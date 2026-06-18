"""Tests for the kanban stage-chain serial workflow (hermes_cli.kanban_db).

Covers the opt-in stage_chain auto-advance added alongside the kanban
serial-workflow feature: first-stage stamping on create, stage advance on
complete (with failure-counter reset), terminal/no-chain done behaviour,
stage-chain sanitisation, and priority dispatch ordering.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB.

    Mirrors the fixture in test_kanban_db.py (which is module-local there and
    therefore not visible here). The default board's kanban.db lands at
    ``<home>/kanban.db``, so a board.json written next to it (``<home>``) is
    picked up by ``_resolve_stage_chain``.
    """
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


# A three-stage chain: worker -> reviewer-cc -> merger. Distinct profiles so
# advance reassignment is observable.
STAGE_CHAIN = [
    {"key": "build", "profile": "worker"},
    {"key": "review", "profile": "reviewer-cc"},
    {"key": "merge", "profile": "merger"},
]


def _write_board_json(home: Path, stage_chain) -> None:
    """Drop a board.json with *stage_chain* next to the default board's
    kanban.db so _resolve_stage_chain (which reads the sibling of the
    connection's own DB) finds it."""
    board = {"stage_chain": stage_chain} if stage_chain is not None else {}
    (home / "board.json").write_text(json.dumps(board))


@pytest.fixture
def stage_chain_home(kanban_home):
    """kanban_home plus a board.json carrying the three-stage STAGE_CHAIN."""
    _write_board_json(kanban_home, STAGE_CHAIN)
    return kanban_home


def _set_failures(conn, task_id: str, n: int) -> None:
    conn.execute(
        "UPDATE tasks SET consecutive_failures = ? WHERE id = ?", (n, task_id)
    )


# ---------------------------------------------------------------------------
# 1. create_task first-stage stamping
# ---------------------------------------------------------------------------

def test_create_stamps_first_stage_for_worker(stage_chain_home):
    """A task assigned to the first stage's profile enters the chain at the
    first stage (current_step_key stamped)."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="build it", assignee="worker")
        task = kb.get_task(conn, t)
        assert task is not None
    assert task.current_step_key == "build"


def test_create_stamps_first_stage_for_null_assignee(stage_chain_home):
    """An unassigned task also enters at the first stage — null-assignee tasks
    default to the worker/first stage at dispatch."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="floater")
        task = kb.get_task(conn, t)
        assert task is not None
    assert task.current_step_key == "build"


def test_create_does_not_stamp_for_reviewer_assignee(stage_chain_home):
    """A task explicitly assigned to a non-first-stage profile (e.g. a reviewer
    child) is left out of the chain — no first-stage key."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="review child", assignee="reviewer-cc")
        task = kb.get_task(conn, t)
        assert task is not None
    assert task.current_step_key is None


# ---------------------------------------------------------------------------
# 2. complete_task on a non-final stage advances (and clears failures)
# ---------------------------------------------------------------------------

def test_complete_advances_non_final_stage(stage_chain_home):
    """Completing a task on a non-final stage advances it: status ready,
    assignee = next stage's profile, current_step_key = next stage."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="build it", assignee="worker")
        assert kb.get_task(conn, t).current_step_key == "build"
        ok = kb.complete_task(conn, t, summary="built the thing")
        task = kb.get_task(conn, t)
        assert task is not None
    assert ok is True
    assert task.status == "ready"
    assert task.assignee == "reviewer-cc"
    assert task.current_step_key == "review"


def test_advance_clears_failure_counter(stage_chain_home):
    """Advancing a stage resets the consecutive-failure counter so the next
    stage starts with a fresh budget."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="build it", assignee="worker")
        _set_failures(conn, t, 3)
        assert kb.get_task(conn, t).consecutive_failures == 3
        kb.complete_task(conn, t, summary="built")
        task = kb.get_task(conn, t)
        assert task is not None
    assert task.current_step_key == "review"  # actually advanced
    assert task.consecutive_failures == 0


# ---------------------------------------------------------------------------
# 3. complete_task on the final stage / no chain -> done
# ---------------------------------------------------------------------------

def test_complete_final_stage_marks_done(stage_chain_home):
    """Completing on the LAST stage marks the task done, not advanced."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="merge it", assignee="worker")
        # Walk to the final stage.
        kb.complete_task(conn, t, summary="built")     # build -> review
        kb.complete_task(conn, t, summary="reviewed")  # review -> merge
        assert kb.get_task(conn, t).current_step_key == "merge"
        ok = kb.complete_task(conn, t, result="merged")  # merge -> done
        task = kb.get_task(conn, t)
        assert task is not None
    assert ok is True
    assert task.status == "done"


def test_complete_no_chain_marks_done(kanban_home):
    """With no stage_chain configured (no board.json), completion is the plain
    done path — zero behaviour change."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="standalone", assignee="worker")
        assert kb.get_task(conn, t).current_step_key is None
        ok = kb.complete_task(conn, t, result="ok")
        task = kb.get_task(conn, t)
        assert task is not None
    assert ok is True
    assert task.status == "done"


# ---------------------------------------------------------------------------
# 4. _resolve_stage_chain sanitisation
# ---------------------------------------------------------------------------

def test_resolve_stage_chain_drops_malformed_and_dedups(kanban_home):
    """_resolve_stage_chain drops non-dict entries, entries missing key/profile,
    and de-duplicates keys (first occurrence wins)."""
    messy = [
        {"key": "a", "profile": "pa"},
        "not-a-dict",
        {"key": "", "profile": "pblank"},          # empty key
        {"key": "b", "profile": ""},               # empty profile
        {"key": "b", "profile": "pb"},             # first valid b
        {"key": "a", "profile": "pa-dup"},         # duplicate key a -> dropped
        {"profile": "pnokey"},                     # missing key
        42,                                        # non-dict
    ]
    _write_board_json(kanban_home, messy)
    with kb.connect() as conn:
        chain = kb._resolve_stage_chain(conn)
    assert [s["key"] for s in chain] == ["a", "b"]
    assert [s["profile"] for s in chain] == ["pa", "pb"]


# ---------------------------------------------------------------------------
# 5. priority dispatch ordering (higher priority is claimed first)
# ---------------------------------------------------------------------------

def test_dispatch_orders_higher_priority_first(kanban_home, all_assignees_spawnable):
    """The dispatcher claims higher-priority tasks first (higher int = more
    urgent, matching the board UI: Higher = claimed first)."""
    spawn_order = []

    def fake_spawn(task, workspace, board=None):
        spawn_order.append(task.id)
        return 42

    with kb.connect() as conn:
        # Create the lower-priority task first so created_at alone would pick
        # it; the priority sort must override and pick the higher-priority one.
        lower = kb.create_task(conn, title="p1", assignee="alice", priority=1)
        higher = kb.create_task(conn, title="p5", assignee="alice", priority=5)
        kb.dispatch_once(conn, spawn_fn=fake_spawn, max_spawn=1)
    assert spawn_order == [higher]


# ---------------------------------------------------------------------------
# 6. handoff visibility — the advance must not drop summary/metadata
# ---------------------------------------------------------------------------

def _runs_for(conn, task_id):
    """All task_runs rows for *task_id*, oldest first, as plain dicts."""
    rows = conn.execute(
        "SELECT id, status, outcome, summary, metadata FROM task_runs "
        "WHERE task_id = ? ORDER BY id",
        (task_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def test_advance_preserves_handoff_for_claimed_worker(stage_chain_home):
    """A dispatcher-claimed worker (has a current_run_id) completing a
    non-final stage must close its run carrying the summary + metadata so the
    next stage can read the handoff out of attempt history."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="build it", assignee="worker")
        claimed = kb.claim_task(conn, t)
        assert claimed is not None
        assert claimed.current_run_id is not None  # a real run is in flight
        ok = kb.complete_task(
            conn, t,
            summary="built X",
            metadata={"changed_files": ["a.py"]},
        )
        task = kb.get_task(conn, t)
        runs = _runs_for(conn, t)
    assert ok is True
    assert task is not None
    assert task.current_step_key == "review"  # advanced, not done
    # The claimed run was closed as a stage advance and carries the handoff.
    advanced = [r for r in runs if r["outcome"] == "stage_advanced"]
    assert len(advanced) == 1
    row = advanced[0]
    assert row["summary"] == "built X"
    assert json.loads(row["metadata"]) == {"changed_files": ["a.py"]}


def test_advance_synthesizes_run_for_ready_task(stage_chain_home):
    """A ready task (no current_run_id) advanced with summary+metadata still
    gets a synthesized run row carrying them (FIX A no-active-run path)."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="build it", assignee="worker")
        assert kb.get_task(conn, t).current_run_id is None  # never claimed
        ok = kb.complete_task(
            conn, t,
            summary="built ready",
            metadata={"changed_files": ["b.py"]},
        )
        task = kb.get_task(conn, t)
        runs = _runs_for(conn, t)
    assert ok is True
    assert task is not None
    assert task.current_step_key == "review"  # advanced
    advanced = [r for r in runs if r["outcome"] == "stage_advanced"]
    assert len(advanced) == 1
    row = advanced[0]
    assert row["summary"] == "built ready"
    assert json.loads(row["metadata"]) == {"changed_files": ["b.py"]}


def test_stage_advanced_event_links_run(stage_chain_home):
    """The stage_advanced event row references the closed/synthesized run id
    so UIs can group it with the rest of the attempt."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="build it", assignee="worker")
        kb.complete_task(conn, t, summary="built", metadata={"k": "v"})
        ev = conn.execute(
            "SELECT run_id, payload FROM task_events "
            "WHERE task_id = ? AND kind = 'stage_advanced' ORDER BY id DESC LIMIT 1",
            (t,),
        ).fetchone()
        run = conn.execute(
            "SELECT id FROM task_runs WHERE task_id = ? AND outcome = 'stage_advanced'",
            (t,),
        ).fetchone()
    assert ev is not None and run is not None
    assert ev["run_id"] == run["id"]


# ---------------------------------------------------------------------------
# 7. FIX D — terminal/vanished-chain done still emits an audit event
# ---------------------------------------------------------------------------

def test_terminal_stage_emits_terminal_event(stage_chain_home):
    """Completing the final stage goes to done AND emits stage_chain_terminal
    with chain_present=True (legit final stage)."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="merge it", assignee="worker")
        kb.complete_task(conn, t, summary="built")     # build -> review
        kb.complete_task(conn, t, summary="reviewed")  # review -> merge
        ok = kb.complete_task(conn, t, result="merged")  # merge -> done
        task = kb.get_task(conn, t)
        ev = conn.execute(
            "SELECT payload FROM task_events "
            "WHERE task_id = ? AND kind = 'stage_chain_terminal' "
            "ORDER BY id DESC LIMIT 1",
            (t,),
        ).fetchone()
    assert ok is True
    assert task is not None and task.status == "done"
    assert ev is not None
    payload = json.loads(ev["payload"])
    assert payload["from_step"] == "merge"
    assert payload["chain_present"] is True


def test_vanished_chain_done_emits_terminal_event(stage_chain_home):
    """A stepped task whose board.json chain is removed before completion still
    goes to done AND emits stage_chain_terminal with chain_present=False, so a
    misconfig-induced early-done is auditable rather than silent."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="build it", assignee="worker")
        assert kb.get_task(conn, t).current_step_key == "build"  # stamped
        # Remove the sibling board.json so the chain resolves empty.
        (stage_chain_home / "board.json").unlink()
        ok = kb.complete_task(conn, t, result="done anyway")
        task = kb.get_task(conn, t)
        ev = conn.execute(
            "SELECT payload FROM task_events "
            "WHERE task_id = ? AND kind = 'stage_chain_terminal' "
            "ORDER BY id DESC LIMIT 1",
            (t,),
        ).fetchone()
    assert ok is True
    assert task is not None and task.status == "done"  # not advanced — chain gone
    assert ev is not None
    payload = json.loads(ev["payload"])
    assert payload["from_step"] == "build"
    assert payload["chain_present"] is False
