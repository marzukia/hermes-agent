"""Tests for the terminal-stage MERGE GATE + merge-review notification.

When a stage_chain task completes its FINAL stage (review/test passed, no
successor), complete_task does NOT mark it done. It HOLDS the card in a
pending-merge state (status='blocked', current_step_key kept at the terminal
stage) and emits a ``merge_review_ready`` event carrying the PR ref + task
title. The gateway kanban notifier renders + delivers that ping to the
configured main session (and to any per-task subscriber). The main agent then
resolves the gate:

  * MERGE  -> ``merge_task`` transitions held -> done (after the PR is merged);
  * REJECT -> ``reject_merge`` sends the card back to the worker/first stage.

These tests exercise:

  * terminal-stage completion HOLDS (not done) + fires the ping once, with PR;
  * non-terminal completion (advance) does NOT fire it;
  * non-chain completion does NOT fire it (and still goes done);
  * a missing PR ref degrades without crashing (event still fires, title-only);
  * merge_task: held -> done, no re-fire of merge_review_ready;
  * reject_merge: held -> ready at the first/build stage with the reason;
  * the configured merge-notify target receives the ping with no auto-sub;
  * gateway renders the merge_review_ready event into a tail message.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


STAGE_CHAIN = [
    {"key": "build", "profile": "worker"},
    {"key": "review", "profile": "reviewer-cc"},
    {"key": "merge", "profile": "merger"},
]


def _write_board_json(home: Path, stage_chain) -> None:
    board = {"stage_chain": stage_chain} if stage_chain is not None else {}
    (home / "board.json").write_text(json.dumps(board))


@pytest.fixture
def stage_chain_home(kanban_home):
    _write_board_json(kanban_home, STAGE_CHAIN)
    return kanban_home


def _events_of_kind(conn, task_id, kind):
    rows = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = ? "
        "ORDER BY id ASC",
        (task_id, kind),
    ).fetchall()
    out = []
    for r in rows:
        out.append(json.loads(r["payload"]) if r["payload"] else {})
    return out


def _drive_to_terminal(conn):
    """Create a task and advance it through build+review to the held merge
    gate. Returns the task id. The card ends blocked at the 'merge' stage."""
    t = kb.create_task(conn, title="Move FAB to bottom-right", assignee="worker")
    kb.add_comment(
        conn, t, author="worker",
        body="Opened https://github.com/acme/app/pull/13 for review.",
    )
    kb.complete_task(conn, t, summary="built")     # build -> review (advance)
    kb.complete_task(conn, t, summary="reviewed")  # review -> merge (advance)
    assert kb.get_task(conn, t).current_step_key == "merge"
    kb.complete_task(conn, t, result="tests pass")  # merge -> HELD (gate)
    return t


# ---------------------------------------------------------------------------
# 1. Terminal-stage completion HOLDS (not done) + fires the ping once.
# ---------------------------------------------------------------------------

def test_terminal_stage_holds_and_fires_merge_review_ready(stage_chain_home):
    with kb.connect() as conn:
        t = _drive_to_terminal(conn)
        task = kb.get_task(conn, t)
        # HELD, not done.
        assert task.status == "blocked"
        assert task.current_step_key == "merge"
        events = _events_of_kind(conn, t, "merge_review_ready")
        # No completed event yet — the card isn't done.
        completed = _events_of_kind(conn, t, "completed")
    assert len(events) == 1
    pl = events[0]
    assert pl.get("pr_number") == 13
    assert pl.get("pr_url") == "https://github.com/acme/app/pull/13"
    assert pl.get("title") == "Move FAB to bottom-right"
    assert pl.get("from_step") == "merge"
    assert completed == []


# ---------------------------------------------------------------------------
# 2. Non-terminal completion (advance) does NOT fire it.
# ---------------------------------------------------------------------------

def test_non_terminal_completion_does_not_fire(stage_chain_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="build it", assignee="worker")
        kb.complete_task(conn, t, summary="built")  # build -> review (advance)
        assert kb.get_task(conn, t).current_step_key == "review"
        assert kb.get_task(conn, t).status == "ready"
        events = _events_of_kind(conn, t, "merge_review_ready")
    assert events == []


# ---------------------------------------------------------------------------
# 3. Non-chain completion does NOT fire it (and still goes done).
# ---------------------------------------------------------------------------

def test_non_chain_completion_does_not_fire(kanban_home):
    # No board.json -> no stage_chain -> plain done path, no current_step_key.
    with kb.connect() as conn:
        t = kb.create_task(conn, title="standalone", assignee="worker")
        assert kb.get_task(conn, t).current_step_key is None
        ok = kb.complete_task(conn, t, result="ok")
        assert ok is True
        assert kb.get_task(conn, t).status == "done"
        events = _events_of_kind(conn, t, "merge_review_ready")
    assert events == []


# ---------------------------------------------------------------------------
# 4. Missing PR ref degrades without crashing (event still fires, title-only).
# ---------------------------------------------------------------------------

def test_terminal_stage_missing_pr_degrades(stage_chain_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="No PR task", assignee="worker")
        # No PR comment, no branch_name.
        kb.complete_task(conn, t, summary="built")
        kb.complete_task(conn, t, summary="reviewed")
        kb.complete_task(conn, t, result="tests pass")
        assert kb.get_task(conn, t).status == "blocked"
        events = _events_of_kind(conn, t, "merge_review_ready")
    assert len(events) == 1
    pl = events[0]
    assert "pr_number" not in pl
    assert "pr_url" not in pl
    assert pl.get("title") == "No PR task"
    assert pl.get("from_step") == "merge"


# ---------------------------------------------------------------------------
# 5. merge_task: held -> done, and does NOT re-fire merge_review_ready.
# ---------------------------------------------------------------------------

def test_merge_task_closes_held_card(stage_chain_home):
    with kb.connect() as conn:
        t = _drive_to_terminal(conn)
        ok = kb.merge_task(conn, t, actor="main", result="PR #13 merged")
        assert ok is True
        task = kb.get_task(conn, t)
        assert task.status == "done"
        assert task.current_step_key is None
        # Exactly one merge_review_ready (from the hold), none re-fired.
        assert len(_events_of_kind(conn, t, "merge_review_ready")) == 1
        decided = _events_of_kind(conn, t, "merge_decided")
        assert len(decided) == 1 and decided[0].get("decision") == "merge"
        # A normal completed event is also emitted for downstream consumers.
        assert len(_events_of_kind(conn, t, "completed")) == 1


def test_merge_task_refuses_non_held(stage_chain_home):
    with kb.connect() as conn:
        # A ready (not held) chain task can't be merged.
        t = kb.create_task(conn, title="x", assignee="worker")
        assert kb.merge_task(conn, t) is False
        assert kb.get_task(conn, t).status != "done"


# ---------------------------------------------------------------------------
# 6. reject_merge: held -> ready at first/build stage with the reason.
# ---------------------------------------------------------------------------

def test_reject_merge_sends_back_to_worker(stage_chain_home):
    with kb.connect() as conn:
        t = _drive_to_terminal(conn)
        ok = kb.reject_merge(
            conn, t, reason="lint fails on src/foo.py", actor="main"
        )
        assert ok is True
        task = kb.get_task(conn, t)
        assert task.status == "ready"
        assert task.current_step_key == "build"
        assert task.assignee == "worker"
        decided = _events_of_kind(conn, t, "merge_decided")
        assert len(decided) == 1
        assert decided[0].get("decision") == "reject"
        assert decided[0].get("to_step") == "build"
        # The reject reason is visible as a comment for the worker.
        comments = kb.list_comments(conn, t)
        assert any("lint fails on src/foo.py" in c.body for c in comments)


def test_reject_merge_requires_reason(stage_chain_home):
    with kb.connect() as conn:
        t = _drive_to_terminal(conn)
        with pytest.raises(ValueError):
            kb.reject_merge(conn, t, reason="   ", actor="main")


def test_reject_merge_refuses_non_held(stage_chain_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="x", assignee="worker")
        assert kb.reject_merge(conn, t, reason="nope") is False


# ---------------------------------------------------------------------------
# 7. _extract_pr_ref returns the most recent PR URL + parsed number.
# ---------------------------------------------------------------------------

def test_extract_pr_ref_prefers_latest(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="x", assignee="worker")
        kb.add_comment(conn, t, author="w", body="https://github.com/a/b/pull/1")
        kb.add_comment(conn, t, author="w", body="reopened https://github.com/a/b/pull/2 now")
        url, num = kb._extract_pr_ref(conn, t)
    assert url == "https://github.com/a/b/pull/2"
    assert num == 2


def test_extract_pr_ref_none_when_absent(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="x", assignee="worker")
        url, num = kb._extract_pr_ref(conn, t)
    assert url is None
    assert num is None


# ---------------------------------------------------------------------------
# 8. The configured merge-notify target receives the ping with NO auto-sub.
# ---------------------------------------------------------------------------

def test_merge_review_sentinel_claims_across_tasks(stage_chain_home):
    with kb.connect() as conn:
        # The gateway registers the always-on target at startup (before any
        # new merge event). No PER-TASK notify sub is ever created —
        # simulating chain tasks created by foreman/worker where the main
        # session never auto-subscribed. The target must still get the ping.
        kb.ensure_merge_review_sub(
            conn, platform="discord", chat_id="main-chat",
        )
        t = _drive_to_terminal(conn)  # fires merge_review_ready after register
        old, new, events = kb.claim_unseen_merge_reviews(
            conn, platform="discord", chat_id="main-chat",
        )
        assert len(events) == 1
        ev = events[0]
        assert ev.task_id == t
        assert ev.kind == "merge_review_ready"
        assert (ev.payload or {}).get("pr_number") == 13
        # Sentinel is excluded from the normal per-task notify loop.
        subs = kb.list_notify_subs(conn)
        assert all(
            s["task_id"] != kb.MERGE_REVIEW_SENTINEL_TASK for s in subs
        )
        # A second claim is a no-op (cursor advanced).
        _, _, again = kb.claim_unseen_merge_reviews(
            conn, platform="discord", chat_id="main-chat",
        )
        assert again == []


def test_merge_review_sub_no_backlog_on_first_register(stage_chain_home):
    # A target registered AFTER a merge event already fired starts at the
    # current max cursor and is not spammed with the historical backlog.
    with kb.connect() as conn:
        _drive_to_terminal(conn)  # fires merge_review_ready #1
        kb.ensure_merge_review_sub(
            conn, platform="discord", chat_id="late-chat",
        )
        _, _, events = kb.claim_unseen_merge_reviews(
            conn, platform="discord", chat_id="late-chat",
        )
    assert events == []


# ---------------------------------------------------------------------------
# 9. Gateway notifier renders the merge_review_ready event into a tail message.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_notifier_renders_and_delivers_merge_review(stage_chain_home):
    import asyncio
    from unittest.mock import AsyncMock, MagicMock
    from gateway.run import GatewayRunner
    from gateway.config import Platform

    with kb.connect() as conn:
        t = kb.create_task(conn, title="Move FAB to bottom-right", assignee="worker")
        kb.add_notify_sub(conn, task_id=t, platform="telegram", chat_id="chat1")
        kb.add_comment(
            conn, t, author="worker",
            body="https://github.com/acme/app/pull/13",
        )
        kb.complete_task(conn, t, summary="built")
        kb.complete_task(conn, t, summary="reviewed")
        kb.complete_task(conn, t, result="tests pass")  # -> HELD + ping

    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner._kanban_sub_fail_counts = {}

    sent = []

    async def _send(chat_id, msg, metadata=None):
        sent.append((chat_id, msg))
        runner._running = False

    fake_adapter = MagicMock()
    fake_adapter.send = AsyncMock(side_effect=_send)
    runner.adapters = {Platform.TELEGRAM: fake_adapter}

    _orig_sleep = asyncio.sleep

    async def _fast_sleep(_):
        await _orig_sleep(0)

    import gateway.run as gr
    orig = gr.asyncio.sleep
    gr.asyncio.sleep = _fast_sleep
    try:
        await asyncio.wait_for(
            runner._kanban_notifier_watcher(interval=0.01), timeout=10
        )
    except asyncio.TimeoutError:
        pass
    finally:
        gr.asyncio.sleep = orig

    joined = "\n".join(m for _, m in sent)
    assert "PR #13" in joined
    assert "Move FAB to bottom-right" in joined
    assert "merge or reject" in joined
