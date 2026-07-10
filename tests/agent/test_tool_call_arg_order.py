# SPDX-License-Identifier: Apache-2.0
"""Regression guard for tool-call argument key ordering in the replayed prompt.

A local inference server reuses the KV cache by matching the token prefix of
one turn against the next. The model emits a tool call in some key order and
the server caches KV from those tokens; if Hermes replays the same historical
call with the keys REORDERED (e.g. sorted alphabetically), the prompt diverges
from the cache at that call and the whole context cold-re-prefills.

``_normalize_api_messages_for_prefix_cache`` must therefore preserve the
model's emission order, never sort. These tests pin that, and the try/except
(clean vs repaired) branch consistency, so a future ``sort_keys=True`` can't
sneak back in unnoticed.
"""

from __future__ import annotations

import json

from agent.conversation_loop import _normalize_api_messages_for_prefix_cache


def _tc(arguments: str) -> list:
    return [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": arguments},
                }
            ],
        }
    ]


def _args_out(api_messages: list) -> str:
    return api_messages[0]["tool_calls"][0]["function"]["arguments"]


def test_emission_order_preserved_not_sorted():
    # Model emitted offset before limit — NOT alphabetical order.
    msgs = _tc('{"offset": 0, "limit": 100, "path": "/x"}')
    _normalize_api_messages_for_prefix_cache(msgs)
    out = _args_out(msgs)
    # compact separators, and offset still before limit (emission order kept)
    assert out == '{"offset":0,"limit":100,"path":"/x"}'
    # explicit anti-regression: must NOT equal the sorted form
    sorted_form = json.dumps(
        json.loads(out), separators=(",", ":"), sort_keys=True
    )
    assert out != sorted_form
    assert out.index('"offset"') < out.index('"limit"')


def test_repaired_args_also_preserve_order():
    # A literal control char forces the except/_repair path; order must hold so
    # a repaired call and a clean call canonicalize identically.
    msgs = _tc('{"offset": 0, "limit": 100, "note": "a\tb"}')
    _normalize_api_messages_for_prefix_cache(msgs)
    out = _args_out(msgs)
    assert out.index('"offset"') < out.index('"limit"') < out.index('"note"')


def test_already_compact_call_is_stable_across_repeated_normalization():
    # Idempotence: normalizing an already-normalized call is a no-op, so a
    # replayed prefix stays byte-identical turn after turn.
    msgs = _tc('{"b":1,"a":2}')
    _normalize_api_messages_for_prefix_cache(msgs)
    first = _args_out(msgs)
    _normalize_api_messages_for_prefix_cache(msgs)
    assert _args_out(msgs) == first == '{"b":1,"a":2}'


def test_content_whitespace_is_stripped():
    msgs = [{"role": "user", "content": "  hello  "}]
    _normalize_api_messages_for_prefix_cache(msgs)
    assert msgs[0]["content"] == "hello"
