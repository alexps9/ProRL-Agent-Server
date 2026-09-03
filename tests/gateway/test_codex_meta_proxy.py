from __future__ import annotations

import importlib.util
import json
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "swebench_verified"
    / "codex_meta_proxy.py"
)
SPEC = importlib.util.spec_from_file_location("codex_meta_proxy", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
proxy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(proxy)


def _multi_agent_tools() -> list[dict]:
    """The two `type: namespace` entries Codex sends with --enable multi_agent_v2."""
    return [
        {
            "type": "namespace",
            "name": "functions",
            "tools": [
                {"type": "custom", "name": "exec", "description": "run"},
                {
                    "type": "function",
                    "name": "wait",
                    "parameters": {"type": "object", "properties": {"timeout_ms": {"type": "number"}}},
                },
            ],
        },
        {
            "type": "namespace",
            "name": "collaboration",
            "tools": [
                {
                    "type": "function",
                    "name": "spawn_agent",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task_name": {"type": "string"},
                            "message": {"type": "string", "encrypted": True},
                        },
                    },
                },
                {"type": "function", "name": "wait_agent",
                 "parameters": {"type": "object", "properties": {"agent_ids": {"type": "array", "items": {"type": "string"}}}}},
            ],
        },
    ]


def test_flatten_namespace_tools_keeps_functions_bare_qualifies_collaboration() -> None:
    out = proxy.flatten_namespace_tools(_multi_agent_tools())
    names = {t.get("name") for t in out}
    assert "exec" in names and "wait" in names  # functions ns -> bare
    assert "collaboration.spawn_agent" in names and "collaboration.wait_agent" in names
    # `encrypted: true` stripped so the payload round-trips as plain text
    spawn = next(t for t in out if t["name"] == "collaboration.spawn_agent")
    assert "encrypted" not in spawn["parameters"]["properties"]["message"]


def test_strip_functions_prefix_on_history_items() -> None:
    body = {
        "input": [
            {"type": "custom_tool_call", "name": "functions.exec", "input": "x", "call_id": "c1"},
            {"type": "function_call", "name": "collaboration.spawn_agent", "arguments": "{}", "call_id": "c2"},
        ]
    }
    proxy.strip_functions_prefix(body)
    assert body["input"][0]["name"] == "exec"
    assert body["input"][1]["name"] == "collaboration.spawn_agent"  # collaboration untouched


def test_split_namespaced_calls_restores_namespace_field() -> None:
    resp = {
        "output": [
            {"type": "function_call", "name": "collaboration.spawn_agent",
             "arguments": '{"task_name":"x","message":"do the thing"}', "call_id": "c1"},
            {"type": "custom_tool_call", "name": "exec", "input": "y", "call_id": "c2"},
        ]
    }
    proxy.split_namespaced_calls(resp)
    fc = resp["output"][0]
    assert fc["name"] == "spawn_agent" and fc["namespace"] == "collaboration"
    assert resp["output"][1]["name"] == "exec"  # not dotted -> untouched


def test_agent_message_payload_reinjection_round_trip() -> None:
    session = "sess-1"
    proxy._SENT_MSGS.clear()
    # 1. root's turn produces a spawn_agent call -> record the plaintext message
    proxy.record_outgoing_agent_messages(
        {"output": [{"type": "function_call", "name": "spawn_agent",
                     "arguments": '{"task_name":"checker","message":"Review calc.py for bugs"}'}]},
        req_input=[],
        session=session,
    )
    # 2. the sub-agent's turn: Codex threads the task as an agent_message whose
    #    payload is an opaque encrypted_content part
    body = {
        "input": [
            {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "sys"}]},
            {
                "type": "agent_message",
                "author": "/root",
                "recipient": "/root/checker",
                "content": [
                    {"type": "input_text", "text": "Message Type: NEW_TASK\nPayload:\n"},
                    {"type": "encrypted_content", "encrypted_content": "gAAAA..."},
                ],
            },
        ]
    }
    proxy.normalize_agent_messages(body, session)
    item = body["input"][1]
    assert item["type"] == "message" and item["role"] == "user"
    assert item["content"][-1] == {"type": "input_text", "text": "Review calc.py for bugs"}


def test_agent_message_reinjection_survives_a_resent_turn() -> None:
    session = "sess-2"
    proxy._SENT_MSGS.clear()
    proxy.record_outgoing_agent_messages(
        {"output": [{"type": "function_call", "name": "spawn_agent",
                     "arguments": '{"task_name":"c","message":"task text"}'}]},
        req_input=[], session=session,
    )

    def _req():
        return {"input": [{
            "type": "agent_message", "author": "/root", "recipient": "/root/c",
            "content": [{"type": "input_text", "text": "NEW_TASK"},
                        {"type": "encrypted_content", "encrypted_content": "blob"}],
        }]}

    for _ in range(3):  # Codex re-sends the full thread every turn
        b = _req()
        proxy.normalize_agent_messages(b, session)
        assert b["input"][0]["content"][-1]["text"] == "task text"


def test_response_sse_events_lifecycle_preserves_items() -> None:
    resp = {
        "id": "r1", "object": "response", "status": "completed", "created_at": 1.0,
        "output": [
            {"type": "reasoning", "id": "rs1", "summary": [], "encrypted_content": "enc"},
            {"type": "function_call", "id": "fc1", "call_id": "c1",
             "name": "spawn_agent", "namespace": "collaboration", "arguments": '{"task_name":"x"}'},
        ],
        "usage": {"input_tokens": 5, "output_tokens": 3},
    }
    evs = proxy.response_sse_events(resp)
    types = [e["type"] for e in evs]
    assert types[0] == "response.created"
    assert types[-1] == "response.completed"
    assert "response.function_call_arguments.done" in types
    # the reasoning item's encrypted_content and the call's namespace survive
    done = [e for e in evs if e["type"] == "response.output_item.done"]
    assert done[0]["item"]["encrypted_content"] == "enc"
    assert done[1]["item"]["namespace"] == "collaboration"
    assert evs[-1]["response"]["output"] == resp["output"]
