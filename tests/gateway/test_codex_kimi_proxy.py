from __future__ import annotations

import importlib.util
import json
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "swebench_verified"
    / "codex_kimi_proxy.py"
)
SPEC = importlib.util.spec_from_file_location("codex_kimi_proxy", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
proxy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(proxy)


def _chat_response(tool_name: str, arguments: str) -> dict:
    return {
        "id": "chat-1",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "reasoning": "Use the requested tool.",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": tool_name, "arguments": arguments},
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
    }


def test_function_tool_response_emits_complete_sse_lifecycle() -> None:
    response = proxy.translate_response(
        _chat_response("exec_command", '{"cmd":"printf ok"}'),
        "kimi-k3",
    )
    events = proxy.response_sse_events(response)
    types = [event["type"] for event in events]

    assert types[0] == "response.created"
    assert "response.function_call_arguments.delta" in types
    assert "response.function_call_arguments.done" in types
    assert types.count("response.output_item.added") == 2  # reasoning + tool
    assert types.count("response.output_item.done") == 2
    assert types[-1] == "response.completed"
    assert response["output"][1]["type"] == "function_call"


def test_custom_tool_round_trip_preserves_freeform_input() -> None:
    freeform = 'const r = await tools.exec_command({"cmd":"printf ok"}); text(r.output)'
    response = proxy.translate_response(
        _chat_response("exec", json.dumps({"input": freeform})),
        "kimi-k3",
        custom_tool_names={"exec"},
    )
    item = response["output"][1]
    events = proxy.response_sse_events(response)
    types = [event["type"] for event in events]

    assert item["type"] == "custom_tool_call"
    assert item["input"] == freeform
    assert "response.custom_tool_call_input.delta" in types
    assert "response.custom_tool_call_input.done" in types
    assert events[-1]["response"]["output"][1] == item


def test_regular_single_input_function_is_not_misclassified_as_custom() -> None:
    response = proxy.translate_response(
        _chat_response("lookup", '{"input":"query"}'),
        "kimi-k3",
        custom_tool_names={"exec"},
    )

    assert response["output"][1]["type"] == "function_call"


def test_multi_agent_namespace_round_trip() -> None:
    tools = [{
        "type": "namespace",
        "name": "collaboration",
        "tools": [{
            "type": "function",
            "name": "spawn_agent",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_name": {"type": "string"},
                    "message": {"type": "string", "encrypted": True},
                },
            },
        }],
    }]
    flattened = proxy.flatten_namespace_tools(tools)
    assert flattened[0]["name"] == "collaboration.spawn_agent"
    assert "encrypted" not in flattened[0]["parameters"]["properties"]["message"]
    translated = proxy.translate_tools(flattened)
    assert translated[0]["function"]["name"] == "collaboration.spawn_agent"

    response = proxy.translate_response(
        _chat_response(
            "collaboration.spawn_agent",
            '{"task_name":"reviewer","message":"Inspect the tests"}',
        ),
        "kimi-k3",
    )
    proxy.split_namespaced_calls(response)
    call = response["output"][1]
    assert call["name"] == "spawn_agent"
    assert call["namespace"] == "collaboration"


def test_agent_message_plaintext_reinjection() -> None:
    proxy._SENT_MSGS.clear()
    session = "kimi-session"
    proxy.record_outgoing_agent_messages(
        {"output": [{
            "type": "function_call",
            "name": "spawn_agent",
            "arguments": '{"task_name":"reviewer","message":"Inspect the tests"}',
        }]},
        [],
        session,
    )
    body = {"input": [{
        "type": "agent_message",
        "recipient": "/root/reviewer",
        "content": [{"type": "encrypted_content", "encrypted_content": "opaque"}],
    }]}
    proxy.normalize_agent_messages(body, session)
    assert body["input"][0]["type"] == "message"
    assert body["input"][0]["content"][0]["text"] == "Inspect the tests"


def test_kimi_default_max_tokens_is_large_enough_for_agent_turns() -> None:
    assert proxy.DEFAULT_MAX_TOKENS >= 8192
