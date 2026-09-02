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
