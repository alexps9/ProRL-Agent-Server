#!/usr/bin/env python3
"""Local proxy: codex's OpenAI Responses API -> chat/completions, for models
only available on Meta's older Llama API endpoint (e.g. Kimi-K3).

Unlike codex_meta_proxy.py (pure passthrough, both sides already speak
Responses API), this endpoint speaks the OLD-style chat/completions
protocol (assistant.tool_calls / role:tool). Codex 0.145.0 has removed
support for configuring `wire_api = "chat"` directly ("no longer
supported... set wire_api = 'responses'"), so there is no way to point
codex at this endpoint without translation. This proxy does that
translation in both directions.

Two real wrinkles, both load-bearing (without either, codex retries the
same turn forever -- prompt_tokens identical across every retry -- until
it gives up with no output):

1. codex's "custom" tools (apply_patch, and the unified_exec
   "exec"/"wait"/"write_stdin" family -- which our earlier trace analysis
   showed carries the large majority of a session's actual tool calls) use
   a Responses-API-only concept: a freeform string payload with no JSON
   schema, not a JSON-schema function call. Chat/completions function
   calling has no equivalent, so each custom tool is represented to the
   model as a normal JSON function with a single string parameter
   ("input") and translated back to a custom_tool_call on the way out.
   This is an approximation (the model sees "call this function with one
   string argument" instead of "write freeform text"), not a faithful
   port.

2. codex requests stream=true and, even after its WebSocket transport
   falls back to plain HTTPS, still expects a real text/event-stream
   response ending in a `response.completed` event -- returning the
   translated result as a single JSON blob makes codex log "stream
   disconnected before completion: stream closed before
   response.completed" and silently retry. We don't stream real
   incremental deltas from the upstream chat/completions call (it's
   already non-streamed by the time we have it); one response.created +
   one response.completed carrying the full object is the minimum codex's
   parser needs (see Handler._send_sse).

Historical upstream issue: Kimi-K3's Fireworks backend previously returned
"Floating point NaN (not-a-number) detected in generation" for Codex's full
system prompt plus tool declarations. The proxy deliberately does not truncate
either input. The backend no longer reproduced that failure in two real
SWE-bench runs on 2026-09-02 (29 full Codex requests, zero NaN errors), but the
error remains classified as retryable together with 429/5xx responses in case
the probabilistic backend failure recurs.

Requires env ``KIMI_API_KEY``. See ``codex_kimi_proxy.env.example`` and
``start_codex_kimi_proxy.sh``.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

HOST = os.environ.get("CODEX_KIMI_PROXY_HOST", "127.0.0.1")
PORT = int(os.environ.get("CODEX_KIMI_PROXY_PORT", "3458"))
UPSTREAM = os.environ.get(
    "CODEX_KIMI_UPSTREAM", "https://api.llama.com/experimental/compat/openai/v1"
).rstrip("/")
# Required: set KIMI_API_KEY in the environment (no default).
API_KEY = os.environ.get("KIMI_API_KEY", "").strip()
# codex's --model value -> this endpoint's model id (only Kimi verified so far;
# add more `fireworks-*` chat/completions-only models here as needed).
MODEL_MAP = {
    "kimi-k3": "fireworks-kimi-k3",
    "fireworks-kimi-k3": "fireworks-kimi-k3",
}
DEBUG = os.environ.get("CODEX_KIMI_DEBUG", "1") not in ("0", "false", "False", "")
DEBUG_LOG_PATH = os.environ.get("CODEX_KIMI_DEBUG_LOG", "/tmp/codex_kimi_proxy_debug.log")

_REQUEST_COUNTER = 0


def _debug(req_id: int, msg: str) -> None:
    if not DEBUG:
        return
    try:
        with open(DEBUG_LOG_PATH, "a") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] req={req_id} {msg}\n")
    except OSError:
        pass


def map_model(name: str | None) -> str:
    if not name:
        return "fireworks-kimi-k3"
    return MODEL_MAP.get(name, name)


# ---------------------------------------------------------------------------
# Responses API tools[] -> chat/completions tools[]
# ---------------------------------------------------------------------------


def _wrap_custom_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """Custom (freeform) Responses tool -> a JSON function with one string arg.

    chat/completions function calling has no freeform-payload concept. The
    model is told to pass its freeform text as the single `input` string.
    """
    description = tool.get("description") or ""
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": (
                description
                + "\n\n(This tool normally takes freeform text, not JSON. "
                "Pass that exact freeform text as the `input` string below.)"
            ),
            "parameters": {
                "type": "object",
                "properties": {"input": {"type": "string", "description": "Freeform tool input."}},
                "required": ["input"],
            },
        },
    }


def translate_tools(tools: list[Any]) -> list[dict[str, Any]]:
    out = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        ttype = t.get("type")
        if ttype == "function":
            out.append({
                "type": "function",
                "function": {
                    "name": t.get("name"),
                    "description": t.get("description") or "",
                    "parameters": t.get("parameters") or {"type": "object", "properties": {}},
                },
            })
        elif ttype == "custom":
            out.append(_wrap_custom_tool(t))
        # tool_search / web_search / other hosted types: no chat/completions
        # equivalent and not needed (SWE-bench runs network-isolated for
        # web_search anyway); silently dropped.
    return out


# ---------------------------------------------------------------------------
# Responses API input[] -> chat/completions messages[]
# ---------------------------------------------------------------------------


def _flatten_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if "text" in block:
                    parts.append(str(block["text"]))
                elif isinstance(block.get("content"), str):
                    parts.append(block["content"])
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return "" if content is None else str(content)


def translate_input(input_items: list[Any], instructions: str | None) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if instructions:
        messages.append({"role": "system", "content": instructions})

    # Pending tool_calls for the *current* assistant turn being assembled,
    # flushed into a single assistant message (chat/completions wants all
    # tool_calls for one turn on one message, not one message per call).
    pending_tool_calls: list[dict[str, Any]] = []

    def flush_pending() -> None:
        if pending_tool_calls:
            messages.append({"role": "assistant", "content": None, "tool_calls": list(pending_tool_calls)})
            pending_tool_calls.clear()

    for item in input_items:
        if not isinstance(item, dict):
            continue
        itype = item.get("type")
        if itype is None and "role" in item:
            # Plain {"role": ..., "content": ...} message.
            flush_pending()
            role = item.get("role")
            role = "system" if role == "developer" else role
            messages.append({"role": role, "content": _flatten_content(item.get("content"))})
        elif itype == "message":
            flush_pending()
            role = item.get("role")
            role = "system" if role == "developer" else role
            messages.append({"role": role, "content": _flatten_content(item.get("content"))})
        elif itype == "reasoning":
            continue  # no chat/completions equivalent to replay; drop.
        elif itype == "function_call":
            pending_tool_calls.append({
                "id": item.get("call_id"),
                "type": "function",
                "function": {"name": item.get("name"), "arguments": item.get("arguments") or "{}"},
            })
        elif itype == "custom_tool_call":
            pending_tool_calls.append({
                "id": item.get("call_id"),
                "type": "function",
                "function": {
                    "name": item.get("name"),
                    "arguments": json.dumps({"input": item.get("input") or ""}),
                },
            })
        elif itype == "function_call_output":
            flush_pending()
            messages.append({
                "role": "tool",
                "tool_call_id": item.get("call_id"),
                "content": _flatten_content(item.get("output")),
            })
        elif itype == "custom_tool_call_output":
            flush_pending()
            messages.append({
                "role": "tool",
                "tool_call_id": item.get("call_id"),
                "content": _flatten_content(item.get("output")),
            })
        # unknown item types are dropped rather than raising -- best-effort.

    flush_pending()
    return messages


# ---------------------------------------------------------------------------
# chat/completions response -> Responses API response
# ---------------------------------------------------------------------------


def translate_response(
    chat_resp: dict[str, Any],
    model: str,
    custom_tool_names: set[str] | None = None,
) -> dict[str, Any]:
    choice = (chat_resp.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    output: list[dict[str, Any]] = []

    # Codex is configured with model_reasoning_effort=xhigh and expects a
    # "reasoning" item per turn (real OpenAI reasoning models return an
    # opaque `encrypted_content` blob for cross-turn CoT continuity). Kimi's
    # chat/completions response instead has a plain-text `reasoning` field.
    # We can't produce real OpenAI encryption, but codex's own encrypted
    # blobs are opaque to it too -- it just stores and replays them
    # verbatim, and we drop "reasoning" input items on the way back in
    # (translate_input) rather than trying to decrypt them, so any
    # placeholder string round-trips fine. Emitting *a* reasoning item
    # (instead of omitting it) is what matters: without one, codex silently
    # discarded every turn and retried the identical prompt (observed: 6x
    # identical prompt_tokens, then task_complete with no final message).
    reasoning_text = message.get("reasoning")
    if reasoning_text:
        output.append({
            "type": "reasoning",
            "id": f"rs_{uuid.uuid4().hex}",
            "summary": [{"type": "summary_text", "text": reasoning_text}],
            "encrypted_content": reasoning_text,
        })

    content = message.get("content")
    if content:
        output.append({
            "type": "message",
            "id": f"msg_{uuid.uuid4().hex}",
            "role": "assistant",
            "content": [{"type": "output_text", "text": content, "annotations": []}],
            "status": "completed",
        })

    for tc in message.get("tool_calls") or []:
        fn = tc.get("function") or {}
        name = fn.get("name")
        raw_args = fn.get("arguments") or "{}"
        call_id = tc.get("id") or f"call_{uuid.uuid4().hex}"
        try:
            parsed = json.loads(raw_args)
        except (TypeError, ValueError):
            parsed = None
        if (
            name in (custom_tool_names or set())
            and isinstance(parsed, dict)
            and set(parsed.keys()) == {"input"}
            and isinstance(parsed["input"], str)
        ):
            # Round-trips a _wrap_custom_tool()-shaped call back to custom_tool_call.
            output.append({
                "type": "custom_tool_call",
                "id": f"ctc_{uuid.uuid4().hex}",
                "call_id": call_id,
                "name": name,
                "input": parsed["input"],
                "status": "completed",
            })
        else:
            output.append({
                "type": "function_call",
                "id": f"fc_{uuid.uuid4().hex}",
                "call_id": call_id,
                "name": name,
                "arguments": raw_args,
                "status": "completed",
            })

    usage = chat_resp.get("usage") or {}
    now = time.time()
    return {
        "id": chat_resp.get("id") or f"resp_{uuid.uuid4().hex}",
        "object": "response",
        "status": "completed",
        "model": model,
        "output": output,
        "output_text": content or "",
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": usage.get("completion_tokens", 0),
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": usage.get("total_tokens", 0),
        },
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "created_at": now,
        "completed_at": now,
        "truncation": "disabled",
        "parallel_tool_calls": False,
        "metadata": {},
    }


def inline_additional_tools(body: dict[str, Any]) -> None:
    """Fold `input` items of type "additional_tools" into the top-level `tools` array.

    Same fixup as codex_meta_proxy.py: newer Codex (unified_exec) sometimes
    declares its tool set via an input[0] item shaped like
    `{"type": "additional_tools", "role": "developer", "tools": [...]}`
    instead of the top-level `tools` field.
    """
    input_items = body.get("input")
    if not isinstance(input_items, list):
        return
    kept: list[Any] = []
    extra_tools: list[Any] = []
    for item in input_items:
        if isinstance(item, dict) and item.get("type") == "additional_tools":
            extra_tools.extend(item.get("tools") or [])
        else:
            kept.append(item)
    if extra_tools:
        body["input"] = kept
        body["tools"] = [*(body.get("tools") or []), *extra_tools]


# ---------------------------------------------------------------------------
# Retry transient capacity errors and the historical Fireworks NaN failure.
# Keep the complete Codex prompt/tool surface intact; silently truncating either
# would change the agent workload whose trace this proxy is meant to capture.
RETRY_MAX_ATTEMPTS = int(os.environ.get("CODEX_KIMI_RETRY_ATTEMPTS", "20"))
RETRY_BASE_DELAY_SECONDS = float(os.environ.get("CODEX_KIMI_RETRY_DELAY", "5.0"))
RETRY_MAX_DELAY_SECONDS = float(os.environ.get("CODEX_KIMI_RETRY_MAX_DELAY", "60.0"))
_RETRYABLE_MESSAGE_SNIPPETS = (
    "floating point nan",
    "app_overload",
    "throttling_error",
)


class UpstreamHTTPError(Exception):
    """Non-retryable (or retries exhausted) upstream HTTP error, body pre-read."""

    def __init__(self, code: int, body: bytes) -> None:
        super().__init__(f"upstream HTTP {code}")
        self.code = code
        self.body = body


def _is_retryable(code: int, body: bytes) -> bool:
    if code in (429, 500, 502, 503, 504):
        return True
    text = body.decode(errors="replace").lower()
    return any(snippet in text for snippet in _RETRYABLE_MESSAGE_SNIPPETS)


def call_upstream_with_retry(chat_body: dict[str, Any], req_id: int) -> dict[str, Any]:
    for attempt in range(1, RETRY_MAX_ATTEMPTS + 1):
        upstream_req = urllib.request.Request(
            f"{UPSTREAM}/chat/completions",
            data=json.dumps(chat_body).encode(),
            method="POST",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(upstream_req, timeout=600) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read()
            if not _is_retryable(e.code, body) or attempt == RETRY_MAX_ATTEMPTS:
                raise UpstreamHTTPError(e.code, body) from None
            retry_after = e.headers.get("Retry-After")
            try:
                retry_after_seconds = float(retry_after) if retry_after else 0.0
            except ValueError:
                retry_after_seconds = 0.0
            delay = max(
                retry_after_seconds,
                min(RETRY_MAX_DELAY_SECONDS, RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1))),
            )
            sys.stderr.write(
                f"req={req_id} attempt {attempt}/{RETRY_MAX_ATTEMPTS} got retryable "
                f"{e.code}, retrying in {delay:.1f}s: {body[:200]!r}\n"
            )
            _debug(req_id, f"RETRY attempt={attempt} code={e.code} body={body[:500]!r}")
            time.sleep(delay)
    raise AssertionError("unreachable")  # loop always returns or raises


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

    @property
    def clean_path(self) -> str:
        return self.path.split("?", 1)[0].rstrip("/") or "/"

    def _send(self, code: int, body: bytes, content_type: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", "0") or "0")
        return self.rfile.read(length) if length > 0 else b""

    def do_GET(self) -> None:  # noqa: N802
        if self.clean_path in ("/", "/health"):
            self._send(200, b'{"ok":true}')
            return
        self._send(404, b'{"error":"not found"}')

    def do_POST(self) -> None:  # noqa: N802
        if not self.clean_path.endswith("/responses"):
            self._send(404, b'{"error":"not found; this proxy only forwards .../responses"}')
            return

        try:
            body = json.loads(self._read_body().decode() or "{}")
        except Exception as e:  # noqa: BLE001
            self._send(400, json.dumps({"error": {"type": "invalid_request", "message": str(e)}}).encode())
            return

        global _REQUEST_COUNTER
        _REQUEST_COUNTER += 1
        req_id = _REQUEST_COUNTER

        inline_additional_tools(body)
        model = map_model(body.get("model"))
        chat_body: dict[str, Any] = {
            "model": model,
            "messages": translate_input(body.get("input") or [], body.get("instructions")),
        }
        response_tools = body.get("tools") or []
        custom_tool_names = {
            str(tool.get("name"))
            for tool in response_tools
            if isinstance(tool, dict) and tool.get("type") == "custom" and tool.get("name")
        }
        tools = translate_tools(response_tools)
        if tools:
            chat_body["tools"] = tools
        if body.get("max_output_tokens"):
            chat_body["max_tokens"] = body["max_output_tokens"]

        sys.stderr.write(
            f"-> upstream model={body.get('model')}->{model} "
            f"messages={len(chat_body['messages'])} tools={len(tools)}\n"
        )
        _debug(req_id, f"CHAT REQUEST {json.dumps(chat_body, ensure_ascii=False)[:4000]}")

        try:
            chat_resp = call_upstream_with_retry(chat_body, req_id)
        except UpstreamHTTPError as e:
            sys.stderr.write(f"upstream HTTP {e.code} (out of retries): {e.body[:500]!r}\n")
            _debug(req_id, f"HTTPError {e.code} (out of retries): {e.body[:2000]!r}")
            self._send(e.code, e.body or json.dumps({"error": str(e)}).encode())
            return
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(f"proxy error: {e}\n")
            _debug(req_id, f"EXCEPTION {type(e).__name__}: {e}")
            self._send(500, json.dumps({"error": {"type": "proxy_error", "message": str(e)}}).encode())
            return

        _debug(req_id, f"CHAT RESPONSE {json.dumps(chat_resp, ensure_ascii=False)[:4000]}")
        resp_body = translate_response(
            chat_resp,
            body.get("model") or model,
            custom_tool_names=custom_tool_names,
        )
        if body.get("stream"):
            self._send_sse(resp_body)
        else:
            self._send(200, json.dumps(resp_body).encode())

    def _send_sse(self, response: dict[str, Any]) -> None:
        """Emit a complete Responses-API SSE lifecycle for Codex.

        Codex requests stream=true and, even after its WebSocket transport
        falls back to plain HTTPS, still expects a real text/event-stream
        response ending in a `response.completed` event -- a single JSON
        blob (what this proxy originally sent) makes it log "stream
        disconnected before completion: stream closed before
        response.completed" and silently retry the whole turn from scratch
        forever (observed: prompt_tokens identical across 6-10 retries,
        response content changes had zero effect). We don't stream real
        incremental deltas from the upstream chat/completions call (it's
        already a single non-streamed response by the time we have it) --
        response.completed event is necessary but not sufficient: Codex builds
        messages and tool calls from the incremental output-item events. Sending
        only created+completed makes it report a successful empty turn and drop
        an upstream tool call without executing it (reproduced with Codex
        0.145.0 and 0.152.1). Emit the same item/content/argument lifecycle as a
        native Responses stream, even though the upstream call was buffered.
        """
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def write_event(event_type: str, data: dict[str, Any]) -> None:
            payload = f"event: {event_type}\ndata: {json.dumps(data)}\n\n".encode()
            self.wfile.write(f"{len(payload):x}\r\n".encode())
            self.wfile.write(payload)
            self.wfile.write(b"\r\n")

        for event in response_sse_events(response):
            write_event(event["type"], event)
        self.wfile.write(b"0\r\n\r\n")


def response_sse_events(response: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand one buffered Responses object into Codex-consumable SSE events."""
    created_response = {**response, "output": [], "status": "in_progress"}
    events: list[dict[str, Any]] = [
        {"type": "response.created", "response": created_response}
    ]

    for output_index, item in enumerate(response.get("output") or []):
        item_type = item.get("type")
        in_progress = {**item, "status": "in_progress"}

        if item_type == "message":
            in_progress["content"] = []
        elif item_type == "function_call":
            in_progress["arguments"] = ""
        elif item_type == "custom_tool_call":
            in_progress["input"] = ""
        elif item_type == "reasoning":
            in_progress["summary"] = []

        events.append(
            {
                "type": "response.output_item.added",
                "output_index": output_index,
                "item": in_progress,
            }
        )

        if item_type == "message":
            for content_index, part in enumerate(item.get("content") or []):
                text = part.get("text") or ""
                empty_part = {**part, "text": ""}
                events.append(
                    {
                        "type": "response.content_part.added",
                        "output_index": output_index,
                        "content_index": content_index,
                        "part": empty_part,
                    }
                )
                if text:
                    events.append(
                        {
                            "type": "response.output_text.delta",
                            "output_index": output_index,
                            "content_index": content_index,
                            "delta": text,
                        }
                    )
                events.append(
                    {
                        "type": "response.output_text.done",
                        "output_index": output_index,
                        "content_index": content_index,
                        "text": text,
                    }
                )
                events.append(
                    {
                        "type": "response.content_part.done",
                        "output_index": output_index,
                        "content_index": content_index,
                        "part": part,
                    }
                )
        elif item_type == "function_call":
            arguments = item.get("arguments") or ""
            if arguments:
                events.append(
                    {
                        "type": "response.function_call_arguments.delta",
                        "output_index": output_index,
                        "item_id": item.get("id"),
                        "delta": arguments,
                    }
                )
            events.append(
                {
                    "type": "response.function_call_arguments.done",
                    "output_index": output_index,
                    "item_id": item.get("id"),
                    "arguments": arguments,
                }
            )
        elif item_type == "custom_tool_call":
            tool_input = item.get("input") or ""
            if tool_input:
                events.append(
                    {
                        "type": "response.custom_tool_call_input.delta",
                        "output_index": output_index,
                        "item_id": item.get("id"),
                        "delta": tool_input,
                    }
                )
            events.append(
                {
                    "type": "response.custom_tool_call_input.done",
                    "output_index": output_index,
                    "item_id": item.get("id"),
                    "input": tool_input,
                }
            )
        elif item_type == "reasoning":
            for summary_index, part in enumerate(item.get("summary") or []):
                text = part.get("text") or ""
                events.append(
                    {
                        "type": "response.reasoning_summary_part.added",
                        "item_id": item.get("id"),
                        "output_index": output_index,
                        "summary_index": summary_index,
                        "part": {**part, "text": ""},
                    }
                )
                if text:
                    events.append(
                        {
                            "type": "response.reasoning_summary_text.delta",
                            "item_id": item.get("id"),
                            "output_index": output_index,
                            "summary_index": summary_index,
                            "delta": text,
                        }
                    )
                events.append(
                    {
                        "type": "response.reasoning_summary_text.done",
                        "item_id": item.get("id"),
                        "output_index": output_index,
                        "summary_index": summary_index,
                        "text": text,
                    }
                )
                events.append(
                    {
                        "type": "response.reasoning_summary_part.done",
                        "item_id": item.get("id"),
                        "output_index": output_index,
                        "summary_index": summary_index,
                        "part": part,
                    }
                )

        events.append(
            {
                "type": "response.output_item.done",
                "output_index": output_index,
                "item": item,
            }
        )

    events.append({"type": "response.completed", "response": response})
    return events


def main() -> None:
    if not API_KEY:
        sys.stderr.write("codex_kimi_proxy: set KIMI_API_KEY (Meta/Llama API bearer token)\n")
        sys.exit(1)
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    sys.stderr.write(f"codex_kimi_proxy listening on {HOST}:{PORT} -> {UPSTREAM}/chat/completions\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
