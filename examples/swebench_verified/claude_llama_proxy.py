#!/usr/bin/env python3
"""Local Anthropic Messages API -> Llama OpenAI-compat proxy for Claude Code.

Requires env ``CLAUDE_LLAMA_API_KEY``. See ``claude_llama_proxy.env.example`` and
``start_claude_llama_proxy.sh``.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

HOST = os.environ.get("CLAUDE_LLAMA_PROXY_HOST", "127.0.0.1")
DEBUG = os.environ.get("CLAUDE_LLAMA_DEBUG", "1") not in ("0", "false", "False", "")
DEBUG_LOG_PATH = os.environ.get("CLAUDE_LLAMA_DEBUG_LOG", "/tmp/claude_llama_proxy_debug.log")
_REQUEST_COUNTER = 0


def _debug(req_id: int, msg: str) -> None:
    if not DEBUG:
        return
    line = f"[{time.strftime('%H:%M:%S')}] req={req_id} {msg}\n"
    try:
        with open(DEBUG_LOG_PATH, "a") as f:
            f.write(line)
    except OSError:
        pass
PORT = int(os.environ.get("CLAUDE_LLAMA_PROXY_PORT", "3456"))
UPSTREAM = os.environ.get(
    "CLAUDE_LLAMA_UPSTREAM",
    "https://api.llama.com/experimental/compat/openai/v1",
).rstrip("/")
# Required: set CLAUDE_LLAMA_API_KEY in the environment (no default).
API_KEY = os.environ.get("CLAUDE_LLAMA_API_KEY", "").strip()
DEFAULT_MODEL = os.environ.get("CLAUDE_LLAMA_MODEL", "claude-4-8-opus-genai")

FORCE_MODEL = os.environ.get("CLAUDE_LLAMA_FORCE_MODEL") or None

MODEL_MAP = {
    "claude-opus-4-8": "claude-4-8-opus-genai",
    "claude-opus-4-6": "claude-4-6-opus-genai",
    "claude-sonnet-4-6": "claude-4-6-sonnet-genai",
    "claude-sonnet-4-5": "claude-4-6-sonnet-genai",
    "claude-haiku-4-5": "claude-4-6-sonnet-genai",
    "claude-3-5-haiku-latest": "claude-4-6-sonnet-genai",
    "claude-3-5-sonnet-latest": "claude-4-6-sonnet-genai",
    "claude-3-opus-latest": "claude-4-8-opus-genai",
    "fireworks-kimi-k3": "fireworks-kimi-k3",
}


def map_model(name: str | None) -> str:
    if FORCE_MODEL:
        return FORCE_MODEL
    if not name:
        return DEFAULT_MODEL
    if name in MODEL_MAP:
        return MODEL_MAP[name]
    # Pass through MetaGen names / already-mapped ids
    if (
        name.endswith("-genai")
        or name.startswith("fireworks-")
        or name in MODEL_MAP.values()
    ):
        return name
    # Fuzzy: prefer opus/sonnet/haiku buckets
    lower = name.lower()
    if "kimi" in lower:
        return "fireworks-kimi-k3"
    if "opus" in lower:
        return "claude-4-8-opus-genai"
    if "haiku" in lower:
        return "claude-4-6-sonnet-genai"
    if "sonnet" in lower:
        return "claude-4-6-sonnet-genai"
    return DEFAULT_MODEL


def content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif block.get("type") == "tool_result":
                    c = block.get("content", "")
                    parts.append(c if isinstance(c, str) else json.dumps(c))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)


def anthropic_to_openai(body: dict[str, Any]) -> dict[str, Any]:

    messages: list[dict[str, Any]] = []
    system = body.get("system")
    if system:
        messages.append({"role": "system", "content": content_to_text(system)})

    for msg in body.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content")
        text_parts: list[str] = []
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    text_parts.append(str(block))
                    continue
                btype = block.get("type")
                if btype == "text":
                    text_parts.append(block.get("text", ""))
                elif btype == "tool_use":
                    text_parts.append(
                        f"[Called tool `{block.get('name', '')}` with input: "
                        f"{json.dumps(block.get('input', {}), ensure_ascii=False)}]"
                    )
                elif btype == "tool_result":
                    label = "Tool error" if block.get("is_error") else "Tool result"
                    text_parts.append(f"[{label}]: {content_to_text(block.get('content'))}")
        else:
            text_parts.append(content_to_text(content))

        text = "\n".join(p for p in text_parts if p)
        norm_role = "assistant" if role == "assistant" else "user"
        if messages and messages[-1]["role"] == norm_role:
            if text:
                messages[-1]["content"] = f"{messages[-1]['content']}\n\n{text}"
        else:
            messages.append({"role": norm_role, "content": text})

    out_body: dict[str, Any] = {
        "model": map_model(body.get("model")),
        "messages": messages,
        "stream": bool(body.get("stream")),
    }
    if "max_tokens" in body:
        # Cap oversized Claude Code defaults for upstream gateways
        out_body["max_tokens"] = min(int(body["max_tokens"]), 16384)
    if "temperature" in body:
        out_body["temperature"] = body["temperature"]
    if "stop_sequences" in body:
        out_body["stop"] = body["stop_sequences"]

    tools = body.get("tools")
    if tools:
        cleaned_tools = []
        for t in tools:
            if not t.get("name"):
                continue
            schema = dict(t.get("input_schema") or {})
            schema.pop("$schema", None)
            cleaned_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": t.get("name", ""),
                        "description": (t.get("description") or "")[:8000],
                        "parameters": schema,
                    },
                }
            )
        if cleaned_tools:
            out_body["tools"] = cleaned_tools
    if body.get("tool_choice"):
        tc = body["tool_choice"]
        if tc == "auto":
            out_body["tool_choice"] = "auto"
        elif tc == "any":
            out_body["tool_choice"] = "required"
        elif isinstance(tc, dict) and tc.get("type") == "tool":
            out_body["tool_choice"] = {
                "type": "function",
                "function": {"name": tc.get("name", "")},
            }

    return out_body


def split_tool_call_args(
    name: str, tool_id: str | None, args_text: str
) -> list[dict[str, Any]]:

    text = (args_text or "").strip()
    if not text:
        return [{"id": tool_id, "name": name, "input": {}}]

    decoder = json.JSONDecoder()
    results: list[dict[str, Any]] = []
    idx = 0
    first = True
    while idx < len(text):
        while idx < len(text) and text[idx] in " \t\r\n":
            idx += 1
        if idx >= len(text):
            break
        try:
            obj, end = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            break
        if first:
            results.append({"id": tool_id, "name": name, "input": obj if isinstance(obj, dict) else {}})
            first = False
        elif isinstance(obj, dict) and obj.get("type") == "tool_use":
            results.append(
                {
                    "id": obj.get("id") or tool_id,
                    "name": obj.get("name") or name,
                    "input": obj.get("input") if isinstance(obj.get("input"), dict) else {},
                }
            )
        idx = end
    if not results:
        results.append({"id": tool_id, "name": name, "input": {}})
    return results


def openai_to_anthropic(data: dict[str, Any], model: str) -> dict[str, Any]:
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    text = message.get("content") or ""
    content: list[dict[str, Any]] = []
    if text:
        content.append({"type": "text", "text": text})
    for i, tc in enumerate(message.get("tool_calls") or []):
        fn = tc.get("function") or {}
        for j, call in enumerate(
            split_tool_call_args(fn.get("name", ""), tc.get("id"), fn.get("arguments") or "{}")
        ):
            content.append(
                {
                    "type": "tool_use",
                    "id": call["id"] or f"toolu_{i}_{j}",
                    "name": call["name"],
                    "input": call["input"],
                }
            )
    if not content:
        content = [{"type": "text", "text": ""}]

    finish = choice.get("finish_reason") or "stop"
    stop_reason = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "end_turn",
    }.get(finish, "end_turn")

    usage = data.get("usage") or {}
    return {
        "id": data.get("id", "msg_proxy"),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


def upstream_request(path: str, payload: dict[str, Any] | None, stream: bool = False):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{UPSTREAM}{path}",
        data=data,
        method="GET" if payload is None else "POST",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if stream else "application/json",
        },
    )
    return urllib.request.urlopen(req, timeout=600)


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
        if length <= 0:
            return b""
        return self.rfile.read(length)

    def _read_json(self) -> dict[str, Any]:
        raw = self._read_body()
        if not raw:
            return {}
        return json.loads(raw.decode())

    def do_HEAD(self) -> None:  # noqa: N802
        # Claude Code probes connectivity via HEAD /api/hello
        if self.clean_path in ("/api/hello", "/health", "/"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", "11")
            self.end_headers()
            return
        self.send_response(404)
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = self.clean_path
        if path in ("/health", "/", "/api/hello"):
            self._send(200, b'{"ok":true}')
            return
        if path.startswith("/v1/models"):
            models = [
                {"id": k, "object": "model", "owned_by": "metagen"}
                for k in sorted(set(MODEL_MAP) | {DEFAULT_MODEL})
            ]
            self._send(200, json.dumps({"object": "list", "data": models}).encode())
            return
        self._send(404, b'{"error":"not found"}')

    def do_POST(self) -> None:  # noqa: N802
        path = self.clean_path
        # Always consume body so keep-alive connections stay aligned
        try:
            body = self._read_json()
        except Exception as e:  # noqa: BLE001
            self._send(400, json.dumps({"error": {"type": "invalid_request", "message": str(e)}}).encode())
            return

        if path.endswith("/messages/count_tokens"):
            text = json.dumps(body)
            est = max(1, len(text) // 4)
            self._send(200, json.dumps({"input_tokens": est}).encode())
            return

        if not path.endswith("/messages"):
            self._send(404, b'{"error":"not found"}')
            return

        global _REQUEST_COUNTER
        _REQUEST_COUNTER += 1
        req_id = _REQUEST_COUNTER

        oai = anthropic_to_openai(body)
        model = oai["model"]
        stream = bool(oai.get("stream"))
        sys.stderr.write(f"→ upstream model={model} stream={stream} tools={len(oai.get('tools') or [])}\n")
        if DEBUG:
            msgs = oai.get("messages", [])
            roles = [m.get("role") for m in msgs]
            tail = [m for m in msgs if m.get("role") in ("assistant", "tool")][-2:]
            tail_json = json.dumps(tail, ensure_ascii=False)
            _debug(req_id, f"REQUEST roles={roles} tail_messages={tail_json[:3000]}")

        try:
            if stream:
                self._proxy_stream(oai, model, req_id)
            else:
                with upstream_request("/chat/completions", oai, stream=False) as resp:
                    raw = resp.read()
                data = json.loads(raw.decode())
                if DEBUG:
                    _debug(req_id, f"RESPONSE (non-stream) {raw[:2000]!r}")
                ant = openai_to_anthropic(data, model)
                self._send(200, json.dumps(ant).encode())
        except urllib.error.HTTPError as e:
            err = e.read()
            sys.stderr.write(f"upstream HTTP {e.code}: {err[:500]!r}\n")
            _debug(req_id, f"HTTPError {e.code}: {err[:2000]!r}")
            self._send(e.code, err or json.dumps({"error": str(e)}).encode())
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(f"proxy error: {e}\n")
            _debug(req_id, f"EXCEPTION {type(e).__name__}: {e}")
            self._send(500, json.dumps({"error": {"type": "proxy_error", "message": str(e)}}).encode())

    def _proxy_stream(self, oai: dict[str, Any], model: str, req_id: int = 0) -> None:
        # Connect first so failures can still return JSON errors
        resp = upstream_request("/chat/completions", oai, stream=True)

        msg_id = "msg_proxy_stream"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        def emit(event: str, payload: dict[str, Any]) -> None:
            chunk = f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()
            self.wfile.write(chunk)
            self.wfile.flush()

        emit(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            },
        )
        emit("ping", {"type": "ping"})

        stop_reason = "end_turn"
        output_tokens = 0
        next_index = 0
        text_index: int | None = None
        # OpenAI tool-call index -> raw accumulated {id, name, args}. We buffer
        # the whole thing (instead of streaming input_json_delta live) because
        # the upstream sometimes concatenates a second tool call's raw JSON
        # onto this one's arguments string; see split_tool_call_args().
        tool_raw: dict[int, dict[str, Any]] = {}
        raw_lines_seen = 0
        data_lines_seen = 0
        error_chunk: dict[str, Any] | None = None

        try:
            while True:
                raw = resp.readline()
                if not raw:
                    break
                raw_lines_seen += 1
                line = raw.decode("utf-8", errors="replace").strip()
                if DEBUG and line:
                    _debug(req_id, f"SSE line: {line[:1000]}")
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                data_lines_seen += 1
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                if chunk.get("error"):
                    error_chunk = chunk
                    _debug(req_id, f"upstream error chunk: {chunk}")

                choice = (chunk.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}
                text = delta.get("content") or ""
                if text:
                    if text_index is None:
                        text_index = next_index
                        next_index += 1
                        emit(
                            "content_block_start",
                            {
                                "type": "content_block_start",
                                "index": text_index,
                                "content_block": {"type": "text", "text": ""},
                            },
                        )
                    emit(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": text_index,
                            "delta": {"type": "text_delta", "text": text},
                        },
                    )

                for tc in delta.get("tool_calls") or []:
                    oi = int(tc.get("index", 0))
                    fn = tc.get("function") or {}
                    if oi not in tool_raw:
                        tool_raw[oi] = {"id": None, "name": "", "args": ""}
                    raw = tool_raw[oi]
                    if tc.get("id"):
                        raw["id"] = tc["id"]
                    if fn.get("name"):
                        raw["name"] += fn["name"]
                    if fn.get("arguments"):
                        raw["args"] += fn["arguments"]

                fr = choice.get("finish_reason")
                if fr:
                    stop_reason = {
                        "stop": "end_turn",
                        "length": "max_tokens",
                        "tool_calls": "tool_use",
                    }.get(fr, "end_turn")
                usage = chunk.get("usage") or {}
                if usage.get("completion_tokens"):
                    output_tokens = usage["completion_tokens"]
        finally:
            resp.close()

        if text_index is not None:
            emit("content_block_stop", {"type": "content_block_stop", "index": text_index})

        any_tool_emitted = False
        for oi in sorted(tool_raw):
            raw = tool_raw[oi]
            if not raw["name"] and not raw["args"]:
                continue
            calls = split_tool_call_args(raw["name"], raw["id"], raw["args"])
            if len(calls) > 1:
                _debug(
                    req_id,
                    f"split {len(calls)} tool calls out of oi={oi} name={raw['name']!r} "
                    f"(upstream glued extra tool_use JSON onto arguments)",
                )
            for call in calls:
                block_index = next_index
                next_index += 1
                any_tool_emitted = True
                tool_id = call["id"] or f"toolu_{oi}_{block_index}"
                emit(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": block_index,
                        "content_block": {
                            "type": "tool_use",
                            "id": tool_id,
                            "name": call["name"],
                            "input": {},
                        },
                    },
                )
                emit(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": block_index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": json.dumps(call["input"]),
                        },
                    },
                )
                emit("content_block_stop", {"type": "content_block_stop", "index": block_index})

        # If model returned nothing, emit empty text so clients still get a block
        if text_index is None and not any_tool_emitted:
            empty_text = ""
            if error_chunk is not None:
                empty_text = f"[proxy: upstream error {json.dumps(error_chunk)[:500]}]"
            elif data_lines_seen == 0:
                empty_text = "[proxy: upstream stream closed with no data lines]"
            _debug(
                req_id,
                "EMPTY completion: "
                f"raw_lines={raw_lines_seen} data_lines={data_lines_seen} "
                f"error_chunk={error_chunk} stop_reason={stop_reason}",
            )
            emit(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            )
            if empty_text:
                emit(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": empty_text},
                    },
                )
            emit("content_block_stop", {"type": "content_block_stop", "index": 0})
        else:
            _debug(
                req_id,
                f"OK completion: text={text_index is not None} tools={len(tool_raw)} "
                f"stop_reason={stop_reason} data_lines={data_lines_seen}",
            )

        emit(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": output_tokens},
            },
        )
        emit("message_stop", {"type": "message_stop"})


def main() -> None:
    if not API_KEY:
        print("CLAUDE_LLAMA_API_KEY is required", file=sys.stderr)
        sys.exit(1)
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(
        f"Claude→Llama proxy on http://{HOST}:{PORT} -> {UPSTREAM} model={DEFAULT_MODEL}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye", flush=True)


if __name__ == "__main__":
    main()
