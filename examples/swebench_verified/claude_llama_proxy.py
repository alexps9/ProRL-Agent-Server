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
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

HOST = os.environ.get("CLAUDE_LLAMA_PROXY_HOST", "127.0.0.1")
PORT = int(os.environ.get("CLAUDE_LLAMA_PROXY_PORT", "3456"))
UPSTREAM = os.environ.get(
    "CLAUDE_LLAMA_UPSTREAM",
    "https://api.llama.com/experimental/compat/openai/v1",
).rstrip("/")
# Required: set CLAUDE_LLAMA_API_KEY in the environment (no default).
API_KEY = os.environ.get("CLAUDE_LLAMA_API_KEY", "").strip()
DEFAULT_MODEL = os.environ.get("CLAUDE_LLAMA_MODEL", "claude-4-8-opus-genai")
FORCE_MODEL = os.environ.get("CLAUDE_LLAMA_FORCE_MODEL") or None
DEBUG = os.environ.get("CLAUDE_LLAMA_DEBUG", "1") not in ("0", "false", "False", "")
DEBUG_LOG_PATH = os.environ.get("CLAUDE_LLAMA_DEBUG_LOG", "/tmp/claude_llama_proxy_debug.log")

_REQUEST_COUNTER = 0


def _debug(req_id: int, msg: str) -> None:
    if not DEBUG:
        return
    try:
        with open(DEBUG_LOG_PATH, "a") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] req={req_id} {msg}\n")
    except OSError:
        pass

# Claude Code's public model ids -> MetaGen ids. The first three are the same
# model under a different name; the rest have no equivalent here (there is no
# Haiku tier at all) and are served by something else -- see _warn_substitution.
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

# Entries of MODEL_MAP that serve a genuinely different model than was asked for.
SUBSTITUTED_MODELS = {
    "claude-sonnet-4-5",
    "claude-haiku-4-5",
    "claude-3-5-haiku-latest",
    "claude-3-5-sonnet-latest",
    "claude-3-opus-latest",
}


def map_model(name: str | None) -> str:
    if FORCE_MODEL:
        return FORCE_MODEL
    if not name:
        return DEFAULT_MODEL
    if name in MODEL_MAP:
        if name in SUBSTITUTED_MODELS:
            _warn_substitution(name, MODEL_MAP[name])
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
        mapped = "fireworks-kimi-k3"
    elif "opus" in lower:
        mapped = "claude-4-8-opus-genai"
    elif "haiku" in lower or "sonnet" in lower:
        mapped = "claude-4-6-sonnet-genai"
    else:
        mapped = DEFAULT_MODEL
    _warn_substitution(name, mapped)
    return mapped


_SUBSTITUTIONS_SEEN: set[str] = set()


def _warn_substitution(requested: str, served: str) -> None:
    """A swapped model changes cost and behaviour -- say so, once per name.

    Only for genuine swaps: mapping `claude-opus-4-8` to its MetaGen id
    `claude-4-8-opus-genai` is a rename and stays quiet. Serving Claude Code's
    haiku-tier background calls with Sonnet, because no Haiku exists here, is not.
    """
    if requested == served or requested in _SUBSTITUTIONS_SEEN:
        return
    _SUBSTITUTIONS_SEEN.add(requested)
    sys.stderr.write(f"[model] {requested!r} is not available here; serving {served!r}\n")


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


def image_parts(content: Any) -> list[dict[str, Any]]:
    """Anthropic `image` blocks -> OpenAI `image_url` parts (data URI)."""
    if not isinstance(content, list):
        return []
    parts: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "image":
            continue
        src = block.get("source") or {}
        if src.get("type") == "base64" and src.get("data"):
            media = src.get("media_type", "image/png")
            parts.append({"type": "image_url",
                          "image_url": {"url": f"data:{media};base64,{src['data']}"}})
        elif src.get("type") == "url" and src.get("url"):
            parts.append({"type": "image_url", "image_url": {"url": src["url"]}})
    return parts


def anthropic_to_openai(body: dict[str, Any]) -> dict[str, Any]:

    messages: list[dict[str, Any]] = []
    system = body.get("system")
    if system:
        messages.append({"role": "system", "content": content_to_text(system)})

    for msg in body.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content")
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []
        images: list[dict[str, Any]] = list(image_parts(content))

        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    text_parts.append(str(block))
                    continue
                btype = block.get("type")
                if btype == "text":
                    text_parts.append(block.get("text", ""))
                elif btype == "tool_use":
                    # Real OpenAI tool_calls, not prose. Flattening these into
                    # "[Called tool `x` with input: ...]" makes the model read its
                    # own tool protocol as narration and, a few turns in, start
                    # *writing* calls as text instead of emitting them.
                    tool_calls.append(
                        {
                            "id": block.get("id") or f"call_{uuid.uuid4().hex[:24]}",
                            "type": "function",
                            "function": {
                                "name": block.get("name", ""),
                                "arguments": json.dumps(
                                    block.get("input") or {}, ensure_ascii=False
                                ),
                            },
                        }
                    )
                elif btype == "tool_result":
                    # OpenAI tool messages are text-only, so any image a tool
                    # returned rides along in a follow-up user turn instead.
                    images.extend(image_parts(block.get("content")))
                    result_text = content_to_text(block.get("content"))
                    if block.get("is_error"):
                        result_text = (
                            f"[Tool error] {result_text}" if result_text else "[Tool error]"
                        )
                    tool_results.append(
                        {
                            "role": "tool",
                            "tool_call_id": block.get("tool_use_id", ""),
                            "content": result_text,
                        }
                    )
        else:
            text_parts.append(content_to_text(content))

        text = "\n".join(p for p in text_parts if p)

        # tool_result blocks answer the preceding assistant turn, so they become
        # their own `role: tool` messages and must never be merged into a text turn.
        messages.extend(tool_results)

        if images:
            # Images never merge into a previous turn: order matters to the model.
            messages.append({
                "role": "user",
                "content": ([{"type": "text", "text": text}] if text else []) + images,
            })
            text = ""

        if role == "assistant" and tool_calls:
            messages.append(
                {"role": "assistant", "content": text or None, "tool_calls": tool_calls}
            )
            continue

        if not text:
            continue

        norm_role = "assistant" if role == "assistant" else "user"
        prev = messages[-1] if messages else None
        if prev and prev.get("role") == norm_role and not prev.get("tool_calls"):
            prev["content"] = f"{prev.get('content') or ''}\n\n{text}".strip()
        else:
            messages.append({"role": norm_role, "content": text})

    out_body: dict[str, Any] = {
        "model": map_model(body.get("model")),
        "messages": messages,
        "stream": bool(body.get("stream")),
    }
    if "max_tokens" in body:
        # Upstream hard limit is 128000 ("max_tokens: N > 128000"); clamping lower
        # would silently truncate long turns.
        out_body["max_tokens"] = min(int(body["max_tokens"]), 128000)
    # Claude 4.8+ reject an explicit `temperature` unless it is exactly the
    # default 1.0 ("deprecated for this model"). Sending the default is a no-op,
    # so drop it; anything else is a deliberate choice and stays, so that an
    # unsupported value fails loudly instead of being silently ignored.
    if "temperature" in body and body["temperature"] != 1:
        out_body["temperature"] = body["temperature"]
    if "stop_sequences" in body:
        out_body["stop"] = body["stop_sequences"]
    # `top_p` is rejected outright by Claude 4.8+ ("deprecated for this model") at
    # every value including the default, so an explicit one is forwarded and fails
    # loudly rather than being silently dropped. `top_k` is accepted as-is.
    if "top_p" in body:
        out_body["top_p"] = body["top_p"]
    if "top_k" in body:
        out_body["top_k"] = body["top_k"]

    # Extended thinking. The Anthropic `thinking` block is silently ignored by this
    # endpoint (accepted with 200, no effect); `reasoning_effort` is what actually
    # raises the thinking budget -- verified by completion_tokens rising 93 -> 172
    # across the tiers on a fixed prompt.
    effort = (body.get("output_config") or {}).get("effort")
    thinking = body.get("thinking") or {}
    if not effort and thinking.get("type") in ("enabled", "adaptive"):
        budget = thinking.get("budget_tokens")
        effort = "high" if budget is None else (
            "low" if budget <= 2048 else "medium" if budget <= 8192 else "high"
        )
    if effort:
        out_body["reasoning_effort"] = effort

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


def openai_to_anthropic(data: dict[str, Any], model: str) -> dict[str, Any]:
    """OpenAI chat completion -> Anthropic Messages response."""
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}

    content: list[dict[str, Any]] = []
    if message.get("content"):
        content.append({"type": "text", "text": message["content"]})
    for tc in message.get("tool_calls") or []:
        fn = tc.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        content.append(
            {
                "type": "tool_use",
                # Claude Code echoes this back as tool_use_id, so it has to be
                # globally unique -- a per-response counter collides across turns.
                "id": tc.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
                "name": fn.get("name", ""),
                "input": args if isinstance(args, dict) else {},
            }
        )

    stop_reason = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "end_turn",
    }.get(choice.get("finish_reason") or "stop", "end_turn")

    return {
        "id": data.get("id") or f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": prompt_completion_tokens(data.get("usage") or {}),
    }


def prompt_completion_tokens(usage: dict[str, Any]) -> dict[str, int]:
    """Anthropic-shaped usage from an OpenAI `usage` block.

    Upstream's `prompt_tokens` under-reports whenever the prompt is served from
    cache -- a 1390-token prompt comes back as 1. `total_tokens` stays correct, so
    the prompt side is derived from it and `prompt_tokens` is only a fallback.
    """
    completion = usage.get("completion_tokens", 0)
    total = usage.get("total_tokens")
    prompt = usage.get("prompt_tokens", 0)
    if isinstance(total, int) and total >= completion:
        prompt = max(prompt, total - completion)
    return {"input_tokens": prompt, "output_tokens": completion}


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
            self._send(200, json.dumps({"input_tokens": self._count_tokens(body)}).encode())
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
            ant = self._complete(oai, model, req_id)
            if stream:
                self._send_sse(ant)
            else:
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

    def _count_tokens(self, body: dict[str, Any]) -> int:
        """Exact prompt token count for Anthropic's /messages/count_tokens.

        The OpenAI-compat upstream has no token-counting endpoint, so the count is
        taken from the only authority there is: a 1-token completion of the same
        prompt, reading `total_tokens - completion_tokens`. Claude Code drives
        compaction off this number, and a guess (the old `len(json)//4`) was off by
        +26% to +88% on ordinary payloads.
        """
        oai = anthropic_to_openai(body)
        oai["stream"] = False
        oai["max_tokens"] = 1
        oai.pop("tool_choice", None)
        with upstream_request("/chat/completions", oai, stream=False) as resp:
            usage = json.loads(resp.read().decode()).get("usage") or {}
        return prompt_completion_tokens(usage)["input_tokens"]

    def _complete(self, oai: dict[str, Any], model: str, req_id: int) -> dict[str, Any]:
        """One upstream turn -> one Anthropic Messages response. Always non-streaming.

        The compat endpoint's *streaming* path drops the assistant `tool_calls`
        message as soon as the history contains a `role: tool` turn, and reports the
        failure as an un-framed JSON error inside an HTTP 200 SSE body -- which every
        conformant SSE parser silently discards, turning a hard failure into "the
        model said nothing". Its non-streaming path converts correctly, so we always
        fetch whole turns and, for streaming clients, synthesise the event stream.

        Cost is token-level TTFT only; total turn latency is unchanged (measured
        ~18s either way on a 600-word generation). For an agent loop that is free:
        a tool call is only actionable once its arguments JSON is complete.
        """
        payload = dict(oai)
        payload["stream"] = False
        with upstream_request("/chat/completions", payload, stream=False) as resp:
            raw = resp.read()
        data = json.loads(raw.decode())
        if DEBUG:
            _debug(req_id, f"RESPONSE {raw[:2000]!r}")

        # Anything that is not a completion -- an error body under HTTP 200, a
        # missing `choices` -- is a failure, and so is a turn carrying neither text
        # nor tool calls. Never hand any of them to the client as a valid empty turn.
        if not data.get("choices"):
            raise RuntimeError(f"upstream returned no completion: {raw[:500]!r}")
        ant = openai_to_anthropic(data, model)
        ant["content"] = [
            b for b in ant["content"]
            if b["type"] != "text" or (b.get("text") or "").strip()
        ]
        if not ant["content"]:
            raise RuntimeError(f"upstream returned an empty completion: {raw[:500]!r}")
        _debug(req_id, f"OK blocks={[b['type'] for b in ant['content']]} "
                       f"stop_reason={ant['stop_reason']} usage={ant['usage']}")
        return ant

    def _send_sse(self, ant: dict[str, Any]) -> None:
        """Serialise a complete Anthropic message as the SSE event sequence."""
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

        usage = ant.get("usage") or {}
        emit(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": ant["id"],
                    "type": "message",
                    "role": "assistant",
                    "model": ant["model"],
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": usage.get("input_tokens", 0),
                              "output_tokens": 0},
                },
            },
        )
        emit("ping", {"type": "ping"})

        for index, block in enumerate(ant["content"]):
            if block["type"] == "text":
                emit("content_block_start", {
                    "type": "content_block_start", "index": index,
                    "content_block": {"type": "text", "text": ""}})
                emit("content_block_delta", {
                    "type": "content_block_delta", "index": index,
                    "delta": {"type": "text_delta", "text": block["text"]}})
            else:
                emit("content_block_start", {
                    "type": "content_block_start", "index": index,
                    "content_block": {"type": "tool_use", "id": block["id"],
                                      "name": block["name"], "input": {}}})
                emit("content_block_delta", {
                    "type": "content_block_delta", "index": index,
                    "delta": {"type": "input_json_delta",
                              "partial_json": json.dumps(block["input"], ensure_ascii=False)}})
            emit("content_block_stop", {"type": "content_block_stop", "index": index})

        emit("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": ant["stop_reason"], "stop_sequence": None},
            "usage": {"output_tokens": usage.get("output_tokens", 0)}})
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
