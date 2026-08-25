#!/usr/bin/env python3
"""Local OpenAI Responses API proxy for the Codex CLI -> Meta Model API.

Unlike claude_llama_proxy.py, this needs no format translation: Codex CLI
already speaks the OpenAI Responses API (POST .../responses with
function_call / function_call_output items), and so does Meta's Model API
(https://api.ai.meta.com/v1/responses) -- see llm_keys/meta_key_modelapi.md.
This proxy only does two things a raw --openai-base-url can't:

1. Rewrite the client's dotted model id (e.g. "gpt-5.4", what the rest of
   this repo's --model-name defaults use) to the Model API's hyphenated id
   ("gpt-5-4"). Get this wrong and it's a 404 that looks identical to "no
   access" -- see meta_key_modelapi.md Troubleshooting.
2. Hold the real API key server-side and inject it, so it isn't threaded
   through Polar's task-submission args/env in plaintext.

Everything else -- request/response bodies, streaming -- is relayed as-is.

Requires env ``CODEX_META_API_KEY``. See ``codex_meta_proxy.env.example`` and
``start_codex_meta_proxy.sh``.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

HOST = os.environ.get("CODEX_META_PROXY_HOST", "127.0.0.1")
PORT = int(os.environ.get("CODEX_META_PROXY_PORT", "3457"))
UPSTREAM = os.environ.get("CODEX_META_UPSTREAM", "https://api.ai.meta.com/v1").rstrip("/")
# Required: set CODEX_META_API_KEY in the environment (no default).
API_KEY = os.environ.get("CODEX_META_API_KEY", "").strip()
DEBUG = os.environ.get("CODEX_META_DEBUG", "1") not in ("0", "false", "False", "")
DEBUG_LOG_PATH = os.environ.get("CODEX_META_DEBUG_LOG", "/tmp/codex_meta_proxy_debug.log")

# Codex's dotted ids (this repo's --model-name convention) -> Model API's
# hyphenated ids. From the GPT table in llm_keys/meta_key_modelapi.md.
MODEL_MAP = {
    "gpt-5.6-sol": "gpt-5-6-sol",
    "gpt-5.6-luna": "gpt-5-6-luna",
    "gpt-5.6-terra": "gpt-5-6-terra",
    "gpt-5.5": "gpt-5-5",
    "gpt-5.4": "gpt-5-4",
    "gpt-5.4-mini": "gpt-5-4-mini",
    "gpt-5.4-nano": "gpt-5-4-nano",
    "gpt-5.4-pro": "gpt-5-4-pro",
    "gpt-5.2": "gpt-5-2",
    "gpt-5.1": "gpt-5-1",
    "gpt-5": "gpt-5",
    "gpt-4.1": "gpt-4-1",
    "gpt-4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-o3": "gpt-o3",
    "gpt-o3-pro": "gpt-o3-pro",
    "gpt-o4-mini": "gpt-o4-mini",
}

# Fallback for a dotted id not in the table above: hyphenate the leading
# "gpt-<digit>(.<digit>)*" version run and pass the rest through untouched.
_VERSION_RUN_RE = re.compile(r"^gpt-\d+(?:\.\d+)*")

_SUBSTITUTIONS_SEEN: set[str] = set()


def map_model(name: str | None) -> str:
    if not name:
        return name
    if name in MODEL_MAP:
        return MODEL_MAP[name]
    if name in MODEL_MAP.values():
        return name  # already a Model API id
    mapped = _VERSION_RUN_RE.sub(lambda m: m.group(0).replace(".", "-"), name)
    if mapped != name and name not in _SUBSTITUTIONS_SEEN:
        _SUBSTITUTIONS_SEEN.add(name)
        sys.stderr.write(f"codex_meta_proxy: no MODEL_MAP entry for {name!r}, guessed {mapped!r}\n")
    return mapped


def inline_additional_tools(body: dict[str, Any]) -> None:
    """Fold `input` items of type "additional_tools" into the top-level `tools` array.

    Newer Codex (observed with unified_exec / the JS "exec" orchestration
    tool) declares its tool set via an `input[0]` item shaped like
    `{"type": "additional_tools", "role": "developer", "tools": [...]}`
    instead of the top-level `tools` field. The Model API doesn't recognize
    that item type at all and 400s with "`input[0]` did not match any
    supported type" before ever looking at the tools inside it. Extract and
    merge them into `tools` (the shape is identical to a normal tool entry),
    then drop the item so the rest of the request looks like the old-style
    request the Model API does understand.
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


def _strictify_schema(schema: Any) -> None:
    """Fill in `required` with every `properties` key, recursively.

    Meta's Model API validates function-call schemas in OpenAI "strict mode"
    unconditionally: `required` must list every key in `properties`. The
    public OpenAI API only enforces that when a tool opts into
    `strict: true`; Codex's built-in tools (tool_search, exec_command, ...)
    leave genuinely-optional params (e.g. tool_search's `limit`, "Defaults
    to 8") out of `required` and rely on that leniency. Against the Model
    API this 400s with e.g. "Missing 'limit'." Codex's tools are otherwise
    already strict-shaped (additionalProperties: false), so completing
    `required` is the only gap.
    """
    if not isinstance(schema, dict):
        return
    props = schema.get("properties")
    if isinstance(props, dict) and props:
        schema["required"] = list(props.keys())
        for sub in props.values():
            _strictify_schema(sub)
    items = schema.get("items")
    if items is not None:
        _strictify_schema(items)


def strictify_tools(tools: list[Any]) -> None:
    for t in tools:
        if isinstance(t, dict) and isinstance(t.get("parameters"), dict):
            _strictify_schema(t["parameters"])


# Codex offers "web_search" as an optional built-in hosted tool. The Model
# API's web_search[_preview] executor doesn't implement it fully -- it 400s
# on Codex's `search_content_types` under either type name, and SWE-bench
# tasks run in a network-isolated container with no use for live search
# anyway -- so just drop it rather than chase its executor's quirks.
_DROPPED_TOOL_TYPES = {"web_search", "web_search_preview"}


def drop_unsupported_tools(tools: list[Any]) -> list[Any]:
    return [t for t in tools if not (isinstance(t, dict) and t.get("type") in _DROPPED_TOOL_TYPES)]


def _debug(req_id: int, msg: str) -> None:
    if not DEBUG:
        return
    try:
        with open(DEBUG_LOG_PATH, "a") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] req={req_id} {msg}\n")
    except OSError:
        pass


_REQUEST_COUNTER = 0


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

        requested_model = body.get("model")
        body["model"] = map_model(requested_model)
        inline_additional_tools(body)
        if body.get("tools"):
            body["tools"] = drop_unsupported_tools(body["tools"])
        strictify_tools(body.get("tools") or [])
        stream = bool(body.get("stream"))
        sys.stderr.write(
            f"-> upstream model={requested_model}->{body['model']} stream={stream} "
            f"tools={len(body.get('tools') or [])}\n"
        )
        _debug(req_id, f"REQUEST model={body['model']} stream={stream} keys={sorted(body.keys())}")
        if body.get("tools"):
            _debug(req_id, "TOOLS " + json.dumps(body["tools"], ensure_ascii=False))

        upstream_req = urllib.request.Request(
            f"{UPSTREAM}/responses",
            data=json.dumps(body).encode(),
            method="POST",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
                "Accept": self.headers.get("Accept", "application/json"),
            },
        )
        try:
            with urllib.request.urlopen(upstream_req, timeout=600) as resp:
                self._relay(resp, resp.status, req_id)
        except urllib.error.HTTPError as e:
            err = e.read()
            sys.stderr.write(f"upstream HTTP {e.code}: {err[:500]!r}\n")
            _debug(req_id, f"HTTPError {e.code}: {err[:2000]!r}")
            _debug(req_id, "FULL REQUEST ON ERROR " + json.dumps(body, ensure_ascii=False)[:6000])
            self._send(e.code, err or json.dumps({"error": str(e)}).encode())
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(f"proxy error: {e}\n")
            _debug(req_id, f"EXCEPTION {type(e).__name__}: {e}")
            self._send(500, json.dumps({"error": {"type": "proxy_error", "message": str(e)}}).encode())

    def _relay(self, resp, status: int, req_id: int) -> None:
        """Byte-for-byte relay: no translation, upstream and client speak the same API."""
        content_type = resp.headers.get("Content-Type", "application/json")
        if "text/event-stream" in content_type:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            total = 0
            while True:
                chunk = resp.read(8192)
                if not chunk:
                    break
                total += len(chunk)
                self.wfile.write(f"{len(chunk):x}\r\n".encode())
                self.wfile.write(chunk)
                self.wfile.write(b"\r\n")
            self.wfile.write(b"0\r\n\r\n")
            _debug(req_id, f"RESPONSE stream bytes={total}")
        else:
            data = resp.read()
            _debug(req_id, f"RESPONSE {status} bytes={len(data)}")
            self._send(status, data, content_type)


def main() -> None:
    if not API_KEY:
        sys.stderr.write("codex_meta_proxy: set CODEX_META_API_KEY (Meta Model API bearer token)\n")
        sys.exit(1)
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    sys.stderr.write(f"codex_meta_proxy listening on {HOST}:{PORT} -> {UPSTREAM}/responses\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
