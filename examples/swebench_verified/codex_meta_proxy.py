#!/usr/bin/env python3
"""Local OpenAI Responses API proxy for the Codex CLI -> Meta Model API.

Codex CLI and Meta's Model API (https://api.ai.meta.com/v1/responses) both
speak the OpenAI Responses API, so no format translation -- but Codex leans
on several proprietary extensions the Model API doesn't implement, so this
proxy does more than pass through:

1. Rewrite the client's dotted model id (e.g. "gpt-5.4", what the rest of
   this repo's --model-name defaults use) to the Model API's hyphenated id
   ("gpt-5-4"). Get this wrong and it's a 404 that looks identical to "no
   access" -- see meta_key_modelapi.md Troubleshooting.
2. Hold the real API key server-side and inject it, so it isn't threaded
   through Polar's task-submission args/env in plaintext.
3. Request schema fix-ups: inline_additional_tools, strictify_tools,
   drop_unsupported_tools (see 08-25 / 08-23 work notes).
4. multi_agent_v2 support: flatten_namespace_tools + normalize_agent_messages
   on the request, split_namespaced_calls on the response -- Codex sends the
   sub-agent collaboration tools as a `type:"namespace"` entry and threads the
   inter-agent conversation as `agent_message` items, neither of which the
   Model API recognises.
5. Buffer + re-synthesise: fetch turns non-streamed and rebuild the SSE
   lifecycle for Codex (which always sets stream=true), so responses can be
   rewritten cleanly -- same approach as claude_llama_proxy / codex_kimi_proxy.

Requires env ``CODEX_META_API_KEY``. See ``codex_meta_proxy.env.example`` and
``start_codex_meta_proxy.sh``.
"""

from __future__ import annotations

import collections
import json
import os
import re
import sys
import threading
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


def flatten_namespace_tools(tools: list[Any]) -> list[Any]:
    """Hoist `{"type": "namespace", "name": ..., "tools": [...]}` entries to top level.

    multi_agent_v2 packs the request `tools` into `type: "namespace"` entries
    -- `functions` (Codex's own exec / wait / request_user_input) and
    `collaboration` (spawn_agent, wait_agent, followup_task, send_message,
    list_agents, interrupt_agent). Meta's Model API doesn't recognise that
    entry type -- no 400, it silently drops it, so the model sees no tools and
    reports "the collaboration tool surface isn't exposed in this session".

    Lift the inner entries to the top level (same idea as
    inline_additional_tools), with fix-ups:

    * `functions` is the default namespace -- keep its tools bare (`exec`), as
      Codex's router expects them and as the Model API wants them.
    * For `collaboration`, keep the qualified name (`collaboration.spawn_agent`):
      Codex's router is keyed on it -- a bare `spawn_agent` call comes back as
      "unsupported call: spawn_agent". split_namespaced_calls() turns the
      dotted name in the response into name + `namespace` on the way back.
    * Strip `"encrypted": true` from collaboration params (Codex marks the
      `message` field so). With it set the Model API returns that field as a
      Fernet blob and threads it back as an `encrypted_content` message part,
      which it then rejects ("`input[N]` did not match any supported type").
    """
    out: list[Any] = []
    for t in tools:
        if isinstance(t, dict) and t.get("type") == "namespace":
            ns = t.get("name") or ""
            for inner in t.get("tools") or []:
                if isinstance(inner, dict):
                    inner = {**inner}
                    if ns and ns != "functions" and isinstance(inner.get("name"), str):
                        inner["name"] = f"{ns}.{inner['name']}"
                    props = (inner.get("parameters") or {}).get("properties")
                    if isinstance(props, dict):
                        for spec in props.values():
                            if isinstance(spec, dict):
                                spec.pop("encrypted", None)
                out.append(inner)
        else:
            out.append(t)
    return out


def strip_functions_prefix(obj: Any) -> None:
    """Drop the `functions.` namespace prefix from tool names, everywhere.

    flatten_namespace_tools() already keeps the `functions` namespace's tools
    bare in the request `tools`, but Codex still writes its *history* items
    (`custom_tool_call` / `function_call`) with the qualified `functions.exec`
    name, and the Model API 400s on those ("`input[N]` did not match any
    supported type"). `functions` is the default namespace, so strip it from
    every `name` in the request and the response -- Codex's router takes the
    bare `exec` back fine.
    """
    if isinstance(obj, dict):
        n = obj.get("name")
        if isinstance(n, str) and n.startswith("functions."):
            obj["name"] = n[len("functions."):]
        for v in obj.values():
            strip_functions_prefix(v)
    elif isinstance(obj, list):
        for v in obj:
            strip_functions_prefix(v)


def split_namespaced_calls(obj: Any) -> None:
    """In-place: split a dotted `function_call` name into name + `namespace`.

    flatten_namespace_tools() sends the collaboration tools to the Model API as
    flat functions named `collaboration.spawn_agent`. The Model API echoes that
    dotted name straight back in the `function_call`. Codex's tool router,
    though, dispatches a namespaced call by matching a bare `name`
    (`spawn_agent`) against a separate `namespace` field on the item -- a
    dotted `name` with no `namespace` is rejected as
    "unsupported call: collaboration.spawn_agent". So walk the response and,
    for every function_call whose name is `<ns>.<tool>`, rewrite it to
    `{"name": "<tool>", "namespace": "<ns>"}`.
    """
    if isinstance(obj, dict):
        if obj.get("type") == "function_call" and isinstance(obj.get("name"), str) and "." in obj["name"] and not obj.get("namespace"):
            ns, _, tool = obj["name"].partition(".")
            obj["name"] = tool
            obj["namespace"] = ns
        for v in obj.values():
            split_namespaced_calls(v)
    elif isinstance(obj, list):
        for v in obj:
            split_namespaced_calls(v)


# Inter-agent message payloads. Codex encrypts every spawn_agent /
# send_message / followup_task `message` and threads it to the recipient as an
# `encrypted_content` part of an `agent_message` item -- the Model API can't
# decrypt it (not its key), so the sub-agent receives a blob and flails. We
# can't decrypt it either, but the plaintext `message` went by in the response
# that made the call. Record those per (session, recipient) in order; on a
# later request, the k-th encrypted agent_message to a recipient gets the k-th
# recorded message (positional, not pop -- Codex re-sends the whole thread
# every turn, so the same item is normalised repeatedly).
_SENT_MSGS: dict[tuple[str, str], list[str]] = collections.defaultdict(list)
_PENDING_LOCK = threading.Lock()


def _session_key(body: dict[str, Any]) -> str:
    """Stable-ish id for one Codex thread tree, to scope _SENT_MSGS so paths
    like /root/checker don't collide across tasks. prompt_cache_key is set
    per Codex session; fall back to a hash of the leading developer text."""
    k = body.get("prompt_cache_key")
    if isinstance(k, str) and k:
        return k
    for it in body.get("input") or []:
        if isinstance(it, dict) and it.get("type") == "message":
            for p in it.get("content") or []:
                if isinstance(p, dict) and p.get("text"):
                    return str(hash(p["text"][:2000]))
    return "_"


def _current_agent_path(inp: list[Any]) -> str:
    """Best guess at whose turn this request is: the recipient of the last
    inter-agent message in `input`, else /root."""
    for it in reversed(inp):
        if isinstance(it, dict) and it.get("type") == "agent_message" and it.get("recipient"):
            return str(it["recipient"])
    return "/root"


def _canon_target(target: str, sender_path: str) -> str:
    target = (target or "").strip()
    if target.startswith("/"):
        return target
    return f"{sender_path.rstrip('/')}/{target}" if target else sender_path


def record_outgoing_agent_messages(response_obj: dict[str, Any], req_input: list[Any], session: str) -> None:
    """Record plaintext `message` values from collab tool calls in this response,
    per (session, recipient agent path), for later payload re-injection."""
    sender = _current_agent_path(req_input if isinstance(req_input, list) else [])
    for item in response_obj.get("output") or []:
        if not (isinstance(item, dict) and item.get("type") == "function_call"):
            continue
        name = item.get("name")
        if name not in ("spawn_agent", "send_message", "followup_task"):
            continue
        try:
            args = json.loads(item.get("arguments") or "{}")
        except ValueError:
            continue
        msg = args.get("message")
        if not isinstance(msg, str) or not msg:
            continue
        if name == "spawn_agent":
            target = _canon_target(args.get("task_name", ""), sender)
        else:
            target = _canon_target(args.get("target", ""), sender)
        with _PENDING_LOCK:
            _SENT_MSGS[(session, target)].append(msg)
        if DEBUG:
            sys.stderr.write(f"[pending] +{name} sender={sender!r} target={target!r} msg={msg[:50]!r}\n")


def normalize_agent_messages(body: dict[str, Any], session: str) -> None:
    """Rewrite Codex inter-agent `agent_message` input items to plain messages.

    multi_agent_v2 threads the /root <-> sub-agent conversation as `input`
    items of type "agent_message" (author / recipient / an `encrypted_content`
    payload part). The Model API rejects that top-level item type
    ("`input[N]` did not match any supported type"), so convert the wrapper to
    `{"type": "message", "role": "user", ...}`. For the encrypted payload part,
    swap in the plaintext `message` recorded from the spawn/send call that
    produced it -- the k-th encrypted message to a recipient maps to the k-th
    recorded one (record_outgoing_agent_messages). Placeholder if we don't
    have it.
    """
    inp = body.get("input")
    if not isinstance(inp, list):
        return
    seen: dict[str, int] = collections.defaultdict(int)
    for i, it in enumerate(inp):
        if not (isinstance(it, dict) and it.get("type") == "agent_message"):
            continue
        recipient = str(it.get("recipient") or "")
        parts: list[Any] = []
        for p in it.get("content") or []:
            if not isinstance(p, dict):
                continue
            if p.get("type") == "encrypted_content":
                k = seen[recipient]
                seen[recipient] += 1
                with _PENDING_LOCK:
                    sent = _SENT_MSGS.get((session, recipient)) or []
                    text = sent[k] if k < len(sent) else None
                if DEBUG:
                    sys.stderr.write(f"[pending] lookup recipient={recipient!r} k={k} hit={text is not None}\n")
                parts.append({"type": "input_text", "text": text or "[cross-agent payload unavailable -- ask the sender to resend as plain text]"})
            else:
                parts.append(p)
        inp[i] = {"type": "message", "role": "user", "content": parts or [{"type": "input_text", "text": ""}]}


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
        if not (isinstance(t, dict) and isinstance(t.get("parameters"), dict)):
            continue
        # Skip the flattened collaboration tools (dotted name -- see
        # flatten_namespace_tools). Forcing their genuinely-optional params
        # (model / reasoning_effort / fork_turns) into `required` makes the
        # model send them as "", and Codex rejects that ("reasoning_effort
        # must not be empty"). The Model API accepts their partial `required`
        # as-is.
        if isinstance(t.get("name"), str) and "." in t["name"]:
            continue
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

# Model API occasionally returns transient capacity failures for long agent
# turns. Retrying the identical Responses request is safe: no hosted tools are
# executed upstream, and Codex only receives the response after this buffered
# call completes.
RETRY_MAX_ATTEMPTS = int(os.environ.get("CODEX_META_RETRY_ATTEMPTS", "20"))
RETRY_BASE_DELAY_SECONDS = float(os.environ.get("CODEX_META_RETRY_DELAY", "5.0"))
RETRY_MAX_DELAY_SECONDS = float(os.environ.get("CODEX_META_RETRY_MAX_DELAY", "60.0"))
_RETRYABLE_MESSAGE_SNIPPETS = ("app_overload", "throttling_error")


class UpstreamHTTPError(Exception):
    """Upstream HTTP error whose response body has already been consumed."""

    def __init__(self, code: int, body: bytes) -> None:
        super().__init__(f"upstream HTTP {code}")
        self.code = code
        self.body = body


def _is_retryable(code: int, body: bytes) -> bool:
    if code in (429, 500, 502, 503, 504):
        return True
    text = body.decode(errors="replace").lower()
    return any(snippet in text for snippet in _RETRYABLE_MESSAGE_SNIPPETS)


def call_upstream_with_retry(body: dict[str, Any], req_id: int) -> tuple[int, bytes]:
    for attempt in range(1, RETRY_MAX_ATTEMPTS + 1):
        upstream_req = urllib.request.Request(
            f"{UPSTREAM}/responses",
            data=json.dumps(body).encode(),
            method="POST",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(upstream_req, timeout=600) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as e:
            error_body = e.read()
            if not _is_retryable(e.code, error_body) or attempt == RETRY_MAX_ATTEMPTS:
                raise UpstreamHTTPError(e.code, error_body) from None
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
                f"{e.code}, retrying in {delay:.1f}s: {error_body[:200]!r}\n"
            )
            _debug(req_id, f"RETRY attempt={attempt} code={e.code} body={error_body[:500]!r}")
            time.sleep(delay)
    raise AssertionError("unreachable")


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
        strip_functions_prefix(body)
        session = _session_key(body)
        # Keep the pre-normalisation input around for
        # record_outgoing_agent_messages (needs the agent_message recipients).
        orig_input = list(body.get("input") or [])
        normalize_agent_messages(body, session)
        if body.get("tools"):
            body["tools"] = flatten_namespace_tools(body["tools"])
            body["tools"] = drop_unsupported_tools(body["tools"])
        strictify_tools(body.get("tools") or [])
        want_stream = bool(body.get("stream"))
        # Fetch whole turns non-streamed and (for Codex, which always sets
        # stream=true) synthesise the event stream from the buffered object --
        # same approach as claude_llama_proxy / codex_kimi_proxy. It's the only
        # place we can reliably rewrite response items (split_namespaced_calls)
        # and it sidesteps partial-JSON-across-chunks parsing.
        body["stream"] = False
        sys.stderr.write(
            f"-> upstream model={requested_model}->{body['model']} want_stream={want_stream} "
            f"tools={len(body.get('tools') or [])}\n"
        )
        _debug(req_id, f"REQUEST model={body['model']} want_stream={want_stream} keys={sorted(body.keys())}")
        if body.get("tools"):
            _debug(req_id, "TOOLS " + json.dumps(body["tools"], ensure_ascii=False))

        try:
            status, data = call_upstream_with_retry(body, req_id)
            try:
                obj = json.loads(data.decode())
            except ValueError:
                _debug(req_id, f"RESPONSE non-JSON bytes={len(data)}")
                self._send(status, data, "application/json")
                return
            strip_functions_prefix(obj)
            split_namespaced_calls(obj)
            record_outgoing_agent_messages(obj, orig_input, session)
            _debug(req_id, f"RESPONSE ok output_items={len(obj.get('output') or [])}")
            if want_stream:
                self._send_sse(obj)
            else:
                self._send(200, json.dumps(obj, ensure_ascii=False).encode())
        except UpstreamHTTPError as e:
            err = e.body
            sys.stderr.write(f"upstream HTTP {e.code}: {err[:500]!r}\n")
            _debug(req_id, f"HTTPError {e.code}: {err[:2000]!r}")
            m = re.search(rb"input\[(\d+)\]", err)
            if m:
                idx = int(m.group(1))
                inp = body.get("input") or []
                if idx < len(inp):
                    _debug(req_id, f"OFFENDING input[{idx}] " + json.dumps(inp[idx], ensure_ascii=False))
            if DEBUG:
                try:
                    with open("/tmp/codex_meta_proxy_last_error.json", "w") as f:
                        json.dump(body, f, ensure_ascii=False, indent=1)
                except OSError:
                    pass
            self._send(e.code, err or json.dumps({"error": str(e)}).encode())
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(f"proxy error: {e}\n")
            _debug(req_id, f"EXCEPTION {type(e).__name__}: {e}")
            self._send(500, json.dumps({"error": {"type": "proxy_error", "message": str(e)}}).encode())

    def _send_sse(self, response: dict[str, Any]) -> None:
        """Emit a full Responses-API SSE lifecycle from one buffered object.

        Codex requests stream=true and, even after its WebSocket transport
        falls back to plain HTTPS, expects a real text/event-stream ending in
        `response.completed` -- and it rebuilds messages / tool calls from the
        incremental output-item events, so created+completed alone makes it
        report an empty turn and drop the tool call (see codex_kimi_proxy for
        the full write-up). Replay the same item/content/argument lifecycle a
        native stream would produce.
        """
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def write_event(event: dict[str, Any]) -> None:
            payload = f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()
            self.wfile.write(f"{len(payload):x}\r\n".encode())
            self.wfile.write(payload)
            self.wfile.write(b"\r\n")

        for event in response_sse_events(response):
            write_event(event)
        self.wfile.write(b"0\r\n\r\n")


def response_sse_events(response: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand one buffered Responses object into Codex-consumable SSE events.

    Adapted from codex_kimi_proxy.response_sse_events; here the upstream already
    speaks the Responses API so items pass through unchanged (a reasoning item
    keeps its encrypted_content, function_call its namespace field, etc.) --
    only the streaming lifecycle around them is synthetic.
    """
    seq = 0

    def ev(d: dict[str, Any]) -> dict[str, Any]:
        nonlocal seq
        d["sequence_number"] = seq
        seq += 1
        return d

    created = {**response, "output": [], "status": "in_progress"}
    events: list[dict[str, Any]] = [ev({"type": "response.created", "response": created})]
    events.append(ev({"type": "response.in_progress", "response": created}))

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
            in_progress.setdefault("summary", [])

        events.append(ev({
            "type": "response.output_item.added",
            "output_index": output_index,
            "item": in_progress,
        }))

        if item_type == "message":
            for content_index, part in enumerate(item.get("content") or []):
                text = part.get("text") or ""
                events.append(ev({
                    "type": "response.content_part.added",
                    "output_index": output_index, "content_index": content_index,
                    "item_id": item.get("id"), "part": {**part, "text": ""},
                }))
                if text:
                    events.append(ev({
                        "type": "response.output_text.delta",
                        "output_index": output_index, "content_index": content_index,
                        "item_id": item.get("id"), "delta": text,
                    }))
                events.append(ev({
                    "type": "response.output_text.done",
                    "output_index": output_index, "content_index": content_index,
                    "item_id": item.get("id"), "text": text,
                }))
                events.append(ev({
                    "type": "response.content_part.done",
                    "output_index": output_index, "content_index": content_index,
                    "item_id": item.get("id"), "part": part,
                }))
        elif item_type == "function_call":
            arguments = item.get("arguments") or ""
            if arguments:
                events.append(ev({
                    "type": "response.function_call_arguments.delta",
                    "output_index": output_index, "item_id": item.get("id"), "delta": arguments,
                }))
            events.append(ev({
                "type": "response.function_call_arguments.done",
                "output_index": output_index, "item_id": item.get("id"), "arguments": arguments,
            }))
        elif item_type == "custom_tool_call":
            tool_input = item.get("input") or ""
            if tool_input:
                events.append(ev({
                    "type": "response.custom_tool_call_input.delta",
                    "output_index": output_index, "item_id": item.get("id"), "delta": tool_input,
                }))
            events.append(ev({
                "type": "response.custom_tool_call_input.done",
                "output_index": output_index, "item_id": item.get("id"), "input": tool_input,
            }))

        events.append(ev({
            "type": "response.output_item.done",
            "output_index": output_index,
            "item": item,
        }))

    events.append(ev({"type": "response.completed", "response": {**response, "status": "completed"}}))
    return events


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
