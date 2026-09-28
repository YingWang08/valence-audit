#!/usr/bin/env python3
"""Reasoning split in front of vLLM, for a reasoning model whose tokenizer has no single <think> token.

NIM returned llama-3.3-nemotron-super-49b-v1.5's reasoning separately from its answer (June 2026), and
the revision round keeps that contract: the answer is read from `content`, the reasoning is logged
from `reasoning_content`. vLLM's own parser for this format (deepseek_r1) refuses to start for this
model (vLLM 0.28, the notebook image: "could not locate think start/end tokens in the tokenizer"),
because <think> and </think> are not single tokens in its Llama-3 tokenizer. So vLLM serves the model without a reasoning parser (the model's text, including its <think> ... </think>
block, comes back in `content`), and this proxy, which the collection talks to, moves the reasoning
out of `content` with the same rule that tools/notebook/hf_server.py --reasoning-split uses
(split_reasoning: text before </think> is reasoning; output that stops inside the reasoning is all
reasoning). Nothing else in the request or the response is changed; token counts and finish_reason
are vLLM's. No request or response text is logged.

    python tools/notebook/think_split_proxy.py --port 8000 --upstream-port 8001
"""
import argparse
import http.client
import json
import pathlib
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from hf_server import split_reasoning  # noqa: E402  (one rule for both engines)

HOP = {"connection", "keep-alive", "transfer-encoding", "content-length", "host"}


def split_choices(obj):
    """Apply split_reasoning to every choice whose server did not already separate the reasoning."""
    for ch in obj.get("choices") or []:
        msg = ch.get("message") or {}
        if msg.get("reasoning_content") or msg.get("reasoning"):
            continue
        text = msg.get("content")
        if not isinstance(text, str):
            continue
        reasoning, answer = split_reasoning(text, ch.get("finish_reason"))
        msg["content"] = answer
        msg["reasoning_content"] = reasoning
    return obj


def make_server(port, upstream_host="127.0.0.1", upstream_port=8001, host="127.0.0.1", timeout=3600):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def _send(self, code, body, ctype="application/json", headers=()):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in headers:
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _forward(self, method):
            body = None
            if method == "POST":
                body = self.rfile.read(int(self.headers.get("Content-Length", "0") or 0))
            is_chat = method == "POST" and self.path.split("?")[0].rstrip("/").endswith("/chat/completions")
            if is_chat:
                try:
                    if json.loads(body or b"{}").get("stream"):
                        return self._send(400, json.dumps({"error": {"message": "streaming is not supported by "
                                                                                "think_split_proxy"}}).encode())
                except ValueError:
                    pass
            fwd = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
            try:
                conn = http.client.HTTPConnection(upstream_host, upstream_port, timeout=timeout)
                conn.request(method, self.path, body=body, headers=fwd)
                r = conn.getresponse()
                data = r.read()
                status, ctype = r.status, r.getheader("Content-Type", "application/json")
                conn.close()
            except Exception as e:  # upstream not up yet, or gone
                return self._send(502, json.dumps({"error": {"message": f"upstream unavailable: "
                                                                        f"{type(e).__name__}"}}).encode())
            if is_chat and status == 200:
                try:
                    data = json.dumps(split_choices(json.loads(data))).encode()
                    ctype = "application/json"
                except ValueError:
                    pass
            self._send(status, data, ctype)

        def do_GET(self):
            self._forward("GET")

        def do_POST(self):
            self._forward("POST")

    srv = ThreadingHTTPServer((host, port), H)
    srv.daemon_threads = True
    return srv


def start_in_thread(port, upstream_port):
    srv = make_server(port, upstream_port=upstream_port)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--upstream-port", type=int, default=8001)
    a = ap.parse_args()
    print(f"[think_split_proxy] http://127.0.0.1:{a.port}/v1 -> http://127.0.0.1:{a.upstream_port}/v1", flush=True)
    make_server(a.port, upstream_port=a.upstream_port).serve_forever()


if __name__ == "__main__":
    main()
