#!/usr/bin/env python3
"""Minimal OpenAI-compatible chat server on Hugging Face transformers.

Fallback used by tools/notebook/run_models.py only when vLLM cannot serve a model on the notebook's
GPU. Same contract as vLLM for this study:
  - the prompt is the model's own chat template (tokenizer_config.json), generation prompt added;
  - the request sets temperature and max_tokens; every other sampling parameter is taken from the
    model's generation_config.json if it sets it (top_p, top_k, repetition_penalty), otherwise it is
    off (top_p 1, no top-k, no repetition penalty) -- the transformers default top_k=50 is NOT used;
  - generation stops at the model's end-of-sequence ids or at max_tokens (finish_reason "length");
  - with --reasoning-split, text before </think> is returned as reasoning_content and the rest as
    content (as vLLM's deepseek_r1 reasoning parser and the June endpoint did).
Requests are batched (same max_tokens and temperature) and run one batch at a time on one GPU.

    python tools/notebook/hf_server.py --model DIR --served-model-name ID --port 8000 [--trust-remote-code] [--reasoning-split]
"""
import argparse
import json
import pathlib
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SAMPLING_KEYS = ("top_p", "top_k", "repetition_penalty")


def load(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=a.trust_remote_code)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    cfg = json.loads((pathlib.Path(a.model) / "config.json").read_text(encoding="utf-8"))
    attn = "eager" if str(cfg.get("model_type", "")).startswith("gemma") else "sdpa"
    kw = dict(trust_remote_code=a.trust_remote_code, attn_implementation=attn)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    import importlib.util
    if importlib.util.find_spec("accelerate"):   # device_map needs accelerate
        kw["device_map"] = dev      # load straight onto the GPU (no full copy in CPU memory)
    try:
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=a.torch_dtype, **kw)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=a.torch_dtype, **kw)
    if "device_map" not in kw:
        model.to(dev)
    model.eval()
    gc_path = pathlib.Path(a.model) / "generation_config.json"
    gc = json.loads(gc_path.read_text(encoding="utf-8")) if gc_path.exists() else {}
    defaults = {k: gc[k] for k in SAMPLING_KEYS if k in gc}
    eos = gc.get("eos_token_id", model.generation_config.eos_token_id)
    eos = [eos] if isinstance(eos, int) else list(eos or [tok.eos_token_id])
    print(f"[hf_server] {a.served_model_name}: attn={attn}, dtype={a.dtype}, device={dev}, "
          f"sampling defaults from generation_config.json: {defaults or 'none'}, eos={eos}", flush=True)
    return tok, model, defaults, eos, dev


def split_reasoning(text, finish):
    """(reasoning, answer). Text before </think> is reasoning; if the output stops inside the reasoning
    (no </think>: truncated, or an unclosed <think>), everything is reasoning and the answer is empty."""
    if "</think>" in text:
        r, ans = text.split("</think>", 1)
        return r.replace("<think>", "", 1).strip(), ans.strip()
    if "<think>" in text or finish == "length":
        return text.replace("<think>", "", 1).strip(), ""
    return "", text


class Job:
    def __init__(self, messages, temperature, max_tokens):
        self.messages, self.temperature, self.max_tokens = messages, temperature, max_tokens
        self.done = threading.Event()
        self.result = None
        self.error = None


def worker(a, tok, model, defaults, eos, dev, q):
    import torch
    max_batch = a.max_batch
    while True:
        first = q.get()
        batch = [first]
        t0 = time.time()
        held = []
        while len(batch) < max_batch and time.time() - t0 < a.wait:
            try:
                j = q.get(timeout=a.wait)
            except queue.Empty:
                break
            if j.max_tokens == first.max_tokens and j.temperature == first.temperature:
                batch.append(j)
            else:
                held.append(j)
        for j in held:
            q.put(j)
        try:
            texts = [tok.apply_chat_template(j.messages, add_generation_prompt=True, tokenize=False) for j in batch]
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
            kw = dict(max_new_tokens=first.max_tokens, eos_token_id=eos, pad_token_id=tok.pad_token_id)
            if first.temperature and first.temperature > 0:
                kw.update(do_sample=True, temperature=first.temperature, top_p=defaults.get("top_p", 1.0),
                          top_k=defaults.get("top_k", 0))
            else:
                kw.update(do_sample=False)
            if "repetition_penalty" in defaults:
                kw["repetition_penalty"] = defaults["repetition_penalty"]
            with torch.no_grad():
                out = model.generate(**enc, **kw)
            n_in = enc["input_ids"].shape[1]
            for i, j in enumerate(batch):
                gen = out[i, n_in:].tolist()
                cut, finish = len(gen), "length"
                for k, t in enumerate(gen):
                    if t in eos:
                        cut, finish = k + 1, "stop"
                        break
                gen = gen[:cut]
                text = tok.decode([t for t in gen if t not in eos and t != tok.pad_token_id], skip_special_tokens=True)
                reasoning = None
                if a.reasoning_split:
                    reasoning, text = split_reasoning(text, finish)
                prompt_tokens = int(enc["attention_mask"][i].sum())
                j.result = dict(content=text, reasoning=reasoning, finish=finish,
                                prompt_tokens=prompt_tokens, completion_tokens=len(gen))
        except Exception as e:  # e.g. out of memory: fail this batch, use smaller batches from now on
            for j in batch:
                j.error = f"{type(e).__name__}: {str(e)[:300]}"
            max_batch = max(1, max_batch // 2)
            print(f"[hf_server] batch failed ({type(e).__name__}); max batch now {max_batch}", flush=True)
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        for j in batch:
            j.done.set()


def serve(a, q):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code, obj):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path.rstrip("/").endswith("/models"):
                return self._send(200, {"object": "list", "data": [{"id": a.served_model_name, "object": "model",
                                                                    "owned_by": "hf_server"}]})
            self._send(404, {"error": {"message": "not found"}})

        def do_POST(self):
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            except Exception:
                return self._send(400, {"error": {"message": "bad json"}})
            if req.get("model") != a.served_model_name:
                return self._send(404, {"error": {"message": f"The model `{req.get('model')}` does not exist"}})
            j = Job(req["messages"], float(req.get("temperature", 1.0)), int(req.get("max_tokens") or 256))
            q.put(j)
            j.done.wait()
            if j.error:
                return self._send(500, {"error": {"message": j.error}})
            r = j.result
            msg = {"role": "assistant", "content": r["content"]}
            if r["reasoning"] is not None:
                msg["reasoning_content"] = r["reasoning"]
            self._send(200, {"id": f"hf-{int(time.time() * 1000)}", "object": "chat.completion",
                             "created": int(time.time()), "model": a.served_model_name,
                             "choices": [{"index": 0, "message": msg, "finish_reason": r["finish"]}],
                             "usage": {"prompt_tokens": r["prompt_tokens"], "completion_tokens": r["completion_tokens"],
                                       "total_tokens": r["prompt_tokens"] + r["completion_tokens"]}})

    ThreadingHTTPServer((a.host, a.port), H).serve_forever()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--served-model-name", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-batch", type=int, default=32)
    ap.add_argument("--wait", type=float, default=0.05)
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--reasoning-split", action="store_true")
    a = ap.parse_args()
    import torch
    a.torch_dtype = getattr(torch, a.dtype)
    tok, model, defaults, eos, dev = load(a)
    q = queue.Queue()
    threading.Thread(target=worker, args=(a, tok, model, defaults, eos, dev, q), daemon=True).start()
    print(f"[hf_server] serving {a.served_model_name} on http://{a.host}:{a.port}/v1", flush=True)
    serve(a, q)


if __name__ == "__main__":
    main()
