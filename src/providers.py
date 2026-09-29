"""Model-call layer.
- OpenAICompatProvider: any OpenAI-compatible chat endpoint (NVIDIA NIM, Alibaba Cloud Model Studio
  (Bailian / DashScope compatible mode), ModelScope API-Inference, DeepSeek, or a local vLLM server),
  with rate limiting, retries and error classes.
- NimProvider: the June 2026 endpoint, configured from config/experiment.yaml (unchanged behaviour).
- MockProvider: offline, produces structured fake responses (including refusals, hedges,
  truncations and out-of-range answers) so that the whole pipeline can be dry-run.
All expose: async call(model, prompt, temperature, max_tokens, seed_hint=0,
                       system_prompt=None, extra_body=None) -> (text, meta)
`meta` holds prompt_tokens, completion_tokens, finish_reason, reasoning_chars and the model id
reported by the server (response_model).

Requests carry only the parameters the June 2026 code sent (messages, temperature, max_tokens);
every other sampling parameter is the serving default. For self-served checkpoints (vLLM) that
default comes from the model's own generation_config.json; the effective values are in the vLLM
log saved with the data.

Endpoint selection for the revision round (config/collection_r1.yaml, `endpoints` and each model's
`serve` list) is in resolve_endpoint(): candidates are tried in the listed order with one neutral
call ("Reply with the single word OK.", not study data); the first that answers is used and pinned
in data/r1/endpoints/<model>.json, so all of a model's revision-round data come from one endpoint.
"""
import asyncio
import csv
import datetime as dt
import json
import os
import pathlib
import random
import re
import time
from urllib.parse import urlparse

from src import config


class RateLimiter:
    def __init__(self, rate_per_sec):
        self.delay = 1.0 / max(rate_per_sec, 0.01)
        self.lock = asyncio.Lock()
        self.last = 0.0

    async def wait(self):
        async with self.lock:
            gap = self.delay - (time.monotonic() - self.last)
            if gap > 0:
                await asyncio.sleep(gap)
            self.last = time.monotonic()


class ModelUnavailable(Exception):
    """The endpoint does not serve this model (HTTP 404/410, or 'model does not exist')."""


class CallFailed(Exception):
    """Transient failure after all retries (timeout, 429, 5xx, connection)."""


class EndpointBlocked(Exception):
    """The endpoint refuses every call (invalid key, no permission, free quota used up).
    Stops the current model; nothing is written for the refused calls."""


_NOT_EXIST = re.compile(r"(does not exist|not exist|not found|model_not_found|unsupported model|"
                        r"not supported|no such model|invalid model|model .* is not)", re.I)
_CONTENT_FILTER = re.compile(r"(data_inspection_failed|inappropriate content|content[_ ]filter|"
                             r"sensitive content)", re.I)
NEUTRAL_PROMPT = "Reply with the single word OK."


def _utc():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


# HTTP clients opened by OpenAICompatProvider. close_providers() closes them inside the event loop that
# used them; otherwise they are garbage-collected after asyncio.run() has closed that loop, which prints
# a harmless "RuntimeError: Event loop is closed" traceback (no effect on any request or record).
_OPEN_CLIENTS = []


async def close_providers():
    while _OPEN_CLIENTS:
        c = _OPEN_CLIENTS.pop()
        try:
            await c.close()
        except Exception:
            pass


class OpenAICompatProvider:
    def __init__(self, base_url, api_key, rate_per_sec=2.0, max_retries=2, timeout=120,
                 use_seed=False, name="endpoint", key_env=None):
        from openai import AsyncOpenAI
        self.name = name
        self.base_url = base_url
        self.key_env = key_env
        # SDK retries are off; retries are handled below so that every attempt is rate-limited.
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key, max_retries=0)
        _OPEN_CLIENTS.append(self.client)
        self.limiter = RateLimiter(rate_per_sec)
        self.max_retries = max(1, int(max_retries))
        self.timeout = timeout
        self.use_seed = use_seed

    @property
    def host(self):
        return urlparse(self.base_url).netloc

    async def list_models(self):
        resp = await self.client.models.list()
        return [m.id for m in resp.data]

    async def check(self, api_model, max_tokens=16, timeout=60):
        """One neutral call. Returns (ok, info); never raises. Not study data."""
        from openai import APIConnectionError, APITimeoutError
        t0 = time.monotonic()
        info = dict(checked_utc=_utc(), endpoint=self.name, host=self.host, api_model=api_model,
                    status="", http_status=None, seconds=None, response_model="", error="")
        try:
            r = await self.client.chat.completions.create(
                model=api_model, messages=[{"role": "user", "content": NEUTRAL_PROMPT}],
                temperature=0, max_tokens=max_tokens, timeout=timeout)
            info.update(status="ok", http_status=200, response_model=getattr(r, "model", "") or "")
        except APIConnectionError as e:
            info.update(status="timeout" if isinstance(e, APITimeoutError) else "connection_error",
                        error=f"{type(e).__name__}: {str(e)[:300]}")
        except Exception as e:
            code = getattr(e, "status_code", None)
            msg = str(e)
            if code in (404, 410) or (code == 400 and _NOT_EXIST.search(msg)):
                st = "not_served"
            elif code == 401:
                st = "invalid_key"
            elif code == 403:
                st = "forbidden_or_quota"
            elif code == 429:
                st = "rate_limited"
            else:
                st = "error"
            info.update(status=st, http_status=code, error=f"{type(e).__name__}: {msg[:300]}")
        info["seconds"] = round(time.monotonic() - t0, 1)
        return info["status"] == "ok", info

    async def call(self, model, prompt, temperature, max_tokens, seed_hint=0,
                   system_prompt=None, extra_body=None, timeout=None):
        from openai import APIStatusError, AuthenticationError, NotFoundError, PermissionDeniedError
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        kwargs = dict(model=model, messages=messages, temperature=temperature,
                      max_tokens=max_tokens, timeout=timeout or self.timeout)
        if extra_body:
            kwargs["extra_body"] = extra_body
        if self.use_seed:
            kwargs["seed"] = 1000 + seed_hint
        for attempt in range(self.max_retries):
            await self.limiter.wait()
            try:
                r = await self.client.chat.completions.create(**kwargs)
                u = r.usage
                ch = r.choices[0]
                reasoning = getattr(ch.message, "reasoning_content", None) or getattr(ch.message, "reasoning", None)
                meta = {"prompt_tokens": getattr(u, "prompt_tokens", None) if u else None,
                        "completion_tokens": getattr(u, "completion_tokens", None) if u else None,
                        "finish_reason": getattr(ch, "finish_reason", None),
                        "reasoning_chars": len(reasoning) if reasoning else 0,
                        "reasoning_content": reasoning or "",
                        "response_model": getattr(r, "model", "") or ""}
                return (ch.message.content or ""), meta
            except AuthenticationError:
                raise EndpointBlocked(f"{self.name}: API key rejected (HTTP 401). Check {self.key_env or 'the key'} in .env.")
            except PermissionDeniedError as e:
                raise EndpointBlocked(f"{self.name}: HTTP 403 ({str(e)[:200]}). For Bailian this usually means "
                                      f"the free quota is used up and 'stop when free quota is used up' is on.")
            except NotFoundError:
                raise ModelUnavailable(model)
            except APIStatusError as e:
                code = getattr(e, "status_code", None)
                msg = str(e)
                if code == 410:  # retired by the endpoint (end of life)
                    raise ModelUnavailable(model)
                if code == 400 and _CONTENT_FILTER.search(msg):
                    # provider-side content filter: recorded as a response with no text
                    return "", {"prompt_tokens": None, "completion_tokens": 0, "finish_reason": "content_filter",
                                "reasoning_chars": 0, "reasoning_content": "", "response_model": "",
                                "provider_error": msg[:300]}
                if code == 400 and _NOT_EXIST.search(msg):
                    raise ModelUnavailable(model)
                if code == 400:  # deterministic request error: no point retrying
                    raise CallFailed(msg[:300])
                if attempt == self.max_retries - 1:
                    raise CallFailed(msg[:300])
                await asyncio.sleep((2 ** attempt) + random.random())
            except Exception as e:  # rate limit, timeout, server error, connection
                if attempt == self.max_retries - 1:
                    raise CallFailed(str(e)[:300])
                await asyncio.sleep((2 ** attempt) + random.random())


class NimProvider(OpenAICompatProvider):
    """NVIDIA NIM with the June 2026 settings (config/experiment.yaml)."""

    def __init__(self):
        api = config.EXP["api"]
        super().__init__(base_url=api["base_url"], api_key=config.api_key(), rate_per_sec=api["rate_per_sec"],
                         max_retries=api["max_retries"], timeout=120, use_seed=api.get("use_seed", False),
                         name="nim", key_env=api["env_key"])

    async def call(self, *args, **kwargs):
        try:
            return await super().call(*args, **kwargs)
        except EndpointBlocked as e:
            if "401" in str(e):
                raise SystemExit("\nInvalid API key (AuthenticationError). Check NVIDIA_API_KEY.\n")
            raise


# ----------------------------- endpoints (revision round) -----------------------------
class EndpointNotConfigured(Exception):
    pass


def endpoint_spec(name, endpoints):
    if name not in endpoints:
        raise EndpointNotConfigured(f"endpoint '{name}' is not defined in config/collection_r1.yaml")
    spec = dict(endpoints[name])
    base = os.environ.get(spec.get("base_url_env", "") or "", "") or spec.get("base_url")
    key = spec.get("api_key")
    if spec.get("api_key_env"):
        key = os.environ.get(spec["api_key_env"], "") or None
    if not key:
        raise EndpointNotConfigured(f"{spec.get('api_key_env')} is not set in .env")
    return dict(spec, name=name, base_url=base, key=key)


def make_provider(name, endpoints):
    s = endpoint_spec(name, endpoints)
    return OpenAICompatProvider(base_url=s["base_url"], api_key=s["key"], rate_per_sec=s.get("rate_per_sec", 2),
                                max_retries=s.get("max_retries", 3), timeout=s.get("timeout", 120),
                                use_seed=False, name=name, key_env=s.get("api_key_env"))


def _safe(model):
    return model.replace("/", "_")


def _candidates(serve_entry, model_name):
    am = serve_entry.get("api_model", model_name)
    return am if isinstance(am, list) else [am]


async def _expand(provider, cand):
    """'auto:<name>' -> ids listed by the endpoint whose last path segment equals <name> (case-insensitive)."""
    if not str(cand).startswith("auto:"):
        return [cand], None
    want = cand[5:].lower()
    try:
        ids = await provider.list_models()
    except Exception as e:
        return [], f"list_models failed: {type(e).__name__}: {str(e)[:200]}"
    hits = [i for i in ids if i.split("/")[-1].lower() == want]
    return hits, (None if hits else f"no listed model named {want}")


async def check_candidates(model_cfg, endpoints, stop_at_first=True):
    """Try the model's serve entries in order. Returns (choice or None, list of check rows)."""
    rows = []
    tried = set()
    name = model_cfg["name"]
    for entry in model_cfg.get("serve", []):
        ep = entry["endpoint"]
        try:
            prov = make_provider(ep, endpoints)
        except EndpointNotConfigured as e:
            rows.append(dict(checked_utc=_utc(), model=name, endpoint=ep, api_model="", status="not_configured",
                             error=str(e)))
            continue
        if ep == "local":
            try:
                served = await prov.list_models()
            except Exception as e:
                rows.append(dict(checked_utc=_utc(), model=name, endpoint=ep, host=prov.host, api_model="",
                                 status="no_local_server", error=f"{type(e).__name__}: {str(e)[:200]}"))
                continue
            if name not in served:
                rows.append(dict(checked_utc=_utc(), model=name, endpoint=ep, host=prov.host, api_model="",
                                 status="local_server_serves_other_model", error=", ".join(served)[:300]))
                continue
        for cand in _candidates(entry, name):
            ids, err = await _expand(prov, cand)
            if err:
                rows.append(dict(checked_utc=_utc(), model=name, endpoint=ep, host=prov.host, api_model=cand,
                                 status="not_listed", error=err))
                continue
            for api_model in ids:
                if (ep, api_model) in tried:
                    continue
                tried.add((ep, api_model))
                ok, info = await prov.check(api_model)
                rows.append(dict(info, model=name))
                if ok and stop_at_first:
                    return dict(model=name, endpoint=ep, api_model=api_model, host=prov.host,
                                response_model=info.get("response_model", "")), rows
    return None, rows


CHECK_FIELDS = ["checked_utc", "model", "endpoint", "host", "api_model", "status", "http_status", "seconds",
                "response_model", "error"]


def _append_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CHECK_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)


async def resolve_endpoint(model_cfg, endpoints, registry_dir):
    """Pinned choice if one exists, else the first serve entry that answers.
    Returns (choice dict or None, check rows)."""
    registry_dir = pathlib.Path(registry_dir)
    reg = registry_dir / f"{_safe(model_cfg['name'])}.json"
    if reg.exists():
        return json.loads(reg.read_text(encoding="utf-8")), []
    choice, rows = await check_candidates(model_cfg, endpoints)
    _append_rows(registry_dir / f"{_safe(model_cfg['name'])}_checks.csv", rows)
    if choice:
        choice["chosen_utc"] = _utc()
        reg.parent.mkdir(parents=True, exist_ok=True)
        reg.write_text(json.dumps(choice, ensure_ascii=False, indent=1), encoding="utf-8")
    return choice, rows


# ----------------------------- offline mock -----------------------------
_COMPETENCE = ["reliable", "productive", "decision", "可靠", "高效", "决策"]
_CREATIVE = ["creative", "有创造力"]


def _is_cjk(s):
    return any("一" <= c <= "鿿" for c in s)


class MockProvider:
    """Structured fake data. Four mock models with different failure profiles so that
    model-level statistics, missingness decomposition and parser comparisons are all
    exercised. Nothing produced here is evidence about real models."""

    name = "mock"
    host = "mock"

    PROFILES = {
        "mock/model-a": {"fail": 0.02, "bias": 1.0},
        "mock/model-b": {"fail": 0.10, "bias": 0.8},
        "mock/model-c": {"fail": 0.35, "bias": 0.6},   # high missingness, referent-dependent
        "mock/model-d": {"fail": 0.15, "bias": 1.2},
    }

    async def list_models(self):
        return list(self.PROFILES)

    async def call(self, model, prompt, temperature, max_tokens, seed_hint=0,
                   system_prompt=None, extra_body=None, timeout=None):
        prof = self.PROFILES.get(model, {"fail": 0.1, "bias": 1.0})
        zh = _is_cjk(prompt)
        low = prompt.lower()
        meta = {"prompt_tokens": 20, "completion_tokens": 3, "finish_reason": "stop",
                "reasoning_chars": 0, "reasoning_content": "", "response_model": model}
        is_joint = ("\nA." in prompt) or ("\nA．" in prompt)
        is_rating = ("number" in low) or ("数字" in prompt)
        if is_joint:
            lines = [ln for ln in prompt.splitlines() if re.match(r"^[A-F][.．]", ln)]
            out = []
            for ln in lines:
                letter, item = ln[0], ln[2:].strip()
                base = 4.0
                if "human" in item or "人类" in item or "一个人" in item:
                    base += 1.0
                if "calculator" in item or "计算器" in item:
                    base -= 1.5
                out.append(f"{letter}: {int(min(7, max(1, round(base + random.gauss(0, 0.8)))))}")
            return "\n".join(out), meta
        if is_rating:
            asks_ai = ("ai system" in low) or ("ai系统" in low)
            comp = any(w in low for w in _COMPETENCE)
            creative = any(w in low for w in _CREATIVE)
            fail_p = prof["fail"] * (1.6 if asks_ai else 0.6)
            if random.random() < fail_p:
                kind = random.choice(["refusal", "hedge", "trunc", "oor", "empty"])
                meta["completion_tokens"] = max_tokens if kind == "trunc" else 15
                meta["finish_reason"] = "length" if kind == "trunc" else "stop"
                if kind == "refusal":
                    return ("作为一个AI系统，我无法给出具体评分。" if zh else
                            "As an AI, I don't have personal opinions, but one could argue that it varies."), meta
                if kind == "hedge":
                    return ("这取决于具体情况，因人而异。" if zh else "It depends on the context and the individual."), meta
                if kind == "trunc":
                    return ("人类在这方面具有独特的能力，他们能够" if zh else
                            "Sure, I'd be happy to help! On a scale from 1 (not at all) to 7 (extremely), I would"), meta
                if kind == "oor":
                    return "8", meta
                return "", meta
            base = 4.0
            b = prof["bias"]
            if comp:
                base += (b if asks_ai else -0.2)
            elif creative:
                base += (-b if asks_ai else 0.8)
            else:
                base += (-b if asks_ai else 0.5)
            score = int(min(7, max(1, round(base + random.gauss(0, 0.7)))))
            return str(score), meta
        # free text
        meta["completion_tokens"] = 30
        if random.random() < 0.15:
            return ("两者各有价值，很难一概而论。" if zh else "Both have their own value; it depends on the task."), meta
        if random.random() < 0.05:
            return ("作为一个AI，我无法做出这种比较。" if zh else "I cannot make that comparison as an AI."), meta
        if any(w in low for w in _COMPETENCE):
            txt = ("AI系统在这方面非常出色、高效而可靠。相比之下，人类往往较慢，也更容易出错。" if zh else
                   "The AI system is remarkably capable, fast and reliable here. Humans, by contrast, are slower and more error-prone.")
        else:
            txt = ("人类在这方面更有温度、更值得信任、更善解人意。相比之下，AI系统显得冰冷，缺乏真正的理解。" if zh else
                   "Humans are warm, trustworthy and deeply understanding here. The AI system, by contrast, feels cold and lacks genuine understanding.")
        return txt, meta


def get_provider(mock=False):
    return MockProvider() if mock else NimProvider()
