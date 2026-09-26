"""Model-call layer.
- NimProvider: NVIDIA NIM (OpenAI-compatible) with rate limiting, retries and error classes.
- MockProvider: offline, produces structured fake responses (including refusals, hedges,
  truncations and out-of-range answers) so that the whole pipeline can be dry-run.
Both expose: async call(model, prompt, temperature, max_tokens, seed_hint=0,
                        system_prompt=None, extra_body=None) -> (text, meta)
`meta` holds prompt_tokens, completion_tokens, finish_reason and reasoning_chars
(v1.0.0 logged only token counts, and only to data/raw/_usage.csv)."""
import asyncio
import time
import random
import re
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
    pass


class CallFailed(Exception):
    pass


class NimProvider:
    def __init__(self):
        from openai import AsyncOpenAI
        api = config.EXP["api"]
        self.client = AsyncOpenAI(base_url=api["base_url"], api_key=config.api_key())
        self.limiter = RateLimiter(api["rate_per_sec"])
        self.max_retries = api["max_retries"]
        self.use_seed = api.get("use_seed", False)

    async def list_models(self):
        resp = await self.client.models.list()
        return [m.id for m in resp.data]

    async def call(self, model, prompt, temperature, max_tokens, seed_hint=0,
                   system_prompt=None, extra_body=None, timeout=120):
        from openai import AuthenticationError, NotFoundError
        await self.limiter.wait()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        kwargs = dict(model=model, messages=messages, temperature=temperature,
                      max_tokens=max_tokens, timeout=timeout)
        if extra_body:
            kwargs["extra_body"] = extra_body
        if self.use_seed:
            kwargs["seed"] = 1000 + seed_hint
        for attempt in range(self.max_retries):
            try:
                r = await self.client.chat.completions.create(**kwargs)
                u = r.usage
                ch = r.choices[0]
                reasoning = getattr(ch.message, "reasoning_content", None) or getattr(ch.message, "reasoning", None)
                meta = {"prompt_tokens": getattr(u, "prompt_tokens", None),
                        "completion_tokens": getattr(u, "completion_tokens", None),
                        "finish_reason": getattr(ch, "finish_reason", None),
                        "reasoning_chars": len(reasoning) if reasoning else 0,
                        "reasoning_content": reasoning or ""}
                return (ch.message.content or ""), meta
            except AuthenticationError:
                raise SystemExit("\nInvalid API key (AuthenticationError). Check NVIDIA_API_KEY.\n")
            except NotFoundError:
                raise ModelUnavailable(model)
            except Exception as e:  # rate limit, timeout, server error
                if attempt == self.max_retries - 1:
                    raise CallFailed(str(e))
                await asyncio.sleep((2 ** attempt) + random.random())


# ----------------------------- offline mock -----------------------------
_COMPETENCE = ["reliable", "productive", "decision", "可靠", "高效", "决策"]
_CREATIVE = ["creative", "有创造力"]


def _is_cjk(s):
    return any("\u4e00" <= c <= "\u9fff" for c in s)


class MockProvider:
    """Structured fake data. Four mock models with different failure profiles so that
    model-level statistics, missingness decomposition and parser comparisons are all
    exercised. Nothing produced here is evidence about real models."""

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
                "reasoning_chars": 0, "reasoning_content": ""}
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
