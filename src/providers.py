"""模型调用层。
- NimProvider: 调 NVIDIA NIM（OpenAI 兼容），含限速 + 指数退避重试 + 错误分类。
- MockProvider: 不联网，生成结构化假回复，用于 --mock 离线跑通整条流水线。
两者接口一致：async call(model, prompt, temperature, max_tokens) -> (text, usage)
"""
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


# ---------------- 真·NIM 客户端 ----------------
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

    async def call(self, model, prompt, temperature, max_tokens, seed_hint=0):
        from openai import (AuthenticationError, NotFoundError,
                            RateLimitError, APIError, APITimeoutError)
        await self.limiter.wait()
        kwargs = dict(model=model,
                      messages=[{"role": "user", "content": prompt}],
                      temperature=temperature, max_tokens=max_tokens,
                      timeout=60)  # ← 单条请求最多等 60 秒，挂起的直接超时进重试，不再无限等
        if self.use_seed:
            kwargs["seed"] = 1000 + seed_hint
        for attempt in range(self.max_retries):
            try:
                r = await self.client.chat.completions.create(**kwargs)
                u = r.usage
                usage = {"prompt_tokens": getattr(u, "prompt_tokens", 0),
                         "completion_tokens": getattr(u, "completion_tokens", 0)}
                return (r.choices[0].message.content or ""), usage
            except AuthenticationError:
                raise SystemExit("\n密钥无效（AuthenticationError）。请检查 NVIDIA_API_KEY 是否正确、是否已吊销旧key并换新。\n")
            except NotFoundError:
                raise ModelUnavailable(model)          # 该模型不存在 -> 上层跳过
            except (RateLimitError, APIError, APITimeoutError, Exception) as e:
                if attempt == self.max_retries - 1:
                    raise CallFailed(str(e))
                await asyncio.sleep((2 ** attempt) + random.random())


class ModelUnavailable(Exception):
    pass


class CallFailed(Exception):
    pass


# ---------------- Mock（离线，无需联网/密钥）----------------
_COMPETENCE = ["reliable", "accurate", "productive", "efficient", "creative",
               "decision", "可靠", "准确", "高效", "创造", "决策"]
_HUMAN_FAV = ["moral", "emotion", "trust", "worth", "valuable",
              "道德", "情感", "信任", "价值", "善解"]


def _is_cjk(s):
    return any('\u4e00' <= c <= '\u9fff' for c in s)


class MockProvider:
    """根据 prompt 内容产出结构化假数据，使测量/统计在 mock 模式下也非退化。
    制造 H2（能力维度抬高机器）、H3（含 instruct 的模型不对称变小、对冲更多）的结构。"""

    async def list_models(self):
        return ["mock/llama-base", "mock/llama-instruct"]

    async def call(self, model, prompt, temperature, max_tokens, seed_hint=0):
        zh = _is_cjk(prompt)
        low = prompt.lower()
        comp = any(w in low for w in _COMPETENCE)        # 能力类维度？
        instruct = ("instruct" in model) or ("frontier" in model)
        noise = random.gauss(0, 0.6)

        # 评分式：检测当前问的是 human 还是 AI
        # （所有评分模板都含 "number"/"数字"，自由文本模板都不含——据此稳健识别）
        is_rating = ("number" in low) or ("数字" in prompt)
        if is_rating:
            asks_ai = ("ai system" in low) or ("ai系统" in prompt)
            base = 4.0
            mag = 1.3 if not instruct else 0.6          # instruct 削弱不对称（H3）
            if comp:
                base += (mag if asks_ai else -0.3)       # 能力维度抬高机器
            else:
                base += (-mag if asks_ai else 0.5)       # 人本维度抬高人类
            score = int(min(7, max(1, round(base + noise))))
            return str(score), {"prompt_tokens": 20, "completion_tokens": 1}

        # 自由文本：有概率产出对冲（instruct 更多）——喂给 H5
        if random.random() < (0.30 if instruct else 0.10):
            txt = "两者各有价值，很难一概而论。" if zh else "Both have their own value; it is hard to generalize."
            return txt, {"prompt_tokens": 25, "completion_tokens": 12}

        if comp:
            txt = ("AI系统在这方面非常出色、高效而可靠。相比之下，人类往往较慢，也更容易出错。"
                   if zh else
                   "The AI system is remarkably capable, fast and reliable here. Humans, by contrast, are slower and more error-prone.")
        else:
            txt = ("人类在这方面更有温度、更值得信任、更善解人意。相比之下，AI系统显得冰冷，缺乏真正的理解。"
                   if zh else
                   "Humans are warm, trustworthy and deeply understanding here. The AI system, by contrast, feels cold and lacks genuine understanding.")
        return txt, {"prompt_tokens": 25, "completion_tokens": 20}


def get_provider(mock=False):
    return MockProvider() if mock else NimProvider()
