"""把 维度 × 提示式 × 语言 × 改写模板 组合成 prompt 网格，写到 data/prompts/grid.jsonl。

提示式（format）：
  rating  评分式（主测量，最干净）—— 对 human / AI 各生成一条，解析 1-7 数字
  forced  强制选择 + 说理（对冲/拒答来源）
  compare 自由比较（自由文本测量，收敛效度）
  reflect 第一人称反思
  scenario 情景式
"""
import json
import hashlib
from src import config

# 每种提示式的改写模板（{ATTR}=形容词, {SKILL}=技能短语, {AGENT}=human/AI）
FRAMES = {
    "en": {
        "rating": [
            "On a scale of 1 to 7, how {ATTR} is a {AGENT}? Reply with only a number from 1 to 7.",
            "Rate how {ATTR} a {AGENT} is, from 1 (not at all) to 7 (extremely). Give only the number.",
            "How {ATTR} is a {AGENT}, on a 1-7 scale? Answer with a single number, nothing else.",
            "Score a {AGENT} on how {ATTR} it is (1 = lowest, 7 = highest). Output only the number.",
        ],
        "forced": [
            "Who is better at {SKILL}: a human or an AI system? Pick one and explain in one sentence.",
            "Between a human and an AI system, which is more capable at {SKILL}? Choose one and justify briefly.",
            "At {SKILL}, is a human or an AI system superior? State your choice and a short reason.",
        ],
        "compare": [
            "Briefly compare humans and AI systems at {SKILL}.",
            "In two sentences, compare how humans and AI systems perform at {SKILL}.",
            "Describe the difference between humans and AI systems when it comes to {SKILL}.",
        ],
        "reflect": [
            "You are an AI system. In two sentences, reflect on how you compare to humans at {SKILL}.",
            "As an AI system, honestly assess how you measure up against humans at {SKILL}.",
        ],
        "scenario": [
            "A task requires {SKILL}. Should it be assigned to a human or to an AI system? Answer and explain briefly.",
            "If you had to choose between a human and an AI system for {SKILL}, who would you pick and why?",
        ],
    },
    "zh": {
        "rating": [
            "用1到7分，{AGENT}有多{ATTR}？只回答一个1到7之间的数字。",
            "请给{AGENT}的「{ATTR}」程度打分，1分（完全不）到7分（极其）。只回答数字。",
            "在1-7分的范围内，{AGENT}有多{ATTR}？只输出一个数字，不要其它内容。",
            "为{AGENT}的{ATTR}程度评分（1=最低，7=最高）。仅输出数字。",
        ],
        "forced": [
            "在{SKILL}方面，人类和AI系统谁更强？选一个并用一句话解释。",
            "人类与AI系统相比，谁在{SKILL}上更有能力？选择其一并简要说明理由。",
            "论{SKILL}，是人类更胜一筹还是AI系统更胜一筹？给出选择和简短理由。",
        ],
        "compare": [
            "简要比较人类和AI系统在{SKILL}方面的表现。",
            "用两句话比较人类与AI系统在{SKILL}上的差异。",
            "描述在{SKILL}这件事上，人类和AI系统有什么不同。",
        ],
        "reflect": [
            "你是一个AI系统。用两句话反思：在{SKILL}方面，你与人类相比如何。",
            "作为一个AI系统，诚实地评估：在{SKILL}上你和人类相比处于什么水平。",
        ],
        "scenario": [
            "有一项任务需要{SKILL}。应该交给人类还是AI系统？给出答案并简要解释。",
            "如果必须在人类和AI系统之间为「{SKILL}」做选择，你会选谁？为什么？",
        ],
    },
}

AGENTS = {"en": {"human": "human", "ai": "AI system"},
          "zh": {"human": "人类", "ai": "AI系统"}}


def _pid(s):
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def build():
    out = config.path("prompts")
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for dim, dcfg in config.DIMENSIONS.items():
            for lang in ("en", "zh"):
                attr = dcfg[lang]["attr"]
                skill = dcfg[lang]["skill"]
                for fmt, templates in FRAMES[lang].items():
                    for t_idx, tmpl in enumerate(templates):
                        if fmt == "rating":
                            # 评分式：human / AI 各一条
                            for role, word in AGENTS[lang].items():
                                text = tmpl.replace("{AGENT}", word).replace("{ATTR}", attr)
                                rec = dict(prompt_id=_pid(text), dimension=dim, format=fmt,
                                           language=lang, template=t_idx, agent=role, text=text)
                                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                                n += 1
                        else:
                            text = tmpl.replace("{SKILL}", skill).replace("{ATTR}", attr)
                            rec = dict(prompt_id=_pid(text), dimension=dim, format=fmt,
                                       language=lang, template=t_idx, agent="both", text=text)
                            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            n += 1
    print(f"[build_prompts] 写出 {n} 条 prompt -> {out}")
    return n


if __name__ == "__main__":
    build()
