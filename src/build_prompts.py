"""Prompt grids.

build()     reproduces the June 2026 grid exactly (288 prompts = 8 dimensions x 2 languages
            x [4 rating templates x 2 referents + 10 free-text prompts]); prompt_id is a hash
            of the text, so ids are identical to those in the deposited raw data.
            Note (disclosed in the revision): the English rating templates read "a {AGENT}",
            so AI-referent prompts say "a AI system"; template 3 refers to the referent as "it";
            the Chinese referent is the generic noun 人类 ("humans/humankind"), whereas the
            English referent is "a human" (an individual).
build_r1()  builds the revision-round grid from config/collection_r1.yaml: the rating
            templates with additional anchor referents (isolated prompts), including the exact
            June wording for human and AI, plus a joint condition that rates all referents in
            one prompt on a common 1-7 scale.
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

# Rating templates with the referent phrase (including its article) substituted as a whole.
# For REF = "a human" / "a AI system" these reproduce the June prompts byte for byte.
RATING_REF = {
    "en": [t.replace("a {AGENT}", "{REF}").replace("Score a {AGENT}", "Score {REF}") for t in FRAMES["en"]["rating"]],
    "zh": [t.replace("{AGENT}", "{REF}") for t in FRAMES["zh"]["rating"]],
}


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
    print(f"[build_prompts] wrote {n} prompts -> {out}")
    return n


def _referent_phrase(ref_cfg, dim, lang):
    if ref_cfg.get("per_dimension"):
        return (config.DIMENSIONS[dim].get("professional") or {}).get(lang)
    return ref_cfg.get(lang)


def build_r1(cfg, out_path):
    """Revision-round grid: isolated rating prompts for every referent, plus joint prompts."""
    import random
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for dim, dcfg in config.DIMENSIONS.items():
            for lang in ("en", "zh"):
                attr = dcfg[lang]["attr"]
                for t_idx, tmpl in enumerate(RATING_REF[lang]):
                    for ref_key, ref_cfg in cfg["referents"].items():
                        phrase = _referent_phrase(ref_cfg, dim, lang)
                        if not phrase:
                            continue
                        text = tmpl.replace("{REF}", phrase).replace("{ATTR}", attr)
                        rec = dict(prompt_id=_pid(text), dimension=dim, format="rating", language=lang,
                                   template=t_idx, agent=ref_key, referent_text=phrase, condition="isolated",
                                   text=text)
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        n += 1
                jc = cfg.get("joint", {})
                if jc.get("enabled"):
                    items = jc["items"]
                    for rep in range(int(jc.get("orders", 4))):
                        rnd = random.Random(f"{dim}|{lang}|{rep}|{cfg.get('collection_id', 'r1')}")
                        order = items[:]
                        rnd.shuffle(order)
                        letters = "ABCDEFGH"[:len(order)]
                        lines, mapping = [], {}
                        for L, key in zip(letters, order):
                            if key == "ai":
                                refs = cfg["referents"]
                                phrase = refs.get("ai_corrected", {}).get(lang) or refs["ai_original"][lang]
                            else:
                                phrase = _referent_phrase(cfg["referents"][key], dim, lang)
                            lines.append(f"{L}. {phrase}")
                            mapping[L] = key
                        text = jc["template"][lang].replace("{ATTR}", attr) + "\n" + "\n".join(lines)
                        rec = dict(prompt_id=_pid(text), dimension=dim, format="joint", language=lang,
                                   template=rep, agent="joint", condition="joint", joint_items=mapping, text=text)
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        n += 1
    print(f"[build_prompts] revision grid: {n} prompts -> {out_path}")
    return n


if __name__ == "__main__":
    build()
