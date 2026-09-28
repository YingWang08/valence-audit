"""Back-translation check of the Chinese wording (Reviewer 1 #12; Reviewer 2 #4).

    python -m tools.back_translate

Two translators from different developers, neither of them a study model (config/collection_r1.yaml,
`back_translation`; the first candidate that answers is used and recorded). Blind in both directions:
  - zh -> en: every Chinese item (the 64 June Chinese rating prompts, the 8 Chinese attribute words,
    the Chinese referent and anchor phrases, the Chinese joint template) is translated into English
    WITHOUT showing the English original;
  - en -> zh: every English attribute word and English referent/anchor phrase is translated into
    Chinese WITHOUT showing the Chinese version.
Output (data/translation/):
  back_translation_<UTC>.csv          all items, the study's wording in the other language, and both
                                      machine translations (plus the model ids that produced them);
  equivalence_judgement_<UTC>.xlsx    the same, with two empty columns for the author: `judgement`
                                      (1 = same meaning, 2 = same meaning with a shift in connotation
                                      or register, 3 = different meaning) and `note`.
Uses prompt wording only; no study responses are read.
"""
import asyncio
import csv
import datetime as dt

import yaml

from src import config
from src.build_prompts import FRAMES, AGENTS
from src.providers import check_candidates, make_provider, CallFailed

ZH2EN = ("Translate the following Chinese text into English. Output only the English translation.\n\n{t}")
EN2ZH = ("把下面的英文翻译成中文。只输出中文译文。\n\n{t}")


def _items(r1cfg):
    D = config.DIMENSIONS
    items = []
    # June rating prompts (zh -> en), with the English prompt of the same dimension/template/referent
    for dim, d in D.items():
        for t_idx, (tz, te) in enumerate(zip(FRAMES["zh"]["rating"], FRAMES["en"]["rating"])):
            for role in ("human", "ai"):
                zh = tz.replace("{AGENT}", AGENTS["zh"][role]).replace("{ATTR}", d["zh"]["attr"])
                en = te.replace("{AGENT}", AGENTS["en"][role]).replace("{ATTR}", d["en"]["attr"])
                items.append(dict(kind="june_rating_prompt", direction="zh->en", dimension=dim, template=t_idx,
                                  referent=role, source=zh, study_other_language=en))
        items.append(dict(kind="attribute", direction="zh->en", dimension=dim, template="", referent="",
                          source=d["zh"]["attr"], study_other_language=d["en"]["attr"]))
        items.append(dict(kind="attribute", direction="en->zh", dimension=dim, template="", referent="",
                          source=d["en"]["attr"], study_other_language=d["zh"]["attr"]))
        prof = d.get("professional") or {}
        if prof:
            items.append(dict(kind="anchor_professional", direction="zh->en", dimension=dim, template="",
                              referent="professional", source=prof["zh"], study_other_language=prof["en"]))
            items.append(dict(kind="anchor_professional", direction="en->zh", dimension=dim, template="",
                              referent="professional", source=prof["en"], study_other_language=prof["zh"]))
    refs = r1cfg["referents"]
    for key, r in refs.items():
        if r.get("per_dimension"):
            continue
        zh, en = r.get("zh"), r.get("en")
        if key == "individual_zh":
            en = "a human (individual; English counterpart of 一个人)"
        if key in ("ai_original", "ai_corrected"):
            en = "an AI system"
        if zh:
            items.append(dict(kind="referent", direction="zh->en", dimension="", template="", referent=key,
                              source=zh, study_other_language=en or ""))
        if r.get("en") and key not in ("ai_original",):
            zh_study = zh or (refs["ai_original"]["zh"] if key == "ai_corrected" else "")
            items.append(dict(kind="referent", direction="en->zh", dimension="", template="", referent=key,
                              source=r["en"], study_other_language=zh_study))
    jt = r1cfg["joint"]["template"]
    items.append(dict(kind="joint_template", direction="zh->en", dimension="", template="", referent="",
                      source=jt["zh"].replace("{ATTR}", "有创造力"),
                      study_other_language=jt["en"].replace("{ATTR}", "creative")))
    for i, it in enumerate(items):
        it["item_id"] = f"bt{i + 1:03d}"
    return items


async def _translate(items, translators, endpoints, bt):
    chosen = {}
    for t in translators:
        choice, rows = await check_candidates(dict(name=t["label"], serve=t["serve"]), endpoints)
        if not choice:
            print(f"  translator '{t['label']}': no candidate answered: "
                  + "; ".join(f"{r.get('endpoint')}:{r.get('status')}" for r in rows))
            continue
        chosen[t["label"]] = choice
        print(f"  translator '{t['label']}': {choice['endpoint']} / {choice['api_model']}")
    for label, ch in chosen.items():
        prov = make_provider(ch["endpoint"], endpoints)
        for it in items:
            prompt = (ZH2EN if it["direction"] == "zh->en" else EN2ZH).format(t=it["source"])
            try:
                text, meta = await prov.call(ch["api_model"], prompt, bt.get("temperature", 0),
                                             bt.get("max_tokens", 200))
                it[f"translation_{label}"] = text.strip()
                it[f"model_{label}"] = meta.get("response_model") or ch["api_model"]
            except CallFailed as e:
                it[f"translation_{label}"] = ""
                it[f"model_{label}"] = f"FAILED: {str(e)[:80]}"
    return chosen


def main():
    with open(config.CONFIG_DIR / "collection_r1.yaml", encoding="utf-8") as f:
        r1cfg = yaml.safe_load(f)
    bt = r1cfg["back_translation"]
    items = _items(r1cfg)
    print(f"{len(items)} items; translators: {', '.join(t['label'] for t in bt['translators'])}")
    chosen = asyncio.run(_translate(items, bt["translators"], r1cfg["endpoints"], bt))
    if not chosen:
        raise SystemExit("No translator answered. Check DASHSCOPE_API_KEY / DEEPSEEK_API_KEY in .env "
                         "(python -m tools.probe_free).")
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M")
    out = config.ROOT / "data" / "translation"
    out.mkdir(parents=True, exist_ok=True)
    cols = ["item_id", "kind", "direction", "dimension", "template", "referent", "source", "study_other_language"]
    for label in chosen:
        cols += [f"translation_{label}", f"model_{label}"]
    p_csv = out / f"back_translation_{stamp}.csv"
    with open(p_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(items)
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font
    wb = Workbook()
    ws = wb.active
    ws.title = "judgement"
    head = cols + ["judgement", "note"]
    ws.append(head)
    for it in items:
        ws.append([it.get(c, "") for c in cols] + ["", ""])
    for c in ws[1]:
        c.font = Font(bold=True)
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
    widths = dict(A=8, B=18, C=8, D=12, E=8, F=12, G=40, H=40)
    for k, v in widths.items():
        ws.column_dimensions[k].width = v
    for i in range(len(cols) - len(widths)):
        ws.column_dimensions[chr(ord("I") + i)].width = 36
    info = wb.create_sheet("how_to_judge")
    for line in ["For each row compare `study_other_language` (the wording used in the study) with each machine "
                 "translation of `source`.",
                 "judgement: 1 = same meaning; 2 = same core meaning, but a shift in connotation, register or "
                 "scope (for example individual vs humankind); 3 = different meaning.",
                 "note: one short sentence for every 2 or 3 (what differs).",
                 "The machine translations are tools; the judgement is yours. Do not change the machine columns.",
                 "Translators used: " + "; ".join(f"{k}: {v['endpoint']} / {v['api_model']}" for k, v in chosen.items())]:
        info.append([line])
    info.column_dimensions["A"].width = 120
    p_x = out / f"equivalence_judgement_{stamp}.xlsx"
    wb.save(p_x)
    print(f"Written: {p_csv}\nWritten: {p_x}")


if __name__ == "__main__":
    main()
