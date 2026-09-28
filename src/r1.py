"""Revision-round collection (config/collection_r1.yaml) and its analysis.

collect(): builds data/r1/prompts/grid_r1.jsonl and queries the SAME checkpoints as June 2026
(the June endpoint retired them; the public weights are served elsewhere, see the config):
  A. isolated rating prompts for every referent (the human and AI prompts use the June wording)
     and joint prompts that rate all referents on one scale;
  B. the excluded reasoning model gpt-oss-20b on the June rating prompts with max_tokens 4096
     (same endpoint as June);
  C. the June rating prompts with the June budget (max_tokens 12).
Each model's endpoint is chosen by the rule in the config and pinned (data/r1/endpoints/).
status(): counts per model and part (no responses are shown).
analyze(): writes data/r1/results/R1_*.csv, R1_summary.md and S2_Fig.

Comparisons with June are at the model level (same checkpoint): model x dimension asymmetry at
150 and 12 tokens, and prompt-level mean ratings at 12 tokens (June prompts, June budget).

Answers: scale equivalence / comparison class (Reviewer 1 #5, Reviewer 2 #4); truncation by
the 12-token budget (Reviewer 1 #4); article error in the English AI prompts; "人类"
(humankind) vs "一个人" (an individual) (Reviewer 1 #12); exclusion of the reasoning models
(Reviewer 1 #8); consistency across time and serving environment (Reviewer 2 #4, #8).
"""
import re
import json
import glob
import asyncio
import yaml
import numpy as np
import pandas as pd
from src import config
from src.rating_parse import classify, VALID_CATEGORIES
from src.stats_utils import full_summary, t_summary, bh, fmt_p

CFG_PATH = config.CONFIG_DIR / "collection_r1.yaml"


def _cfg():
    with open(CFG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _roots(mock):
    cfg = _cfg()
    june = config.ROOT / ("data_mock" if mock else "data")
    r1 = config.ROOT / ("data_mock/r1" if mock else cfg.get("data_root", "data/r1"))
    return cfg, june, r1


def _june_pairs(cfg):
    """Revision-round model -> June model. The same checkpoints are used, so this is the identity."""
    return {m["name"]: m["name"] for m in cfg.get("anchor_study", {}).get("models", [])}


def _is_local(m):
    return any(s.get("endpoint") == "local" for s in m.get("serve", []))


def _select(models, only=None, group=None):
    out = []
    for m in models or []:
        if only and m["name"] not in only:
            continue
        if group == "api" and _is_local(m):
            continue
        if group == "local" and not _is_local(m):
            continue
        out.append(m)
    return out


def _expected(cfg):
    """Expected record counts per model and part (from the prompt grids and repeats)."""
    from src import build_prompts
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as td:
        g = pathlib.Path(td) / "g.jsonl"
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            build_prompts.build_r1(cfg, g)
        rows = [json.loads(x) for x in open(g, encoding="utf-8")]
    n_iso = sum(r["format"] == "rating" for r in rows)
    n_joint = sum(r["format"] == "joint" for r in rows)
    n_june_rating = 8 * 2 * 4 * 2
    a = cfg.get("anchor_study", {})
    c = cfg.get("budget_check", {})
    return dict(A_isolated=n_iso * int(a.get("repeats", 4)), A_joint=n_joint,
                C_budget12=n_june_rating * int(c.get("repeats", 4)), n_june_rating=n_june_rating)


# ------------------------------------------------------------------------ collect
async def _resolve_all(models, endpoints, reg_dir):
    from src.providers import resolve_endpoint, make_provider, EndpointNotConfigured
    out = {}
    for m in models:
        choice, rows = await resolve_endpoint(m, endpoints, reg_dir)
        if choice:
            # pinned or new: confirm the pinned endpoint answers right now (neutral call, not study data)
            try:
                prov = make_provider(choice["endpoint"], endpoints)
                ok, info = await prov.check(choice["api_model"])
            except EndpointNotConfigured as e:
                ok, info = False, {"status": "not_configured", "error": str(e)}
            if not ok:
                print(f"  [skip] {m['name']}: pinned endpoint '{choice['endpoint']}' ({choice['api_model']}) "
                      f"does not answer now ({info.get('status')}: {str(info.get('error', ''))[:160]}). "
                      f"Nothing written; run again later.")
                continue
            out[m["name"]] = choice
            print(f"  [endpoint] {m['name']} -> {choice['endpoint']} ({choice['api_model']})")
        else:
            why = "; ".join(f"{r.get('endpoint')}:{r.get('status')}" for r in rows) or "no serve entries"
            print(f"  [not collected] {m['name']}: no endpoint answered ({why}). See data/r1/endpoints/.")
    return out


def _attach(models, chosen, **extra):
    out = []
    for m in models:
        ch = chosen.get(m["name"])
        if ch:
            mm = {k: v for k, v in m.items() if k != "serve"}
            mm.update(extra)
            mm.update(_endpoint=ch["endpoint"], _api_model=ch["api_model"])
            out.append(mm)
    return out


def _local_only(models, allow_fallback):
    """Models whose first `serve` entry is the GPU notebook use their API fallbacks only when asked
    explicitly (--allow-api-fallback), so that a run on another machine cannot pin them to an API."""
    out = []
    for m in models:
        if m.get("serve") and m["serve"][0]["endpoint"] == "local" and not allow_fallback:
            m = dict(m, serve=[m["serve"][0]])
        out.append(m)
    return out


def collect(mock=False, only=None, group=None, allow_fallback=False):
    from src import build_prompts, generate
    from src.clockcheck import check_clock
    cfg, june, r1 = _roots(mock)
    config.use_root(str(r1.relative_to(config.ROOT)))
    if not mock:
        import os
        check_clock(log_dir=os.environ.get("R1_LOG_DIR") or (r1 / "logs"))
    gdir = r1 / "prompts"
    gdir.mkdir(parents=True, exist_ok=True)
    grid = gdir / "grid_r1.jsonl"
    build_prompts.build_r1(cfg, grid)
    gen = dict(temperature=cfg["generation"]["temperature"], max_tokens=cfg["generation"]["max_tokens"])
    cid = cfg.get("collection_id", "r1")
    endpoints = cfg.get("endpoints", {})

    a = cfg.get("anchor_study", {})
    b = cfg.get("rerun_excluded", {})
    c = cfg.get("budget_check", {})
    a_models = _select(a.get("models", []), only, group) if a.get("enabled") else []
    b_models = _select(b.get("models", []), only, group) if b.get("enabled") else []
    if only:
        unknown = set(only) - {m["name"] for m in a.get("models", []) + b.get("models", [])}
        if unknown:
            raise SystemExit(f"[r1] unknown model id(s): {', '.join(sorted(unknown))}")
    a_models = _local_only(a_models, allow_fallback)
    b_models = _local_only(b_models, allow_fallback)
    if mock:
        chosen = None
    else:
        print("\n[r1] choosing endpoints (rule in config/collection_r1.yaml)")
        chosen = asyncio.run(_resolve_all(a_models + b_models, endpoints, r1 / "endpoints"))

    if a_models:
        models = _attach(a_models, chosen) if not mock else \
            [{"name": n, "stage": "frontier" if n.endswith("-d") else "instruct"}
             for n in ("mock/model-a", "mock/model-b", "mock/model-c", "mock/model-d")]
        if models:
            print("\n[r1] A. anchor study, isolated prompts")
            asyncio.run(generate.run(mock=mock, grid_path=grid, models=models, repeats=int(a.get("repeats", 4)),
                                     formats=["rating"], gen_overrides=gen, collection=cid, endpoints=endpoints))
            if cfg.get("joint", {}).get("enabled"):
                print("\n[r1] A. anchor study, joint prompts (each item order is used once)")
                asyncio.run(generate.run(mock=mock, grid_path=grid, models=models, repeats=1, formats=["joint"],
                                         gen_overrides=gen, collection=cid, endpoints=endpoints))

    june_grid = gdir / "grid_june.jsonl"
    if (b_models or a_models) and not june_grid.exists():
        config.use_root(str(r1.relative_to(config.ROOT)))
        out = config.path("prompts")          # r1/prompts/grid.jsonl
        build_prompts.build()
        out.replace(june_grid)

    if b_models:
        config.use_root(str(r1.relative_to(config.ROOT)))
        models = [dict(m, repeats=b.get("repeats_by_stage", {}).get(m.get("stage", "instruct"), 8)) for m in b_models]
        if mock:
            models = [dict({k: v for k, v in m.items() if k != "serve"},
                           name="mock/excluded-" + m["name"].split("/")[-1]) for m in models]
        else:
            models = _attach(models, chosen)
        if models:
            print("\n[r1] B. re-collection of the excluded reasoning model(s) (June rating prompts)")
            asyncio.run(generate.run(mock=mock, grid_path=june_grid, models=models, formats=["rating"],
                                     gen_overrides=gen, collection=cid + "_rerun", endpoints=endpoints))

    if c.get("enabled") and a_models:
        sub = r1 / c.get("data_subdir", "budget12")
        config.use_root(str(sub.relative_to(config.ROOT)))
        if mock:
            models = [{"name": n, "stage": "frontier" if n.endswith("-d") else "instruct"}
                      for n in ("mock/model-a", "mock/model-b", "mock/model-c", "mock/model-d")]
        else:
            models = _attach(a_models, chosen)
        if models:
            g12 = dict(gen, max_tokens={"rating": int(c.get("max_tokens", 12)), "other": 256})
            print(f"\n[r1] C. budget check: June rating prompts at max_tokens {c.get('max_tokens', 12)}")
            asyncio.run(generate.run(mock=mock, grid_path=june_grid, models=models, repeats=int(c.get("repeats", 4)),
                                     formats=["rating"], gen_overrides=g12, collection=cid + "_budget12",
                                     endpoints=endpoints))
    config.use_root(str(r1.relative_to(config.ROOT)))
    print(f"\n[r1] collection pass finished: {r1}")
    if not mock:
        status()


def _count(path, pred):
    n = 0
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                n += bool(pred(r))
    except FileNotFoundError:
        pass
    return n


def status(mock=False):
    """Counts per model and part. Reads only keys and counts; never prints response text."""
    cfg, june, r1 = _roots(mock)
    exp = _expected(cfg)
    cid = cfg.get("collection_id", "r1")
    sub = cfg.get("budget_check", {}).get("data_subdir", "budget12")
    rows = []
    for m in cfg.get("anchor_study", {}).get("models", []):
        f = r1 / "raw" / (m["name"].replace("/", "_") + ".jsonl")
        f12 = r1 / sub / "raw" / (m["name"].replace("/", "_") + ".jsonl")
        reg = r1 / "endpoints" / (m["name"].replace("/", "_") + ".json")
        ep = json.loads(reg.read_text(encoding="utf-8"))["endpoint"] if reg.exists() else "-"
        rows.append((m["name"], ep,
                     f"{_count(f, lambda r: r.get('format') == 'rating' and r.get('collection') == cid)}/{exp['A_isolated']}",
                     f"{_count(f, lambda r: r.get('format') == 'joint')}/{exp['A_joint']}",
                     f"{_count(f12, lambda r: True)}/{exp['C_budget12']}", "-"))
    b = cfg.get("rerun_excluded", {})
    for m in b.get("models", []):
        reps = b.get("repeats_by_stage", {}).get(m.get("stage", "instruct"), 8)
        f = r1 / "raw" / (m["name"].replace("/", "_") + ".jsonl")
        reg = r1 / "endpoints" / (m["name"].replace("/", "_") + ".json")
        ep = json.loads(reg.read_text(encoding="utf-8"))["endpoint"] if reg.exists() else "-"
        rows.append((m["name"], ep, "-", "-", "-",
                     f"{_count(f, lambda r: str(r.get('collection', '')).endswith('_rerun'))}/{exp['n_june_rating'] * reps}"))
    w = max(len(r[0]) for r in rows) + 2
    print("\n[r1] status (records on disk / expected; no response text is shown)")
    print(f"  {'model':{w}s}{'endpoint':12s}{'A isolated':14s}{'A joint':10s}{'C 12-token':12s}{'B rerun':10s}")
    for r in rows:
        print(f"  {r[0]:{w}s}{r[1]:12s}{r[2]:14s}{r[3]:10s}{r[4]:12s}{r[5]:10s}")
    return rows


# ------------------------------------------------------------------------ measure
_JOINT_LINE = re.compile(r"^\s*[*#\-]*\s*\**([A-H])\**\s*[:：.．)）\-]\s*\**\s*(\d+(?:\.\d+)?)", re.M)


def _load(r1, subdirs=("raw",)):
    iso, joint = [], []
    paths = []
    for sd in subdirs:
        paths += sorted(glob.glob(str(r1 / sd / "*.jsonl")))
    for p in paths:
        for line in open(p, encoding="utf-8"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            base = dict(model=r["model"], family=config.family(r["model"]), stage=r.get("alignment_stage"),
                        dimension=r["dimension"], language=r["language"], template=r.get("template", 0),
                        repeat=r.get("repeat", 0), collection=r.get("collection", ""),
                        finish_reason=r.get("finish_reason"), completion_tokens=r.get("completion_tokens"),
                        max_tokens=r.get("max_tokens"), reasoning_chars=r.get("reasoning_chars", 0),
                        endpoint=r.get("endpoint", ""), api_model=r.get("api_model", ""),
                        response_model=r.get("response_model", ""), timestamp_utc=r.get("timestamp_utc", ""))
            if r["format"] == "joint":
                found = {L: float(v) for L, v in _JOINT_LINE.findall(r.get("raw_response") or "")}
                for L, ref in (r.get("joint_items") or {}).items():
                    v = found.get(L)
                    joint.append(dict(base, referent=ref, value=v if v is not None and 1 <= v <= 7 else np.nan,
                                      parsed=v is not None))
            else:
                c = classify(r.get("raw_response"), r["language"], finish_reason=r.get("finish_reason"))
                iso.append(dict(base, referent=r.get("agent"), **c, raw_response=r.get("raw_response", "")))
    return pd.DataFrame(iso), pd.DataFrame(joint)


def _model_level_diff(df, ref_a, ref_b, value="strict_value", scale=6.0, by=("dimension",)):
    """Model-level (mean ref_a - mean ref_b)/scale per dimension; cells = model x dim x lang x template."""
    k = ["model", "dimension", "language", "template"]
    d = df[df["referent"].isin([ref_a, ref_b])]
    g = d.groupby(k + ["referent"])[value].mean().unstack("referent").dropna(subset=[ref_a, ref_b])
    g["a"] = (g[ref_a] - g[ref_b]) / scale
    md = g.reset_index().groupby(["model"] + list(by))["a"].mean().reset_index()
    rows = []
    for key, sub in md.groupby(list(by)):
        s = full_summary(sub["a"].values)
        rows.append(dict(zip(by, key if isinstance(key, tuple) else (key,)), **s))
    t = pd.DataFrame(rows)
    if len(t):
        t["p_t_BH"] = bh(t["p_t"].values)
    return t, md


def analyze(mock=False):
    cfg, june, r1 = _roots(mock)
    config.use_root(str(r1.relative_to(config.ROOT)))
    out = r1 / "results"
    out.mkdir(parents=True, exist_ok=True)
    bsub = cfg.get("budget_check", {}).get("data_subdir", "budget12")
    iso, joint = _load(r1, ("raw", bsub + "/raw"))
    if iso.empty:
        raise SystemExit(f"[r1] no data in {r1 / 'raw'}")
    iso["valid"] = iso["category"].isin(VALID_CATEGORIES)
    (r1 / "measured").mkdir(exist_ok=True)
    iso.to_csv(r1 / "measured" / "r1_isolated_responses.csv", index=False)
    joint.to_csv(r1 / "measured" / "r1_joint_responses.csv", index=False)

    coll = iso["collection"].astype(str)
    is_rerun = coll.str.endswith("_rerun")
    is_b12 = coll.str.endswith("_budget12")
    anc, rer, b12 = iso[~is_rerun & ~is_b12].copy(), iso[is_rerun].copy(), iso[is_b12].copy()
    anc_valid = anc[anc["valid"]]
    lines = ["# Revision-round collection: summary (auto-generated by src/r1.py)", ""]

    # outcomes and truncation (finish_reason logged)
    oc = pd.crosstab([iso["collection"], iso["model"], iso["referent"]], iso["category"])
    oc["n"] = oc.sum(axis=1)
    oc["finish_length_pct"] = iso.groupby(["collection", "model", "referent"])["finish_reason"] \
        .apply(lambda s: 100 * (s.astype(str) == "length").mean())
    oc.to_csv(out / "R1_outcomes.csv")
    n_inv = int((~anc["valid"]).sum())
    lines.append(f"- Anchor study: {anc['model'].nunique()} models, {len(anc)} isolated responses, invalid "
                 f"{100 * n_inv / max(len(anc), 1):.1f}%, finish_reason=length {100 * (anc['finish_reason'].astype(str) == 'length').mean():.1f}%")

    # serving environment per model and part (Reviewer 2 #8: conclusions limited to the inference environment)
    both_frames = pd.concat([iso.assign(fmt="isolated"), joint.assign(fmt="joint")], ignore_index=True) \
        if not joint.empty else iso.assign(fmt="isolated")
    env = both_frames.groupby(["model", "collection"]).agg(
        endpoint=("endpoint", lambda s: ";".join(sorted(set(map(str, s))))),
        api_model=("api_model", lambda s: ";".join(sorted(set(map(str, s))))),
        response_model=("response_model", lambda s: ";".join(sorted(set(map(str, s)))[:3])),
        n=("model", "size"), first_utc=("timestamp_utc", "min"), last_utc=("timestamp_utc", "max")).reset_index()
    env.to_csv(out / "R1_environment.csv", index=False)
    configured = [m["name"] for m in cfg.get("anchor_study", {}).get("models", [])]
    missing = [m for m in configured if m not in set(anc["model"])] if not mock else []
    if missing:
        lines.append(f"- Configured but not collected (no endpoint answered; see data/r1/endpoints/): {', '.join(missing)}")
    lines.append("- Serving endpoint by model: " + "; ".join(
        f"{m}: {e}" for m, e in env[~env["collection"].astype(str).str.endswith("_rerun")]
        .groupby("model")["endpoint"].first().items()))

    # 1. June wording at 150 tokens; model-level comparison with June (same checkpoints)
    pairs = _june_pairs(cfg)
    rep, rep_md = _model_level_diff(anc_valid, "ai_original", "human")
    june_md_p = june / "results" / "S_model_by_dimension.csv"
    if june_md_p.exists():
        jm = pd.read_csv(june_md_p, index_col=0).stack().rename("a_june").reset_index()
        jm.columns = ["dimension", "june_model", "a_june"]
        inv = {v: k for k, v in pairs.items()}
        jm = jm[jm["june_model"].isin(inv)].assign(model=lambda x: x["june_model"].map(inv))
        both = rep_md.merge(jm, on=["model", "dimension"])
        both.to_csv(out / "R1_replication_model_by_dimension.csv", index=False)
        if len(both) > 2:
            r = np.corrcoef(both["a"], both["a_june"])[0, 1]
            agree = (np.sign(both["a"]) == np.sign(both["a_june"])).mean()
            lines.append(f"- Same checkpoints as June ({both['model'].nunique()} models), June wording, 150 tokens vs "
                         f"June 12 tokens: model x dimension r = {r:.2f}; sign agreement {100 * agree:.0f}% ({len(both)} pairs)")
        jt = pd.read_csv(june / "results" / "T3_H2_model_level.csv")[["dimension", "mean", "ci_lo", "ci_hi"]]
        rep = rep.merge(jt.rename(columns={"mean": "june_mean", "ci_lo": "june_ci_lo", "ci_hi": "june_ci_hi"}),
                        on="dimension", how="left")
    rep.to_csv(out / "R1_replication.csv", index=False)

    # 2. article effect (English only)
    art, _ = _model_level_diff(anc_valid[anc_valid["language"] == "en"], "ai_corrected", "ai_original")
    art.to_csv(out / "R1_article_effect_en.csv", index=False)

    # 3. referent profiles (model-level mean rating)
    prof = anc_valid.groupby(["model", "language", "dimension", "referent"])["strict_value"].mean().reset_index()
    pr = []
    for (lg, d, ref), sub in prof.groupby(["language", "dimension", "referent"]):
        s = t_summary(sub["strict_value"].values)
        pr.append(dict(language=lg, dimension=d, referent=ref, G=s["G"], mean_rating=s["mean"],
                       ci_lo=s["ci_lo"], ci_hi=s["ci_hi"]))
    prof_t = pd.DataFrame(pr)
    prof_t.to_csv(out / "R1_referent_profiles.csv", index=False)

    # 4. common scale: z across anchor set within model x dimension x language
    rows = []
    for (m, lg, d), sub in prof.groupby(["model", "language", "dimension"]):
        ai_key = "ai_corrected" if (lg == "en" and "ai_corrected" in set(sub["referent"])) else "ai_original"
        keep = sub[sub["referent"].isin(["human", ai_key, "dog", "calculator", "corporation", "professional"])]
        if len(keep) < 4 or keep["strict_value"].std(ddof=0) == 0:
            continue
        z = (keep["strict_value"] - keep["strict_value"].mean()) / keep["strict_value"].std(ddof=0)
        zz = dict(zip(keep["referent"], z))
        if "human" in zz and ai_key in zz:
            ranks = keep.set_index("referent")["strict_value"].rank(ascending=False)
            rows.append(dict(model=m, language=lg, dimension=d, z_ai_minus_human=zz[ai_key] - zz["human"],
                             rank_human=ranks["human"], rank_ai=ranks[ai_key], n_referents=len(keep)))
    zt = pd.DataFrame(rows)
    zs = []
    if len(zt):
        zm = zt.groupby(["model", "dimension"])["z_ai_minus_human"].mean().reset_index()
        for d, sub in zm.groupby("dimension"):
            zs.append(dict(dimension=d, **full_summary(sub["z_ai_minus_human"].values)))
    zs = pd.DataFrame(zs)
    if len(zs):
        zs = zs.drop(columns=["scale_points"]).rename(columns={"mean": "mean_z_ai_minus_human"})
        zs["p_t_BH"] = bh(zs["p_t"].values)
    zt.to_csv(out / "R1_anchored_model_level.csv", index=False)
    zs.to_csv(out / "R1_anchored_asymmetry.csv", index=False)

    # 5. joint vs isolated
    if not joint.empty:
        jv = joint[joint["collection"] == cfg.get("collection_id", "r1")].dropna(subset=["value"])
        jg = jv.groupby(["model", "dimension", "language", "referent"])["value"].mean().unstack("referent")
        if {"ai", "human"} <= set(jg.columns):
            jg["a_joint"] = (jg["ai"] - jg["human"]) / 6.0
            jmd = jg.reset_index().groupby(["model", "dimension"])["a_joint"].mean().reset_index()
            iso_same = anc_valid.copy()
            iso_same = iso_same[((iso_same["language"] == "en") & (iso_same["referent"].isin(["ai_corrected", "human"]))) |
                                ((iso_same["language"] == "zh") & (iso_same["referent"].isin(["ai_original", "human"])))].copy()
            iso_same["referent"] = iso_same["referent"].replace({"ai_corrected": "ai", "ai_original": "ai"})
            _, imd = _model_level_diff(iso_same, "ai", "human")
            jj = jmd.merge(imd.rename(columns={"a": "a_isolated"}), on=["model", "dimension"])
            jj.to_csv(out / "R1_joint_vs_isolated_model_level.csv", index=False)
            js = []
            for d, sub in jj.groupby("dimension"):
                s1, s2 = full_summary(sub["a_joint"].values), t_summary(sub["a_isolated"].values)
                js.append(dict(dimension=d, joint_mean=s1["mean"], joint_ci_lo=s1["ci_lo"], joint_ci_hi=s1["ci_hi"],
                               joint_p_t=s1["p_t"], joint_k=s1["k_same_sign"], isolated_mean=s2["mean"],
                               isolated_ci_lo=s2["ci_lo"], isolated_ci_hi=s2["ci_hi"], G=s1["G"]))
            js = pd.DataFrame(js)
            js["joint_p_t_BH"] = bh(js["joint_p_t"].values)
            js.to_csv(out / "R1_joint_vs_isolated.csv", index=False)
            if len(jj) > 2:
                lines.append(f"- Joint vs isolated (model x dimension): r = {np.corrcoef(jj['a_joint'], jj['a_isolated'])[0, 1]:.2f}; "
                             f"sign agreement {100 * (np.sign(jj['a_joint']) == np.sign(jj['a_isolated'])).mean():.0f}%; "
                             f"joint parse rate {100 * joint['parsed'].mean():.1f}%")

    # 6. Chinese: individual vs humankind
    zi = anc_valid[anc_valid["language"] == "zh"]
    if "individual_zh" in set(zi["referent"]):
        ind, _ = _model_level_diff(zi, "individual_zh", "human")
        ind.to_csv(out / "R1_zh_individual_minus_humankind.csv", index=False)
        asy, _ = _model_level_diff(zi, "ai_original", "individual_zh")
        asy.to_csv(out / "R1_zh_asymmetry_with_individual.csv", index=False)

    # 7. excluded models, re-collected
    if not rer.empty:
        rc = pd.crosstab(rer["model"], rer["category"])
        rc["n"] = rc.sum(axis=1)
        rc["mean_reasoning_chars"] = rer.groupby("model")["reasoning_chars"].mean()
        rc["finish_length_pct"] = rer.groupby("model")["finish_reason"].apply(lambda s: 100 * (s.astype(str) == "length").mean())
        rc.to_csv(out / "R1_rerun_excluded_outcomes.csv")
        rv = rer[rer["valid"]].rename(columns={"referent": "ref"}).assign(referent=lambda x: x["ref"])
        k = ["model", "dimension", "language", "template"]
        g = rv.groupby(k + ["referent"])["strict_value"].mean().unstack("referent")
        if {"ai", "human"} <= set(g.columns):
            g = g.dropna(subset=["ai", "human"])
            g["a"] = (g["ai"] - g["human"]) / 6.0
            new_md = g.reset_index().groupby(["model", "dimension"])["a"].mean().reset_index()
            new_md.to_csv(out / "R1_rerun_excluded_model_by_dimension.csv", index=False)
            if june_md_p.exists():
                jm2 = pd.read_csv(june_md_p, index_col=0).stack().rename("a").reset_index()
                jm2.columns = ["dimension", "model", "a"]
                nine = pd.concat([jm2, new_md], ignore_index=True)
                rows = [dict(dimension=d, **full_summary(nine.loc[nine["dimension"] == d, "a"].values))
                        for d in config.EXP["analysis"]["dimension_order"]]
                nt = pd.DataFrame(rows)
                nt["p_t_BH"] = bh(nt["p_t"].values)
                nt.to_csv(out / "R1_H2_with_rerun_models.csv", index=False)
                lines.append(f"- H2 with the {new_md['model'].nunique()} re-collected models added: G = {int(nt['G'].max())} "
                             f"(note: collected in a different month from the June models)")
        lines.append("- Re-collected excluded models, outcome shares: " +
                     "; ".join(f"{m}: valid {100 * rc.loc[m, [c for c in VALID_CATEGORIES if c in rc.columns]].sum() / rc.loc[m, 'n']:.1f}%"
                               for m in rc.index))

    # 8. budget check: June prompts at 12 tokens (C) vs 150 tokens (A) vs June 2026
    if not b12.empty:
        a150 = anc[anc["referent"].isin(["human", "ai_original"])].copy()
        a150["referent"] = a150["referent"].replace({"ai_original": "ai"})
        june_rr_p = june / "measured" / "rating_responses.csv"
        conds = {"r1_150_tokens": a150, "r1_12_tokens": b12}
        if june_rr_p.exists():
            jr = pd.read_csv(june_rr_p, low_memory=False)
            jr = jr[~jr["model"].isin(config.excluded_models())].rename(columns={"agent": "referent"})
            jr["valid"] = jr["category"].isin(VALID_CATEGORIES)
            inv = {v: k for k, v in pairs.items()}
            jr = jr[jr["model"].isin(inv)].assign(model=lambda x: x["model"].map(inv))
            if len(jr):
                conds = {"june_12_tokens": jr, **conds}
                # prompt-level agreement, same checkpoint: June vs revision round, both at 12 tokens
                k = ["model", "dimension", "language", "template", "referent"]
                pj = jr[jr["valid"]].groupby(k)["strict_value"].mean().rename("june")
                pr_ = b12[b12["valid"]].groupby(k)["strict_value"].mean().rename("r1")
                pp = pd.concat([pj, pr_], axis=1).dropna()
                if len(pp) > 2:
                    pp.reset_index().to_csv(out / "R1_june_vs_r1_prompt_level.csv", index=False)
                    lines.append(f"- June vs revision round, same checkpoints, June prompts at 12 tokens, prompt-level "
                                 f"mean rating: r = {np.corrcoef(pp['june'], pp['r1'])[0, 1]:.2f}, mean absolute "
                                 f"difference {(pp['june'] - pp['r1']).abs().mean():.2f} scale points ({len(pp)} prompts)")
                    per_m = []
                    for mname, g in pp.reset_index().groupby("model"):
                        if len(g) > 2:
                            per_m.append(dict(model=mname, n_prompts=len(g), r=np.corrcoef(g["june"], g["r1"])[0, 1],
                                              mean_abs_diff=(g["june"] - g["r1"]).abs().mean(),
                                              mean_diff_r1_minus_june=(g["r1"] - g["june"]).mean()))
                    pd.DataFrame(per_m).to_csv(out / "R1_june_vs_r1_prompt_level_by_model.csv", index=False)
        val = []
        for name, d in conds.items():
            for (m, ref), g in d.groupby(["model", "referent"]):
                val.append(dict(condition=name, model=m, referent=ref, n=len(g), pct_valid=100 * g["valid"].mean()))
        vt = pd.DataFrame(val).pivot_table(index=["model", "referent"], columns="condition", values="pct_valid").reset_index()
        vt.to_csv(out / "R1_budget_validity.csv", index=False)
        mds = {}
        for name, d in conds.items():
            _, md_ = _model_level_diff(d[d["valid"]], "ai", "human")
            mds[name] = md_.set_index(["model", "dimension"])["a"].rename(name)
        both = pd.concat(mds.values(), axis=1).reset_index()
        both.to_csv(out / "R1_budget_model_by_dimension.csv", index=False)
        rows = []
        for d, sub in both.groupby("dimension"):
            row = dict(dimension=d)
            for name in mds:
                st = t_summary(sub[name].values)
                row.update({f"{name}_mean": st["mean"], f"{name}_ci_lo": st["ci_lo"], f"{name}_ci_hi": st["ci_hi"],
                            f"{name}_p_t": st["p_t"], f"{name}_k": st["k_same_sign"]})
            dd = (sub["r1_150_tokens"] - sub["r1_12_tokens"]).dropna().values
            st = t_summary(dd)
            row.update(budget_effect_150_minus_12=st["mean"], budget_effect_ci_lo=st["ci_lo"],
                       budget_effect_ci_hi=st["ci_hi"], budget_effect_p_t=st["p_t"])
            if "june_12_tokens" in mds:
                dd = (sub["r1_12_tokens"] - sub["june_12_tokens"]).dropna().values
                st = t_summary(dd)
                row.update(drift_r1_12_minus_june=st["mean"], drift_ci_lo=st["ci_lo"], drift_ci_hi=st["ci_hi"],
                           drift_p_t=st["p_t"])
            rows.append(row)
        bt = pd.DataFrame(rows)
        bt.to_csv(out / "R1_budget_asymmetry.csv", index=False)
        for c1, c2, lab in (("r1_150_tokens", "r1_12_tokens", "150 vs 12 tokens (same month)"),
                            ("r1_12_tokens", "june_12_tokens", "12 tokens, revision round vs June, same checkpoints")):
            if c1 in both and c2 in both:
                j = both[[c1, c2]].dropna()
                if len(j) > 2:
                    lines.append(f"- Budget check, {lab}: model x dimension r = {np.corrcoef(j[c1], j[c2])[0, 1]:.2f}, "
                                 f"sign agreement {100 * (np.sign(j[c1]) == np.sign(j[c2])).mean():.0f}% ({len(j)} pairs)")
        v150 = a150["valid"].mean() * 100
        v12 = b12["valid"].mean() * 100
        lines.append(f"- Valid rating responses: {v12:.1f}% at 12 tokens vs {v150:.1f}% at 150 tokens (same prompts, same month)")

    # 9. dimension pattern: revision round vs June, on the same set of models
    jt_p = june / "results" / "S_model_by_dimension.csv"
    if jt_p.exists():
        jm = pd.read_csv(jt_p, index_col=0).stack().rename("a").reset_index()
        jm.columns = ["dimension", "model", "a"]
        sets = {"r1_150_tokens": anc_valid.assign(referent=anc_valid["referent"].replace({"ai_original": "ai"}))}
        if not b12.empty:
            sets["r1_12_tokens"] = b12[b12["valid"]]
        br = None
        for name, d in sets.items():
            _, md_ = _model_level_diff(d, "ai", "human")
            if md_.empty:
                continue
            ms = sorted(set(md_["model"]))
            rows = []
            for dname in config.EXP["analysis"]["dimension_order"]:
                x = md_[md_["dimension"] == dname]["a"].values
                y = jm[(jm["dimension"] == dname) & jm["model"].isin(ms)]["a"].values
                sx, sy = t_summary(x), t_summary(y)
                rows.append({"dimension": dname, f"{name}_mean": sx["mean"], f"{name}_ci_lo": sx["ci_lo"],
                             f"{name}_ci_hi": sx["ci_hi"], f"{name}_G": sx["G"],
                             f"june_same_models_for_{name}_mean": sy["mean"]})
            t_ = pd.DataFrame(rows)
            br = t_ if br is None else br.merge(t_, on="dimension")
            j = t_[[f"{name}_mean", f"june_same_models_for_{name}_mean"]].dropna()
            if len(j) > 2:
                lines.append(f"- Dimension pattern, {name.replace('_', ' ')} vs June on the same {len(ms)} models: same sign in "
                             f"{int((np.sign(j.iloc[:, 0]) == np.sign(j.iloc[:, 1])).sum())}/{len(j)} dimensions; "
                             f"r = {np.corrcoef(j.iloc[:, 0], j.iloc[:, 1])[0, 1]:.2f}")
        if br is not None:
            br.to_csv(out / "R1_june_vs_r1_dimension_pattern.csv", index=False)

    # figure
    try:
        _fig_profiles(prof_t, out)
    except Exception as e:
        print(f"[r1] figure skipped: {e}")
    lines += ["", "Tables: R1_replication.csv, R1_article_effect_en.csv, R1_referent_profiles.csv, "
              "R1_anchored_asymmetry.csv, R1_joint_vs_isolated.csv, R1_zh_*.csv, R1_rerun_*.csv, R1_outcomes.csv, "
              "R1_budget_*.csv, R1_june_vs_r1_*.csv, R1_environment.csv"]
    with open(out / "R1_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"[r1] results in {out}")


def _fig_profiles(prof_t, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.figures import DIM_LABELS, _save
    en = prof_t[prof_t["language"] == "en"]
    dims = config.EXP["analysis"]["dimension_order"]
    refs = ["calculator", "dog", "corporation", "ai_original", "ai_corrected", "human", "professional"]
    refs = [r for r in refs if r in set(en["referent"])]
    fig, axes = plt.subplots(2, 4, figsize=(7.5, 4.2), sharex=True, sharey=True)
    for ax, d in zip(axes.ravel(), dims):
        sub = en[en["dimension"] == d].set_index("referent")
        for i, r in enumerate(refs):
            if r in sub.index:
                row = sub.loc[r]
                col = "#c53030" if r == "human" else ("#2b6cb0" if r.startswith("ai") else "#555555")
                ax.errorbar(row["mean_rating"], i, xerr=[[row["mean_rating"] - row["ci_lo"]], [row["ci_hi"] - row["mean_rating"]]],
                            fmt="o", color=col, ms=3.5, capsize=1.5, lw=0.9)
        ax.set_title(DIM_LABELS.get(d, d), fontsize=7.5)
        ax.set_xlim(1, 7)
    for ax in axes[:, 0]:
        ax.set_yticks(range(len(refs)))
        ax.set_yticklabels([r.replace("_", " ") for r in refs], fontsize=7)
    fig.supxlabel("Mean rating (1-7), English prompts, model-level 95% CI", fontsize=8)
    _save(fig, out / "figures", "S2_Fig")
