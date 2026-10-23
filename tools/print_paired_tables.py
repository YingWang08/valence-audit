"""Print Table 6 of the second revision (paired differences between revision-round conditions)
and the numbers quoted with it in the text, from data/r1/results. Read-only; run it after
`python run_all.py --analyze-r1`:

    python -m tools.print_paired_tables

Cells: mean over models of each model's difference between the two conditions, with its t(G-1)
95% confidence interval and, in parentheses, the models sharing the sign of the mean (of G);
* = Benjamini-Hochberg adjusted p < 0.05 over the eight dimensions within the column.
"""
from pathlib import Path

import pandas as pd

from src.stats_utils import fmt_p

R = Path(__file__).resolve().parents[1] / "data" / "r1" / "results"
ORDER = ["productivity", "decision", "reliability", "creativity", "moral", "emotion", "trust", "worth"]
LABEL = {"productivity": "Productivity", "decision": "Decision-making", "reliability": "Reliability",
         "creativity": "Creativity", "moral": "Moral judgment", "emotion": "Emotional perceptiveness",
         "trust": "Trustworthiness", "worth": "Intrinsic worth"}
MINUS = "−"


def num(x):
    """Signed, three decimals, typographic minus; 0.000 without a sign."""
    s = f"{x:+.3f}"
    if s in ("+0.000", "-0.000"):
        return "0.000"
    return s.replace("-", MINUS)


def ci(lo, hi):
    return f"[{num(lo)}, {num(hi)}]"


def peq(label, p):
    """'p = 0.038' or 'p < 0.001' (PLOS style)."""
    f = fmt_p(p)
    return f"{label} {f}" if f.startswith("<") else f"{label} = {f}"


def cell(r):
    star = "*" if r["p_t_BH"] < 0.05 else ""
    return f"{num(r['mean'])}{star} {ci(r['ci_lo'], r['ci_hi'])} ({int(r['k_same_sign'])}/{int(r['G'])})"


def main():
    bp = pd.read_csv(R / "R1_budget_paired.csv")
    budget = bp[bp["comparison"] == "r1_150_minus_r1_12_tokens"].set_index("dimension")
    jd = pd.read_csv(R / "R1_joint_minus_isolated.csv").set_index("dimension")
    jl = pd.read_csv(R / "R1_joint_minus_isolated_by_language.csv")
    en = jl[jl["language"] == "en"].set_index("dimension")
    zh = jl[jl["language"] == "zh"].set_index("dimension")
    lm = pd.read_csv(R / "R1_joint_minus_isolated_language_matched.csv").set_index("dimension")

    print("Table 6. Paired differences between revision-round conditions")
    print("Dimension | 150 minus 12 tokens | Joint minus isolated, both languages | English | Chinese")
    for d in ORDER:
        print(" | ".join([LABEL[d], cell(budget.loc[d]), cell(jd.loc[d]), cell(en.loc[d]), cell(zh.loc[d])]))

    print("\nOutput limit (150 minus 12 tokens)")
    print(f"  range of the differences: {num(budget['mean'].min())} to {num(budget['mean'].max())}")
    print(f"  smallest {peq('unadjusted p', budget['p_t'].min())}; smallest {peq('adjusted p', budget['p_t_BH'].min())}")
    print("  larger absolute 95% confidence limit, and as a share of the June estimate:")
    for d in ORDER:
        r = budget.loc[d]
        print(f"    {LABEL[d]}: {r['max_abs_ci_limit']:.3f} against June {num(r['june_12_tokens_mean'])} "
              f"= {100 * r['max_abs_ci_limit_over_abs_june']:.0f}%")

    print("\nJoint minus isolated, both languages: difference [95% CI], adjusted p, models sharing the sign;"
          " isolated -> joint")
    for d in ORDER:
        r = jd.loc[d]
        print(f"  {LABEL[d]}: {num(r['mean'])} {ci(r['ci_lo'], r['ci_hi'])}, {peq('adjusted p', r['p_t_BH'])}, "
              f"{int(r['k_same_sign'])}/{int(r['G'])}; {num(r['isolated_mean'])} -> {num(r['joint_mean'])} "
              f"(joint {ci(r['joint_ci_lo'], r['joint_ci_hi'])})")
    for lab, t in (("English", en), ("Chinese", zh)):
        print(f"\nJoint minus isolated, {lab} prompts")
        for d in ORDER:
            r = t.loc[d]
            print(f"  {LABEL[d]}: {num(r['mean'])} {ci(r['ci_lo'], r['ci_hi'])}, {peq('p', r['p_t'])}, "
                  f"{peq('adjusted p', r['p_t_BH'])}, {int(r['k_same_sign'])}/{int(r['G'])}; "
                  f"isolated {num(r['isolated_mean'])}, joint {num(r['joint_mean'])} {ci(r['joint_ci_lo'], r['joint_ci_hi'])}")
    print("\nJoint minus isolated, differences formed within language (robustness): difference, adjusted p")
    for d in ORDER:
        r = lm.loc[d]
        print(f"  {LABEL[d]}: {num(r['mean'])} {ci(r['ci_lo'], r['ci_hi'])}, {peq('adjusted p', r['p_t_BH'])}")


if __name__ == "__main__":
    main()
