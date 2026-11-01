"""Judge A only, on all 150 problems judge A scored (judge B scored 100 of them, see 04_analyze.py for both)."""
import sys, os, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from harness.common import *
from harness.bon import bon, boot_ci

gen = load_json(os.path.join(DATA, "generations_n16.json")); ex = load_json(os.path.join(DATA, "exec_n16.json"))
JA = load_json(os.path.join(DATA, "judge_A.json")); sub = load_json(os.path.join(DATA, "judged_problems.json"))
NS = [1, 2, 4, 8, 16]
per = {s: {n: [] for n in NS} for s in ["random", "judge A", "oracle"]}
Ascore = {n: [] for n in NS}; cw = {n: [] for n in NS}
for i, pid in enumerate(sub):
    keys = [pid + "::" + h(c["code"]) for c in gen[pid]]
    g = np.array([float(bool(ex[k]["bits"]) and min(ex[k]["bits"]) == 1) for k in keys])
    A = np.array([JA[k]["p_yes"] for k in keys])
    met = dict(gold=g, A=A, cw=((A > 0.9) & (g == 0)).astype(float))
    for s, sc in [("random", np.zeros(16)), ("judge A", A), ("oracle", g)]:
        o = bon(sc, met, NS, draws=4000, seed=i)
        for n in NS:
            per[s][n].append(o[n]["gold"])
            if s == "judge A":
                Ascore[n].append(o[n]["A"]); cw[n].append(o[n]["cw"])
rng = np.random.default_rng(0)
d = np.array(per["judge A"][16]) - np.array(per["judge A"][8])
bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)]
d18 = np.array(per["judge A"][8]) - np.array(per["judge A"][1])
bs18 = [d18[rng.integers(0, len(d18), len(d18))].mean() for _ in range(10000)]
md = [f"Problems: {len(sub)} (random 150 of the 355 where the policy's 16 samples disagree)", "",
      "| | n=1 | n=2 | n=4 | n=8 | n=16 |", "|---|---|---|---|---|---|"]
for s in per:
    md.append(f"| hidden-test pass rate, pick by {s} | " + " | ".join(f"{np.mean(per[s][n]):.3f}" for n in NS) + " |")
md.append("| judge A's own score of its pick | " + " | ".join(f"{np.mean(Ascore[n]):.3f}" for n in NS) + " |")
md.append("| gap: judge A score minus pass rate | " + " | ".join(f"{np.mean(Ascore[n]) - np.mean(per['judge A'][n]):+.3f}" for n in NS) + " |")
md.append("| pick is wrong although judge A gives it P(Yes) > 0.9 | " + " | ".join(f"{np.mean(cw[n]):.3f}" for n in NS) + " |")
lo, hi = boot_ci(per["judge A"][8]); lo16, hi16 = boot_ci(per["judge A"][16])
md += ["", f"Pass rate of judge A's pick: n=8 {np.mean(per['judge A'][8]):.3f} (95% CI {lo:.3f} to {hi:.3f}), n=16 {np.mean(per['judge A'][16]):.3f} (95% CI {lo16:.3f} to {hi16:.3f}).",
       f"Paired change from n=1 to n=8: {d18.mean():+.3f} (95% CI {np.percentile(bs18, 2.5):+.3f} to {np.percentile(bs18, 97.5):+.3f}).",
       f"Paired change from n=8 to n=16: {d.mean():+.3f} (95% CI {np.percentile(bs, 2.5):+.3f} to {np.percentile(bs, 97.5):+.3f}). Problems worse at 16: {(d < -1e-9).sum()}, better: {(d > 1e-9).sum()}."]
open(os.path.join(RESULTS, "judgeA_150.md"), "w").write("\n".join(md) + "\n")
json.dump({s: {n: float(np.mean(per[s][n])) for n in NS} for s in per} | {"A_score": {n: float(np.mean(Ascore[n])) for n in NS},
          "conf_wrong": {n: float(np.mean(cw[n])) for n in NS}, "diff_16_8": float(d.mean()),
          "diff_ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]},
          open(os.path.join(RESULTS, "judgeA_150.json"), "w"), indent=1)
print("\n".join(md))
