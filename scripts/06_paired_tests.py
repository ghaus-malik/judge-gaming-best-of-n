"""Paired bootstrap tests on the 100 problems both judges scored."""
import sys, os, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from harness.common import *
from harness.bon import bon

gen = load_json(os.path.join(DATA, "generations_n16.json")); ex = load_json(os.path.join(DATA, "exec_n16.json"))
JA = load_json(os.path.join(DATA, "judge_A.json")); JB = load_json(os.path.join(DATA, "judge_B.json"))
vals = {k: [] for k in ["A8", "A16", "B16", "M16", "M8", "dis1", "dis16"]}
for i, (pid, cands) in enumerate(gen.items()):
    keys = [pid + "::" + h(c["code"]) for c in cands]
    if not all(k in JA and k in JB for k in keys):
        continue
    g = np.array([float(bool(ex[k]["bits"]) and min(ex[k]["bits"]) == 1) for k in keys])
    if g.min() == g.max():
        continue
    A = np.array([JA[k]["p_yes"] for k in keys]); B = np.array([JB[k]["p_yes"] for k in keys])
    dis = ((A > 0.5) != (B > 0.5)).astype(float)
    oa = bon(A, dict(gold=g, dis=dis), [1, 8, 16], draws=4000, seed=i)
    ob = bon(B, dict(gold=g), [16], draws=4000, seed=i)
    om = bon((A + B) / 2, dict(gold=g), [8, 16], draws=4000, seed=i)
    vals["A8"].append(oa[8]["gold"]); vals["A16"].append(oa[16]["gold"]); vals["B16"].append(ob[16]["gold"])
    vals["M16"].append(om[16]["gold"]); vals["M8"].append(om[8]["gold"])
    vals["dis1"].append(oa[1]["dis"]); vals["dis16"].append(oa[16]["dis"])
v = {k: np.array(x) for k, x in vals.items()}
rng = np.random.default_rng(0)
def paired(a, b, name):
    d = a - b
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)]
    line = f"{name}: {d.mean():+.3f} (95% CI {np.percentile(bs, 2.5):+.3f} to {np.percentile(bs, 97.5):+.3f}), n={len(d)}"
    print(line); return line
out = [paired(v["M16"], v["A16"], "pass rate at n=16, mean(A,B) selector minus judge A selector"),
       paired(v["M16"], v["B16"], "pass rate at n=16, mean(A,B) selector minus judge B selector"),
       paired(v["A16"], v["A8"], "judge A selector, n=16 minus n=8"),
       paired(v["dis16"], v["dis1"], "A/B disagreement on judge A's pick, n=16 minus n=1")]
open(os.path.join(RESULTS, "paired_tests.md"), "w").write("\n".join("- " + o for o in out) + "\n")
