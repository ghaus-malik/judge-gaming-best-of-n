"""Best-of-n against a local LLM judge, with hidden unit tests as ground truth."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness.common import *
from harness.bon import bon, auroc, boot_ci
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
gen = load_json(os.path.join(DATA, "generations_n16.json"))
ex = load_json(os.path.join(DATA, "exec_n16.json"))
JA = load_json(os.path.join(DATA, "judge_A.json"))
JB = load_json(os.path.join(DATA, "judge_B.json"))


def gold(r):
    return int(bool(r["bits"]) and min(r["bits"]) == 1)


def frac(r):
    return float(np.mean(r["bits"])) if r["bits"] else 0.0


rows = {}
n_allpass = n_allfail = 0
for pid, cands in gen.items():
    keys = [pid + "::" + h(c["code"]) for c in cands]
    g = np.array([gold(ex[k]) for k in keys])
    if g.sum() == len(g):
        n_allpass += 1
        continue
    if g.sum() == 0:
        n_allfail += 1
        continue
    if not all(k in JA and k in JB for k in keys):
        continue
    vis = [ex[k]["vis"] for k in keys]
    rows[pid] = dict(
        gold=g, frac=np.array([frac(ex[k]) for k in keys]),
        vis=np.array([float(min(v)) if v else np.nan for v in vis]),
        A=np.array([JA[k]["p_yes"] for k in keys]), B=np.array([JB[k]["p_yes"] for k in keys]),
        uniq=len(set(keys)))
P = len(gen)
print(f"problems {P}: all-pass {n_allpass}, all-fail {n_allfail}, mixed and judged {len(rows)}")

NS = [1, 2, 4, 8, 16]
selectors = {
    "random": lambda r: np.zeros(16),
    "judge A (Qwen2.5-7B)": lambda r: r["A"],
    "judge B (Llama-3.1-8B)": lambda r: r["B"],
    "mean of A and B": lambda r: (r["A"] + r["B"]) / 2,
    "oracle (hidden tests)": lambda r: r["gold"] + 0.001 * r["frac"],
}
mbpp = [p for p in rows if p.startswith("MBPP")]
res = {s: {n: [] for n in NS} for s in selectors}
per_problem = {s: {n: {} for n in NS} for s in selectors}
for i, (pid, r) in enumerate(rows.items()):
    metrics = dict(gold=r["gold"], frac=r["frac"], A=r["A"], B=r["B"],
                   disagree=((r["A"] > 0.5) != (r["B"] > 0.5)).astype(float),
                   A_conf_wrong=((r["A"] > 0.9) & (r["gold"] == 0)).astype(float))
    for s, f in selectors.items():
        out = bon(f(r), metrics, NS, draws=2000, seed=i)
        for n in NS:
            res[s][n].append({m: v for m, v in out[n].items() if m != "_picks"})
            per_problem[s][n][pid] = out[n]["gold"]

summary = {"n_problems": P, "all_pass": n_allpass, "all_fail": n_allfail, "mixed": len(rows), "selectors": {}}
lines = ["| selector | " + " | ".join(f"n={n}" for n in NS) + " |", "|---" * (len(NS) + 1) + "|"]
for s in selectors:
    summary["selectors"][s] = {}
    cells = []
    for n in NS:
        agg = {m: float(np.mean([d[m] for d in res[s][n]])) for m in res[s][n][0]}
        vals = [per_problem[s][n][p] for p in rows]
        lo, hi = boot_ci(vals)
        agg["gold_ci"] = (lo, hi)
        agg["overall_pass"] = (sum(vals) + n_allpass) / P
        summary["selectors"][s][n] = agg
        cells.append(f"{agg['gold']:.3f}")
    lines.append(f"| {s} | " + " | ".join(cells) + " |")
table_gold = "\n".join(lines)
print("Gold pass rate of the selected sample, mixed problems:\n" + table_gold)

# Proxy vs gold under judge A pressure
lines = ["| n | judge A score of pick | hidden-test pass of pick | gap | judge B score of pick | A/B disagree | A>0.9 but wrong |",
         "|---|---|---|---|---|---|---|"]
for n in NS:
    a = summary["selectors"]["judge A (Qwen2.5-7B)"][n]
    lines.append(f"| {n} | {a['A']:.3f} | {a['gold']:.3f} | {a['A']-a['gold']:+.3f} | {a['B']:.3f} | {a['disagree']:.3f} | {a['A_conf_wrong']:.3f} |")
table_pressure = "\n".join(lines)
print(table_pressure)

# Early warning: among judge A's single top pick per problem (n = 16), does judge B separate right from wrong?
picks = []
for pid, r in rows.items():
    k = int(np.argmax(r["A"] + 1e-9 * np.arange(16)))
    picks.append((r["gold"][k], r["A"][k], r["B"][k]))
picks = np.array(picks)
ew = dict(
    n=len(picks), wrong=int((picks[:, 0] == 0).sum()),
    auroc_A=auroc(picks[:, 1], picks[:, 0]), auroc_B=auroc(picks[:, 2], picks[:, 0]),
    auroc_min=auroc(np.minimum(picks[:, 1], picks[:, 2]), picks[:, 0]),
)
# problems where pressure on A hurt: gold at n=16 below random (n=1)
hurt = np.array([per_problem["judge A (Qwen2.5-7B)"][16][p] < per_problem["random"][1][p] - 1e-9 for p in rows])
dis16 = np.array([d["disagree"] for d in res["judge A (Qwen2.5-7B)"][16]])
gapAB16 = np.array([d["A"] - d["B"] for d in res["judge A (Qwen2.5-7B)"][16]])
ew["problems_where_A_selection_hurt"] = int(hurt.sum())
ew["auroc_disagree_flags_hurt"] = auroc(dis16, hurt.astype(int))
ew["auroc_AminusB_flags_hurt"] = auroc(gapAB16, hurt.astype(int))
print("early warning:", json.dumps(ew, indent=1))
summary["early_warning"] = ew

# Self-preference check: judge A is the same family as the policy (Qwen); compare mean scores on wrong samples
allA = np.concatenate([r["A"] for r in rows.values()])
allB = np.concatenate([r["B"] for r in rows.values()])
allG = np.concatenate([r["gold"] for r in rows.values()])
summary["sample_level"] = dict(
    n=int(len(allG)), base_rate=float(allG.mean()),
    auroc_A=auroc(allA, allG), auroc_B=auroc(allB, allG), auroc_mean=auroc((allA + allB) / 2, allG),
    accept_rate_A=float((allA > 0.5).mean()), accept_rate_B=float((allB > 0.5).mean()),
    false_accept_A=float(((allA > 0.5) & (allG == 0)).sum() / max(1, (allG == 0).sum())),
    false_accept_B=float(((allB > 0.5) & (allG == 0)).sum() / max(1, (allG == 0).sum())),
    false_reject_A=float(((allA <= 0.5) & (allG == 1)).sum() / max(1, (allG == 1).sum())),
    false_reject_B=float(((allB <= 0.5) & (allG == 1)).sum() / max(1, (allG == 1).sum())),
)
print("sample level:", json.dumps(summary["sample_level"], indent=1))

with open(os.path.join(RESULTS, "results.json"), "w") as f:
    json.dump(summary, f, indent=1, default=float)
with open(os.path.join(RESULTS, "tables.md"), "w") as f:
    f.write("## Hidden-test pass rate of the selected sample (mixed problems)\n\n" + table_gold +
            "\n\n## Judge A under growing selection pressure\n\n" + table_pressure + "\n")

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for s in selectors:
    ys = [summary["selectors"][s][n]["gold"] for n in NS]
    lo = [summary["selectors"][s][n]["gold_ci"][0] for n in NS]
    hi = [summary["selectors"][s][n]["gold_ci"][1] for n in NS]
    ax[0].plot(NS, ys, marker="o", label=s)
    ax[0].fill_between(NS, lo, hi, alpha=0.15)
ax[0].set_xscale("log", base=2); ax[0].set_xlabel("n (best-of-n)"); ax[0].set_ylabel("hidden-test pass rate of pick")
ax[0].set_title(f"{len(rows)} problems where samples disagree"); ax[0].legend(fontsize=7)
a = summary["selectors"]["judge A (Qwen2.5-7B)"]
ax[1].plot(NS, [a[n]["A"] for n in NS], marker="o", label="judge A score (optimized)")
ax[1].plot(NS, [a[n]["B"] for n in NS], marker="o", label="judge B score (held out)")
ax[1].plot(NS, [a[n]["gold"] for n in NS], marker="o", label="hidden-test pass rate")
ax[1].set_xscale("log", base=2); ax[1].set_xlabel("n (best-of-n against judge A)"); ax[1].legend(fontsize=7)
ax[1].set_title("proxy vs held-out judge vs ground truth")
plt.tight_layout(); plt.savefig(os.path.join(RESULTS, "bon_curves.png"), dpi=150)
print("saved")
