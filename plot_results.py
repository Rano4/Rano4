"""
Comprehensive plotting for IEEE paper — 6 methods, 6 metrics.
Run after all experiments complete.
"""
import json, os, numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

METHODS = {
    "TCP-CUBIC":          "results/tcp_cubic",
    "FedAvg":             "results/fedavg",
    "Ind-PPO":            "results/indppo_isl",
    "CA-PFedAvg":         "results/capfedavg",
    "Flat-FRL (proposed)": "results/flat_frl",
    "GNN-FRL (proposed)": "results/gnn_frl",
}
COLORS = ["#7f7f7f", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
LABELS = list(METHODS.keys())
SMOOTH = 50


def load(path):
    p = Path(path) / "metrics.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def smooth(arr, w=SMOOTH):
    if len(arr) < w:
        return np.array(arr)
    return np.convolve(arr, np.ones(w)/w, mode="valid")


def save_fig(fig, name, out_dir="results/figures"):
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(f"{out_dir}/{name}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(f"{out_dir}/{name}.png", bbox_inches="tight", dpi=150)
    print(f"  saved {out_dir}/{name}.pdf/.png")


data = {}
for label, path in METHODS.items():
    d = load(path)
    if d:
        data[label] = d
        print(f"[OK] {label:30s} rounds={len(d['round'])}")
    else:
        print(f"[--] {label:30s} (no data)")


# Fig 1: Learning curves
fig, ax = plt.subplots(figsize=(8, 4))
for lbl, clr in zip(LABELS, COLORS):
    if lbl not in data: continue
    y = smooth(data[lbl]["mean_reward"])
    ax.plot(np.arange(len(y)), y, color=clr, lw=1.6, label=lbl)
ax.set_xlabel("Training Round", fontsize=12)
ax.set_ylabel("Mean Reward", fontsize=12)
ax.set_title("Learning Curves", fontsize=13)
ax.legend(fontsize=9, ncol=2); ax.grid(alpha=0.3)
save_fig(fig, "fig1_learning_curves"); plt.close(fig)


# Fig 2: Throughput
fig, ax = plt.subplots(figsize=(8, 4))
for lbl, clr in zip(LABELS, COLORS):
    if lbl not in data or "throughput_gbps" not in data[lbl]: continue
    y = smooth(data[lbl]["throughput_gbps"])
    ax.plot(np.arange(len(y)), y, color=clr, lw=1.6, label=lbl)
ax.set_xlabel("Training Round", fontsize=12)
ax.set_ylabel("Aggregate Throughput (Gbps)", fontsize=12)
ax.set_title("Network Throughput", fontsize=13)
ax.legend(fontsize=9, ncol=2); ax.grid(alpha=0.3)
save_fig(fig, "fig2_throughput"); plt.close(fig)


# Fig 3: Drop rate
fig, ax = plt.subplots(figsize=(8, 4))
for lbl, clr in zip(LABELS, COLORS):
    if lbl not in data or "drop_rate" not in data[lbl]: continue
    y = smooth(data[lbl]["drop_rate"])
    ax.plot(np.arange(len(y)), y, color=clr, lw=1.6, label=lbl)
ax.set_xlabel("Training Round", fontsize=12)
ax.set_ylabel("Packet Drop Rate", fontsize=12)
ax.set_title("Packet Drop Rate", fontsize=13)
ax.legend(fontsize=9, ncol=2); ax.grid(alpha=0.3)
save_fig(fig, "fig3_drop_rate"); plt.close(fig)


# Fig 4: Queue occupancy
fig, ax = plt.subplots(figsize=(8, 4))
for lbl, clr in zip(LABELS, COLORS):
    if lbl not in data or "mean_kappa" not in data[lbl]: continue
    y = smooth(data[lbl]["mean_kappa"])
    ax.plot(np.arange(len(y)), y, color=clr, lw=1.6, label=lbl)
ax.set_xlabel("Training Round", fontsize=12)
ax.set_ylabel("Mean Queue Occupancy κ", fontsize=12)
ax.set_title("Congestion Level (κ)", fontsize=13)
ax.legend(fontsize=9, ncol=2); ax.grid(alpha=0.3)
save_fig(fig, "fig4_kappa"); plt.close(fig)


# Fig 5: Link utilization
fig, ax = plt.subplots(figsize=(8, 4))
for lbl, clr in zip(LABELS, COLORS):
    if lbl not in data or "link_utilization" not in data[lbl]: continue
    y = smooth(data[lbl]["link_utilization"])
    ax.plot(np.arange(len(y)), y, color=clr, lw=1.6, label=lbl)
ax.set_xlabel("Training Round", fontsize=12)
ax.set_ylabel("Link Utilization", fontsize=12)
ax.set_title("ISL Link Utilization", fontsize=13)
ax.legend(fontsize=9, ncol=2); ax.grid(alpha=0.3)
save_fig(fig, "fig5_utilization"); plt.close(fig)


# Fig 6: Bar comparison
WIN = 200
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for (metric, ylabel, ax) in [("throughput_gbps", "Throughput (Gbps)", axes[0]),
                              ("drop_rate", "Packet Drop Rate", axes[1]),
                              ("link_utilization", "Link Utilization", axes[2])]:
    vals, errs, lbls, clrs = [], [], [], []
    for lbl, clr in zip(LABELS, COLORS):
        if lbl not in data or metric not in data[lbl]: continue
        arr = np.array(data[lbl][metric])
        tail = arr[-WIN:] if len(arr) >= WIN else arr
        vals.append(tail.mean()); errs.append(tail.std())
        lbls.append(lbl); clrs.append(clr)
    x = np.arange(len(vals))
    ax.bar(x, vals, yerr=errs, capsize=4, color=clrs, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(lbls, fontsize=8, rotation=30, ha="right")
    ax.set_ylabel(ylabel, fontsize=10); ax.grid(axis="y", alpha=0.3)
fig.suptitle("Steady-State Performance (last %d rounds)" % WIN, fontsize=12)
fig.tight_layout()
save_fig(fig, "fig6_bar_comparison"); plt.close(fig)


# Fig 7: Convergence speed
rl_methods = [l for l in LABELS if l != "TCP-CUBIC"]
best_final = max((np.mean(data[l]["mean_reward"][-200:]) for l in rl_methods if l in data), default=0.18)
THRESH = 0.90 * best_final
print("Convergence threshold (90%% of best): %.4f" % THRESH)
fig, ax = plt.subplots(figsize=(7, 4))
for i, (lbl, clr) in enumerate(zip(LABELS, COLORS)):
    if lbl not in data: continue
    r = np.array(data[lbl]["mean_reward"])
    crossed = np.where(r >= THRESH)[0]
    conv_round = int(crossed[0]) if len(crossed) > 0 else len(r)
    ax.barh(i, conv_round, color=clr, alpha=0.8)
    ax.text(conv_round + 10, i, str(conv_round), va="center", fontsize=9)
ax.set_yticks(range(len(LABELS)))
ax.set_yticklabels(LABELS, fontsize=10)
ax.set_xlabel("Round to reach 90%% of best reward", fontsize=11)
ax.set_title("Convergence Speed", fontsize=12)
ax.axvline(3000, ls="--", color="k", alpha=0.3)
ax.grid(axis="x", alpha=0.3)
save_fig(fig, "fig7_convergence"); plt.close(fig)

print("\nAll figures saved to results/figures/")
