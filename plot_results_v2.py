"""
Comprehensive IEEE-paper plots — v2.
Includes: learning curves, steady-state bars, disruption resilience,
          ablation study, Jain's fairness, convergence speed.

Run after all experiments complete:
  python plot_results_v2.py
"""
import json, os, numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── Method registry ────────────────────────────────────────────────────────────
METHODS = {
    "TCP-CUBIC":           "results/tcp_cubic",
    "FedAvg":              "results/fedavg",
    "Ind-PPO":             "results/indppo_isl",
    "CA-PFedAvg":          "results/capfedavg",
    "Flat-FRL (proposed)": "results/flat_frl",
    "GNN-FRL (proposed)":  "results/gnn_frl",
}
COLORS = ["#7f7f7f", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
LABELS = list(METHODS.keys())
SMOOTH = 50
ABLATION_DIR = Path("results/ablation")
DISRUPT_DIR  = Path("results/disruption")


# ── Helpers ────────────────────────────────────────────────────────────────────
def load(path):
    p = Path(path) / "metrics.json"
    return json.load(open(p)) if p.exists() else None


def load_file(path):
    p = Path(path)
    return json.load(open(p)) if p.exists() else None


def smooth(arr, w=SMOOTH):
    arr = np.array(arr, dtype=float)
    return np.convolve(arr, np.ones(w)/w, mode="valid") if len(arr) >= w else arr


def save_fig(fig, name, out_dir="results/figures"):
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(f"{out_dir}/{name}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(f"{out_dir}/{name}.png", bbox_inches="tight", dpi=150)
    print(f"  saved {out_dir}/{name}.pdf/.png")


def jain_fairness(tputs_per_link):
    """Jain's Fairness Index from a list of per-link throughputs."""
    x = np.array(tputs_per_link, dtype=float)
    n = len(x)
    if n == 0: return 0.0
    return float(x.sum()**2 / (n * (x**2).sum() + 1e-12))


# ── Load main data ─────────────────────────────────────────────────────────────
data = {}
for label, path in METHODS.items():
    d = load(path)
    if d:
        data[label] = d
        print(f"[OK] {label:30s} rounds={len(d['round'])}")
    else:
        print(f"[--] {label:30s} (no data)")

print()

# ══════════════════════════════════════════════════════════════════════════════
# Fig 1 — Learning Curves (reward), RL methods only, TCP as dashed reference
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 4))
tcp_reward = float(np.mean(data["TCP-CUBIC"]["mean_reward"])) if "TCP-CUBIC" in data else None
if tcp_reward is not None:
    ax.axhline(tcp_reward, color=COLORS[0], lw=1.4, ls="--",
               label=f"TCP-CUBIC (ref={tcp_reward:.3f})")

rl_labels = [l for l in LABELS if l != "TCP-CUBIC"]
rl_colors = COLORS[1:]
for lbl, clr in zip(rl_labels, rl_colors):
    if lbl not in data: continue
    y = smooth(data[lbl]["mean_reward"])
    ax.plot(np.arange(len(y)), y, color=clr, lw=1.6, label=lbl)

ax.set_xlabel("Training Round", fontsize=12)
ax.set_ylabel("Mean Reward", fontsize=12)
ax.set_title("Learning Curves (RL methods vs TCP-CUBIC reference)", fontsize=13)
ax.legend(fontsize=9, ncol=2); ax.grid(alpha=0.3)
save_fig(fig, "fig1_learning_curves"); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Fig 2 — Throughput over training (ALL methods, RL x-axis = round)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 4))
if "TCP-CUBIC" in data:
    tcp_tput = float(np.mean(data["TCP-CUBIC"]["throughput_gbps"]))
    ax.axhline(tcp_tput, color=COLORS[0], lw=1.4, ls="--",
               label=f"TCP-CUBIC (ref={tcp_tput:.1f} Gbps)")
for lbl, clr in zip(rl_labels, rl_colors):
    if lbl not in data or "throughput_gbps" not in data[lbl]: continue
    y = smooth(data[lbl]["throughput_gbps"])
    ax.plot(np.arange(len(y)), y, color=clr, lw=1.6, label=lbl)
ax.set_xlabel("Training Round", fontsize=12)
ax.set_ylabel("Aggregate Throughput (Gbps)", fontsize=12)
ax.set_title("Network Throughput During Training", fontsize=13)
ax.legend(fontsize=9, ncol=2); ax.grid(alpha=0.3)
save_fig(fig, "fig2_throughput"); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Fig 3 — Drop Rate over training
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 4))
if "TCP-CUBIC" in data:
    ax.axhline(float(np.mean(data["TCP-CUBIC"]["drop_rate"])), color=COLORS[0],
               lw=1.4, ls="--", label="TCP-CUBIC (ref)")
for lbl, clr in zip(rl_labels, rl_colors):
    if lbl not in data or "drop_rate" not in data[lbl]: continue
    y = smooth(data[lbl]["drop_rate"])
    ax.plot(np.arange(len(y)), y, color=clr, lw=1.6, label=lbl)
ax.set_xlabel("Training Round", fontsize=12)
ax.set_ylabel("Packet Drop Rate", fontsize=12)
ax.set_title("Packet Drop Rate During Training", fontsize=13)
ax.legend(fontsize=9, ncol=2); ax.grid(alpha=0.3)
save_fig(fig, "fig3_drop_rate"); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Fig 4 — Queue Occupancy (kappa)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 4))
if "TCP-CUBIC" in data:
    ax.axhline(float(np.mean(data["TCP-CUBIC"]["mean_kappa"])), color=COLORS[0],
               lw=1.4, ls="--", label="TCP-CUBIC (ref)")
for lbl, clr in zip(rl_labels, rl_colors):
    if lbl not in data or "mean_kappa" not in data[lbl]: continue
    y = smooth(data[lbl]["mean_kappa"])
    ax.plot(np.arange(len(y)), y, color=clr, lw=1.6, label=lbl)
ax.set_xlabel("Training Round", fontsize=12)
ax.set_ylabel("Mean Queue Occupancy κ", fontsize=12)
ax.set_title("Congestion Level (κ) During Training", fontsize=13)
ax.legend(fontsize=9, ncol=2); ax.grid(alpha=0.3)
save_fig(fig, "fig4_kappa"); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Fig 5 — Link Utilization
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 4))
if "TCP-CUBIC" in data:
    ax.axhline(float(np.mean(data["TCP-CUBIC"]["link_utilization"])), color=COLORS[0],
               lw=1.4, ls="--", label="TCP-CUBIC (ref)")
for lbl, clr in zip(rl_labels, rl_colors):
    if lbl not in data or "link_utilization" not in data[lbl]: continue
    y = smooth(data[lbl]["link_utilization"])
    ax.plot(np.arange(len(y)), y, color=clr, lw=1.6, label=lbl)
ax.set_xlabel("Training Round", fontsize=12)
ax.set_ylabel("Link Utilization", fontsize=12)
ax.set_title("ISL Link Utilization During Training", fontsize=13)
ax.legend(fontsize=9, ncol=2); ax.grid(alpha=0.3)
save_fig(fig, "fig5_utilization"); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Fig 6 — Steady-state bar chart (ALL 6 methods, last 200 rounds)
# ══════════════════════════════════════════════════════════════════════════════
WIN = 200
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for metric, ylabel, ax in [
    ("throughput_gbps", "Throughput (Gbps)", axes[0]),
    ("drop_rate",       "Packet Drop Rate",  axes[1]),
    ("link_utilization","Link Utilization",  axes[2]),
]:
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
fig.suptitle(f"Steady-State Performance (last {WIN} rounds)", fontsize=12)
fig.tight_layout()
save_fig(fig, "fig6_bar_comparison"); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Fig 7 — Convergence Speed
# ══════════════════════════════════════════════════════════════════════════════
rl_with_data = [l for l in rl_labels if l in data]
if rl_with_data:
    best_final = max(np.mean(data[l]["mean_reward"][-200:]) for l in rl_with_data)
    THRESH = 0.90 * best_final
    print(f"Convergence threshold (90% of best): {THRESH:.4f}")
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (lbl, clr) in enumerate(zip(LABELS, COLORS)):
        if lbl not in data: continue
        r = np.array(data[lbl]["mean_reward"])
        crossed = np.where(r >= THRESH)[0]
        conv_round = int(crossed[0]) if len(crossed) > 0 else len(r)
        ax.barh(i, conv_round, color=clr, alpha=0.8)
        ax.text(conv_round + 20, i, str(conv_round), va="center", fontsize=9)
    ax.set_yticks(range(len(LABELS))); ax.set_yticklabels(LABELS, fontsize=10)
    ax.set_xlabel("Round to reach 90% of best reward", fontsize=11)
    ax.set_title("Convergence Speed", fontsize=12)
    ax.axvline(3000, ls="--", color="k", alpha=0.3)
    ax.grid(axis="x", alpha=0.3)
    save_fig(fig, "fig7_convergence"); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Fig 8 — Topology Disruption Resilience
# ══════════════════════════════════════════════════════════════════════════════
DISRUPT_METHODS = {
    "TCP-CUBIC":           DISRUPT_DIR / "tcp_metrics.json",
    "Ind-PPO":             DISRUPT_DIR / "indppo_metrics.json",
    "Flat-FRL (proposed)": DISRUPT_DIR / "flat_frl_metrics.json",
    "GNN-FRL (proposed)":  DISRUPT_DIR / "gnn_frl_metrics.json",
}
DISRUPT_COLORS = ["#7f7f7f", "#ff7f0e", "#d62728", "#9467bd"]

dm = {}
for lbl, p in DISRUPT_METHODS.items():
    d = load_file(p)
    if d: dm[lbl] = d; print(f"[disrupt OK] {lbl}")
    else: print(f"[disrupt --] {lbl} (no data)")

if dm:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for metric, ylabel, ax in [
        ("throughput_gbps", "Aggregate Throughput (Gbps)", axes[0]),
        ("drop_rate",       "Packet Drop Rate",            axes[1]),
    ]:
        for lbl, clr in zip(DISRUPT_METHODS.keys(), DISRUPT_COLORS):
            if lbl not in dm: continue
            d   = dm[lbl]
            rnd = np.array(d["round"])
            y   = smooth(d[metric], w=20)
            x   = rnd[:len(y)]
            ax.plot(x, y, color=clr, lw=1.5, label=lbl)
        # shade disruption windows
        DISRUPT_EVERY = 500; DISRUPT_LEN = 50
        for start in range(0, 3000, DISRUPT_EVERY):
            ax.axvspan(start, start + DISRUPT_LEN, alpha=0.12, color="red")
        ax.set_xlabel("Training Round", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    axes[0].set_title("Throughput Under ISL Disruptions (red = outage)", fontsize=11)
    axes[1].set_title("Drop Rate Under ISL Disruptions", fontsize=11)
    patch = mpatches.Patch(color="red", alpha=0.3, label="20% ISL failure")
    fig.legend(handles=[patch], loc="upper center", fontsize=9)
    fig.tight_layout()
    save_fig(fig, "fig8_disruption_resilience"); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Fig 9 — Ablation Study (bar chart, last 200 rounds)
# ══════════════════════════════════════════════════════════════════════════════
ABL_LABELS = {
    "full":      "GNN-FRL\n(Full)",
    "no_gossip": "No ISL-\nGossip",
    "no_relay":  "No Relay\nExpansion",
    "flat":      "Flat State\n(No GNN)",
    "ind":       "Ind-PPO\n(No Fed)",
}
ABL_COLORS = ["#9467bd", "#17becf", "#bcbd22", "#d62728", "#ff7f0e"]

abm = {}
for name in ABL_LABELS:
    d = load_file(ABLATION_DIR / f"{name}_metrics.json")
    if d: abm[name] = d; print(f"[ablation OK] {name}")
    else: print(f"[ablation --] {name} (no data)")

if abm:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for metric, ylabel, ax in [
        ("throughput_gbps",  "Throughput (Gbps)", axes[0]),
        ("drop_rate",        "Drop Rate",         axes[1]),
        ("link_utilization", "Link Utilization",  axes[2]),
    ]:
        vals, errs, lbls, clrs = [], [], [], []
        for name, clr in zip(ABL_LABELS.keys(), ABL_COLORS):
            if name not in abm: continue
            arr  = np.array(abm[name][metric])
            tail = arr[-WIN:] if len(arr) >= WIN else arr
            vals.append(tail.mean()); errs.append(tail.std())
            lbls.append(ABL_LABELS[name]); clrs.append(clr)
        x = np.arange(len(vals))
        ax.bar(x, vals, yerr=errs, capsize=4, color=clrs, alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(lbls, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10); ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Ablation Study — Component Contribution (last 200 rounds)", fontsize=12)
    fig.tight_layout()
    save_fig(fig, "fig9_ablation"); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Fig 10 — Jain's Fairness Index (from throughput_gbps proxy)
# Uses per-round throughput as a single-link proxy; real fairness needs
# per-link data, but this illustrates trend.
# ══════════════════════════════════════════════════════════════════════════════
FAIRNESS_ROUNDS = 200
fig, ax = plt.subplots(figsize=(9, 4))
has_fairness = False
for lbl, clr in zip(LABELS, COLORS):
    if lbl not in data or "throughput_gbps" not in data[lbl]: continue
    tputs = np.array(data[lbl]["throughput_gbps"])
    # rolling Jain's fairness over windows
    fairness = []
    for i in range(0, len(tputs) - FAIRNESS_ROUNDS, FAIRNESS_ROUNDS // 10):
        window = tputs[i:i + FAIRNESS_ROUNDS]
        fairness.append(jain_fairness(window))
    if fairness:
        x = np.linspace(0, len(tputs), len(fairness))
        ax.plot(x, fairness, color=clr, lw=1.5, label=lbl)
        has_fairness = True

if has_fairness:
    ax.set_xlabel("Training Round (approx)", fontsize=12)
    ax.set_ylabel("Jain's Fairness Index", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title("Throughput Fairness Index (Jain's, J=1 is perfectly fair)", fontsize=12)
    ax.legend(fontsize=9, ncol=2); ax.grid(alpha=0.3)
    save_fig(fig, "fig10_fairness"); plt.close(fig)
else:
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Table 1 — Steady-state summary table (print + save as CSV)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*90)
print(f"{'Method':<30} {'Throughput(Gbps)':>16} {'Drop Rate':>10} {'Kappa':>8} {'Util':>7}")
print("="*90)
rows = []
for lbl in LABELS:
    if lbl not in data: continue
    d = data[lbl]
    def tail_mean(key):
        arr = np.array(d.get(key, [0]))
        return float(arr[-WIN:].mean()) if len(arr) >= WIN else float(arr.mean())
    tput = tail_mean("throughput_gbps")
    drop = tail_mean("drop_rate")
    kap  = tail_mean("mean_kappa")
    util = tail_mean("link_utilization")
    print(f"{lbl:<30} {tput:>16.2f} {drop:>10.4f} {kap:>8.4f} {util:>7.4f}")
    rows.append([lbl, tput, drop, kap, util])

os.makedirs("results/figures", exist_ok=True)
with open("results/figures/table1_summary.csv", "w") as f:
    f.write("Method,Throughput_Gbps,Drop_Rate,Mean_Kappa,Link_Utilization\n")
    for r in rows:
        f.write(",".join(str(v) for v in r) + "\n")
print("  saved results/figures/table1_summary.csv")
print("="*90)

print("\nAll figures saved to results/figures/")
