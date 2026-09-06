"""TCP-CUBIC baseline — static additive-increase / multiplicative-decrease sending rate."""
import numpy as np, json
from pathlib import Path
from ..env.network import ISLNetwork, DT, B
from ..env.constellation import N_SAT

K_ROUNDS        = 3000
STEPS_PER_ROUND = 10
LOG_EVERY       = 100
SAVE_DIR        = Path("results/tcp_cubic")

# CUBIC parameters
C_CUBIC = 0.4
BETA_CUBIC = 0.7
W_MAX_INIT = 1.0   # fraction of capacity


def run():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    net   = ISLNetwork(seed=42)
    x_prev = np.zeros((N_SAT, N_SAT))
    w      = np.ones((N_SAT, N_SAT)) * W_MAX_INIT
    w_max  = w.copy()
    t_loss = np.zeros((N_SAT, N_SAT))
    metrics = {"round": [], "mean_reward": [], "drop_rate": [],
               "mean_kappa": [], "link_utilization": [], "throughput_gbps": []}

    for k in range(K_ROUNDS):
        round_reward = []; round_kappa = []; round_drop = []
        round_util = []; round_tput = []
        for step in range(STEPS_PER_ROUND):
            obs   = net.step(x_prev)
            kappa = obs["kappa"]; cap = obs["cap"]
            avail = obs["avail"]; drop = obs["drop"]; S = obs["S"]

            loss_event = drop > 0
            w_max  = np.where(loss_event, w, w_max)
            w      = np.where(loss_event, w * BETA_CUBIC, w)
            t_loss = np.where(loss_event, 0.0, t_loss + DT)
            K_val  = (w_max * (1 - BETA_CUBIC) / C_CUBIC) ** (1/3)
            w_cubic = C_CUBIC * (t_loss - K_val)**3 + w_max
            w = np.clip(w_cubic, 0.01, 1.0)

            x_new = w * cap * avail
            mask   = avail > 0; cap_sum = cap[mask].sum() + 1e-9
            round_reward.append(float(-drop[mask].sum() / (N_SAT * 1e9)))
            round_kappa.append(float(kappa[mask].mean()) if mask.any() else 0.0)
            round_drop.append(float(drop[mask].sum() / (cap_sum * DT)))
            round_util.append(float(x_new[mask].sum() / cap_sum))
            round_tput.append(float(S[mask].sum() / DT / 1e9))
            x_prev = x_new.copy()

        metrics["round"].append(k)
        metrics["mean_reward"].append(float(np.mean(round_reward)))
        metrics["drop_rate"].append(float(np.mean(round_drop)))
        metrics["mean_kappa"].append(float(np.mean(round_kappa)))
        metrics["link_utilization"].append(float(np.mean(round_util)))
        metrics["throughput_gbps"].append(float(np.mean(round_tput)))

        if k % LOG_EVERY == 0:
            print("Round %4d | reward %+.4f | kappa %.3f | drop %.3f | util %.3f | tput %.2fG" % (
                k, metrics["mean_reward"][-1], metrics["mean_kappa"][-1],
                metrics["drop_rate"][-1], metrics["link_utilization"][-1], metrics["throughput_gbps"][-1]))
        if k % 500 == 0:
            with open(SAVE_DIR / "metrics.json", "w") as f:
                json.dump(metrics, f)

    with open(SAVE_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f)
    print("TCP-CUBIC done. Saved to", SAVE_DIR)


if __name__ == "__main__":
    run()
