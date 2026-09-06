"""Vanilla FedAvg baseline — flat-state PPO, standard FedAvg (no CA weighting, no gossip)."""
import numpy as np, torch, json
from pathlib import Path
from .env.network import ISLNetwork, DT, B
from .env.constellation import N_SAT, compute_isl_graph, _sat_positions
from .agents.ppo import PPOAgent
from .fed.fedavg import visible_satellites, fedavg_round, broadcast
from .utils.state import build_state, STATE_DIM
from .utils.reward import compute_reward

K_ROUNDS = 3000; STEPS_PER_ROUND = 10; ACTION_DIM = 4
LOG_EVERY = 100; SAVE_DIR = Path("results/fedavg")


def run():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    net    = ISLNetwork(seed=42)
    agents = [PPOAgent(STATE_DIM, ACTION_DIM, i) for i in range(N_SAT)]
    init_w = agents[0].get_weights()
    for ag in agents: ag.set_weights(init_w)

    x_prev = np.zeros((N_SAT, N_SAT)); kappa_prev = np.zeros((N_SAT, N_SAT))
    metrics = {"round": [], "mean_reward": [], "loss": [], "contact_size": [],
               "mean_kappa": [], "drop_rate": [], "link_utilization": [], "throughput_gbps": []}

    for k in range(K_ROUNDS):
        round_rewards = []; round_kappa = []; round_drop = []
        round_util = []; round_tput = []
        for _ in range(STEPS_PER_ROUND):
            obs = net.step(x_prev)
            kappa=obs["kappa"]; cap=obs["cap"]; avail=obs["avail"]
            rtt=obs["rtt"]; S=obs["S"]; drop=obs["drop"]
            x_new = np.zeros((N_SAT, N_SAT)); step_rewards = []
            for i in range(N_SAT):
                nbrs = [j for j in range(N_SAT) if avail[i, j]]
                if not nbrs: continue
                state = build_state(i, nbrs, kappa, cap, avail, rtt, kappa_prev, drop)
                s_pad = np.zeros(STATE_DIM, dtype=np.float32)
                s_pad[:len(state)] = state[:STATE_DIM]
                action, lp, val = agents[i].select_action(s_pad)
                for idx, j in enumerate(nbrs[:ACTION_DIM]):
                    x_new[i, j] = action[idx] * cap[i, j]
                r = compute_reward(i, nbrs, S, cap, kappa, drop, B, x_new, x_prev, DT)
                agents[i].store(s_pad, action[:ACTION_DIM], r, lp, val, done=0.0)
                step_rewards.append(r)
            mask = avail > 0; cap_sum = cap[mask].sum() + 1e-9
            round_rewards.append(np.mean(step_rewards) if step_rewards else 0)
            round_kappa.append(float(kappa[mask].mean()) if mask.any() else 0.0)
            round_drop.append(float(drop[mask].sum() / (cap_sum * DT)))
            round_util.append(float(x_new[mask].sum() / cap_sum))
            round_tput.append(float(S[mask].sum() / DT / 1e9))
            kappa_prev = kappa.copy(); x_prev = x_new.copy()

        losses = [ag.update() for ag in agents]
        pos = _sat_positions(net.t)
        direct = visible_satellites(pos)
        _, _, avail_g = compute_isl_graph(net.t)
        trunk = fedavg_round(agents, direct)
        if trunk: broadcast(agents, trunk, direct)

        mean_r = float(np.mean(round_rewards)); mean_l = float(np.mean(losses))
        metrics["round"].append(k); metrics["mean_reward"].append(mean_r)
        metrics["loss"].append(mean_l); metrics["contact_size"].append(len(direct))
        metrics["mean_kappa"].append(float(np.mean(round_kappa)))
        metrics["drop_rate"].append(float(np.mean(round_drop)))
        metrics["link_utilization"].append(float(np.mean(round_util)))
        metrics["throughput_gbps"].append(float(np.mean(round_tput)))
        if k % LOG_EVERY == 0:
            print("Round %4d | reward %+.4f | kappa %.3f | drop %.3f | util %.3f | tput %.2fG | contact %d/%d" % (
                k, mean_r, metrics["mean_kappa"][-1], metrics["drop_rate"][-1],
                metrics["link_utilization"][-1], metrics["throughput_gbps"][-1], len(direct), N_SAT))
        if k % 500 == 0:
            with open(SAVE_DIR / "metrics.json", "w") as f: json.dump(metrics, f)

    with open(SAVE_DIR / "metrics.json", "w") as f: json.dump(metrics, f)
    print("FedAvg done. Saved to", SAVE_DIR)


if __name__ == "__main__":
    run()
