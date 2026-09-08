"""
Ablation Study — removes one component at a time from GNN-FRL.
Variants:
  full      — GNN-FRL full (CA-PFedAvg + ISL-Gossip + GNN)
  no_gossip — GNN-FRL without ISL-Gossip (only CA-PFedAvg)
  no_relay  — without relay expansion (only direct visible sats)
  flat      — flat-state PPO + CA-PFedAvg + ISL-Gossip (no GNN)
  ind       — no federation at all (Ind-PPO)

Usage: python -m sim.train_ablation
"""
import numpy as np, json
from pathlib import Path
from .env.network import ISLNetwork, DT, B
from .env.constellation import N_SAT, compute_isl_graph, _sat_positions
from .agents.ppo import PPOAgent
from .agents.gnn_ppo import GNNPPOAgent
from .fed.fedavg import visible_satellites, relay_contact_set, fedavg_round, broadcast
from .fed.gossip import isl_gossip_round
from .utils.state import build_state, STATE_DIM
from .utils.reward import compute_reward

K_ROUNDS        = 3000
STEPS_PER_ROUND = 10
ACTION_DIM      = 4
LOG_EVERY       = 100
SAVE_DIR        = Path("results/ablation")
SEED            = 42

VARIANTS = ["full", "no_gossip", "no_relay", "flat", "ind"]


def _blank():
    return {"round": [], "mean_reward": [], "loss": [],
            "mean_kappa": [], "drop_rate": [], "link_utilization": [], "throughput_gbps": []}


def _gnn_step(agents, net, x_prev, kappa_prev):
    obs   = net.step(x_prev)
    kappa = obs["kappa"]; cap  = obs["cap"]
    avail = obs["avail"]; rtt  = obs["rtt"]; S = obs["S"]; drop = obs["drop"]
    x_new = np.zeros((N_SAT, N_SAT)); step_rewards = []
    C_MAX = 10e9; RTT_MAX = 0.1
    for i in range(N_SAT):
        nbrs = [j for j in range(N_SAT) if avail[i, j]]
        if not nbrs: continue
        nbr_feats = np.array([[float(kappa[i,j]), float(cap[i,j]/C_MAX),
                               float(min(rtt[i,j]/RTT_MAX,1.0)),
                               float(np.clip(kappa[i,j]-kappa_prev[i,j],-1,1))]
                              for j in nbrs], dtype=np.float32)
        sf = nbr_feats.mean(axis=0)
        action, lp, val = agents[i].select_action(sf, nbr_feats)
        for idx, j in enumerate(nbrs[:ACTION_DIM]):
            x_new[i, j] = action[idx] * cap[i, j]
        r = compute_reward(i, nbrs, S, cap, kappa, drop, B, x_new, x_prev, DT)
        agents[i].store(sf, nbr_feats, action[:ACTION_DIM], r, lp, val, done=0.0)
        step_rewards.append(r)
    mask = avail > 0; cap_sum = cap[mask].sum() + 1e-9
    return x_new, kappa.copy(), dict(
        reward=np.mean(step_rewards) if step_rewards else 0,
        kappa=float(kappa[mask].mean()) if mask.any() else 0.0,
        drop=float(drop[mask].sum() / (cap_sum * DT)),
        util=float(x_new[mask].sum() / cap_sum),
        tput=float(S[mask].sum() / DT / 1e9),
    )


def _flat_step(agents, net, x_prev, kappa_prev):
    obs   = net.step(x_prev)
    kappa = obs["kappa"]; cap  = obs["cap"]
    avail = obs["avail"]; rtt  = obs["rtt"]; S = obs["S"]; drop = obs["drop"]
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
    return x_new, kappa.copy(), dict(
        reward=np.mean(step_rewards) if step_rewards else 0,
        kappa=float(kappa[mask].mean()) if mask.any() else 0.0,
        drop=float(drop[mask].sum() / (cap_sum * DT)),
        util=float(x_new[mask].sum() / cap_sum),
        tput=float(S[mask].sum() / DT / 1e9),
    )


def run():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    net = ISLNetwork(seed=SEED)

    # initialise all variant agents
    def _make_gnn(): return [GNNPPOAgent(ACTION_DIM, i) for i in range(N_SAT)]
    def _make_flat(): return [PPOAgent(STATE_DIM, ACTION_DIM, i) for i in range(N_SAT)]
    def _sync(ags):
        w = ags[0].get_weights()
        for ag in ags: ag.set_weights(w)

    v = {
        "full":      _make_gnn(),
        "no_gossip": _make_gnn(),
        "no_relay":  _make_gnn(),
        "flat":      _make_flat(),
        "ind":       _make_flat(),
    }
    for ags in v.values(): _sync(ags)

    x_prev = {name: np.zeros((N_SAT, N_SAT)) for name in VARIANTS}
    k_prev = {name: np.zeros((N_SAT, N_SAT)) for name in VARIANTS}
    metrics = {name: _blank() for name in VARIANTS}

    for k in range(K_ROUNDS):
        _, _, avail_g = compute_isl_graph(net.t)
        pos    = _sat_positions(net.t)
        direct = visible_satellites(pos)

        for name in VARIANTS:
            is_gnn  = name in ("full", "no_gossip", "no_relay")
            step_fn = _gnn_step if is_gnn else _flat_step
            r=[]; kp=[]; dp=[]; ul=[]; tp=[]

            for _ in range(STEPS_PER_ROUND):
                x_new, k_new, row = step_fn(v[name], net, x_prev[name], k_prev[name])
                x_prev[name] = x_new; k_prev[name] = k_new
                r.append(row["reward"]); kp.append(row["kappa"])
                dp.append(row["drop"]);  ul.append(row["util"]); tp.append(row["tput"])

            losses = [ag.update() for ag in v[name]]

            # federation step (variant-specific)
            if name == "full":
                isl_gossip_round(v[name], avail_g)
                contact = relay_contact_set(direct, avail_g)
                trunk   = fedavg_round(v[name], contact)
                if trunk: broadcast(v[name], trunk, contact)

            elif name == "no_gossip":
                # CA-PFedAvg only, no gossip
                contact = relay_contact_set(direct, avail_g)
                trunk   = fedavg_round(v[name], contact)
                if trunk: broadcast(v[name], trunk, contact)

            elif name == "no_relay":
                # gossip + direct-only fedavg (no relay expansion)
                isl_gossip_round(v[name], avail_g)
                contact = set(direct)           # direct only, no relay_contact_set
                trunk   = fedavg_round(v[name], contact)
                if trunk: broadcast(v[name], trunk, contact)

            elif name == "flat":
                # flat PPO + full federation
                isl_gossip_round(v[name], avail_g)
                contact = relay_contact_set(direct, avail_g)
                trunk   = fedavg_round(v[name], contact)
                if trunk: broadcast(v[name], trunk, contact)

            # "ind" — no federation

            m = metrics[name]
            m["round"].append(k)
            m["mean_reward"].append(float(np.mean(r)))
            m["loss"].append(float(np.mean(losses)))
            m["mean_kappa"].append(float(np.mean(kp)))
            m["drop_rate"].append(float(np.mean(dp)))
            m["link_utilization"].append(float(np.mean(ul)))
            m["throughput_gbps"].append(float(np.mean(tp)))

        if k % LOG_EVERY == 0:
            row = {n: metrics[n]["throughput_gbps"][-1] for n in VARIANTS}
            print(f"Round {k:4d} | " +
                  " | ".join(f"{n}={row[n]:.1f}G" for n in VARIANTS))

        if k % 500 == 0:
            for name in VARIANTS:
                with open(SAVE_DIR / f"{name}_metrics.json", "w") as f:
                    json.dump(metrics[name], f)

    for name in VARIANTS:
        with open(SAVE_DIR / f"{name}_metrics.json", "w") as f:
            json.dump(metrics[name], f)
    print("Ablation done. Saved to", SAVE_DIR)


if __name__ == "__main__":
    run()
