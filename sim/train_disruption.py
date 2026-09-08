"""
Topology Disruption Experiment — Best Paper Figure.
Every DISRUPT_EVERY rounds, 20% of ISLs fail for DISRUPT_LEN rounds.
Run ALL methods in parallel; TCP collapses, FRL adapts.

Usage: python -m sim.train_disruption
"""
import numpy as np, json, random
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
DISRUPT_EVERY   = 500   # trigger disruption every N rounds
DISRUPT_LEN     = 50    # disruption lasts N rounds
DISRUPT_FRAC    = 0.20  # fraction of ISLs killed
SAVE_DIR        = Path("results/disruption")
SEED            = 42


# ─── TCP-CUBIC disruption baseline ────────────────────────────────────────────
def _tcp_cubic_step(avail, kappa, cap, S, W, t_since_loss):
    """Returns (x_new, W_new, t_since_loss_new)."""
    CUBIC_C = 0.4; CUBIC_BETA = 0.7; W_MAX = 1.0; T_K = 5.0
    x_new = np.zeros((N_SAT, N_SAT))
    W_new = np.zeros((N_SAT, N_SAT))
    t_new = np.zeros((N_SAT, N_SAT))
    for i in range(N_SAT):
        for j in range(N_SAT):
            if not avail[i, j]:
                continue
            t = t_since_loss[i, j] + DT
            w_cubic = CUBIC_C * (t - T_K) ** 3 + W_MAX
            w_new = float(np.clip(w_cubic, 0.01, W_MAX))
            if kappa[i, j] > 0.8 or (cap[i, j] > 0 and S[i, j] > 0.95 * cap[i, j] * DT):
                w_new = W[i, j] * CUBIC_BETA
                t = 0.0
            W_new[i, j] = w_new
            t_new[i, j] = t
            x_new[i, j] = w_new * cap[i, j]
    return x_new, W_new, t_new


def _run_tcp(net, disruption_mask):
    """One round of TCP-CUBIC with optional disruption mask applied to avail."""
    rewards = []; kappas = []; drops = []; utils = []; tputs = []
    x_prev = np.zeros((N_SAT, N_SAT))
    W      = np.ones((N_SAT, N_SAT)) * 0.1
    t_sl   = np.zeros((N_SAT, N_SAT))
    for _ in range(STEPS_PER_ROUND):
        obs   = net.step(x_prev)
        kappa = obs["kappa"]; cap = obs["cap"]
        avail = obs["avail"] * disruption_mask   # apply outage
        S     = obs["S"];     drop = obs["drop"]
        x_new, W, t_sl = _tcp_cubic_step(avail, kappa, cap, S, W, t_sl)
        mask    = avail > 0; cap_sum = cap[mask].sum() + 1e-9
        step_r  = []
        for i in range(N_SAT):
            nbrs = [j for j in range(N_SAT) if avail[i, j]]
            if nbrs:
                step_r.append(compute_reward(i, nbrs, S, cap, kappa, drop, B, x_new, x_prev, DT))
        rewards.append(np.mean(step_r) if step_r else 0)
        kappas.append(float(kappa[mask].mean()) if mask.any() else 0.0)
        drops.append(float(drop[mask].sum() / (cap_sum * DT)))
        utils.append(float(x_new[mask].sum() / cap_sum))
        tputs.append(float(S[mask].sum() / DT / 1e9))
        x_prev = x_new.copy()
    return rewards, kappas, drops, utils, tputs


# ─── Generic RL step ──────────────────────────────────────────────────────────
def _rl_step(net, agents, x_prev, kappa_prev, disruption_mask, use_gnn=False):
    obs   = net.step(x_prev)
    kappa = obs["kappa"]; cap  = obs["cap"]
    avail = obs["avail"] * disruption_mask
    rtt   = obs["rtt"];   S    = obs["S"]; drop = obs["drop"]
    x_new = np.zeros((N_SAT, N_SAT)); step_rewards = []
    for i in range(N_SAT):
        nbrs = [j for j in range(N_SAT) if avail[i, j]]
        if not nbrs:
            continue
        if use_gnn:
            C_MAX = 10e9; RTT_MAX = 0.1
            nbr_feats = np.array([[float(kappa[i,j]), float(cap[i,j]/C_MAX),
                                   float(min(rtt[i,j]/RTT_MAX,1.0)),
                                   float(np.clip(kappa[i,j]-kappa_prev[i,j],-1,1))]
                                  for j in nbrs], dtype=np.float32)
            sf = nbr_feats.mean(axis=0)
            action, lp, val = agents[i].select_action(sf, nbr_feats)
        else:
            state = build_state(i, nbrs, kappa, cap, avail, rtt, kappa_prev, drop)
            s_pad = np.zeros(STATE_DIM, dtype=np.float32)
            s_pad[:len(state)] = state[:STATE_DIM]
            action, lp, val = agents[i].select_action(s_pad)
            sf = s_pad; nbr_feats = None
        for idx, j in enumerate(nbrs[:ACTION_DIM]):
            x_new[i, j] = action[idx] * cap[i, j]
        r = compute_reward(i, nbrs, S, cap, kappa, drop, B, x_new, x_prev, DT)
        if use_gnn:
            agents[i].store(sf, nbr_feats, action[:ACTION_DIM], r, lp, val, done=0.0)
        else:
            agents[i].store(sf, action[:ACTION_DIM], r, lp, val, done=0.0)
        step_rewards.append(r)
    mask    = avail > 0; cap_sum = cap[mask].sum() + 1e-9
    row = dict(
        reward = np.mean(step_rewards) if step_rewards else 0,
        kappa  = float(kappa[mask].mean()) if mask.any() else 0.0,
        drop   = float(drop[mask].sum() / (cap_sum * DT)),
        util   = float(x_new[mask].sum() / cap_sum),
        tput   = float(S[mask].sum() / DT / 1e9),
    )
    return x_new, kappa.copy(), row


# ─── Build disruption mask ────────────────────────────────────────────────────
def _disruption_mask(seed):
    rng  = random.Random(seed)
    mask = np.ones((N_SAT, N_SAT))
    edges = [(i, j) for i in range(N_SAT) for j in range(N_SAT) if i != j]
    kill  = rng.sample(edges, int(len(edges) * DISRUPT_FRAC))
    for i, j in kill:
        mask[i, j] = 0
    return mask


def _blank():
    return {"round": [], "mean_reward": [], "loss": [],
            "mean_kappa": [], "drop_rate": [], "link_utilization": [],
            "throughput_gbps": [], "disrupted": []}


def _append(m, k, disrupted, rewards, kappas, drops, utils, tputs, losses=None):
    m["round"].append(k)
    m["disrupted"].append(int(disrupted))
    m["mean_reward"].append(float(np.mean(rewards)))
    m["mean_kappa"].append(float(np.mean(kappas)))
    m["drop_rate"].append(float(np.mean(drops)))
    m["link_utilization"].append(float(np.mean(utils)))
    m["throughput_gbps"].append(float(np.mean(tputs)))
    m["loss"].append(float(np.mean(losses)) if losses is not None else 0.0)


def run():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # one shared network — all methods observe same environment
    net = ISLNetwork(seed=SEED)

    # initialise agents
    flat_agents = [PPOAgent(STATE_DIM, ACTION_DIM, i) for i in range(N_SAT)]
    gnn_agents  = [GNNPPOAgent(ACTION_DIM, i)         for i in range(N_SAT)]
    ind_agents  = [PPOAgent(STATE_DIM, ACTION_DIM, i) for i in range(N_SAT)]

    for agents in [flat_agents, gnn_agents, ind_agents]:
        w = agents[0].get_weights()
        for ag in agents: ag.set_weights(w)

    x_flat = np.zeros((N_SAT, N_SAT)); k_flat = np.zeros((N_SAT, N_SAT))
    x_gnn  = np.zeros((N_SAT, N_SAT)); k_gnn  = np.zeros((N_SAT, N_SAT))
    x_ind  = np.zeros((N_SAT, N_SAT)); k_ind  = np.zeros((N_SAT, N_SAT))

    m_tcp  = _blank(); m_flat = _blank()
    m_gnn  = _blank(); m_ind  = _blank()

    for k in range(K_ROUNDS):
        # determine disruption
        phase     = k % DISRUPT_EVERY
        disrupted = (phase < DISRUPT_LEN)
        dmask     = _disruption_mask(k // DISRUPT_EVERY) if disrupted else np.ones((N_SAT, N_SAT))

        # ── TCP ──
        tcp_r, tcp_k, tcp_d, tcp_u, tcp_t = _run_tcp(net, dmask)
        _append(m_tcp, k, disrupted, tcp_r, tcp_k, tcp_d, tcp_u, tcp_t)

        # ── Ind-PPO ──
        ind_r=[]; ind_k=[]; ind_d=[]; ind_u=[]; ind_t=[]
        for _ in range(STEPS_PER_ROUND):
            x_ind, k_ind_new, row = _rl_step(net, ind_agents, x_ind, k_ind, dmask)
            k_ind = k_ind_new
            ind_r.append(row["reward"]); ind_k.append(row["kappa"])
            ind_d.append(row["drop"]);   ind_u.append(row["util"])
            ind_t.append(row["tput"])
        ind_losses = [ag.update() for ag in ind_agents]
        _append(m_ind, k, disrupted, ind_r, ind_k, ind_d, ind_u, ind_t, ind_losses)

        # ── Flat-FRL ──
        flat_r=[]; flat_k=[]; flat_d=[]; flat_u=[]; flat_t=[]
        for _ in range(STEPS_PER_ROUND):
            x_flat, k_flat_new, row = _rl_step(net, flat_agents, x_flat, k_flat, dmask)
            k_flat = k_flat_new
            flat_r.append(row["reward"]); flat_k.append(row["kappa"])
            flat_d.append(row["drop"]);   flat_u.append(row["util"])
            flat_t.append(row["tput"])
        flat_losses = [ag.update() for ag in flat_agents]
        _, _, avail_g = compute_isl_graph(net.t)
        isl_gossip_round(flat_agents, avail_g)
        pos     = _sat_positions(net.t)
        contact = relay_contact_set(visible_satellites(pos), avail_g)
        trunk   = fedavg_round(flat_agents, contact)
        if trunk: broadcast(flat_agents, trunk, contact)
        _append(m_flat, k, disrupted, flat_r, flat_k, flat_d, flat_u, flat_t, flat_losses)

        # ── GNN-FRL ──
        gnn_r=[]; gnn_k=[]; gnn_d=[]; gnn_u=[]; gnn_t=[]
        for _ in range(STEPS_PER_ROUND):
            x_gnn, k_gnn_new, row = _rl_step(net, gnn_agents, x_gnn, k_gnn, dmask, use_gnn=True)
            k_gnn = k_gnn_new
            gnn_r.append(row["reward"]); gnn_k.append(row["kappa"])
            gnn_d.append(row["drop"]);   gnn_u.append(row["util"])
            gnn_t.append(row["tput"])
        gnn_losses = [ag.update() for ag in gnn_agents]
        isl_gossip_round(gnn_agents, avail_g)
        trunk = fedavg_round(gnn_agents, contact)
        if trunk: broadcast(gnn_agents, trunk, contact)
        _append(m_gnn, k, disrupted, gnn_r, gnn_k, gnn_d, gnn_u, gnn_t, gnn_losses)

        if k % LOG_EVERY == 0:
            tag = "[DISRUPT]" if disrupted else "         "
            print(f"Round {k:4d} {tag} | "
                  f"TCP tput={m_tcp['throughput_gbps'][-1]:.1f}G drop={m_tcp['drop_rate'][-1]:.3f} | "
                  f"Flat-FRL tput={m_flat['throughput_gbps'][-1]:.1f}G drop={m_flat['drop_rate'][-1]:.3f} | "
                  f"GNN-FRL tput={m_gnn['throughput_gbps'][-1]:.1f}G")

        if k % 500 == 0:
            for name, m in [("tcp", m_tcp), ("indppo", m_ind),
                             ("flat_frl", m_flat), ("gnn_frl", m_gnn)]:
                with open(SAVE_DIR / f"{name}_metrics.json", "w") as f:
                    json.dump(m, f)

    for name, m in [("tcp", m_tcp), ("indppo", m_ind),
                    ("flat_frl", m_flat), ("gnn_frl", m_gnn)]:
        with open(SAVE_DIR / f"{name}_metrics.json", "w") as f:
            json.dump(m, f)
    print("Disruption experiment done. Saved to", SAVE_DIR)


if __name__ == "__main__":
    run()
