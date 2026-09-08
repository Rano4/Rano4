"""
Handover Scenario Experiment — Best Paper Figure.

Models the critical LEO challenge: satellites continuously enter/exit
Ground Station (GS) visibility windows, breaking direct aggregation.

Variants tested:
  - TCP-CUBIC          : no learning, just rate control
  - FedAvg (direct)   : aggregates only via direct GS contact — FAILS during eclipse
  - CA-PFedAvg        : relay expansion keeps federation alive during handover
  - Flat-FRL          : CA-PFedAvg + ISL-Gossip
  - GNN-FRL           : full proposal

Key mechanic:
  - 3 Ground Stations (Europe, California, Sydney)
  - Each GS can only communicate with satellites in its visibility cone
  - As the constellation orbits, satellites rotate in/out of GS range
  - During eclipse: FedAvg has no aggregator → stale weights → degraded performance
  - CA-PFedAvg uses ISL relay to bridge the gap: out-of-range sats relay via in-range peers

Usage: python -m sim.train_handover
"""
import numpy as np, json, math
from pathlib import Path
from .env.network import ISLNetwork, DT, B
from .env.constellation import N_SAT, compute_isl_graph, _sat_positions
from .agents.ppo import PPOAgent
from .agents.gnn_ppo import GNNPPOAgent
from .fed.fedavg import relay_contact_set, fedavg_round, broadcast
from .fed.gossip import isl_gossip_round
from .utils.state import build_state, STATE_DIM
from .utils.reward import compute_reward

K_ROUNDS        = 3000
STEPS_PER_ROUND = 10
ACTION_DIM      = 4
LOG_EVERY       = 100
SAVE_DIR        = Path("results/handover")
SEED            = 42

# Ground station positions (lat, lon in degrees)
GROUND_STATIONS = [
    {"name": "Europe",     "lat":  48.8,  "lon":   2.3},
    {"name": "California", "lat":  37.4,  "lon": -122.1},
    {"name": "Sydney",     "lat": -33.9,  "lon":  151.2},
]
GS_ELEVATION_MIN = 5.0   # degrees, minimum elevation angle for GS contact
GS_MAX_RANGE_KM  = 2500  # km — GS can only reach sats within this slant range

EARTH_RADIUS_KM  = 6371.0
LEO_ALT_KM       = 550.0


def _gs_ecef(lat_deg, lon_deg):
    """Ground station position in ECEF (km)."""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    r   = EARTH_RADIUS_KM
    return np.array([r * math.cos(lat) * math.cos(lon),
                     r * math.cos(lat) * math.sin(lon),
                     r * math.sin(lat)])


def visible_from_gs(sat_positions_ecef, t=None):
    """
    Returns set of satellite indices visible to AT LEAST ONE ground station.
    sat_positions_ecef: (N_SAT, 3) array of satellite ECEF positions in km.
    """
    gs_positions = [_gs_ecef(gs["lat"], gs["lon"]) for gs in GROUND_STATIONS]
    visible = set()
    for i, sat_pos in enumerate(sat_positions_ecef):
        for gs_pos in gs_positions:
            diff = sat_pos - gs_pos
            dist = float(np.linalg.norm(diff))
            if dist > GS_MAX_RANGE_KM:
                continue
            # elevation angle check
            gs_norm = gs_pos / np.linalg.norm(gs_pos)
            elev = math.degrees(math.asin(float(np.dot(diff / dist, gs_norm))))
            if elev >= GS_ELEVATION_MIN:
                visible.add(i)
                break
    return visible


def _sat_ecef(positions_skyfield):
    """
    Convert skyfield positions to ECEF km array.
    positions_skyfield is the output of _sat_positions(t) — (N_SAT, 3) in km ECEF.
    """
    return positions_skyfield  # already ECEF km from constellation module


def _blank():
    return {"round": [], "mean_reward": [], "loss": [],
            "mean_kappa": [], "drop_rate": [], "link_utilization": [],
            "throughput_gbps": [], "gs_visible_count": [], "fed_participants": []}


def _rl_step(agents, net, x_prev, kappa_prev, use_gnn=False):
    obs   = net.step(x_prev)
    kappa = obs["kappa"]; cap  = obs["cap"]
    avail = obs["avail"]; rtt  = obs["rtt"]; S = obs["S"]; drop = obs["drop"]
    x_new = np.zeros((N_SAT, N_SAT)); step_rewards = []
    C_MAX = 10e9; RTT_MAX = 0.1

    for i in range(N_SAT):
        nbrs = [j for j in range(N_SAT) if avail[i, j]]
        if not nbrs: continue
        if use_gnn:
            nbr_feats = np.array([[float(kappa[i,j]), float(cap[i,j]/C_MAX),
                                   float(min(rtt[i,j]/RTT_MAX,1.0)),
                                   float(np.clip(kappa[i,j]-kappa_prev[i,j],-1,1))]
                                  for j in nbrs], dtype=np.float32)
            sf = nbr_feats.mean(axis=0)
            action, lp, val = agents[i].select_action(sf, nbr_feats)
            r = compute_reward(i, nbrs, S, cap, kappa, drop, B, x_new, x_prev, DT)
            agents[i].store(sf, nbr_feats, action[:ACTION_DIM], r, lp, val, done=0.0)
        else:
            state = build_state(i, nbrs, kappa, cap, avail, rtt, kappa_prev, drop)
            s_pad = np.zeros(STATE_DIM, dtype=np.float32)
            s_pad[:len(state)] = state[:STATE_DIM]
            action, lp, val = agents[i].select_action(s_pad)
            r = compute_reward(i, nbrs, S, cap, kappa, drop, B, x_new, x_prev, DT)
            agents[i].store(s_pad, action[:ACTION_DIM], r, lp, val, done=0.0)
        for idx, j in enumerate(nbrs[:ACTION_DIM]):
            x_new[i, j] = action[idx] * cap[i, j]
        step_rewards.append(r)

    mask = avail > 0; cap_sum = cap[mask].sum() + 1e-9
    return x_new, kappa.copy(), dict(
        reward = float(np.mean(step_rewards)) if step_rewards else 0.0,
        kappa  = float(kappa[mask].mean()) if mask.any() else 0.0,
        drop   = float(drop[mask].sum() / (cap_sum * DT)),
        util   = float(x_new[mask].sum() / cap_sum),
        tput   = float(S[mask].sum() / DT / 1e9),
    )


def _fedavg_direct_only(agents, gs_visible):
    """
    Vanilla FedAvg: only satellites currently visible to a GS participate.
    If no satellite is visible (full eclipse), skip — weights go stale.
    Returns number of participants.
    """
    if not gs_visible:
        return 0  # eclipse — no aggregation this round
    participant_weights = [agents[i].get_trunk_weights() for i in gs_visible]
    if not participant_weights:
        return 0
    # simple mean
    avg = {k: sum(w[k] for w in participant_weights) / len(participant_weights)
           for k in participant_weights[0]}
    # broadcast to ALL agents (models telemetry'd to ground, re-uplinked)
    for ag in agents:
        ag.set_trunk_weights(avg)
    return len(gs_visible)


def _append(m, k, rewards, kappas, drops, utils, tputs, losses, gs_vis, fed_n):
    m["round"].append(k)
    m["mean_reward"].append(float(np.mean(rewards)))
    m["loss"].append(float(np.mean(losses)) if losses is not None else 0.0)
    m["mean_kappa"].append(float(np.mean(kappas)))
    m["drop_rate"].append(float(np.mean(drops)))
    m["link_utilization"].append(float(np.mean(utils)))
    m["throughput_gbps"].append(float(np.mean(tputs)))
    m["gs_visible_count"].append(int(gs_vis))
    m["fed_participants"].append(int(fed_n))


def run():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    net = ISLNetwork(seed=SEED)

    def _make_gnn():
        ags = [GNNPPOAgent(ACTION_DIM, i) for i in range(N_SAT)]
        w = ags[0].get_weights()
        for a in ags: a.set_weights(w)
        return ags

    def _make_flat():
        ags = [PPOAgent(STATE_DIM, ACTION_DIM, i) for i in range(N_SAT)]
        w = ags[0].get_weights()
        for a in ags: a.set_weights(w)
        return ags

    agents = {
        "fedavg":   _make_flat(),
        "capfedavg": _make_flat(),
        "flat_frl": _make_flat(),
        "gnn_frl":  _make_gnn(),
    }
    x_prev = {n: np.zeros((N_SAT, N_SAT)) for n in agents}
    k_prev = {n: np.zeros((N_SAT, N_SAT)) for n in agents}
    metrics = {n: _blank() for n in agents}

    # TCP state
    m_tcp = _blank()
    W_tcp = np.ones((N_SAT, N_SAT)) * 0.1
    t_tcp = np.zeros((N_SAT, N_SAT))
    x_tcp = np.zeros((N_SAT, N_SAT))

    for k in range(K_ROUNDS):
        # ── topology at this round ──
        _, _, avail_g = compute_isl_graph(net.t)
        pos_ecef = _sat_positions(net.t)            # (N_SAT, 3) ECEF km
        gs_visible = visible_from_gs(pos_ecef)      # set of sat indices

        # relay-expanded contact set for CA-PFedAvg methods
        contact_relay = relay_contact_set(gs_visible, avail_g)

        # ── TCP-CUBIC ──
        CUBIC_C = 0.4; CUBIC_BETA = 0.7; W_MAX = 1.0; T_K = 5.0
        tcp_r=[]; tcp_k=[]; tcp_d=[]; tcp_u=[]; tcp_t=[]
        for _ in range(STEPS_PER_ROUND):
            obs = net.step(x_tcp)
            kappa = obs["kappa"]; cap = obs["cap"]
            avail = obs["avail"]; S = obs["S"]; drop = obs["drop"]
            x_new = np.zeros((N_SAT, N_SAT))
            for i in range(N_SAT):
                for j in range(N_SAT):
                    if not avail[i, j]: continue
                    tt = t_tcp[i,j] + DT
                    wc = float(np.clip(CUBIC_C * (tt - T_K)**3 + W_MAX, 0.01, W_MAX))
                    if kappa[i,j] > 0.8:
                        wc = W_tcp[i,j] * CUBIC_BETA; tt = 0.0
                    W_tcp[i,j] = wc; t_tcp[i,j] = tt
                    x_new[i,j] = wc * cap[i,j]
            x_tcp = x_new
            mask = avail > 0; cap_sum = cap[mask].sum() + 1e-9
            step_r = []
            for i in range(N_SAT):
                nbrs = [j for j in range(N_SAT) if avail[i,j]]
                if nbrs: step_r.append(compute_reward(i, nbrs, S, cap, kappa, drop, B, x_new, x_tcp, DT))
            tcp_r.append(np.mean(step_r) if step_r else 0)
            tcp_k.append(float(kappa[mask].mean()) if mask.any() else 0.0)
            tcp_d.append(float(drop[mask].sum()/(cap_sum*DT)))
            tcp_u.append(float(x_new[mask].sum()/cap_sum))
            tcp_t.append(float(S[mask].sum()/DT/1e9))
        _append(m_tcp, k, tcp_r, tcp_k, tcp_d, tcp_u, tcp_t,
                None, len(gs_visible), len(gs_visible))

        # ── RL variants ──
        for name, use_gnn in [("fedavg", False), ("capfedavg", False),
                               ("flat_frl", False), ("gnn_frl", True)]:
            r=[]; kp=[]; dp=[]; ul=[]; tp=[]
            for _ in range(STEPS_PER_ROUND):
                x_new, k_new, row = _rl_step(agents[name], net,
                                             x_prev[name], k_prev[name], use_gnn)
                x_prev[name] = x_new; k_prev[name] = k_new
                r.append(row["reward"]); kp.append(row["kappa"])
                dp.append(row["drop"]);  ul.append(row["util"]); tp.append(row["tput"])

            losses = [ag.update() for ag in agents[name]]

            # federation — the key difference between methods
            if name == "fedavg":
                # DIRECT only — breaks during handover eclipse
                n_fed = _fedavg_direct_only(agents[name], gs_visible)

            elif name == "capfedavg":
                # relay expansion — survives handover
                trunk = fedavg_round(agents[name], contact_relay)
                n_fed = len(contact_relay)
                if trunk: broadcast(agents[name], trunk, contact_relay)

            elif name in ("flat_frl", "gnn_frl"):
                # ISL-Gossip + relay CA-PFedAvg
                isl_gossip_round(agents[name], avail_g)
                trunk = fedavg_round(agents[name], contact_relay)
                n_fed = len(contact_relay)
                if trunk: broadcast(agents[name], trunk, contact_relay)

            _append(metrics[name], k, r, kp, dp, ul, tp, losses,
                    len(gs_visible), n_fed)

        if k % LOG_EVERY == 0:
            print(f"Round {k:4d} | GS-visible={len(gs_visible):2d}/{N_SAT} "
                  f"relay={len(contact_relay):2d} | "
                  f"FedAvg={metrics['fedavg']['throughput_gbps'][-1]:.1f}G "
                  f"CA-PFed={metrics['capfedavg']['throughput_gbps'][-1]:.1f}G "
                  f"GNN-FRL={metrics['gnn_frl']['throughput_gbps'][-1]:.1f}G")

        if k % 500 == 0:
            for name, m in list(metrics.items()) + [("tcp", m_tcp)]:
                with open(SAVE_DIR / f"{name}_metrics.json", "w") as f:
                    json.dump(m, f)

    for name, m in list(metrics.items()) + [("tcp", m_tcp)]:
        with open(SAVE_DIR / f"{name}_metrics.json", "w") as f:
            json.dump(m, f)
    print("Handover experiment done. Saved to", SAVE_DIR)


if __name__ == "__main__":
    run()
