# ══════════════════════════════════════════════════════════════════════════════
# HOW TO USE:
#   Open a NEW Google Colab notebook.
#   Each block below = one Colab cell.
#   Copy everything between the ═══ lines into one cell, then run.
#   For cells starting with %%writefile: that line MUST be the very first line.
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
# CELL 1 — Setup
# ══════════════════════════════════════════════════════════════════════════════

from google.colab import drive
drive.mount('/content/drive')

import os
os.makedirs('sim/env',    exist_ok=True)
os.makedirs('sim/agents', exist_ok=True)
os.makedirs('sim/fed',    exist_ok=True)
os.makedirs('sim/utils',  exist_ok=True)
for d in ['sim','sim/env','sim/agents','sim/fed','sim/utils']:
    open(f'{d}/__init__.py','a').close()
os.makedirs('results', exist_ok=True)
os.makedirs('/content/drive/MyDrive/Rano4_results', exist_ok=True)

import subprocess
subprocess.run(['pip','install','torch','numpy','matplotlib','-q'])
print("Setup done")


# ══════════════════════════════════════════════════════════════════════════════
# CELL 2 — constellation.py   (%%writefile — first line must stay as is)
# ══════════════════════════════════════════════════════════════════════════════

%%writefile sim/env/constellation.py
import numpy as np
import math

N_PLANES    = 6
N_PER_PLANE = 11
N_SAT       = N_PLANES * N_PER_PLANE   # 66
ALT_KM      = 550.0
INC_DEG     = 53.0
EARTH_R_KM  = 6371.0
MU          = 3.986004418e14
MAX_ISL_KM  = 4500.0

_R  = EARTH_R_KM + ALT_KM
_T  = 2 * math.pi * math.sqrt((_R * 1e3)**3 / MU)
_OM = 2 * math.pi / _T


def _sat_positions(t_sec):
    pos = np.zeros((N_SAT, 3))
    inc = math.radians(INC_DEG)
    for p in range(N_PLANES):
        raan = 2 * math.pi * p / N_PLANES
        for s in range(N_PER_PLANE):
            M0    = 2 * math.pi * s / N_PER_PLANE + math.pi / N_PLANES * p
            M     = (M0 + _OM * t_sec) % (2 * math.pi)
            x_orb = _R * math.cos(M)
            y_orb = _R * math.sin(M)
            cos_r, sin_r = math.cos(raan), math.sin(raan)
            cos_i, sin_i = math.cos(inc),  math.sin(inc)
            x_eci = cos_r * x_orb - sin_r * cos_i * y_orb
            y_eci = sin_r * x_orb + cos_r * cos_i * y_orb
            z_eci = sin_i * y_orb
            pos[p * N_PER_PLANE + s] = [x_eci, y_eci, z_eci]
    return pos


def compute_isl_graph(t_sec):
    pos   = _sat_positions(t_sec)
    avail = np.zeros((N_SAT, N_SAT), dtype=bool)
    for i in range(N_SAT):
        for j in range(i + 1, N_SAT):
            d = float(np.linalg.norm(pos[i] - pos[j]))
            if d <= MAX_ISL_KM:
                avail[i, j] = avail[j, i] = True
    return None, None, avail


# ══════════════════════════════════════════════════════════════════════════════
# CELL 3 — network.py
# ══════════════════════════════════════════════════════════════════════════════

%%writefile sim/env/network.py
import numpy as np
from .constellation import N_SAT, compute_isl_graph, _sat_positions, MAX_ISL_KM

DT       = 1.0
B        = 1e8
C_LEVELS = [10e9, 5e9, 2e9, 1e9]
RTT_BASE = 0.02


class ISLNetwork:
    def __init__(self, seed=42):
        self.rng   = np.random.default_rng(seed)
        self.t     = 0.0
        self.kappa = np.zeros((N_SAT, N_SAT))
        self.S     = np.zeros((N_SAT, N_SAT))

    def step(self, x_alloc):
        self.t += DT
        _, _, avail = compute_isl_graph(self.t)
        pos = _sat_positions(self.t)
        cap = np.zeros((N_SAT, N_SAT))
        rtt = np.zeros((N_SAT, N_SAT))
        for i in range(N_SAT):
            for j in range(N_SAT):
                if not avail[i, j]:
                    continue
                d    = float(np.linalg.norm(pos[i] - pos[j]))
                frac = d / MAX_ISL_KM
                if   frac < 0.25: cap[i, j] = C_LEVELS[0]
                elif frac < 0.50: cap[i, j] = C_LEVELS[1]
                elif frac < 0.75: cap[i, j] = C_LEVELS[2]
                else:             cap[i, j] = C_LEVELS[3]
                rtt[i, j] = RTT_BASE + 2 * d / 3e5
        demand    = self.rng.poisson(cap * 0.7 * DT / 1e9).astype(float) * 1e9
        new_kappa = np.zeros((N_SAT, N_SAT))
        S         = np.zeros((N_SAT, N_SAT))
        drop      = np.zeros((N_SAT, N_SAT))
        for i in range(N_SAT):
            for j in range(N_SAT):
                if not avail[i, j]:
                    continue
                buf          = self.kappa[i, j] * B + demand[i, j]
                served       = min(x_alloc[i, j] * DT, buf, cap[i, j] * DT)
                buf         -= served
                overflow     = max(buf - B, 0)
                drop[i, j]      = overflow
                new_kappa[i, j] = max(0, buf - overflow) / B
                S[i, j]         = served
        self.kappa = new_kappa
        self.S     = S
        return dict(kappa=new_kappa, cap=cap,
                    avail=avail.astype(float), rtt=rtt, S=S, drop=drop)


# ══════════════════════════════════════════════════════════════════════════════
# CELL 4 — state.py + reward.py
# ══════════════════════════════════════════════════════════════════════════════

%%writefile sim/utils/state.py
import numpy as np

STATE_DIM = 18
C_MAX     = 10e9
RTT_MAX   = 0.1


def build_state(sat_id, neighbors, kappa, cap, avail, rtt, kappa_prev, drop):
    feats = []
    for j in neighbors[:4]:
        phi = float(np.clip(kappa[sat_id, j] - kappa_prev[sat_id, j], -1, 1))
        feats += [float(kappa[sat_id, j]),
                  float(cap[sat_id, j] / C_MAX),
                  float(min(rtt[sat_id, j] / RTT_MAX, 1.0)),
                  phi]
    while len(feats) < 16:
        feats += [0.0, 0.0, 0.0, 0.0]
    mask = avail > 0
    feats.append(float(kappa[mask].mean()) if mask.any() else 0.0)
    feats.append(float(drop[mask].mean())  if mask.any() else 0.0)
    return np.array(feats[:STATE_DIM], dtype=np.float32)


%%writefile sim/utils/reward.py
import numpy as np


def compute_reward(sat_id, neighbors, S, cap, kappa, drop, B, x_new, x_prev, DT):
    r = 0.0
    for j in neighbors:
        c = cap[sat_id, j]
        if c < 1:
            continue
        tput = S[sat_id, j] / (c * DT + 1e-9)
        cong = -kappa[sat_id, j]
        drp  = -drop[sat_id, j] / (B + 1e-9)
        chg  = -abs(x_new[sat_id, j] - x_prev[sat_id, j]) / (c + 1e-9) * 0.1
        r   += tput + cong + drp + chg
    return r / (len(neighbors) + 1e-9)


# ══════════════════════════════════════════════════════════════════════════════
# CELL 5 — ppo.py
# ══════════════════════════════════════════════════════════════════════════════

%%writefile sim/agents/ppo.py
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Beta

GAMMA = 0.99
LAM   = 0.95
CLIP  = 0.2
LR    = 3e-4
EPOCHS = 4
BATCH  = 64


class _Trunk(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 64),        nn.ReLU())

    def forward(self, x):
        return self.net(x)


class _Head(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.alpha = nn.Linear(64, action_dim)
        self.beta  = nn.Linear(64, action_dim)
        self.value = nn.Linear(64, 1)

    def forward(self, h):
        a = torch.clamp(torch.softplus(self.alpha(h)) + 1.0, 1.0, 10.0)
        b = torch.clamp(torch.softplus(self.beta(h))  + 1.0, 1.0, 10.0)
        return a, b, self.value(h)


class PPOAgent:
    def __init__(self, state_dim, action_dim, sat_id):
        self.sat_id     = sat_id
        self.action_dim = action_dim
        self.trunk = _Trunk(state_dim)
        self.head  = _Head(action_dim)
        self.opt   = torch.optim.Adam(
            list(self.trunk.parameters()) + list(self.head.parameters()), lr=LR)
        self.buf = []

    def get_trunk_weights(self):
        return {k: v.clone() for k, v in self.trunk.state_dict().items()}

    def set_trunk_weights(self, w):
        self.trunk.load_state_dict(w)

    def get_weights(self):
        d = {}
        d.update({f"trunk.{k}": v.clone() for k, v in self.trunk.state_dict().items()})
        d.update({f"head.{k}":  v.clone() for k, v in self.head.state_dict().items()})
        return d

    def set_weights(self, w):
        self.trunk.load_state_dict(
            {k[6:]: v for k, v in w.items() if k.startswith("trunk.")})
        self.head.load_state_dict(
            {k[5:]: v for k, v in w.items() if k.startswith("head.")})

    def select_action(self, state):
        s = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            h = self.trunk(s)
            a, b, v = self.head(h)
        dist   = Beta(a, b)
        action = dist.sample()
        lp     = dist.log_prob(action).sum(-1)
        return action.squeeze(0).numpy(), float(lp), float(v)

    def store(self, s, a, r, lp, v, done):
        self.buf.append((s, a, float(r), float(lp), float(v), float(done)))

    def update(self):
        if len(self.buf) < 2:
            return 0.0
        S, A, R, LP, V, D = map(np.array, zip(*self.buf))
        self.buf = []
        adv = np.zeros_like(R)
        last = 0
        for t in reversed(range(len(R))):
            nv    = V[t + 1] if t + 1 < len(V) else 0
            delta = R[t] + GAMMA * (1 - D[t]) * nv - V[t]
            last  = delta + GAMMA * LAM * (1 - D[t]) * last
            adv[t] = last
        ret = adv + V
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        S_t   = torch.FloatTensor(S)
        A_t   = torch.FloatTensor(A)
        LP_t  = torch.FloatTensor(LP)
        adv_t = torch.FloatTensor(adv)
        ret_t = torch.FloatTensor(ret)
        losses = []
        for _ in range(EPOCHS):
            idx = np.random.permutation(len(S))
            for start in range(0, len(S), BATCH):
                b       = idx[start:start + BATCH]
                h       = self.trunk(S_t[b])
                al, be, v = self.head(h)
                dist    = Beta(al, be)
                nlp     = dist.log_prob(torch.clamp(A_t[b], 1e-6, 1 - 1e-6)).sum(-1)
                ratio   = torch.exp(nlp - LP_t[b])
                s1      = ratio * adv_t[b]
                s2      = torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * adv_t[b]
                pl      = -torch.min(s1, s2).mean()
                vl      = (v.squeeze() - ret_t[b]).pow(2).mean()
                loss    = pl + 0.5 * vl
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.trunk.parameters()) + list(self.head.parameters()), 0.5)
                self.opt.step()
                losses.append(float(loss))
        return float(np.mean(losses))


# ══════════════════════════════════════════════════════════════════════════════
# CELL 6 — gnn_ppo.py
# ══════════════════════════════════════════════════════════════════════════════

%%writefile sim/agents/gnn_ppo.py
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Beta

GAMMA  = 0.99
LAM    = 0.95
CLIP   = 0.2
LR     = 3e-4
EPOCHS = 4
BATCH  = 64


class _GNNTrunk(nn.Module):
    def __init__(self):
        super().__init__()
        self.node_proj = nn.Linear(4, 32)
        self.agg_mlp   = nn.Sequential(
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU())

    def forward(self, sf, nf):
        sf_h = torch.relu(self.node_proj(sf))
        nf_h = torch.relu(self.node_proj(nf))
        agg  = nf_h.mean(0)
        return self.agg_mlp(torch.cat([sf_h, agg]))


class _Head(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.alpha = nn.Linear(64, action_dim)
        self.beta  = nn.Linear(64, action_dim)
        self.value = nn.Linear(64, 1)

    def forward(self, h):
        a = torch.clamp(torch.softplus(self.alpha(h)) + 1, 1, 10)
        b = torch.clamp(torch.softplus(self.beta(h))  + 1, 1, 10)
        return a, b, self.value(h)


class GNNPPOAgent:
    def __init__(self, action_dim, sat_id):
        self.sat_id     = sat_id
        self.action_dim = action_dim
        self.trunk = _GNNTrunk()
        self.head  = _Head(action_dim)
        self.opt   = torch.optim.Adam(
            list(self.trunk.parameters()) + list(self.head.parameters()), lr=LR)
        self.buf = []

    def get_trunk_weights(self):
        return {k: v.clone() for k, v in self.trunk.state_dict().items()}

    def set_trunk_weights(self, w):
        self.trunk.load_state_dict(w)

    def get_weights(self):
        d = {}
        d.update({f"trunk.{k}": v.clone() for k, v in self.trunk.state_dict().items()})
        d.update({f"head.{k}":  v.clone() for k, v in self.head.state_dict().items()})
        return d

    def set_weights(self, w):
        self.trunk.load_state_dict(
            {k[6:]: v for k, v in w.items() if k.startswith("trunk.")})
        self.head.load_state_dict(
            {k[5:]: v for k, v in w.items() if k.startswith("head.")})

    def select_action(self, sf, nf):
        sf_t = torch.FloatTensor(sf)
        nf_t = torch.FloatTensor(nf) if len(nf) > 0 else torch.zeros(1, 4)
        with torch.no_grad():
            h = self.trunk(sf_t, nf_t)
            a, b, v = self.head(h)
        dist   = Beta(a, b)
        action = dist.sample()
        lp     = dist.log_prob(action).sum(-1)
        return action.numpy(), float(lp), float(v)

    def store(self, sf, nf, a, r, lp, v, done):
        self.buf.append((sf, nf, a, float(r), float(lp), float(v), float(done)))

    def update(self):
        if len(self.buf) < 2:
            return 0.0
        SFs, NFs, A, R, LP, V, D = zip(*self.buf)
        self.buf = []
        A  = np.array(A)
        R  = np.array(R)
        LP = np.array(LP)
        V  = np.array(V)
        D  = np.array(D)
        adv = np.zeros_like(R)
        last = 0
        for t in reversed(range(len(R))):
            nv    = V[t + 1] if t + 1 < len(V) else 0
            delta = R[t] + GAMMA * (1 - D[t]) * nv - V[t]
            last  = delta + GAMMA * LAM * (1 - D[t]) * last
            adv[t] = last
        ret   = adv + V
        adv   = (adv - adv.mean()) / (adv.std() + 1e-8)
        A_t   = torch.FloatTensor(A)
        LP_t  = torch.FloatTensor(LP)
        adv_t = torch.FloatTensor(adv)
        ret_t = torch.FloatTensor(ret)
        losses = []
        for _ in range(EPOCHS):
            idx = np.random.permutation(len(A))
            for start in range(0, len(A), BATCH):
                b  = idx[start:start + BATCH]
                hs = []
                for i in b:
                    sf_t = torch.FloatTensor(SFs[i])
                    nf_t = torch.FloatTensor(NFs[i]) if len(NFs[i]) > 0 else torch.zeros(1, 4)
                    hs.append(self.trunk(sf_t, nf_t))
                H = torch.stack(hs)
                al, be, vv = self.head(H)
                dist  = Beta(al, be)
                nlp   = dist.log_prob(torch.clamp(A_t[b], 1e-6, 1 - 1e-6)).sum(-1)
                ratio = torch.exp(nlp - LP_t[b])
                s1    = ratio * adv_t[b]
                s2    = torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * adv_t[b]
                pl    = -torch.min(s1, s2).mean()
                vl    = (vv.squeeze() - ret_t[b]).pow(2).mean()
                loss  = pl + 0.5 * vl
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.trunk.parameters()) + list(self.head.parameters()), 0.5)
                self.opt.step()
                losses.append(float(loss))
        return float(np.mean(losses))


# ══════════════════════════════════════════════════════════════════════════════
# CELL 7 — fedavg.py + gossip.py
# ══════════════════════════════════════════════════════════════════════════════

%%writefile sim/fed/fedavg.py
import numpy as np
import math
from ..env.constellation import N_SAT, EARTH_R_KM

GS_LIST = [
    {"lat":  48.8, "lon":   2.3},
    {"lat":  37.4, "lon": -122.1},
    {"lat": -33.9, "lon":  151.2},
]
GS_MAX_KM = 2500.0
GS_ELEV   = 5.0


def _gs_ecef(gs):
    lat = math.radians(gs["lat"])
    lon = math.radians(gs["lon"])
    r   = EARTH_R_KM
    return np.array([r * math.cos(lat) * math.cos(lon),
                     r * math.cos(lat) * math.sin(lon),
                     r * math.sin(lat)])


def visible_satellites(pos_ecef):
    gps = [_gs_ecef(g) for g in GS_LIST]
    vis = set()
    for i, sp in enumerate(pos_ecef):
        for gp in gps:
            diff = sp - gp
            d    = float(np.linalg.norm(diff))
            if d > GS_MAX_KM:
                continue
            gn   = gp / np.linalg.norm(gp)
            elev = math.degrees(math.asin(float(np.clip(np.dot(diff / d, gn), -1, 1))))
            if elev >= GS_ELEV:
                vis.add(i)
                break
    return vis


def relay_contact_set(direct, avail_g, hops=2):
    contact = set(direct)
    for _ in range(hops):
        expand = set()
        for i in contact:
            for j in range(N_SAT):
                if avail_g[i, j]:
                    expand.add(j)
        contact |= expand
    return contact


def fedavg_round(agents, contact):
    if not contact:
        return None
    ids     = list(contact)
    weights = [agents[i].get_trunk_weights() for i in ids]
    avg     = {k: sum(w[k] for w in weights) / len(weights) for k in weights[0]}
    return avg


def broadcast(agents, trunk, contact):
    for i in contact:
        agents[i].set_trunk_weights(trunk)


%%writefile sim/fed/gossip.py
DELTA_THRESH = 1e-4


def isl_gossip_round(agents, avail_g):
    n = len(agents)
    for i in range(n):
        wi = agents[i].get_trunk_weights()
        for j in range(n):
            if not avail_g[i, j]:
                continue
            wj     = agents[j].get_trunk_weights()
            update = {k: wi[k] - wj[k] for k in wi
                      if (wi[k] - wj[k]).abs().max() > DELTA_THRESH}
            if not update:
                continue
            merged = {k: wj[k] + 0.5 * update[k] if k in update else wj[k] for k in wj}
            agents[j].set_trunk_weights(merged)


# ══════════════════════════════════════════════════════════════════════════════
# CELL 8 — Imports + shared helpers (run once before any training cell)
# ══════════════════════════════════════════════════════════════════════════════

import numpy as np
import json
from pathlib import Path

from sim.env.network import ISLNetwork, DT, B
from sim.env.constellation import N_SAT, compute_isl_graph, _sat_positions
from sim.agents.ppo import PPOAgent
from sim.agents.gnn_ppo import GNNPPOAgent
from sim.fed.fedavg import visible_satellites, relay_contact_set, fedavg_round, broadcast
from sim.fed.gossip import isl_gossip_round
from sim.utils.state import build_state, STATE_DIM
from sim.utils.reward import compute_reward

K_ROUNDS        = 3000
STEPS_PER_ROUND = 10
ACTION_DIM      = 4
LOG_EVERY       = 100
C_MAX           = 10e9
RTT_MAX         = 0.1


def blank():
    return {"round": [], "mean_reward": [], "loss": [],
            "mean_kappa": [], "drop_rate": [], "link_utilization": [],
            "throughput_gbps": [], "p99_rtt": []}


def collect(obs, x_new, x_prev):
    kappa = obs["kappa"]; cap = obs["cap"]
    avail = obs["avail"]; rtt = obs["rtt"]
    mask     = avail > 0
    cap_sum  = cap[mask].sum() + 1e-9
    return dict(
        kappa = float(kappa[mask].mean()) if mask.any() else 0.0,
        drop  = float(obs["drop"][mask].sum() / (cap_sum * DT)),
        util  = float(x_new[mask].sum() / cap_sum),
        tput  = float(obs["S"][mask].sum() / DT / 1e9),
        p99   = float(np.percentile(rtt[mask], 99)) if mask.any() else 0.0,
    )


def record(m, k, rewards, rows, losses):
    m["round"].append(k)
    m["mean_reward"].append(float(np.mean(rewards)) if rewards else 0.0)
    m["loss"].append(float(np.mean(losses)) if losses else 0.0)
    m["mean_kappa"].append(float(np.mean([r["kappa"] for r in rows])))
    m["drop_rate"].append(float(np.mean([r["drop"]  for r in rows])))
    m["link_utilization"].append(float(np.mean([r["util"] for r in rows])))
    m["throughput_gbps"].append(float(np.mean([r["tput"] for r in rows])))
    m["p99_rtt"].append(float(np.mean([r["p99"]  for r in rows])))


def save(m, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(m, open(path, "w"))


def make_flat():
    ags = [PPOAgent(STATE_DIM, ACTION_DIM, i) for i in range(N_SAT)]
    w   = ags[0].get_weights()
    for a in ags: a.set_weights(w)
    return ags


def make_gnn():
    ags = [GNNPPOAgent(ACTION_DIM, i) for i in range(N_SAT)]
    w   = ags[0].get_weights()
    for a in ags: a.set_weights(w)
    return ags


def flat_step(agents, obs, x_prev, kappa_prev):
    kappa = obs["kappa"]; cap = obs["cap"]
    avail = obs["avail"]; rtt = obs["rtt"]; S = obs["S"]; drop = obs["drop"]
    x_new = np.zeros((N_SAT, N_SAT))
    rewards = []
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
        agents[i].store(s_pad, action[:ACTION_DIM], r, lp, val, 0.0)
        rewards.append(r)
    return x_new, kappa.copy(), rewards


def gnn_step(agents, obs, x_prev, kappa_prev):
    kappa = obs["kappa"]; cap = obs["cap"]
    avail = obs["avail"]; rtt = obs["rtt"]; S = obs["S"]; drop = obs["drop"]
    x_new = np.zeros((N_SAT, N_SAT))
    rewards = []
    for i in range(N_SAT):
        nbrs = [j for j in range(N_SAT) if avail[i, j]]
        if not nbrs: continue
        nf = np.array([[float(kappa[i,j]), float(cap[i,j]/C_MAX),
                        float(min(rtt[i,j]/RTT_MAX, 1.0)),
                        float(np.clip(kappa[i,j]-kappa_prev[i,j], -1, 1))]
                       for j in nbrs], dtype=np.float32)
        sf = nf.mean(axis=0)
        action, lp, val = agents[i].select_action(sf, nf)
        for idx, j in enumerate(nbrs[:ACTION_DIM]):
            x_new[i, j] = action[idx] * cap[i, j]
        r = compute_reward(i, nbrs, S, cap, kappa, drop, B, x_new, x_prev, DT)
        agents[i].store(sf, nf, action[:ACTION_DIM], r, lp, val, 0.0)
        rewards.append(r)
    return x_new, kappa.copy(), rewards

print("Helpers ready — N_SAT=%d  STATE_DIM=%d  K_ROUNDS=%d" % (N_SAT, STATE_DIM, K_ROUNDS))


# ══════════════════════════════════════════════════════════════════════════════
# CELL 9 — Train TCP-CUBIC
# ══════════════════════════════════════════════════════════════════════════════

OUT = "results/tcp_cubic/metrics.json"
net = ISLNetwork(seed=42)
W   = np.ones((N_SAT, N_SAT)) * 0.1
TSL = np.zeros((N_SAT, N_SAT))
x_prev = np.zeros((N_SAT, N_SAT))
m = blank()
CUBIC_C = 0.4; CUBIC_BETA = 0.7; W_MAX = 1.0; T_K = 5.0

for k in range(K_ROUNDS):
    rewards = []; rows = []
    for _ in range(STEPS_PER_ROUND):
        obs   = net.step(x_prev)
        kappa = obs["kappa"]; cap = obs["cap"]
        avail = obs["avail"]; S   = obs["S"]; drop = obs["drop"]
        x_new = np.zeros((N_SAT, N_SAT))
        for i in range(N_SAT):
            for j in range(N_SAT):
                if not avail[i, j]: continue
                tt = TSL[i,j] + DT
                wc = float(np.clip(CUBIC_C*(tt-T_K)**3 + W_MAX, 0.01, W_MAX))
                if kappa[i,j] > 0.8:
                    wc = W[i,j] * CUBIC_BETA; tt = 0.0
                W[i,j] = wc; TSL[i,j] = tt
                x_new[i,j] = wc * cap[i,j]
        step_r = []
        for i in range(N_SAT):
            nbrs = [j for j in range(N_SAT) if avail[i,j]]
            if nbrs:
                step_r.append(compute_reward(i, nbrs, S, cap, kappa, drop, B, x_new, x_prev, DT))
        rewards.append(np.mean(step_r) if step_r else 0)
        rows.append(collect(obs, x_new, x_prev))
        x_prev = x_new.copy()
    record(m, k, rewards, rows, [])
    if k % LOG_EVERY == 0:
        print(f"TCP  Round {k:4d} | tput={m['throughput_gbps'][-1]:.1f}G  drop={m['drop_rate'][-1]:.3f}")
    if k % 500 == 0: save(m, OUT)

save(m, OUT)
print("TCP-CUBIC done →", OUT)


# ══════════════════════════════════════════════════════════════════════════════
# CELL 10 — Train Ind-PPO
# ══════════════════════════════════════════════════════════════════════════════

OUT    = "results/indppo_isl/metrics.json"
net    = ISLNetwork(seed=42)
agents = make_flat()
x_prev = np.zeros((N_SAT, N_SAT))
kp     = np.zeros((N_SAT, N_SAT))
m      = blank()

for k in range(K_ROUNDS):
    rewards = []; rows = []
    for _ in range(STEPS_PER_ROUND):
        obs = net.step(x_prev)
        x_new, kp, step_r = flat_step(agents, obs, x_prev, kp)
        rows.append(collect(obs, x_new, x_prev))
        rewards.extend(step_r)
        x_prev = x_new.copy()
    losses = [ag.update() for ag in agents]
    record(m, k, rewards, rows, losses)
    if k % LOG_EVERY == 0:
        print(f"IndPPO Round {k:4d} | reward={m['mean_reward'][-1]:+.4f}  tput={m['throughput_gbps'][-1]:.1f}G")
    if k % 500 == 0: save(m, OUT)

save(m, OUT)
print("Ind-PPO done →", OUT)


# ══════════════════════════════════════════════════════════════════════════════
# CELL 11 — Train FedAvg
# ══════════════════════════════════════════════════════════════════════════════

OUT    = "results/fedavg/metrics.json"
net    = ISLNetwork(seed=42)
agents = make_flat()
x_prev = np.zeros((N_SAT, N_SAT))
kp     = np.zeros((N_SAT, N_SAT))
m      = blank()

for k in range(K_ROUNDS):
    rewards = []; rows = []
    for _ in range(STEPS_PER_ROUND):
        obs = net.step(x_prev)
        x_new, kp, step_r = flat_step(agents, obs, x_prev, kp)
        rows.append(collect(obs, x_new, x_prev))
        rewards.extend(step_r)
        x_prev = x_new.copy()
    losses = [ag.update() for ag in agents]
    pos    = _sat_positions(net.t)
    direct = visible_satellites(pos)
    trunk  = fedavg_round(agents, direct)
    if trunk: broadcast(agents, trunk, direct)
    record(m, k, rewards, rows, losses)
    if k % LOG_EVERY == 0:
        print(f"FedAvg Round {k:4d} | reward={m['mean_reward'][-1]:+.4f}  contact={len(direct)}")
    if k % 500 == 0: save(m, OUT)

save(m, OUT)
print("FedAvg done →", OUT)


# ══════════════════════════════════════════════════════════════════════════════
# CELL 12 — Train CA-PFedAvg
# ══════════════════════════════════════════════════════════════════════════════

OUT    = "results/capfedavg/metrics.json"
net    = ISLNetwork(seed=42)
agents = make_flat()
x_prev = np.zeros((N_SAT, N_SAT))
kp     = np.zeros((N_SAT, N_SAT))
m      = blank()

for k in range(K_ROUNDS):
    rewards = []; rows = []
    for _ in range(STEPS_PER_ROUND):
        obs = net.step(x_prev)
        x_new, kp, step_r = flat_step(agents, obs, x_prev, kp)
        rows.append(collect(obs, x_new, x_prev))
        rewards.extend(step_r)
        x_prev = x_new.copy()
    losses  = [ag.update() for ag in agents]
    _, _, avail_g = compute_isl_graph(net.t)
    pos     = _sat_positions(net.t)
    direct  = visible_satellites(pos)
    contact = relay_contact_set(direct, avail_g)
    trunk   = fedavg_round(agents, contact)
    if trunk: broadcast(agents, trunk, contact)
    record(m, k, rewards, rows, losses)
    if k % LOG_EVERY == 0:
        print(f"CAPFed Round {k:4d} | reward={m['mean_reward'][-1]:+.4f}  contact={len(contact)}")
    if k % 500 == 0: save(m, OUT)

save(m, OUT)
print("CA-PFedAvg done →", OUT)


# ══════════════════════════════════════════════════════════════════════════════
# CELL 13 — Train Flat-FRL
# ══════════════════════════════════════════════════════════════════════════════

OUT    = "results/flat_frl/metrics.json"
net    = ISLNetwork(seed=42)
agents = make_flat()
x_prev = np.zeros((N_SAT, N_SAT))
kp     = np.zeros((N_SAT, N_SAT))
m      = blank()

for k in range(K_ROUNDS):
    rewards = []; rows = []
    for _ in range(STEPS_PER_ROUND):
        obs = net.step(x_prev)
        x_new, kp, step_r = flat_step(agents, obs, x_prev, kp)
        rows.append(collect(obs, x_new, x_prev))
        rewards.extend(step_r)
        x_prev = x_new.copy()
    losses  = [ag.update() for ag in agents]
    _, _, avail_g = compute_isl_graph(net.t)
    isl_gossip_round(agents, avail_g)
    pos     = _sat_positions(net.t)
    direct  = visible_satellites(pos)
    contact = relay_contact_set(direct, avail_g)
    trunk   = fedavg_round(agents, contact)
    if trunk: broadcast(agents, trunk, contact)
    record(m, k, rewards, rows, losses)
    if k % LOG_EVERY == 0:
        print(f"FlatFRL Round {k:4d} | reward={m['mean_reward'][-1]:+.4f}  tput={m['throughput_gbps'][-1]:.1f}G")
    if k % 500 == 0: save(m, OUT)

save(m, OUT)
print("Flat-FRL done →", OUT)


# ══════════════════════════════════════════════════════════════════════════════
# CELL 14 — Train GNN-FRL
# ══════════════════════════════════════════════════════════════════════════════

OUT    = "results/gnn_frl/metrics.json"
net    = ISLNetwork(seed=42)
agents = make_gnn()
x_prev = np.zeros((N_SAT, N_SAT))
kp     = np.zeros((N_SAT, N_SAT))
m      = blank()

for k in range(K_ROUNDS):
    rewards = []; rows = []
    for _ in range(STEPS_PER_ROUND):
        obs = net.step(x_prev)
        x_new, kp, step_r = gnn_step(agents, obs, x_prev, kp)
        rows.append(collect(obs, x_new, x_prev))
        rewards.extend(step_r)
        x_prev = x_new.copy()
    losses  = [ag.update() for ag in agents]
    _, _, avail_g = compute_isl_graph(net.t)
    isl_gossip_round(agents, avail_g)
    pos     = _sat_positions(net.t)
    direct  = visible_satellites(pos)
    contact = relay_contact_set(direct, avail_g)
    trunk   = fedavg_round(agents, contact)
    if trunk: broadcast(agents, trunk, contact)
    record(m, k, rewards, rows, losses)
    if k % LOG_EVERY == 0:
        print(f"GNN-FRL Round {k:4d} | reward={m['mean_reward'][-1]:+.4f}  tput={m['throughput_gbps'][-1]:.1f}G  contact={len(contact)}")
    if k % 500 == 0: save(m, OUT)

save(m, OUT)
print("GNN-FRL done →", OUT)


# ══════════════════════════════════════════════════════════════════════════════
# CELL 15 — Disruption Experiment
# ══════════════════════════════════════════════════════════════════════════════

import random as _random

DISRUPT_EVERY = 500
DISRUPT_LEN   = 50
DISRUPT_FRAC  = 0.20


def make_dmask(seed):
    rng   = _random.Random(seed)
    mask  = np.ones((N_SAT, N_SAT))
    edges = [(i, j) for i in range(N_SAT) for j in range(N_SAT) if i != j]
    for i, j in rng.sample(edges, int(len(edges) * DISRUPT_FRAC)):
        mask[i, j] = 0
    return mask


net = ISLNetwork(seed=42)

ag_tcp  = {"W": np.ones((N_SAT,N_SAT))*0.1, "tsl": np.zeros((N_SAT,N_SAT)), "x": np.zeros((N_SAT,N_SAT))}
ag_ind  = make_flat(); x_ind  = np.zeros((N_SAT,N_SAT)); kp_ind  = np.zeros((N_SAT,N_SAT))
ag_flat = make_flat(); x_flat = np.zeros((N_SAT,N_SAT)); kp_flat = np.zeros((N_SAT,N_SAT))
ag_gnn  = make_gnn();  x_gnn  = np.zeros((N_SAT,N_SAT)); kp_gnn  = np.zeros((N_SAT,N_SAT))

m_tcp  = blank(); m_tcp["disrupted"]  = []
m_ind  = blank(); m_ind["disrupted"]  = []
m_flat = blank(); m_flat["disrupted"] = []
m_gnn  = blank(); m_gnn["disrupted"]  = []

CUBIC_C = 0.4; CUBIC_BETA = 0.7; W_MAX = 1.0; T_K = 5.0

for k in range(K_ROUNDS):
    phase     = k % DISRUPT_EVERY
    disrupted = phase < DISRUPT_LEN
    dmask     = make_dmask(k // DISRUPT_EVERY) if disrupted else np.ones((N_SAT, N_SAT))

    _, _, avail_g = compute_isl_graph(net.t)
    pos     = _sat_positions(net.t)
    direct  = visible_satellites(pos)
    contact = relay_contact_set(direct, avail_g)

    # TCP
    tcp_r = []; tcp_rows = []
    for _ in range(STEPS_PER_ROUND):
        obs   = net.step(ag_tcp["x"])
        kappa = obs["kappa"]; cap = obs["cap"]
        avail = obs["avail"] * dmask; S = obs["S"]; drop = obs["drop"]
        x_new = np.zeros((N_SAT, N_SAT))
        for i in range(N_SAT):
            for j in range(N_SAT):
                if not avail[i,j]: continue
                tt = ag_tcp["tsl"][i,j] + DT
                wc = float(np.clip(CUBIC_C*(tt-T_K)**3 + W_MAX, 0.01, W_MAX))
                if kappa[i,j] > 0.8:
                    wc = ag_tcp["W"][i,j] * CUBIC_BETA; tt = 0.0
                ag_tcp["W"][i,j] = wc; ag_tcp["tsl"][i,j] = tt; x_new[i,j] = wc*cap[i,j]
        step_r = [compute_reward(i,[j for j in range(N_SAT) if avail[i,j]],S,cap,kappa,drop,B,x_new,ag_tcp["x"],DT)
                  for i in range(N_SAT) if [j for j in range(N_SAT) if avail[i,j]]]
        tcp_r.append(np.mean(step_r) if step_r else 0)
        tcp_rows.append(collect(obs, x_new, ag_tcp["x"]))
        ag_tcp["x"] = x_new.copy()
    record(m_tcp, k, tcp_r, tcp_rows, []); m_tcp["disrupted"].append(int(disrupted))

    # Ind-PPO
    ind_r = []; ind_rows = []
    for _ in range(STEPS_PER_ROUND):
        obs = net.step(x_ind)
        obs_masked = dict(obs); obs_masked["avail"] = obs["avail"] * dmask
        x_new, kp_ind, step_r = flat_step(ag_ind, obs_masked, x_ind, kp_ind)
        ind_r.extend(step_r); ind_rows.append(collect(obs_masked, x_new, x_ind))
        x_ind = x_new.copy()
    losses = [ag.update() for ag in ag_ind]
    record(m_ind, k, ind_r, ind_rows, losses); m_ind["disrupted"].append(int(disrupted))

    # Flat-FRL
    flat_r = []; flat_rows = []
    for _ in range(STEPS_PER_ROUND):
        obs = net.step(x_flat)
        obs_masked = dict(obs); obs_masked["avail"] = obs["avail"] * dmask
        x_new, kp_flat, step_r = flat_step(ag_flat, obs_masked, x_flat, kp_flat)
        flat_r.extend(step_r); flat_rows.append(collect(obs_masked, x_new, x_flat))
        x_flat = x_new.copy()
    losses = [ag.update() for ag in ag_flat]
    isl_gossip_round(ag_flat, avail_g)
    trunk = fedavg_round(ag_flat, contact)
    if trunk: broadcast(ag_flat, trunk, contact)
    record(m_flat, k, flat_r, flat_rows, losses); m_flat["disrupted"].append(int(disrupted))

    # GNN-FRL
    gnn_r = []; gnn_rows = []
    for _ in range(STEPS_PER_ROUND):
        obs = net.step(x_gnn)
        obs_masked = dict(obs); obs_masked["avail"] = obs["avail"] * dmask
        x_new, kp_gnn, step_r = gnn_step(ag_gnn, obs_masked, x_gnn, kp_gnn)
        gnn_r.extend(step_r); gnn_rows.append(collect(obs_masked, x_new, x_gnn))
        x_gnn = x_new.copy()
    losses = [ag.update() for ag in ag_gnn]
    isl_gossip_round(ag_gnn, avail_g)
    trunk = fedavg_round(ag_gnn, contact)
    if trunk: broadcast(ag_gnn, trunk, contact)
    record(m_gnn, k, gnn_r, gnn_rows, losses); m_gnn["disrupted"].append(int(disrupted))

    if k % LOG_EVERY == 0:
        tag = "[DISRUPT]" if disrupted else "         "
        print(f"Round {k:4d} {tag} | TCP={m_tcp['throughput_gbps'][-1]:.1f}G  Flat={m_flat['throughput_gbps'][-1]:.1f}G  GNN={m_gnn['throughput_gbps'][-1]:.1f}G")
    if k % 500 == 0:
        for n, mm in [("tcp",m_tcp),("indppo",m_ind),("flat_frl",m_flat),("gnn_frl",m_gnn)]:
            save(mm, f"results/disruption/{n}_metrics.json")

for n, mm in [("tcp",m_tcp),("indppo",m_ind),("flat_frl",m_flat),("gnn_frl",m_gnn)]:
    save(mm, f"results/disruption/{n}_metrics.json")
print("Disruption experiment done")


# ══════════════════════════════════════════════════════════════════════════════
# CELL 16 — Handover Experiment
# ══════════════════════════════════════════════════════════════════════════════

import math as _math

GS_LIST_HO = [{"lat":48.8,"lon":2.3}, {"lat":37.4,"lon":-122.1}, {"lat":-33.9,"lon":151.2}]
GS_MAX_KM  = 2500.0
GS_ELEV    = 5.0
EARTH_R    = 6371.0


def _ecef(gs):
    lat = _math.radians(gs["lat"]); lon = _math.radians(gs["lon"])
    return np.array([EARTH_R*_math.cos(lat)*_math.cos(lon),
                     EARTH_R*_math.cos(lat)*_math.sin(lon),
                     EARTH_R*_math.sin(lat)])


def vis_gs(pos_ecef):
    gps = [_ecef(g) for g in GS_LIST_HO]; vis = set()
    for i, sp in enumerate(pos_ecef):
        for gp in gps:
            diff = sp - gp; d = float(np.linalg.norm(diff))
            if d > GS_MAX_KM: continue
            gn   = gp / np.linalg.norm(gp)
            elev = _math.degrees(_math.asin(float(np.clip(np.dot(diff/d, gn), -1, 1))))
            if elev >= GS_ELEV: vis.add(i); break
    return vis


def direct_fedavg(agents, gs_vis):
    if not gs_vis: return 0
    ws  = [agents[i].get_trunk_weights() for i in gs_vis]
    avg = {k: sum(w[k] for w in ws)/len(ws) for k in ws[0]}
    for ag in agents: ag.set_trunk_weights(avg)
    return len(gs_vis)


def blank_ho():
    m = blank(); m["gs_visible_count"] = []; m["fed_participants"] = []
    return m


net = ISLNetwork(seed=42)

ag_fv = make_flat(); x_fv = np.zeros((N_SAT,N_SAT)); kp_fv = np.zeros((N_SAT,N_SAT))
ag_ca = make_flat(); x_ca = np.zeros((N_SAT,N_SAT)); kp_ca = np.zeros((N_SAT,N_SAT))
ag_fl = make_flat(); x_fl = np.zeros((N_SAT,N_SAT)); kp_fl = np.zeros((N_SAT,N_SAT))
ag_gn = make_gnn();  x_gn = np.zeros((N_SAT,N_SAT)); kp_gn = np.zeros((N_SAT,N_SAT))

m_fv = blank_ho(); m_ca = blank_ho(); m_fl = blank_ho(); m_gn = blank_ho()

for k in range(K_ROUNDS):
    _, _, avail_g = compute_isl_graph(net.t)
    pos_ecef = _sat_positions(net.t)
    gs_vis   = vis_gs(pos_ecef)
    contact  = relay_contact_set(gs_vis, avail_g)

    for agents, x_p, kp, m, mode in [
        (ag_fv, x_fv, kp_fv, m_fv, "direct"),
        (ag_ca, x_ca, kp_ca, m_ca, "relay"),
        (ag_fl, x_fl, kp_fl, m_fl, "relay_gossip"),
        (ag_gn, x_gn, kp_gn, m_gn, "relay_gossip_gnn"),
    ]:
        use_gnn = (mode == "relay_gossip_gnn")
        r = []; rows = []
        for _ in range(STEPS_PER_ROUND):
            obs = net.step(x_p)
            if use_gnn:
                x_new, kp_new, step_r = gnn_step(agents, obs, x_p, kp)
            else:
                x_new, kp_new, step_r = flat_step(agents, obs, x_p, kp)
            r.extend(step_r); rows.append(collect(obs, x_new, x_p))
            x_p[:] = x_new; kp[:] = kp_new

        losses = [ag.update() for ag in agents]
        if mode == "direct":
            n_fed = direct_fedavg(agents, gs_vis)
        elif mode == "relay":
            trunk = fedavg_round(agents, contact); n_fed = len(contact)
            if trunk: broadcast(agents, trunk, contact)
        else:
            isl_gossip_round(agents, avail_g)
            trunk = fedavg_round(agents, contact); n_fed = len(contact)
            if trunk: broadcast(agents, trunk, contact)

        record(m, k, r, rows, losses)
        m["gs_visible_count"].append(len(gs_vis))
        m["fed_participants"].append(n_fed)

    if k % LOG_EVERY == 0:
        print(f"HO Round {k:4d} | GS={len(gs_vis)} relay={len(contact)} | FedAvg={m_fv['throughput_gbps'][-1]:.1f}G  CA={m_ca['throughput_gbps'][-1]:.1f}G  GNN={m_gn['throughput_gbps'][-1]:.1f}G")
    if k % 500 == 0:
        for n, mm in [("fedavg",m_fv),("capfedavg",m_ca),("flat_frl",m_fl),("gnn_frl",m_gn)]:
            save(mm, f"results/handover/{n}_metrics.json")

for n, mm in [("fedavg",m_fv),("capfedavg",m_ca),("flat_frl",m_fl),("gnn_frl",m_gn)]:
    save(mm, f"results/handover/{n}_metrics.json")
print("Handover experiment done")


# ══════════════════════════════════════════════════════════════════════════════
# CELL 17 — Ablation Study
# ══════════════════════════════════════════════════════════════════════════════

VARIANTS = ["full", "no_gossip", "no_relay", "flat", "ind"]

abl_ag = {
    "full":      make_gnn(),
    "no_gossip": make_gnn(),
    "no_relay":  make_gnn(),
    "flat":      make_flat(),
    "ind":       make_flat(),
}
abl_x  = {n: np.zeros((N_SAT, N_SAT)) for n in VARIANTS}
abl_kp = {n: np.zeros((N_SAT, N_SAT)) for n in VARIANTS}
abl_m  = {n: blank() for n in VARIANTS}

net2 = ISLNetwork(seed=42)

for k in range(K_ROUNDS):
    _, _, avail_g = compute_isl_graph(net2.t)
    pos     = _sat_positions(net2.t)
    direct  = visible_satellites(pos)
    c_relay = relay_contact_set(direct, avail_g)
    c_dir   = set(direct)

    for name in VARIANTS:
        agents  = abl_ag[name]
        use_gnn = name in ("full", "no_gossip", "no_relay")
        r = []; rows = []
        for _ in range(STEPS_PER_ROUND):
            obs = net2.step(abl_x[name])
            if use_gnn:
                x_new, kp_new, step_r = gnn_step(agents, obs, abl_x[name], abl_kp[name])
            else:
                x_new, kp_new, step_r = flat_step(agents, obs, abl_x[name], abl_kp[name])
            r.extend(step_r); rows.append(collect(obs, x_new, abl_x[name]))
            abl_x[name][:] = x_new; abl_kp[name][:] = kp_new

        losses = [ag.update() for ag in agents]

        if name == "full":
            isl_gossip_round(agents, avail_g)
            trunk = fedavg_round(agents, c_relay)
            if trunk: broadcast(agents, trunk, c_relay)
        elif name == "no_gossip":
            trunk = fedavg_round(agents, c_relay)
            if trunk: broadcast(agents, trunk, c_relay)
        elif name == "no_relay":
            isl_gossip_round(agents, avail_g)
            trunk = fedavg_round(agents, c_dir)
            if trunk: broadcast(agents, trunk, c_dir)
        elif name == "flat":
            isl_gossip_round(agents, avail_g)
            trunk = fedavg_round(agents, c_relay)
            if trunk: broadcast(agents, trunk, c_relay)
        # "ind": no federation

        record(abl_m[name], k, r, rows, losses)

    if k % LOG_EVERY == 0:
        print(f"Ablation Round {k:4d} | " + "  ".join(f"{n}={abl_m[n]['throughput_gbps'][-1]:.1f}G" for n in VARIANTS))
    if k % 500 == 0:
        for n in VARIANTS: save(abl_m[n], f"results/ablation/{n}_metrics.json")

for n in VARIANTS: save(abl_m[n], f"results/ablation/{n}_metrics.json")
print("Ablation study done")


# ══════════════════════════════════════════════════════════════════════════════
# CELL 18 — All Plots (run after all experiments complete)
# ══════════════════════════════════════════════════════════════════════════════

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

FIGS = "results/figures"
os.makedirs(FIGS, exist_ok=True)
SMOOTH = 50
WIN    = 200

METHODS = {
    "TCP-CUBIC":  "results/tcp_cubic",
    "FedAvg":     "results/fedavg",
    "Ind-PPO":    "results/indppo_isl",
    "CA-PFedAvg": "results/capfedavg",
    "Flat-FRL":   "results/flat_frl",
    "GNN-FRL":    "results/gnn_frl",
}
COLORS = ["#7f7f7f", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
LABELS = list(METHODS.keys())
RL     = [l for l in LABELS if l != "TCP-CUBIC"]
RC     = COLORS[1:]


def _load(path):
    p = Path(path) / "metrics.json"
    return json.load(open(p)) if p.exists() else None


def _loadf(path):
    p = Path(path)
    return json.load(open(p)) if p.exists() else None


def _sm(arr, w=SMOOTH):
    arr = np.array(arr, dtype=float)
    return np.convolve(arr, np.ones(w)/w, mode="valid") if len(arr) >= w else arr


def _savefig(fig, name):
    fig.savefig(f"{FIGS}/{name}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(f"{FIGS}/{name}.png", bbox_inches="tight", dpi=150)
    print(f"  saved {name}")


data = {}
for lbl, path in METHODS.items():
    d = _load(path)
    if d:
        data[lbl] = d
        print(f"[OK] {lbl:25s} rounds={len(d['round'])}")
    else:
        print(f"[--] {lbl}")

# Fig 1 — Learning curves
fig, ax = plt.subplots(figsize=(9, 4))
if "TCP-CUBIC" in data:
    ax.axhline(np.mean(data["TCP-CUBIC"]["mean_reward"]), color=COLORS[0], ls="--", lw=1.4, label="TCP-CUBIC (ref)")
for lbl, clr in zip(RL, RC):
    if lbl not in data: continue
    y = _sm(data[lbl]["mean_reward"])
    ax.plot(np.arange(len(y)), y, color=clr, lw=1.6, label=lbl)
ax.set_xlabel("Training Round"); ax.set_ylabel("Mean Reward")
ax.set_title("Learning Curves"); ax.legend(fontsize=9, ncol=2); ax.grid(alpha=0.3)
_savefig(fig, "fig01_learning_curves"); plt.close(fig)

# Fig 2 — Throughput
fig, ax = plt.subplots(figsize=(9, 4))
if "TCP-CUBIC" in data:
    ax.axhline(np.mean(data["TCP-CUBIC"]["throughput_gbps"]), color=COLORS[0], ls="--", lw=1.4, label="TCP-CUBIC (ref)")
for lbl, clr in zip(RL, RC):
    if lbl not in data: continue
    y = _sm(data[lbl]["throughput_gbps"])
    ax.plot(np.arange(len(y)), y, color=clr, lw=1.6, label=lbl)
ax.set_xlabel("Training Round"); ax.set_ylabel("Throughput (Gbps)")
ax.set_title("Network Throughput"); ax.legend(fontsize=9, ncol=2); ax.grid(alpha=0.3)
_savefig(fig, "fig02_throughput"); plt.close(fig)

# Fig 3 — Drop rate
fig, ax = plt.subplots(figsize=(9, 4))
if "TCP-CUBIC" in data:
    ax.axhline(np.mean(data["TCP-CUBIC"]["drop_rate"]), color=COLORS[0], ls="--", lw=1.4, label="TCP-CUBIC (ref)")
for lbl, clr in zip(RL, RC):
    if lbl not in data: continue
    y = _sm(data[lbl]["drop_rate"])
    ax.plot(np.arange(len(y)), y, color=clr, lw=1.6, label=lbl)
ax.set_xlabel("Training Round"); ax.set_ylabel("Packet Drop Rate")
ax.set_title("Packet Drop Rate"); ax.legend(fontsize=9, ncol=2); ax.grid(alpha=0.3)
_savefig(fig, "fig03_drop_rate"); plt.close(fig)

# Fig 4 — Congestion kappa
fig, ax = plt.subplots(figsize=(9, 4))
if "TCP-CUBIC" in data:
    ax.axhline(np.mean(data["TCP-CUBIC"]["mean_kappa"]), color=COLORS[0], ls="--", lw=1.4, label="TCP-CUBIC (ref)")
for lbl, clr in zip(RL, RC):
    if lbl not in data: continue
    y = _sm(data[lbl]["mean_kappa"])
    ax.plot(np.arange(len(y)), y, color=clr, lw=1.6, label=lbl)
ax.set_xlabel("Training Round"); ax.set_ylabel("Mean Queue Occupancy k")
ax.set_title("Congestion Level"); ax.legend(fontsize=9, ncol=2); ax.grid(alpha=0.3)
_savefig(fig, "fig04_kappa"); plt.close(fig)

# Fig 5 — Link utilization
fig, ax = plt.subplots(figsize=(9, 4))
if "TCP-CUBIC" in data:
    ax.axhline(np.mean(data["TCP-CUBIC"]["link_utilization"]), color=COLORS[0], ls="--", lw=1.4, label="TCP-CUBIC (ref)")
for lbl, clr in zip(RL, RC):
    if lbl not in data: continue
    y = _sm(data[lbl]["link_utilization"])
    ax.plot(np.arange(len(y)), y, color=clr, lw=1.6, label=lbl)
ax.set_xlabel("Training Round"); ax.set_ylabel("Link Utilization")
ax.set_title("ISL Link Utilization"); ax.legend(fontsize=9, ncol=2); ax.grid(alpha=0.3)
_savefig(fig, "fig05_utilization"); plt.close(fig)

# Fig 6 — P99 tail latency
fig, ax = plt.subplots(figsize=(9, 4))
for lbl, clr in zip(LABELS, COLORS):
    if lbl not in data or "p99_rtt" not in data[lbl]: continue
    y = _sm(data[lbl]["p99_rtt"])
    ax.plot(np.arange(len(y)), y, color=clr, lw=1.6, label=lbl)
ax.set_xlabel("Training Round"); ax.set_ylabel("P99 RTT (seconds)")
ax.set_title("Tail Latency (P99 RTT)"); ax.legend(fontsize=9, ncol=2); ax.grid(alpha=0.3)
_savefig(fig, "fig06_p99_latency"); plt.close(fig)

# Fig 7 — Steady-state bar chart
fig, axes = plt.subplots(1, 4, figsize=(17, 4))
for metric, ylabel, ax in [
    ("throughput_gbps",  "Throughput (Gbps)", axes[0]),
    ("drop_rate",        "Drop Rate",         axes[1]),
    ("link_utilization", "Link Utilization",  axes[2]),
    ("p99_rtt",          "P99 RTT (s)",       axes[3]),
]:
    vals, errs, lbls, clrs = [], [], [], []
    for lbl, clr in zip(LABELS, COLORS):
        if lbl not in data or metric not in data[lbl]: continue
        arr  = np.array(data[lbl][metric])
        tail = arr[-WIN:] if len(arr) >= WIN else arr
        vals.append(tail.mean()); errs.append(tail.std())
        lbls.append(lbl); clrs.append(clr)
    x = np.arange(len(vals))
    ax.bar(x, vals, yerr=errs, capsize=4, color=clrs, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(lbls, fontsize=7, rotation=35, ha="right")
    ax.set_ylabel(ylabel, fontsize=9); ax.grid(axis="y", alpha=0.3)
fig.suptitle(f"Steady-State Performance (last {WIN} rounds)"); fig.tight_layout()
_savefig(fig, "fig07_steady_state"); plt.close(fig)

# Fig 8 — Jain's Fairness Index
def jain(arr):
    x = np.array(arr, dtype=float); n = len(x)
    return float(x.sum()**2 / (n * (x**2).sum() + 1e-12)) if n > 0 else 0.0

fig, ax = plt.subplots(figsize=(9, 4))
has = False
for lbl, clr in zip(LABELS, COLORS):
    if lbl not in data or "throughput_gbps" not in data[lbl]: continue
    t  = np.array(data[lbl]["throughput_gbps"]); fw = WIN // 2
    fv = [jain(t[i:i+fw]) for i in range(0, len(t)-fw, fw//5)]
    if fv:
        ax.plot(np.linspace(0, len(t), len(fv)), fv, color=clr, lw=1.5, label=lbl)
        has = True
if has:
    ax.set_xlabel("Training Round"); ax.set_ylabel("Jain's Fairness Index")
    ax.set_ylim(0, 1.05); ax.set_title("Throughput Fairness (J=1 is perfect)")
    ax.legend(fontsize=9, ncol=2); ax.grid(alpha=0.3)
_savefig(fig, "fig08_fairness"); plt.close(fig)

# Fig 9 — Convergence speed
rl_d = [l for l in RL if l in data]
if rl_d:
    best   = max(np.mean(data[l]["mean_reward"][-200:]) for l in rl_d)
    THRESH = 0.90 * best
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (lbl, clr) in enumerate(zip(LABELS, COLORS)):
        if lbl not in data: continue
        r  = np.array(data[lbl]["mean_reward"])
        cr = np.where(r >= THRESH)[0]
        cv = int(cr[0]) if len(cr) > 0 else len(r)
        ax.barh(i, cv, color=clr, alpha=0.8)
        ax.text(cv + 20, i, str(cv), va="center", fontsize=9)
    ax.set_yticks(range(len(LABELS))); ax.set_yticklabels(LABELS, fontsize=10)
    ax.set_xlabel("Rounds to 90% of best reward")
    ax.set_title("Convergence Speed"); ax.axvline(3000, ls="--", color="k", alpha=0.3)
    ax.grid(axis="x", alpha=0.3)
    _savefig(fig, "fig09_convergence"); plt.close(fig)

# Fig 10 — Disruption resilience
DM = {"TCP-CUBIC":"results/disruption/tcp_metrics.json",
      "Ind-PPO":"results/disruption/indppo_metrics.json",
      "Flat-FRL":"results/disruption/flat_frl_metrics.json",
      "GNN-FRL":"results/disruption/gnn_frl_metrics.json"}
DC  = ["#7f7f7f","#ff7f0e","#d62728","#9467bd"]
dm  = {lbl: _loadf(p) for lbl, p in DM.items() if _loadf(p)}
if dm:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for metric, ylabel, ax in [("throughput_gbps","Throughput (Gbps)",axes[0]),
                                ("drop_rate","Drop Rate",axes[1])]:
        for lbl, clr in zip(DM.keys(), DC):
            if lbl not in dm: continue
            y = _sm(dm[lbl][metric], w=20)
            ax.plot(np.arange(len(y)), y, color=clr, lw=1.5, label=lbl)
        for s in range(0, 3000, 500):
            ax.axvspan(s, s + 50, alpha=0.12, color="red")
        ax.set_xlabel("Round"); ax.set_ylabel(ylabel)
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
    axes[0].set_title("Throughput Under Disruptions (red=outage)")
    axes[1].set_title("Drop Rate Under Disruptions")
    fig.tight_layout()
    _savefig(fig, "fig10_disruption"); plt.close(fig)

# Fig 11 — Handover
HM = {"FedAvg":"results/handover/fedavg_metrics.json",
      "CA-PFedAvg":"results/handover/capfedavg_metrics.json",
      "Flat-FRL":"results/handover/flat_frl_metrics.json",
      "GNN-FRL":"results/handover/gnn_frl_metrics.json"}
HC  = ["#1f77b4","#2ca02c","#d62728","#9467bd"]
hm  = {lbl: _loadf(p) for lbl, p in HM.items() if _loadf(p)}
if hm:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for lbl, clr in zip(HM.keys(), HC):
        if lbl not in hm: continue
        y = _sm(hm[lbl]["throughput_gbps"], w=20)
        axes[0].plot(np.arange(len(y)), y, color=clr, lw=1.5, label=lbl)
    axes[0].set_title("Throughput During Handover"); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    axes[0].set_xlabel("Round"); axes[0].set_ylabel("Throughput (Gbps)")
    for lbl, clr in zip(HM.keys(), HC):
        if lbl not in hm or "fed_participants" not in hm[lbl]: continue
        y = _sm(hm[lbl]["fed_participants"], w=20)
        axes[1].plot(np.arange(len(y)), y, color=clr, lw=1.5, label=lbl)
    axes[1].set_title("Federation Participants"); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    axes[1].set_xlabel("Round"); axes[1].set_ylabel("Satellites in Federation")
    if "FedAvg" in hm and "gs_visible_count" in hm["FedAvg"]:
        vis = np.array(hm["FedAvg"]["gs_visible_count"]); y = _sm(vis, w=20)
        axes[2].fill_between(np.arange(len(y)), y, alpha=0.4, color="#1f77b4", label="Direct only")
    if "CA-PFedAvg" in hm and "fed_participants" in hm["CA-PFedAvg"]:
        rel = np.array(hm["CA-PFedAvg"]["fed_participants"]); y2 = _sm(rel, w=20)
        axes[2].fill_between(np.arange(len(y2)), y2, alpha=0.4, color="#2ca02c", label="Relay-expanded")
    axes[2].set_title("Direct vs Relay Contact"); axes[2].legend(fontsize=9); axes[2].grid(alpha=0.3)
    axes[2].set_xlabel("Round"); axes[2].set_ylabel("Satellite Count")
    fig.suptitle("Handover: FedAvg Fails in Eclipse, CA-PFedAvg Survives via Relay")
    fig.tight_layout()
    _savefig(fig, "fig11_handover"); plt.close(fig)

# Fig 12 — Ablation
ABL = {"full":"GNN-FRL\n(Full)","no_gossip":"No\nGossip",
       "no_relay":"No\nRelay","flat":"Flat\nState","ind":"No\nFed"}
AC  = ["#9467bd","#17becf","#bcbd22","#d62728","#ff7f0e"]
abm = {n: _loadf(f"results/ablation/{n}_metrics.json") for n in ABL}
abm = {n: v for n, v in abm.items() if v}
if abm:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for metric, ylabel, ax in [("throughput_gbps","Throughput (Gbps)",axes[0]),
                                ("drop_rate","Drop Rate",axes[1]),
                                ("link_utilization","Link Util",axes[2])]:
        vals, errs, lbls, clrs = [], [], [], []
        for n, clr in zip(ABL.keys(), AC):
            if n not in abm: continue
            arr  = np.array(abm[n][metric])
            tail = arr[-WIN:] if len(arr) >= WIN else arr
            vals.append(tail.mean()); errs.append(tail.std())
            lbls.append(ABL[n]); clrs.append(clr)
        x = np.arange(len(vals))
        ax.bar(x, vals, yerr=errs, capsize=4, color=clrs, alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(lbls, fontsize=9)
        ax.set_ylabel(ylabel); ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Ablation Study — Component Contribution"); fig.tight_layout()
    _savefig(fig, "fig12_ablation"); plt.close(fig)

# Summary table
print("\n" + "="*80)
print(f"{'Method':<22} {'Tput(Gbps)':>12} {'Drop':>8} {'Kappa':>8} {'Util':>8} {'P99-RTT':>10}")
print("="*80)
rows = []
for lbl in LABELS:
    if lbl not in data: continue
    def tm(key):
        arr = np.array(data[lbl].get(key, [0]))
        return float(arr[-WIN:].mean() if len(arr) >= WIN else arr.mean())
    row = [lbl, tm("throughput_gbps"), tm("drop_rate"), tm("mean_kappa"), tm("link_utilization"), tm("p99_rtt")]
    print(f"{lbl:<22} {row[1]:>12.2f} {row[2]:>8.4f} {row[3]:>8.4f} {row[4]:>8.4f} {row[5]:>10.5f}")
    rows.append(row)
with open(f"{FIGS}/table1_summary.csv", "w") as f:
    f.write("Method,Throughput_Gbps,Drop_Rate,Mean_Kappa,Link_Util,P99_RTT_s\n")
    for r in rows:
        f.write(",".join(str(v) for v in r) + "\n")
print("="*80)
print(f"\n12 figures + table1_summary.csv saved to {FIGS}/")


# ══════════════════════════════════════════════════════════════════════════════
# CELL 19 — Save to Google Drive
# ══════════════════════════════════════════════════════════════════════════════

import shutil
DRIVE = "/content/drive/MyDrive/Rano4_results"
shutil.copytree("results", DRIVE, dirs_exist_ok=True)
print("All results saved to", DRIVE)
