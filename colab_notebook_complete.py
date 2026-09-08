# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  COMPLETE COLAB NOTEBOOK — FRL for ISL Congestion Control                  ║
# ║  Copy each CELL into a separate Colab cell                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 1 — Mount Drive + Install
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
from google.colab import drive
drive.mount('/content/drive')
import os
os.makedirs('/content/drive/MyDrive/Rano4_results', exist_ok=True)
!pip install skyfield torch numpy matplotlib -q
print("Done")
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 2 — Write sim package: constellation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CELL2 = r"""
import os, math, numpy as np
os.makedirs('sim/env', exist_ok=True)
os.makedirs('sim/agents', exist_ok=True)
os.makedirs('sim/fed', exist_ok=True)
os.makedirs('sim/utils', exist_ok=True)
for d in ['sim','sim/env','sim/agents','sim/fed','sim/utils']:
    open(f'{d}/__init__.py','a').close()

# ── constellation.py ──────────────────────────────────────────────────────────
open('sim/env/constellation.py','w').write(r'''
import numpy as np, math

N_PLANES     = 6
N_PER_PLANE  = 11
N_SAT        = N_PLANES * N_PER_PLANE   # 66
ALT_KM       = 550.0
INC_DEG      = 53.0
EARTH_R_KM   = 6371.0
MU           = 3.986004418e14           # m^3/s^2
MAX_ISL_KM   = 4500.0

_R   = EARTH_R_KM + ALT_KM
_T   = 2 * math.pi * math.sqrt((_R * 1e3)**3 / MU)   # orbital period (s)
_OM  = 2 * math.pi / _T                               # mean motion (rad/s)


def _sat_positions(t_sec):
    """Return (N_SAT, 3) ECEF positions in km at time t_sec."""
    pos = np.zeros((N_SAT, 3))
    inc = math.radians(INC_DEG)
    for p in range(N_PLANES):
        raan = 2 * math.pi * p / N_PLANES
        for s in range(N_PER_PLANE):
            # Walker Delta: phasing offset = pi/N_PLANES * plane_index
            M0   = 2 * math.pi * s / N_PER_PLANE + math.pi / N_PLANES * p
            M    = (M0 + _OM * t_sec) % (2 * math.pi)
            # position in orbital plane
            x_orb = _R * math.cos(M)
            y_orb = _R * math.sin(M)
            # rotate to ECI
            cos_r, sin_r = math.cos(raan), math.sin(raan)
            cos_i, sin_i = math.cos(inc),  math.sin(inc)
            x_eci = cos_r * x_orb - sin_r * cos_i * y_orb
            y_eci = sin_r * x_orb + cos_r * cos_i * y_orb
            z_eci = sin_i * y_orb
            # ECI ≈ ECEF (ignore Earth rotation for simplicity; negligible for LEO RL)
            idx = p * N_PER_PLANE + s
            pos[idx] = [x_eci, y_eci, z_eci]
    return pos


def compute_isl_graph(t_sec):
    """Returns dist (N,N), cap_base (N,N), avail (N,N bool)."""
    pos   = _sat_positions(t_sec)
    dist  = np.zeros((N_SAT, N_SAT))
    avail = np.zeros((N_SAT, N_SAT), dtype=bool)
    for i in range(N_SAT):
        for j in range(i+1, N_SAT):
            d = float(np.linalg.norm(pos[i] - pos[j]))
            dist[i,j] = dist[j,i] = d
            if d <= MAX_ISL_KM:
                avail[i,j] = avail[j,i] = True
    return dist, None, avail
''')
print("constellation.py written")
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 3 — Write sim package: network + utils
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CELL3 = r"""
# ── network.py ────────────────────────────────────────────────────────────────
open('sim/env/network.py','w').write(r'''
import numpy as np
from .constellation import N_SAT, compute_isl_graph, _sat_positions, MAX_ISL_KM

DT = 1.0        # seconds per step
B  = 1e8        # buffer size (bits)

C_LEVELS = [10e9, 5e9, 2e9, 1e9]   # AMC capacity levels (bps)
RTT_BASE = 0.02                     # 20 ms baseline RTT


class ISLNetwork:
    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)
        self.t   = 0.0
        self.kappa = np.zeros((N_SAT, N_SAT))   # queue occupancy [0,1]
        self.S     = np.zeros((N_SAT, N_SAT))   # served bits/step

    def step(self, x_alloc):
        """
        Advance one DT step.
        x_alloc: (N_SAT, N_SAT) requested rate allocation (bps)
        Returns obs dict.
        """
        self.t += DT
        _, _, avail = compute_isl_graph(self.t)
        pos  = _sat_positions(self.t)

        # AMC: capacity depends on distance
        cap = np.zeros((N_SAT, N_SAT))
        rtt = np.zeros((N_SAT, N_SAT))
        for i in range(N_SAT):
            for j in range(N_SAT):
                if not avail[i,j]: continue
                d = float(np.linalg.norm(pos[i]-pos[j]))
                frac = d / MAX_ISL_KM
                if   frac < 0.25: cap[i,j] = C_LEVELS[0]
                elif frac < 0.50: cap[i,j] = C_LEVELS[1]
                elif frac < 0.75: cap[i,j] = C_LEVELS[2]
                else:             cap[i,j] = C_LEVELS[3]
                rtt[i,j] = RTT_BASE + 2 * d / (3e5)  # propagation

        # traffic demand: random Poisson arrivals
        demand = self.rng.poisson(cap * 0.7 * DT / 1e9).astype(float) * 1e9

        # queue model
        new_kappa = np.zeros((N_SAT, N_SAT))
        S = np.zeros((N_SAT, N_SAT))
        drop = np.zeros((N_SAT, N_SAT))
        for i in range(N_SAT):
            for j in range(N_SAT):
                if not avail[i,j]: continue
                buf      = self.kappa[i,j] * B + demand[i,j]
                served   = min(x_alloc[i,j] * DT, buf, cap[i,j] * DT)
                buf     -= served
                overflow = max(buf - B, 0)
                drop[i,j]     = overflow
                new_kappa[i,j] = max(0, buf - overflow) / B
                S[i,j]        = served

        self.kappa = new_kappa
        self.S     = S
        return dict(kappa=new_kappa, cap=cap, avail=avail.astype(float),
                    rtt=rtt, S=S, drop=drop)
''')

# ── utils/state.py ────────────────────────────────────────────────────────────
open('sim/utils/state.py','w').write(r'''
import numpy as np

STATE_DIM = 18   # 4 neighbors × 4 features + 2 global
C_MAX   = 10e9
RTT_MAX = 0.1

def build_state(sat_id, neighbors, kappa, cap, avail, rtt, kappa_prev, drop):
    feats = []
    for j in neighbors[:4]:
        phi = float(np.clip(kappa[sat_id,j] - kappa_prev[sat_id,j], -1, 1))
        feats += [float(kappa[sat_id,j]),
                  float(cap[sat_id,j] / C_MAX),
                  float(min(rtt[sat_id,j] / RTT_MAX, 1.0)),
                  phi]
    while len(feats) < 16:
        feats += [0.0, 0.0, 0.0, 0.0]
    # 2 global features
    mask = avail > 0
    feats.append(float(kappa[mask].mean()) if mask.any() else 0.0)
    feats.append(float(drop[mask].mean())  if mask.any() else 0.0)
    return np.array(feats[:STATE_DIM], dtype=np.float32)
''')

# ── utils/reward.py ───────────────────────────────────────────────────────────
open('sim/utils/reward.py','w').write(r'''
import numpy as np

def compute_reward(sat_id, neighbors, S, cap, kappa, drop, B, x_new, x_prev, DT):
    r = 0.0
    for j in neighbors:
        c = cap[sat_id, j]
        if c < 1: continue
        tput   =  S[sat_id, j] / (c * DT + 1e-9)
        cong   = -kappa[sat_id, j]
        drp    = -drop[sat_id, j] / (B + 1e-9)
        chg    = -abs(x_new[sat_id,j] - x_prev[sat_id,j]) / (c + 1e-9) * 0.1
        r += tput + cong + drp + chg
    return r / (len(neighbors) + 1e-9)
''')
print("network.py + utils written")
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 4 — Write sim package: PPO agent
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CELL4 = r"""
open('sim/agents/ppo.py','w').write(r'''
import torch, torch.nn as nn, numpy as np
from torch.distributions import Beta

GAMMA=0.99; LAM=0.95; CLIP=0.2; LR=3e-4; EPOCHS=4; BATCH=64
DELTA_THRESH = 1e-4

class _Trunk(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 64),        nn.ReLU())
    def forward(self, x): return self.net(x)

class _Head(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.alpha = nn.Linear(64, action_dim)
        self.beta  = nn.Linear(64, action_dim)
        self.value = nn.Linear(64, 1)
    def forward(self, h):
        a = torch.clamp(torch.softplus(self.alpha(h)) + 1.0, 1.0, 10.0)
        b = torch.clamp(torch.softplus(self.beta(h))  + 1.0, 1.0, 10.0)
        v = self.value(h)
        return a, b, v


class PPOAgent:
    def __init__(self, state_dim, action_dim, sat_id):
        self.sat_id = sat_id
        self.action_dim = action_dim
        self.trunk = _Trunk(state_dim)
        self.head  = _Head(action_dim)
        self.opt   = torch.optim.Adam(
            list(self.trunk.parameters()) + list(self.head.parameters()), lr=LR)
        self.buf   = []

    def get_trunk_weights(self):
        return {k: v.clone() for k, v in self.trunk.state_dict().items()}

    def set_trunk_weights(self, w):
        self.trunk.load_state_dict(w)

    def get_weights(self):
        d = {}
        d.update({f"trunk.{k}": v.clone() for k,v in self.trunk.state_dict().items()})
        d.update({f"head.{k}":  v.clone() for k,v in self.head.state_dict().items()})
        return d

    def set_weights(self, w):
        self.trunk.load_state_dict({k[6:]: v for k,v in w.items() if k.startswith("trunk.")})
        self.head.load_state_dict( {k[5:]: v for k,v in w.items() if k.startswith("head.")})

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
        S,A,R,LP,V,D = map(np.array, zip(*self.buf))
        self.buf = []
        # GAE
        adv = np.zeros_like(R); last = 0
        for t in reversed(range(len(R))):
            nv   = V[t+1] if t+1 < len(V) else 0
            delta= R[t] + GAMMA*(1-D[t])*nv - V[t]
            last = delta + GAMMA*LAM*(1-D[t])*last
            adv[t] = last
        ret = adv + V
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        S  = torch.FloatTensor(S)
        A  = torch.FloatTensor(A)
        LP = torch.FloatTensor(LP)
        adv= torch.FloatTensor(adv)
        ret= torch.FloatTensor(ret)
        losses = []
        for _ in range(EPOCHS):
            idx = np.random.permutation(len(S))
            for start in range(0, len(S), BATCH):
                b = idx[start:start+BATCH]
                h     = self.trunk(S[b])
                al,be,v = self.head(h)
                dist  = Beta(al, be)
                nlp   = dist.log_prob(torch.clamp(A[b],1e-6,1-1e-6)).sum(-1)
                ratio = torch.exp(nlp - LP[b])
                s1    = ratio * adv[b]
                s2    = torch.clamp(ratio,1-CLIP,1+CLIP) * adv[b]
                pl    = -torch.min(s1,s2).mean()
                vl    = (v.squeeze()-ret[b]).pow(2).mean()
                loss  = pl + 0.5*vl
                self.opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.trunk.parameters())+list(self.head.parameters()), 0.5)
                self.opt.step()
                losses.append(float(loss))
        return float(np.mean(losses))
''')
print("ppo.py written")
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 5 — Write GNN-PPO agent
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CELL5 = r"""
open('sim/agents/gnn_ppo.py','w').write(r'''
import torch, torch.nn as nn, numpy as np
from torch.distributions import Beta

GAMMA=0.99; LAM=0.95; CLIP=0.2; LR=3e-4; EPOCHS=4; BATCH=64
NODE_DIM=4   # per-neighbor feature dim

class _GNNTrunk(nn.Module):
    def __init__(self):
        super().__init__()
        self.node_proj = nn.Linear(NODE_DIM, 32)
        self.agg_mlp   = nn.Sequential(
            nn.Linear(32+32, 64), nn.ReLU(),
            nn.Linear(64, 64),   nn.ReLU())
    def forward(self, self_feat, nbr_feats):
        # self_feat: (4,)  nbr_feats: (K,4)
        sf = torch.relu(self.node_proj(self_feat))       # (32,)
        nf = torch.relu(self.node_proj(nbr_feats))       # (K,32)
        agg= nf.mean(0)                                  # (32,)
        return self.agg_mlp(torch.cat([sf, agg]))        # (64,)

class _Head(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.alpha = nn.Linear(64, action_dim)
        self.beta  = nn.Linear(64, action_dim)
        self.value = nn.Linear(64, 1)
    def forward(self, h):
        a = torch.clamp(torch.softplus(self.alpha(h))+1, 1, 10)
        b = torch.clamp(torch.softplus(self.beta(h)) +1, 1, 10)
        return a, b, self.value(h)

class GNNPPOAgent:
    def __init__(self, action_dim, sat_id):
        self.sat_id = sat_id
        self.action_dim = action_dim
        self.trunk = _GNNTrunk()
        self.head  = _Head(action_dim)
        self.opt   = torch.optim.Adam(
            list(self.trunk.parameters())+list(self.head.parameters()), lr=LR)
        self.buf   = []

    def get_trunk_weights(self):
        return {k: v.clone() for k,v in self.trunk.state_dict().items()}

    def set_trunk_weights(self, w):
        self.trunk.load_state_dict(w)

    def get_weights(self):
        d = {}
        d.update({f"trunk.{k}": v.clone() for k,v in self.trunk.state_dict().items()})
        d.update({f"head.{k}":  v.clone() for k,v in self.head.state_dict().items()})
        return d

    def set_weights(self, w):
        self.trunk.load_state_dict({k[6:]: v for k,v in w.items() if k.startswith("trunk.")})
        self.head.load_state_dict( {k[5:]: v for k,v in w.items() if k.startswith("head.")})

    def select_action(self, sf, nf):
        sf_t = torch.FloatTensor(sf)
        nf_t = torch.FloatTensor(nf) if len(nf) else torch.zeros(1, 4)
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
        SFs,NFs,A,R,LP,V,D = zip(*self.buf)
        self.buf = []
        A  = np.array(A); R = np.array(R); LP = np.array(LP)
        V  = np.array(V); D = np.array(D)
        adv = np.zeros_like(R); last = 0
        for t in reversed(range(len(R))):
            nv    = V[t+1] if t+1 < len(V) else 0
            delta = R[t] + GAMMA*(1-D[t])*nv - V[t]
            last  = delta + GAMMA*0.95*(1-D[t])*last
            adv[t]= last
        ret = adv + V
        adv = (adv - adv.mean())/(adv.std()+1e-8)
        A_t  = torch.FloatTensor(A)
        LP_t = torch.FloatTensor(LP)
        adv_t= torch.FloatTensor(adv)
        ret_t= torch.FloatTensor(ret)
        losses = []
        for _ in range(EPOCHS):
            idx = np.random.permutation(len(A))
            for start in range(0, len(A), BATCH):
                b = idx[start:start+BATCH]
                hs = []
                for i in b:
                    sf_t = torch.FloatTensor(SFs[i])
                    nf_t = torch.FloatTensor(NFs[i]) if len(NFs[i]) else torch.zeros(1,4)
                    hs.append(self.trunk(sf_t, nf_t))
                H = torch.stack(hs)
                al,be,vv = self.head(H)
                dist = Beta(al, be)
                nlp  = dist.log_prob(torch.clamp(A_t[b],1e-6,1-1e-6)).sum(-1)
                ratio= torch.exp(nlp - LP_t[b])
                s1   = ratio * adv_t[b]
                s2   = torch.clamp(ratio,1-CLIP,1+CLIP)*adv_t[b]
                pl   = -torch.min(s1,s2).mean()
                vl   = (vv.squeeze()-ret_t[b]).pow(2).mean()
                loss = pl + 0.5*vl
                self.opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.trunk.parameters())+list(self.head.parameters()),0.5)
                self.opt.step()
                losses.append(float(loss))
        return float(np.mean(losses))
''')
print("gnn_ppo.py written")
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 6 — Write fed/ modules
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CELL6 = r"""
open('sim/fed/fedavg.py','w').write(r'''
import numpy as np, math
from ..env.constellation import N_SAT, _sat_positions, EARTH_R_KM, ALT_KM

GS_LIST = [
    {"lat":  48.8, "lon":   2.3},   # Europe
    {"lat":  37.4, "lon":-122.1},   # California
    {"lat": -33.9, "lon": 151.2},   # Sydney
]
GS_MAX_KM = 2500.0
GS_ELEV   = 5.0

def _gs_ecef(gs):
    lat,lon = math.radians(gs["lat"]), math.radians(gs["lon"])
    r = EARTH_R_KM
    return np.array([r*math.cos(lat)*math.cos(lon),
                     r*math.cos(lat)*math.sin(lon),
                     r*math.sin(lat)])

def visible_satellites(pos_ecef):
    """Set of sat indices with direct GS visibility."""
    gs_pos = [_gs_ecef(g) for g in GS_LIST]
    vis = set()
    for i, sp in enumerate(pos_ecef):
        for gp in gs_pos:
            diff = sp - gp; d = float(np.linalg.norm(diff))
            if d > GS_MAX_KM: continue
            gn   = gp / np.linalg.norm(gp)
            elev = math.degrees(math.asin(float(np.clip(np.dot(diff/d, gn),-1,1))))
            if elev >= GS_ELEV: vis.add(i); break
    return vis

def relay_contact_set(direct, avail_g, hops=2):
    """Expand direct set by ISL relay up to `hops` hops."""
    contact = set(direct)
    for _ in range(hops):
        expand = set()
        for i in contact:
            for j in range(N_SAT):
                if avail_g[i,j]: expand.add(j)
        contact |= expand
    return contact

def fedavg_round(agents, contact):
    """Weighted average of trunk weights among contact set."""
    if not contact: return None
    ids = list(contact)
    weights = [agents[i].get_trunk_weights() for i in ids]
    avg = {k: sum(w[k] for w in weights)/len(weights) for k in weights[0]}
    return avg

def broadcast(agents, trunk, contact):
    for i in contact:
        agents[i].set_trunk_weights(trunk)
''')

open('sim/fed/gossip.py','w').write(r'''
import numpy as np
DELTA_THRESH = 1e-4

def isl_gossip_round(agents, avail_g):
    """Topology-aware peer FL: exchange sparse weight deltas over ISLs."""
    n = len(agents)
    base = agents[0].get_trunk_weights()
    for i in range(n):
        wi = agents[i].get_trunk_weights()
        for j in range(n):
            if not avail_g[i,j]: continue
            wj   = agents[j].get_trunk_weights()
            delta= {k: wi[k]-wj[k] for k in wi}
            # sparse: only transmit layers with significant change
            update = {k: v for k,v in delta.items() if v.abs().max() > DELTA_THRESH}
            if not update: continue
            merged = {k: wj[k] + 0.5*update[k] if k in update else wj[k] for k in wj}
            agents[j].set_trunk_weights(merged)
''')
print("fed/ written")
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 7 — Common training helpers (paste once, used by all experiments)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CELL7 = r"""
import numpy as np, json
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

def _blank_metrics():
    return {"round":[], "mean_reward":[], "loss":[],
            "mean_kappa":[], "drop_rate":[], "link_utilization":[],
            "throughput_gbps":[], "p99_rtt":[]}

def _collect_step(obs, x_new, x_prev):
    kappa=obs["kappa"]; cap=obs["cap"]; avail=obs["avail"]
    rtt=obs["rtt"]; S=obs["S"]; drop=obs["drop"]
    mask = avail > 0; cap_sum = cap[mask].sum() + 1e-9
    return dict(
        kappa = float(kappa[mask].mean()) if mask.any() else 0.0,
        drop  = float(drop[mask].sum() / (cap_sum * DT)),
        util  = float(x_new[mask].sum() / cap_sum),
        tput  = float(S[mask].sum() / DT / 1e9),
        p99   = float(np.percentile(rtt[mask], 99)) if mask.any() else 0.0,
    )

def _append_metrics(m, k, rewards, rows, losses):
    m["round"].append(k)
    m["mean_reward"].append(float(np.mean(rewards)))
    m["loss"].append(float(np.mean(losses)) if losses else 0.0)
    m["mean_kappa"].append(float(np.mean([r["kappa"] for r in rows])))
    m["drop_rate"].append(float(np.mean([r["drop"]  for r in rows])))
    m["link_utilization"].append(float(np.mean([r["util"] for r in rows])))
    m["throughput_gbps"].append(float(np.mean([r["tput"] for r in rows])))
    m["p99_rtt"].append(float(np.mean([r["p99"]  for r in rows])))

def _save(m, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path,"w") as f: json.dump(m, f)

def _make_flat(n=N_SAT): return [PPOAgent(STATE_DIM, ACTION_DIM, i) for i in range(n)]
def _make_gnn(n=N_SAT):  return [GNNPPOAgent(ACTION_DIM, i) for i in range(n)]
def _sync(ags):
    w = ags[0].get_weights()
    for a in ags: a.set_weights(w)

print("Helpers ready — K_ROUNDS=%d STEPS=%d N_SAT=%d" % (K_ROUNDS, STEPS_PER_ROUND, N_SAT))
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 8 — TCP-CUBIC baseline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CELL8 = r"""
SAVE = "results/tcp_cubic/metrics.json"
net  = ISLNetwork(seed=42)
W    = np.ones((N_SAT,N_SAT))*0.1; t_sl = np.zeros((N_SAT,N_SAT))
x_prev = np.zeros((N_SAT,N_SAT))
m = _blank_metrics()

CUBIC_C=0.4; CUBIC_BETA=0.7; W_MAX=1.0; T_K=5.0

for k in range(K_ROUNDS):
    rewards=[]; rows=[]
    for _ in range(STEPS_PER_ROUND):
        obs   = net.step(x_prev)
        kappa=obs["kappa"]; cap=obs["cap"]; avail=obs["avail"]; S=obs["S"]; drop=obs["drop"]
        x_new = np.zeros((N_SAT,N_SAT))
        for i in range(N_SAT):
            for j in range(N_SAT):
                if not avail[i,j]: continue
                tt = t_sl[i,j]+DT
                wc = float(np.clip(CUBIC_C*(tt-T_K)**3+W_MAX, 0.01, W_MAX))
                if kappa[i,j]>0.8: wc=W[i,j]*CUBIC_BETA; tt=0.0
                W[i,j]=wc; t_sl[i,j]=tt
                x_new[i,j]=wc*cap[i,j]
        step_r=[]
        for i in range(N_SAT):
            nbrs=[j for j in range(N_SAT) if avail[i,j]]
            if nbrs: step_r.append(compute_reward(i,nbrs,S,cap,kappa,drop,B,x_new,x_prev,DT))
        rewards.append(np.mean(step_r) if step_r else 0)
        rows.append(_collect_step(obs,x_new,x_prev))
        x_prev=x_new.copy()
    _append_metrics(m, k, rewards, rows, [])
    if k%LOG_EVERY==0:
        print(f"TCP Round {k:4d} | tput={m['throughput_gbps'][-1]:.1f}G drop={m['drop_rate'][-1]:.3f}")
    if k%500==0: _save(m, SAVE)

_save(m, SAVE)
print("TCP-CUBIC done →", SAVE)
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 9 — Ind-PPO baseline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CELL9 = r"""
SAVE = "results/indppo_isl/metrics.json"
net  = ISLNetwork(seed=42)
agents = _make_flat(); _sync(agents)
x_prev=np.zeros((N_SAT,N_SAT)); kappa_prev=np.zeros((N_SAT,N_SAT))
m = _blank_metrics()

for k in range(K_ROUNDS):
    rewards=[]; rows=[]
    for _ in range(STEPS_PER_ROUND):
        obs=net.step(x_prev)
        kappa=obs["kappa"]; cap=obs["cap"]; avail=obs["avail"]
        rtt=obs["rtt"]; S=obs["S"]; drop=obs["drop"]
        x_new=np.zeros((N_SAT,N_SAT))
        for i in range(N_SAT):
            nbrs=[j for j in range(N_SAT) if avail[i,j]]
            if not nbrs: continue
            state=build_state(i,nbrs,kappa,cap,avail,rtt,kappa_prev,drop)
            s_pad=np.zeros(STATE_DIM,dtype=np.float32); s_pad[:len(state)]=state[:STATE_DIM]
            action,lp,val=agents[i].select_action(s_pad)
            for idx,j in enumerate(nbrs[:ACTION_DIM]): x_new[i,j]=action[idx]*cap[i,j]
            r=compute_reward(i,nbrs,S,cap,kappa,drop,B,x_new,x_prev,DT)
            agents[i].store(s_pad,action[:ACTION_DIM],r,lp,val,done=0.0)
            rewards.append(r)
        rows.append(_collect_step(obs,x_new,x_prev))
        kappa_prev=kappa.copy(); x_prev=x_new.copy()
    losses=[ag.update() for ag in agents]
    _append_metrics(m,k,rewards,rows,losses)
    if k%LOG_EVERY==0:
        print(f"IndPPO Round {k:4d} | reward={m['mean_reward'][-1]:+.4f} tput={m['throughput_gbps'][-1]:.1f}G")
    if k%500==0: _save(m,SAVE)

_save(m,SAVE); print("Ind-PPO done →", SAVE)
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 10 — FedAvg baseline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CELL10 = r"""
SAVE = "results/fedavg/metrics.json"
net  = ISLNetwork(seed=42)
agents=_make_flat(); _sync(agents)
x_prev=np.zeros((N_SAT,N_SAT)); kappa_prev=np.zeros((N_SAT,N_SAT))
m=_blank_metrics()

for k in range(K_ROUNDS):
    rewards=[]; rows=[]
    for _ in range(STEPS_PER_ROUND):
        obs=net.step(x_prev)
        kappa=obs["kappa"]; cap=obs["cap"]; avail=obs["avail"]
        rtt=obs["rtt"]; S=obs["S"]; drop=obs["drop"]
        x_new=np.zeros((N_SAT,N_SAT))
        for i in range(N_SAT):
            nbrs=[j for j in range(N_SAT) if avail[i,j]]
            if not nbrs: continue
            state=build_state(i,nbrs,kappa,cap,avail,rtt,kappa_prev,drop)
            s_pad=np.zeros(STATE_DIM,dtype=np.float32); s_pad[:len(state)]=state[:STATE_DIM]
            action,lp,val=agents[i].select_action(s_pad)
            for idx,j in enumerate(nbrs[:ACTION_DIM]): x_new[i,j]=action[idx]*cap[i,j]
            r=compute_reward(i,nbrs,S,cap,kappa,drop,B,x_new,x_prev,DT)
            agents[i].store(s_pad,action[:ACTION_DIM],r,lp,val,done=0.0)
            rewards.append(r)
        rows.append(_collect_step(obs,x_new,x_prev))
        kappa_prev=kappa.copy(); x_prev=x_new.copy()
    losses=[ag.update() for ag in agents]
    # direct-only FedAvg
    pos=_sat_positions(net.t); direct=visible_satellites(pos)
    trunk=fedavg_round(agents,direct)
    if trunk: broadcast(agents,trunk,direct)
    _append_metrics(m,k,rewards,rows,losses)
    if k%LOG_EVERY==0:
        print(f"FedAvg Round {k:4d} | reward={m['mean_reward'][-1]:+.4f} tput={m['throughput_gbps'][-1]:.1f}G")
    if k%500==0: _save(m,SAVE)

_save(m,SAVE); print("FedAvg done →", SAVE)
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 11 — CA-PFedAvg (relay only, no gossip)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CELL11 = r"""
SAVE="results/capfedavg/metrics.json"
net=ISLNetwork(seed=42)
agents=_make_flat(); _sync(agents)
x_prev=np.zeros((N_SAT,N_SAT)); kappa_prev=np.zeros((N_SAT,N_SAT))
m=_blank_metrics()

for k in range(K_ROUNDS):
    rewards=[]; rows=[]
    for _ in range(STEPS_PER_ROUND):
        obs=net.step(x_prev)
        kappa=obs["kappa"]; cap=obs["cap"]; avail=obs["avail"]
        rtt=obs["rtt"]; S=obs["S"]; drop=obs["drop"]
        x_new=np.zeros((N_SAT,N_SAT))
        for i in range(N_SAT):
            nbrs=[j for j in range(N_SAT) if avail[i,j]]
            if not nbrs: continue
            state=build_state(i,nbrs,kappa,cap,avail,rtt,kappa_prev,drop)
            s_pad=np.zeros(STATE_DIM,dtype=np.float32); s_pad[:len(state)]=state[:STATE_DIM]
            action,lp,val=agents[i].select_action(s_pad)
            for idx,j in enumerate(nbrs[:ACTION_DIM]): x_new[i,j]=action[idx]*cap[i,j]
            r=compute_reward(i,nbrs,S,cap,kappa,drop,B,x_new,x_prev,DT)
            agents[i].store(s_pad,action[:ACTION_DIM],r,lp,val,done=0.0)
            rewards.append(r)
        rows.append(_collect_step(obs,x_new,x_prev))
        kappa_prev=kappa.copy(); x_prev=x_new.copy()
    losses=[ag.update() for ag in agents]
    _,_,avail_g=compute_isl_graph(net.t)
    pos=_sat_positions(net.t); direct=visible_satellites(pos)
    contact=relay_contact_set(direct,avail_g)
    trunk=fedavg_round(agents,contact)
    if trunk: broadcast(agents,trunk,contact)
    _append_metrics(m,k,rewards,rows,losses)
    if k%LOG_EVERY==0:
        print(f"CAPFedAvg Round {k:4d} | reward={m['mean_reward'][-1]:+.4f} contact={len(contact)}")
    if k%500==0: _save(m,SAVE)

_save(m,SAVE); print("CA-PFedAvg done →", SAVE)
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 12 — Flat-FRL (CA-PFedAvg + ISL-Gossip)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CELL12 = r"""
SAVE="results/flat_frl/metrics.json"
net=ISLNetwork(seed=42)
agents=_make_flat(); _sync(agents)
x_prev=np.zeros((N_SAT,N_SAT)); kappa_prev=np.zeros((N_SAT,N_SAT))
m=_blank_metrics()

for k in range(K_ROUNDS):
    rewards=[]; rows=[]
    for _ in range(STEPS_PER_ROUND):
        obs=net.step(x_prev)
        kappa=obs["kappa"]; cap=obs["cap"]; avail=obs["avail"]
        rtt=obs["rtt"]; S=obs["S"]; drop=obs["drop"]
        x_new=np.zeros((N_SAT,N_SAT))
        for i in range(N_SAT):
            nbrs=[j for j in range(N_SAT) if avail[i,j]]
            if not nbrs: continue
            state=build_state(i,nbrs,kappa,cap,avail,rtt,kappa_prev,drop)
            s_pad=np.zeros(STATE_DIM,dtype=np.float32); s_pad[:len(state)]=state[:STATE_DIM]
            action,lp,val=agents[i].select_action(s_pad)
            for idx,j in enumerate(nbrs[:ACTION_DIM]): x_new[i,j]=action[idx]*cap[i,j]
            r=compute_reward(i,nbrs,S,cap,kappa,drop,B,x_new,x_prev,DT)
            agents[i].store(s_pad,action[:ACTION_DIM],r,lp,val,done=0.0)
            rewards.append(r)
        rows.append(_collect_step(obs,x_new,x_prev))
        kappa_prev=kappa.copy(); x_prev=x_new.copy()
    losses=[ag.update() for ag in agents]
    _,_,avail_g=compute_isl_graph(net.t)
    isl_gossip_round(agents,avail_g)
    pos=_sat_positions(net.t); direct=visible_satellites(pos)
    contact=relay_contact_set(direct,avail_g)
    trunk=fedavg_round(agents,contact)
    if trunk: broadcast(agents,trunk,contact)
    _append_metrics(m,k,rewards,rows,losses)
    if k%LOG_EVERY==0:
        print(f"FlatFRL Round {k:4d} | reward={m['mean_reward'][-1]:+.4f} tput={m['throughput_gbps'][-1]:.1f}G")
    if k%500==0: _save(m,SAVE)

_save(m,SAVE); print("Flat-FRL done →", SAVE)
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 13 — GNN-FRL (full proposal)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CELL13 = r"""
SAVE="results/gnn_frl/metrics.json"
net=ISLNetwork(seed=42)
agents=_make_gnn(); _sync(agents)
x_prev=np.zeros((N_SAT,N_SAT)); kappa_prev=np.zeros((N_SAT,N_SAT))
m=_blank_metrics(); C_MAX=10e9; RTT_MAX=0.1

for k in range(K_ROUNDS):
    rewards=[]; rows=[]
    for _ in range(STEPS_PER_ROUND):
        obs=net.step(x_prev)
        kappa=obs["kappa"]; cap=obs["cap"]; avail=obs["avail"]
        rtt=obs["rtt"]; S=obs["S"]; drop=obs["drop"]
        x_new=np.zeros((N_SAT,N_SAT))
        for i in range(N_SAT):
            nbrs=[j for j in range(N_SAT) if avail[i,j]]
            if not nbrs: continue
            nbr_feats=np.array([[float(kappa[i,j]),float(cap[i,j]/C_MAX),
                                  float(min(rtt[i,j]/RTT_MAX,1.0)),
                                  float(np.clip(kappa[i,j]-kappa_prev[i,j],-1,1))]
                                 for j in nbrs],dtype=np.float32)
            sf=nbr_feats.mean(axis=0)
            action,lp,val=agents[i].select_action(sf,nbr_feats)
            for idx,j in enumerate(nbrs[:ACTION_DIM]): x_new[i,j]=action[idx]*cap[i,j]
            r=compute_reward(i,nbrs,S,cap,kappa,drop,B,x_new,x_prev,DT)
            agents[i].store(sf,nbr_feats,action[:ACTION_DIM],r,lp,val,done=0.0)
            rewards.append(r)
        rows.append(_collect_step(obs,x_new,x_prev))
        kappa_prev=kappa.copy(); x_prev=x_new.copy()
    losses=[ag.update() for ag in agents]
    _,_,avail_g=compute_isl_graph(net.t)
    isl_gossip_round(agents,avail_g)
    pos=_sat_positions(net.t); direct=visible_satellites(pos)
    contact=relay_contact_set(direct,avail_g)
    trunk=fedavg_round(agents,contact)
    if trunk: broadcast(agents,trunk,contact)
    _append_metrics(m,k,rewards,rows,losses)
    if k%LOG_EVERY==0:
        print(f"GNN-FRL Round {k:4d} | reward={m['mean_reward'][-1]:+.4f} tput={m['throughput_gbps'][-1]:.1f}G contact={len(contact)}")
    if k%500==0: _save(m,SAVE)

_save(m,SAVE); print("GNN-FRL done →", SAVE)
"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 14 — Disruption Experiment
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CELL14 = r"""
import random
DISRUPT_EVERY=500; DISRUPT_LEN=50; DISRUPT_FRAC=0.20
C_MAX=10e9; RTT_MAX=0.1

def _dmask(seed):
    rng=random.Random(seed); mask=np.ones((N_SAT,N_SAT))
    edges=[(i,j) for i in range(N_SAT) for j in range(N_SAT) if i!=j]
    for i,j in rng.sample(edges, int(len(edges)*DISRUPT_FRAC)): mask[i,j]=0
    return mask

net=ISLNetwork(seed=42)
ag_tcp_W=np.ones((N_SAT,N_SAT))*0.1; ag_tcp_t=np.zeros((N_SAT,N_SAT))
x_tcp=np.zeros((N_SAT,N_SAT))

ag_ind=_make_flat(); _sync(ag_ind); x_ind=np.zeros((N_SAT,N_SAT)); kp_ind=np.zeros((N_SAT,N_SAT))
ag_flat=_make_flat(); _sync(ag_flat); x_flat=np.zeros((N_SAT,N_SAT)); kp_flat=np.zeros((N_SAT,N_SAT))
ag_gnn=_make_gnn();  _sync(ag_gnn);  x_gnn=np.zeros((N_SAT,N_SAT));  kp_gnn=np.zeros((N_SAT,N_SAT))

m_tcp=_blank_metrics(); m_ind=_blank_metrics()
m_flat=_blank_metrics(); m_gnn=_blank_metrics()
for m in [m_tcp,m_ind,m_flat,m_gnn]: m["disrupted"]=[]

CUBIC_C=0.4; CUBIC_BETA=0.7; W_MAX=1.0; T_K=5.0

for k in range(K_ROUNDS):
    phase=k%DISRUPT_EVERY; disrupted=(phase<DISRUPT_LEN)
    dmask=_dmask(k//DISRUPT_EVERY) if disrupted else np.ones((N_SAT,N_SAT))

    # TCP
    tcp_r=[]; tcp_rows=[]
    for _ in range(STEPS_PER_ROUND):
        obs=net.step(x_tcp)
        kappa=obs["kappa"]; cap=obs["cap"]; avail=obs["avail"]*dmask; S=obs["S"]; drop=obs["drop"]
        x_new=np.zeros((N_SAT,N_SAT))
        for i in range(N_SAT):
            for j in range(N_SAT):
                if not avail[i,j]: continue
                tt=ag_tcp_t[i,j]+DT
                wc=float(np.clip(CUBIC_C*(tt-T_K)**3+W_MAX,0.01,W_MAX))
                if kappa[i,j]>0.8: wc=ag_tcp_W[i,j]*CUBIC_BETA; tt=0.0
                ag_tcp_W[i,j]=wc; ag_tcp_t[i,j]=tt; x_new[i,j]=wc*cap[i,j]
        step_r=[compute_reward(i,[j for j in range(N_SAT) if avail[i,j]],S,cap,kappa,drop,B,x_new,x_tcp,DT)
                for i in range(N_SAT) if [j for j in range(N_SAT) if avail[i,j]]]
        tcp_r.append(np.mean(step_r) if step_r else 0)
        tcp_rows.append(_collect_step(obs,x_new,x_tcp)); x_tcp=x_new.copy()
    _append_metrics(m_tcp,k,tcp_r,tcp_rows,[]); m_tcp["disrupted"].append(int(disrupted))

    # Ind-PPO
    ind_r=[]; ind_rows=[]
    for _ in range(STEPS_PER_ROUND):
        obs=net.step(x_ind)
        kappa=obs["kappa"]; cap=obs["cap"]; avail=obs["avail"]*dmask; rtt=obs["rtt"]; S=obs["S"]; drop=obs["drop"]
        x_new=np.zeros((N_SAT,N_SAT))
        for i in range(N_SAT):
            nbrs=[j for j in range(N_SAT) if avail[i,j]]
            if not nbrs: continue
            state=build_state(i,nbrs,kappa,cap,avail,rtt,kp_ind,drop)
            s_pad=np.zeros(STATE_DIM,dtype=np.float32); s_pad[:len(state)]=state[:STATE_DIM]
            action,lp,val=ag_ind[i].select_action(s_pad)
            for idx,j in enumerate(nbrs[:ACTION_DIM]): x_new[i,j]=action[idx]*cap[i,j]
            ag_ind[i].store(s_pad,action[:ACTION_DIM],compute_reward(i,nbrs,S,cap,kappa,drop,B,x_new,x_ind,DT),lp,val,0.0)
            ind_r.append(compute_reward(i,nbrs,S,cap,kappa,drop,B,x_new,x_ind,DT))
        ind_rows.append(_collect_step(obs,x_new,x_ind)); kp_ind=kappa.copy(); x_ind=x_new.copy()
    losses=[ag.update() for ag in ag_ind]
    _append_metrics(m_ind,k,ind_r,ind_rows,losses); m_ind["disrupted"].append(int(disrupted))

    # Flat-FRL
    flat_r=[]; flat_rows=[]
    for _ in range(STEPS_PER_ROUND):
        obs=net.step(x_flat)
        kappa=obs["kappa"]; cap=obs["cap"]; avail=obs["avail"]*dmask; rtt=obs["rtt"]; S=obs["S"]; drop=obs["drop"]
        x_new=np.zeros((N_SAT,N_SAT))
        for i in range(N_SAT):
            nbrs=[j for j in range(N_SAT) if avail[i,j]]
            if not nbrs: continue
            state=build_state(i,nbrs,kappa,cap,avail,rtt,kp_flat,drop)
            s_pad=np.zeros(STATE_DIM,dtype=np.float32); s_pad[:len(state)]=state[:STATE_DIM]
            action,lp,val=ag_flat[i].select_action(s_pad)
            for idx,j in enumerate(nbrs[:ACTION_DIM]): x_new[i,j]=action[idx]*cap[i,j]
            ag_flat[i].store(s_pad,action[:ACTION_DIM],compute_reward(i,nbrs,S,cap,kappa,drop,B,x_new,x_flat,DT),lp,val,0.0)
            flat_r.append(compute_reward(i,nbrs,S,cap,kappa,drop,B,x_new,x_flat,DT))
        flat_rows.append(_collect_step(obs,x_new,x_flat)); kp_flat=kappa.copy(); x_flat=x_new.copy()
    losses=[ag.update() for ag in ag_flat]
    _,_,avail_g=compute_isl_graph(net.t); isl_gossip_round(ag_flat,avail_g)
    pos=_sat_positions(net.t); contact=relay_contact_set(visible_satellites(pos),avail_g)
    trunk=fedavg_round(ag_flat,contact)
    if trunk: broadcast(ag_flat,trunk,contact)
    _append_metrics(m_flat,k,flat_r,flat_rows,losses); m_flat["disrupted"].append(int(disrupted))

    # GNN-FRL
    gnn_r=[]; gnn_rows=[]
    for _ in range(STEPS_PER_ROUND):
        obs=net.step(x_gnn)
        kappa=obs["kappa"]; cap=obs["cap"]; avail=obs["avail"]*dmask; rtt=obs["rtt"]; S=obs["S"]; drop=obs["drop"]
        x_new=np.zeros((N_SAT,N_SAT))
        for i in range(N_SAT):
            nbrs=[j for j in range(N_SAT) if avail[i,j]]
            if not nbrs: continue
            nbr_feats=np.array([[float(kappa[i,j]),float(cap[i,j]/C_MAX),float(min(rtt[i,j]/RTT_MAX,1.0)),float(np.clip(kappa[i,j]-kp_gnn[i,j],-1,1))] for j in nbrs],dtype=np.float32)
            sf=nbr_feats.mean(axis=0)
            action,lp,val=ag_gnn[i].select_action(sf,nbr_feats)
            for idx,j in enumerate(nbrs[:ACTION_DIM]): x_new[i,j]=action[idx]*cap[i,j]
            ag_gnn[i].store(sf,nbr_feats,action[:ACTION_DIM],compute_reward(i,nbrs,S,cap,kappa,drop,B,x_new,x_gnn,DT),lp,val,0.0)
            gnn_r.append(compute_reward(i,nbrs,S,cap,kappa,drop,B,x_new,x_gnn,DT))
        gnn_rows.append(_collect_step(obs,x_new,x_gnn)); kp_gnn=kappa.copy(); x_gnn=x_new.copy()
    losses=[ag.update() for ag in ag_gnn]
    isl_gossip_round(ag_gnn,avail_g)
    trunk=fedavg_round(ag_gnn,contact)
    if trunk: broadcast(ag_gnn,trunk,contact)
    _append_metrics(m_gnn,k,gnn_r,gnn_rows,losses); m_gnn["disrupted"].append(int(disrupted))

    if k%LOG_EVERY==0:
        tag="[DISRUPT]" if disrupted else "         "
        print(f"Round {k:4d} {tag} | TCP={m_tcp['throughput_gbps'][-1]:.1f}G Flat={m_flat['throughput_gbps'][-1]:.1f}G GNN={m_gnn['throughput_gbps'][-1]:.1f}G")
    if k%500==0:
        for n,m in [("tcp",m_tcp),("indppo",m_ind),("flat_frl",m_flat),("gnn_frl",m_gnn)]:
            _save(m, f"results/disruption/{n}_metrics.json")

for n,m in [("tcp",m_tcp),("indppo",m_ind),("flat_frl",m_flat),("gnn_frl",m_gnn)]:
    _save(m, f"results/disruption/{n}_metrics.json")
print("Disruption experiment done")
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 15 — Handover Experiment
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CELL15 = r"""
import math as _math

GS_LIST=[{"lat":48.8,"lon":2.3},{"lat":37.4,"lon":-122.1},{"lat":-33.9,"lon":151.2}]
EARTH_R_KM=6371.0; GS_MAX_KM=2500.0; GS_ELEV=5.0

def _gs_ecef(gs):
    lat,lon=_math.radians(gs["lat"]),_math.radians(gs["lon"]); r=EARTH_R_KM
    return np.array([r*_math.cos(lat)*_math.cos(lon),r*_math.cos(lat)*_math.sin(lon),r*_math.sin(lat)])

def vis_from_gs(pos_ecef):
    gps=[_gs_ecef(g) for g in GS_LIST]; vis=set()
    for i,sp in enumerate(pos_ecef):
        for gp in gps:
            diff=sp-gp; d=float(np.linalg.norm(diff))
            if d>GS_MAX_KM: continue
            gn=gp/np.linalg.norm(gp)
            elev=_math.degrees(_math.asin(float(np.clip(np.dot(diff/d,gn),-1,1))))
            if elev>=GS_ELEV: vis.add(i); break
    return vis

def fedavg_direct(agents, gs_vis):
    if not gs_vis: return 0
    ws=[agents[i].get_trunk_weights() for i in gs_vis]
    avg={k:sum(w[k] for w in ws)/len(ws) for k in ws[0]}
    for ag in agents: ag.set_trunk_weights(avg)
    return len(gs_vis)

net=ISLNetwork(seed=42)
ag_fv=_make_flat(); _sync(ag_fv); x_fv=np.zeros((N_SAT,N_SAT)); kp_fv=np.zeros((N_SAT,N_SAT))
ag_ca=_make_flat(); _sync(ag_ca); x_ca=np.zeros((N_SAT,N_SAT)); kp_ca=np.zeros((N_SAT,N_SAT))
ag_fl=_make_flat(); _sync(ag_fl); x_fl=np.zeros((N_SAT,N_SAT)); kp_fl=np.zeros((N_SAT,N_SAT))
ag_gn=_make_gnn();  _sync(ag_gn); x_gn=np.zeros((N_SAT,N_SAT)); kp_gn=np.zeros((N_SAT,N_SAT))

def _blank_ho():
    m=_blank_metrics(); m["gs_visible_count"]=[]; m["fed_participants"]=[]
    return m

m_fv=_blank_ho(); m_ca=_blank_ho(); m_fl=_blank_ho(); m_gn=_blank_ho()

C_MAX=10e9; RTT_MAX=0.1

for k in range(K_ROUNDS):
    _,_,avail_g=compute_isl_graph(net.t)
    pos_ecef=_sat_positions(net.t); gs_vis=vis_from_gs(pos_ecef)
    contact=relay_contact_set(gs_vis,avail_g)

    for (agents,x_prev,kp,m,fed_type) in [
        (ag_fv,x_fv,kp_fv,m_fv,"direct"),
        (ag_ca,x_ca,kp_ca,m_ca,"relay"),
        (ag_fl,x_fl,kp_fl,m_fl,"relay_gossip"),
        (ag_gn,x_gn,kp_gn,m_gn,"relay_gossip_gnn"),
    ]:
        use_gnn=(fed_type=="relay_gossip_gnn")
        r=[]; rows=[]
        for _ in range(STEPS_PER_ROUND):
            obs=net.step(x_prev)
            kappa=obs["kappa"]; cap=obs["cap"]; avail=obs["avail"]; rtt=obs["rtt"]; S=obs["S"]; drop=obs["drop"]
            x_new=np.zeros((N_SAT,N_SAT))
            for i in range(N_SAT):
                nbrs=[j for j in range(N_SAT) if avail[i,j]]
                if not nbrs: continue
                if use_gnn:
                    nf=np.array([[float(kappa[i,j]),float(cap[i,j]/C_MAX),float(min(rtt[i,j]/RTT_MAX,1.0)),float(np.clip(kappa[i,j]-kp[i,j],-1,1))] for j in nbrs],dtype=np.float32)
                    sf=nf.mean(axis=0); action,lp,val=agents[i].select_action(sf,nf)
                    for idx,j in enumerate(nbrs[:ACTION_DIM]): x_new[i,j]=action[idx]*cap[i,j]
                    agents[i].store(sf,nf,action[:ACTION_DIM],compute_reward(i,nbrs,S,cap,kappa,drop,B,x_new,x_prev,DT),lp,val,0.0)
                else:
                    state=build_state(i,nbrs,kappa,cap,avail,rtt,kp,drop)
                    s_pad=np.zeros(STATE_DIM,dtype=np.float32); s_pad[:len(state)]=state[:STATE_DIM]
                    action,lp,val=agents[i].select_action(s_pad)
                    for idx,j in enumerate(nbrs[:ACTION_DIM]): x_new[i,j]=action[idx]*cap[i,j]
                    agents[i].store(s_pad,action[:ACTION_DIM],compute_reward(i,nbrs,S,cap,kappa,drop,B,x_new,x_prev,DT),lp,val,0.0)
                r.append(compute_reward(i,nbrs,S,cap,kappa,drop,B,x_new,x_prev,DT))
            rows.append(_collect_step(obs,x_new,x_prev))
            kp[:]=kappa; x_prev[:]=x_new

        losses=[ag.update() for ag in agents]
        if fed_type=="direct":
            n_fed=fedavg_direct(agents,gs_vis)
        elif fed_type=="relay":
            trunk=fedavg_round(agents,contact); n_fed=len(contact)
            if trunk: broadcast(agents,trunk,contact)
        else:
            isl_gossip_round(agents,avail_g)
            trunk=fedavg_round(agents,contact); n_fed=len(contact)
            if trunk: broadcast(agents,trunk,contact)
        _append_metrics(m,k,r,rows,losses)
        m["gs_visible_count"].append(len(gs_vis)); m["fed_participants"].append(n_fed)

    if k%LOG_EVERY==0:
        print(f"HO Round {k:4d} | GS-vis={len(gs_vis)} relay={len(contact)} | FedAvg={m_fv['throughput_gbps'][-1]:.1f}G CA={m_ca['throughput_gbps'][-1]:.1f}G GNN={m_gn['throughput_gbps'][-1]:.1f}G")
    if k%500==0:
        for n,m in [("fedavg",m_fv),("capfedavg",m_ca),("flat_frl",m_fl),("gnn_frl",m_gn)]:
            _save(m, f"results/handover/{n}_metrics.json")

for n,m in [("fedavg",m_fv),("capfedavg",m_ca),("flat_frl",m_fl),("gnn_frl",m_gn)]:
    _save(m, f"results/handover/{n}_metrics.json")
print("Handover experiment done")
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 16 — Ablation Study
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CELL16 = r"""
VARIANTS=["full","no_gossip","no_relay","flat","ind"]
abl_ag={
    "full":_make_gnn(),"no_gossip":_make_gnn(),"no_relay":_make_gnn(),
    "flat":_make_flat(),"ind":_make_flat()}
for ags in abl_ag.values(): _sync(ags)
abl_x={n:np.zeros((N_SAT,N_SAT)) for n in VARIANTS}
abl_kp={n:np.zeros((N_SAT,N_SAT)) for n in VARIANTS}
abl_m={n:_blank_metrics() for n in VARIANTS}
C_MAX=10e9; RTT_MAX=0.1; net2=ISLNetwork(seed=42)

for k in range(K_ROUNDS):
    _,_,avail_g=compute_isl_graph(net2.t)
    pos=_sat_positions(net2.t); direct=visible_satellites(pos)
    contact_relay=relay_contact_set(direct,avail_g)
    contact_direct=set(direct)

    for name in VARIANTS:
        agents=abl_ag[name]; x_prev=abl_x[name]; kp=abl_kp[name]; m=abl_m[name]
        use_gnn=(name in ("full","no_gossip","no_relay"))
        r=[]; rows=[]
        for _ in range(STEPS_PER_ROUND):
            obs=net2.step(x_prev)
            kappa=obs["kappa"]; cap=obs["cap"]; avail=obs["avail"]; rtt=obs["rtt"]; S=obs["S"]; drop=obs["drop"]
            x_new=np.zeros((N_SAT,N_SAT))
            for i in range(N_SAT):
                nbrs=[j for j in range(N_SAT) if avail[i,j]]
                if not nbrs: continue
                if use_gnn:
                    nf=np.array([[float(kappa[i,j]),float(cap[i,j]/C_MAX),float(min(rtt[i,j]/RTT_MAX,1.0)),float(np.clip(kappa[i,j]-kp[i,j],-1,1))] for j in nbrs],dtype=np.float32)
                    sf=nf.mean(axis=0); action,lp,val=agents[i].select_action(sf,nf)
                    for idx,j in enumerate(nbrs[:ACTION_DIM]): x_new[i,j]=action[idx]*cap[i,j]
                    agents[i].store(sf,nf,action[:ACTION_DIM],compute_reward(i,nbrs,S,cap,kappa,drop,B,x_new,x_prev,DT),lp,val,0.0)
                else:
                    state=build_state(i,nbrs,kappa,cap,avail,rtt,kp,drop)
                    s_pad=np.zeros(STATE_DIM,dtype=np.float32); s_pad[:len(state)]=state[:STATE_DIM]
                    action,lp,val=agents[i].select_action(s_pad)
                    for idx,j in enumerate(nbrs[:ACTION_DIM]): x_new[i,j]=action[idx]*cap[i,j]
                    agents[i].store(s_pad,action[:ACTION_DIM],compute_reward(i,nbrs,S,cap,kappa,drop,B,x_new,x_prev,DT),lp,val,0.0)
                r.append(compute_reward(i,nbrs,S,cap,kappa,drop,B,x_new,x_prev,DT))
            rows.append(_collect_step(obs,x_new,x_prev)); kp[:]=kappa; x_prev[:]=x_new
        losses=[ag.update() for ag in agents]
        if name=="full":
            isl_gossip_round(agents,avail_g)
            trunk=fedavg_round(agents,contact_relay)
            if trunk: broadcast(agents,trunk,contact_relay)
        elif name=="no_gossip":
            trunk=fedavg_round(agents,contact_relay)
            if trunk: broadcast(agents,trunk,contact_relay)
        elif name=="no_relay":
            isl_gossip_round(agents,avail_g)
            trunk=fedavg_round(agents,contact_direct)
            if trunk: broadcast(agents,trunk,contact_direct)
        elif name=="flat":
            isl_gossip_round(agents,avail_g)
            trunk=fedavg_round(agents,contact_relay)
            if trunk: broadcast(agents,trunk,contact_relay)
        # ind: no federation
        _append_metrics(m,k,r,rows,losses)

    if k%LOG_EVERY==0:
        print(f"Ablation Round {k:4d} | "+' | '.join(f"{n}={abl_m[n]['throughput_gbps'][-1]:.1f}G" for n in VARIANTS))
    if k%500==0:
        for n in VARIANTS: _save(abl_m[n], f"results/ablation/{n}_metrics.json")

for n in VARIANTS: _save(abl_m[n], f"results/ablation/{n}_metrics.json")
print("Ablation done")
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 17 — ALL PLOTS (run after all experiments)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CELL17 = r"""
import json, os, numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

FIGS="results/figures"; os.makedirs(FIGS,exist_ok=True)
SMOOTH=50; WIN=200

METHODS={"TCP-CUBIC":"results/tcp_cubic","FedAvg":"results/fedavg",
          "Ind-PPO":"results/indppo_isl","CA-PFedAvg":"results/capfedavg",
          "Flat-FRL":"results/flat_frl","GNN-FRL":"results/gnn_frl"}
COLORS=["#7f7f7f","#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd"]
LABELS=list(METHODS.keys())

def _load(path):
    p=Path(path)/"metrics.json"
    return json.load(open(p)) if p.exists() else None

def _loadf(path):
    p=Path(path); return json.load(open(p)) if p.exists() else None

def _sm(arr,w=SMOOTH):
    arr=np.array(arr,dtype=float)
    return np.convolve(arr,np.ones(w)/w,mode="valid") if len(arr)>=w else arr

def _save(fig,name):
    fig.savefig(f"{FIGS}/{name}.pdf",bbox_inches="tight",dpi=300)
    fig.savefig(f"{FIGS}/{name}.png",bbox_inches="tight",dpi=150)
    print(f"  saved {name}")

data={}
for lbl,path in METHODS.items():
    d=_load(path)
    if d: data[lbl]=d; print(f"[OK] {lbl}")
    else: print(f"[--] {lbl}")

RL=[l for l in LABELS if l!="TCP-CUBIC"]
RC=COLORS[1:]

# ── Fig 1: Learning Curves ────────────────────────────────────────────────────
fig,ax=plt.subplots(figsize=(9,4))
if "TCP-CUBIC" in data:
    ax.axhline(np.mean(data["TCP-CUBIC"]["mean_reward"]),color=COLORS[0],ls="--",lw=1.4,label=f"TCP-CUBIC (ref)")
for lbl,clr in zip(RL,RC):
    if lbl not in data: continue
    y=_sm(data[lbl]["mean_reward"]); ax.plot(np.arange(len(y)),y,color=clr,lw=1.6,label=lbl)
ax.set_xlabel("Training Round",fontsize=12); ax.set_ylabel("Mean Reward",fontsize=12)
ax.set_title("Learning Curves",fontsize=13); ax.legend(fontsize=9,ncol=2); ax.grid(alpha=0.3)
_save(fig,"fig01_learning_curves"); plt.close(fig)

# ── Fig 2: Throughput ─────────────────────────────────────────────────────────
fig,ax=plt.subplots(figsize=(9,4))
if "TCP-CUBIC" in data:
    ax.axhline(np.mean(data["TCP-CUBIC"]["throughput_gbps"]),color=COLORS[0],ls="--",lw=1.4,label="TCP-CUBIC (ref)")
for lbl,clr in zip(RL,RC):
    if lbl not in data: continue
    y=_sm(data[lbl]["throughput_gbps"]); ax.plot(np.arange(len(y)),y,color=clr,lw=1.6,label=lbl)
ax.set_xlabel("Training Round",fontsize=12); ax.set_ylabel("Throughput (Gbps)",fontsize=12)
ax.set_title("Network Throughput",fontsize=13); ax.legend(fontsize=9,ncol=2); ax.grid(alpha=0.3)
_save(fig,"fig02_throughput"); plt.close(fig)

# ── Fig 3: Drop Rate ──────────────────────────────────────────────────────────
fig,ax=plt.subplots(figsize=(9,4))
if "TCP-CUBIC" in data:
    ax.axhline(np.mean(data["TCP-CUBIC"]["drop_rate"]),color=COLORS[0],ls="--",lw=1.4,label="TCP-CUBIC (ref)")
for lbl,clr in zip(RL,RC):
    if lbl not in data: continue
    y=_sm(data[lbl]["drop_rate"]); ax.plot(np.arange(len(y)),y,color=clr,lw=1.6,label=lbl)
ax.set_xlabel("Training Round",fontsize=12); ax.set_ylabel("Packet Drop Rate",fontsize=12)
ax.set_title("Packet Drop Rate",fontsize=13); ax.legend(fontsize=9,ncol=2); ax.grid(alpha=0.3)
_save(fig,"fig03_drop_rate"); plt.close(fig)

# ── Fig 4: Congestion κ ───────────────────────────────────────────────────────
fig,ax=plt.subplots(figsize=(9,4))
if "TCP-CUBIC" in data:
    ax.axhline(np.mean(data["TCP-CUBIC"]["mean_kappa"]),color=COLORS[0],ls="--",lw=1.4,label="TCP-CUBIC (ref)")
for lbl,clr in zip(RL,RC):
    if lbl not in data: continue
    y=_sm(data[lbl]["mean_kappa"]); ax.plot(np.arange(len(y)),y,color=clr,lw=1.6,label=lbl)
ax.set_xlabel("Training Round",fontsize=12); ax.set_ylabel("Mean Queue Occupancy κ",fontsize=12)
ax.set_title("Congestion Level (κ)",fontsize=13); ax.legend(fontsize=9,ncol=2); ax.grid(alpha=0.3)
_save(fig,"fig04_kappa"); plt.close(fig)

# ── Fig 5: Link Utilization ───────────────────────────────────────────────────
fig,ax=plt.subplots(figsize=(9,4))
if "TCP-CUBIC" in data:
    ax.axhline(np.mean(data["TCP-CUBIC"]["link_utilization"]),color=COLORS[0],ls="--",lw=1.4,label="TCP-CUBIC (ref)")
for lbl,clr in zip(RL,RC):
    if lbl not in data: continue
    y=_sm(data[lbl]["link_utilization"]); ax.plot(np.arange(len(y)),y,color=clr,lw=1.6,label=lbl)
ax.set_xlabel("Training Round",fontsize=12); ax.set_ylabel("Link Utilization",fontsize=12)
ax.set_title("ISL Link Utilization",fontsize=13); ax.legend(fontsize=9,ncol=2); ax.grid(alpha=0.3)
_save(fig,"fig05_utilization"); plt.close(fig)

# ── Fig 6: P99 Tail Latency ───────────────────────────────────────────────────
fig,ax=plt.subplots(figsize=(9,4))
for lbl,clr in zip(LABELS,COLORS):
    if lbl not in data or "p99_rtt" not in data[lbl]: continue
    y=_sm(data[lbl]["p99_rtt"]); ax.plot(np.arange(len(y)),y,color=clr,lw=1.6,label=lbl)
ax.set_xlabel("Training Round",fontsize=12); ax.set_ylabel("P99 RTT (seconds)",fontsize=12)
ax.set_title("Tail Latency (P99 RTT)",fontsize=13); ax.legend(fontsize=9,ncol=2); ax.grid(alpha=0.3)
_save(fig,"fig06_p99_latency"); plt.close(fig)

# ── Fig 7: Steady-state bar chart ─────────────────────────────────────────────
fig,axes=plt.subplots(1,4,figsize=(17,4))
for metric,ylabel,ax in [("throughput_gbps","Throughput (Gbps)",axes[0]),
                          ("drop_rate","Drop Rate",axes[1]),
                          ("link_utilization","Link Utilization",axes[2]),
                          ("p99_rtt","P99 RTT (s)",axes[3])]:
    vals,errs,lbls,clrs=[],[],[],[]
    for lbl,clr in zip(LABELS,COLORS):
        if lbl not in data or metric not in data[lbl]: continue
        arr=np.array(data[lbl][metric]); tail=arr[-WIN:] if len(arr)>=WIN else arr
        vals.append(tail.mean()); errs.append(tail.std()); lbls.append(lbl); clrs.append(clr)
    x=np.arange(len(vals))
    ax.bar(x,vals,yerr=errs,capsize=4,color=clrs,alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(lbls,fontsize=7,rotation=35,ha="right")
    ax.set_ylabel(ylabel,fontsize=9); ax.grid(axis="y",alpha=0.3)
fig.suptitle(f"Steady-State Performance (last {WIN} rounds)",fontsize=12); fig.tight_layout()
_save(fig,"fig07_steady_state_bar"); plt.close(fig)

# ── Fig 8: Jain's Fairness Index ──────────────────────────────────────────────
def _jain(arr): x=np.array(arr,dtype=float); n=len(x); return float(x.sum()**2/(n*(x**2).sum()+1e-12)) if n>0 else 0.0
fig,ax=plt.subplots(figsize=(9,4))
has=False
for lbl,clr in zip(LABELS,COLORS):
    if lbl not in data or "throughput_gbps" not in data[lbl]: continue
    t=np.array(data[lbl]["throughput_gbps"]); fw=WIN//2
    fairness=[_jain(t[i:i+fw]) for i in range(0,len(t)-fw,fw//5)]
    if fairness:
        ax.plot(np.linspace(0,len(t),len(fairness)),fairness,color=clr,lw=1.5,label=lbl); has=True
if has:
    ax.set_xlabel("Training Round",fontsize=12); ax.set_ylabel("Jain's Fairness Index",fontsize=12)
    ax.set_ylim(0,1.05); ax.set_title("Throughput Fairness (J=1 is perfectly fair)",fontsize=12)
    ax.legend(fontsize=9,ncol=2); ax.grid(alpha=0.3)
_save(fig,"fig08_fairness"); plt.close(fig)

# ── Fig 9: Convergence Speed ──────────────────────────────────────────────────
rl_d=[l for l in RL if l in data]
if rl_d:
    best=max(np.mean(data[l]["mean_reward"][-200:]) for l in rl_d)
    THRESH=0.90*best; print(f"Convergence threshold: {THRESH:.4f}")
    fig,ax=plt.subplots(figsize=(7,4))
    for i,(lbl,clr) in enumerate(zip(LABELS,COLORS)):
        if lbl not in data: continue
        r=np.array(data[lbl]["mean_reward"]); cr=np.where(r>=THRESH)[0]
        cv=int(cr[0]) if len(cr)>0 else len(r)
        ax.barh(i,cv,color=clr,alpha=0.8); ax.text(cv+20,i,str(cv),va="center",fontsize=9)
    ax.set_yticks(range(len(LABELS))); ax.set_yticklabels(LABELS,fontsize=10)
    ax.set_xlabel("Rounds to 90% of best reward",fontsize=11)
    ax.set_title("Convergence Speed",fontsize=12); ax.axvline(3000,ls="--",color="k",alpha=0.3)
    ax.grid(axis="x",alpha=0.3); _save(fig,"fig09_convergence"); plt.close(fig)

# ── Fig 10: Disruption Resilience ────────────────────────────────────────────
DM={"TCP-CUBIC":"results/disruption/tcp_metrics.json",
    "Ind-PPO":"results/disruption/indppo_metrics.json",
    "Flat-FRL":"results/disruption/flat_frl_metrics.json",
    "GNN-FRL":"results/disruption/gnn_frl_metrics.json"}
DC=["#7f7f7f","#ff7f0e","#d62728","#9467bd"]
dm={lbl:_loadf(p) for lbl,p in DM.items() if _loadf(p)}
if dm:
    fig,axes=plt.subplots(1,2,figsize=(13,4))
    for metric,ylabel,ax in [("throughput_gbps","Throughput (Gbps)",axes[0]),("drop_rate","Drop Rate",axes[1])]:
        for lbl,clr in zip(DM.keys(),DC):
            if lbl not in dm: continue
            y=_sm(dm[lbl][metric],w=20); ax.plot(np.arange(len(y)),y,color=clr,lw=1.5,label=lbl)
        for s in range(0,3000,500): ax.axvspan(s,s+50,alpha=0.12,color="red")
        ax.set_xlabel("Round",fontsize=11); ax.set_ylabel(ylabel,fontsize=11); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    axes[0].set_title("Throughput Under ISL Disruptions (red=outage)",fontsize=11)
    axes[1].set_title("Drop Rate Under Disruptions",fontsize=11); fig.tight_layout()
    _save(fig,"fig10_disruption"); plt.close(fig)

# ── Fig 11: Handover Scenario ─────────────────────────────────────────────────
HM={"FedAvg":"results/handover/fedavg_metrics.json",
    "CA-PFedAvg":"results/handover/capfedavg_metrics.json",
    "Flat-FRL":"results/handover/flat_frl_metrics.json",
    "GNN-FRL":"results/handover/gnn_frl_metrics.json"}
HC=["#1f77b4","#2ca02c","#d62728","#9467bd"]
hm={lbl:_loadf(p) for lbl,p in HM.items() if _loadf(p)}
if hm:
    fig,axes=plt.subplots(1,3,figsize=(16,4))
    for lbl,clr in zip(HM.keys(),HC):
        if lbl not in hm: continue
        y=_sm(hm[lbl]["throughput_gbps"],w=20); axes[0].plot(np.arange(len(y)),y,color=clr,lw=1.5,label=lbl)
    axes[0].set_title("Throughput During Handover",fontsize=11); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    axes[0].set_xlabel("Round"); axes[0].set_ylabel("Throughput (Gbps)")
    for lbl,clr in zip(HM.keys(),HC):
        if lbl not in hm or "fed_participants" not in hm[lbl]: continue
        y=_sm(hm[lbl]["fed_participants"],w=20); axes[1].plot(np.arange(len(y)),y,color=clr,lw=1.5,label=lbl)
    axes[1].set_title("Federation Participants",fontsize=11); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    axes[1].set_xlabel("Round"); axes[1].set_ylabel("# Satellites in Federation")
    if "FedAvg" in hm and "gs_visible_count" in hm["FedAvg"]:
        vis=np.array(hm["FedAvg"]["gs_visible_count"]); y=_sm(vis,w=20); axes[2].fill_between(np.arange(len(y)),y,alpha=0.4,color="#1f77b4",label="Direct only")
    if "CA-PFedAvg" in hm and "fed_participants" in hm["CA-PFedAvg"]:
        rel=np.array(hm["CA-PFedAvg"]["fed_participants"]); y2=_sm(rel,w=20); axes[2].fill_between(np.arange(len(y2)),y2,alpha=0.4,color="#2ca02c",label="Relay-expanded")
    axes[2].set_title("Direct vs Relay Contact",fontsize=11); axes[2].legend(fontsize=9); axes[2].grid(alpha=0.3)
    axes[2].set_xlabel("Round"); axes[2].set_ylabel("Satellite Count")
    fig.suptitle("Handover: FedAvg Fails During Eclipse, CA-PFedAvg Survives",fontsize=12); fig.tight_layout()
    _save(fig,"fig11_handover"); plt.close(fig)

# ── Fig 12: Ablation Study ────────────────────────────────────────────────────
ABL={"full":"GNN-FRL\n(Full)","no_gossip":"No\nGossip","no_relay":"No\nRelay","flat":"Flat\nState","ind":"No\nFed"}
AC=["#9467bd","#17becf","#bcbd22","#d62728","#ff7f0e"]
abm={n:_loadf(f"results/ablation/{n}_metrics.json") for n in ABL if _loadf(f"results/ablation/{n}_metrics.json")}
if abm:
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for metric,ylabel,ax in [("throughput_gbps","Throughput (Gbps)",axes[0]),("drop_rate","Drop Rate",axes[1]),("link_utilization","Link Util",axes[2])]:
        vals,errs,lbls,clrs=[],[],[],[]
        for n,clr in zip(ABL.keys(),AC):
            if n not in abm: continue
            arr=np.array(abm[n][metric]); tail=arr[-WIN:] if len(arr)>=WIN else arr
            vals.append(tail.mean()); errs.append(tail.std()); lbls.append(ABL[n]); clrs.append(clr)
        x=np.arange(len(vals)); ax.bar(x,vals,yerr=errs,capsize=4,color=clrs,alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(lbls,fontsize=9); ax.set_ylabel(ylabel,fontsize=10); ax.grid(axis="y",alpha=0.3)
    fig.suptitle("Ablation Study — Each Component's Contribution",fontsize=12); fig.tight_layout()
    _save(fig,"fig12_ablation"); plt.close(fig)

# ── Table 1: CSV summary ──────────────────────────────────────────────────────
print("\n"+"="*85)
print(f"{'Method':<25}{'Throughput':>14}{'Drop':>10}{'Kappa':>8}{'Util':>8}{'P99-RTT':>10}")
print("="*85)
rows=[]
for lbl in LABELS:
    if lbl not in data: continue
    d=data[lbl]
    def tm(key): arr=np.array(d.get(key,[0])); return float(arr[-WIN:].mean() if len(arr)>=WIN else arr.mean())
    row=[lbl,tm("throughput_gbps"),tm("drop_rate"),tm("mean_kappa"),tm("link_utilization"),tm("p99_rtt")]
    print(f"{lbl:<25}{row[1]:>14.2f}{row[2]:>10.4f}{row[3]:>8.4f}{row[4]:>8.4f}{row[5]:>10.5f}")
    rows.append(row)
with open(f"{FIGS}/table1_summary.csv","w") as f:
    f.write("Method,Throughput_Gbps,Drop_Rate,Mean_Kappa,Link_Util,P99_RTT_s\n")
    for r in rows: f.write(",".join(str(v) for v in r)+"\n")
print("="*85)
print(f"\nAll 12 figures + table saved to {FIGS}/")
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 18 — Save everything to Google Drive
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CELL18 = r"""
import shutil
DRIVE_OUT = '/content/drive/MyDrive/Rano4_results'
shutil.copytree('results', DRIVE_OUT, dirs_exist_ok=True)
print("All results copied to Google Drive:", DRIVE_OUT)
"""

