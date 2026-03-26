"""
Commodity Portfolio Q-Learning API — V2: Endogenous Supply Dynamics

- Suppliers: fixed cost per unit capacity + variable cost per unit produced
- Suppliers adjust capacity each period based on prior utilisation
- Domestic: fixed cost for maintained capacity + variable cost for production
- New state dimension: supply tightness (240 states total)
- Feedback: storage → reduced offtake → capacity cuts → tighter market → worse crises
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import time

app = FastAPI(title="Commodity QL API v2")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class SupplierIn(BaseModel):
    name: str
    max_capacity: float
    init_capacity: float
    fixed_cost: float
    variable_cost: float
    p_d: float
    p_r: float
    capacity_adjust_rate: float = 0.08

class CountryIn(BaseModel):
    name: str
    demand: float
    elasticity: float
    dom_variable_cost: float
    dom_fixed_cost: float
    dom_max_capacity: float
    dom_init_capacity: float
    max_dom: float
    stor_max: float = 0
    stor_cost: float = 0
    stor_init: float = 0

class RunRequest(BaseModel):
    suppliers: list[SupplierIn]
    countries: list[CountryIn]
    base_spot: float = 55.0
    train_episodes: int = 2000
    eval_sims: int = 300
    ep_length: int = 36

N_SB, N_PB, N_DB, N_TB = 5, 4, 4, 3
N_ST = N_SB * N_PB * N_DB * N_TB
PROF_NAMES = ['diversified','reliable','heavy-spot','heavy-domestic','balanced','min-cost']
STOR_ACTS = ['store','hold','draw']
N_ACT = len(PROF_NAMES) * len(STOR_ACTS)

def d_stor(lv, cap):
    if cap <= 0: return 0
    return min(int(lv / cap * N_SB), N_SB - 1)

def d_price(sp, base):
    r = sp / base
    if r < 0.8: return 0
    if r < 1.3: return 1
    if r < 3.0: return 2
    return 3

def d_disrupt(states):
    f = sum(states) / max(len(states), 1)
    if f < 0.05: return 0
    if f < 0.15: return 1
    if f < 0.30: return 2
    return 3

def d_tight(tot_cap, tot_dem):
    if tot_dem <= 0: return 1
    r = tot_cap / tot_dem
    if r > 1.3: return 0
    if r > 0.95: return 1
    return 2

def s_idx(sb, pb, db, tb):
    return sb * (N_PB * N_DB * N_TB) + pb * (N_DB * N_TB) + db * N_TB + tb

def dec_act(a): return a // 3, a % 3

def get_alloc(pi, sups, max_dom):
    ns = len(sups); nd = 1.0 - max_dom; al = np.zeros(ns + 1)
    if pi == 0: al[:] = nd / (ns + 1)
    elif pi == 1:
        rel = np.array([1.0 / (s.p_d + 0.01) for s in sups]); rel /= rel.sum()
        al[:ns] = rel * nd * 0.85; al[ns] = nd * 0.15
    elif pi == 2: al[:ns] = nd * 0.1 / max(ns, 1); al[ns] = nd * 0.9
    elif pi == 3: al[:] = nd / (ns + 1)
    elif pi == 4: al[:ns] = nd * 0.7 / max(ns, 1); al[ns] = nd * 0.3
    else:
        ip = np.array([1.0 / s.variable_cost for s in sups]); ip /= ip.sum()
        al[:ns] = ip * nd * 0.9; al[ns] = nd * 0.1
    return al

class Env:
    def __init__(self, sups, ctrs, bs, rng):
        self.sups, self.ctrs, self.bs, self.rng = sups, ctrs, bs, rng
        self.ns, self.nc = len(sups), len(ctrs)

    def reset(self):
        self.sup_st = np.zeros(self.ns, dtype=int)
        self.sup_cap = np.array([s.init_capacity for s in self.sups], dtype=float)
        self.dom_cap = np.array([c.dom_init_capacity for c in self.ctrs], dtype=float)
        self.stor_lv = np.array([c.stor_init for c in self.ctrs], dtype=float)
        self.prev_spot = self.bs
        self.prev_sup_util = np.full(self.ns, 0.7)
        self.prev_dom_util = np.full(self.nc, 0.5)
        return self._obs()

    def _obs(self):
        db = d_disrupt(self.sup_st); pb = d_price(self.prev_spot, self.bs)
        op = sum(self.sup_cap[i] for i in range(self.ns) if self.sup_st[i] == 0)
        td = sum(c.demand for c in self.ctrs)
        tb = d_tight(op + sum(self.dom_cap), td)
        return [s_idx(d_stor(self.stor_lv[ci], self.ctrs[ci].stor_max), pb, db, tb) for ci in range(self.nc)]

    def step(self, actions):
        ns, nc, sups, ctrs, bs, rng = self.ns, self.nc, self.sups, self.ctrs, self.bs, self.rng

        # Supplier capacity adjustment
        for i in range(ns):
            s, u, r = sups[i], self.prev_sup_util[i], sups[i].capacity_adjust_rate
            if u < 0.5: self.sup_cap[i] = max(s.max_capacity * 0.2, self.sup_cap[i] * (1 - r * (0.5 - u) / 0.5))
            elif u > 0.85: self.sup_cap[i] = min(s.max_capacity, self.sup_cap[i] * (1 + r * (u - 0.85) / 0.15))

        # Domestic capacity adjustment
        for ci in range(nc):
            c, u, r = ctrs[ci], self.prev_dom_util[ci], 0.06
            if u < 0.3: self.dom_cap[ci] = max(0, self.dom_cap[ci] * (1 - r * (0.3 - u) / 0.3))
            elif u > 0.8: self.dom_cap[ci] = min(c.dom_max_capacity, self.dom_cap[ci] * (1 + r * (u - 0.8) / 0.2))

        # Markov transitions
        for i in range(ns):
            if self.sup_st[i] == 0: self.sup_st[i] = 1 if rng.random() < sups[i].p_d else 0
            else: self.sup_st[i] = 0 if rng.random() < sups[i].p_r else 1

        avail = np.array([self.sup_cap[i] if self.sup_st[i] == 0 else 0 for i in range(ns)])

        # Decode actions
        allocs, stor_dec = [], []
        for ci in range(nc):
            pi, si = dec_act(actions[ci])
            allocs.append(get_alloc(pi, sups, ctrs[ci].max_dom)); stor_dec.append(STOR_ACTS[si])

        # Contracted (variable cost)
        c_qty, c_vc = np.zeros(nc), np.zeros(nc)
        per_sup = np.zeros((nc, ns))
        for ci in range(nc):
            for si in range(ns):
                desired = allocs[ci][si] * ctrs[ci].demand
                f = min(desired, avail[si]) if avail[si] > 0 else 0
                c_qty[ci] += f; c_vc[ci] += f * sups[si].variable_cost; per_sup[ci, si] = f

        # Domestic (capped by capacity)
        dom_qty = np.zeros(nc)
        for ci in range(nc):
            c = ctrs[ci]; da = min(max(0, 1 - allocs[ci].sum()), c.max_dom)
            dom_qty[ci] = min(da * c.demand, self.dom_cap[ci])

        # Storage pre-clearing
        s_draw, s_si = np.zeros(nc), np.zeros(nc)
        for ci in range(nc):
            c = ctrs[ci]
            if stor_dec[ci] == 'draw' and self.stor_lv[ci] > 0:
                dr = min(self.stor_lv[ci], max(0, c.demand - c_qty[ci] - dom_qty[ci]) * 0.6)
                s_draw[ci] = dr; self.stor_lv[ci] -= dr
            elif stor_dec[ci] == 'store':
                s_si[ci] = min(c.stor_max - self.stor_lv[ci], c.demand * 0.12)

        # Spot clearing
        sp_dem = np.array([max(0, ctrs[ci].demand - c_qty[ci] - dom_qty[ci] - s_draw[ci]) + s_si[ci] for ci in range(nc)])
        sp_sup = max(0, avail.sum() - c_qty.sum())
        sp_p, sp_q = bs, np.zeros(nc)
        if sp_dem.sum() > 0 and sp_sup > 0:
            pL, pH = bs * 0.5, bs * 20
            for _ in range(50):
                pm = (pL + pH) / 2
                td = sum(sp_dem[i] * (pm / bs) ** ctrs[i].elasticity for i in range(nc))
                if td > sp_sup: pL = pm
                else: pH = pm
                if abs(td - sp_sup) / max(sp_sup, 1e-9) < 1e-6: break
            sp_p = (pL + pH) / 2
            sp_q = np.array([sp_dem[i] * (sp_p / bs) ** ctrs[i].elasticity for i in range(nc)])
        elif sp_dem.sum() > 0:
            sp_p = bs * 10
            sp_q = np.array([sp_dem[i] * (sp_p / bs) ** ctrs[i].elasticity for i in range(nc)])

        # Storage purchases
        a_st = np.zeros(nc)
        for ci in range(nc):
            if s_si[ci] > 0:
                ratio = s_si[ci] / max(sp_dem[ci], 1e-9)
                a_st[ci] = sp_q[ci] * ratio
                self.stor_lv[ci] = min(ctrs[ci].stor_max, self.stor_lv[ci] + a_st[ci])

        # Update utilisation
        for i in range(ns):
            sold = per_sup[:, i].sum()
            if avail.sum() > 0 and avail[i] > 0: sold += avail[i] / avail.sum() * sp_q.sum()
            self.prev_sup_util[i] = sold / max(self.sup_cap[i], 1e-9)
        for ci in range(nc):
            self.prev_dom_util[ci] = dom_qty[ci] / max(self.dom_cap[ci], 1e-9)

        # Costs
        sup_fc_alloc = np.zeros(nc)
        for si in range(ns):
            tfc = self.sup_cap[si] * sups[si].fixed_cost
            ts = per_sup[:, si].sum()
            if ts > 0:
                for ci in range(nc): sup_fc_alloc[ci] += tfc * (per_sup[ci, si] / ts)

        rewards, details = np.zeros(nc), []
        for ci in range(nc):
            c = ctrs[ci]
            sc = sp_q[ci] * sp_p
            dvc = dom_qty[ci] * c.dom_variable_cost
            dfc = self.dom_cap[ci] * c.dom_fixed_cost
            hc = self.stor_lv[ci] * c.stor_cost
            tc = c_vc[ci] + sup_fc_alloc[ci] + sc + dvc + dfc + hc
            cons = c_qty[ci] + dom_qty[ci] + s_draw[ci] + sp_q[ci] - a_st[ci]
            rewards[ci] = -tc / c.demand
            details.append({
                'avg_cost': tc / max(cons, 1e-9), 'dem_pct': cons / max(c.demand, 1e-9) * 100,
                'stor_lv': float(self.stor_lv[ci]), 'per_sup': per_sup[ci].tolist(),
                'spot_q': float(sp_q[ci]), 'dom_q': float(dom_qty[ci]), 'draw_q': float(s_draw[ci]),
                'action': actions[ci], 'sup_caps': self.sup_cap.tolist(), 'dom_cap': float(self.dom_cap[ci]),
                'dom_fc': float(dfc), 'sup_fc': float(sup_fc_alloc[ci]),
                'contract_vc': float(c_vc[ci]), 'spot_cost': float(sc),
            })
        self.prev_spot = sp_p
        return self._obs(), rewards, sp_p, details

class QAgent:
    def __init__(self, ns, na, lr=0.12, g=0.95, es=1.0, ee=0.03, ed=0.9992):
        self.q = np.zeros((ns, na)); self.lr, self.g, self.eps, self.ee, self.ed = lr, g, es, ee, ed
    def act(self, s, rng, explore=True):
        if explore and rng.random() < self.eps: return int(rng.integers(0, self.q.shape[1]))
        return int(np.argmax(self.q[s]))
    def update(self, s, a, r, ns): self.q[s, a] += self.lr * (r + self.g * np.max(self.q[ns]) - self.q[s, a])
    def decay(self): self.eps = max(self.ee, self.eps * self.ed)

def run_model(sups, ctrs, base_spot, train_eps, eval_sims, ep_len):
    nc, ns = len(ctrs), len(sups)
    rng = np.random.default_rng(42)
    agents = [QAgent(N_ST, N_ACT) for _ in ctrs]
    tr_spot, tr_costs = [], {c.name: [] for c in ctrs}

    for ep in range(train_eps):
        env = Env(sups, ctrs, base_spot, np.random.default_rng(42 + ep))
        obs = env.reset(); ep_c, ep_s = np.zeros(nc), 0
        for t in range(ep_len):
            acts = [agents[ci].act(obs[ci], rng) for ci in range(nc)]
            n_obs, rew, sp, det = env.step(acts)
            for ci in range(nc): agents[ci].update(obs[ci], acts[ci], rew[ci], n_obs[ci]); ep_c[ci] += det[ci]['avg_cost']
            ep_s += sp; obs = n_obs
        for a in agents: a.decay()
        if ep % 30 == 0:
            tr_spot.append(round(ep_s / ep_len, 1))
            for ci, c in enumerate(ctrs): tr_costs[c.name].append(round(ep_c[ci] / ep_len, 1))

    def evl(act_fn):
        spt = [[] for _ in range(ep_len)]
        ct = {c.name: [[] for _ in range(ep_len)] for c in ctrs}
        st = {c.name: [[] for _ in range(ep_len)] for c in ctrs}
        sct = [[] for _ in range(ep_len)]
        dct = {c.name: [[] for _ in range(ep_len)] for c in ctrs}
        sv = {c.name: np.zeros(ns) for c in ctrs}
        spv = {c.name: 0.0 for c in ctrs}; dv = {c.name: 0.0 for c in ctrs}; drv = {c.name: 0.0 for c in ctrs}
        ac = {c.name: np.zeros(N_ACT) for c in ctrs}
        cb = {c.name: {'contract_vc':0,'sup_fc':0,'spot':0,'dom_vc':0,'dom_fc':0,'holding':0} for c in ctrs}

        for sim in range(eval_sims):
            env = Env(sups, ctrs, base_spot, np.random.default_rng(99 + sim))
            obs = env.reset()
            for t in range(ep_len):
                acts = act_fn(obs); obs, _, sp, det = env.step(acts)
                spt[t].append(sp); sct[t].append(sum(det[0]['sup_caps']))
                for ci, c in enumerate(ctrs):
                    d = det[ci]; ct[c.name][t].append(d['avg_cost']); st[c.name][t].append(d['stor_lv'])
                    dct[c.name][t].append(d['dom_cap'])
                    for si in range(ns): sv[c.name][si] += d['per_sup'][si]
                    spv[c.name] += d['spot_q']; dv[c.name] += d['dom_q']; drv[c.name] += d['draw_q']
                    ac[c.name][d['action']] += 1
                    cb[c.name]['contract_vc'] += d['contract_vc']; cb[c.name]['sup_fc'] += d['sup_fc']
                    cb[c.name]['spot'] += d['spot_cost']; cb[c.name]['dom_fc'] += d['dom_fc']
                    cb[c.name]['holding'] += d['stor_lv'] * ctrs[ci].stor_cost

        N = eval_sims * ep_len
        ag = lambda arrs: {'m': [round(np.mean(a), 1) for a in arrs], 'h': [round(np.percentile(a, 95), 1) for a in arrs]}
        fs = [v for a in spt for v in a]; sm = np.mean(fs)
        res = {'spot': ag(spt), 'spotS': {'mean': round(sm, 2), 'std': round(float(np.std(fs)), 2), 'p95': round(float(np.percentile(fs, 95)), 2)},
               'sup_cap': ag(sct), 'countries': {}}

        for ci, c in enumerate(ctrs):
            cn = c.name; fc = [v for a in ct[cn] for v in a]; mc = np.mean(fc)
            svd = {}; tot = 0
            for si in range(ns): v = sv[cn][si] / N; svd[sups[si].name] = round(v, 2); tot += v
            svd['spot'] = round(spv[cn] / N, 2); tot += svd['spot']
            svd['domestic'] = round(dv[cn] / N, 2); tot += svd['domestic']
            svd['storage_draw'] = round(drv[cn] / N, 2); tot += svd['storage_draw']
            mx = {k: round(v / max(tot, 1e-9), 4) for k, v in svd.items()}
            ta = N; pc = np.zeros(len(PROF_NAMES)); sc = np.zeros(3)
            for a in range(N_ACT): pi, si = dec_act(a); pc[pi] += ac[cn][a]; sc[si] += ac[cn][a]
            af = {'allocation': {PROF_NAMES[i]: round(pc[i] / ta, 3) for i in range(len(PROF_NAMES))},
                  'storage': {STOR_ACTS[i]: round(sc[i] / ta, 3) for i in range(3)}}
            bl = np.zeros(ns + 1)
            for a in range(N_ACT): pi, _ = dec_act(a); bl += get_alloc(pi, sups, c.max_dom) * ac[cn][a]
            bl /= ta
            bw = {sups[si].name: round(float(bl[si]), 4) for si in range(ns)}
            bw['spot'] = round(float(bl[ns]), 4); bw['domestic'] = round(max(0, 1 - bl.sum()), 4)
            cbd = {k: round(v / N / max(c.demand, 1), 2) for k, v in cb[cn].items()}
            res['countries'][cn] = {
                'cost': ag(ct[cn]), 'stor': ag(st[cn]), 'dom_cap_ts': ag(dct[cn]),
                's': {'avg_mean': round(mc, 2), 'avg_std': round(float(np.std(fc)), 2), 'demand_met': round(tot / c.demand * 100, 2)},
                'sv': svd, 'mx': mx, 'af': af, 'bw': bw, 'cost_breakdown': cbd,
            }
        return res

    ql = evl(lambda obs: [agents[ci].act(obs[ci], rng, explore=False) for ci in range(nc)])
    nv = evl(lambda obs: [1] * nc)
    return {'tr': {'spot': tr_spot, 'costs': tr_costs}, 'nv': nv, 'ql': ql}

@app.get("/api/health")
def health(): return {"status": "ok"}

@app.post("/api/run")
def run(req: RunRequest):
    t0 = time.time()
    r = run_model(req.suppliers, req.countries, req.base_spot, min(req.train_episodes, 5000), min(req.eval_sims, 1000), min(req.ep_length, 60))
    r['meta'] = {'elapsed_s': round(time.time() - t0, 2), 'train_episodes': req.train_episodes, 'eval_sims': req.eval_sims}
    return r
