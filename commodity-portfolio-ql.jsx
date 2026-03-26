import { useState, useCallback, useRef, useEffect } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Area, AreaChart, BarChart, Bar, Cell, Legend, CartesianGrid, PieChart, Pie } from "recharts";

/* ═══════════════════════════════════════════════════════════════════
   Q-LEARNING ENGINE — ported from Python
   ═══════════════════════════════════════════════════════════════════ */

const N_SB = 5, N_PB = 4, N_DB = 4, N_TB = 3;
const N_ST = N_SB * N_PB * N_DB * N_TB;
const PROF_NAMES = ['diversified','reliable','heavy-spot','heavy-domestic','balanced','min-cost'];
const STOR_ACTS = ['store','hold','draw'];
const N_ACT = PROF_NAMES.length * STOR_ACTS.length;

function mulberry32(seed) {
  let s = seed | 0;
  return () => { s = (s + 0x6D2B79F5) | 0; let t = Math.imul(s ^ (s >>> 15), 1 | s); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

function rngIntegers(rng, max) { return Math.floor(rng() * max); }

const dStor = (lv, cap) => cap <= 0 ? 0 : Math.min(Math.floor(lv / cap * N_SB), N_SB - 1);
const dPrice = (sp, base) => { const r = sp / base; return r < 0.8 ? 0 : r < 1.3 ? 1 : r < 3.0 ? 2 : 3; };
const dDisrupt = (states) => { const f = states.reduce((a, b) => a + b, 0) / Math.max(states.length, 1); return f < 0.05 ? 0 : f < 0.15 ? 1 : f < 0.30 ? 2 : 3; };
const dTight = (totCap, totDem) => { if (totDem <= 0) return 1; const r = totCap / totDem; return r > 1.3 ? 0 : r > 0.95 ? 1 : 2; };
const sIdx = (sb, pb, db, tb) => sb * (N_PB * N_DB * N_TB) + pb * (N_DB * N_TB) + db * N_TB + tb;
const decAct = (a) => [Math.floor(a / 3), a % 3];

function getAlloc(pi, sups, maxDom) {
  const ns = sups.length; const nd = 1.0 - maxDom; const al = new Float64Array(ns + 1);
  if (pi === 0) { al.fill(nd / (ns + 1)); }
  else if (pi === 1) {
    let rel = sups.map(s => 1.0 / (s.p_d + 0.01)); const rs = rel.reduce((a, b) => a + b); rel = rel.map(r => r / rs);
    for (let i = 0; i < ns; i++) al[i] = rel[i] * nd * 0.85; al[ns] = nd * 0.15;
  } else if (pi === 2) { for (let i = 0; i < ns; i++) al[i] = nd * 0.1 / Math.max(ns, 1); al[ns] = nd * 0.9; }
  else if (pi === 3) { al.fill(nd / (ns + 1)); }
  else if (pi === 4) { for (let i = 0; i < ns; i++) al[i] = nd * 0.7 / Math.max(ns, 1); al[ns] = nd * 0.3; }
  else {
    let ip = sups.map(s => 1.0 / s.variable_cost); const ips = ip.reduce((a, b) => a + b); ip = ip.map(v => v / ips);
    for (let i = 0; i < ns; i++) al[i] = ip[i] * nd * 0.9; al[ns] = nd * 0.1;
  }
  return al;
}

class Env {
  constructor(sups, ctrs, bs, rng) { this.sups = sups; this.ctrs = ctrs; this.bs = bs; this.rng = rng; this.ns = sups.length; this.nc = ctrs.length; }
  reset() {
    this.supSt = new Int32Array(this.ns); this.supCap = Float64Array.from(this.sups.map(s => s.init_capacity));
    this.domCap = Float64Array.from(this.ctrs.map(c => c.dom_init_capacity)); this.storLv = Float64Array.from(this.ctrs.map(c => c.stor_init));
    this.prevSpot = this.bs; this.prevSupUtil = new Float64Array(this.ns).fill(0.7); this.prevDomUtil = new Float64Array(this.nc).fill(0.5);
    return this._obs();
  }
  _obs() {
    const db = dDisrupt(this.supSt); const pb = dPrice(this.prevSpot, this.bs);
    let op = 0; for (let i = 0; i < this.ns; i++) if (this.supSt[i] === 0) op += this.supCap[i];
    const td = this.ctrs.reduce((a, c) => a + c.demand, 0); const tb = dTight(op + this.domCap.reduce((a, b) => a + b), td);
    return this.ctrs.map((_, ci) => sIdx(dStor(this.storLv[ci], this.ctrs[ci].stor_max), pb, db, tb));
  }
  step(actions) {
    const { ns, nc, sups, ctrs, bs, rng } = this;
    for (let i = 0; i < ns; i++) { const s = sups[i], u = this.prevSupUtil[i], r = s.capacity_adjust_rate || 0.08;
      if (u < 0.5) this.supCap[i] = Math.max(s.max_capacity * 0.2, this.supCap[i] * (1 - r * (0.5 - u) / 0.5));
      else if (u > 0.85) this.supCap[i] = Math.min(s.max_capacity, this.supCap[i] * (1 + r * (u - 0.85) / 0.15));
    }
    for (let ci = 0; ci < nc; ci++) { const c = ctrs[ci], u = this.prevDomUtil[ci], r = 0.06;
      if (u < 0.3) this.domCap[ci] = Math.max(0, this.domCap[ci] * (1 - r * (0.3 - u) / 0.3));
      else if (u > 0.8) this.domCap[ci] = Math.min(c.dom_max_capacity, this.domCap[ci] * (1 + r * (u - 0.8) / 0.2));
    }
    for (let i = 0; i < ns; i++) {
      if (this.supSt[i] === 0) this.supSt[i] = rng() < sups[i].p_d ? 1 : 0;
      else this.supSt[i] = rng() < sups[i].p_r ? 0 : 1;
    }
    const avail = Float64Array.from(sups.map((_, i) => this.supSt[i] === 0 ? this.supCap[i] : 0));
    const allocs = [], storDec = [];
    for (let ci = 0; ci < nc; ci++) { const [pi, si] = decAct(actions[ci]); allocs.push(getAlloc(pi, sups, ctrs[ci].max_dom)); storDec.push(STOR_ACTS[si]); }
    const cQty = new Float64Array(nc), cVc = new Float64Array(nc), perSup = Array.from({length: nc}, () => new Float64Array(ns));
    for (let ci = 0; ci < nc; ci++) for (let si = 0; si < ns; si++) {
      const desired = allocs[ci][si] * ctrs[ci].demand; const f = avail[si] > 0 ? Math.min(desired, avail[si]) : 0;
      cQty[ci] += f; cVc[ci] += f * sups[si].variable_cost; perSup[ci][si] = f;
    }
    const domQty = new Float64Array(nc);
    for (let ci = 0; ci < nc; ci++) { const c = ctrs[ci]; const da = Math.min(Math.max(0, 1 - allocs[ci].reduce((a, b) => a + b)), c.max_dom); domQty[ci] = Math.min(da * c.demand, this.domCap[ci]); }
    const sDraw = new Float64Array(nc), sSi = new Float64Array(nc);
    for (let ci = 0; ci < nc; ci++) { const c = ctrs[ci];
      if (storDec[ci] === 'draw' && this.storLv[ci] > 0) { const dr = Math.min(this.storLv[ci], Math.max(0, c.demand - cQty[ci] - domQty[ci]) * 0.6); sDraw[ci] = dr; this.storLv[ci] -= dr; }
      else if (storDec[ci] === 'store') { sSi[ci] = Math.min(c.stor_max - this.storLv[ci], c.demand * 0.12); }
    }
    const spDem = Float64Array.from(ctrs.map((c, ci) => Math.max(0, c.demand - cQty[ci] - domQty[ci] - sDraw[ci]) + sSi[ci]));
    const spSup = Math.max(0, avail.reduce((a, b) => a + b) - cQty.reduce((a, b) => a + b));
    let spP = bs; const spQ = new Float64Array(nc);
    if (spDem.reduce((a, b) => a + b) > 0 && spSup > 0) {
      let pL = bs * 0.5, pH = bs * 20;
      for (let it = 0; it < 50; it++) { const pm = (pL + pH) / 2; let td = 0; for (let i = 0; i < nc; i++) td += spDem[i] * Math.pow(pm / bs, ctrs[i].elasticity);
        if (td > spSup) pL = pm; else pH = pm; if (Math.abs(td - spSup) / Math.max(spSup, 1e-9) < 1e-6) break;
      } spP = (pL + pH) / 2; for (let i = 0; i < nc; i++) spQ[i] = spDem[i] * Math.pow(spP / bs, ctrs[i].elasticity);
    } else if (spDem.reduce((a, b) => a + b) > 0) { spP = bs * 10; for (let i = 0; i < nc; i++) spQ[i] = spDem[i] * Math.pow(spP / bs, ctrs[i].elasticity); }
    const aSt = new Float64Array(nc);
    for (let ci = 0; ci < nc; ci++) if (sSi[ci] > 0) { const ratio = sSi[ci] / Math.max(spDem[ci], 1e-9); aSt[ci] = spQ[ci] * ratio; this.storLv[ci] = Math.min(ctrs[ci].stor_max, this.storLv[ci] + aSt[ci]); }
    for (let i = 0; i < ns; i++) { let sold = 0; for (let ci = 0; ci < nc; ci++) sold += perSup[ci][i]; const as2 = avail.reduce((a, b) => a + b);
      if (as2 > 0 && avail[i] > 0) sold += avail[i] / as2 * spQ.reduce((a, b) => a + b); this.prevSupUtil[i] = sold / Math.max(this.supCap[i], 1e-9);
    }
    for (let ci = 0; ci < nc; ci++) this.prevDomUtil[ci] = domQty[ci] / Math.max(this.domCap[ci], 1e-9);
    const supFcAlloc = new Float64Array(nc);
    for (let si = 0; si < ns; si++) { const tfc = this.supCap[si] * sups[si].fixed_cost; let ts = 0; for (let ci = 0; ci < nc; ci++) ts += perSup[ci][si];
      if (ts > 0) for (let ci = 0; ci < nc; ci++) supFcAlloc[ci] += tfc * (perSup[ci][si] / ts);
    }
    const rewards = new Float64Array(nc), details = [];
    for (let ci = 0; ci < nc; ci++) { const c = ctrs[ci]; const sc = spQ[ci] * spP; const dvc = domQty[ci] * c.dom_variable_cost; const dfc = this.domCap[ci] * c.dom_fixed_cost;
      const hc = this.storLv[ci] * c.stor_cost; const tc = cVc[ci] + supFcAlloc[ci] + sc + dvc + dfc + hc; const cons = cQty[ci] + domQty[ci] + sDraw[ci] + spQ[ci] - aSt[ci];
      rewards[ci] = -tc / c.demand;
      details.push({ avg_cost: tc / Math.max(cons, 1e-9), dem_pct: cons / Math.max(c.demand, 1e-9) * 100, stor_lv: this.storLv[ci], per_sup: Array.from(perSup[ci]),
        spot_q: spQ[ci], dom_q: domQty[ci], draw_q: sDraw[ci], action: actions[ci], sup_caps: Array.from(this.supCap), dom_cap: this.domCap[ci],
        dom_fc: dfc, sup_fc: supFcAlloc[ci], contract_vc: cVc[ci], spot_cost: sc });
    }
    this.prevSpot = spP; return [this._obs(), rewards, spP, details];
  }
}

class QAgent {
  constructor(ns, na) { this.q = Array.from({length: ns}, () => new Float64Array(na)); this.lr = 0.12; this.g = 0.95; this.eps = 1.0; this.ee = 0.03; this.ed = 0.9992; }
  act(s, rng, explore = true) { if (explore && rng() < this.eps) return rngIntegers(rng, this.q[0].length); let mx = -Infinity, mi = 0; for (let i = 0; i < this.q[s].length; i++) if (this.q[s][i] > mx) { mx = this.q[s][i]; mi = i; } return mi; }
  update(s, a, r, ns) { let mx = -Infinity; for (let i = 0; i < this.q[ns].length; i++) if (this.q[ns][i] > mx) mx = this.q[ns][i]; this.q[s][a] += this.lr * (r + this.g * mx - this.q[s][a]); }
  decay() { this.eps = Math.max(this.ee, this.eps * this.ed); }
}

function runModel(sups, ctrs, baseSpot, trainEps, evalSims, epLen, onProgress) {
  const nc = ctrs.length, ns = sups.length;
  const rng = mulberry32(42);
  const agents = ctrs.map(() => new QAgent(N_ST, N_ACT));
  const trSpot = [], trCosts = {}; ctrs.forEach(c => trCosts[c.name] = []);

  for (let ep = 0; ep < trainEps; ep++) {
    const env = new Env(sups, ctrs, baseSpot, mulberry32(42 + ep));
    let obs = env.reset(), epC = new Float64Array(nc), epS = 0;
    for (let t = 0; t < epLen; t++) {
      const acts = agents.map((a, ci) => a.act(obs[ci], rng));
      const [nObs, rew, sp, det] = env.step(acts);
      for (let ci = 0; ci < nc; ci++) { agents[ci].update(obs[ci], acts[ci], rew[ci], nObs[ci]); epC[ci] += det[ci].avg_cost; }
      epS += sp; obs = nObs;
    }
    agents.forEach(a => a.decay());
    if (ep % 30 === 0) { trSpot.push(Math.round(epS / epLen * 10) / 10); ctrs.forEach((c, ci) => trCosts[c.name].push(Math.round(epC[ci] / epLen * 10) / 10)); }
  }

  function evl(actFn) {
    const spt = Array.from({length: epLen}, () => []), sct = Array.from({length: epLen}, () => []);
    const ct = {}, st = {}, dct = {}, sv = {}, spv = {}, dv = {}, drv = {}, ac = {}, cb = {};
    ctrs.forEach(c => { ct[c.name] = Array.from({length: epLen}, () => []); st[c.name] = Array.from({length: epLen}, () => []);
      dct[c.name] = Array.from({length: epLen}, () => []); sv[c.name] = new Float64Array(ns); spv[c.name] = 0; dv[c.name] = 0; drv[c.name] = 0;
      ac[c.name] = new Float64Array(N_ACT); cb[c.name] = {contract_vc:0,sup_fc:0,spot:0,dom_vc:0,dom_fc:0,holding:0};
    });
    for (let sim = 0; sim < evalSims; sim++) {
      const env = new Env(sups, ctrs, baseSpot, mulberry32(99 + sim)); let obs = env.reset();
      for (let t = 0; t < epLen; t++) {
        const acts = actFn(obs); const [nObs, , sp, det] = env.step(acts); obs = nObs;
        spt[t].push(sp); sct[t].push(det[0].sup_caps.reduce((a, b) => a + b));
        ctrs.forEach((c, ci) => { const d = det[ci]; ct[c.name][t].push(d.avg_cost); st[c.name][t].push(d.stor_lv); dct[c.name][t].push(d.dom_cap);
          for (let si = 0; si < ns; si++) sv[c.name][si] += d.per_sup[si]; spv[c.name] += d.spot_q; dv[c.name] += d.dom_q; drv[c.name] += d.draw_q;
          ac[c.name][d.action] += 1; cb[c.name].contract_vc += d.contract_vc; cb[c.name].sup_fc += d.sup_fc; cb[c.name].spot += d.spot_cost; cb[c.name].dom_fc += d.dom_fc;
          cb[c.name].holding += d.stor_lv * ctrs[ci].stor_cost;
        });
      }
    }
    const N = evalSims * epLen;
    const mean = arr => arr.reduce((a, b) => a + b, 0) / arr.length;
    const pct = (arr, p) => { const s = [...arr].sort((a, b) => a - b); return s[Math.floor(s.length * p / 100)]; };
    const ag = arrs => ({ m: arrs.map(a => Math.round(mean(a) * 10) / 10), h: arrs.map(a => Math.round(pct(a, 95) * 10) / 10) });
    const fs = spt.flat(); const sm = mean(fs);
    const res = { spot: ag(spt), spotS: { mean: Math.round(sm * 100) / 100, std: Math.round(Math.sqrt(fs.reduce((a, v) => a + (v - sm) ** 2, 0) / fs.length) * 100) / 100, p95: Math.round(pct(fs, 95) * 100) / 100 }, sup_cap: ag(sct), countries: {} };
    ctrs.forEach((c, ci) => {
      const cn = c.name; const fc = ct[cn].flat(); const mc = mean(fc);
      const svd = {}; let tot = 0;
      for (let si = 0; si < ns; si++) { const v = sv[cn][si] / N; svd[sups[si].name] = Math.round(v * 100) / 100; tot += v; }
      svd.spot = Math.round(spv[cn] / N * 100) / 100; tot += svd.spot; svd.domestic = Math.round(dv[cn] / N * 100) / 100; tot += svd.domestic;
      svd.storage_draw = Math.round(drv[cn] / N * 100) / 100; tot += svd.storage_draw;
      const mx = {}; Object.entries(svd).forEach(([k, v]) => mx[k] = Math.round(v / Math.max(tot, 1e-9) * 10000) / 10000);
      const ta = N; const pc = new Float64Array(PROF_NAMES.length), sc = new Float64Array(3);
      for (let a = 0; a < N_ACT; a++) { const [pi, si] = decAct(a); pc[pi] += ac[cn][a]; sc[si] += ac[cn][a]; }
      const af = { allocation: Object.fromEntries(PROF_NAMES.map((k, i) => [k, Math.round(pc[i] / ta * 1000) / 1000])), storage: Object.fromEntries(STOR_ACTS.map((k, i) => [k, Math.round(sc[i] / ta * 1000) / 1000])) };
      const bl = new Float64Array(ns + 1);
      for (let a = 0; a < N_ACT; a++) { const [pi] = decAct(a); const al = getAlloc(pi, sups, c.max_dom); for (let j = 0; j <= ns; j++) bl[j] += al[j] * ac[cn][a]; }
      for (let j = 0; j <= ns; j++) bl[j] /= ta;
      const bw = {}; for (let si = 0; si < ns; si++) bw[sups[si].name] = Math.round(bl[si] * 10000) / 10000;
      bw.spot = Math.round(bl[ns] * 10000) / 10000; bw.domestic = Math.round(Math.max(0, 1 - bl.reduce((a, b) => a + b)) * 10000) / 10000;
      const cbd = {}; Object.entries(cb[cn]).forEach(([k, v]) => cbd[k] = Math.round(v / N / Math.max(c.demand, 1) * 100) / 100);
      const stdfc = Math.sqrt(fc.reduce((a, v) => a + (v - mc) ** 2, 0) / fc.length);
      res.countries[cn] = { cost: ag(ct[cn]), stor: ag(st[cn]), dom_cap_ts: ag(dct[cn]),
        s: { avg_mean: Math.round(mc * 100) / 100, avg_std: Math.round(stdfc * 100) / 100, demand_met: Math.round(tot / c.demand * 100 * 100) / 100 },
        sv: svd, mx, af, bw, cost_breakdown: cbd };
    });
    return res;
  }
  const ql = evl(obs => agents.map((a, ci) => a.act(obs[ci], rng, false)));
  const nv = evl(() => new Array(nc).fill(1));
  return { tr: { spot: trSpot, costs: trCosts }, nv, ql };
}

/* ═══════════════════════════════════════════════════════════════════
   DEFAULTS
   ═══════════════════════════════════════════════════════════════════ */
const D_SUP = [
  {name:"Gulf-A",max_capacity:120,init_capacity:110,fixed_cost:8,variable_cost:48,p_d:0.03,p_r:0.60},
  {name:"Gulf-B",max_capacity:100,init_capacity:90,fixed_cost:9,variable_cost:50,p_d:0.04,p_r:0.55},
  {name:"Caspian-1",max_capacity:80,init_capacity:70,fixed_cost:10,variable_cost:52,p_d:0.06,p_r:0.40},
  {name:"West-Africa",max_capacity:70,init_capacity:60,fixed_cost:11,variable_cost:55,p_d:0.08,p_r:0.35},
  {name:"LatAm-North",max_capacity:60,init_capacity:55,fixed_cost:8,variable_cost:47,p_d:0.05,p_r:0.50},
  {name:"LatAm-South",max_capacity:50,init_capacity:45,fixed_cost:9,variable_cost:49,p_d:0.04,p_r:0.55},
  {name:"Arctic-Basin",max_capacity:40,init_capacity:30,fixed_cost:14,variable_cost:62,p_d:0.10,p_r:0.25},
  {name:"SE-Asia",max_capacity:55,init_capacity:50,fixed_cost:10,variable_cost:51,p_d:0.05,p_r:0.45},
  {name:"Australasia",max_capacity:45,init_capacity:42,fixed_cost:11,variable_cost:54,p_d:0.03,p_r:0.65},
  {name:"North-Sea",max_capacity:35,init_capacity:33,fixed_cost:12,variable_cost:58,p_d:0.02,p_r:0.70},
];
const D_CTR = [
  {name:"Industria",demand:200,elasticity:-0.25,dom_variable_cost:72,dom_fixed_cost:10,dom_max_capacity:80,dom_init_capacity:60,max_dom:0.30,stor_max:80,stor_cost:2.0,stor_init:20},
  {name:"Pacifica",demand:150,elasticity:-0.35,dom_variable_cost:68,dom_fixed_cost:9,dom_max_capacity:50,dom_init_capacity:30,max_dom:0.20,stor_max:50,stor_cost:2.5,stor_init:10},
  {name:"Europa",demand:180,elasticity:-0.30,dom_variable_cost:75,dom_fixed_cost:11,dom_max_capacity:70,dom_init_capacity:45,max_dom:0.25,stor_max:70,stor_cost:1.8,stor_init:15},
  {name:"Emergent",demand:120,elasticity:-0.40,dom_variable_cost:65,dom_fixed_cost:8,dom_max_capacity:30,dom_init_capacity:18,max_dom:0.15,stor_max:30,stor_cost:3.0,stor_init:5},
];
const D_SPOT = 55.0;

/* ═══════════════════════════════════════════════════════════════════
   DESIGN TOKENS — editorial dark palette
   ═══════════════════════════════════════════════════════════════════ */
const T = {
  bg: '#0c0f14', card: '#151921', cardBorder: '#1e2533', cardHover: '#1a2230',
  text: '#c8cdd8', textMuted: '#6b7280', textDim: '#3d4554',
  accent: '#4f8fea', accentSoft: '#4f8fea22', green: '#34d399', greenBg: '#34d39918',
  red: '#f87171', redBg: '#f8717118', amber: '#fbbf24', amberBg: '#fbbf2418',
  gridLine: '#1e2533', inputBg: '#0f1319', inputBorder: '#252d3d',
  pillActive: '#4f8fea', pillActiveTxt: '#0c0f14',
};
const XCC = ["#f87171","#4f8fea","#fbbf24","#34d399","#a78bfa","#fb923c","#2dd4bf","#94a3b8"];
const ALLOC_C = {"diversified":"#94a3b8","reliable":"#4f8fea","heavy-spot":"#a78bfa","heavy-domestic":"#fbbf24","balanced":"#34d399","min-cost":"#f87171"};
const STOR_CC = {"store":"#4f8fea","hold":"#94a3b8","draw":"#f87171"};
const MIX_C = ["#f87171","#4f8fea","#fbbf24","#34d399","#a78bfa","#fb923c","#2dd4bf","#94a3b8","#e879f9","#38bdf8","#6b7280","#c084fc","#f97316"];
const TABS = ["Parameters","Overview","Spot Market","Costs","Supply Mix","Storage","Strategy","Training"];
const ff = (v, d=1) => typeof v === 'number' ? v.toFixed(d) : v;
const gcc = (n, i) => ({"Industria":"#f87171","Pacifica":"#4f8fea","Europa":"#fbbf24","Emergent":"#34d399"}[n] || XCC[i % XCC.length]);

/* ═══════════════════════════════════════════════════════════════════
   UI PRIMITIVES
   ═══════════════════════════════════════════════════════════════════ */
function Tip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background: T.card, border: `1px solid ${T.cardBorder}`, borderRadius: 8, padding: '10px 16px', fontSize: 11, boxShadow: '0 8px 32px #0006', backdropFilter: 'blur(12px)' }}>
      <div style={{ color: T.textMuted, fontWeight: 600, marginBottom: 6, fontSize: 10, textTransform: 'uppercase', letterSpacing: '.06em' }}>{label}</div>
      {payload.filter(p => p.name && p.value !== undefined).map((p, i) => (
        <div key={i} style={{ color: p.color || T.text, marginTop: 2 }}>{p.name}: <b>{ff(p.value, 2)}</b></div>
      ))}
    </div>
  );
}

function Stat({ label, nv, ql, unit = '', lower = true }) {
  const d = (ql - nv) / Math.abs(nv || 1) * 100;
  const good = lower ? d < 0 : d > 0;
  return (
    <div style={{ background: T.card, borderRadius: 10, padding: '16px 20px', flex: 1, minWidth: 140, border: `1px solid ${T.cardBorder}` }}>
      <div style={{ fontSize: 9, color: T.textMuted, textTransform: 'uppercase', letterSpacing: '.1em', marginBottom: 10, fontWeight: 600 }}>{label}</div>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 10, flexWrap: 'wrap' }}>
        <span style={{ fontSize: 24, fontWeight: 700, color: T.text, fontFamily: "'JetBrains Mono', monospace" }}>{ff(ql)}{unit}</span>
        <span style={{ fontSize: 10, fontWeight: 700, color: good ? T.green : T.red, background: good ? T.greenBg : T.redBg, padding: '3px 10px', borderRadius: 20 }}>{d > 0 ? '+' : ''}{ff(d)}%</span>
      </div>
      <div style={{ fontSize: 10, color: T.textDim, marginTop: 6, fontFamily: "'JetBrains Mono', monospace" }}>naive: {ff(nv)}{unit}</div>
    </div>
  );
}

function Pill({ active, onClick, children, disabled }) {
  return (
    <button onClick={onClick} disabled={disabled} style={{ padding: '8px 18px', border: 'none', borderRadius: 6, cursor: disabled ? 'not-allowed' : 'pointer', fontSize: 11.5,
      fontWeight: active ? 700 : 500, background: active ? T.pillActive : 'transparent', color: active ? T.pillActiveTxt : disabled ? T.textDim : T.textMuted,
      transition: 'all .2s', opacity: disabled ? .35 : 1, letterSpacing: '.02em', fontFamily: "'JetBrains Mono', monospace" }}>
      {children}
    </button>
  );
}

function Card({ children, style }) {
  return <div style={{ background: T.card, borderRadius: 12, border: `1px solid ${T.cardBorder}`, padding: '20px 18px', ...style }}>{children}</div>;
}

const IS = { background: T.inputBg, border: `1px solid ${T.inputBorder}`, borderRadius: 6, color: T.text, padding: '7px 9px', fontSize: 12, width: '100%', fontFamily: "'JetBrains Mono', monospace", boxSizing: 'border-box', outline: 'none', transition: 'border-color .2s' };

function NI({ value, onChange, step, min, max, style: es }) {
  return <input type="number" value={value} onChange={e => onChange(parseFloat(e.target.value) || 0)} step={step} min={min} max={max} style={{ ...IS, ...es }}
    onFocus={e => { e.target.style.borderColor = T.accent; }} onBlur={e => { e.target.style.borderColor = T.inputBorder; }} />;
}

const GR = <CartesianGrid strokeDasharray="3 3" stroke={T.gridLine} />;
const XP = { tick: { fill: T.textMuted, fontSize: 10, fontFamily: "'JetBrains Mono', monospace" }, stroke: T.gridLine };
const YP = { tick: { fill: T.textMuted, fontSize: 10, fontFamily: "'JetBrains Mono', monospace" }, stroke: T.gridLine };

/* ═══════════════════════════════════════════════════════════════════
   PARAMETER TAB
   ═══════════════════════════════════════════════════════════════════ */
function ParamTab({ sups, setSups, ctrs, setCtrs, bSpot, setBSpot, onRun, running, progress }) {
  const uS = (i, k, v) => setSups(sups.map((s, j) => j === i ? { ...s, [k]: v } : s));
  const uC = (i, k, v) => setCtrs(ctrs.map((c, j) => j === i ? { ...c, [k]: v } : c));
  const th = { padding: '10px 5px', textAlign: 'left', color: T.textMuted, fontWeight: 600, fontSize: 9, textTransform: 'uppercase', letterSpacing: '.08em', whiteSpace: 'nowrap', borderBottom: `1px solid ${T.cardBorder}` };
  const td = { padding: '5px 3px' };
  return (
    <div>
      <div style={{ display: 'flex', gap: 16, alignItems: 'center', marginBottom: 24, flexWrap: 'wrap' }}>
        <div>
          <label style={{ fontSize: 9, color: T.textMuted, textTransform: 'uppercase', letterSpacing: '.08em', display: 'block', marginBottom: 5, fontWeight: 600 }}>Base Spot Price</label>
          <NI value={bSpot} onChange={setBSpot} step={1} min={1} style={{ width: 90 }} />
        </div>
        <div style={{ marginLeft: 'auto', display: 'flex', gap: 10 }}>
          <button onClick={() => { setSups(D_SUP.map(s => ({ ...s }))); setCtrs(D_CTR.map(c => ({ ...c }))); setBSpot(D_SPOT); }}
            style={{ background: 'transparent', border: `1px solid ${T.cardBorder}`, borderRadius: 8, color: T.textMuted, padding: '10px 20px', fontSize: 11, cursor: 'pointer', fontFamily: "'JetBrains Mono', monospace", fontWeight: 500 }}>Reset</button>
          <button onClick={onRun} disabled={running} style={{ background: running ? T.cardBorder : T.accent, border: 'none', borderRadius: 8, color: running ? T.textMuted : T.bg,
            padding: '10px 28px', fontSize: 12, fontWeight: 700, cursor: running ? 'wait' : 'pointer', boxShadow: running ? 'none' : `0 4px 20px ${T.accentSoft}`,
            fontFamily: "'JetBrains Mono', monospace", letterSpacing: '.02em', transition: 'all .2s' }}>
            {running ? progress : 'Train & Evaluate'}
          </button>
        </div>
      </div>
      <Card style={{ marginBottom: 20 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
          <div style={{ fontSize: 14, fontWeight: 700, color: T.text }}>Suppliers <span style={{ color: T.textMuted, fontWeight: 400, fontSize: 12 }}>({sups.length})</span></div>
          <button onClick={() => sups.length < 15 && setSups([...sups, { name: `Sup-${sups.length + 1}`, max_capacity: 50, init_capacity: 45, fixed_cost: 10, variable_cost: 55, p_d: 0.05, p_r: 0.5 }])}
            style={{ background: T.greenBg, border: `1px solid ${T.green}33`, borderRadius: 6, color: T.green, padding: '6px 16px', fontSize: 11, cursor: 'pointer', fontWeight: 600, fontFamily: "'JetBrains Mono', monospace" }}>+ Add</button>
        </div>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
            <thead><tr>{["Name","Max Cap","Init Cap","Fix $/u","Var $/u","P(Disrupt)","P(Recover)","SS Off%",""].map(h => <th key={h} style={th}>{h}</th>)}</tr></thead>
            <tbody>{sups.map((s, i) => { const ss = s.p_d / (s.p_d + s.p_r); return (
              <tr key={i} style={{ borderBottom: `1px solid ${T.cardBorder}` }}>
                <td style={td}><input type="text" value={s.name} onChange={e => uS(i, 'name', e.target.value)} style={{ ...IS, width: 95, fontFamily: "'JetBrains Mono', monospace" }} /></td>
                <td style={td}><NI value={s.max_capacity} onChange={v => uS(i, 'max_capacity', v)} step={5} min={1} style={{ width: 60 }} /></td>
                <td style={td}><NI value={s.init_capacity} onChange={v => uS(i, 'init_capacity', v)} step={5} min={1} style={{ width: 60 }} /></td>
                <td style={td}><NI value={s.fixed_cost} onChange={v => uS(i, 'fixed_cost', v)} step={1} min={0} style={{ width: 60 }} /></td>
                <td style={td}><NI value={s.variable_cost} onChange={v => uS(i, 'variable_cost', v)} step={1} min={1} style={{ width: 60 }} /></td>
                <td style={td}><NI value={s.p_d} onChange={v => uS(i, 'p_d', v)} step={0.01} min={0} max={1} style={{ width: 60 }} /></td>
                <td style={td}><NI value={s.p_r} onChange={v => uS(i, 'p_r', v)} step={0.05} min={0.01} max={1} style={{ width: 60 }} /></td>
                <td style={{ ...td, fontFamily: "'JetBrains Mono', monospace", fontWeight: 700, color: ss > .15 ? T.red : ss > .08 ? T.amber : T.green, textAlign: 'center', fontSize: 11 }}>{(ss * 100).toFixed(1)}%</td>
                <td style={td}><button onClick={() => sups.length > 2 && setSups(sups.filter((_, j) => j !== i))} style={{ background: 'none', border: 'none', color: T.red, cursor: 'pointer', fontSize: 14, opacity: sups.length <= 2 ? .3 : 1, fontWeight: 700 }}>✕</button></td>
              </tr>);
            })}</tbody>
          </table>
        </div>
      </Card>
      <Card>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
          <div style={{ fontSize: 14, fontWeight: 700, color: T.text }}>Countries <span style={{ color: T.textMuted, fontWeight: 400, fontSize: 12 }}>({ctrs.length})</span></div>
          <button onClick={() => ctrs.length < 8 && setCtrs([...ctrs, { name: `Country-${ctrs.length + 1}`, demand: 100, elasticity: -.3, dom_variable_cost: 70, dom_fixed_cost: 10, dom_max_capacity: 40, dom_init_capacity: 30, max_dom: .2, stor_max: 40, stor_cost: 2, stor_init: 10 }])}
            style={{ background: T.greenBg, border: `1px solid ${T.green}33`, borderRadius: 6, color: T.green, padding: '6px 16px', fontSize: 11, cursor: 'pointer', fontWeight: 600, fontFamily: "'JetBrains Mono', monospace" }}>+ Add</button>
        </div>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
            <thead><tr>{["Name","Demand","Elast.","Dom Var$","Dom Fix$","Dom Max","Dom Init","Max Dom%","Stor Cap","$/per","Init",""].map(h => <th key={h} style={th}>{h}</th>)}</tr></thead>
            <tbody>{ctrs.map((c, i) => (
              <tr key={i} style={{ borderBottom: `1px solid ${T.cardBorder}` }}>
                <td style={td}><input type="text" value={c.name} onChange={e => uC(i, 'name', e.target.value)} style={{ ...IS, width: 95, fontFamily: "'JetBrains Mono', monospace" }} /></td>
                <td style={td}><NI value={c.demand} onChange={v => uC(i, 'demand', v)} step={10} min={1} style={{ width: 58 }} /></td>
                <td style={td}><NI value={c.elasticity} onChange={v => uC(i, 'elasticity', v)} step={.05} min={-2} max={0} style={{ width: 58 }} /></td>
                <td style={td}><NI value={c.dom_variable_cost} onChange={v => uC(i, 'dom_variable_cost', v)} step={1} min={1} style={{ width: 58 }} /></td>
                <td style={td}><NI value={c.dom_fixed_cost} onChange={v => uC(i, 'dom_fixed_cost', v)} step={1} min={0} style={{ width: 58 }} /></td>
                <td style={td}><NI value={c.dom_max_capacity} onChange={v => uC(i, 'dom_max_capacity', v)} step={5} min={0} style={{ width: 58 }} /></td>
                <td style={td}><NI value={c.dom_init_capacity} onChange={v => uC(i, 'dom_init_capacity', v)} step={5} min={0} style={{ width: 58 }} /></td>
                <td style={td}><NI value={c.max_dom} onChange={v => uC(i, 'max_dom', v)} step={.05} min={0} max={1} style={{ width: 58 }} /></td>
                <td style={td}><NI value={c.stor_max} onChange={v => uC(i, 'stor_max', v)} step={5} min={0} style={{ width: 55 }} /></td>
                <td style={td}><NI value={c.stor_cost} onChange={v => uC(i, 'stor_cost', v)} step={.5} min={0} style={{ width: 55 }} /></td>
                <td style={td}><NI value={c.stor_init} onChange={v => uC(i, 'stor_init', v)} step={5} min={0} style={{ width: 55 }} /></td>
                <td style={td}><button onClick={() => ctrs.length > 2 && setCtrs(ctrs.filter((_, j) => j !== i))} style={{ background: 'none', border: 'none', color: T.red, cursor: 'pointer', fontSize: 14, opacity: ctrs.length <= 2 ? .3 : 1, fontWeight: 700 }}>✕</button></td>
              </tr>
            ))}</tbody>
          </table>
        </div>
      </Card>
      <div style={{ marginTop: 20, padding: 16, background: T.accentSoft, borderRadius: 10, fontSize: 11, color: T.textMuted, lineHeight: 1.8, border: `1px solid ${T.accent}22` }}>
        <b style={{ color: T.accent }}>In-browser training</b> — runs 3,000 Q-learning episodes and 400 Monte Carlo evaluations entirely in your browser using JavaScript. No server needed. Takes ~5–15 seconds depending on your device.
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════
   RESULTS TABS
   ═══════════════════════════════════════════════════════════════════ */
function Overview({ R, ctrs }) {
  const CN = ctrs.map(c => c.name);
  return (<div>
    <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap', marginBottom: 26 }}>
      <Stat label="Avg Spot" nv={R.nv.spotS.mean} ql={R.ql.spotS.mean} />
      <Stat label="Spot Vol" nv={R.nv.spotS.std} ql={R.ql.spotS.std} />
      <Stat label="Spot P95" nv={R.nv.spotS.p95} ql={R.ql.spotS.p95} />
    </div>
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(240px,1fr))', gap: 14 }}>
      {CN.map((n, ci) => { const ns = R.nv.countries[n]?.s, qs = R.ql.countries[n]?.s; if (!ns || !qs) return null; const col = gcc(n, ci); return (
        <div key={n} style={{ background: T.card, borderRadius: 10, padding: 18, borderLeft: `3px solid ${col}`, border: `1px solid ${T.cardBorder}`, borderLeftColor: col }}>
          <div style={{ fontSize: 14, fontWeight: 700, color: col, marginBottom: 10 }}>{n}</div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, fontSize: 11 }}>
            <span style={{ color: T.textMuted }}>Avg Cost</span><span style={{ fontFamily: "'JetBrains Mono', monospace", fontWeight: 600, color: T.text }}>{ff(qs.avg_mean)} <span style={{ color: T.textDim }}>({ff(ns.avg_mean)})</span></span>
            <span style={{ color: T.textMuted }}>Saving</span><span style={{ fontFamily: "'JetBrains Mono', monospace", fontWeight: 700, color: T.green }}>{ff((1 - qs.avg_mean / ns.avg_mean) * 100)}%</span>
          </div>
        </div>);
      })}
    </div>
  </div>);
}

function SpotTab({ R }) {
  const d = R.nv.spot.m.map((v, i) => ({ t: i + 1, naive: v, nv95: R.nv.spot.h[i], ql: R.ql.spot.m[i], ql95: R.ql.spot.h[i] }));
  return (<Card><ResponsiveContainer width="100%" height={370}>
    <AreaChart data={d} margin={{ left: 10, right: 10, top: 5, bottom: 5 }}>{GR}<XAxis dataKey="t" {...XP} /><YAxis {...YP} /><Tooltip content={<Tip />} />
      <Area dataKey="nv95" stroke="none" fill={T.red} fillOpacity={.08} name="Naive P95" /><Area dataKey="ql95" stroke="none" fill={T.accent} fillOpacity={.08} name="QL P95" />
      <Line type="monotone" dataKey="naive" stroke={T.red} strokeWidth={2.5} dot={false} name="Naive" /><Line type="monotone" dataKey="ql" stroke={T.accent} strokeWidth={2.5} dot={false} name="Q-Learned" />
      <Legend wrapperStyle={{ fontSize: 11, paddingTop: 10, fontFamily: "'JetBrains Mono', monospace" }} /></AreaChart></ResponsiveContainer></Card>);
}

function CostTab({ R, ctrs }) {
  const CN = ctrs.map(c => c.name); const [sel, setSel] = useState(0); const n = CN[sel];
  const nc = R.nv.countries[n], qc = R.ql.countries[n]; if (!nc || !qc) return null;
  const d = nc.cost.m.map((v, i) => ({ t: i + 1, naive: v, nv95: nc.cost.h[i], ql: qc.cost.m[i], ql95: qc.cost.h[i] }));
  return (<div>
    <div style={{ display: 'flex', gap: 4, marginBottom: 18, background: T.card, borderRadius: 8, padding: 4, border: `1px solid ${T.cardBorder}`, width: 'fit-content' }}>
      {CN.map((c, i) => <Pill key={c} active={sel === i} onClick={() => setSel(i)}>{c}</Pill>)}</div>
    <Card><ResponsiveContainer width="100%" height={320}>
      <AreaChart data={d} margin={{ left: 10, right: 10, top: 5, bottom: 5 }}>{GR}<XAxis dataKey="t" {...XP} /><YAxis {...YP} domain={['auto','auto']} /><Tooltip content={<Tip />} />
        <Area dataKey="nv95" stroke="none" fill={T.red} fillOpacity={.07} name="Naive P95" /><Area dataKey="ql95" stroke="none" fill={T.accent} fillOpacity={.07} name="QL P95" />
        <Line type="monotone" dataKey="naive" stroke={T.red} strokeWidth={2.5} dot={false} name="Naive" /><Line type="monotone" dataKey="ql" stroke={T.accent} strokeWidth={2.5} dot={false} name="Q-Learned" />
        <Legend wrapperStyle={{ fontSize: 11, paddingTop: 10, fontFamily: "'JetBrains Mono', monospace" }} /></AreaChart></ResponsiveContainer></Card>
    <div style={{ display: 'flex', gap: 14, marginTop: 18, flexWrap: 'wrap' }}><Stat label="Mean Cost" nv={nc.s.avg_mean} ql={qc.s.avg_mean} /><Stat label="Cost Vol" nv={nc.s.avg_std} ql={qc.s.avg_std} /></div>
  </div>);
}

function MixTab({ R, ctrs, sups }) {
  const CN = ctrs.map(c => c.name), SN = sups.map(s => s.name); const [sel, setSel] = useState(0); const n = CN[sel];
  const sources = [...SN, 'spot', 'domestic', 'storage_draw'];
  const nMx = R.nv.countries[n]?.mx || {}, qMx = R.ql.countries[n]?.mx || {};
  const barD = sources.filter(s => ((nMx[s]||0) + (qMx[s]||0)) > .005).map(s => ({ name: s, naive: Math.round((nMx[s]||0)*1000)/10, ql: Math.round((qMx[s]||0)*1000)/10 }));
  const pieD = sources.filter(s => (qMx[s]||0) > .01).map((s, i) => ({ name: s, value: Math.round((qMx[s]||0)*1000)/10, fill: MIX_C[i % MIX_C.length] }));
  return (<div>
    <div style={{ display: 'flex', gap: 4, marginBottom: 18, background: T.card, borderRadius: 8, padding: 4, border: `1px solid ${T.cardBorder}`, width: 'fit-content' }}>
      {CN.map((c, i) => <Pill key={c} active={sel === i} onClick={() => setSel(i)}>{c}</Pill>)}</div>
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
      <Card><div style={{ fontSize: 11, fontWeight: 600, color: T.textMuted, marginBottom: 10, textTransform: 'uppercase', letterSpacing: '.06em' }}>Supply Mix (%)</div>
        <ResponsiveContainer width="100%" height={Math.max(200, barD.length * 30 + 60)}>
          <BarChart data={barD} layout="vertical" margin={{ left: 85, right: 15, top: 5, bottom: 5 }}>{GR}<XAxis type="number" {...XP} unit="%" /><YAxis type="category" dataKey="name" tick={{ fill: T.textMuted, fontSize: 10, fontFamily: "'JetBrains Mono', monospace" }} stroke={T.gridLine} width={80} /><Tooltip content={<Tip />} />
            <Bar dataKey="naive" fill={T.red} fillOpacity={.5} name="Naive" barSize={8} radius={[0,3,3,0]} /><Bar dataKey="ql" fill={T.accent} name="Q-Learned" barSize={8} radius={[0,3,3,0]} />
            <Legend wrapperStyle={{ fontSize: 10, paddingTop: 6, fontFamily: "'JetBrains Mono', monospace" }} /></BarChart></ResponsiveContainer></Card>
      <Card><div style={{ fontSize: 11, fontWeight: 600, color: T.textMuted, marginBottom: 10, textTransform: 'uppercase', letterSpacing: '.06em' }}>Q-Learned Composition</div>
        <ResponsiveContainer width="100%" height={Math.max(200, barD.length * 30 + 60)}>
          <PieChart><Pie data={pieD} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={90} innerRadius={45} paddingAngle={2} label={({ name, value }) => `${name} ${value}%`} fontSize={9}>
            {pieD.map((e, i) => <Cell key={i} fill={e.fill} />)}</Pie><Tooltip /></PieChart></ResponsiveContainer></Card>
    </div></div>);
}

function StorTab({ R, ctrs }) {
  const CN = ctrs.map(c => c.name); const [sel, setSel] = useState(0); const n = CN[sel];
  const nc = R.nv.countries[n], qc = R.ql.countries[n]; if (!nc || !qc) return null;
  const d = nc.stor.m.map((v, i) => ({ t: i + 1, naive: v, ql: qc.stor.m[i], ql95: qc.stor.h[i] }));
  return (<div>
    <div style={{ display: 'flex', gap: 4, marginBottom: 18, background: T.card, borderRadius: 8, padding: 4, border: `1px solid ${T.cardBorder}`, width: 'fit-content' }}>
      {CN.map((c, i) => <Pill key={c} active={sel === i} onClick={() => setSel(i)}>{c}</Pill>)}</div>
    <Card><ResponsiveContainer width="100%" height={300}>
      <AreaChart data={d} margin={{ left: 10, right: 10, top: 5, bottom: 5 }}>{GR}<XAxis dataKey="t" {...XP} /><YAxis {...YP} domain={[0,'auto']} /><Tooltip content={<Tip />} />
        <Area type="monotone" dataKey="ql95" stroke="none" fill={T.accent} fillOpacity={.1} name="QL P95" />
        <Area type="monotone" dataKey="ql" stroke={T.accent} fill={T.accent} fillOpacity={.15} strokeWidth={2.5} name="Q-Learned" dot={false} />
        <Line type="monotone" dataKey="naive" stroke={T.red} strokeWidth={2} strokeDasharray="6 3" dot={false} name="Naive" />
        <Legend wrapperStyle={{ fontSize: 11, paddingTop: 10, fontFamily: "'JetBrains Mono', monospace" }} /></AreaChart></ResponsiveContainer></Card>
    <div style={{ marginTop: 18, display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(200px,1fr))', gap: 14 }}>
      {CN.map((n, ci) => { const qc = R.ql.countries[n]; if (!qc) return null; const avgQ = qc.stor.m.reduce((a, b) => a + b) / qc.stor.m.length; return (
        <div key={n} style={{ background: T.card, borderRadius: 10, padding: 16, borderLeft: `3px solid ${gcc(n, ci)}`, border: `1px solid ${T.cardBorder}`, borderLeftColor: gcc(n, ci) }}>
          <div style={{ fontWeight: 700, color: gcc(n, ci), fontSize: 12, marginBottom: 8 }}>{n}</div>
          <div style={{ fontSize: 11, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 5 }}>
            <span style={{ color: T.textMuted }}>Cap</span><span style={{ fontFamily: "'JetBrains Mono', monospace", fontWeight: 600, color: T.text }}>{ctrs[ci].stor_max}</span>
            <span style={{ color: T.textMuted }}>Avg</span><span style={{ fontFamily: "'JetBrains Mono', monospace", fontWeight: 600, color: T.text }}>{ff(avgQ)}</span>
            <span style={{ color: T.textMuted }}>Util</span><span style={{ fontFamily: "'JetBrains Mono', monospace", fontWeight: 700, color: T.green }}>{ctrs[ci].stor_max ? ff(avgQ / ctrs[ci].stor_max * 100) : 0}%</span>
          </div>
        </div>);
      })}
    </div></div>);
}

function StratTab({ R, ctrs, sups }) {
  const CN = ctrs.map(c => c.name), SN = sups.map(s => s.name);
  const allocD = CN.map(n => { const af = R.ql.countries[n]?.af?.allocation || {}; return { name: n, ...Object.fromEntries(PROF_NAMES.map(k => [k, Math.round((af[k]||0)*1000)/10])) }; });
  const storD = CN.map(n => { const sf = R.ql.countries[n]?.af?.storage || {}; return { name: n, ...Object.fromEntries(STOR_ACTS.map(k => [k, Math.round((sf[k]||0)*1000)/10])) }; });
  const allocSrc = [...SN, 'spot', 'domestic'];
  const blendD = CN.map(n => { const bw = R.ql.countries[n]?.bw || {}; return { name: n, ...Object.fromEntries(allocSrc.map(s => [s, Math.round((bw[s]||0)*1000)/10])) }; });
  const h = Math.max(180, CN.length * 50 + 60);
  return (<div>
    <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 14, color: T.text }}>Allocation Profile Frequency</div>
    <Card style={{ marginBottom: 22 }}><ResponsiveContainer width="100%" height={h}>
      <BarChart data={allocD} layout="vertical" margin={{ left: 75, right: 15, top: 5, bottom: 5 }}>{GR}<XAxis type="number" {...XP} domain={[0,100]} unit="%" /><YAxis type="category" dataKey="name" tick={{ fill: T.text, fontSize: 11, fontWeight: 600 }} stroke={T.gridLine} /><Tooltip content={<Tip />} />
        {PROF_NAMES.map(k => <Bar key={k} dataKey={k} stackId="a" fill={ALLOC_C[k]} name={k} />)}<Legend wrapperStyle={{ fontSize: 9, paddingTop: 6, fontFamily: "'JetBrains Mono', monospace" }} /></BarChart></ResponsiveContainer></Card>
    <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 14, color: T.text }}>Storage Action Frequency</div>
    <Card style={{ marginBottom: 22 }}><ResponsiveContainer width="100%" height={Math.max(150, CN.length * 45 + 60)}>
      <BarChart data={storD} layout="vertical" margin={{ left: 75, right: 15, top: 5, bottom: 5 }}>{GR}<XAxis type="number" {...XP} domain={[0,100]} unit="%" /><YAxis type="category" dataKey="name" tick={{ fill: T.text, fontSize: 11, fontWeight: 600 }} stroke={T.gridLine} /><Tooltip content={<Tip />} />
        {STOR_ACTS.map(k => <Bar key={k} dataKey={k} stackId="a" fill={STOR_CC[k]} name={k} />)}<Legend wrapperStyle={{ fontSize: 9, paddingTop: 6, fontFamily: "'JetBrains Mono', monospace" }} /></BarChart></ResponsiveContainer></Card>
    <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 14, color: T.text }}>Blended Allocation Weights</div>
    <Card><ResponsiveContainer width="100%" height={h}>
      <BarChart data={blendD} layout="vertical" margin={{ left: 75, right: 15, top: 5, bottom: 5 }}>{GR}<XAxis type="number" {...XP} domain={[0,100]} unit="%" /><YAxis type="category" dataKey="name" tick={{ fill: T.text, fontSize: 11, fontWeight: 600 }} stroke={T.gridLine} /><Tooltip content={<Tip />} />
        {allocSrc.map((s, i) => <Bar key={s} dataKey={s} stackId="a" fill={MIX_C[i % MIX_C.length]} name={s} />)}<Legend wrapperStyle={{ fontSize: 8, paddingTop: 6, fontFamily: "'JetBrains Mono', monospace" }} /></BarChart></ResponsiveContainer></Card>
  </div>);
}

function TrainTab({ R, ctrs }) {
  const CN = ctrs.map(c => c.name);
  const ma = (arr, w = 6) => arr.map((_, i) => { const s = arr.slice(Math.max(0, i - w), i + 1); return s.reduce((a, b) => a + b) / s.length; });
  const spotMA = ma(R.tr.spot);
  const spotD = R.tr.spot.map((v, i) => ({ ep: i * 30, raw: v, ma: spotMA[i] }));
  const costD = (R.tr.costs[CN[0]] || []).map((_, i) => { const r = { ep: i * 30 }; CN.forEach(n => { r[n] = (R.tr.costs[n] || [])[i] || 0; }); return r; });
  return (<div>
    <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 14, color: T.text }}>Spot Price Convergence</div>
    <Card style={{ marginBottom: 22 }}><ResponsiveContainer width="100%" height={260}>
      <LineChart data={spotD} margin={{ left: 10, right: 10, top: 5, bottom: 5 }}>{GR}<XAxis dataKey="ep" {...XP} /><YAxis {...YP} /><Tooltip content={<Tip />} />
        <Line type="monotone" dataKey="raw" stroke={T.accent} strokeWidth={1} dot={false} opacity={.25} name="Raw" />
        <Line type="monotone" dataKey="ma" stroke={T.accent} strokeWidth={2.5} dot={false} name="MA" /></LineChart></ResponsiveContainer></Card>
    <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 14, color: T.text }}>Country Cost Convergence</div>
    <Card><ResponsiveContainer width="100%" height={280}>
      <LineChart data={costD} margin={{ left: 10, right: 10, top: 5, bottom: 5 }}>{GR}<XAxis dataKey="ep" {...XP} /><YAxis {...YP} /><Tooltip content={<Tip />} />
        {CN.map((n, i) => <Line key={n} type="monotone" dataKey={n} stroke={gcc(n, i)} strokeWidth={1.5} dot={false} name={n} />)}
        <Legend wrapperStyle={{ fontSize: 11, paddingTop: 10, fontFamily: "'JetBrains Mono', monospace" }} /></LineChart></ResponsiveContainer></Card>
    {R.meta && (
      <div style={{ marginTop: 18, padding: 16, background: T.accentSoft, borderRadius: 10, fontSize: 11, color: T.textMuted, border: `1px solid ${T.accent}22`, fontFamily: "'JetBrains Mono', monospace" }}>
        Completed in <b style={{ color: T.accent }}>{R.meta.elapsed_s}s</b> — {R.meta.train_episodes} training episodes, {R.meta.eval_sims} evaluation sims, entirely in-browser.
      </div>
    )}
  </div>);
}

/* ═══════════════════════════════════════════════════════════════════
   MAIN APP
   ═══════════════════════════════════════════════════════════════════ */
export default function App() {
  const [tab, setTab] = useState(0);
  const [sups, setSups] = useState(() => D_SUP.map(s => ({ ...s })));
  const [ctrs, setCtrs] = useState(() => D_CTR.map(c => ({ ...c })));
  const [bSpot, setBSpot] = useState(D_SPOT);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState('');
  const [R, setR] = useState(null);

  const handleRun = useCallback(async () => {
    setRunning(true); setProgress('Training agents...');
    await new Promise(r => setTimeout(r, 50));
    try {
      const t0 = performance.now();
      const res = runModel(sups, ctrs, bSpot, 3000, 400, 36);
      res.meta = { elapsed_s: Math.round((performance.now() - t0) / 10) / 100, train_episodes: 3000, eval_sims: 400 };
      setR(res); setTab(1);
    } catch (e) { alert('Error: ' + e.message); }
    finally { setRunning(false); setProgress(''); }
  }, [sups, ctrs, bSpot]);

  const has = R !== null;
  const renderTab = () => {
    if (tab === 0) return <ParamTab sups={sups} setSups={setSups} ctrs={ctrs} setCtrs={setCtrs} bSpot={bSpot} setBSpot={setBSpot} onRun={handleRun} running={running} progress={progress} />;
    if (!has) return <div style={{ textAlign: 'center', padding: 60, color: T.textMuted, fontFamily: "'JetBrains Mono', monospace" }}>No results yet — configure parameters and run.</div>;
    switch (tab) {
      case 1: return <Overview R={R} ctrs={ctrs} />;
      case 2: return <SpotTab R={R} />;
      case 3: return <CostTab R={R} ctrs={ctrs} />;
      case 4: return <MixTab R={R} ctrs={ctrs} sups={sups} />;
      case 5: return <StorTab R={R} ctrs={ctrs} />;
      case 6: return <StratTab R={R} ctrs={ctrs} sups={sups} />;
      case 7: return <TrainTab R={R} ctrs={ctrs} />;
      default: return null;
    }
  };

  return (
    <div style={{ background: T.bg, color: T.text, minHeight: '100vh', fontFamily: "'DM Sans', -apple-system, sans-serif", padding: 28 }}>
      <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet" />
      <div style={{ maxWidth: 1080, margin: '0 auto' }}>
        <div style={{ marginBottom: 28 }}>
          <div style={{ fontSize: 10, fontWeight: 700, color: T.accent, textTransform: 'uppercase', letterSpacing: '.14em', marginBottom: 6, fontFamily: "'JetBrains Mono', monospace" }}>Multi-Agent Reinforcement Learning</div>
          <h1 style={{ fontSize: 26, fontWeight: 700, margin: 0, letterSpacing: '-.03em', color: T.text }}>Commodity Offtake Portfolio Model</h1>
          <p style={{ fontSize: 11, color: T.textDim, margin: '8px 0 0', fontFamily: "'JetBrains Mono', monospace" }}>Q-learning agents · in-browser computation · zero server dependencies</p>
        </div>
        <div style={{ display: 'flex', gap: 3, marginBottom: 22, background: T.card, borderRadius: 10, padding: 5, border: `1px solid ${T.cardBorder}`, flexWrap: 'wrap', width: 'fit-content' }}>
          {TABS.map((t, i) => <Pill key={t} active={tab === i} onClick={() => setTab(i)} disabled={i > 0 && !has}>{t}</Pill>)}
        </div>
        {renderTab()}
      </div>
    </div>
  );
}
