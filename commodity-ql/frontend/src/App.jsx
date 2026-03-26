import React, { useState, useCallback } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Area, AreaChart, BarChart, Bar, Cell, Legend, CartesianGrid, PieChart, Pie } from "recharts";

/* ═══════════════════════════════════════════════════════════════════
   DEFAULTS — V2: separate fixed/variable costs, init/max capacity
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
   CONSTANTS (for strategy tab labels)
   ═══════════════════════════════════════════════════════════════════ */
const PROF_NAMES = ['diversified','reliable','heavy-spot','heavy-domestic','balanced','min-cost'];
const STOR_ACTS = ['store','hold','draw'];

/* ═══════════════════════════════════════════════════════════════════
   API CLIENT
   ═══════════════════════════════════════════════════════════════════ */
const API_BASE = (typeof import.meta !== 'undefined' && import.meta.env?.VITE_API_URL) || '';

async function runViaAPI(sups, ctrs, bSpot) {
  const resp = await fetch(`${API_BASE}/api/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      suppliers: sups.map(s => ({
        name: s.name,
        max_capacity: s.max_capacity,
        init_capacity: s.init_capacity,
        fixed_cost: s.fixed_cost,
        variable_cost: s.variable_cost,
        p_d: s.p_d,
        p_r: s.p_r,
      })),
      countries: ctrs.map(c => ({
        name: c.name,
        demand: c.demand,
        elasticity: c.elasticity,
        dom_variable_cost: c.dom_variable_cost,
        dom_fixed_cost: c.dom_fixed_cost,
        dom_max_capacity: c.dom_max_capacity,
        dom_init_capacity: c.dom_init_capacity,
        max_dom: c.max_dom,
        stor_max: c.stor_max,
        stor_cost: c.stor_cost,
        stor_init: c.stor_init,
      })),
      base_spot: bSpot,
      train_episodes: 3000,
      eval_sims: 400,
      ep_length: 36,
    }),
  });
  if (!resp.ok) {
    const err = await resp.text();
    throw new Error(`Server error ${resp.status}: ${err}`);
  }
  return await resp.json();
}

/* ═══════════════════════════════════════════════════════════════════
   UI PRIMITIVES
   ═══════════════════════════════════════════════════════════════════ */
const XCC = ["#c0392b","#2471a3","#d4890a","#1e8449","#8e44ad","#d35400","#16a085","#2c3e50"];
const ALLOC_C = {"diversified":"#7f8c8d","reliable":"#2471a3","heavy-spot":"#8e44ad","heavy-domestic":"#d4890a","balanced":"#1e8449","min-cost":"#c0392b"};
const STOR_CC = {"store":"#2471a3","hold":"#95a5a6","draw":"#c0392b"};
const MIX_C = ["#c0392b","#2471a3","#d4890a","#1e8449","#8e44ad","#d35400","#16a085","#2c3e50","#e74c3c","#3498db","#95a5a6","#34495e","#e67e22"];
const TABS = ["Parameters","Overview","Spot Market","Costs","Supply Mix","Storage","Strategy","Training"];
const f = (v, d=1) => typeof v === 'number' ? v.toFixed(d) : v;
const gcc = (n, i) => ({"Industria":"#c0392b","Pacifica":"#2471a3","Europa":"#d4890a","Emergent":"#1e8449"}[n] || XCC[i % XCC.length]);
const GS = "#e8eaef";

function Tip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background: '#fff', border: '1px solid #ddd', borderRadius: 6, padding: '8px 14px', fontSize: 12, boxShadow: '0 2px 10px #0001' }}>
      <div style={{ color: '#888', fontWeight: 600, marginBottom: 4 }}>{label}</div>
      {payload.filter(p => p.name && p.value !== undefined).map((p, i) => (
        <div key={i} style={{ color: p.color || '#333' }}>{p.name}: <b>{f(p.value, 2)}</b></div>
      ))}
    </div>
  );
}

function Stat({ label, nv, ql, unit = '', lower = true }) {
  const d = (ql - nv) / Math.abs(nv || 1) * 100;
  const good = lower ? d < 0 : d > 0;
  return (
    <div style={{ background: '#fff', borderRadius: 8, padding: '14px 18px', flex: 1, minWidth: 130, border: '1px solid #e4e6eb' }}>
      <div style={{ fontSize: 10, color: '#999', textTransform: 'uppercase', letterSpacing: '.08em', marginBottom: 8 }}>{label}</div>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, flexWrap: 'wrap' }}>
        <span style={{ fontSize: 22, fontWeight: 700, color: '#1a1f2e', fontFamily: 'monospace' }}>{f(ql)}{unit}</span>
        <span style={{ fontSize: 11, fontWeight: 700, color: good ? '#1e8449' : '#c0392b', background: good ? '#e8f5e9' : '#fbe9e7', padding: '2px 8px', borderRadius: 10 }}>{d > 0 ? '+' : ''}{f(d)}%</span>
      </div>
      <div style={{ fontSize: 11, color: '#bbb', marginTop: 5 }}>Naive: {f(nv)}{unit}</div>
    </div>
  );
}

function Pill({ active, onClick, children, disabled }) {
  return (
    <button onClick={onClick} disabled={disabled} style={{ padding: '7px 16px', border: 'none', borderRadius: 6, cursor: disabled ? 'not-allowed' : 'pointer', fontSize: 12.5, fontWeight: active ? 600 : 400, background: active ? '#1a1f2e' : 'transparent', color: active ? '#fff' : disabled ? '#ccc' : '#667', transition: 'all .15s', opacity: disabled ? .4 : 1 }}>
      {children}
    </button>
  );
}

function Card({ children, style }) {
  return <div style={{ background: '#fff', borderRadius: 10, border: '1px solid #e4e6eb', padding: '18px 14px', ...style }}>{children}</div>;
}

const IS = { background: '#f8f9fa', border: '1px solid #dde', borderRadius: 5, color: '#1a1f2e', padding: '6px 8px', fontSize: 13, width: '100%', fontFamily: 'monospace', boxSizing: 'border-box', outline: 'none' };

function NI({ value, onChange, step, min, max, style: es }) {
  return (
    <input type="number" value={value} onChange={e => onChange(parseFloat(e.target.value) || 0)}
      step={step} min={min} max={max} style={{ ...IS, ...es }}
      onFocus={e => { e.target.style.borderColor = '#2471a3'; }}
      onBlur={e => { e.target.style.borderColor = '#dde'; }} />
  );
}

const GR = <CartesianGrid strokeDasharray="3 3" stroke={GS} />;
const XP = { tick: { fill: '#999', fontSize: 11 }, stroke: GS };
const YP = { tick: { fill: '#999', fontSize: 11 }, stroke: GS };

/* ═══════════════════════════════════════════════════════════════════
   PARAMETER TAB — V2 fields
   ═══════════════════════════════════════════════════════════════════ */
function ParamTab({ sups, setSups, ctrs, setCtrs, bSpot, setBSpot, onRun, running, progress }) {
  const uS = (i, k, v) => setSups(sups.map((s, j) => j === i ? { ...s, [k]: v } : s));
  const uC = (i, k, v) => setCtrs(ctrs.map((c, j) => j === i ? { ...c, [k]: v } : c));
  const th = { padding: '8px 5px', textAlign: 'left', color: '#999', fontWeight: 600, fontSize: 10, textTransform: 'uppercase', letterSpacing: '.06em', whiteSpace: 'nowrap', borderBottom: '2px solid #e4e6eb' };
  const td = { padding: '4px 3px' };

  return (
    <div>
      <div style={{ display: 'flex', gap: 16, alignItems: 'center', marginBottom: 22, flexWrap: 'wrap' }}>
        <div>
          <label style={{ fontSize: 10, color: '#999', textTransform: 'uppercase', letterSpacing: '.06em', display: 'block', marginBottom: 4 }}>Base Spot Price</label>
          <NI value={bSpot} onChange={setBSpot} step={1} min={1} style={{ width: 90 }} />
        </div>
        <div style={{ marginLeft: 'auto', display: 'flex', gap: 10 }}>
          <button onClick={() => { setSups(D_SUP.map(s => ({ ...s }))); setCtrs(D_CTR.map(c => ({ ...c }))); setBSpot(D_SPOT); }}
            style={{ background: '#fff', border: '1px solid #dde', borderRadius: 6, color: '#667', padding: '9px 18px', fontSize: 12, cursor: 'pointer' }}>Reset</button>
          <button onClick={onRun} disabled={running} style={{ background: running ? '#e4e6eb' : '#2471a3', border: 'none', borderRadius: 6, color: running ? '#999' : '#fff', padding: '9px 24px', fontSize: 13, fontWeight: 700, cursor: running ? 'wait' : 'pointer', boxShadow: running ? 'none' : '0 2px 8px #2471a333' }}>
            {running ? progress : 'Train & Evaluate'}
          </button>
        </div>
      </div>

      <Card style={{ marginBottom: 20 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
          <div style={{ fontSize: 15, fontWeight: 700 }}>Suppliers <span style={{ color: '#999', fontWeight: 400, fontSize: 13 }}>({sups.length})</span></div>
          <button onClick={() => sups.length < 15 && setSups([...sups, { name: `Sup-${sups.length + 1}`, max_capacity: 50, init_capacity: 45, fixed_cost: 10, variable_cost: 55, p_d: 0.05, p_r: 0.5 }])}
            style={{ background: '#e8f5e9', border: 'none', borderRadius: 5, color: '#1e8449', padding: '5px 14px', fontSize: 12, cursor: 'pointer', fontWeight: 600 }}>+ Add</button>
        </div>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead><tr>{["Name", "Max Cap", "Init Cap", "Fixed $/u", "Var $/u", "P(Disrupt)", "P(Recover)", "SS Off%", ""].map(h => <th key={h} style={th}>{h}</th>)}</tr></thead>
            <tbody>{sups.map((s, i) => {
              const ss = s.p_d / (s.p_d + s.p_r);
              return (
                <tr key={i} style={{ borderBottom: '1px solid #f5f5f5' }}>
                  <td style={td}><input type="text" value={s.name} onChange={e => uS(i, 'name', e.target.value)} style={{ ...IS, width: 100, fontFamily: 'inherit' }} /></td>
                  <td style={td}><NI value={s.max_capacity} onChange={v => uS(i, 'max_capacity', v)} step={5} min={1} style={{ width: 62 }} /></td>
                  <td style={td}><NI value={s.init_capacity} onChange={v => uS(i, 'init_capacity', v)} step={5} min={1} style={{ width: 62 }} /></td>
                  <td style={td}><NI value={s.fixed_cost} onChange={v => uS(i, 'fixed_cost', v)} step={1} min={0} style={{ width: 62 }} /></td>
                  <td style={td}><NI value={s.variable_cost} onChange={v => uS(i, 'variable_cost', v)} step={1} min={1} style={{ width: 62 }} /></td>
                  <td style={td}><NI value={s.p_d} onChange={v => uS(i, 'p_d', v)} step={0.01} min={0} max={1} style={{ width: 62 }} /></td>
                  <td style={td}><NI value={s.p_r} onChange={v => uS(i, 'p_r', v)} step={0.05} min={0.01} max={1} style={{ width: 62 }} /></td>
                  <td style={{ ...td, fontFamily: 'monospace', fontWeight: 600, color: ss > .15 ? '#c0392b' : ss > .08 ? '#d4890a' : '#1e8449', textAlign: 'center' }}>{(ss * 100).toFixed(1)}%</td>
                  <td style={td}><button onClick={() => sups.length > 2 && setSups(sups.filter((_, j) => j !== i))} style={{ background: 'none', border: 'none', color: '#c0392b', cursor: 'pointer', fontSize: 14, opacity: sups.length <= 2 ? .3 : 1, fontWeight: 700 }}>✕</button></td>
                </tr>
              );
            })}</tbody>
          </table>
        </div>
      </Card>

      <Card>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
          <div style={{ fontSize: 15, fontWeight: 700 }}>Countries <span style={{ color: '#999', fontWeight: 400, fontSize: 13 }}>({ctrs.length})</span></div>
          <button onClick={() => ctrs.length < 8 && setCtrs([...ctrs, { name: `Country-${ctrs.length + 1}`, demand: 100, elasticity: -.3, dom_variable_cost: 70, dom_fixed_cost: 10, dom_max_capacity: 40, dom_init_capacity: 30, max_dom: .2, stor_max: 40, stor_cost: 2, stor_init: 10 }])}
            style={{ background: '#e8f5e9', border: 'none', borderRadius: 5, color: '#1e8449', padding: '5px 14px', fontSize: 12, cursor: 'pointer', fontWeight: 600 }}>+ Add</button>
        </div>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead><tr>{["Name", "Demand", "Elast.", "Dom Var$", "Dom Fix$", "Dom Max", "Dom Init", "Max Dom%", "Stor Cap", "$/per", "Init", ""].map(h => <th key={h} style={th}>{h}</th>)}</tr></thead>
            <tbody>{ctrs.map((c, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f5f5f5' }}>
                <td style={td}><input type="text" value={c.name} onChange={e => uC(i, 'name', e.target.value)} style={{ ...IS, width: 100, fontFamily: 'inherit' }} /></td>
                <td style={td}><NI value={c.demand} onChange={v => uC(i, 'demand', v)} step={10} min={1} style={{ width: 62 }} /></td>
                <td style={td}><NI value={c.elasticity} onChange={v => uC(i, 'elasticity', v)} step={.05} min={-2} max={0} style={{ width: 62 }} /></td>
                <td style={td}><NI value={c.dom_variable_cost} onChange={v => uC(i, 'dom_variable_cost', v)} step={1} min={1} style={{ width: 62 }} /></td>
                <td style={td}><NI value={c.dom_fixed_cost} onChange={v => uC(i, 'dom_fixed_cost', v)} step={1} min={0} style={{ width: 62 }} /></td>
                <td style={td}><NI value={c.dom_max_capacity} onChange={v => uC(i, 'dom_max_capacity', v)} step={5} min={0} style={{ width: 62 }} /></td>
                <td style={td}><NI value={c.dom_init_capacity} onChange={v => uC(i, 'dom_init_capacity', v)} step={5} min={0} style={{ width: 62 }} /></td>
                <td style={td}><NI value={c.max_dom} onChange={v => uC(i, 'max_dom', v)} step={.05} min={0} max={1} style={{ width: 62 }} /></td>
                <td style={td}><NI value={c.stor_max} onChange={v => uC(i, 'stor_max', v)} step={5} min={0} style={{ width: 58 }} /></td>
                <td style={td}><NI value={c.stor_cost} onChange={v => uC(i, 'stor_cost', v)} step={.5} min={0} style={{ width: 58 }} /></td>
                <td style={td}><NI value={c.stor_init} onChange={v => uC(i, 'stor_init', v)} step={5} min={0} style={{ width: 58 }} /></td>
                <td style={td}><button onClick={() => ctrs.length > 2 && setCtrs(ctrs.filter((_, j) => j !== i))} style={{ background: 'none', border: 'none', color: '#c0392b', cursor: 'pointer', fontSize: 14, opacity: ctrs.length <= 2 ? .3 : 1, fontWeight: 700 }}>✕</button></td>
              </tr>
            ))}</tbody>
          </table>
        </div>
      </Card>

      <div style={{ marginTop: 18, padding: 14, background: '#eef6fb', borderRadius: 8, fontSize: 12, color: '#556', lineHeight: 1.7, border: '1px solid #d0e4f0' }}>
        <b style={{ color: '#2471a3' }}>Server-side training:</b> Click "Train & Evaluate" to send parameters to the Python backend. The server runs 3,000 Q-learning episodes and 400 Monte Carlo evaluation simulations with numpy — much faster and more accurate than in-browser computation. Results populate all tabs once complete.
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════
   RESULTS TABS
   ═══════════════════════════════════════════════════════════════════ */
function Overview({ R, ctrs }) {
  const CN = ctrs.map(c => c.name);
  return (
    <div>
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 24 }}>
        <Stat label="Avg Spot" nv={R.nv.spotS.mean} ql={R.ql.spotS.mean} />
        <Stat label="Spot Vol" nv={R.nv.spotS.std} ql={R.ql.spotS.std} />
        <Stat label="Spot P95" nv={R.nv.spotS.p95} ql={R.ql.spotS.p95} />
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(225px,1fr))', gap: 12 }}>
        {CN.map((n, ci) => {
          const ns = R.nv.countries[n]?.s, qs = R.ql.countries[n]?.s;
          if (!ns || !qs) return null;
          const col = gcc(n, ci);
          return (
            <div key={n} style={{ background: '#fff', borderRadius: 8, padding: 16, borderLeft: `4px solid ${col}`, border: '1px solid #e4e6eb', borderLeftColor: col }}>
              <div style={{ fontSize: 15, fontWeight: 700, color: col, marginBottom: 8 }}>{n}</div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 5, fontSize: 12 }}>
                <span style={{ color: '#999' }}>Avg Cost</span><span style={{ fontFamily: 'monospace', fontWeight: 600 }}>{f(qs.avg_mean)} <span style={{ color: '#ccc' }}>({f(ns.avg_mean)})</span></span>
                <span style={{ color: '#999' }}>Saving</span><span style={{ fontFamily: 'monospace', fontWeight: 700, color: '#1e8449' }}>{f((1 - qs.avg_mean / ns.avg_mean) * 100)}%</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function SpotTab({ R }) {
  const d = R.nv.spot.m.map((v, i) => ({ t: i + 1, naive: v, nv95: R.nv.spot.h[i], ql: R.ql.spot.m[i], ql95: R.ql.spot.h[i] }));
  return (
    <Card>
      <ResponsiveContainer width="100%" height={370}>
        <AreaChart data={d} margin={{ left: 10, right: 10, top: 5, bottom: 5 }}>
          {GR}<XAxis dataKey="t" {...XP} /><YAxis {...YP} /><Tooltip content={<Tip />} />
          <Area dataKey="nv95" stroke="none" fill="#c0392b" fillOpacity={.07} name="Naive P95" />
          <Area dataKey="ql95" stroke="none" fill="#2471a3" fillOpacity={.07} name="QL P95" />
          <Line type="monotone" dataKey="naive" stroke="#c0392b" strokeWidth={2.5} dot={false} name="Naive" />
          <Line type="monotone" dataKey="ql" stroke="#2471a3" strokeWidth={2.5} dot={false} name="Q-Learned" />
          <Legend wrapperStyle={{ fontSize: 12, paddingTop: 8 }} />
        </AreaChart>
      </ResponsiveContainer>
    </Card>
  );
}

function CostTab({ R, ctrs }) {
  const CN = ctrs.map(c => c.name);
  const [sel, setSel] = useState(0);
  const n = CN[sel];
  const nc = R.nv.countries[n], qc = R.ql.countries[n];
  if (!nc || !qc) return null;
  const d = nc.cost.m.map((v, i) => ({ t: i + 1, naive: v, nv95: nc.cost.h[i], ql: qc.cost.m[i], ql95: qc.cost.h[i] }));
  return (
    <div>
      <div style={{ display: 'flex', gap: 4, marginBottom: 16, background: '#fff', borderRadius: 8, padding: 4, border: '1px solid #e4e6eb', width: 'fit-content' }}>
        {CN.map((c, i) => <Pill key={c} active={sel === i} onClick={() => setSel(i)}>{c}</Pill>)}
      </div>
      <Card>
        <ResponsiveContainer width="100%" height={320}>
          <AreaChart data={d} margin={{ left: 10, right: 10, top: 5, bottom: 5 }}>
            {GR}<XAxis dataKey="t" {...XP} /><YAxis {...YP} domain={['auto', 'auto']} /><Tooltip content={<Tip />} />
            <Area dataKey="nv95" stroke="none" fill="#c0392b" fillOpacity={.06} name="Naive P95" />
            <Area dataKey="ql95" stroke="none" fill="#2471a3" fillOpacity={.06} name="QL P95" />
            <Line type="monotone" dataKey="naive" stroke="#c0392b" strokeWidth={2.5} dot={false} name="Naive" />
            <Line type="monotone" dataKey="ql" stroke="#2471a3" strokeWidth={2.5} dot={false} name="Q-Learned" />
            <Legend wrapperStyle={{ fontSize: 12, paddingTop: 8 }} />
          </AreaChart>
        </ResponsiveContainer>
      </Card>
      <div style={{ display: 'flex', gap: 12, marginTop: 16, flexWrap: 'wrap' }}>
        <Stat label="Mean Cost" nv={nc.s.avg_mean} ql={qc.s.avg_mean} />
        <Stat label="Cost Vol" nv={nc.s.avg_std} ql={qc.s.avg_std} />
      </div>
    </div>
  );
}

function MixTab({ R, ctrs, sups }) {
  const CN = ctrs.map(c => c.name), SN = sups.map(s => s.name);
  const [sel, setSel] = useState(0);
  const n = CN[sel];
  const sources = [...SN, 'spot', 'domestic', 'storage_draw'];
  const nMx = R.nv.countries[n]?.mx || {}, qMx = R.ql.countries[n]?.mx || {};
  const barD = sources.filter(s => ((nMx[s] || 0) + (qMx[s] || 0)) > .005).map(s => ({ name: s, naive: Math.round((nMx[s] || 0) * 1000) / 10, ql: Math.round((qMx[s] || 0) * 1000) / 10 }));
  const pieD = sources.filter(s => (qMx[s] || 0) > .01).map((s, i) => ({ name: s, value: Math.round((qMx[s] || 0) * 1000) / 10, fill: MIX_C[i % MIX_C.length] }));
  return (
    <div>
      <div style={{ display: 'flex', gap: 4, marginBottom: 16, background: '#fff', borderRadius: 8, padding: 4, border: '1px solid #e4e6eb', width: 'fit-content' }}>
        {CN.map((c, i) => <Pill key={c} active={sel === i} onClick={() => setSel(i)}>{c}</Pill>)}
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        <Card>
          <div style={{ fontSize: 12, fontWeight: 600, color: '#666', marginBottom: 8 }}>Supply Mix (%)</div>
          <ResponsiveContainer width="100%" height={Math.max(200, barD.length * 30 + 60)}>
            <BarChart data={barD} layout="vertical" margin={{ left: 85, right: 15, top: 5, bottom: 5 }}>
              {GR}<XAxis type="number" {...XP} unit="%" /><YAxis type="category" dataKey="name" tick={{ fill: '#555', fontSize: 10 }} stroke={GS} width={80} /><Tooltip content={<Tip />} />
              <Bar dataKey="naive" fill="#c0392b" fillOpacity={.5} name="Naive" barSize={8} radius={[0, 3, 3, 0]} />
              <Bar dataKey="ql" fill="#2471a3" name="Q-Learned" barSize={8} radius={[0, 3, 3, 0]} />
              <Legend wrapperStyle={{ fontSize: 11, paddingTop: 4 }} />
            </BarChart>
          </ResponsiveContainer>
        </Card>
        <Card>
          <div style={{ fontSize: 12, fontWeight: 600, color: '#666', marginBottom: 8 }}>Q-Learned Composition</div>
          <ResponsiveContainer width="100%" height={Math.max(200, barD.length * 30 + 60)}>
            <PieChart>
              <Pie data={pieD} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={90} innerRadius={45} paddingAngle={2} label={({ name, value }) => `${name} ${value}%`} fontSize={9}>
                {pieD.map((e, i) => <Cell key={i} fill={e.fill} />)}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </Card>
      </div>
    </div>
  );
}

function StorTab({ R, ctrs }) {
  const CN = ctrs.map(c => c.name);
  const [sel, setSel] = useState(0);
  const n = CN[sel], nc = R.nv.countries[n], qc = R.ql.countries[n];
  if (!nc || !qc) return null;
  const d = nc.stor.m.map((v, i) => ({ t: i + 1, naive: v, ql: qc.stor.m[i], ql95: qc.stor.h[i] }));
  return (
    <div>
      <div style={{ display: 'flex', gap: 4, marginBottom: 16, background: '#fff', borderRadius: 8, padding: 4, border: '1px solid #e4e6eb', width: 'fit-content' }}>
        {CN.map((c, i) => <Pill key={c} active={sel === i} onClick={() => setSel(i)}>{c}</Pill>)}
      </div>
      <Card>
        <ResponsiveContainer width="100%" height={300}>
          <AreaChart data={d} margin={{ left: 10, right: 10, top: 5, bottom: 5 }}>
            {GR}<XAxis dataKey="t" {...XP} /><YAxis {...YP} domain={[0, 'auto']} /><Tooltip content={<Tip />} />
            <Area type="monotone" dataKey="ql95" stroke="none" fill="#2471a3" fillOpacity={.08} name="QL P95" />
            <Area type="monotone" dataKey="ql" stroke="#2471a3" fill="#2471a3" fillOpacity={.15} strokeWidth={2.5} name="Q-Learned" dot={false} />
            <Line type="monotone" dataKey="naive" stroke="#c0392b" strokeWidth={2} strokeDasharray="6 3" dot={false} name="Naive" />
            <Legend wrapperStyle={{ fontSize: 12, paddingTop: 8 }} />
          </AreaChart>
        </ResponsiveContainer>
      </Card>
      <div style={{ marginTop: 16, display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(190px,1fr))', gap: 12 }}>
        {CN.map((n, ci) => {
          const qc = R.ql.countries[n]; if (!qc) return null;
          const avgQ = qc.stor.m.reduce((a, b) => a + b) / qc.stor.m.length;
          return (
            <div key={n} style={{ background: '#fff', borderRadius: 8, padding: 14, borderLeft: `4px solid ${gcc(n, ci)}`, border: '1px solid #e4e6eb', borderLeftColor: gcc(n, ci) }}>
              <div style={{ fontWeight: 700, color: gcc(n, ci), fontSize: 13, marginBottom: 6 }}>{n}</div>
              <div style={{ fontSize: 12, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4 }}>
                <span style={{ color: '#999' }}>Cap</span><span style={{ fontFamily: 'monospace', fontWeight: 600 }}>{ctrs[ci].stor_max}</span>
                <span style={{ color: '#999' }}>Avg</span><span style={{ fontFamily: 'monospace', fontWeight: 600 }}>{f(avgQ)}</span>
                <span style={{ color: '#999' }}>Util</span><span style={{ fontFamily: 'monospace', fontWeight: 700, color: '#1e8449' }}>{ctrs[ci].stor_max ? f(avgQ / ctrs[ci].stor_max * 100) : 0}%</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function StratTab({ R, ctrs, sups }) {
  const CN = ctrs.map(c => c.name), SN = sups.map(s => s.name);
  const allocD = CN.map(n => { const af = R.ql.countries[n]?.af?.allocation || {}; return { name: n, ...Object.fromEntries(PROF_NAMES.map(k => [k, Math.round((af[k] || 0) * 1000) / 10])) }; });
  const storD = CN.map(n => { const sf = R.ql.countries[n]?.af?.storage || {}; return { name: n, ...Object.fromEntries(STOR_ACTS.map(k => [k, Math.round((sf[k] || 0) * 1000) / 10])) }; });
  const allocSrc = [...SN, 'spot', 'domestic'];
  const blendD = CN.map(n => { const bw = R.ql.countries[n]?.bw || {}; return { name: n, ...Object.fromEntries(allocSrc.map(s => [s, Math.round((bw[s] || 0) * 1000) / 10])) }; });
  const h = Math.max(180, CN.length * 50 + 60);
  return (
    <div>
      <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 12 }}>Allocation Profile Frequency</div>
      <Card style={{ marginBottom: 20 }}>
        <ResponsiveContainer width="100%" height={h}>
          <BarChart data={allocD} layout="vertical" margin={{ left: 75, right: 15, top: 5, bottom: 5 }}>
            {GR}<XAxis type="number" {...XP} domain={[0, 100]} unit="%" /><YAxis type="category" dataKey="name" tick={{ fill: '#333', fontSize: 12, fontWeight: 600 }} stroke={GS} /><Tooltip content={<Tip />} />
            {PROF_NAMES.map(k => <Bar key={k} dataKey={k} stackId="a" fill={ALLOC_C[k]} name={k} />)}
            <Legend wrapperStyle={{ fontSize: 10, paddingTop: 6 }} />
          </BarChart>
        </ResponsiveContainer>
      </Card>
      <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 12 }}>Storage Action Frequency</div>
      <Card style={{ marginBottom: 20 }}>
        <ResponsiveContainer width="100%" height={Math.max(150, CN.length * 45 + 60)}>
          <BarChart data={storD} layout="vertical" margin={{ left: 75, right: 15, top: 5, bottom: 5 }}>
            {GR}<XAxis type="number" {...XP} domain={[0, 100]} unit="%" /><YAxis type="category" dataKey="name" tick={{ fill: '#333', fontSize: 12, fontWeight: 600 }} stroke={GS} /><Tooltip content={<Tip />} />
            {STOR_ACTS.map(k => <Bar key={k} dataKey={k} stackId="a" fill={STOR_CC[k]} name={k} />)}
            <Legend wrapperStyle={{ fontSize: 10, paddingTop: 6 }} />
          </BarChart>
        </ResponsiveContainer>
      </Card>
      <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 12 }}>Blended Allocation Weights</div>
      <Card>
        <ResponsiveContainer width="100%" height={h}>
          <BarChart data={blendD} layout="vertical" margin={{ left: 75, right: 15, top: 5, bottom: 5 }}>
            {GR}<XAxis type="number" {...XP} domain={[0, 100]} unit="%" /><YAxis type="category" dataKey="name" tick={{ fill: '#333', fontSize: 12, fontWeight: 600 }} stroke={GS} /><Tooltip content={<Tip />} />
            {allocSrc.map((s, i) => <Bar key={s} dataKey={s} stackId="a" fill={MIX_C[i % MIX_C.length]} name={s} />)}
            <Legend wrapperStyle={{ fontSize: 9, paddingTop: 6 }} />
          </BarChart>
        </ResponsiveContainer>
      </Card>
    </div>
  );
}

function TrainTab({ R, ctrs }) {
  const CN = ctrs.map(c => c.name);
  const ma = (arr, w = 6) => arr.map((_, i) => { const s = arr.slice(Math.max(0, i - w), i + 1); return s.reduce((a, b) => a + b) / s.length; });
  const spotMA = ma(R.tr.spot);
  const spotD = R.tr.spot.map((v, i) => ({ ep: i * 30, raw: v, ma: spotMA[i] }));
  const costD = (R.tr.costs[CN[0]] || []).map((_, i) => { const r = { ep: i * 30 }; CN.forEach(n => { r[n] = (R.tr.costs[n] || [])[i] || 0; }); return r; });
  return (
    <div>
      <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 12 }}>Spot Price Convergence</div>
      <Card style={{ marginBottom: 20 }}>
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={spotD} margin={{ left: 10, right: 10, top: 5, bottom: 5 }}>
            {GR}<XAxis dataKey="ep" {...XP} /><YAxis {...YP} /><Tooltip content={<Tip />} />
            <Line type="monotone" dataKey="raw" stroke="#2471a3" strokeWidth={1} dot={false} opacity={.25} name="Raw" />
            <Line type="monotone" dataKey="ma" stroke="#2471a3" strokeWidth={2.5} dot={false} name="MA" />
          </LineChart>
        </ResponsiveContainer>
      </Card>
      <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 12 }}>Country Cost Convergence</div>
      <Card>
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={costD} margin={{ left: 10, right: 10, top: 5, bottom: 5 }}>
            {GR}<XAxis dataKey="ep" {...XP} /><YAxis {...YP} /><Tooltip content={<Tip />} />
            {CN.map((n, i) => <Line key={n} type="monotone" dataKey={n} stroke={gcc(n, i)} strokeWidth={1.5} dot={false} name={n} />)}
            <Legend wrapperStyle={{ fontSize: 12, paddingTop: 8 }} />
          </LineChart>
        </ResponsiveContainer>
      </Card>
      {R.meta && (
        <div style={{ marginTop: 16, padding: 14, background: '#faf5ee', borderRadius: 8, fontSize: 12, color: '#665', border: '1px solid #e8dcc0' }}>
          Server completed in <b>{R.meta.elapsed_s}s</b> — {R.meta.train_episodes} training episodes, {R.meta.eval_sims} evaluation simulations.
        </div>
      )}
    </div>
  );
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
    setRunning(true);
    setProgress('Sending to server...');
    try {
      const res = await runViaAPI(sups, ctrs, bSpot);
      setR(res);
      setTab(1);
    } catch (e) {
      alert('Error: ' + e.message);
    } finally {
      setRunning(false);
      setProgress('');
    }
  }, [sups, ctrs, bSpot]);

  const has = R !== null;

  const renderTab = () => {
    if (tab === 0) return <ParamTab sups={sups} setSups={setSups} ctrs={ctrs} setCtrs={setCtrs} bSpot={bSpot} setBSpot={setBSpot} onRun={handleRun} running={running} progress={progress} />;
    if (!has) return <div style={{ textAlign: 'center', padding: 60, color: '#999' }}>No results yet — configure and run on the Parameters tab.</div>;
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
    <div style={{ background: '#f4f5f7', color: '#1a1f2e', minHeight: '100vh', fontFamily: "'Libre Franklin',-apple-system,sans-serif", padding: 24 }}>
      <link href="https://fonts.googleapis.com/css2?family=Libre+Franklin:wght@300;400;500;600;700&display=swap" rel="stylesheet" />
      <div style={{ maxWidth: 1060, margin: '0 auto' }}>
        <div style={{ marginBottom: 24 }}>
          <div style={{ fontSize: 11, fontWeight: 600, color: '#2471a3', textTransform: 'uppercase', letterSpacing: '.1em', marginBottom: 5 }}>Multi-Agent Reinforcement Learning</div>
          <h1 style={{ fontSize: 24, fontWeight: 700, margin: 0, letterSpacing: '-.02em' }}>Commodity Offtake Portfolio Model</h1>
          <p style={{ fontSize: 12, color: '#999', margin: '6px 0 0' }}>Q-learning agents · Python backend · numpy-accelerated training</p>
        </div>
        <div style={{ display: 'flex', gap: 3, marginBottom: 20, background: '#fff', borderRadius: 8, padding: 4, border: '1px solid #e4e6eb', flexWrap: 'wrap', width: 'fit-content' }}>
          {TABS.map((t, i) => <Pill key={t} active={tab === i} onClick={() => setTab(i)} disabled={i > 0 && !has}>{t}</Pill>)}
        </div>
        {renderTab()}
      </div>
    </div>
  );
}
