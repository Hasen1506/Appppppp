#!/usr/bin/env python3
"""
Enterprise Inventory Simulator — Full-Stack App
=================================================
Single file: serves HTML frontend + MILP solver API.

DEPLOY TO REPLIT:
  1. Create new Python Replit
  2. Upload: app.py + Enterprise_Simulator_v78.html + milp_v3_fixed.py
  3. In shell: pip install flask flask-cors pulp numpy
  4. Click Run → app serves on port 5000
  5. The simulator's "🖥 Solve MILP (server)" button auto-connects

LOCAL:
  pip install flask flask-cors pulp numpy
  python app.py
  → Open http://localhost:5000
"""
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pulp
import math
import json
import time
import os

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    """Serve the simulator HTML"""
    html_path = os.path.join(os.path.dirname(__file__), 'Enterprise_Simulator_v78.html')
    if os.path.exists(html_path):
        return send_file(html_path)
    return "Upload Enterprise_Simulator_v78.html to the same directory", 404

@app.route('/health')
def health():
    return jsonify({"status": "ok", "solver": "CBC"})

@app.route('/solve', methods=['POST'])
def solve():
    try:
        return _solve_inner()
    except Exception as exc:
        import traceback
        return jsonify({"error": str(exc), "trace": traceback.format_exc()}), 500

def _solve_inner():
    t0 = time.time()
    data = request.json
    products = data.get('products', [])
    params = data.get('params', {})
    cap_mode = data.get('capacity_mode', 'parallel')
    
    T = params.get('T', 261)
    shelf = params.get('shelf', 75)
    carry = params.get('carry_rate', 0.24)
    sw_cost = params.get('switch_cost', 50)
    wh_max = params.get('wh_max', 500)
    fixed = params.get('fixed_daily', 100)
    bo_on = params.get('backorder_on', False)
    salv = params.get('salvage_rate', 0.80)
    mape = params.get('mape_pct', 15)
    sl = params.get('service_level', 0.95)
    base_cap = params.get('capacity', 21)
    
    n = len(products)
    if not n: return jsonify({"error": "No products"}), 400
    
    # Weekly aggregation
    bk = 5 if T > 30 else 1
    T_orig = T
    if bk > 1: T = (T + bk - 1) // bk; shelf = max(1, shelf // bk)
    
    z = {0.90:1.282, 0.95:1.645, 0.99:2.326}.get(sl, 1.645)
    wd = [22,20,23,21,22,21,23,22,20,23,21,20]
    sm_offset = params.get('start_month', 0)  # planning horizon start month (0=Jan)

    demand = {}; all_parts = []
    for k, prod in enumerate(products):
        dd = []
        monthly = prod.get('demand_monthly', prod.get('demand', []))
        for m in range(min(12, len(monthly))):
            wday = wd[(sm_offset + m) % 12]  # correct working-days for actual calendar month
            per = monthly[m] // wday; rem = monthly[m] - per * wday
            dd += [per + (1 if i < rem else 0) for i in range(wday)]
        dd = dd[:T_orig]
        while len(dd) < T_orig: dd.append(dd[-1] if dd else 8)

        if bk > 1:
            bucketed = [sum(dd[w*bk:min((w+1)*bk, T_orig)]) for w in range(T)]
            demand[k] = bucketed
        else:
            demand[k] = dd

        fy = prod.get('yield_pct', 0.95)
        parts = prod.get('parts', [])
        # Split transport into variable (perUnit/ltl/contract) vs fixed-per-order (perShip/ftl)
        for p in parts:
            mode = p.get('trans_mode', 'perUnit')
            if mode in ('perShip', 'ftl'):
                p['uc_trans'] = 0.0
                p['ord_trans'] = p.get('trans', 0.5)
            else:  # perUnit, ltl, contract → variable landed cost
                p['uc_trans'] = p.get('trans', 0.5)
                p['ord_trans'] = 0.0
        rm = sum(p['qty'] / p.get('partYield', 0.97) / fy * (p['cost'] + p['uc_trans']) for p in parts)
        prod['uc'] = rm
        prod['fh'] = rm * carry / 365 * bk
        prod['wc'] = rm * (1 - salv)
        avg = sum(demand[k]) / T
        sig = avg * mape / 100
        alt = sum(p.get('lt', 3) for p in parts) / max(len(parts), 1)
        if bk > 1: alt = max(1, alt / bk)
        prod['ss'] = max(1, math.ceil(z * sig * math.sqrt(alt)))
        sc = prod.get('bo_scenario', 'A')
        sp = prod.get('sell_price', 100)
        prod['bp'] = (sp - rm) if sc == 'A' else (sp * 0.10 if sc == 'B' else sp * 0.20 + rm)
        
        for p in parts:
            p['ki'] = k
            p['ltb'] = max(1, p.get('lt', 3) // bk) if bk > 1 else p.get('lt', 3)
            all_parts.append(p)
    
    np2 = len(all_parts)
    mlt = max((p['ltb'] for p in all_parts), default=1)
    days = list(range(T))
    rng = range(-mlt, T)
    cap = [base_cap * bk] * T if bk > 1 else [base_cap] * T
    
    m = pulp.LpProblem("MILP", pulp.LpMinimize)
    p_v = {(k,t): pulp.LpVariable(f"p_{k}_{t}", 0, cat="Integer") for k in range(n) for t in days}
    if cap_mode == 'parallel':
        y = {(k,t): pulp.LpVariable(f"y_{k}_{t}", cat="Binary") for k in range(n) for t in days}
        sw = {(k,t): pulp.LpVariable(f"sw_{k}_{t}", cat="Binary") for k in range(n) for t in days}
    else:
        y = {t: pulp.LpVariable(f"y_{t}", cat="Binary") for t in days}
        sw = {t: pulp.LpVariable(f"sw_{t}", cat="Binary") for t in days}
    Inv = {(k,t): pulp.LpVariable(f"I_{k}_{t}", 0) for k in range(n) for t in range(T+1)}
    e = {(k,t): pulp.LpVariable(f"e_{k}_{t}", 0) for k in range(n) for t in days}
    s = {(k,t): pulp.LpVariable(f"s_{k}_{t}", 0) for k in range(n) for t in days}
    ssv = {(k,t): pulp.LpVariable(f"ssv_{k}_{t}", 0) for k in range(n) for t in days}
    r_v = {(i,t): pulp.LpVariable(f"r_{i}_{t}", 0, cat="Integer") for i in range(np2) for t in rng}
    RI = {(i,t): pulp.LpVariable(f"RI_{i}_{t}", 0) for i in range(np2) for t in range(T+1)}
    zo = {(i,t): pulp.LpVariable(f"zo_{i}_{t}", cat="Binary") for i in range(np2) for t in rng}
    
    for k in range(n): m += Inv[k,0] == 0
    for i in range(np2): m += RI[i,0] == 0
    
    obj = []
    for t in days:
        if cap_mode == 'parallel': obj.append(sw_cost * pulp.lpSum(sw[k,t] for k in range(n)))
        else: obj.append(sw_cost * sw[t])
        obj += [pulp.lpSum(products[k]['fh']*Inv[k,t+1] for k in range(n)),
                pulp.lpSum(products[k]['uc']*p_v[k,t] for k in range(n)),
                pulp.lpSum(products[k]['wc']*e[k,t] for k in range(n)),
                pulp.lpSum(products[k]['bp']*s[k,t] for k in range(n)),
                pulp.lpSum(3*products[k]['fh']*ssv[k,t] for k in range(n)),
                fixed * bk]
    for i in range(np2):
        pt = all_parts[i]; rh = pt['cost'] * pt.get('hold_pct',24)/100/365*bk
        obj += [pulp.lpSum(rh*RI[i,t+1] for t in days),
                pulp.lpSum(pt['cost']*r_v[i,t]+(pt.get('ord_cost',50)+pt.get('ord_trans',0))*zo[i,t] for t in rng)]
    m += pulp.lpSum(obj)
    
    for t in days:
        for k in range(n):
            fy = products[k].get('yield_pct', 0.95)
            m += Inv[k,t+1] == Inv[k,t]+p_v[k,t]*fy-demand[k][t]-e[k,t]+s[k,t]
            m += ssv[k,t] >= products[k]['ss']-Inv[k,t+1]
            if not bo_on: m += s[k,t] == 0
            if t >= shelf:
                m += pulp.lpSum(e[k,tau] for tau in range(max(0,t-shelf+1),t+1)) >= pulp.lpSum(p_v[k,tau]*fy for tau in range(max(0,t-2*shelf+1),t-shelf+1)) - pulp.lpSum(demand[k][tau] for tau in range(max(0,t-2*shelf+1),t-shelf+1))
        if cap_mode == 'parallel':
            for k in range(n): m += p_v[k,t] <= cap[t]*y[k,t]
        else: m += pulp.lpSum(p_v[k,t] for k in range(n)) <= cap[t]*y[t]
        if cap_mode == 'parallel':
            for k in range(n): m += sw[k,t] >= y[k,t]-(y[k,t-1] if t>0 else 0)
        else: m += sw[t] >= y[t]-(y[t-1] if t>0 else 0)
        m += pulp.lpSum(Inv[k,t+1] for k in range(n)) <= wh_max*bk
    
    for i in range(np2):
        pt=all_parts[i]; lt2=pt['ltb']; ki=pt['ki']
        for t in days:
            od=t-lt2
            if -mlt<=od<T: m+=r_v[i,od]*pt.get('partYield',0.97)>=p_v[ki,t]*pt['qty']
            if t>0: m+=RI[i,t]==RI[i,t-1]+(r_v[i,od]*pt.get('partYield',0.97) if -mlt<=od<T else 0)-p_v[ki,t]*pt['qty']
            m+=RI[i,t]<=pt.get('rm_cap',1000)
        for t in rng:
            m+=r_v[i,t]>=pt.get('moq',20)*zo[i,t]
            m+=r_v[i,t]<=pt.get('max_order',200)*zo[i,t]
    
    m.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=240, gapRel=0.05))
    
    status = pulp.LpStatus[m.status]
    cost = round(pulp.value(m.objective), 2) if m.status == 1 else None
    
    mps = {}; fills = {}
    for k in range(n):
        wp = [int(p_v[k,t].varValue or 0) for t in days]
        if bk > 1:
            dl = []
            for w, wv in enumerate(wp):
                wd2 = min(bk, T_orig-w*bk)
                pd = wv//wd2 if wd2>0 else 0; rm2=wv-pd*wd2
                for d2 in range(wd2): dl.append(pd+(1 if d2<rm2 else 0))
            mps[products[k].get('name',f'P{k}')] = dl
        else:
            mps[products[k].get('name',f'P{k}')] = wp
        td=sum(demand[k]); ts=sum(s[k,t].varValue or 0 for t in days)
        fills[products[k].get('name',f'P{k}')] = round((td-ts)/td*100,1) if td>0 else 100
    
    proc = {}
    for i in range(np2):
        pt=all_parts[i]; nm=f"{products[pt['ki']].get('name','?')} <- {pt.get('name','?')}"
        pos=[]
        for t in rng:
            rv=r_v[i,t].varValue or 0
            if rv>0.5:
                ad=t*bk if bk>1 else t
                pos.append({"day":ad+1,"qty":int(rv),"arrives":ad+pt.get('lt',3)+1})
        if pos: proc[nm]=pos
    
    return jsonify({
        "source":"MILP_server_v3","status":status,"cost":cost,
        "mps_plan":mps,"procurement":proc,"fill_rates":fills,
        "ss_used":{products[k].get('name',f'P{k}'):products[k]['ss'] for k in range(n)},
        "solve_time":round(time.time()-t0,2),"periods":T,"bucketed":bk>1
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Enterprise Simulator — http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
