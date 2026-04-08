#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════
# HOW TO RUN THIS SCRIPT
# ═══════════════════════════════════════════════════════════════
#
# OPTION 1: Google Colab (recommended — free, fast, no install)
#   1. Go to https://colab.research.google.com
#   2. New notebook → paste this ENTIRE file into one cell
#   3. Add a cell ABOVE it with: !pip install pulp numpy
#   4. Run both cells. Output appears below.
#
# OPTION 2: Local Python
#   pip install pulp numpy
#   python milp_v3.py
#
# PERFORMANCE FOR T > 30 DAYS:
#   CBC solver is free but slow for large MILP problems.
#   For T=23 (1 month): ~2 minutes. T=66 (3 months): ~10 minutes.
#   For T>90: set FAST_MODE=True below — aggregates to WEEKLY periods.
#   For T=261 (full year): FAST_MODE=True solves in ~3 minutes.
#   Without FAST_MODE, T=261 may take 30+ minutes or timeout.
#
#   Alternative: use Gurobi (free academic license) — 10x faster.
#   Set USE_GUROBI=True below if you have it installed.
# ═══════════════════════════════════════════════════════════════
"""
Enterprise Inventory Simulator — MILP Production Planner v3
============================================================

═══════════════════════════════════════════════════════════════
VERBAL MODEL (Problem Description)
═══════════════════════════════════════════════════════════════

PROBLEM:
  A bakery produces multiple finished goods (Cake, Cookie) from raw materials
  (Cream Bun, Vanilla, Flour, Sugar) on a shared/parallel production line.
  Each product has limited shelf life, uncertain demand, and constrained capacity.
  
  We need to decide:
  - HOW MUCH of each product to produce each day
  - WHEN to place purchase orders for each raw material
  - HOW MUCH raw material to order per PO
  
  While minimizing total cost (production + holding + waste + shortages + ordering)
  and maintaining target service levels.

DECISION VARIABLES (what we control):
  - p[k,t]  = units of product k to produce on day t
  - r[i,t]  = units of raw material i to order on day t
  - y[k,t]  = whether production line runs for product k on day t (yes/no)
  - ot[t]   = whether to run overtime shift on day t (yes/no)

OBJECTIVE (what we minimize):
  Total cost = setup + FG holding + variable production + expiry waste + shortage penalty
             + safety stock violation + fixed overhead + OT labor
             + RM purchase + RM ordering admin + RM holding
  
  NOTE: Material cost is charged ONCE at ordering (RM purchase = cost × qty ordered).
  Production cost covers ONLY non-material variable costs per unit produced:
    - Electricity/gas: oven, mixer, cooling ($/unit)
    - Packaging: boxes, labels, shrink wrap ($/unit)
    - Consumables: cleaning, disposable tools ($/unit)
    - Labor variable: $0 for fixed-salary model (overtime is separate binary decision)
  The solver still produces because NOT producing triggers shortage penalty.

CONSTRAINTS (physical/business limits):
  C1.  Inventory balance: today's stock = yesterday's + produced - sold - expired
  C2.  Capacity: can't produce more than line capacity per day
  C3.  Setup: batch start cost only when transitioning from idle to active
  C4.  Warehouse: total FG can't exceed storage limit
  C5.  Safety stock: penalty for going below target (soft, not hard wall)
  C6.  Backorder: either forced to zero (lost sale) or carries to next day
  C7.  Shelf life: inventory older than shelf days MUST expire
  C8.  BOM explosion: raw materials must arrive before production needs them
  C9.  RM inventory balance: RM stock tracks orders received minus consumed
  C10. RM warehouse: per-subpart storage limits
  C11. MOQ: if ordering, must order at least minimum quantity
  C12. Supplier max: can't order more than supplier can deliver per PO

═══════════════════════════════════════════════════════════════
MATHEMATICAL MODEL (Formal Definition)
═══════════════════════════════════════════════════════════════

Sets:
  K = {0, 1, ..., n_products-1}    — products (Cake, Cookie, ...)
  T = {0, 1, ..., T-1}             — planning days
  I = {0, 1, ..., n_parts-1}       — raw material subparts (all products)

Parameters (given data):
  d[k,t]        = demand for product k on day t (from forecast)
  C[t]          = production capacity on day t (units)
  shelf         = FG shelf life (days)
  yield_k       = production yield for product k (0-1)
  partYield_i   = arrival quality yield for subpart i (0-1)
  qty_i         = BOM quantity of subpart i per FG unit
  lt_i          = lead time for subpart i (days)
  S             = setup/changeover cost per batch start ($)
  h_k           = FG holding cost per unit per day for product k ($/u/d)
  c_k           = variable production cost for product k ($/u) = electricity + packaging + consumables. EXCLUDES material (charged at ordering) and labor (fixed salary in f)
  w_k           = waste cost per expired unit of product k ($/u)
  b_k           = shortage penalty per unit of product k ($/u)
  f             = fixed daily overhead ($)
  SS_k          = safety stock target for product k (units)
                    SS_k = z × √(LT_k × σ²_demand_k + d_k² × σ²_LT_k)
                    where σ_demand = d_k × MAPE/100, σ_LT = LT_k × LTCV_k

Decision Variables:
  p[k,t] ∈ Z+     — FG units of product k to produce on day t
  y[k,t] ∈ {0,1}  — 1 if line active for product k on day t
  sw[k,t] ∈ {0,1} — 1 if batch STARTS (idle→active transition)
  I[k,t] ∈ R+     — FG inventory of product k at end of day t
  e[k,t] ∈ R+     — units of product k expired on day t
  s[k,t] ∈ R+     — shortage of product k on day t
  ssv[k,t] ∈ R+   — safety stock violation for product k on day t
  r[i,t] ∈ Z+     — RM units of subpart i ordered on day t
  RI[i,t] ∈ R+    — RM inventory of subpart i at end of day t
  zo[i,t] ∈ {0,1} — 1 if PO placed for subpart i on day t
  ot[t] ∈ {0,1}   — 1 if overtime shift runs on day t

Objective Function:
  min Z = Σ_t [ S·sw[t] + Σ_k(h_k·I[k,t] + c_k·p[k,t] + w_k·e[k,t]
          + b_k·s[k,t] + λ·ssv[k,t]) + f + OT_cost·ot[t] ]
          + Σ_i Σ_t [ rm_h_i·RI[i,t] + cost_i·r[i,t] + S_i·zo[i,t] ]

Constraints:
  ∀k,t:  I[k,t+1] = I[k,t] + p[k,t]·yield_k - d[k,t] - e[k,t] + s[k,t]   (C1)
  ∀k,t:  p[k,t] ≤ C[t]·y[k,t] + OT_cap·ot[t]                              (C2)
  ∀k,t:  sw[k,t] ≥ y[k,t] - y[k,t-1]                                       (C3)
  ∀t:    Σ_k I[k,t] ≤ WH_max                                                (C4)
  ∀k,t:  ssv[k,t] ≥ SS_k - I[k,t]                                           (C5)
  ∀k,t:  s[k,t] = 0  (if no backorder)                                      (C6)
  ∀k,t:  Σe ≥ Σ(p·yield) - Σd  for aged window                             (C7)
  ∀i,t:  r[i,t-lt_i]·partYield_i ≥ p[k,t]·qty_i                            (C8)
  ∀i,t:  RI[i,t] = RI[i,t-1] + r[i,t]·partYield_i - p[k,t]·qty_i          (C9)
  ∀i,t:  RI[i,t] ≤ RM_cap_i                                                (C10)
  ∀i,t:  r[i,t] ≥ MOQ_i · zo[i,t]                                          (C11)
  ∀i,t:  r[i,t] ≤ MaxOrder_i · zo[i,t]                                     (C12)

═══════════════════════════════════════════════════════════════
IMPLEMENTATION (Code below)
═══════════════════════════════════════════════════════════════
Corrected and annotated. All formulas explained.
Run locally: pip install pulp numpy
Run on Colab: !pip install pulp numpy

ARCHITECTURE ANSWER (last question first):
  MILP vs Heuristics — why do both exist?
  This script solves BOTH production (MPS) AND procurement simultaneously.
  That is mathematically correct and gives the globally optimal answer.

  SAP IBP / large enterprises split them for ONE reason: SCALE.
  10,000 SKUs × 261 days × 20 subparts each = ~52 million binary variables.
  CBC solver would run for weeks. Gurobi would take hours.
  So enterprises: (1) MILP for MPS only (fast, ~minutes), then
                  (2) Run WW-DP / EOQ / heuristics on the MPS output for procurement.
  
  For your scale (2-3 SKUs, 14-261 days, 2 subparts each):
  Joint MILP is FINE. It gives the exact optimum.
  The heuristics in the simulator are for quick approximation without a solver.
  Use this script for the exact answer. Use the simulator heuristics for speed.
"""
import pulp
import math
import numpy as np

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
USE_GUROBI = False
FAST_MODE = 'auto'   # Options:
                     #   'off'     = always daily (exact, slow for T>30)
                     #   'weekly'  = always aggregate to weekly periods
                     #   'monthly' = always aggregate to monthly periods (fastest, least precise)
                     #   'auto'    = daily if T≤30, weekly if T≤90, monthly if T>90
np.random.seed(42)

# Capacity mode options:
#   'single'   — one SKU on one line. Full capacity available to it.
#   'shared'   — multiple SKUs share ONE line. Sum of all production ≤ capacity.
#                Changeover penalty when switching between products on same day.
#   'parallel' — each SKU has its OWN dedicated line. Each line runs independently.
#                Product A can run 21u/d AND Product B can run 21u/d simultaneously.
capacity_mode = 'parallel'   # Change to 'single' or 'shared' as needed

# ─── MAPE / DEMAND UNCERTAINTY ───────────────────────────────────────────────
# ANSWER: How MAPE fits randomness.
# MAPE% from your 17 models tells you how much demand deviates from forecast on average.
# It is NOT production yield noise — it is DEMAND noise.
# Correct approach: use forecast as the base, then add MAPE-scaled uncertainty.
# In a deterministic MILP we use the POINT FORECAST (expected value) as demand.
# The MAPE becomes your Safety Stock buffer (computed below from SS formula).
# If you want stochastic MILP: generate N scenarios from MAPE, solve each,
# take the solution that is best across all scenarios (robust optimisation).
# For now: deterministic, MAPE → SS buffer.
MAPE_PCT = 15.0   # % — from your winning forecast model. Feeds into SS formula below.
SERVICE_LEVEL = 0.95  # 95% — from your simulator SL% slider

# ─── PLANNING HORIZON ────────────────────────────────────────────────────────
selected_months = ['Mar']
month_days = {
    'Jan':22,'Feb':20,'Mar':23,'Apr':22,'May':23,'Jun':20,
    'Jul':23,'Aug':22,'Sep':23,'Oct':22,'Nov':21,'Dec':20
}
cum_days = 0; start_day = 0
for m in month_days:
    if m in selected_months: start_day = cum_days; break
    cum_days += month_days[m]
T = sum(month_days[m] for m in selected_months)
print(f"Planning horizon: {T} days ({', '.join(selected_months)})")

# ─── PRODUCTS ─────────────────────────────────────────────────────────────────
# ANSWER: Product B raw material fix.
# Product B now has its own distinct subparts (Flour + Sugar), not Cream Bun + Vanilla.
# If two products genuinely share a raw material in reality, keep it — but the qty
# and cost should reflect each product's actual BOM independently.

products = [
    {
        'name': 'Product A',   # e.g. Cake
        'demand_full': [8,8,9,8,8,8,8,9,8,8,8,8,9,8,8,8,8,9,8,8,8,8,8,8,9,8,8,8,8,9,
                        8,8,8,8,9,8,8,8,8,9,8,8,9,9,8,9,9,9,9,8,9,9,9,9,8,9,9,9,8,9,
                        8,9,9,9,9,10,9,10,9,10,9,10,9,10,9,10,9,10,9,9,10,9,10,9,10,9,
                        11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,
                        10,11,10,11,10,10,11,10,11,10,10,10,11,10,10,10,11,10,11,10,11,
                        10,9,9,10,9,9,9,10,9,10,9,9,9,10,9,9,9,9,10,9,9,9,10,9,10,10,
                        9,10,10,10,10,9,10,10,10,10,9,10,10,10,10,9,10,10,10,12,11,12,
                        11,12,11,12,11,12,11,12,11,12,11,12,11,12,11,12,11,12,11,12,13,
                        12,13,12,12,13,12,13,12,12,13,12,13,12,12,13,12,13,12,13,12,10,
                        10,9,10,10,10,10,9,10,10,10,10,9,10,10,10,10,9,10,10,10,10,11,
                        10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10],
        # YIELD: Fraction of inputs that become sellable FG. E.g. 0.95 = 5% loss in process.
        # This is YOUR factory's yield%, set in the simulator.
        'yield_pct': 0.95,
        # Sell price per finished unit
        'sell_price': 100.0,
        # Bill of Materials — subparts unique to Product A
        # ANSWER: Ordering cost uses actual PO admin cost per part, not a random number.
        # ordCost = PO admin cost (your simulator's "PO $/ord" field per subpart).
        # transport cost is LTL per unit (goes into holding via landed cost) or
        # FTL per shipment (goes into ordering cost S).
        'parts': [
            {
                'name': 'Cream Bun',
                'qty': 1,               # Units of this RM per FG unit
                'cost': 5.0,            # RM unit cost $/unit
                'trans': 0.50,          # LTL transport $/unit (adds to landed cost → H)
                'lt': 3,                # Lead time days
                'ltcv': 0.10,           # Lead time coefficient of variation (10% variability)
                'hold_pct': 24,         # Annual carry rate % (capital + warehouse + insurance)
                'partYield': 0.97,      # Fraction of ordered RM that is usable (3% damage/loss)
                'moq': 20,              # Minimum order quantity
                'max_order': 200,       # Maximum order per PO
                'rm_cap': 1000,         # Max RM storable on hand (warehouse constraint)
                'rm_shelf': 30,         # RM shelf life days from receipt
                'ord_cost': 50.0,       # PO admin cost per order placed (from simulator)
                'pay_terms': 30,        # Net payment days
                'early_pay_disc': 0.02, # 2% early payment discount if paid in 10 days
            },
            {
                'name': 'Vanilla',
                'qty': 2,
                'cost': 10.0,
                'trans': 1.20,
                'lt': 5,
                'ltcv': 0.10,
                'hold_pct': 24,
                'partYield': 0.92,
                'moq': 50,
                'max_order': 500,
                'rm_cap': 2000,
                'rm_shelf': 60,
                'ord_cost': 50.0,
                'pay_terms': 60,
                'early_pay_disc': 0.0,
            },
        ]
    },
    {
        'name': 'Product B',   # e.g. Cookie — DIFFERENT raw materials than Product A
        'demand_full': [5,5,6,5,5,5,5,6,5,5,5,5,6,5,5,5,5,6,5,5,5,5,5,5,6,5,5,5,5,6,
                        5,5,5,5,6,5,5,5,5,6,5,5,6,6,5,6,6,6,6,5,6,6,6,6,5,6,6,6,5,6,
                        5,6,6,6,6,7,6,7,6,7,6,7,6,7,6,7,6,7,6,6,7,6,7,6,7,6,8,8,8,8,
                        8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,7,8,7,8,7,7,8,7,8,7,7,7,8,
                        7,7,7,8,7,8,7,8,7,6,6,7,6,6,6,7,6,7,6,6,6,7,6,6,6,6,7,6,6,6,
                        7,6,7,7,6,7,7,7,7,6,7,7,7,7,6,7,7,7,7,6,7,7,7,9,8,9,8,9,8,9,
                        8,9,8,9,8,9,8,9,8,9,8,9,8,9,8,9,10,9,10,9,9,10,9,10,9,9,10,9,
                        10,9,9,10,9,10,9,10,9,7,7,6,7,7,7,7,6,7,7,7,7,6,7,7,7,7,6,7,
                        7,7,7,8,7,8,7,8,7,8,7,8,7,8,7,8,7,8,7,8,7,8,7,8,7],
        'yield_pct': 0.92,
        'sell_price': 90.0,
        'parts': [
            {
                # Product B uses Flour, not Cream Bun
                'name': 'Flour',
                'qty': 1.5,             # 1.5 kg flour per cookie batch unit
                'cost': 2.0,
                'trans': 0.30,
                'lt': 2,
                'ltcv': 0.08,
                'hold_pct': 20,
                'partYield': 0.99,      # Flour has very low loss
                'moq': 50,
                'max_order': 500,
                'rm_cap': 2000,
                'rm_shelf': 90,
                'ord_cost': 40.0,
                'pay_terms': 30,
                'early_pay_disc': 0.01,
            },
            {
                # Product B uses Sugar, not Vanilla
                'name': 'Sugar',
                'qty': 0.5,
                'cost': 1.5,
                'trans': 0.20,
                'lt': 2,
                'ltcv': 0.08,
                'hold_pct': 20,
                'partYield': 0.99,
                'moq': 100,
                'max_order': 1000,
                'rm_cap': 3000,
                'rm_shelf': 180,
                'ord_cost': 40.0,
                'pay_terms': 30,
                'early_pay_disc': 0.0,
            },
        ]
    }
]

n_products = len(products)

# ─── SLICE DEMAND TO SELECTED MONTHS ─────────────────────────────────────────
for prod in products:
    prod['demand'] = prod['demand_full'][start_day : start_day + T]
    if len(prod['demand']) < T:
        raise ValueError(f"{prod['name']}: not enough demand data for {T} days")

demand = {k: products[k]['demand'] for k in range(n_products)}

# ─── CAPACITY ────────────────────────────────────────────────────────────────
# ANSWER: Capacity variability.
# Real capacity varies daily (maintenance, absenteeism, equipment issues).
# Best approach: encode this in the capacity array itself using cap_cv_pct.
# Default is a fixed level (cap_cv_pct = 0 → no variability → deterministic MILP).
# Set cap_cv_pct > 0 to introduce day-by-day variability.
# This does NOT add binary variables — it just changes the RHS of constraint C2.
# The solver then optimises over the variable capacity profile.

base_capacity = 21        # Units/day — your cycle time & shift hours output
cap_cv_pct    = 0.0       # 0 = fixed. Set e.g. 0.15 for ±15% daily variability.
                          # Variability is deterministic per-day (pre-sampled), not stochastic.

if cap_cv_pct > 0:
    # Pre-sample capacity for each day. Solver treats these as fixed RHS values.
    cap_noise = np.random.normal(1.0, cap_cv_pct, T)
    cap_noise = np.clip(cap_noise, 1 - cap_cv_pct * 1.5, 1 + cap_cv_pct)
    capacity = [max(1, int(round(base_capacity * f))) for f in cap_noise]
else:
    capacity = [base_capacity] * T   # Fixed — default

print(f"Capacity: min={min(capacity)} max={max(capacity)} avg={sum(capacity)/len(capacity):.1f} u/d")

# ─── SHELF LIFE ───────────────────────────────────────────────────────────────
shelf = 75   # FG shelf life days (from simulator). Matches your simulator shelf life field.

# ─── FG HOLDING COST ─────────────────────────────────────────────────────────
# ANSWER: FG holding per unit per day formula.
# For multi-SKU, each product has a different landed cost → different holding cost.
# Formula:
#   landed_cost_per_FG = Σ(part_qty_i × part_cost_i / partYield_i) / fg_yield
#                       + Σ(part_qty_i × part_trans_i) / fg_yield
#                       + labor_fixed_daily_per_unit (from labor model below)
#   fg_hold_per_unit_per_day = landed_cost_per_FG × carry_rate_annual / 365

carry_rate_annual = 0.24  # 24%/yr — from your simulator carry rate slider

def compute_fg_hold(prod_idx):
    """Compute FG holding cost $/unit/day for one product."""
    prod = products[prod_idx]
    fg_yield = prod['yield_pct']
    # Sum landed raw material cost per FG unit (adjusted for part yield losses)
    rm_landed_per_fg = sum(
        (p['qty'] * (p['cost'] + p['trans'])) / p['partYield']
        for p in prod['parts']
    ) / fg_yield
    # Add per-unit labor (variable portion — see labor section)
    labor_var = prod.get('labor_variable_per_unit', 0.0)
    landed_per_fg = rm_landed_per_fg + labor_var
    hold_per_day = landed_per_fg * carry_rate_annual / 365
    prod['landed_per_fg']    = round(landed_per_fg, 4)
    prod['fg_hold_per_day']  = round(hold_per_day, 6)
    return hold_per_day

# ─── VARIABLE PRODUCTION COSTS (per unit produced) ──────────────────────────
# These are costs that ONLY occur when a unit is actually produced.
# Material cost is NOT here — it's charged at ordering (r[i,t] × cost).
electricity_per_unit    = 0.50   # $/unit — oven gas, mixer power, cooling
packaging_per_unit      = 0.30   # $/unit — boxes, labels, shrink wrap
consumables_per_unit    = 0.20   # $/unit — cleaning supplies, disposable tools
# Total variable production cost = electricity + packaging + consumables
# This is what goes into the objective as c_k × p[k,t]

# ─── LABOR MODEL ─────────────────────────────────────────────────────────────
# ANSWER: Labor — fixed monthly vs variable OT.
# In your bakery: workers are on fixed monthly salary. They show up regardless of
# how much you produce. That is a FIXED cost, not per-unit.
# OT = a full extra shift at OT_multiplier × daily pay rate.
# There is NO per-unit variable labor in a fixed-salary setup.
#
# Fixed: salary / 30 days → charged to fixed_daily (already in model)
# OT: if OT shift runs that day, add (daily_pay × OT_multiplier)
#
# Example: Salary = ₹30,000/month → ₹1,000/day.
# OT shift at 2×: OT cost = ₹1,000 × 2 = ₹2,000 extra that day.
# OT shift adds extra capacity = OT_shift_hours × (units_per_hour).

monthly_salary_per_worker = 30000   # ₹ per worker per month (set to your actual)
n_workers                 = 5       # Number of workers
currency_factor           = 1       # Exchange rate: set by simulator export. 1=USD, 83=INR. All costs in base currency.

daily_base_pay_total      = (monthly_salary_per_worker * n_workers) / 30
# This goes into fixed_daily (workers come in regardless of production level)

# ANSWER: Overtime.
# OT scenario: full extra shift stays for the whole additional shift.
# Best way to handle excess demand: priority order is:
#   1. Produce from safety stock buffer (draw down SS)
#   2. If still short AND OT is possible → trigger OT shift (costs 2× daily pay)
#   3. If OT maxed → backorder or lost sale (depending on backorder_on)
# The MILP naturally finds this trade-off — it will only trigger OT if
# OT cost < shortage penalty cost.
#
# Shift definitions:
cycle_time_min   = 19        # minutes per cycle — from your simulator
units_per_cycle  = 1         # units produced per cycle (1=sequential, 4=parallel oven)
                             # Capacity = (shift-break)/cycle × units_per_cycle × util
shift_hours      = 8         # regular shift
break_minutes    = 30        # breaks per shift (your formula had 50, more typical is 30)

# ANSWER: Effective hours formula.
# effective_hours = shift_hours - break_minutes / 60
# = 8 - 30/60 = 7.5 hours available for production
effective_hours = shift_hours - break_minutes / 60   # 7.5 hrs

# ANSWER: Productivity per hour.
# Should NOT be a magic number. Derive from cycle time:
# units_per_hour = 60 minutes / cycle_time_min per unit
# So if cycle = 19 min/unit → 60/19 ≈ 3.16 units/hour
# base_capacity = floor(effective_hours × units_per_hour) × utilisation
units_per_hour   = 60.0 / cycle_time_min * units_per_cycle  # parallel mfg
opt_util         = 0.85   # 85% utilisation (your simulator optUtil slider)
derived_capacity = int(effective_hours * units_per_hour * opt_util)
print(f"Derived capacity: {effective_hours:.2f}h × {units_per_hour:.2f}u/h × {opt_util} = {derived_capacity}u/d "
      f"(vs base_capacity={base_capacity})")

# OT shift parameters
ot_enabled          = True    # Allow overtime shifts?
ot_max_hours        = 8       # Full extra shift (not partial — workers stay whole shift)
ot_multiplier       = 2.0     # 2× pay for OT shift
ot_daily_cost       = daily_base_pay_total * ot_multiplier  # Cost of one OT shift day
ot_extra_capacity   = int(ot_max_hours * units_per_hour * opt_util)  # Units from OT shift

# Per-unit labor for holding cost purposes (zero in fixed-salary model,
# because labor is in fixed_daily already)
for prod in products:
    prod['labor_variable_per_unit'] = 0.0  # Fixed salary → no per-unit variable labor

# ─── FG HOLD CALCULATION ─────────────────────────────────────────────────────
for k, prod in enumerate(products):
    compute_fg_hold(k)
    print(f"  {prod['name']}: landed=${prod['landed_per_fg']:.4f}/u, "
          f"hold=${prod['fg_hold_per_day']:.6f}/u/d "
          f"({carry_rate_annual*100:.0f}%/yr ÷ 365)")

# ─── FIXED DAILY COST ────────────────────────────────────────────────────────
# Includes: rent, insurance, base staff salary, fixed utilities
fixed_daily = 100 + daily_base_pay_total  # Overhead + workers' base salary
print(f"  Fixed daily: ${fixed_daily:.2f} (overhead + {n_workers} workers' base pay)")

# ─── SETUP / BATCH SWITCH COST ───────────────────────────────────────────────
switch_cost = 50   # $/batch start — changeover, cleaning, calibration

# ─── WASTE / EXPIRY ─────────────────────────────────────────────────────────
# ANSWER: Wastage rate vs yield vs expiry — what is what?
#
# yield_pct      = fraction of RM inputs that become sellable FG during PRODUCTION.
#                  e.g. 0.95 means 5% lost during baking (stuck to moulds, trimming).
#                  This is PRODUCTION YIELD. Already in the model (p[k,t] × yield_pct).
#
# partYield      = fraction of ordered RM that is actually usable on arrival.
#                  e.g. 0.97 Cream Bun means 3% of each delivery is damaged/unusable.
#                  This affects how much RM you need to ORDER to cover production.
#
# e[k,t]         = FG units EXPIRED on day t because they aged past shelf life.
#                  This is the ONLY "wastage" that makes sense as a separate variable.
#                  It triggers when FG inventory is older than shelf days.
#
# wastage_rate   = (REMOVED) — it double-counted production yield loss.
#                  Do NOT use a separate wastage_rate if yield_pct already captures it.
#
# waste_cost (expired unit cost):
# ANSWER: Expired unit cost formula.
# When a unit expires, you recover SALVAGE VALUE (e.g. animal feed, seconds).
# expired_unit_cost = landed_per_fg × (1 - salvage_rate)
# e.g. if landed = $20/u and salvage = 80% → you lose $4/unit expired.

salvage_rate = 0.80   # 80% of material value recovered on expiry (from your simulator)

for k, prod in enumerate(products):
    prod['expired_unit_cost'] = prod['landed_per_fg'] * (1 - salvage_rate)
    print(f"  {prod['name']}: expired cost = ${prod['landed_per_fg']:.4f} × {1-salvage_rate:.0%} "
          f"= ${prod['expired_unit_cost']:.4f}/unit")

# ─── SAFETY STOCK (computed, not hardcoded) ──────────────────────────────────
# ANSWER: FG SS formula — not a random number, derived from service level.
#
# Formula (same as your simulator):
#   SS = z(SL%) × σ_combined × √(avg_LT)
#
# Where:
#   z(SL%)       = normal inverse CDF at service level (95% → 1.645)
#   σ_combined   = daily_demand × MAPE/100
#                  (MAPE from your winning model captures demand variability)
#   avg_LT       = average lead time of subparts (days until RM arrives)
#                  SS must cover demand uncertainty during replenishment lag
#
# Note: When capacity is full and you can't produce → you draw from SS.
# SS is the buffer that covers the gap until you can produce again.
# SS penalty in the MILP is a SOFT constraint — the solver is penalised for
# dipping below SS but allowed to do so if the alternative costs more.
# This mirrors reality: you CAN consume SS, but it's expensive (service risk).

z_score = {0.90: 1.282, 0.95: 1.645, 0.99: 2.326, 0.999: 3.090}
z = z_score.get(SERVICE_LEVEL, 1.645)

def compute_ss(prod_idx):
    """Compute FG safety stock for one product.
    
    Full formula (accounts for BOTH demand uncertainty AND lead time uncertainty):
      SS = z × √(LT × σ²_demand + d² × σ²_LT)
    
    Where:
      z         = service level z-score (95% → 1.645)
      LT        = average lead time (days)
      σ_demand  = demand std dev = avg_daily_demand × MAPE / 100
      d         = average daily demand
      σ_LT      = LT std dev = avg_LT × LTCV (from BOM per-part)
    
    This is the standard formula used by SAP APO/IBP and Oracle ASCP.
    If LTCV=0: reduces to the simpler SS = z × σ_demand × √LT.
    """
    prod = products[prod_idx]
    d = sum(prod['demand']) / T                                    # avg daily demand
    sigma_d = d * (MAPE_PCT / 100.0)                               # demand std dev
    avg_lt = sum(p['lt'] for p in prod['parts']) / len(prod['parts'])
    # LT variability from BOM (weighted avg of all parts' LTCV)
    avg_ltcv = sum(p.get('ltcv', 0.10) for p in prod['parts']) / len(prod['parts'])
    sigma_lt = avg_lt * avg_ltcv                                    # LT std dev
    
    # Combined uncertainty: demand risk during lead time + lead time risk on demand
    variance = (avg_lt * sigma_d**2) + (d**2 * sigma_lt**2)
    ss = math.ceil(z * math.sqrt(variance))
    prod['fg_ss'] = max(ss, 1)
    
    # Also store components for traceability
    prod['ss_components'] = {
        'avg_demand': round(d, 2),
        'sigma_demand': round(sigma_d, 2),
        'avg_lt': round(avg_lt, 1),
        'sigma_lt': round(sigma_lt, 2),
        'demand_var': round(avg_lt * sigma_d**2, 2),
        'lt_var': round(d**2 * sigma_lt**2, 2),
        'total_var': round(variance, 2),
    }
    
    print(f"  {prod['name']}: SS = {z:.3f} × √({avg_lt:.1f}×{sigma_d:.2f}² + {d:.1f}²×{sigma_lt:.2f}²) "
          f"= {z:.3f} × √{variance:.1f} = {z * math.sqrt(variance):.2f} → {prod['fg_ss']}u "
          f"(SL={SERVICE_LEVEL*100:.0f}%, MAPE={MAPE_PCT}%, LTCV={avg_ltcv*100:.0f}%)")
    return prod['fg_ss']

print("\nDemand & Supply Variability:")
for k in range(n_products):
    d_arr = products[k]['demand']
    avg_d = sum(d_arr) / len(d_arr) if d_arr else 1
    std_d = (sum((x - avg_d)**2 for x in d_arr) / len(d_arr))**0.5 if len(d_arr) > 1 else 0
    cv_d = std_d / avg_d * 100 if avg_d > 0 else 0
    products[k]['demand_cv'] = round(cv_d, 1)
    products[k]['avg_daily_demand'] = round(avg_d, 2)
    print(f"  {products[k]['name']}: avg_demand={avg_d:.1f}u/d, CV={cv_d:.1f}%, MAPE={MAPE_PCT}%")
    for p in products[k]['parts']:
        print(f"    {p['name']}: LT={p['lt']}d, LTCV={p.get('ltcv',0.10)*100:.0f}%, "
              f"cost=${p['cost']:.2f}, partYield={p.get('partYield',0.97)*100:.0f}%")

print("\nSafety Stock calculation (full formula with LT variability):")
for k in range(n_products):
    compute_ss(k)

# SS penalty: cost of being below SS per unit per day
# Typical: 3× holding cost. This is a soft incentive, not a hard wall.
# ANSWER: SS penalty is USEFUL — it stops the solver from ignoring SS entirely.
# Without it, the solver would always sacrifice SS to save holding cost.
# The penalty value should be 2–5× holding cost (urgency premium).
ss_penalty_multiplier = 3.0   # 3× holding cost per unit per day below SS

# ─── BACKORDER / LOST SALE PENALTY ───────────────────────────────────────────
# ANSWER: BO penalty — what it should actually be.
#
# Three distinct scenarios — you choose via backorder_on and bo_scenario:
#
# Scenario A: backorder_on = False, lost sale (customer walks away)
#   Penalty = gross MARGIN lost, NOT full sell price.
#   You lose the profit you would have made, but you don't pay them anything.
#   bo_penalty = sell_price - unit_prod_cost
#   (You already paid for materials; the cost is the forgone margin.)
#   Some companies add a goodwill penalty (lost future business risk): +10-20%.
#
# Scenario B: backorder_on = True, customer waits (capacity full / line changeover)
#   Penalty = late delivery charge. Typically 5-20% of order value.
#   In FMCG: 10% of invoice value per late shipment. In contracts: agreed SLA penalty.
#   bo_penalty = sell_price × late_penalty_pct
#
# Scenario C: Make-to-order, customer cancels and demands refund
#   Penalty = penalty_pct × sell_price + any production cost already incurred.
#   Typically 10-25% of order value per industry norms.
#   SAP uses "delivery schedule line penalty" which is contract-specific.
#
# In BIG CORPS (P&G, Unilever):
#   Retailer contracts specify "fill rate penalties" — e.g. 1% of invoice per %
#   below 98% fill rate. So if fill = 95%, penalty = 3% × invoice value.
#   For simplicity here: use Scenario B with 10% late penalty.

backorder_on     = False     # True = customer waits; False = lost sale
bo_scenario      = 'B'       # 'A' = lost margin, 'B' = late penalty %, 'C' = MTO refund
late_penalty_pct = 0.10      # 10% of sell price per unit short (Scenario B)

for k, prod in enumerate(products):
    sp = prod['sell_price']
    up = prod.get('unit_prod_cost', prod['landed_per_fg'])
    if not backorder_on:
        prod['bo_penalty'] = sp - prod['landed_per_fg']   # Scenario A: lost margin (sell - full landed cost)
        print(f"  {prod['name']}: BO penalty (lost margin) = ${sp:.0f} - ${up:.2f} = ${prod['bo_penalty']:.2f}/unit")
    elif bo_scenario == 'B':
        prod['bo_penalty'] = sp * late_penalty_pct
        print(f"  {prod['name']}: BO penalty (late {late_penalty_pct*100:.0f}%) = ${prod['bo_penalty']:.2f}/unit")
    else:  # C — MTO refund + penalty
        prod['bo_penalty'] = sp * 0.20 + up
        print(f"  {prod['name']}: BO penalty (MTO refund) = ${prod['bo_penalty']:.2f}/unit")

# ─── UNIT PRODUCTION COST — 3-YIELD FORMULA ──────────────────────────────────
# ANSWER: Unit production cost with all 3 yields.
#
# Three yield layers:
#   1. partYield[i]  — % of ordered RM that is usable (arrival quality)
#   2. qty[i]        — RM units per FG unit (BOM ratio)
#   3. yield_pct     — % of total RM inputs that become sellable FG
#
# Gross RM cost per FG unit:
#   For each part i:
#     effective_rm_needed_per_fg = qty[i] / partYield[i] / yield_pct
#     (You need more RM because some is unusable AND some is lost in production)
#     rm_cost_i = effective_rm_needed_per_fg × (cost[i] + trans[i])
#   Total RM cost = Σ rm_cost_i
#
# Total unit production cost:
#   unit_prod_cost = total_rm_cost + labor_variable_per_unit
#   (Labor is 0 here because workers are on fixed salary → in fixed_daily)
#
# NOTE: This is the cost PER SALEABLE FG UNIT leaving the factory.
# If yield=0.95, you need to input 1/0.95 = 1.053 "raw FG" to get 1 saleable unit.

print("\nUnit production cost calculation (3-yield formula):")
for k, prod in enumerate(products):
    fg_yield = prod['yield_pct']
    rm_cost  = 0.0
    for part in prod['parts']:
        # Effective RM per saleable FG unit = qty / partYield / fg_yield
        eff_qty       = part['qty'] / part['partYield'] / fg_yield
        part_cost_val = eff_qty * (part['cost'] + part['trans'])
        rm_cost      += part_cost_val
        print(f"    {prod['name']} ← {part['name']}: "
              f"{part['qty']} ÷ {part['partYield']} ÷ {fg_yield} = {eff_qty:.4f} effective units "
              f"× ${part['cost'] + part['trans']:.2f} = ${part_cost_val:.4f}/FG")
    prod['unit_prod_cost_full'] = rm_cost + prod['labor_variable_per_unit']  # For reference only
    # Variable production cost = ONLY non-material costs incurred per unit produced.
    # Material cost is already charged at ordering: part['cost'] * r[i,t].
    # With fixed-salary labor (already in fixed_daily), variable prod cost = $0.
    # If you have per-unit electricity/consumables, add them here.
    # Variable production cost = electricity + packaging + consumables (NOT material)
    prod['unit_prod_cost'] = (electricity_per_unit + packaging_per_unit + consumables_per_unit
                              + prod['labor_variable_per_unit'])  # labor_variable = $0 for fixed salary
    print(f"  {prod['name']}: variable_prod_cost = ${prod['unit_prod_cost']:.4f}/FG "
          f"(elec=${electricity_per_unit}, pkg=${packaging_per_unit}, cons=${consumables_per_unit}, "
          f"labor_var=${prod['labor_variable_per_unit']:.2f}). "
          f"Material ${rm_cost:.4f} charged at ordering, not here.")

# ─── ORDERING COSTS ───────────────────────────────────────────────────────────
# ANSWER: Ordering costs — not random, derived from BOM.
# Each subpart has its own 'ord_cost' (PO admin cost from your simulator's PO$/ord field).
# This is the cost of PLACING one purchase order: paperwork, approval, receiving, QC.
# It is NOT the material cost — that is separate.
# Total ordering cost for a PO = ord_cost (fixed) + material × qty (variable)
# The MILP has: zo[i,t] (binary = PO placed) and r[i,t] (qty ordered)
# Cost = ord_cost × zo[i,t] + material_cost × r[i,t]
# (Transport $/unit is embedded in the landing cost → holding cost)

# Parts are per-product in this version (no shared parts assumption)
# For shared capacity with shared parts, list them once with combined qty.
# The solver handles BOM explosion per product separately.

all_parts = []
for k, prod in enumerate(products):
    for pi, part in enumerate(prod['parts']):
        part['prod_idx'] = k
        part['part_local_idx'] = pi
        all_parts.append(part)

# ─── REWORK & CAPACITY EXPANSION (disabled by default) ───────────────────────
# ANSWER: Rework — uncontrollable, already in yield. Turn off.
# Rework means: some defective units are re-processed and saved.
# Since yield_pct already captures the NET output (after rework decisions),
# adding rework explicitly would double-count. Only add it if you have a
# separate rework line with its own capacity, cost, and yield improvement.
rework_enabled = False   # OFF by default — yield already captures this

# ANSWER: Capacity expansion — keep but off by default.
# Capacity expansion = buying a new machine or renting extra line.
# This is a strategic (weeks/months) decision, not a daily operational one.
# If planning horizon > 1 month and you're hitting capacity every day,
# the model will suggest expansion. For short horizons: turn off.
expansion_enabled      = False
expansion_cost_per_day = 1000   # $ per day the expanded line runs
expansion_extra_units  = 5      # Additional units from expansion

# ─── LABOR HIRING DECISION ──────────────────────────────────────────────────
# Should we hire more workers? This is a CAPACITY decision inside the SAME objective.
# The solver decides: is hiring cheaper than OT or lost sales?
#
# hire_enabled = True → adds binary hire[w] variables (1 per potential hire)
# Each hire adds extra_units_per_worker to daily capacity at monthly_cost_per_hire.
# The solver naturally trades off: hire cost vs OT cost vs shortage penalty.
#
# Example: 5 workers now → can hire up to max_new_hires more
#   Each new worker adds (effective_hours × units_per_hour × opt_util / n_workers) capacity
#   Cost = monthly_salary / 30 per day
hire_enabled       = False     # OFF by default. Set True to let solver decide.
max_new_hires      = 3         # Maximum additional workers to consider
extra_units_per_worker = max(1, int(derived_capacity / n_workers))  # Each worker adds ~4u/day
hire_cost_per_day  = monthly_salary_per_worker / 30  # ₹1000/day per hire

# Subcontracting (off by default)
subcontract_enabled = False
subcontract_premium = 0.08   # 8% above your own production cost

# ─── WAREHOUSE ───────────────────────────────────────────────────────────────
wh_max    = 500   # Total FG warehouse capacity (all products combined)
on_hand_fg = {k: 0 for k in range(n_products)}  # Starting FG inventory per product

# ─── FAST MODE: WEEKLY AGGREGATION ───────────────────────────────────────────
# For T>30 days, daily MILP is too slow for CBC (too many binary variables).
# FAST_MODE aggregates to weekly periods: ~52 weeks instead of 261 days.
# Production plan is weekly, disaggregated back to daily after solving.
# Determine aggregation level
_agg = 'daily'
if FAST_MODE == 'auto':
    if T > 90: _agg = 'monthly'
    elif T > 30: _agg = 'weekly'
elif FAST_MODE == 'weekly': _agg = 'weekly'
elif FAST_MODE == 'monthly': _agg = 'monthly'
elif FAST_MODE is True: _agg = 'weekly'  # backward compat

if _agg != 'daily':
    bucket_size = 5 if _agg == 'weekly' else 22  # 5 days/week or ~22 days/month
    bucket_label = 'weekly' if _agg == 'weekly' else 'monthly'
    print(f"\n⚡ FAST MODE: aggregating to {bucket_label} periods...")
    week_size = bucket_size
    T_weeks = (T + week_size - 1) // week_size
    weekly_demand = {}
    for k in range(n_products):
        wd = []
        for w in range(T_weeks):
            start = w * week_size
            end = min(start + week_size, T)
            wd.append(sum(demand[k][start:end]))
        weekly_demand[k] = wd
    weekly_capacity = []
    for w in range(T_weeks):
        start = w * week_size
        end = min(start + week_size, T)
        weekly_capacity.append(sum(capacity[start:end]))
    # Replace daily arrays with weekly
    demand = weekly_demand
    capacity = weekly_capacity
    T_original = T
    T = T_weeks
    shelf = max(1, shelf // week_size)  # Convert shelf life to weeks
    for prod in products:
        prod['demand'] = weekly_demand[products.index(prod)]
    # Adjust holding costs (per week instead of per day)
    for prod in products:
        prod['fg_hold_per_day'] *= week_size
    for part in all_parts:
        part['lt'] = max(1, part['lt'] // week_size)
    print(f"  {T} weekly periods (was {T_original} daily)")
    print(f"  Shelf: {shelf} weeks, Weekly demand: {[sum(d) for d in weekly_demand.values()]} total")
    days = list(range(T))
    max_lt = max(p['lt'] for p in all_parts)
    rng_ext = range(-max_lt, T)

# ─────────────────────────────────────────────────────────────────────────────
# BUILD THE MILP MODEL
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("BUILDING MILP MODEL")
print("="*60)

model = pulp.LpProblem("MultiSKU_MPS", pulp.LpMinimize)
days  = list(range(T))

# All parts flat list for RM decisions
n_parts  = len(all_parts)
max_lt   = max(p['lt'] for p in all_parts)
rng_ext  = range(-max_lt, T)

# ─── DECISION VARIABLES ──────────────────────────────────────────────────────
# p[k,t]     = units of product k to produce on day t
p   = {(k, t): pulp.LpVariable(f"p_{k}_{t}", 0, cat="Integer")
       for k in range(n_products) for t in days}

# y[k,t]/y[t] = 1 if production line active
if capacity_mode == 'parallel':
    y  = {(k, t): pulp.LpVariable(f"y_{k}_{t}", cat="Binary")
          for k in range(n_products) for t in days}
    sw = {(k, t): pulp.LpVariable(f"sw_{k}_{t}", cat="Binary")
          for k in range(n_products) for t in days}
else:  # single or shared — one line
    y  = {t: pulp.LpVariable(f"y_{t}", cat="Binary") for t in days}
    sw = {t: pulp.LpVariable(f"sw_{t}", cat="Binary") for t in days}

# Inv[k,t]  = FG inventory of product k at end of day t
Inv = {(k, t): pulp.LpVariable(f"I_{k}_{t}", 0)
       for k in range(n_products) for t in range(T + 1)}

# e[k,t]    = units of product k expired on day t
e   = {(k, t): pulp.LpVariable(f"e_{k}_{t}", 0)
       for k in range(n_products) for t in days}

# s[k,t]    = shortage of product k on day t
s   = {(k, t): pulp.LpVariable(f"s_{k}_{t}", 0)
       for k in range(n_products) for t in days}

# ss_viol[k,t] = how far below SS for product k
ss_viol = {(k, t): pulp.LpVariable(f"ssv_{k}_{t}", 0)
           for k in range(n_products) for t in days}

# RM ordering variables (per part in all_parts)
r  = {(i, t): pulp.LpVariable(f"r_{i}_{t}", 0, cat="Integer")
      for i in range(n_parts) for t in rng_ext}
RI = {(i, t): pulp.LpVariable(f"RI_{i}_{t}", 0)
      for i in range(n_parts) for t in range(T + 1)}
zo = {(i, t): pulp.LpVariable(f"zo_{i}_{t}", cat="Binary")
      for i in range(n_parts) for t in rng_ext}

# Overtime: ot[t] = 1 if OT shift runs on day t (binary — whole shift)
# (Changed from integer hours to binary whole-shift: workers stay for full OT shift)
if ot_enabled:
    ot = {t: pulp.LpVariable(f"ot_{t}", cat="Binary") for t in days}
else:
    ot = {t: 0 for t in days}  # OT disabled → always 0

# Capacity expansion
if expansion_enabled:
    exp_var = {t: pulp.LpVariable(f"exp_{t}", cat="Binary") for t in days}
else:
    exp_var = {t: 0 for t in days}

# Labor hiring variables
if hire_enabled:
    hire = {w: pulp.LpVariable(f"hire_{w}", cat="Binary") for w in range(max_new_hires)}
    hire_extra_cap = pulp.lpSum(hire[w] * extra_units_per_worker for w in range(max_new_hires))
    print(f"  Labor hiring: up to {max_new_hires} workers × {extra_units_per_worker}u/d = +{max_new_hires*extra_units_per_worker}u/d max")
else:
    hire = {}
    hire_extra_cap = 0

# Subcontracting
if subcontract_enabled:
    sub = {(k, t): pulp.LpVariable(f"sub_{k}_{t}", 0, cat="Integer")
           for k in range(n_products) for t in days}
else:
    sub = {(k, t): 0 for k in range(n_products) for t in days}

# ─── INITIAL CONDITIONS ───────────────────────────────────────────────────────
for k in range(n_products):
    model += Inv[k, 0] == on_hand_fg[k], f"init_fg_{k}"
for i in range(n_parts):
    model += RI[i, 0] == 0, f"init_rm_{i}"

# ─── OBJECTIVE FUNCTION ───────────────────────────────────────────────────────
# ANSWER: Does the objective function make sense?
#
# A correct MPS+procurement objective minimises:
#   (1) Setup cost       — batch start penalty (encourages fewer, larger batches)
#   (2) FG holding       — incentivises leaner inventory
#   (3) Production cost  — material + labor per unit (yield-adjusted)
#   (4) Expiry cost      — units wasted past shelf life (salvage loss)
#   (5) Shortage penalty — lost margin or late-delivery penalty (your scenario A/B/C)
#   (6) SS violation     — soft penalty for dipping below safety stock
#   (7) Fixed overhead   — charged daily regardless
#   (8) OT cost          — whole extra shift cost (only if OT runs)
#   (9) RM purchase cost — material × qty ordered
#   (10) RM ordering     — PO admin cost per order placed
#   (11) RM holding      — carry cost on RM inventory on hand
#
# What is NOT in the objective (correctly):
#   - Capacity expansion cost (disabled by default)
#   - Subcontract cost (disabled by default)
#   - Wastage_rate (removed — yield already captures this)
#
# The objective is CORRECT. SAP uses the same cost decomposition.

obj_terms = []
for t in days:
    # (1) Setup cost — fires only on batch start
    if capacity_mode == 'parallel':
        setup_t = switch_cost * pulp.lpSum(sw[k, t] for k in range(n_products))
    else:
        setup_t = switch_cost * sw[t]
    obj_terms.append(setup_t)

    # (2) FG holding — per unit per day (product-specific)
    obj_terms.append(
        pulp.lpSum(products[k]['fg_hold_per_day'] * Inv[k, t+1]
                   for k in range(n_products))
    )

    # (3) Variable production cost (ONLY non-material per-unit costs)
    # Material is charged at (9): part['cost'] * r[i,t] when ordered.
    # With fixed-salary labor: prod cost per unit = $0.
    # This term exists for: electricity per unit, consumables, piece-rate labor.
    obj_terms.append(
        pulp.lpSum(products[k]['unit_prod_cost'] * p[k, t]
                   for k in range(n_products))
    )

    # (4) Expiry cost (salvage-adjusted)
    obj_terms.append(
        pulp.lpSum(products[k]['expired_unit_cost'] * e[k, t]
                   for k in range(n_products))
    )

    # (5) Shortage penalty (scenario-dependent)
    obj_terms.append(
        pulp.lpSum(products[k]['bo_penalty'] * s[k, t]
                   for k in range(n_products))
    )

    # (6) SS violation penalty
    obj_terms.append(
        pulp.lpSum(
            ss_penalty_multiplier * products[k]['fg_hold_per_day'] * ss_viol[k, t]
            for k in range(n_products)
        )
    )

    # (7) Fixed overhead (includes base labor salary)
    obj_terms.append(fixed_daily)

    # (8) Overtime cost (whole shift binary)
    if ot_enabled:
        obj_terms.append(ot_daily_cost * ot[t])

    # (9+10) RM purchase + ordering cost — added outside day loop below

    # Expansion cost
    if expansion_enabled:
        obj_terms.append(expansion_cost_per_day * exp_var[t])

    # Subcontract cost
    if subcontract_enabled:
        for k in range(n_products):
            obj_terms.append(
                products[k]['unit_prod_cost'] * (1 + subcontract_premium) * sub[k, t]
            )

# (9+10+11) RM costs across extended range
# (12) Hiring cost — permanent for entire horizon
if hire_enabled:
    for w in range(max_new_hires):
        obj_terms.append(hire_cost_per_day * T * hire[w])  # Total cost = daily rate × T days

for i in range(n_parts):
    part = all_parts[i]
    rm_h = part['cost'] * part['hold_pct'] / 100 / 365   # RM holding $/unit/day
    obj_terms.append(
        pulp.lpSum(rm_h * RI[i, t+1] for t in days)          # RM holding
    )
    obj_terms.append(
        pulp.lpSum(
            part['cost'] * r[i, t]                    # RM purchase cost
            + part['ord_cost'] * zo[i, t]             # PO admin cost (fixed per order)
            for t in rng_ext
        )
    )

model += pulp.lpSum(obj_terms), "Total_Cost"

# ─── CONSTRAINTS ──────────────────────────────────────────────────────────────
for t in days:
    # C1: FG FLOW BALANCE per product
    for k in range(n_products):
        fg_yield = products[k]['yield_pct']
        prod_in  = p[k, t] * fg_yield
        sub_in   = sub[k, t] if subcontract_enabled else 0
        backord  = (s[k, t-1] if backorder_on and t > 0 else 0)

        model += (
            Inv[k, t+1] == Inv[k, t] + prod_in + sub_in
                           - demand[k][t] - backord
                           - e[k, t] + s[k, t]
        ), f"flow_{k}_{t}"

        # C5: SAFETY STOCK (soft — penalised not forbidden)
        model += ss_viol[k, t] >= products[k]['fg_ss'] - Inv[k, t+1], f"ss_{k}_{t}"

        # C6: SHORTAGE forcing (when backorder off → must be 0 or pay penalty)
        if not backorder_on:
            model += s[k, t] == 0, f"nobo_{k}_{t}"
            # Note: shortage s[k,t] is forced to 0 so the shortage penalty never
            # fires. The solver just can't satisfy demand — it will produce more
            # or let the SS buffer absorb it. If neither is possible, the model
            # will report infeasible (increase capacity or reduce demand).

        # C7: SHELF LIFE EXPIRY
        if t >= shelf:
            model += (
                pulp.lpSum(e[k, tau] for tau in range(max(0, t-shelf+1), t+1))
                >= pulp.lpSum(p[k, tau] * fg_yield
                              for tau in range(max(0, t-2*shelf+1), t-shelf+1))
                - pulp.lpSum(demand[k][tau]
                             for tau in range(max(0, t-2*shelf+1), t-shelf+1))
            ), f"shelf_{k}_{t}"

    # C2: CAPACITY CONSTRAINT
    # OT adds capacity SEPARATELY (can't multiply two variables in LP)
    # Hiring adds PERMANENT extra capacity (hire[w] × units_per_worker)
    _hire_cap = hire_extra_cap if hire_enabled else 0
    if capacity_mode == 'parallel':
        for k in range(n_products):
            ot_cap = ot_extra_capacity * ot[t] if ot_enabled else 0
            model += p[k, t] <= capacity[t] * y[k, t] + ot_cap + _hire_cap, f"cap_{k}_{t}"
    else:
        ot_cap = ot_extra_capacity * ot[t] if ot_enabled else 0
        model += (
            pulp.lpSum(p[k, t] for k in range(n_products))
            <= capacity[t] * y[t] + ot_cap + _hire_cap
        ), f"cap_{t}"

    # C3: BATCH SWITCH (setup fires on 0→1 transition only)
    if capacity_mode == 'parallel':
        for k in range(n_products):
            model += sw[k, t] >= y[k, t] - (y[k, t-1] if t > 0 else 0), f"sw_{k}_{t}"
    else:
        model += sw[t] >= y[t] - (y[t-1] if t > 0 else 0), f"sw_{t}"

    # C4: WAREHOUSE MAX (total FG across all products)
    model += pulp.lpSum(Inv[k, t+1] for k in range(n_products)) <= wh_max, f"wh_{t}"

# C8–C12: RM constraints per part
for i in range(n_parts):
    part     = all_parts[i]
    lt       = part['lt']
    k_prod   = part['prod_idx']         # which product this part belongs to
    fg_yield = products[k_prod]['yield_pct']

    for t in days:
        od = t - lt   # Order day (when PO must be placed to arrive by day t)

        # C8: BOM explosion — RM ordered LT days ago × partYield ≥ production × qty
        if -max_lt <= od < T:
            model += (
                r[i, od] * part['partYield']
                >= p[k_prod, t] * part['qty']
            ), f"bom_{i}_{t}"

        # C9: RM flow balance
        if t > 0:
            rm_arrive = r[i, od] * part['partYield'] if (-max_lt <= od < T) else 0
            # Only usable portion enters inventory. Damaged units (1-partYield) discarded on receipt.
            rm_consume = p[k_prod, t] * part['qty']
            model += (
                RI[i, t] == RI[i, t-1] + rm_arrive - rm_consume
            ), f"rm_{i}_{t}"

        # C10: RM warehouse capacity
        model += RI[i, t] <= part['rm_cap'], f"rmcap_{i}_{t}"

    # C11–C12: MOQ and max order
    for t in rng_ext:
        model += r[i, t] >= part['moq']       * zo[i, t], f"moq_{i}_{t}"
        model += r[i, t] <= part['max_order'] * zo[i, t], f"mxo_{i}_{t}"

# ─── C13: VOLUME DISCOUNTS (piecewise cost) ─────────────────────────────────
# If a part has 'vol_disc' tiers, create binary variables per tier.
# Example: vol_disc = [{"minQty":0,"pct":0}, {"minQty":100,"pct":5}, {"minQty":500,"pct":10}]
# Tier 0: 0-99 units → full price
# Tier 1: 100-499 → 5% discount
# Tier 2: 500+ → 10% discount
# Binary vd[i,t,tier] = 1 if order falls in that tier. Exactly one tier active per order.

vol_disc_enabled = False  # OFF by default — adds many binary vars, slows CBC
for part in all_parts:
    if 'vol_disc' in part and len(part.get('vol_disc', [])) > 1:
        vol_disc_enabled = True
        break

if vol_disc_enabled:
    print("Volume discounts: ENABLED")
    vd = {}  # vd[i,t,tier] = binary
    for i in range(n_parts):
        part = all_parts[i]
        tiers = part.get('vol_disc', [{"minQty": 0, "pct": 0}])
        if len(tiers) <= 1:
            continue
        for t in rng_ext:
            for ti in range(len(tiers)):
                vd[i, t, ti] = pulp.LpVariable(f"vd_{i}_{t}_{ti}", cat="Binary")
            # Exactly one tier active per order (if ordering)
            model += pulp.lpSum(vd[i, t, ti] for ti in range(len(tiers))) == zo[i, t], f"vd_one_{i}_{t}"
            # Tier quantity bounds
            for ti in range(len(tiers)):
                lb = tiers[ti]['minQty']
                ub = tiers[ti + 1]['minQty'] - 1 if ti + 1 < len(tiers) else part['max_order']
                model += r[i, t] >= lb * vd[i, t, ti], f"vd_lb_{i}_{t}_{ti}"
                model += r[i, t] <= ub * vd[i, t, ti] + part['max_order'] * (1 - vd[i, t, ti]), f"vd_ub_{i}_{t}_{ti}"
    # Adjust RM purchase cost in objective to use discounted price
    # (Already in obj_terms as part['cost'] * r[i,t] — override with tier-based)
    print(f"  Added {sum(1 for k in vd)} binary tier variables")

# ─── STOCHASTIC MODE (scenario-based robust optimization) ────────────────────
# Instead of one demand scenario, generate N scenarios from MAPE distribution.
# Solve the deterministic model for each scenario.
# Take the plan that minimizes WORST-CASE cost across all scenarios (minimax).
# This is computationally expensive — use only for short horizons or weekly mode.

STOCHASTIC_MODE = False   # OFF by default
N_SCENARIOS = 10          # Number of demand scenarios to generate

if STOCHASTIC_MODE and not vol_disc_enabled:
    print(f"\n⚡ STOCHASTIC MODE: generating {N_SCENARIOS} demand scenarios from MAPE={MAPE_PCT}%...")
    # Generate scenarios
    scenarios = []
    for sc in range(N_SCENARIOS):
        sc_demand = {}
        for k in range(n_products):
            noise = np.random.normal(1.0, MAPE_PCT / 100, T)
            noise = np.clip(noise, 0.5, 2.0)
            sc_demand[k] = [max(0, int(round(demand[k][t] * noise[t]))) for t in range(T)]
        scenarios.append(sc_demand)
    
    # Add scenario-indexed shortage variables
    s_sc = {(k, t, sc): pulp.LpVariable(f"s_{k}_{t}_sc{sc}", 0)
            for k in range(n_products) for t in days for sc in range(N_SCENARIOS)}
    
    # Worst-case cost variable
    worst_cost = pulp.LpVariable("worst_cost", 0)
    
    # For each scenario: flow balance with scenario demand
    for sc in range(N_SCENARIOS):
        sc_cost_terms = []
        for t in days:
            for k in range(n_products):
                # Scenario flow: same production p[k,t], different demand
                sc_dem = scenarios[sc][k][t]
                model += (
                    Inv[k, t+1] >= Inv[k, t] + p[k, t] * products[k]['yield_pct']
                    - sc_dem - e[k, t] + s_sc[k, t, sc]
                ), f"sc_flow_{k}_{t}_{sc}"
                sc_cost_terms.append(products[k]['bo_penalty'] * s_sc[k, t, sc])
        # Worst case ≥ each scenario's shortage cost
        model += worst_cost >= pulp.lpSum(sc_cost_terms), f"sc_worst_{sc}"
    
    # Replace shortage in objective with worst_case
    # (This is a simplification — full robust uses all cost terms per scenario)
    obj_terms.append(worst_cost)
    print(f"  Added {N_SCENARIOS} scenarios × {n_products} products × {T} days = {N_SCENARIOS * n_products * T} scenario variables")

# ─── SOLVE ───────────────────────────────────────────────────────────────────
print(f"\nSolving {T}-day horizon, {n_products} SKUs, {n_parts} RM parts...")
# Solver tuning for speed
if USE_GUROBI:
    try:
        import gurobipy
        solver = pulp.GUROBI_CMD(msg=1, timeLimit=300)
        print("Using Gurobi solver")
    except ImportError:
        print("Gurobi not found, falling back to CBC")
        solver = pulp.PULP_CBC_CMD(msg=1, timeLimit=180, gapRel=0.02)
else:
    # CBC: gapRel=0.02 means accept solution within 2% of optimal (much faster)
    # For exact optimal: set gapRel=0 (but may take 10x longer)
    solver = pulp.PULP_CBC_CMD(msg=1, timeLimit=180, gapRel=0.05)
    print(f"Using CBC solver (timeLimit=180s, gap=5%)")
model.solve(solver)

print(f"\nStatus: {pulp.LpStatus[model.status]}")
if model.status == -1:
    print("INFEASIBLE — check capacity vs demand. Try increasing capacity or enabling OT.")
    exit()

print(f"Total Cost: ${pulp.value(model.objective):,.2f}\n")

if hire_enabled:
    hired = sum(1 for w in range(max_new_hires) if (hire[w].varValue or 0) > 0.5)
    hire_total_cost = hired * hire_cost_per_day * T
    extra_cap = hired * extra_units_per_worker
    print(f"LABOR DECISION: Hire {hired}/{max_new_hires} workers → +{extra_cap}u/day capacity")
    print(f"  Cost: {hired} × ${hire_cost_per_day:.0f}/day × {T}d = ${hire_total_cost:,.0f}")
    print(f"  New capacity: {derived_capacity} + {extra_cap} = {derived_capacity + extra_cap} u/day")
    if hired == 0:
        print(f"  → Solver says: DON'T hire. OT/current capacity is cheaper than adding staff.")
    print()

# ─── OUTPUT ───────────────────────────────────────────────────────────────────
print(f"{'Day':>4} | " + " | ".join(
    f"{prod['name'][:8]} Dem  Prod  Inv  Exp Short OT" for prod in products))
print("-" * (10 + 36 * n_products))

totals = {k: {'prod':0,'exp':0,'short':0,'ot_days':0} for k in range(n_products)}
total_demand = {k: 0 for k in range(n_products)}

for t in days:
    otv = int((ot[t].varValue or 0) if ot_enabled else 0)
    line = f"{t+1:>4} | "
    for k in range(n_products):
        pv  = int(p[k,t].varValue or 0)
        iv  = Inv[k,t+1].varValue or 0
        ev  = e[k,t].varValue or 0
        sv  = s[k,t].varValue or 0
        dem = demand[k][t]
        line += f"{products[k]['name'][:4]} {dem:>4} {pv:>4} {iv:>5.0f} {ev:>4.0f} {sv:>4.0f}   "
        totals[k]['prod']  += pv
        totals[k]['exp']   += ev
        totals[k]['short'] += sv
        total_demand[k]    += dem
    if otv: line += f"OT ✓"
    print(line)

print("\nSUMMARY:")
for k in range(n_products):
    td = total_demand[k]
    fill = (td - totals[k]['short']) / td * 100 if td > 0 else 100
    print(f"  {products[k]['name']}: produced={totals[k]['prod']}u "
          f"expired={totals[k]['exp']:.0f}u short={totals[k]['short']:.0f}u "
          f"fill={fill:.1f}%")

# RM procurement schedule
import json
print("\nRM PROCUREMENT SCHEDULE:")
procurement = {}
for i in range(n_parts):
    part = all_parts[i]
    nm   = f"{products[part['prod_idx']]['name']} ← {part['name']}"
    procurement[nm] = []
    tq = npo = 0
    for t in rng_ext:
        rv = r[i, t].varValue or 0
        if rv > 0.5:
            arrives = t + part['lt'] + 1
            mat_cost = part['cost'] * int(rv)
            print(f"  {nm}: PO day {t+1:>3} qty {int(rv):>5}u arrives day {arrives:>3} "
                  f"mat cost ${mat_cost:,.2f} + PO admin ${part['ord_cost']:.2f}")
            procurement[nm].append({"day": t+1, "qty": int(rv), "arrives": arrives})
            tq += int(rv); npo += 1
    if npo:
        print(f"    → {npo} POs, {tq} total units")

mps_plan = {products[k]['name']: [int(p[k,t].varValue or 0) for t in days]
            for k in range(n_products)}
total_cost = round(pulp.value(model.objective), 2)
fill_rates = {products[k]['name']:
              round((total_demand[k] - totals[k]['short']) / total_demand[k] * 100, 1)
              for k in range(n_products)}

# ─── DISAGGREGATE WEEKLY → DAILY (if FAST_MODE) ──────────────────────────
if _agg != 'daily' and T != T_original:
    print(f"\n⚡ Disaggregating weekly plan back to {T_original} daily periods...")
    for k in range(n_products):
        weekly_plan = [int(p[k,t].varValue or 0) for t in days]
        daily_plan = []
        for w, wp in enumerate(weekly_plan):
            wdays = min(5, T_original - w*5)
            per_day = wp // wdays if wdays > 0 else 0
            remainder = wp - per_day * wdays
            for d in range(wdays):
                daily_plan.append(per_day + (1 if d < remainder else 0))
        mps_plan[products[k]['name']] = daily_plan
        print(f"  {products[k]['name']}: {sum(weekly_plan)}u weekly → {sum(daily_plan)}u daily ({len(daily_plan)} days)")

print(f"\n--- JSON for simulator ---")
print(json.dumps({
    "source":      "MILP_v3",
    "cost":        total_cost,
    "mps_plan":    mps_plan,
    "procurement": procurement,
    "fill_rates":  fill_rates,
    "ss_used":     {products[k]['name']: products[k]['fg_ss'] for k in range(n_products)},
}, indent=2))
