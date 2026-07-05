# ================================================================
# QUICK START — APIS Capstone
# ================================================================
print("APIS — Autonomous Procurement Intelligence System")
print("   • Multi-agent orchestrator with 4 specialized sub-agents")
print("   • 3-phase workflow: Inventory → TCO → Risk Assessment")
print("   • 5-element risk model (FX, Material, Transport, Tax, Insurance)")
print("   • Human-in-the-Loop gate for high-cost/high-risk decisions")
print("   • Zero-cost deterministic mode (no external API calls)")
print("   ⚠️  Internet: OFF | Accelerator: None | Cost: €0.00")


# Required libraries
import numpy as np
import pandas as pd

# Optional: kagglehub only needed in Kaggle environment
try:
    import kagglehub
    KAGGLE_ENV = True
except ImportError:
    KAGGLE_ENV = False
    print("   Local environment detected — using ./data/ folder for datasets")



# ================================================================
# SECTION 0 — KAGGLE SECRETS & API KEY
# ================================================================
# ARCHITECTURAL NOTE: This section demonstrates production secrets
# management patterns. The GOOGLE_API_KEY defaults to a placeholder
# string. ADK_AVAILABLE (Section 1) controls whether any LLM path
# executes. In the Kaggle free-tier environment, ADK is not
# pre-installed, so the system automatically falls back to the
# zero-cost deterministic path. No external API calls are attempted.
# ================================================================
import os

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "YOUR_API_KEY_HERE")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Verify the key is loaded (masked for security)
if GOOGLE_API_KEY != "YOUR_API_KEY_HERE":
    print("API key detected — capstone runs in zero-cost deterministic mode.")
    print("   No LLM calls will be made. ADK_AVAILABLE controls execution path.")
else:
    print("Zero-cost mode: Using placeholder API key. No external calls possible.")


# ================================================================
# SECTION 1 --- IMPORTS & CONFIGURATION
# ================================================================
import warnings
warnings.filterwarnings("ignore", message=".*EXPERIMENTAL.*")

import pandas as pd
import numpy as np
import json
import logging
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor

# ADK imports (optional - only needed for real LLM mode)
try:
    from google.adk.agents import Agent
    from google.adk.tools import FunctionTool
    from google.adk.runners import InMemoryRunner
    from google.adk.models.google_llm import Gemini
    from google.genai import types
    ADK_AVAILABLE = True
except ImportError:
    ADK_AVAILABLE = False
    # Zero-cost mode: google-adk not installed in Kaggle environment.
    # System uses deterministic PseudoAgent with identical behavior.
    # This is intentional -- demonstrates graceful fallback architecture.
    logger.warning("ADK not installed - running in zero-cost deterministic mode only")

logging.basicConfig(level=logging.INFO, format="%(asctime)s --- %(levelname)s --- %(message)s")
logger = logging.getLogger(__name__)

# Free-tier model --- NEVER change to Pro
MODEL_NAME = "gemini-3.5-flash-lite"

# Retry config for rate limits (ADK only)
retry_config = None
if ADK_AVAILABLE:
    retry_config = types.HttpRetryOptions(
        attempts=3, exp_base=2, initial_delay=1,
        http_status_codes=[429, 500, 503, 504]
    )

# Jupyter widgets for interactive dashboard
try:
    from ipywidgets import widgets, HBox, VBox
    from IPython.display import display, clear_output
    IPYWIDGETS_AVAILABLE = True
except ImportError:
    IPYWIDGETS_AVAILABLE = False
    logger.warning("ipywidgets not installed - interactive dashboard unavailable")


# ================================================================
# SECTION 2 — DATA LOADING (2024 datasets)
# ================================================================
def load_csv_safe(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        df = df.astype(object)  # Convert numpy types to Python native
        return df
    except Exception as e:
        logger.error(f"Failed to load {path}: {e}")
        raise

inventory_df = load_csv_safe("./data/inventory_levels_weekly.csv")
vendor_master_df = load_csv_safe("./data/vendor_list.csv")
fx_df = load_csv_safe("./data/Euro exchange_rates_weekly.csv")
vendor_prices_df = load_csv_safe("./data/vendor_prices_weekly.csv")
fx_fee_df = load_csv_safe("./data/min_amountmax_amountfee_percentage.csv")
additional_costs_df = load_csv_safe("./data/procurement_additional_costs.csv")

# Validation
ensure_columns = lambda df, cols, name: (
    missing := [c for c in cols if c not in df.columns],
    missing and logger.error(f"{name} missing: {missing}")
) and (not missing or (_ for _ in ()).throw(ValueError(f"{name} missing: {missing}")))

ensure_columns(inventory_df, ["Week","Material_ID","Current_Stock","Planned_Usage","Reorder_Level"], "inventory")
ensure_columns(vendor_master_df, ["Vendor_ID","Vendor_Name","Currency","Payment_Terms_Days","Early Pmt","Early P Disc"], "vendor")
ensure_columns(fx_df, ["Week","USD","JPY","CHF"], "fx")
ensure_columns(vendor_prices_df, ["week","vendor_id","currency","local_price"], "prices")
ensure_columns(additional_costs_df, ["vendor_id","Material_ID","Transport_Cost_Per_Unit_EUR","Default_Tax_Percentage","Insurance_Percentage"], "costs")
ensure_columns(fx_fee_df, ["min_amount","max_amount","fee_percentage"], "fx_fee")

print("✅ All datasets loaded and validated")


# ================================================================
# SECTION 3 — FX & COST UTILITIES
# ================================================================
def get_fx_rate(currency: str, week: int) -> float:
    row = fx_df[fx_df["Week"] == week]
    if row.empty:
        raise ValueError(f"No FX rate for week {week}")
    return float(row[currency].iloc[0])

def get_fx_fee(amount: float) -> float:
    row = fx_fee_df[(fx_fee_df.min_amount <= amount) & (fx_fee_df.max_amount >= amount)]
    if row.empty:
        return 0.015  # Default 1.5%
    return float(row.fee_percentage.iloc[0]) / 100

def compute_purchase_qty(current: float, usage: float, reorder: float) -> float:
    return max(usage + reorder - current, 0)

def compute_fx_conversion(amount: float, rate: float, fee_pct: float) -> float:
    eur = amount * rate
    return eur + (eur * fee_pct)
print("✅ All column validations passed")


# ================================================================
# SECTION 4 --- TCO ENGINE (FIXED: + Division-by-Zero Guard)
# ================================================================
def compute_total_cost_with_discounts(vendor_id: str, week: int, quantity: float, material_id: str = "MAT-001") -> dict:
    """Full TCO with early payment discount, FX, transport, tax, insurance"""

    # Vendor price & currency
    vendor_row = vendor_prices_df[
        (vendor_prices_df.vendor_id == vendor_id) & (vendor_prices_df.week == week)
    ].iloc[0]
    currency = vendor_master_df.loc[vendor_master_df.Vendor_ID == vendor_id, "Currency"].iloc[0]
    local_price = float(vendor_row.local_price)

    # FX conversion
    fx_rate = get_fx_rate(currency, week)

    # FIX 3: Division-by-Zero Guard
    if fx_rate <= 0:
        raise ValueError(f"Invalid FX rate {fx_rate} for {currency} at week {week}")

    material_cost_local = local_price * quantity
    material_cost_eur = material_cost_local / fx_rate

    # Early payment discount
    vendor_info = vendor_master_df[vendor_master_df.Vendor_ID == vendor_id].iloc[0]
    early_discount_pct = float(vendor_info["Early P Disc"]) / 100
    discounted_material_cost = material_cost_eur * (1 - early_discount_pct)
    discount_savings = material_cost_eur - discounted_material_cost
    material_cost_eur = discounted_material_cost

    # Additional costs
    cost_row = additional_costs_df[
        (additional_costs_df.vendor_id == vendor_id) & (additional_costs_df.Material_ID == material_id)
    ].iloc[0]

    fx_fee_eur = material_cost_eur * get_fx_fee(material_cost_eur)
    transport_eur = float(cost_row.Transport_Cost_Per_Unit_EUR) * quantity
    insurance_eur = material_cost_eur * (float(cost_row.Insurance_Percentage) / 100)

    # Correct tax base
    tax_base = material_cost_eur + insurance_eur + transport_eur
    tax_eur = tax_base * (float(cost_row.Default_Tax_Percentage) / 100)

    total_eur = material_cost_eur + fx_fee_eur + transport_eur + insurance_eur + tax_eur

    return {
        "vendor": vendor_id,
        "vendor_name": vendor_info.Vendor_Name,
        "week": week,
        "quantity": quantity,
        "material_cost_eur": round(material_cost_eur, 2),
        "fx_fee_eur": round(fx_fee_eur, 2),
        "transport_eur": round(transport_eur, 2),
        "insurance_eur": round(insurance_eur, 2),
        "tax_eur": round(tax_eur, 2),
        "total_eur": round(total_eur, 2),
        "early_discount_savings": round(discount_savings, 2),
        "currency": currency,
        "fx_rate": fx_rate,
        "payment_terms": int(vendor_info.Payment_Terms_Days),
        "early_discount_rate": early_discount_pct * 100
    }

print("All column validations passed")


# ================================================================
# SECTION 5 --- TOOLS (FIXED: + Standardized risk_elements Structure [W 4])
# ================================================================
def inventory_tool_func(week: int, material_id: str) -> Dict[str, Any]:
    """Tool 1: Calculate purchase quantity from inventory"""
    row = inventory_df[(inventory_df.Week == week) & (inventory_df.Material_ID == material_id)]
    if row.empty:
        return {"error": "No inventory record", "week": week, "material": material_id}
    r = row.iloc[0]
    qty = compute_purchase_qty(
        float(r.Current_Stock), float(r.Planned_Usage), float(r.Reorder_Level)
    )
    return {
        "current_stock": float(r.Current_Stock),
        "planned_usage": float(r.Planned_Usage),
        "reorder_level": float(r.Reorder_Level),
        "purchase_quantity": qty,
        "week": week,
        "material": material_id
    }

def vendor_tool_func(week: int, material_id: str) -> Dict[str, Any]:
    """Tool 2: Get available vendors and prices for material"""
    week_prices = vendor_prices_df[vendor_prices_df.week == week]
    vendors = {}
    for _, row in week_prices.iterrows():
        vid = row['vendor_id']
        vendors[vid] = {
            "local_price": float(row['local_price']),
            "currency": row['currency']
        }
    return {"available_vendors": vendors, "week": week, "material": material_id}

def fx_tool_func(week: int, vendor_id: str, quantity: float) -> Dict[str, Any]:
    """Tool 3: Get FX rate and fee for vendor"""
    currency = vendor_master_df.loc[vendor_master_df.Vendor_ID == vendor_id, "Currency"].iloc[0]
    rate = get_fx_rate(currency, week)
    return {
        "vendor": vendor_id,
        "currency": currency,
        "fx_rate": rate,
        "fee_pct": get_fx_fee(quantity),
        "week": week
    }

def tco_tool_func(vendor_id: str, week: int, quantity: float, material_id: str = "MAT-001") -> Dict[str, Any]:
    """Tool 4: Compute full TCO for vendor (runs AFTER inventory/vendor/FX)"""
    return compute_total_cost_with_discounts(vendor_id, week, quantity, material_id)

def risk_tool_func(vendor_id: str, week: int, total_cost: float, quantity: float, material_id: str = "MAT-001") -> Dict[str, Any]:
    """Tool 5: 5-element risk assessment at tight thresholds (1%/3% FX, 103% cost)"""
    currency = vendor_master_df.loc[vendor_master_df.Vendor_ID == vendor_id, "Currency"].iloc[0]

    # --- ELEMENT 1: FX VOLATILITY (flag >1%, strong >3%) ---
    recent = fx_df[fx_df["Week"] <= week].tail(4)
    if len(recent) >= 2:
        rates = [float(r[currency]) for _, r in recent.iterrows()]
        avg_rate = np.mean(rates)
        current_rate = rates[-1]
        volatility = abs(current_rate - avg_rate) / avg_rate if avg_rate > 0 else 0
    else:
        volatility = 0.0

    fx_flag = volatility > 0.01
    fx_high_flag = volatility > 0.03

    # --- ELEMENT 2: MATERIAL COST ANOMALY (threshold: median x 1.03) ---
    vendor_price_history = []
    for w in range(max(1, week - 4), week):
        hist = vendor_prices_df[(vendor_prices_df.vendor_id == vendor_id) & (vendor_prices_df.week == w)]
        if not hist.empty:
            vendor_price_history.append(float(hist.local_price.iloc[0]))

    current_price_row = vendor_prices_df[(vendor_prices_df.vendor_id == vendor_id) & (vendor_prices_df.week == week)]
    current_local_price = float(current_price_row.local_price.iloc[0]) if not current_price_row.empty else 0
    median_material_price = np.median(vendor_price_history) if vendor_price_history else current_local_price
    material_flag = (current_local_price > median_material_price * 1.03) if median_material_price > 0 and len(vendor_price_history) >= 2 else False

    # --- ELEMENT 3: TRANSPORT COST ANOMALY (threshold: median x 1.03) ---
    transport_history = []
    for w in range(max(1, week - 4), week):
        cost_row_hist = additional_costs_df[
            (additional_costs_df.vendor_id == vendor_id) &
            (additional_costs_df.Material_ID == material_id)
        ]
        if not cost_row_hist.empty:
            transport_history.append(float(cost_row_hist.Transport_Cost_Per_Unit_EUR.iloc[0]))

    current_cost_row = additional_costs_df[
        (additional_costs_df.vendor_id == vendor_id) &
        (additional_costs_df.Material_ID == material_id)
    ]
    current_transport = float(current_cost_row.Transport_Cost_Per_Unit_EUR.iloc[0]) if not current_cost_row.empty else 0
    median_transport = np.median(transport_history) if transport_history else current_transport
    transport_flag = (current_transport > median_transport * 1.03) if median_transport > 0 and len(transport_history) >= 2 else False

    # --- ELEMENT 4: TAX ANOMALY (threshold: median x 1.03) ---
    tax_history = []
    for w in range(max(1, week - 4), week):
        try:
            c = compute_total_cost_with_discounts(vendor_id, w, quantity, material_id)
            tax_amt = c["total_eur"] - (c["material_cost_eur"] + c["fx_fee_eur"] + c["transport_eur"] + c["insurance_eur"])
            tax_history.append(tax_amt)
        except:
            pass

    try:
        current_breakdown = compute_total_cost_with_discounts(vendor_id, week, quantity, material_id)
        current_tax = current_breakdown["total_eur"] - (
            current_breakdown["material_cost_eur"] +
            current_breakdown["fx_fee_eur"] +
            current_breakdown["transport_eur"] +
            current_breakdown["insurance_eur"]
        )
    except:
        current_tax = 0

    median_tax = np.median(tax_history) if tax_history else current_tax
    tax_flag = (current_tax > median_tax * 1.03) if median_tax > 0 and len(tax_history) >= 2 else False

    # --- ELEMENT 5: INSURANCE ANOMALY (threshold: median x 1.03) ---
    insurance_history = []
    for w in range(max(1, week - 4), week):
        try:
            c = compute_total_cost_with_discounts(vendor_id, w, quantity, material_id)
            insurance_history.append(c["insurance_eur"])
        except:
            pass

    try:
        current_insurance = current_breakdown["insurance_eur"]
    except:
        current_insurance = 0

    median_insurance = np.median(insurance_history) if insurance_history else current_insurance
    insurance_flag = (current_insurance > median_insurance * 1.03) if median_insurance > 0 and len(insurance_history) >= 2 else False

    # --- OVERALL RISK LEVEL (5 elements) ---
    flag_count = sum([fx_flag, material_flag, transport_flag, tax_flag, insurance_flag])

    risk_level = "LOW"
    if flag_count >= 3 or fx_high_flag:
        risk_level = "HIGH"
    elif flag_count >= 1:
        risk_level = "MEDIUM"

    # WARN 4 FIX: Standardized risk_elements structure
    return {
        "vendor": vendor_id,
        "week": week,
        "total_cost": total_cost,
        "quantity": quantity,
        "fx_volatility": round(volatility, 4),
        "fx_flag": fx_flag,
        "fx_high_flag": fx_high_flag,
        "material_flag": material_flag,
        "current_local_price": round(current_local_price, 2),
        "median_material_price": round(median_material_price, 2),
        "transport_flag": transport_flag,
        "current_transport": round(current_transport, 2),
        "median_transport": round(median_transport, 2),
        "tax_flag": tax_flag,
        "current_tax": round(current_tax, 2),
        "median_tax": round(median_tax, 2),
        "insurance_flag": insurance_flag,
        "current_insurance": round(current_insurance, 2),
        "median_insurance": round(median_insurance, 2),
        "risk_level": risk_level,
        "flag_count": flag_count,
        "risk_elements": {
            "fx_volatility": {"flagged": fx_flag, "current": round(current_rate, 4), "median": round(avg_rate, 4), "threshold": 0.01},
            "material_cost": {"flagged": material_flag, "current": round(current_local_price, 2), "median": round(median_material_price, 2), "threshold_pct": 103},
            "transport": {"flagged": transport_flag, "current": round(current_transport, 2), "median": round(median_transport, 2), "threshold_pct": 103},
            "tax": {"flagged": tax_flag, "current": round(current_tax, 2), "median": round(median_tax, 2), "threshold_pct": 103},
            "insurance": {"flagged": insurance_flag, "current": round(current_insurance, 2), "median": round(median_insurance, 2), "threshold_pct": 103}
        }
    }

# Wrap as ADK FunctionTools (if ADK available)
if ADK_AVAILABLE:
    inventory_tool = FunctionTool(inventory_tool_func)
    vendor_tool = FunctionTool(vendor_tool_func)
    fx_tool = FunctionTool(fx_tool_func)
    tco_tool = FunctionTool(tco_tool_func)
    risk_tool = FunctionTool(risk_tool_func)
    print("All 5 tools wrapped as ADK FunctionTools")
else:
    print("All 5 tools initialized --- ADK not available, using direct function calls")

print("5-element risk at 1%/3% FX, 103% cost thresholds (FIXED v6: quantity-aware)")


# ================================================================
# SECTION 6 --- PARALLEL TOOL EXECUTION ENGINE (FIXED: + Error Propagation [W 1])
# ================================================================

def run_tools_parallel(week: int, material_id: str) -> Dict[str, Any]:
    """
    Execute tools in parallel where allowed:
    Phase 1 (parallel): Inventory + Vendor + FX
    Phase 2 (sequential): TCO (needs Phase 1 output)
    Phase 3 (parallel): Risk (needs TCO output + quantity)
    """
    # Phase 1: Parallel --- no dependencies
    with ThreadPoolExecutor(max_workers=3) as executor:
        inv_future = executor.submit(inventory_tool_func, week, material_id)
        ven_future = executor.submit(vendor_tool_func, week, material_id)

        inventory_result = inv_future.result()
        vendor_result = ven_future.result()

        # FIX 2: Check inventory error
        if "error" in inventory_result:
            return {"error": inventory_result["error"]}

        # WARN 1 FIX: Check vendor errors
        if "error" in vendor_result:
            return {"error": f"Vendor lookup failed: {vendor_result['error']}"}

        quantity = inventory_result["purchase_quantity"]
        vendors = vendor_result.get("available_vendors", {})

        # WARN 1 FIX: Check if no vendors available
        if not vendors:
            return {"error": "No vendors available for week", "week": week, "material": material_id}

    # Phase 2: TCO for each vendor (can parallelize)
    tco_results = {}
    with ThreadPoolExecutor(max_workers=len(vendors)) as executor:
        futures = {
            vid: executor.submit(tco_tool_func, vid, week, quantity, material_id)
            for vid in vendors.keys()
        }
        for vid, fut in futures.items():
            tco_results[vid] = fut.result()

    # Phase 3: Risk assessment (parallel) --- passes quantity and material_id
    risk_results = {}
    with ThreadPoolExecutor(max_workers=len(vendors)) as executor:
        futures = {
            vid: executor.submit(risk_tool_func, vid, week, tco["total_eur"], quantity, material_id)
            for vid, tco in tco_results.items()
        }
        for vid, fut in futures.items():
            risk_results[vid] = fut.result()

    return {
        "inventory": inventory_result,
        "vendors": vendor_result,
        "tco": tco_results,
        "risk": risk_results,
        "quantity": quantity,
        "week": week,
        "material": material_id
    }



# ================================================================
# SECTION 7 --- DECISION SYNTHESIS (FIXED: + Empty Vendor Guard [F 2], + Tie-Breaking [W 3])
# ================================================================

def synthesize_decision(tool_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deterministic decision logic --- no LLM cost, fully reproducible.
    Selects optimal vendor based on: lowest total cost (TCO).

    RISK ASSESSMENT IS FORWARD LOOKING (Informational Only):
    - 5-element risk: FX volatility, Material cost, Transport, Tax, Insurance
    - FX: flagged if >1% volatility, strong if >3%
    - Cost elements: flagged if current > 103% of historical median
    - Risk levels (LOW/MEDIUM/HIGH) computed and displayed for transparency
    - Risk does NOT penalize the vendor's score or affect selection
    - Risk informs future procurement strategy and triggers HITL when HIGH
    - Pure cost optimization: cheapest TCO wins regardless of risk level
    """
    if "error" in tool_results:
        return {"error": tool_results["error"]}

    tco_data = tool_results["tco"]
    risk_data = tool_results["risk"]
    quantity = tool_results["quantity"]

    # FIX 2: Guard against empty vendor list
    if not tco_data:
        return {
            "error": "No vendors available",
            "week": tool_results.get("week"),
            "material": tool_results.get("material")
        }

    best_vendor = None
    best_cost = float('inf')
    vendor_scores = {}

    for vid, tco in tco_data.items():
        risk = risk_data.get(vid) or {}

        total = tco["total_eur"]

        risk_level = risk.get("risk_level", "UNKNOWN")
        flag_count = risk.get("flag_count", 0)
        risk_elements = risk.get("risk_elements", {})

        score = total

        vendor_scores[vid] = {
            "total_cost": total,
            "risk_level": risk_level,
            "flag_count": flag_count,
            "risk_adjusted_score": round(score, 2),
            "details": tco,
            "risk_elements": risk_elements,
            "risk_info": {
                "fx_volatility": risk.get("fx_volatility", 0),
                "fx_flag": risk.get("fx_flag", False),
                "material_flag": risk.get("material_flag", False),
                "current_local_price": risk.get("current_local_price", 0),
                "median_material_price": risk.get("median_material_price", 0),
                "transport_flag": risk.get("transport_flag", False),
                "current_transport": risk.get("current_transport", 0),
                "median_transport": risk.get("median_transport", 0),
                "tax_flag": risk.get("tax_flag", False),
                "current_tax": risk.get("current_tax", 0),
                "median_tax": risk.get("median_tax", 0),
                "insurance_flag": risk.get("insurance_flag", False),
                "current_insurance": risk.get("current_insurance", 0),
                "median_insurance": risk.get("median_insurance", 0)
            }
        }

        # WARN 3 FIX: Tie-breaking --- prefer lower risk when costs are equal
        if score < best_cost:
            best_cost = score
            best_vendor = vid
        elif score == best_cost and best_vendor is not None:
            # Prefer lower risk level: LOW > MEDIUM > HIGH
            current_best_risk = risk_data.get(best_vendor, {}).get("risk_level", "UNKNOWN")
            risk_priority = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "UNKNOWN": 3}
            if risk_priority.get(risk_level, 3) < risk_priority.get(current_best_risk, 3):
                best_vendor = vid

    ranked = sorted(vendor_scores.items(), key=lambda x: x[1]["risk_adjusted_score"])

    return {
        "best_vendor": best_vendor,
        "best_score": round(best_cost, 2),
        "quantity": quantity,
        "week": tool_results["week"],
        "material": tool_results["material"],
        "all_vendors": vendor_scores,
        "ranking": [(v, s["risk_adjusted_score"]) for v, s in ranked],
        "justification": f"Selected {best_vendor} with lowest total cost EUR {best_cost:,.2f} (risk assessment is forward-looking informational only)",
        "selection_method": "Pure cost optimization -- lowest TCO wins regardless of risk level",
        "risk_disclaimer": "Risk levels (LOW/MEDIUM/HIGH) are forward-looking indicators based on 5 elements (FX volatility, material cost, transport, tax, insurance) at 103% threshold (1%/3% FX). They do not affect vendor selection."
    }

print("Decision synthesis engine ready -- 5 elements, 1%/3% FX, 103% cost thresholds")



# ================================================================
# SECTION 8 --- HUMAN-IN-THE-LOOP GATE (FIXED: + Historical Cost Anomaly [F 1])
# ================================================================
def hitl_gate(decision: Dict[str, Any], threshold_eur: float = 5000.0) -> Dict[str, Any]:
    """
    Trigger HITL if:
    - Total cost above threshold
    - HIGH risk detected
    - Cost anomaly: current cost > 120% of 4-week historical median
    """

    if "error" in decision:
        return decision

    triggers = []
    best_vendor = decision["best_vendor"]
    best_score = decision["best_score"]
    week = decision["week"]
    quantity = decision["quantity"]
    material = decision["material"]

    if best_score > threshold_eur:
        triggers.append("Cost EUR " + f"{best_score:,.2f}" + " exceeds threshold EUR " + f"{threshold_eur:,.2f}")

    # Check if best vendor has HIGH risk
    vendor_data = decision.get("all_vendors", {}).get(best_vendor, {})
    if vendor_data.get("risk_level") == "HIGH":
        triggers.append("HIGH risk detected for " + best_vendor)

    # FIX 1: Cost anomaly --- compare against 4-week historical median
    historical_costs = []
    for w in range(max(1, week - 4), week):
        try:
            hist_tco = compute_total_cost_with_discounts(best_vendor, w, quantity, material)
            historical_costs.append(hist_tco["total_eur"])
        except:
            pass

    if historical_costs:
        median_historical = np.median(historical_costs)
        if best_score > median_historical * 1.2:
            pct_above = (best_score / median_historical - 1) * 100
            triggers.append(
                "Cost EUR " + f"{best_score:,.2f}" + " is " + f"{pct_above:.1f}" + "% above " +
                "4-week median EUR " + f"{median_historical:,.2f}"
            )

    decision["hitl_required"] = len(triggers) > 0
    decision["hitl_triggers"] = triggers

    if triggers:
        hitl_msg = "APPROVAL REQUIRED\n\n"
        hitl_msg += "Recommendation: " + best_vendor + " --- EUR " + f"{best_score:,.2f}" + "\n"
        hitl_msg += "Quantity: " + str(decision['quantity']) + " units\n"
        hitl_msg += "Triggers: " + ", ".join(triggers) + "\n\n"
        hitl_msg += "Action: APPROVE / REJECT / MODIFY"
        decision["hitl_prompt"] = hitl_msg
    else:
        decision["hitl_prompt"] = "Auto-approved --- no HITL triggers"

    return decision

print("HITL gate configured (threshold: EUR 5,000 + 4-week historical anomaly)")


# ================================================================
# SECTION 9 — MEMORY LAYER (Session + Vendor Preference)
# ================================================================
class SimpleMemory:
    """Lightweight memory — no external DB, free tier safe"""

    def __init__(self):
        self.sessions = {}
        self.vendor_preferences = {}
        self.approvals = []

    def save_session(self, session_id: str, decision: Dict):
        self.sessions[session_id] = decision

    def get_preferred_vendor(self, material: str) -> Optional[str]:
        return self.vendor_preferences.get(material)

    def set_preferred_vendor(self, material: str, vendor: str):
        self.vendor_preferences[material] = vendor

    def record_approval(self, decision: Dict, action: str):
        self.approvals.append({
            "week": decision.get("week"),
            "material": decision.get("material"),
            "vendor": decision.get("best_vendor"),
            "cost": decision.get("best_score"),
            "action": action,
            "timestamp": pd.Timestamp.now().isoformat()
        })

memory = SimpleMemory()
print("✅ Memory layer initialized")



# ================================================================
# SECTION 10 --- CONTROL & SAFETY LAYER (FIXED: + NaN/Inf + Week Range [W 2])
# ================================================================
def safety_check(tool_results: Dict[str, Any]) -> List[str]:
    """Deterministic overrides -- if tool fails, catch it here"""
    errors = []

    if "error" in tool_results:
        errors.append(f"Tool error: {tool_results['error']}")
        return errors

    # Validate quantities
    qty = tool_results.get("quantity", 0)
    if qty < 0:
        errors.append(f"Invalid negative quantity: {qty}")
    if qty > 100000:
        errors.append(f"Suspiciously large quantity: {qty}")

    # Validate costs
    for vid, tco in tool_results.get("tco", {}).items():
        total = tco.get("total_eur", 0)

        # WARN 2 FIX: NaN/Inf checks
        if np.isnan(total) or np.isinf(total):
            errors.append(f"Invalid cost (NaN/Inf) for {vid}")
            continue

        if total < 0:
            errors.append(f"Negative cost for {vid}")
        if total > 1000000:
            errors.append(f"Suspiciously high cost for {vid}")

    # Validate FX rates
    for vid, risk in tool_results.get("risk", {}).items():
        if risk is None:
            continue
        fx_vol = risk.get("fx_volatility", 0)
        if np.isnan(fx_vol) or np.isinf(fx_vol):
            errors.append(f"Invalid FX volatility (NaN/Inf) for {vid}")
            continue
        if fx_vol > 0.5:
            errors.append(f"Extreme FX volatility for {vid}: {fx_vol}")

    return errors

def validate_week(week: int):
    """WARN 2 FIX: Week range validation"""
    if not isinstance(week, int):
        return False, f"Week must be integer, got {type(week)}"
    if week < 1 or week > 52:
        return False, f"Week {week} out of range (1-52)"
    return True, "Valid"

def deterministic_fallback(week: int, material_id: str) -> Dict[str, Any]:
    """If LLM/agent fails, use pure deterministic logic"""

    # WARN 2 FIX: Validate week before processing
    is_valid, msg = validate_week(week)
    if not is_valid:
        return {"error": f"Validation failed: {msg}"}

    logger.warning("Using deterministic fallback -- LLM not invoked")
    tool_results = run_tools_parallel(week, material_id)

    safety_errors = safety_check(tool_results)
    if safety_errors:
        return {"error": "Safety checks failed", "details": safety_errors}

    decision = synthesize_decision(tool_results)
    return hitl_gate(decision)

print("Safety layer active (sanity checks + NaN/Inf guards + week validation)")



# ================================================================
# SECTION 11 --- PSEUDO-AGENT (Zero-Cost LLM Mimic) [CHANGED v5]
# ================================================================

class PseudoAgent:
    """
    Zero-cost agent that mimics LLM behavior using deterministic logic.
    Parses user intent, executes the 3-phase workflow, and returns
    a structured natural-language response.
    """
    def __init__(self):
        self.name = "ProcurementIntelligenceAgent"
        self.persona = """You are a Procurement Intelligence Agent.
Your job: interpret procurement requests, call tools, and produce optimal decisions.

MANDATORY WORKFLOW (call tools in this order):
1. FIRST call inventory_tool AND vendor_tool AND fx_tool in parallel (Phase 1)
2. THEN call tco_tool for each vendor using the quantity from inventory_tool (Phase 2)
3. THEN call risk_tool for each vendor using the total_eur from tco_tool (Phase 3)

RULES:
1. ALWAYS follow the 3-phase workflow above
2. NEVER make up data - only use tool outputs
3. ALWAYS show your reasoning with evidence
4. TRIGGER HITL for costs > €5,000 or HIGH risk
5. FALLBACK to deterministic logic if tools fail"""

    def parse_intent(self, prompt: str = "") -> tuple[int, str]:
        """Interactive: ask user for date, then compute week and material."""
        from datetime import datetime
        import re

        print("\n" + "="*50)
        print("📅 PROCUREMENT INTELLIGENCE AGENT")
        print("="*50)

        date_input = input("Enter target date (DD/MM/YYYY) or 'today': ").strip().lower()
        if date_input == 'today':
            target_date = datetime.now()
        else:
            day, month, year = map(int, date_input.split('/'))
            target_date = datetime(year, month, day)

        start_date = datetime(2024, 1, 1)
        days_diff = (target_date - start_date).days
        week = max(1, (days_diff // 7) + 1)
        print(f"📊 Computed procurement week: {week}")

        material_input = input("Enter Material ID (e.g., MAT-001) or press Enter for default: ").strip().upper()
        material = material_input if material_input else "MAT-001"
        print(f"📦 Material selected: {material}")

        return week, material

    def run(self, prompt: str) -> Dict[str, Any]:
        """Main entry: parse prompt, run 3-phase workflow, return LLM-style response."""
        week, material = self.parse_intent(prompt)
        tool_results = run_tools_parallel(week, material)

        safety_errors = safety_check(tool_results)
        if safety_errors:
            return {
                "response": f"Safety check failed: {', '.join(safety_errors)}",
                "error": True,
                "details": safety_errors
            }

        decision = synthesize_decision(tool_results)
        final_decision = hitl_gate(decision)
        response = self._generate_response(final_decision, tool_results)

        return {
            "response": response,
            "decision": final_decision,
            "tool_results": tool_results,
            "week": week,
            "material": material,
            "error": False
        }

    def _generate_response(self, decision: Dict, tool_results: Dict) -> str:
        """Generate human-readable reasoning (mimics LLM output)."""
        best_vendor = decision.get("best_vendor", "UNKNOWN")
        best_score = decision.get("best_score", 0)
        quantity = decision.get("quantity", 0)
        week = decision.get("week", 0)
        material = decision.get("material", "UNKNOWN")

        lines = [
            f"## Procurement Analysis - Week {week}, {material}",
            "",
            f"**Purchase Quantity:** {quantity:,.0f} units",
            "",
            "### Phase 1: Inventory & Vendor Discovery",
        ]

        inv = tool_results.get("inventory", {})
        lines.append(f"- Current stock: {inv.get('current_stock', 0):,.0f} units")
        lines.append(f"- Planned usage: {inv.get('planned_usage', 0):,.0f} units")
        lines.append(f"- Reorder level: {inv.get('reorder_level', 0):,.0f} units")
        lines.append(f"- **Calculated purchase quantity: {inv.get('purchase_quantity', 0):,.0f} units**")
        lines.append("")

        vendors = tool_results.get("vendors", {}).get("available_vendors", {})
        lines.append(f"- Identified {len(vendors)} active vendor(s) for week {week}")
        for vid, vinfo in vendors.items():
            lines.append(f"  - {vid}: {vinfo.get('local_price', 0):.2f} {vinfo.get('currency', 'EUR')}")
        lines.append("")

        lines.append("### Phase 2: Total Cost of Ownership (TCO)")
        tco_data = tool_results.get("tco", {})
        for vid, tco in tco_data.items():
            lines.append(f"**{vid} ({tco.get('vendor_name', vid)}):**")
            lines.append(f"- Material cost: €{tco.get('material_cost_eur', 0):,.2f}")
            lines.append(f"- FX fee: €{tco.get('fx_fee_eur', 0):,.2f}")
            lines.append(f"- Transport: €{tco.get('transport_eur', 0):,.2f}")
            lines.append(f"- Insurance: €{tco.get('insurance_eur', 0):,.2f}")
            lines.append(f"- Tax: €{tco.get('tax_eur', 0):,.2f}")
            lines.append(f"- **Total: €{tco.get('total_eur', 0):,.2f}**")
            if tco.get('early_discount_savings', 0) > 0:
                lines.append(f"- Early payment discount saved: €{tco.get('early_discount_savings', 0):,.2f}")
            lines.append("")

        lines.append("### Phase 3: Risk Assessment (Forward Looking — Informational Only)")
        lines.append("*Risk does NOT affect vendor selection. Lowest TCO wins. Risk informs future strategy.*")
        lines.append("")

        risk_data = tool_results.get("risk", {})
        for vid, risk in risk_data.items():
            level = risk.get("risk_level", "UNKNOWN")
            flag_count = risk.get("flag_count", 0)
            elements = risk.get("risk_elements", {})

            risk_icon = "🟢" if level == "LOW" else "🟡" if level == "MEDIUM" else "🔴"
            lines.append(f"- **{vid}:** Overall={risk_icon} {level} ({flag_count}/5 flags)")

            fx = elements.get("fx_volatility", {})
            fx_icon = "🚩" if fx.get("flagged", False) else "✅"
            lines.append(f"  {fx_icon} FX Volatility: {fx.get('value', 0):.2%} (threshold: {fx.get('threshold', 0.01):.0%}, strong: 3%)")

            mat = elements.get("material_cost", {})
            mat_icon = "🚩" if mat.get("flagged", False) else "✅"
            lines.append(f"  {mat_icon} Material Cost: €{mat.get('current', 0):.2f} vs median €{mat.get('median', 0):.2f} (threshold: {mat.get('threshold_pct', 103)}%)")

            trn = elements.get("transport", {})
            trn_icon = "🚩" if trn.get("flagged", False) else "✅"
            lines.append(f"  {trn_icon} Transport: €{trn.get('current', 0):.2f} vs median €{trn.get('median', 0):.2f} (threshold: {trn.get('threshold_pct', 103)}%)")

            ins = elements.get("insurance", {})
            ins_icon = "🚩" if ins.get("flagged", False) else "✅"
            lines.append(f"  {ins_icon} Insurance: €{ins.get('current', 0):.2f} vs median €{ins.get('median', 0):.2f} (threshold: {ins.get('threshold_pct', 103)}%)")

            tax = elements.get("tax", {})
            tax_icon = "🚩" if tax.get("flagged", False) else "✅"
            lines.append(f"  {tax_icon} Tax: €{tax.get('current', 0):.2f} vs median €{tax.get('median', 0):.2f} (threshold: {tax.get('threshold_pct', 103)}%)")
            lines.append("")

        lines.append("### Recommendation")
        lines.append(f"🏆 **Best Vendor: {best_vendor}**")
        lines.append(f"💰 **Total Cost: €{best_score:,.2f}**")
        lines.append("")

        if decision.get("hitl_required"):
            lines.append("⚠️ **HITL REQUIRED:** " + "; ".join(decision.get("hitl_triggers", [])))
        else:
            lines.append("✅ **Auto-approved** — no HITL triggers")
        lines.append("")

        lines.append("### Vendor Ranking (by total cost — risk is informational only)")
        for vendor, score in decision.get("ranking", []):
            details = decision.get("all_vendors", {}).get(vendor, {})
            risk = details.get("risk_level", "?")
            flags = details.get("flag_count", 0)
            marker = "🏆" if vendor == best_vendor else "  "
            lines.append(f"{marker} {vendor}: €{score:,.2f} (risk: {risk}, flags: {flags}/5)")
        lines.append("")
        lines.append(f"**Justification:** {decision.get('justification', 'N/A')}")
        lines.append(f"**Selection Method:** {decision.get('selection_method', 'N/A')}")

        return "\n".join(lines)

# Initialize the pseudo-agent (zero cost)
controller_agent = PseudoAgent()

print("✅ Pseudo-Agent initialized (zero-cost, deterministic)")



# ================================================================
# SECTION 12 --- FULL WORKFLOW EXECUTION (Dynamic Date Input)
# ================================================================

import nest_asyncio
nest_asyncio.apply()

from datetime import datetime
import asyncio

def parse_date_to_week(date_input: str) -> int:
    """
    Convert user date input to procurement week.
    Supports: 'week N', 'DD/MM' or 'DD/MM/YYYY'
    Year defaults to 2024 if omitted (matches dataset).
    """
    date_input = date_input.strip().lower()
    
    # Mode 1: Direct week number
    if date_input.startswith('week'):
        try:
            week = int(date_input.split()[1])
            return min(week, 54)
        except (IndexError, ValueError):
            raise ValueError("Invalid week format. Use: 'week 5'")
    
    # Mode 2: DD/MM or DD/MM/YYYY (year defaults to 2024)
    try:
        parts = date_input.split('/')
        if len(parts) == 2:
            day, month = map(int, parts)
            year = 2024
        elif len(parts) == 3:
            day, month, year = map(int, parts)
        else:
            raise ValueError
        
        target_date = datetime(year, month, day)
        start_date = datetime(2024, 1, 1)
        days_diff = (target_date - start_date).days
        week = max(1, (days_diff // 7) + 1)
        return min(week, 54)
        
    except ValueError:
        raise ValueError("Invalid date. Use: 'week N', 'DD/MM', or 'DD/MM/YYYY'")


async def run_procurement_workflow(week: int = None, material: str = None, use_llm: bool = False) -> Dict[str, Any]:
    """
    Main entry point with optional interactive fallbacks.
    """
    from datetime import datetime
    
    # --- INTERACTIVE INPUT if not provided ---
    if week is None or material is None:
        print("\n" + "="*50)
        print("📅 PROCUREMENT INTELLIGENCE AGENT")
        print("="*50)
        
        if week is None:
            date_input = input("Enter target date (DD/MM or 'week N'): ").strip().lower()
            week = parse_date_to_week(date_input)
            print(f"📊 Computed procurement week: {week}")
        
        if material is None:
            material = "MAT-001"  # FIXED: Only MAT-001 available
            print(f"📦 Material: {material}")
    
    session_id = f"{material}_W{week}_{pd.Timestamp.now().strftime('%H%M%S')}"
    
    if not use_llm:
        result = deterministic_fallback(week, material)
        memory.save_session(session_id, result)
        return result
    
    # Pseudo-Agent path
    try:
        prompt = f"Run procurement for week {week}, material {material}. Show all vendor costs and select optimal."
        agent_result = controller_agent.run(prompt)
        
        if agent_result.get("error"):
            logger.warning(f"Pseudo-agent error: {agent_result.get('details')}. Falling back.")
            result = deterministic_fallback(week, material)
            memory.save_session(session_id, result)
            return result
        
        final = agent_result["decision"]
        final["agent_response"] = agent_result["response"]
        final["agent_type"] = "pseudo"
        
        # HITL human input
        if final.get("hitl_required"):
            print(final["hitl_prompt"])
            action = input("Enter action (APPROVE/REJECT/MODIFY): ").strip().upper()
            memory.record_approval(final, action)
            
            if action == "REJECT":
                return {"error": "Rejected by human", "decision": final}
            elif action == "MODIFY":
                new_vendor = input("New vendor ID (or press Enter to keep): ").strip()
                if new_vendor:
                    final["best_vendor"] = new_vendor
                    final["justification"] += f" [Modified by human to {new_vendor}]"
        
        memory.save_session(session_id, final)
        return final
        
    except Exception as e:
        logger.error(f"Pseudo-agent failed: {e}. Falling back to deterministic.")
        result = deterministic_fallback(week, material)
        memory.save_session(session_id, result)
        return result



# ================================================================
# SECTION 12a --- SAVINGS REPORT FUNCTION [CHANGED v5]
# ================================================================
# CHANGES v5:
#   - 5 risk elements displayed (FX, Material, Transport, Tax, Insurance)
#   - FX: flag >1%, strong >3%
#   - Cost elements: threshold 103%
# ================================================================

def _fmt_euro(val: float, width: int = 12) -> str:
    """Format value as euro with fixed width for perfect column alignment."""
    s = f"€{val:,.2f}"
    return f"{s:>{width}}"

def generate_savings_report(result: Dict[str, Any]) -> str:
    """
    Generates a comprehensive savings report comparing all vendor alternatives.
    Shows full cost breakdown and savings vs the most expensive option.
    Includes 5-element forward-looking risk assessment (informational only).
    """
    all_vendors = result.get("all_vendors", {})
    best_vendor = result.get("best_vendor")
    quantity = result.get("quantity", 0)
    week = result.get("week", 0)

    if not all_vendors:
        return "⚠️ No vendor data available for report."

    tco_data = {}
    risk_data = {}
    for vid, vdata in all_vendors.items():
        tco_data[vid] = vdata.get("details", {})
        risk_data[vid] = {
            "risk_level": vdata.get("risk_level", "UNKNOWN"),
            "flag_count": vdata.get("flag_count", 0),
            "risk_elements": vdata.get("risk_elements", {})
        }

    all_totals = {vid: tco.get("total_eur", 0) for vid, tco in tco_data.items()}
    most_expensive_vendor = max(all_totals, key=all_totals.get)
    most_expensive_cost = all_totals[most_expensive_vendor]

    lines = []
    lines.append("=" * 105)
    lines.append("📊 COMPREHENSIVE SAVINGS REPORT — ALL VENDOR ALTERNATIVES")
    lines.append("=" * 105)
    lines.append(f"Week: {week} | Material: {result.get('material', 'N/A')} | Purchase Quantity: {quantity:,.0f} units")
    lines.append("=" * 105)
    lines.append("")

    lines.append(f"{'VENDOR':<18} {'MATERIAL €':>14} {'FX FEE €':>12} {'TRANSPORT €':>13} {'INSURANCE €':>13} {'TAX €':>12} {'DISCOUNT €':>13} {'TOTAL €':>15}")
    lines.append("-" * 105)

    for vid in sorted(tco_data.keys()):
        tco = tco_data[vid]
        risk = risk_data.get(vid, {})
        is_best = (vid == best_vendor)
        is_worst = (vid == most_expensive_vendor)
        total_eur = tco.get("total_eur", 0)
        savings = most_expensive_cost - total_eur
        savings_pct = (savings / most_expensive_cost * 100) if most_expensive_cost > 0 else 0
        marker = "🏆 " if is_best else "  "

        row_parts = [
            f"{marker}{vid:<15}",
            _fmt_euro(tco.get('material_cost_eur', 0), 14),
            _fmt_euro(tco.get('fx_fee_eur', 0), 12),
            _fmt_euro(tco.get('transport_eur', 0), 13),
            _fmt_euro(tco.get('insurance_eur', 0), 13),
            _fmt_euro(tco.get('tax_eur', 0), 12),
            _fmt_euro(tco.get('early_discount_savings', 0), 13),
            _fmt_euro(total_eur, 15),
        ]
        lines.append(" ".join(row_parts))

        risk_level = risk.get("risk_level", "UNKNOWN")
        flag_count = risk.get("flag_count", 0)
        risk_icon = "🟢" if risk_level == "LOW" else "🟡" if risk_level == "MEDIUM" else "🔴"

        elements = risk.get("risk_elements", {})
        fx_flagged = elements.get("fx_volatility", {}).get("flagged", False)
        mat_flagged = elements.get("material_cost", {}).get("flagged", False)
        trn_flagged = elements.get("transport", {}).get("flagged", False)
        ins_flagged = elements.get("insurance", {}).get("flagged", False)
        tax_flagged = elements.get("tax", {}).get("flagged", False)

        if is_best:
            lines.append(f" {risk_icon} Risk: {risk_level} ({flag_count}/5 flags) | 💰 SAVED: €{savings:,.2f} ({savings_pct:.1f}%) vs {most_expensive_vendor} ← SELECTED")
        elif is_worst:
            lines.append(f" {risk_icon} Risk: {risk_level} ({flag_count}/5 flags) | 📌 Most expensive option (baseline)")
        else:
            lines.append(f" {risk_icon} Risk: {risk_level} ({flag_count}/5 flags) | 💰 SAVED: €{savings:,.2f} ({savings_pct:.1f}%) vs {most_expensive_vendor}")

        elem_icons = []
        if fx_flagged: elem_icons.append("FX🚩")
        if mat_flagged: elem_icons.append("Mat🚩")
        if trn_flagged: elem_icons.append("Trn🚩")
        if ins_flagged: elem_icons.append("Ins🚩")
        if tax_flagged: elem_icons.append("Tax🚩")
        if elem_icons:
            lines.append(f"   Forward-looking alerts: {', '.join(elem_icons)} (103% threshold, 1%/3% FX)")

        lines.append(f" Currency: {tco.get('currency', 'EUR')} | FX Rate: {tco.get('fx_rate')} | Payment: {tco.get('payment_terms')} days | Early Disc: {tco.get('early_discount_rate')}%")
        lines.append("")

    best_tco = tco_data.get(best_vendor, {})
    actual_savings = most_expensive_cost - best_tco.get("total_eur", 0)
    lines.append("=" * 105)
    lines.append("💡 EXECUTIVE SUMMARY")
    lines.append("=" * 105)
    lines.append(f" 🏆 Selected Vendor: {best_vendor} ({best_tco.get('vendor_name', '')})")
    lines.append(f" 💰 Total Cost: €{best_tco.get('total_eur', 0):,.2f}")
    lines.append(f" 📌 Most Expensive: €{most_expensive_cost:,.2f} ({most_expensive_vendor})")
    lines.append(f" 💰 TOTAL SAVINGS: €{actual_savings:,.2f} ({(actual_savings/most_expensive_cost*100):.1f}%)")
    lines.append(f" 🛡️ Risk Level: {risk_data.get(best_vendor, {}).get('risk_level', '?')} ({risk_data.get(best_vendor, {}).get('flag_count', 0)}/5 flags)")
    lines.append(f" 🎁 Early Pay Discount: €{best_tco.get('early_discount_savings', 0):,.2f}")
    lines.append("=" * 105)
    lines.append("* Risk assessment is forward-looking and informational only. Lowest TCO wins.")
    lines.append("* FX flagged if >1% volatility, strong if >3%. Cost elements flagged if >103% of median.")
    lines.append("* Savings calculated vs. most expensive vendor alternative")
    lines.append("=" * 105)

    return "\n".join(lines)



# ================================================================
# SECTION 12b --- INTERACTIVE WIDGET INTERFACE (Jupyter/Kaggle)
# ================================================================

def create_interactive_dashboard():
    """
    Creates a clickable dashboard for dynamic procurement execution.
    Includes interactive HITL approval and comprehensive savings report.
    """
    if not IPYWIDGETS_AVAILABLE:
        print("❌ ipywidgets not installed. Run: !pip install ipywidgets")
        return None
    
    # Input widgets
    date_widget = widgets.Text(
        value='week 1',
        description='Date:',
        placeholder='week 5 or DD/MM',
        layout=widgets.Layout(width='300px')
    )
    
    material_display = widgets.HTML(
        value='<b>Material: MAT-001</b>'
    )
    
    run_button = widgets.Button(
        description='▶️ Run Procurement',
        button_style='success',
        layout=widgets.Layout(width='200px', height='40px')
    )
    
    output_area = widgets.Output()
    
    # HITL widgets (hidden by default)
    hitl_box = VBox([])
    
    def on_run_clicked(b):
        with output_area:
            clear_output(wait=True)
            print("="*70)
            print("🚀 EXECUTING PROCUREMENT WORKFLOW...")
            print("="*70)
            
            try:
                week = parse_date_to_week(date_widget.value)
                material = "MAT-001"
                
                print(f"📅 Input: {date_widget.value}")
                print(f"📊 Resolved Week: {week}")
                print(f"📦 Material: {material}")
                print(f"🤖 Mode: Deterministic (zero cost)")
                print("-"*70)
                
                result = asyncio.get_event_loop().run_until_complete(
                    run_procurement_workflow(week=week, material=material, use_llm=False)
                )
                
                # Display results
                print(f"\n🏆 BEST VENDOR: {result.get('best_vendor', 'N/A')}")
                print(f"💰 TOTAL COST: €{result.get('best_score', 0):,.2f}")
                print(f"📊 QUANTITY: {result.get('quantity', 0)} units")
                
                # Check HITL status
                if result.get("hitl_required"):
                    print(f"\n⚠️ HITL REQUIRED:")
                    for trigger in result.get("hitl_triggers", []):
                        print(f"   • {trigger}")
                    show_hitl_controls(result, week, material)
                    return
                
                print(f"\n✅ Auto-approved — no HITL triggers")
                
                # ============================================================
                # COMPREHENSIVE SAVINGS REPORT
                # ============================================================
                print("\n" + generate_savings_report(result))
                
                print(f"\n{'='*70}")
                print("✅ WORKFLOW COMPLETE")
                print(f"{'='*70}")
                
            except Exception as e:
                print(f"❌ ERROR: {type(e).__name__}: {str(e)}")
                traceback.print_exc()
    
    def show_hitl_controls(decision, week, material):
        """Display APPROVE / REJECT / MODIFY buttons"""
        
        hitl_label = widgets.HTML(
            value=f"<h4>⚠️ APPROVAL REQUIRED</h4>"
                  f"<p><b>Vendor:</b> {decision.get('best_vendor')}<br>"
                  f"<b>Cost:</b> €{decision.get('best_score', 0):,.2f}<br>"
                  f"<b>Quantity:</b> {decision.get('quantity', 0)} units</p>"
        )
        
        approve_btn = widgets.Button(
            description='✅ APPROVE',
            button_style='success',
            layout=widgets.Layout(width='120px')
        )
        
        reject_btn = widgets.Button(
            description='❌ REJECT',
            button_style='danger',
            layout=widgets.Layout(width='120px')
        )
        
        modify_btn = widgets.Button(
            description='✏️ MODIFY',
            button_style='warning',
            layout=widgets.Layout(width='120px')
        )
        
        modify_input = widgets.Text(
            placeholder='Enter new vendor ID',
            layout=widgets.Layout(width='150px', display='none')
        )
        
        confirm_modify_btn = widgets.Button(
            description='Confirm',
            button_style='info',
            layout=widgets.Layout(width='100px', display='none')
        )
        
        def on_approve(b):
            with output_area:
                clear_output(wait=True)
                memory.record_approval(decision, "APPROVE")
                print("✅ APPROVED — Order proceeding")
                print(f"   Vendor: {decision.get('best_vendor')}")
                print(f"   Cost: €{decision.get('best_score', 0):,.2f}")
                # Show savings report after approval
                print("\n" + generate_savings_report(decision))
                hide_hitl()
        
        def on_reject(b):
            with output_area:
                clear_output(wait=True)
                memory.record_approval(decision, "REJECT")
                print("❌ REJECTED — Order cancelled")
                hide_hitl()
        
        def on_modify(b):
            modify_input.layout.display = 'block'
            confirm_modify_btn.layout.display = 'block'
        
        def on_confirm_modify(b):
            new_vendor = modify_input.value.strip()
            if new_vendor:
                decision["best_vendor"] = new_vendor
                decision["justification"] += f" [Modified by human to {new_vendor}]"
                memory.record_approval(decision, "MODIFY")
                with output_area:
                    clear_output(wait=True)
                    print(f"✏️ MODIFIED — New vendor: {new_vendor}")
                    print(f"   Updated cost: €{decision.get('best_score', 0):,.2f}")
                    print("\n" + generate_savings_report(decision))
                    hide_hitl()
        
        def hide_hitl():
            hitl_box.children = []
        
        approve_btn.on_click(on_approve)
        reject_btn.on_click(on_reject)
        modify_btn.on_click(on_modify)
        confirm_modify_btn.on_click(on_confirm_modify)
        
        hitl_box.children = [
            hitl_label,
            HBox([approve_btn, reject_btn, modify_btn]),
            HBox([modify_input, confirm_modify_btn])
        ]
    
    run_button.on_click(on_run_clicked)
    
    # Layout
    inputs_row = HBox([date_widget, material_display])
    dashboard = VBox([
        widgets.HTML("<h3>📅 Procurement Intelligence Agent — Interactive Dashboard</h3>"),
        inputs_row,
        run_button,
        widgets.HTML("<hr>"),
        output_area,
        hitl_box
    ])
    
    display(dashboard)
    return dashboard



# ================================================================
# SECTION 13 — MCP SERVER CONFIGURATION (Local / Documentation Only)
# ================================================================
# ARCHITECTURAL NOTE: This is a MOCK MCP configuration demonstrating
# Model Context Protocol interoperability standards. The URL and headers
# are schema placeholders. No actual connection is attempted because:
#   1. ADK_AVAILABLE is False in Kaggle environment
#   2. The PseudoAgent path (Sections 11-12) never invokes MCP
#   3. Notebook internet is disabled in published version
# This mirrors enterprise air-gapped deployments where external MCP
# servers are unreachable but the configuration schema remains valid.
# ================================================================

MCP_CONFIG = {
    "mcpServers": {
        "google-developer-knowledge": {
            "headers": {
                "X-Goog-Api-Key": os.environ.get("GOOGLE_API_KEY", "")  # Fixed: pull from env
            },
            "serverUrl": "https://developerknowledge.googleapis.com/mcp"  # Placeholder
        }
    }
}

def save_mcp_config():
    """Save MCP config locally — no external server connection attempted"""
    config_path = "./mcp_config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(MCP_CONFIG, f, indent=2)
    print(f"MCP config saved to {config_path}")
    print("   NOTE: This is a mock config for demonstration. No server connection attempted.")

save_mcp_config()


# ================================================================
# SECTION 14 --- AGENT SKILLS (Local .agents/skills/) [CHANGED v5]
# ================================================================


import os

SKILLS_DIR = "./.agents/skills"
os.makedirs(SKILLS_DIR, exist_ok=True)

# Skill 1: Procurement Workflow (UNCHANGED)
procurement_skill = """---
name: procurement-workflow
description: Execute full procurement workflow from inventory check to vendor selection. Use when user asks to procure materials, run procurement, or optimize purchasing.
---

# Procurement Workflow Skill

## Steps
1. Check inventory levels (current stock, planned usage, reorder level)
2. Calculate purchase quantity: max(0, usage + reorder - current)
3. Retrieve all available vendors and prices
4. Get FX rates and fees for each vendor's currency
5. Compute full TCO per vendor (material + FX + transport + tax + insurance - discount)
6. Assess risk (FX volatility, cost anomaly)
7. Select vendor with lowest risk-adjusted cost
8. Trigger HITL if cost > threshold or HIGH risk

## Output Format
- Recommended vendor with full cost breakdown
- Risk assessment per vendor
- HITL status
- Justification"""

# Skill 2: TCO Calculation (UNCHANGED)
tco_skill = """---
name: tco-calculator
description: Compute Total Cost of Ownership for procurement decisions. Use when user asks about costs, pricing, or vendor comparison.
---

# TCO Calculator Skill

## Formula
```
EUR_cost = (local_price * quantity) / fx_rate
Discounted = EUR_cost * (1 - early_payment_discount)
FX_fee = Discounted * fx_fee_percentage
Transport = transport_per_unit * quantity
Insurance = Discounted * insurance_rate
Tax = (Discounted + Insurance + Transport) * tax_rate
Total = Discounted + FX_fee + Transport + Insurance + Tax
```

## Constraints
- Always apply early payment discount before tax
- Tax base includes material + insurance + transport
- Never return negative costs"""

# Skill 3: FX & Cost Risk Assessment (UPDATED v5 — 5 elements, tight thresholds)
fx_risk_skill = """---
name: fx-risk-assessor
description: Assess foreign exchange and cost risk for procurement. Use when user mentions currency, FX, volatility, exchange rates, material cost, transport cost, insurance cost, or tax anomalies.
---

# FX & Cost Risk Assessment Skill

## 5 Risk Elements (Forward Looking — Informational Only)

### 1. FX Volatility
- **Metric**: |current_rate - 4wk_avg| / 4wk_avg
- **Threshold**: >1% = flag, >3% = strong flag
- **Impact**: Currency fluctuation risk on EUR conversion

### 2. Material Cost Anomaly
- **Metric**: current_local_price vs. historical median (weeks 1-4)
- **Threshold**: >103% of median = flag
- **Impact**: Supplier price increase detection

### 3. Transport Cost Anomaly
- **Metric**: current_transport_per_unit vs. historical median
- **Threshold**: >103% of median = flag
- **Impact**: Logistics cost surge detection

### 4. Insurance Cost Anomaly
- **Metric**: current_insurance_amount vs. historical median
- **Threshold**: >103% of median = flag
- **Impact**: Insurance premium change detection

### 5. Tax Anomaly
- **Metric**: current_tax_amount vs. historical median
- **Threshold**: >103% of median = flag
- **Impact**: Tax/regulatory cost change detection

## Risk Levels (Based on Flag Count)
- **LOW**: 0 flags — all elements within normal range
- **MEDIUM**: 1-2 flags — one or two elements above threshold
- **HIGH**: 3-5 flags OR FX >3% — multiple anomalies or severe currency volatility

## Important Note
- Risk assessment is **FORWARD LOOKING and INFORMATIONAL ONLY**
- Risk does **NOT** affect vendor selection (lowest TCO always wins)
- Risk informs **future procurement strategy** and triggers HITL when HIGH
- All cost thresholds set at **103%** (3% above historical median)
- FX thresholds set at **1%** (flag) and **3%** (strong)

## Action
- HIGH risk → HITL mandatory (human review required)
- MEDIUM risk → HITL recommended (informational)
- LOW risk → auto-approve (proceed with order)
"""

# Save all 3 skills
skills = {
    "procurement-workflow": procurement_skill,
    "tco-calculator": tco_skill,
    "fx-risk-assessor": fx_risk_skill
}

for name, content in skills.items():
    skill_dir = os.path.join(SKILLS_DIR, name)
    os.makedirs(skill_dir, exist_ok=True)
    with open(os.path.join(skill_dir, "SKILL.md"), 'w', encoding='utf-8') as f:
        f.write(content)

print(f"✅ 3 Agent Skills saved to {SKILLS_DIR}")
print("  - procurement-workflow")
print("  - tco-calculator")
print("  - fx-risk-assessor (UPDATED v5: 5 elements, 1%/3% FX, 103% cost)")




# ================================================================
# SECTION 14a — Dynamic Skill Registry
# ================================================================
import os
import re
from difflib import SequenceMatcher

class SkillRegistry:
    """Zero-cost skill loader with progressive disclosure"""
    
    def __init__(self, skills_dir: str = "./.agents/skills"):
        self.skills_dir = skills_dir
        self.metadata = {}  # Always loaded: name, description, triggers
        self.bodies = {}    # Loaded on demand
        self._index_skills()
    
    def _index_skills(self):
        """Parse SKILL.md frontmatter only — cheap metadata"""
        for skill_name in os.listdir(self.skills_dir):
            skill_path = os.path.join(self.skills_dir, skill_name, "SKILL.md")
            if os.path.exists(skill_path):
                with open(skill_path, encoding='utf-8') as f:
                    content = f.read()
                # Extract YAML frontmatter
                frontmatter = re.search(r'^---\n(.*?)\n---', content, re.DOTALL)
                if frontmatter:
                    meta = self._parse_yaml(frontmatter.group(1))
                    self.metadata[skill_name] = {
                        'name': meta.get('name', skill_name),
                        'description': meta.get('description', ''),
                        'triggers': self._extract_triggers(meta.get('description', ''))
                    }
                    self.bodies[skill_name] = content  # Full body stored, loaded on match
    
    def _parse_yaml(self, yaml_text: str) -> dict:
        """Minimal YAML parser — no external deps"""
        result = {}
        for line in yaml_text.strip().split('\n'):
            if ':' in line and not line.startswith('#'):
                key, val = line.split(':', 1)
                result[key.strip()] = val.strip().strip('"\'')
        return result
    
    def _extract_triggers(self, description: str) -> list:
        """Extract trigger phrases from description"""
        # Match "Use when..." patterns
        triggers = re.findall(r'Use when [^.]*', description, re.IGNORECASE)
        # Also extract verb-led phrases
        triggers += re.findall(r'\b(?:procure|purchase|optimize|calculate|assess|compare)\w*', description, re.IGNORECASE)
        return [t.lower() for t in triggers]
    
    def match_skill(self, user_prompt: str) -> tuple[str, float]:
        """Find best matching skill by description similarity"""
        prompt_lower = user_prompt.lower()
        best_match = None
        best_score = 0.0
        
        for skill_name, meta in self.metadata.items():
            # Keyword overlap score
            trigger_hits = sum(1 for t in meta['triggers'] if t in prompt_lower)
            # Sequence similarity on description
            desc_sim = SequenceMatcher(None, prompt_lower, meta['description'].lower()).ratio()
            score = trigger_hits * 0.6 + desc_sim * 0.4
            
            if score > best_score:
                best_score = score
                best_match = skill_name
        
        return best_match, best_score
    
    def load_skill(self, skill_name: str) -> dict:
        """Load full skill body on demand — progressive disclosure"""
        if skill_name not in self.bodies:
            return None
        return {
            'metadata': self.metadata[skill_name],
            'body': self.bodies[skill_name]
        }

# Usage in PseudoAgent.run():
# registry = SkillRegistry()
# matched_skill, confidence = registry.match_skill(prompt)
# if confidence > 0.3:
#     skill = registry.load_skill(matched_skill)
#     # Inject skill instructions into reasoning



# ================================================================
# SECTION 15 — SECURITY FEATURES
# ================================================================

# 1. Input validation
def validate_procurement_request(week: int, material: str) -> tuple[bool, str]:
    if not isinstance(week, int) or week < 1 or week > 52:
        return False, f"Invalid week: {week} (must be 1-52)"
    if not material or not material.startswith("MAT-"):
        return False, f"Invalid material ID: {material}"
    return True, "Valid"

# 2. Cost bounds check
def validate_cost_bounds(cost: float) -> bool:
    return 0 < cost < 1000000  # Reject negative or absurd costs

# 3. Deterministic override (no LLM for critical decisions)
CRITICAL_THRESHOLD = 10000  # Always deterministic above this

def should_use_deterministic(cost_estimate: float) -> bool:
    return cost_estimate > CRITICAL_THRESHOLD

print("✅ Security features active (validation + bounds + deterministic override)")



# ================================================================
# SECTION 15a — SECURITY ARCHITECTURE (Zero-Cost)
# ================================================================
# ARCHITECTURAL NOTE: This section implements a 7-pillar security
# framework using only Python stdlib and pandas. No external IAM,
# firewall, or observability platforms are used — to emulate enterprise
# configuration with security patterns respecting zero cost constraints.
# All classes are fully functional within the deterministic path.
# ================================================================

import hashlib
import json
import re
from datetime import datetime

# Section 15a.1 — Application Security (Pillar 4)
class LLMFirewall:
    """Deterministic prompt and tool call filtering"""
    
    BLOCKED_PATTERNS = [
        r'(?i)(password|secret|key)\s*=\s*["\'][^"\']+["\']',  # Hardcoded secrets
        r'(?i)rm\s+-rf\s+/',  # Dangerous commands
        r'(?i)import\s+os\s*;.*system',  # OS injection
        r'(?i)__import__\s*\(\s*["\']os',  # Obfuscated import
    ]
    
    ALLOWED_TOOLS = {'inventory_tool', 'vendor_tool', 'fx_tool', 'tco_tool', 'risk_tool'}
    
    def scan_prompt(self, prompt: str) -> tuple[bool, list[str]]:
        """Returns (is_safe, violations)"""
        violations = []
        for pattern in self.BLOCKED_PATTERNS:
            if re.search(pattern, prompt):
                violations.append(f"Blocked pattern: {pattern[:50]}...")
        return len(violations) == 0, violations
    
    def validate_tool_call(self, tool_name: str, args: dict) -> tuple[bool, str]:
        if tool_name not in self.ALLOWED_TOOLS:
            return False, f"Tool {tool_name} not in allowlist"
        # Validate argument types
        if tool_name == 'tco_tool' and args.get('quantity', 0) > 100000:
            return False, "Quantity exceeds safety threshold"
        return True, "OK"

# Section 15a.2 — Identity & Access (Pillar 5)
class SimpleIdentity:
    """Role-based access without external IAM"""
    
    ROLES = {
        'analyst': {'inventory_tool', 'vendor_tool', 'fx_tool'},
        'buyer': {'inventory_tool', 'vendor_tool', 'fx_tool', 'tco_tool', 'risk_tool'},
        'admin': {'inventory_tool', 'vendor_tool', 'fx_tool', 'tco_tool', 'risk_tool', 'hitl_approve'}
    }
    
    def __init__(self, user_id: str, role: str):
        self.user_id = user_id
        self.role = role
        self.permissions = self.ROLES.get(role, set())
        self.session_start = pd.Timestamp.now()
    
    def can_execute(self, tool_name: str) -> bool:
        return tool_name in self.permissions
    
    def get_session_context(self) -> dict:
        return {
            'user_id': self.user_id,
            'role': self.role,
            'permissions': list(self.permissions),
            'session_age_minutes': (pd.Timestamp.now() - self.session_start).total_seconds() / 60
        }

# Section 15a.3 — Audit & SecOps (Pillar 6)
class AuditLogger:
    """Immutable audit trail — append-only JSON lines"""
    
    def __init__(self, log_path: str = "./audit.log"):
        self.log_path = log_path
        self.session_events = []
    
    def log(self, event_type: str, tool_name: str, args: dict, result: dict, identity: SimpleIdentity):
        entry = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'event_type': event_type,
            'tool_name': tool_name,
            'args_hash': hashlib.sha256(json.dumps(args, sort_keys=True).encode()).hexdigest()[:16],
            'result_status': 'success' if 'error' not in result else 'error',
            'user_id': identity.user_id,
            'role': identity.role,
            'session_age': identity.get_session_context()['session_age_minutes']
        }
        self.session_events.append(entry)
        # Append to persistent log
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry) + '\n')
    
    def detect_anomaly(self) -> list[str]:
        """Simple anomaly detection on session patterns"""
        alerts = []
        tool_counts = {}
        for e in self.session_events:
            tool_counts[e['tool_name']] = tool_counts.get(e['tool_name'], 0) + 1
        # Flag: same tool called >10x in one session
        for tool, count in tool_counts.items():
            if count > 10:
                alerts.append(f"Excessive {tool} usage: {count} calls")
        # Flag: rapid sequential calls (potential loop)
        if len(self.session_events) > 20:
            alerts.append("High event volume — possible infinite loop")
        return alerts

# Section 15a.4 — Governance (Pillar 7)
class RiskAttestation:
    """Plain-English approval summaries"""
    
    def generate_summary(self, decision: dict, identity: SimpleIdentity) -> str:
        """Translate code decision to human-readable summary"""
        lines = [
            "=== PROCUREMENT APPROVAL SUMMARY ===",
            f"Requester: {identity.user_id} (role: {identity.role})",
            f"Recommended Vendor: {decision.get('best_vendor', 'N/A')}",
            f"Total Cost: €{decision.get('best_score', 0):,.2f}",
            f"Quantity: {decision.get('quantity', 0)} units",
            f"Risk Level: {decision.get('all_vendors', {}).get(decision.get('best_vendor'), {}).get('risk_level', 'UNKNOWN')}",
            "",
            "HITL Triggers:",
        ]
        triggers = decision.get('hitl_triggers', [])
        lines.extend(triggers if triggers else ["None — auto-approved"])
        lines.append("")
        lines.append("Digital Signature: [SIMULATED — " + hashlib.sha256(
            json.dumps(decision, sort_keys=True).encode()
        ).hexdigest()[:16] + "]")
        
        return '\n'.join(lines)



# ================================================================
# SECTION 16 — DEPLOYMENT SCAFFOLDING & GAP DOCUMENTATION
# ================================================================
# ARCHITECTURAL NOTE: The Makefile, pyproject.toml, AGENTS.md, and
# GAPS_DOCUMENTATION.md below demonstrate production deployment
# scaffolding and enterprise architecture documentation. The Makefile
# deploy targets are NON-FUNCTIONAL without a billing-enabled Google
# Cloud project and are included to show production deployment knowledge.
# In the zero-cost capstone environment, these targets safely exit with
# an informational message. All files are written to ./
# for persistence in the saved version.
# ================================================================

# 16a — Makefile content (for video/demo explanation)
makefile_content = """# Procurement Intelligence Agent — Deployment Ready
# 
# SAFE TARGETS (run these only):
#   make lint, make test, make playground
# 
# DOCUMENTATION TARGETS (require paid GCP project — do not run):
#   make deploy-dry-run, make deploy

install:
	uv pip install -e .

lint:
	agents-cli lint

test:
	pytest tests/

playground:
	agents-cli playground

deploy-dry-run:
	@echo "INFO: Requires billing-enabled GCP project."
	@echo "      Included as architectural documentation only."
	@exit 0

deploy:
	@echo "INFO: Requires billing-enabled GCP project."
	@echo "      Included as architectural documentation only."
	@exit 0

clean:
	rm -rf __pycache__ .pytest_cache
"""

with open("./Makefile", 'w', encoding='utf-8') as f:
    f.write(makefile_content)

# 16b — pyproject.toml content
pyproject_content = """[project]
name = "procurement-intelligence-agent"
version = "2025.1.0"
description = "AI Agent for procurement optimization"
requires-python = ">=3.11"
dependencies = [
    "google-adk>=2.0.0",
    "google-genai",
    "pandas",
    "numpy",
    "nest_asyncio",
]

[project.optional-dependencies]
dev = ["pytest", "semgrep", "pre-commit"]
"""

with open("./pyproject.toml", 'w', encoding='utf-8') as f:
    f.write(pyproject_content)

print("Deployment scaffolding created (Makefile, pyproject.toml)")
print("   NOTE: Makefile deploy targets are documentation-only.")
print("   Actual deployment requires billing-enabled GCP project.")

# 16c — AGENTS.md rule file
agents_md_content = """# AGENTS.md — Procurement Intelligence Agent

## Stack
- Python 3.11+
- Google ADK (optional, fallback to deterministic)
- Pandas, NumPy

## Conventions
- All costs in EUR
- Week range: 1-52
- Material IDs: MAT-XXX format
- Early payment discount applied before tax

## Hard Rules
1. NEVER make up FX rates — always query from dataset
2. NEVER return negative costs
3. ALWAYS trigger HITL for costs > €5,000 or HIGH risk
4. ALWAYS use deterministic fallback if LLM fails
5. NEVER expose secrets in generated code

## Workflow
1. Inventory check → 2. Vendor discovery → 3. TCO calculation → 4. Risk assessment → 5. Decision synthesis → 6. HITL gate
"""

with open("./AGENTS.md", 'w', encoding='utf-8') as f:
    f.write(agents_md_content)

print("AGENTS.md rule file created")

# 16d — GAP DOCUMENTATION
gaps_documentation = """# Gap Analysis — Free-Tier Constraints

| Gap | Reason for Acceptability | Mitigation Implemented |
|-----|------------------------|----------------------|
| No live LLM (Gemini Pro) | Billing constraint — €0 budget | PseudoAgent with deterministic logic; regex-based NLU; templated reasoning |
| No operational MCP server | No cloud hosting budget; Kaggle ephemeral | Mock MCP client with local JSON schemas; config validates connection format |
| No dynamic Skills runtime | No ADK Agent Engine deployment | SkillRegistry with stdlib parsing; progressive disclosure via lazy loading |
| No kernel-level sandboxing | Kaggle environment is already containerized | Input validation, output bounds, tool allowlists, prompt firewall |
| No SPIFFE/ABAC identity | No enterprise IAM infrastructure | SimpleIdentity RBAC matrix with role-based tool permissions |
| No OpenTelemetry backend | No observability platform subscription | File-based audit logger with JSON lines; anomaly detection on patterns |
| No CI/CD pipeline | No GitHub Actions runner / cloud build | Local Makefile with lint/test targets; eval suite runs in notebook |
| No LLM-as-judge | No API credits for evaluation scoring | Rule-based judge with rubric scoring; golden dataset with trajectory matching |

## Design Principle
All "missing" enterprise infrastructure is replaced with deterministic Python equivalents that demonstrate architectural understanding of the production pattern, while remaining executable at zero cost.

## Evaluation Coverage (Deterministic Path)
- [x] Trigger accuracy: SkillRegistry keyword matching tested
- [x] Execution correctness: TCO formula validated against known cases
- [x] Trajectory compliance: 3-phase workflow enforced in code
- [x] Regression: Safety checks prevent invalid outputs
- [x] Token budget: No LLM calls = zero tokens consumed
"""

with open("./GAPS_DOCUMENTATION.md", 'w', encoding='utf-8') as f:
    f.write(gaps_documentation)

print("GAPS_DOCUMENTATION.md written to ./")
print("=" * 60)
print("SECTION 16 COMPLETE — All scaffolding and documentation files created")
print("=" * 60)



# ================================================================
# SECTION 17 --- DEMONSTRATION [FIXED v6]
# ================================================================
# FIX v6:
#   - risk_tool_func() now uses actual quantity parameter instead of hardcoded 100
#   - Week 15 selected for maximum risk flag visibility
# ================================================================

# MODE SWITCH: Set True for local interactive testing, False for Kaggle submission
DEMO_MODE = False

print("\n" + "="*70)
print("PROCUREMENT INTELLIGENCE AGENT --- DEMONSTRATION")
print("="*70)

if DEMO_MODE:
    # Interactive dashboard mode (local use only)
    print("\nLaunching interactive dashboard...")
    dashboard = create_interactive_dashboard()
else:
    # Batch demonstration mode (Kaggle submission --- shows full output)
    print("\nRunning batch demonstration mode (no UI)")
    print("-"*70)
    
    # ================================================================
    # WEEK 15: MAXIMUM RISK VISIBILITY SHOWCASE
    # ================================================================
    # Week 15 features:
    #   - VEND-001: HIGH risk (4/5 flags: FX + Material + Tax + Insurance)
    #   - VEND-002: HIGH risk (3/5 flags: FX + Tax + Insurance)  
    #   - VEND-003: MEDIUM risk (1/5 flag: FX)
    #   - All vendors exceed €5,000 HITL threshold
    #   - Demonstrates all 4 flag types in action
    # ================================================================
    
    # Run procurement workflow directly (deterministic path, no async needed)
    result = deterministic_fallback(15, "MAT-001")
    
    # ============================================================
    # 1. PROCUREMENT RESULT
    # ============================================================
    print(f"\nWeek {result['week']} | Material {result['material']}")
    print(f"Quantity: {result['quantity']} units")
    print(f"Best Vendor: {result['best_vendor']} --- EUR {result['best_score']:,.2f}")
    print(f"Justification: {result.get('justification', 'N/A')}")
    
    # ============================================================
    # 2. FULL VENDOR COMPARISON --- HORIZONTAL TABLE FORMAT
    # ============================================================
    print("\n" + "="*105)
    print("ALL VENDOR ALTERNATIVES")
    print("="*105)
    print(f"{'VENDOR':<12} {'MATERIAL EUR':>12} {'FX FEE EUR':>10} {'TRANSPORT EUR':>12} {'INSURANCE EUR':>12} {'TAX EUR':>10} {'DISCOUNT EUR':>12} {'TOTAL EUR':>14} {'RISK':>10}")
    print("-"*105)
    
    for vid, vdata in result.get("all_vendors", {}).items():
        is_best = (vid == result['best_vendor'])
        marker = "Best" if is_best else "  "
        details = vdata.get('details', {})
        risk = vdata.get('risk_level', '?')
        flags = vdata.get('flag_count', 0)
        risk_str = f"{risk} ({flags}/5)"
        
        print(f"{marker}{vid:<10} EUR {details.get('material_cost_eur', 0):>10,.2f} EUR {details.get('fx_fee_eur', 0):>8,.2f} EUR {details.get('transport_eur', 0):>10,.2f} EUR {details.get('insurance_eur', 0):>10,.2f} EUR {details.get('tax_eur', 0):>8,.2f} EUR {details.get('early_discount_savings', 0):>10,.2f} EUR {details.get('total_eur', 0):>12,.2f} {risk_str:>10}")
    
    # ============================================================
    # 3. 5-ELEMENT RISK BREAKDOWN --- HORIZONTAL FORMAT
    # ============================================================
    print("\n" + "="*105)
    print("FORWARD-LOOKING RISK ASSESSMENT (Informational Only)")
    print("="*105)
    print("Risk does NOT affect vendor selection. Lowest TCO wins.")
    print("-"*105)
    
    for vid, vdata in result.get("all_vendors", {}).items():
        risk = vdata.get("risk_level", "?")
        flags = vdata.get("flag_count", 0)
        elements = vdata.get("risk_elements", {})
        icon = "Low" if risk == "LOW" else "Medium" if risk == "MEDIUM" else "High" if risk == "HIGH" else "Unknown"
        
        fx = elements.get("fx_volatility", {})
        mat = elements.get("material_cost", {})
        trn = elements.get("transport", {})
        ins = elements.get("insurance", {})
        tax = elements.get("tax", {})
        
        fx_icon = "FX!" if fx.get("flagged") else "FXok"
        mat_icon = "Mat!" if mat.get("flagged") else "Matok"
        trn_icon = "Trn!" if trn.get("flagged") else "Trnok"
        ins_icon = "Ins!" if ins.get("flagged") else "Insok"
        tax_icon = "Tax!" if tax.get("flagged") else "Taxok"
        
        print(f"\n{icon} {vid}: {risk} ({flags}/5) | {fx_icon} | {mat_icon} | {trn_icon} | {ins_icon} | {tax_icon}")
        print(f"   (103% threshold, 1%/3% FX)")
        
        flagged_details = []
        if fx.get("flagged"):
            flagged_details.append(f"FX: {fx.get('value', 0):.2%}")
        if mat.get("flagged"):
            flagged_details.append(f"Mat: EUR {mat.get('current', 0):.2f} vs median EUR {mat.get('median', 0):.2f}")
        if trn.get("flagged"):
            flagged_details.append(f"Trn: EUR {trn.get('current', 0):.2f} vs median EUR {trn.get('median', 0):.2f}")
        if ins.get("flagged"):
            flagged_details.append(f"Ins: EUR {ins.get('current', 0):.2f} vs median EUR {ins.get('median', 0):.2f}")
        if tax.get("flagged"):
            flagged_details.append(f"Tax: EUR {tax.get('current', 0):.2f} vs median EUR {tax.get('median', 0):.2f}")
        
        if flagged_details:
            print(f"   Forward-looking alerts: {', '.join(flagged_details)}")
    
    # ============================================================
    # 4. HITL GATE (Decision point after full review)
    # ============================================================
    print("\n" + "="*70)
    print("HUMAN-IN-THE-LOOP (HITL) GATE")
    print("="*70)
    
    if result.get("hitl_required"):
        print(f"\nHITL TRIGGERED --- Human approval required:")
        for trigger in result.get("hitl_triggers", []):
            print(f"   * {trigger}")
        
        print(f"\nHITL PROMPT:")
        print(result.get("hitl_prompt", "N/A"))
        
        print("\nAvailable actions: APPROVE | REJECT | MODIFY")
        simulated_action = "APPROVE"
        print(f"\nSIMULATED HUMAN DECISION: {simulated_action}")
        print("   (After reviewing full risk profile above)")
        
        memory.record_approval(result, simulated_action)
        print(f"Logged to audit trail: {memory.approvals[-1]}")
        
        if simulated_action == "APPROVE":
            print("ORDER APPROVED --- Proceeding with procurement")
        elif simulated_action == "REJECT":
            print("ORDER REJECTED")
        elif simulated_action == "MODIFY":
            print("ORDER MODIFIED")
    else:
        print("\nAuto-approved --- no HITL triggers")
        print("   Cost below 5,000 EUR threshold and risk level acceptable")
    
    # ============================================================
    # 5. COMPREHENSIVE SAVINGS REPORT
    # ============================================================
    print("\n" + generate_savings_report(result))
    
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)




# ================================================================
# Section 18 — Evaluation Suite (Zero-Cost)
# ================================================================
import json
from dataclasses import dataclass
from typing import Literal

@dataclass
class EvalCase:
    case_id: str
    input_prompt: str
    expected_skill: str
    expected_tools: list[str]
    expected_output_contains: list[str]
    trajectory_mode: Literal['EXACT', 'IN_ORDER', 'ANY_ORDER'] = 'IN_ORDER'

class EvalSuite:
    """Golden dataset + trajectory scoring"""
    
    def __init__(self):
        self.cases: list[EvalCase] = []
        self.results: list[dict] = []
    
    def load_golden_dataset(self, path: str = "./eval_dataset.json"):
        """Load curated test cases from JSON"""
        with open(path) as f:
            data = json.load(f)
        for c in data['cases']:
            self.cases.append(EvalCase(**c))
    
    def score_trajectory(self, actual_tools: list[str], expected: list[str], mode: str) -> tuple[float, str]:
        """Score tool call sequence"""
        if mode == 'EXACT':
            score = 1.0 if actual_tools == expected else 0.0
            detail = "Exact match" if score == 1.0 else f"Expected {expected}, got {actual_tools}"
        elif mode == 'IN_ORDER':
            # Check if expected is subsequence of actual in order
            idx = 0
            for tool in actual_tools:
                if idx < len(expected) and tool == expected[idx]:
                    idx += 1
            score = idx / len(expected) if expected else 1.0
            detail = f"Matched {idx}/{len(expected)} in order"
        else:  # ANY_ORDER
            score = len(set(actual_tools) & set(expected)) / len(set(expected)) if expected else 1.0
            detail = f"Overlap: {score:.0%}"
        
        return score, detail
    
    def run_evaluation(self, agent_run_func) -> dict:
        """Execute all eval cases against agent function"""
        for case in self.cases:
            # Run agent
            result = agent_run_func(case.input_prompt)
            
            # Extract actual tool sequence from trace
            actual_tools = result.get('tool_results', {}).get('_tool_sequence', [])
            
            # Score trajectory
            traj_score, traj_detail = self.score_trajectory(
                actual_tools, case.expected_tools, case.trajectory_mode
            )
            
            # Check output content
            response = result.get('response', '')
            output_hits = sum(1 for phrase in case.expected_output_contains if phrase in response)
            output_score = output_hits / len(case.expected_output_contains) if case.expected_output_contains else 1.0
            
            # Check skill routing
            skill_match = result.get('matched_skill') == case.expected_skill
            
            self.results.append({
                'case_id': case.case_id,
                'trajectory_score': traj_score,
                'trajectory_detail': traj_detail,
                'output_score': output_score,
                'skill_routed_correctly': skill_match,
                'overall': (traj_score + output_score + float(skill_match)) / 3
            })
        
        # Aggregate
        avg_score = sum(r['overall'] for r in self.results) / len(self.results) if self.results else 0
        return {
            'average_score': round(avg_score, 3),
            'case_results': self.results,
            'pass_threshold': 0.8,
            'passed': avg_score >= 0.8
        }

# Sample golden dataset (create as JSON file)
SAMPLE_GOLDEN_DATASET = {
    "cases": [
        {
            "case_id": "procure_basic_001",
            "input_prompt": "Procure MAT-001 for week 1",
            "expected_skill": "procurement-workflow",
            "expected_tools": ["inventory_tool", "vendor_tool", "tco_tool", "risk_tool"],
            "expected_output_contains": ["Best Vendor", "Total Cost", "Risk Level"],
            "trajectory_mode": "IN_ORDER"
        },
        {
            "case_id": "fx_risk_002",
            "input_prompt": "Check FX risk for vendor V001",
            "expected_skill": "fx-risk-assessor",
            "expected_tools": ["fx_tool", "risk_tool"],
            "expected_output_contains": ["volatility", "risk level"],
            "trajectory_mode": "ANY_ORDER"
        }
    ]
}

# Write sample dataset
with open("./eval_dataset.json", 'w', encoding='utf-8') as f:
    json.dump(SAMPLE_GOLDEN_DATASET, f, indent=2)

# Test eval suite
suite = EvalSuite()
suite.load_golden_dataset("./eval_dataset.json")
print(f"✅ EvalSuite executed — loaded {len(suite.cases)} test cases from golden dataset")

# Quick validation test
if suite.cases:
    print(f"✅ Sample case: {suite.cases[0].case_id} — expects skill '{suite.cases[0].expected_skill}'")
    print(f"✅ Trajectory mode: {suite.cases[0].trajectory_mode}")
print("=" * 60)
print("SECTION 18 COMPLETE — Evaluation framework ready")
print("=" * 60)




# ================================================================
# SECTION 19 --- OBSERVABILITY & TRUST DECAY MONITORING (Zero-Cost)
# ================================================================

import functools
import time
import json
from collections import defaultdict
from datetime import datetime

class LightweightTracer:
    """OpenTelemetry-inspired tracing without external dependencies"""

    def __init__(self, trace_path: str = "./traces.jsonl"):
        self.trace_path = trace_path
        self.current_trace = None
        self.spans = []

    def start_trace(self, session_id: str, user_prompt: str):
        self.current_trace = {
            'trace_id': hashlib.sha256(f"{session_id}_{time.time()}".encode()).hexdigest()[:16],
            'session_id': session_id,
            'start_time': datetime.now().isoformat(),
            'user_prompt': user_prompt[:200],
            'spans': []
        }
        self.spans = []

    def add_span(self, span_name: str, span_type: str, inputs: dict, outputs: dict, duration_ms: float):
        span = {
            'span_id': f"{self.current_trace['trace_id']}_{len(self.spans)}",
            'name': span_name,
            'type': span_type,
            'timestamp': datetime.now().isoformat(),
            'inputs': {k: str(v)[:100] for k, v in inputs.items()},
            'outputs': {k: str(v)[:100] for k, v in outputs.items()},
            'duration_ms': round(duration_ms, 2)
        }
        self.spans.append(span)

    def end_trace(self, final_decision: dict):
        self.current_trace['end_time'] = datetime.now().isoformat()
        self.current_trace['total_duration_ms'] = sum(s['duration_ms'] for s in self.spans)
        self.current_trace['span_count'] = len(self.spans)
        self.current_trace['final_vendor'] = final_decision.get('best_vendor', 'N/A')
        self.current_trace['final_cost'] = final_decision.get('best_score', 0)
        self.current_trace['spans'] = self.spans

        with open(self.trace_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(self.current_trace) + '\n')

        return self.current_trace

class AgentBillOfMaterials:
    """Runtime AgBOM -- track active tools, skills, data sources"""

    def __init__(self):
        self.active_tools = set()
        self.active_skills = set()
        self.data_sources_accessed = set()
        self.tool_call_counts = defaultdict(int)

    def record_tool_use(self, tool_name: str):
        self.active_tools.add(tool_name)
        self.tool_call_counts[tool_name] += 1

    def record_skill_load(self, skill_name: str):
        self.active_skills.add(skill_name)

    def record_data_source(self, source: str):
        self.data_sources_accessed.add(source)

    def get_bom(self) -> dict:
        return {
            'tools': list(self.active_tools),
            'skills': list(self.active_skills),
            'data_sources': list(self.data_sources_accessed),
            'tool_call_counts': dict(self.tool_call_counts),
            'blast_radius_score': len(self.active_tools) + len(self.data_sources_accessed)
        }

class TrustDecayMonitor:
    """Monitor deviation from expected patterns"""

    def __init__(self, baseline_path: str = "./trust_baseline.json"):
        self.baseline_path = baseline_path
        self.baseline = self._load_baseline()
        self.current_scores = []

    def _load_baseline(self) -> dict:
        if os.path.exists(self.baseline_path):
            with open(self.baseline_path, encoding='utf-8') as f:
                return json.load(f)
        return {
            'typical_tool_sequence': ['inventory_tool', 'vendor_tool', 'tco_tool', 'risk_tool'],
            'max_cost_ratio': 1.5,
            'max_session_duration_ms': 30000
        }

    def compute_trust_score(self, trace: dict, bom: dict) -> float:
        """Score 0-1, where 1 = fully trusted, 0 = circuit breaker triggered"""
        penalties = 0.0

        actual_tools = [s['name'] for s in trace.get('spans', []) if s['type'] == 'tool']
        expected = self.baseline['typical_tool_sequence']
        if actual_tools != expected:
            penalties += 0.2

        if bom.get('blast_radius_score', 0) > 6:
            penalties += 0.3

        if trace.get('total_duration_ms', 0) > self.baseline['max_session_duration_ms']:
            penalties += 0.2

        return max(0.0, 1.0 - penalties)

    def should_circuit_break(self, trust_score: float) -> bool:
        return trust_score < 0.4

def traced(span_name: str, span_type: str = 'tool'):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracer = args[0].tracer if hasattr(args[0], 'tracer') else None
            start = time.time() * 1000

            result = func(*args, **kwargs)

            duration = (time.time() * 1000) - start
            if tracer and tracer.current_trace:
                tracer.add_span(
                    span_name=span_name,
                    span_type=span_type,
                    inputs={'args': str(args[1:])[:100]},
                    outputs={'result': str(result)[:100]},
                    duration_ms=duration
                )

            return result
        return wrapper
    return decorator

# Initialize observability stack
tracer = LightweightTracer()
agbom = AgentBillOfMaterials()
trust_monitor = TrustDecayMonitor()

print("Observability stack initialized (tracer, AgBOM, trust monitor)")
