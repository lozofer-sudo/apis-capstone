---
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
