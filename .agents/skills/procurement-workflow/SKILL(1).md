---
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
- Justification