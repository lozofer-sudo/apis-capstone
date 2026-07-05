---
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
- Never return negative costs