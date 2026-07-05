# AGENTS.md — Procurement Intelligence Agent

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
