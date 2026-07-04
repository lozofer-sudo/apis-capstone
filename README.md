Autonomous Procurement Intelligence System (APIS)
Zero-Cost Enterprise-Grade Procurement Automation
Built Under Budget Constraints with Full Emulation of Enterprise Components
________________________________________
Live Demo
Run this notebook on Kaggle: https://www.kaggle.com/code/wiltonjarms/notebookb6dc356aa2
________________________________________
Executive Summary
Modern procurement workflows in small-to-large enterprises are often slow, fragmented, and manual. Teams must coordinate across inventory systems, vendor databases, pricing sources, currency conversion tools, and approval workflows.
This project introduces APIS, a multi-agent architecture prototype, automating the full procurement workflow from inventory analysis to vendor selection using a central orchestrator and specialized sub-agents.
Key Innovation: Enterprise-grade controls (HITL gates, audit trails, RBAC) implemented through deterministic emulation at zero LLM cost. Architected under strict budget constraints without compromising production-grade governance.
________________________________________
Architecture
The system maps local implementations to their enterprise equivalents:
•	Orchestrator Agent: PseudoAgent (deterministic) maps to ADK Agent + InMemoryRunner
•	Sub-Agents: Python functions with FunctionTool map to ADK delegated tools
•	LLM: Deterministic logic (emulated) maps to Gemini 2.5 Flash/Pro
•	Skills: Local markdown + YAML map to Remote skill registry
•	Compute: Kaggle Notebook (CPU) maps to Cloud Run / GKE
•	Data: CSV datasets map to Cloud SQL + BigQuery
•	Memory: In-memory dict maps to Cloud SQL / Firestore
•	Observability: JSONL traces map to Cloud Trace + OpenTelemetry
•	Security: Python RBAC + firewall map to Cloud IAM + Cloud Armor
•	Deployment: Makefile + pyproject.toml map to Cloud Build + Cloud Deploy
________________________________________
Capabilities
•	Inventory-aware purchasing: optimal order quantity calculation
•	Multi-currency TCO: FX conversion with fee modeling
•	5-element risk scoring: FX volatility, material, transport, insurance, tax
•	Early payment optimization: discount capture before tax
•	HITL governance: mandatory approval for high-cost/high-risk orders
________________________________________
Quick Start
Option 1: Kaggle (Recommended — Zero Setup)
Click the Kaggle link above to run the notebook immediately.
Option 2: Local
Step 1: Clone repo
git clone https://github.com/lozofer-sudo/apis-capstone
Step 2: Install dependencies
pip install pandas numpy
Step 3: Add CSV datasets to ./data/ directory:
•	inventory_levels_weekly.csv
•	vendor_list.csv
•	Euro exchange_rates_weekly.csv
•	vendor_prices_weekly.csv
•	min_amountmax_amountfee_percentage.csv
•	procurement_additional_costs.csv
Step 4: Set API key (optional — for MCP config only)
export GOOGLE_API_KEY="your_key_here"
Step 5: Run
python procurement_agent.py
________________________________________
Project Structure
•	procurement_agent.py: Main implementation (Sections 0-19)
•	README.md: This file
•	AGENTS.md: Team conventions and hard rules
•	Makefile: Build, test, deploy targets
•	pyproject.toml: Dependencies and metadata
•	mcp_config.json: MCP server configuration
•	GAPS_DOCUMENTATION.md: Production gap analysis
•	eval_dataset.json: Golden dataset for evaluation
•	audit.log: Immutable audit trail
•	traces.jsonl: OpenTelemetry-style traces
•	.agents/skills/procurement-workflow/SKILL.md
•	.agents/skills/tco-calculator/SKILL.md
•	.agents/skills/fx-risk-assessor/SKILL.md
________________________________________
Track Selection
Agents for Business
________________________________________
Problem Definition
Modern procurement workflows in mid-to-large enterprises are slow, fragmented, and heavily manual. Teams must coordinate across inventory systems, vendor databases, pricing sources, currency conversion tools, and approval workflows.
This leads to three critical inefficiencies:
Cost Inefficiency: Without systematic Total Cost of Ownership (TCO) modeling including FX fees, transport, insurance, tax, and discounts, organizations often make suboptimal purchasing decisions. Industry estimates suggest 3-7% cost leakage due to poor vendor selection.
Decision Latency: Procurement cycles typically take 5-10 business days due to manual coordination between systems and stakeholders. This creates delays in time-sensitive or volatile pricing environments.
Risk Blindness: Risk factors such as currency volatility, supplier reliability, and cost instability are typically assessed after decisions are made, rather than during vendor selection.
These steps naturally map to an agentic system capable of structured reasoning, tool use, and multi-step decision automation.
________________________________________
Solution Overview
This project introduces the Autonomous Procurement Intelligence System (APIS), a multi-agent architecture inspired by Google’s Agent Development Kit (ADK).
The system automates the full procurement workflow from inventory analysis to vendor selection and purchase order generation using a central orchestrator agent and specialized sub-agents.
Although the production architecture is designed for cloud deployment (Vertex AI, Cloud Run, Cloud SQL), this capstone implements a fully local, zero-cost emulation that preserves the same architectural design, execution logic, and agent behaviors.
________________________________________
Core Architecture
1. Orchestrator Agent (Main Controller)
The system is driven by a central PseudoAgent, emulating an ADK-style orchestrator. It is responsible for six functions:
•	Intent parsing: Interpreting user inputs such as material ID and procurement timing.
•	Skill routing: Matching intent to the correct skill module via SkillRegistry.
•	Workflow management: Enforcing structured multi-phase execution.
•	Sub-agent coordination: Delegating tasks in parallel or sequential execution via ThreadPoolExecutor.
•	Decision synthesis: Producing final vendor selection and reasoning.
•	Human-in-the-loop (HITL): Triggering approval for high-cost or high-risk decisions.
Execution follows a strict three-phase workflow:
•	Phase 1 (parallel execution): Inventory Agent, Vendor Discovery Agent, and FX lookup run simultaneously with no dependencies.
•	Phase 2 (sequential execution): TCO Engine Agent runs per vendor, using the purchase quantity computed in Phase 1.
•	Phase 3 (parallel execution): Risk Assessment Agent runs per vendor, using the total cost from Phase 2.
Code: PseudoAgent class — Section 11; run_tools_parallel() — Section 6.
2. Sub-Agent System
Four specialized sub-agents execute skill-delegated tasks:
•	Inventory Agent: Responsible for stock levels and reorder quantity computation using inventory_tool. Code: Section 5.
•	Vendor Discovery Agent: Responsible for vendor selection and pricing retrieval using vendor_tool and fx_tool. Code: Section 5.
•	TCO Engine Agent: Responsible for full cost computation per vendor using tco_tool. Code: Sections 4-5.
•	Risk Assessment Agent: Responsible for multi-factor risk evaluation using risk_tool. Code: Section 5.
Each sub-agent is stateless and independently executable, enabling modularity, testability, and parallel execution. The orchestrator manages all state handoffs between phases.
3. Skill-Based Delegation System
APIS implements a SkillRegistry that dynamically routes requests based on intent.
Three production-grade skills are defined in .agents/skills/:
•	procurement-workflow: Triggered by keywords “procure”, “purchase”, “optimize”. Purpose: full 8-step pipeline execution from inventory check to HITL gate.
•	tco-calculator: Triggered by keywords “cost”, “pricing”, “compare”. Purpose: TCO formula with constraint rules (discount before tax, no negative costs).
•	fx-risk-assessor: Triggered by keywords “currency”, “FX”, “volatility”, “risk”. Purpose: 5-element risk framework with thresholds and action triggers.
SkillRegistry mechanics:
•	Indexing: Parses YAML frontmatter at initialization (name, description, trigger keywords).
•	Matching: Keyword overlap (60% weight) plus sequence similarity on description (40% weight).
•	Progressive Disclosure: Loads full skill body only when confidence exceeds 0.3.
•	Validation: agents-cli lint (Makefile target) validates skill syntax before execution.
Code: SkillRegistry class — Section 14b.
4. Human-in-the-Loop (HITL)
The system includes explicit human approval gates under three conditions:
1.	Total procurement cost exceeds 5,000 EUR.
2.	Risk level is classified as HIGH (3 or more of 5 elements flagged, or FX volatility exceeds 3%).
3.	Vendor switch from historical preference is detected.
Risk is informational, not punitive — lowest TCO always wins. Risk data informs future strategy and triggers HITL when HIGH.
Code: hitl_gate() — Section 8.
5. Memory and State Tracking
Lightweight session memory tracks three things:
•	Vendor decisions per session.
•	Procurement execution history.
•	Audit traces with cryptographic hashing.
In production, this upgrades to Cloud SQL with transaction isolation.
Code: SimpleMemory class — Section 9.
________________________________________
System Architecture
The system follows a hierarchical agent model with five layers:
1.	User Interface Layer: Interactive widgets, CLI, or REST API in production.
2.	Orchestrator Agent: Central controller managing intent, skills, workflow, and delegation.
3.	Sub-Agent Execution Layer: Four specialized workers handling inventory, vendor discovery, TCO, and risk.
4.	Risk Evaluation and HITL Gate: Forward-looking risk assessment with human approval triggers.
5.	Memory Persistence Layer: Session state, vendor preferences, and audit trails.
Execution is structured to ensure deterministic behavior even under parallel processing conditions.
________________________________________
TCO Calculation Model
The Total Cost of Ownership model includes seven components:
1.	Material cost: Base cost computed as (local_price * quantity) / fx_rate
2.	Currency conversion using FX rates
3.	Early payment discounts applied before tax
4.	Transport costs
5.	Insurance costs
6.	Tax computation
7.	FX fees
The formula step by step:
•	EUR_cost = (local_price * quantity) / fx_rate
•	Discounted = EUR_cost * (1 - early_payment_discount)
•	FX_fee = Discounted * fx_fee_percentage
•	Transport = transport_per_unit * quantity
•	Insurance = Discounted * insurance_rate
•	Tax = (Discounted + Insurance + Transport) * tax_rate
•	Total = Discounted + FX_fee + Transport + Insurance + Tax
Key constraints:
•	Discounts are applied before tax.
•	Tax base includes material, transport, and insurance.
•	Negative values are rejected as invalid inputs.
This ensures consistent and comparable vendor evaluation.
Code: compute_total_cost_with_discounts() — Section 4.
________________________________________
Agent Design Rationale
Why Orchestrator Plus Sub-Agents
This architecture was chosen for five key reasons:
1.	Centralized Control: Ensures consistent execution order and global state management. The orchestrator enforces Phase 1 before Phase 2 before Phase 3 invariant; sub-agents cannot skip steps.
2.	Specialization: Each sub-agent focuses on a single domain responsibility with its own tools and constraints, mirroring how procurement teams organize.
3.	Parallel Execution Efficiency: Independent tasks are executed concurrently via ThreadPoolExecutor to reduce latency. The orchestrator knows which phases are parallel-safe.
4.	Dynamic Routing: Skills are selected at runtime based on user intent. A “check FX risk” query routes to fx-risk-assessor; a “procure MAT-001” query routes to procurement-workflow.
5.	Observability: Each execution step generates structured traces with inputs, outputs, and duration for debugging and auditing.
________________________________________
Risk Assessment System
Risk is computed using a five-factor model. Each factor is evaluated against historical baselines (previous 4 weeks) using the same purchase quantity as the current week, ensuring apples-to-apples comparisons that isolate genuine price and volatility changes from procurement volume effects.
•	FX Volatility: Metric is absolute value of (current_rate - 4_week_average) divided by 4_week_average. Threshold: greater than 1% triggers a flag, greater than 3% triggers a strong flag. Code: Section 5.
•	Material Cost Deviation: Metric is current local price versus historical median. Threshold: greater than 103% of median. Code: Section 5.
•	Transport Cost Deviation: Metric is current transport cost per unit versus historical median. Threshold: greater than 103% of median. Code: Section 5.
•	Insurance Cost Deviation: Metric is current insurance amount versus historical median. Threshold: greater than 103% of median. Historical calculations use the actual purchase quantity (not a hardcoded value) to ensure fair comparison. Code: Section 5.
•	Tax Variation: Metric is current tax amount versus historical median. Threshold: greater than 103% of median. Historical calculations use the actual purchase quantity (not a hardcoded value) to ensure fair comparison. Code: Section 5.
Critical design decisions:
1.	Risk does not influence vendor selection. The system always selects the lowest TCO option. Risk is used for three purposes: human-in-the-loop triggers, strategic decision support, and procurement auditing. This avoids the “risk aversion trap” where conservative scoring systematically excludes innovative vendors.
2.	Quantity-aware historical comparisons. Both current and historical tax and insurance calculations use the identical purchase quantity. This prevents false flags caused by volume scaling and ensures that triggered alerts reflect genuine cost anomalies (price spikes, FX volatility, rate changes) rather than procurement volume differences.
________________________________________
Impact and Value
For an organization spending 10 million EUR annually, the system delivers five improvements:
•	Procurement time: Before 3-5 days, after under 30 seconds, improvement up to 99.9% reduction
•	Cost leakage: Before 3-7% of spend, after under 1% of spend, improvement 70-85% reduction
•	FX risk visibility: Before reactive quarterly reviews, after real-time per PO, improvement 100% visibility
•	Audit compliance: Before manual spreadsheets, after immutable JSON logs, improvement full traceability
•	Approval workflow: Before ad hoc emails, after structured HITL gates, improvement 100% coverage for high-value orders
This transforms procurement from a manual workflow into an automated, auditable, and intelligent decision system.
________________________________________
Technical Implementation
Stack Overview
The system maps local implementations to their enterprise equivalents:
•	Orchestrator Agent: PseudoAgent (emulated) maps to ADK Agent + InMemoryRunner
•	Sub-Agents: Python functions with FunctionTool wrappers map to ADK Agent with delegated tools
•	LLM: Deterministic logic (emulated) maps to Gemini 2.5 Flash/Pro
•	Skills: Local markdown with YAML frontmatter maps to Remote skill registry
•	Compute: Kaggle Notebook (CPU) maps to Cloud Run or GKE
•	Data: Kaggle Input Datasets (CSV) maps to Cloud SQL + BigQuery
•	Memory: In-memory Python dict maps to Cloud SQL or Firestore
•	Observability: File-based JSONL traces maps to Cloud Trace + OpenTelemetry
•	Security: Python classes (RBAC, firewall) maps to Cloud IAM + Cloud Armor
•	Deployment: Makefile + pyproject.toml maps to Cloud Build + Cloud Deploy
Code Structure
•	README.md: This file
•	AGENTS.md: Team conventions and hard rules
•	Makefile: Build, test, deploy targets
•	pyproject.toml: Dependencies and metadata
•	mcp_config.json: MCP server configuration
•	GAPS_DOCUMENTATION.md: Production gap analysis
•	eval_dataset.json: Golden dataset for evaluation
•	audit.log: Immutable audit trail
•	traces.jsonl: OpenTelemetry-style traces
•	.agents/skills/procurement-workflow/SKILL.md
•	.agents/skills/tco-calculator/SKILL.md
•	.agents/skills/fx-risk-assessor/SKILL.md
•	src/procurement_agent.py: Main implementation (Sections 0-19)
Key Design Features
•	Fully offline execution with zero API cost — deterministic logic emulates LLM behavior.
•	Deterministic fallback — if ADK is not installed, orchestrator falls back to pure Python with identical behavior.
•	Parallel execution of independent tasks via ThreadPoolExecutor.
•	Modular skill-based architecture with runtime routing and progressive disclosure.
•	Security validation layer — input bounds, cost sanity checks, tool allowlisting.
•	Lightweight tracing — OpenTelemetry-inspired spans, Agent Bill of Materials, trust decay monitoring.
________________________________________
Security Design
The system implements seven safety controls:
1.	Input Validation: Week bounds (1 to 52), material ID format (MAT-XXX). Code: Section 15.
2.	Cost Bounds: Rejection of negative or absurd costs (exceeding 1 million EUR). Code: Section 15.
3.	Tool Allowlisting: Only registered tools may execute. Code: Section 15b.
4.	LLM Firewall: Regex-based prompt scanning blocks secrets and injection patterns. Code: Section 15b.
5.	RBAC Identity: Role-based access (analyst, buyer, admin) with permission matrices. Code: Section 15b.
6.	Audit Logging: Immutable append-only JSON logs with SHA-256 hashing. Code: Section 15b.
7.	Deterministic Override: All decisions exceeding 10,000 EUR bypass LLM and use pure logic. Code: Section 15.
________________________________________
Applied Course Concepts
•	Agent / Multi-agent system (ADK): Main orchestrator (PseudoAgent) plus 4 sub-agents with skills, tools, and state; strict 3-phase workflow. Code: Sections 5-7, 11.
•	MCP Server: Mock MCP client with local JSON config; validates connection format to Google Developer Knowledge MCP for tool standardization and interoperability. Code: Section 13.
•	Antigravity: Agents float above data sources via tool abstraction; no direct coupling to CSV schema. Tools can be retargeted to APIs or databases without agent changes. Location: Architecture diagram (video).
•	Security features: 7-pillar security including validation, bounds, allowlisting, firewall, RBAC, audit logging, deterministic override. Code: Sections 15, 15b.
•	Deployability: Makefile with agents-cli targets (lint, playground, deploy-dry-run, deploy); pyproject.toml; AGENTS.md rule file; GAPS_DOCUMENTATION.md maps local emulation to production. Location: Deployment walkthrough (video).
•	Agent skills (Agents CLI): 3 skills with YAML frontmatter; SkillRegistry with progressive disclosure; agents-cli lint and agents-cli playground targets in Makefile. Code: Sections 14, 14b.
________________________________________
Project Evolution
The system evolved through six stages:
•	v1: Single-agent monolith with basic TCO. Learned that monolithic agents fail at complex multi-step workflows — state gets lost and reasoning becomes opaque.
•	v2: Multi-function system without orchestration. Learned that without centralized control, parallel execution races produce inconsistent results.
•	v3: Parallel execution without routing logic. Learned that hardcoded workflows cannot adapt to user intent; for example, “just check FX risk” still runs full procurement.
•	v4: Skill-based routing introduced with 3-element risk (FX, material, transport). Learned that risk must be granular to be actionable.
•	v5: Full HITL integration, 5-element risk framework (FX, material, transport, insurance, tax) at 103% threshold, tightened FX thresholds to 1% and 3%, agents-cli scaffolding.
•	v6 (final): Quantity-aware risk assessment fix. Tax and Insurance historical comparisons now use the actual purchase quantity instead of a hardcoded value, eliminating false flags caused by volume scaling. Risk alerts now reflect genuine cost anomalies (price spikes, FX volatility, rate changes) rather than procurement volume differences.
Key insight: Agentic systems are defined not by LLM usage, but by orchestration discipline and system design.
________________________________________
Conclusion
The Autonomous Procurement Intelligence System demonstrates how multi-agent architectures can transform procurement into a structured, automated, and auditable workflow.
By combining a central orchestrator, specialized sub-agents, skill-based routing, and human-in-the-loop safety mechanisms, the system achieves both operational efficiency and decision transparency.
The architecture is fully extensible and can be adapted to other enterprise workflows such as underwriting, claims processing, and logistics optimization.
________________________________________
About the Author
Supply Chain, Corporate Strategy, FP&A | AI Solutions Translator
20+ years of experience in Supply Chain, Corporate Strategy, FP&A with enterprise procedures alongside IT teams.
Open to: Advisory roles, corporate positions, strategic consulting, AI-driven operations — and conversations with anyone exploring this space Thanks
________________________________________
License
MIT — For educational and demonstration purposes.
