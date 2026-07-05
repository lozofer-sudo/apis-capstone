# Gap Analysis — Free-Tier Constraints

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
