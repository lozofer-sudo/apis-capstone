# Procurement Intelligence Agent — Deployment Ready
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
