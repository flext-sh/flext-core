# flext-core - Foundation Library
PROJECT_NAME := flext-core
COV_DIR := flext_core
MIN_COVERAGE := 80

include ../base.mk

# === PROJECT-SPECIFIC TARGETS ===
.PHONY: complexity docstring-check test-unit test-integration coverage-html
.PHONY: build docs docs-serve shell diagnose doctor deps-update deps-show

complexity: ## Code complexity analysis
	$(Q)$(POETRY) run radon cc $(SRC_DIR) -a -nb --total-average
	$(Q)$(POETRY) run radon mi $(SRC_DIR) -nb

docstring-check: ## Docstring coverage (80%)
	$(Q)$(POETRY) run interrogate $(SRC_DIR) --fail-under=80 --ignore-init-method --ignore-magic -q

coverage-html: ## HTML coverage report
	$(Q)PYTHONPATH=$(SRC_DIR) $(POETRY) run pytest --cov=$(COV_DIR) --cov-report=html -q

docs: ## Build documentation
	$(Q)$(POETRY) run mkdocs build

docs-serve: ## Serve documentation
	$(Q)$(POETRY) run mkdocs serve

diagnose: ## Project diagnostics
	$(Q)echo "Python: $$(python --version)"
	$(Q)echo "Poetry: $$($(POETRY) --version)"
	$(Q)$(POETRY) env info

doctor: diagnose check ## Health check

deps-update: ## Update dependencies
	$(Q)$(POETRY) update

deps-show: ## Show dependency tree
	$(Q)$(POETRY) show --tree

# Override validate to include complexity and docstrings
validate: lint format-check type-check complexity docstring-check security test ## Full validation

.DEFAULT_GOAL := help
