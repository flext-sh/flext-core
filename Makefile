# flext-core - Foundation Library
PROJECT_NAME := flext-core
ifneq ("$(wildcard ../base.mk)", "")
include ../base.mk
else
include base.mk
endif

# === PROJECT-SPECIFIC TARGETS ===
.PHONY: docs-serve diagnose doctor deps-update deps-show validate-typings

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

validate-typings: ## Validate TypeAlias syntax rules in typings.py
	$(Q)bash scripts/validate_typings.sh
.DEFAULT_GOAL := help
