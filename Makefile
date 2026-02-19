# flext-core - Foundation Library
PROJECT_NAME := flext-core
ifneq ("$(wildcard ../base.mk)", "")
include ../base.mk
else
include base.mk
endif

# === PROJECT-SPECIFIC TARGETS ===
.PHONY: docs-serve diagnose doctor deps-update deps-show

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

.DEFAULT_GOAL := help
