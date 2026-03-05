# =============================================================================
# flext-core - Foundation Library
# =============================================================================
# Prefer workspace base.mk when present; fallback to local generated template.
# This file adds project-level automation wrappers around flext-infra tooling.
# =============================================================================

PROJECT_NAME := flext-core
PYTHON_VERSION ?= 3.13
SRC_DIR ?= src
TESTS_DIR ?= tests
FLEXT_MODERNIZE_FLAGS ?= --skip-check
ifneq ("$(wildcard ../base.mk)", "")
include ../base.mk
else
include base.mk
endif

FLEXT_INFRA_PYTHON = PYTHONPATH=$(CURDIR)/src $(POETRY) run python

# -----------------------------------------------------------------------------
# Project-specific convenience targets (on top of standardized FLEXT verbs)
# -----------------------------------------------------------------------------
.PHONY: docs-serve diagnose doctor deps-update deps-show validate-typings precommit quick-check ci ci-fast activate tooling-sync setup-project

docs-serve: ## Serve MkDocs locally
	$(Q)$(POETRY) run mkdocs serve

diagnose: ## Show local runtime and Poetry diagnostics
	$(Q)echo "Project: $(PROJECT_NAME)"
	$(Q)echo "Python: $$($(POETRY) run python --version)"
	$(Q)echo "Poetry: $$($(POETRY) --version)"
	$(Q)$(POETRY) env info

activate: ## Pre-activate environment for flext-infra automations
	$(Q)$(POETRY) install --all-extras --all-groups
	$(Q)if git rev-parse --git-dir >/dev/null 2>&1; then \
		$(POETRY) run pre-commit install; \
	else \
		echo "INFO: skipping pre-commit install (no git repository)"; \
	fi

tooling-sync: ## Regenerate pyproject/base.mk through flext-infra generators
	$(Q)$(FLEXT_INFRA_PYTHON) -m flext_infra deps modernize $(FLEXT_MODERNIZE_FLAGS)
	$(Q)$(FLEXT_INFRA_PYTHON) -m flext_infra basemk generate --project-name $(PROJECT_NAME) --output base.mk

setup-project: ## Recommended setup: activate env + generate files + internal sync
	$(Q)$(MAKE) activate
	$(Q)$(MAKE) tooling-sync
	$(Q)$(FLEXT_INFRA_PYTHON) -m flext_infra deps internal-sync --project-root "$(CURDIR)"
	$(Q)$(POETRY) lock
	$(Q)$(POETRY) install --all-extras --all-groups

precommit: ## Run all pre-commit hooks
	$(Q)$(POETRY) run pre-commit run --all-files

quick-check: ## Fast gates for local development loop
	$(Q)$(MAKE) check CHECK_GATES=lint,format,pyrefly,mypy,pyright

ci-fast: ## CI-style check without docs generation
	$(Q)$(MAKE) check

ci: ## Full CI-style check including docs
	$(Q)$(MAKE) check
	$(Q)$(MAKE) docs DOCS_PHASE=all

# Extended health check for contributors
doctor: ## Project health check
	$(Q)$(MAKE) diagnose
	$(Q)$(MAKE) validate
	$(Q)$(MAKE) test

deps-update: ## Refresh lockfile and dependency graph
	$(Q)$(POETRY) update
	$(Q)$(POETRY) lock

deps-show: ## Show dependency tree
	$(Q)$(POETRY) show --tree

validate-typings: ## Validate TypeAlias syntax rules in typings.py
	$(Q)bash scripts/validate_typings.sh

.DEFAULT_GOAL := help
