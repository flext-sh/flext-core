# Private project handlers for flext-core.
# Strict extension: only `_custom_<verb>_<what>` handlers and `(pre|post)-<verb>[-<what>]`
# hooks. Public targets, toolchain vars, .DEFAULT_GOAL, includes, and help are
# invalid (base.mk owns those). Each handler maps to `make <verb> WHAT=<what>`.
# NOTE: legacy activate/setup-project/tooling-sync/precommit/quick-check/ci/
# ci-fast/doctor reimplemented standard verbs (boot/check/test) and were removed;
# base.mk owns them. Toolchain vars and .DEFAULT_GOAL removed.
.PHONY: _custom_run_diagnose _custom_run_deps-update _custom_run_deps-show _custom_run_validate-typings
_custom_run_diagnose: ## make run WHAT=diagnose — local runtime and Poetry diagnostics
	$(Q)echo "Project: $(PROJECT_NAME)"
	$(Q)echo "Python: $$($(POETRY) run python --version)"
	$(Q)echo "Poetry: $$($(POETRY) --version)"
	$(Q)$(POETRY) env info
_custom_run_deps-update: ## make run WHAT=deps-update — refresh lockfile
	$(Q)$(POETRY) update
	$(Q)$(POETRY) lock
_custom_run_deps-show: ## make run WHAT=deps-show — show dependency tree
	$(Q)$(POETRY) show --tree
_custom_run_validate-typings: ## make run WHAT=validate-typings — validate TypeAlias rules
	$(Q)bash scripts/validate_typings.sh
