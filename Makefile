# @flext-managed: continuous
# @flext-regenerate: make codegen WHAT=apply APPLY=Y
# @flext-ssot: flext-infra/config/codegen.yaml + flext-infra/src/flext_infra/templates/project/base/Makefile.j2
# @flext-maintenance: do not edit generated projections; edit the SSOT and regenerate
# flext-core — generated project interface.
# Managed by flext-infra codegen conform for new and existing repositories.

SHELL := /bin/sh
.DEFAULT_GOAL := help

PROJECT_NAME := flext-core
MAKE_PROFILE := workspace-member
WORKSPACE_ROOT_REL := ..
WORKSPACE_MEMBERS :=
WORKSPACE_EDITABLES := $(PROJECT_NAME):.
UV_LINK_MODE := copy
UV_VERSION := 0.11.32
MISE_VERSION := 2026.7.16

APPLY ?= N
ARGS ?=
CHECK_GATES ?=
FAIL_FAST ?= 0
FILE ?=
FIX ?= 0
MATCH ?=
PROJECT ?=
PROJECTS ?=
BASE ?=
BRANCH ?=
# Public selector documented by base.mk. Forwarded to the test recipe so a
# focused run stays inside the canonical Make surface instead of forcing a
# loose pytest invocation.
PYTEST_ARGS ?=
PYTEST_TARGETS ?= $(PROJECT_ROOT)/tests
PYTEST_DIAG_ARGS ?= -rA --durations=0 --tb=long --showlocals
PYTEST_REPORT_ARGS ?= -ra --durations=25 --durations-min=0.001 --tb=short
PYTEST_REPORTS_DIR ?= .reports/tests
WHAT ?=

PROJECT_ROOT := $(shell pwd -P)
SELF_MAKEFILE := $(abspath $(firstword $(MAKEFILE_LIST)))
MAKEFILE_ROOT := $(patsubst %/,%,$(dir $(SELF_MAKEFILE)))
WORKSPACE ?= $(PROJECT_ROOT)
PUBLIC_VERBS := help setup deps build check test fmt run status docs clean release codegen worktree
CHECK_GATES_ALLOWED := lint format pyrefly mypy pyright security markdown smells
CHECK_GATES_DEFAULT := lint format pyrefly mypy pyright security markdown smells
DOCS_ACTIONS := generate fix audit build validate
SERIALIZED_VERBS := check test codegen
SERIALIZED_TARGETS := _serialized_check _serialized_test _serialized_codegen
RUFF_PATHS := $(PROJECT_ROOT)/src $(PROJECT_ROOT)/tests
MYPY_PATHS := $(PROJECT_ROOT)/src $(PROJECT_ROOT)/tests
CALLER_PATH := $(PATH)
CALLER_VIRTUAL_ENV := $(patsubst %/,%,$(VIRTUAL_ENV))
FLEXT_INFRA_BOOTSTRAP_REQUIREMENT := flext-infra @ git+https://github.com/flext-sh/flext-infra.git@0.12.0-dev
FLEXT_INFRA_SOURCE_ROOT_REL := 

# === MYPY RESOURCE LIMIT ===
# mro-0ftd.3.11: every Mypy process inherits validated memory and time caps.
MYPY_MEMORY_LIMIT_MB ?= 6144
MYPY_TIMEOUT_SECONDS ?= 600
MYPY_BOUNDED = timeout --signal=TERM --kill-after=5s "$(MYPY_TIMEOUT_SECONDS)s" prlimit --as=$$(( $(MYPY_MEMORY_LIMIT_MB) * 1024 * 1024 )):$$(( $(MYPY_MEMORY_LIMIT_MB) * 1024 * 1024 )) --
VALIDATE_MYPY_LIMITS = case "$(MYPY_MEMORY_LIMIT_MB)" in ""|*[!0-9]*) echo "ERROR: MYPY_MEMORY_LIMIT_MB must be a positive integer"; exit 2;; esac; [ "$(MYPY_MEMORY_LIMIT_MB)" -gt 0 ] || { echo "ERROR: MYPY_MEMORY_LIMIT_MB must be greater than zero"; exit 2; }; [ "$(MYPY_MEMORY_LIMIT_MB)" -le 6144 ] || { echo "ERROR: MYPY_MEMORY_LIMIT_MB must be less than or equal to 6144"; exit 2; }; case "$(MYPY_TIMEOUT_SECONDS)" in ""|*[!0-9]*) echo "ERROR: MYPY_TIMEOUT_SECONDS must be a positive integer"; exit 2;; esac; [ "$(MYPY_TIMEOUT_SECONDS)" -gt 0 ] || { echo "ERROR: MYPY_TIMEOUT_SECONDS must be greater than zero"; exit 2; }; [ "$(MYPY_TIMEOUT_SECONDS)" -le 600 ] || { echo "ERROR: MYPY_TIMEOUT_SECONDS must be less than or equal to 600"; exit 2; }; command -v timeout >/dev/null 2>&1 || { echo "ERROR: required executable not found: timeout"; exit 2; }; command -v prlimit >/dev/null 2>&1 || { echo "ERROR: required executable not found: prlimit"; exit 2; }
REPORT_MYPY_FAILURE = code=$$?; signal=none; if [ "$$code" -ge 128 ]; then signal=$$(( $$code - 128 )); fi; if [ "$$code" -eq 124 ] || [ "$$signal" != none ]; then reason="resource limit triggered"; else reason="type check failed under enforced limits"; fi; echo "ERROR: Mypy $$reason: memory_limit=$(MYPY_MEMORY_LIMIT_MB) MiB; timeout=$(MYPY_TIMEOUT_SECONDS)s; exit=$$code; signal=$$signal" >&2
export MYPY_MEMORY_LIMIT_MB MYPY_TIMEOUT_SECONDS


_DEFAULT_help := usage
_DEFAULT_deps := check
_DEFAULT_build := artifacts
_DEFAULT_check := all
_DEFAULT_test := all
_DEFAULT_fmt := check
_DEFAULT_run := default
_DEFAULT_status := diagnostics
_DEFAULT_docs := all
_DEFAULT_clean := generated
_DEFAULT_release := status
_DEFAULT_codegen := check
_DEFAULT_worktree := list


ifneq ($(filter $(MAKE_PROFILE),workspace-root workspace-member standalone),$(MAKE_PROFILE))
$(error Invalid MAKE_PROFILE '$(MAKE_PROFILE)')
endif

ifeq ($(MAKE_PROFILE),workspace-member)
DECLARED_WORKSPACE_ROOT := $(shell cd "$(PROJECT_ROOT)/$(WORKSPACE_ROOT_REL)" 2>/dev/null && pwd -P)
SUPERPROJECT_ROOT_RAW := $(shell git rev-parse --show-superproject-working-tree 2>/dev/null)
SUPERPROJECT_ROOT := $(shell test -n "$(SUPERPROJECT_ROOT_RAW)" && cd "$(SUPERPROJECT_ROOT_RAW)" 2>/dev/null && pwd -P)
ifeq ($(SUPERPROJECT_ROOT),$(DECLARED_WORKSPACE_ROOT))
ATTACHED_MEMBER := Y
RUNTIME_ROOT := $(DECLARED_WORKSPACE_ROOT)
else
ATTACHED_MEMBER := N
RUNTIME_ROOT := $(PROJECT_ROOT)
endif
else
ATTACHED_MEMBER := N
RUNTIME_ROOT := $(PROJECT_ROOT)
endif

RUNTIME_VENV := $(RUNTIME_ROOT)/.venv
ifeq ($(MAKE_PROFILE),standalone)
TOOLS_BIN := $(PROJECT_ROOT)/.bin
MISE_DATA_DIR := $(PROJECT_ROOT)/.tools
else
TOOLS_BIN := $(RUNTIME_ROOT)/.bin
MISE_DATA_DIR := $(RUNTIME_ROOT)/.tools
endif
MISE_SHIMS := $(MISE_DATA_DIR)/shims
FLEXT_INFRA_RUNTIME_ROOT := $(if $(filter $(MAKEFILE_ROOT),$(PROJECT_ROOT)),$(RUNTIME_ROOT),$(MAKEFILE_ROOT))
ifeq ($(OS),Windows_NT)
MISE ?= $(TOOLS_BIN)/mise.exe
RUNTIME_BIN := $(RUNTIME_VENV)/Scripts
RUNTIME_PYTHON := $(RUNTIME_BIN)/python.exe
FLEXT_INFRA_RUNTIME_PYTHON := $(FLEXT_INFRA_RUNTIME_ROOT)/.venv/Scripts/python.exe
NORMALIZED_CALLER_PATH := $(shell cygpath --path "$(CALLER_PATH)" 2>/dev/null)
NORMALIZED_CALLER_VIRTUAL_ENV := $(shell cygpath --unix "$(CALLER_VIRTUAL_ENV)" 2>/dev/null)
CALLER_VIRTUAL_ENV_BIN := $(NORMALIZED_CALLER_VIRTUAL_ENV)/Scripts
else
MISE ?= $(TOOLS_BIN)/mise
RUNTIME_BIN := $(RUNTIME_VENV)/bin
RUNTIME_PYTHON := $(RUNTIME_BIN)/python
FLEXT_INFRA_RUNTIME_PYTHON := $(FLEXT_INFRA_RUNTIME_ROOT)/.venv/bin/python
NORMALIZED_CALLER_PATH := $(CALLER_PATH)
NORMALIZED_CALLER_VIRTUAL_ENV := $(CALLER_VIRTUAL_ENV)
CALLER_VIRTUAL_ENV_BIN := $(NORMALIZED_CALLER_VIRTUAL_ENV)/bin
endif
SANITIZED_CALLER_PATH := $(NORMALIZED_CALLER_PATH)
ifneq ($(strip $(NORMALIZED_CALLER_VIRTUAL_ENV)),)
SANITIZED_CALLER_PATH := $(subst $(CALLER_VIRTUAL_ENV_BIN):,,$(SANITIZED_CALLER_PATH))
SANITIZED_CALLER_PATH := $(subst :$(CALLER_VIRTUAL_ENV_BIN),,$(SANITIZED_CALLER_PATH))
ifeq ($(SANITIZED_CALLER_PATH),$(CALLER_VIRTUAL_ENV_BIN))
SANITIZED_CALLER_PATH :=
endif
endif
override UV := $(MISE_SHIMS)/uv
override FLEXT_INFRA_PYTHON := $(FLEXT_INFRA_RUNTIME_PYTHON)
override UV_PROJECT := $(RUNTIME_ROOT)
override UV_PROJECT_ENVIRONMENT := $(RUNTIME_VENV)
override VIRTUAL_ENV := $(RUNTIME_VENV)
override PATH := $(RUNTIME_BIN):$(TOOLS_BIN):$(MISE_SHIMS):$(SANITIZED_CALLER_PATH)
export FLEXT_INFRA_PYTHON MISE_DATA_DIR UV UV_PROJECT UV_PROJECT_ENVIRONMENT VIRTUAL_ENV PATH

ifneq ($(strip $(FLEXT_INFRA_SOURCE_ROOT_REL)),)
FLEXT_INFRA_SOURCE_ROOT := $(abspath $(PROJECT_ROOT)/$(FLEXT_INFRA_SOURCE_ROOT_REL))
FLEXT_INFRA_BOOTSTRAP := env -u PYTHONPATH -u MYPYPATH -u VIRTUAL_ENV -u UV_PROJECT -u UV_PROJECT_ENVIRONMENT PATH="$(TOOLS_BIN):$(MISE_SHIMS):$(SANITIZED_CALLER_PATH)" $(UV) run --no-project --with-editable "$(FLEXT_INFRA_SOURCE_ROOT)" python -m flext_infra
else
FLEXT_INFRA_SOURCE_ROOT :=
FLEXT_INFRA_BOOTSTRAP := env -u PYTHONPATH -u MYPYPATH -u VIRTUAL_ENV -u UV_PROJECT -u UV_PROJECT_ENVIRONMENT PATH="$(TOOLS_BIN):$(MISE_SHIMS):$(SANITIZED_CALLER_PATH)" $(UV) run --no-project --with "$(FLEXT_INFRA_BOOTSTRAP_REQUIREMENT)" python -m flext_infra
endif

ifeq ($(MAKE_PROFILE),workspace-root)
CODEGEN_SCOPE := all
ALLOWED_PROJECTS := . $(WORKSPACE_MEMBERS)
else
CODEGEN_SCOPE := self
ALLOWED_PROJECTS := .
endif

# Workspace-root gate verbs fan out across declared members through the generic
# `flext-infra workspace orchestrate` primitive (verb allowlist + CLI group come
# from the constants SSOT, never hardcoded here). Members and standalone projects
# run the gate locally. FAIL_FAST forwards the stop-on-first-failure policy.
WORKSPACE_ORCHESTRATE = $(UV_RUN) python -m flext_infra workspace orchestrate
REQUESTED_PROJECTS := $(strip $(if $(PROJECT),$(PROJECT),$(PROJECTS)))
FILE_MEMBER := $(firstword $(foreach member,$(WORKSPACE_MEMBERS),$(if $(filter $(member)/%,$(FILE)),$(member))))
FILE_PROJECT := $(if $(strip $(FILE_MEMBER)),$(FILE_MEMBER),.)
FILE_RELATIVE := $(if $(filter .,$(FILE_PROJECT)),$(FILE),$(patsubst $(FILE_PROJECT)/%,%,$(FILE)))
DEFAULT_PROJECTS := $(WORKSPACE_MEMBERS) .
SELECTED_PROJECTS := $(if $(strip $(FILE)),$(FILE_PROJECT),$(if $(strip $(REQUESTED_PROJECTS)),$(REQUESTED_PROJECTS),$(DEFAULT_PROJECTS)))
WORKSPACE_PROJECT_ARGS := $(foreach project,$(SELECTED_PROJECTS),--projects $(project))
WORKSPACE_CHECK_ARGS := $(if $(strip $(CHECK_GATES)),--make-arg "CHECK_GATES=$(strip $(CHECK_GATES))")
WORKSPACE_TEST_ARGS := $(if $(strip $(FILE)),--make-arg "FILE=$(FILE_RELATIVE)") $(if $(strip $(MATCH)),--make-arg "MATCH=$(MATCH)") $(if $(strip $(PYTEST_ARGS)),--make-arg "PYTEST_ARGS=$(strip $(PYTEST_ARGS))")
DOCS_PROJECT_ARGS := $(foreach project,$(REQUESTED_PROJECTS),--projects $(project))
ORCHESTRATED_VERBS := build check clean docs scan test val

UV_RUN := env -u PYTHONPATH -u MYPYPATH $(UV) run --project "$(RUNTIME_ROOT)" --no-sync
PROJECT_INFRA_PYTHONPATH ?= $(MAKEFILE_ROOT)/src
PROJECT_FLEXT_INFRA := test -x "$(FLEXT_INFRA_PYTHON)" || { printf 'ERROR: FLEXT_INFRA_PYTHON must name an executable managed Python\n' >&2; exit 2; }; env -u PYTHONPATH -u MYPYPATH -u VIRTUAL_ENV -u UV_PROJECT -u UV_PROJECT_ENVIRONMENT PATH="$(dir $(FLEXT_INFRA_PYTHON)):$(MISE_SHIMS):$(SANITIZED_CALLER_PATH)" PYTHONPATH="$(PROJECT_INFRA_PYTHONPATH)" $(FLEXT_INFRA_PYTHON) -m flext_infra
# mro-j47u (codex): scaffold dev tools live in the validated optional dev
# profile; a fresh project must create its lock before later check-mode locks.
UV_SYNC_FLAGS := --all-extras --all-groups

ifneq ($(strip $(PROJECT)),)
ifneq ($(strip $(PROJECTS)),)
$(error ERROR: Cannot use PROJECT and PROJECTS together)
endif
endif


-include custom.mk

_BUILTIN_HANDLERS := \
	_builtin_help_usage \
	_builtin_deps_check \
	_builtin_deps_lock \
	_builtin_deps_upgrade \
	_builtin_build_artifacts \
	_builtin_check_all \
	_builtin_test_all \
	_builtin_fmt_check \
	_builtin_fmt_apply \
	_builtin_run_default \
	_builtin_status_diagnostics \
	_builtin_docs_all \
_builtin_docs_generate \
_builtin_docs_fix \
_builtin_docs_audit \
_builtin_docs_build \
_builtin_docs_validate \
_builtin_clean_generated \
	_builtin_release_status \
	_builtin_codegen_check \
	_builtin_codegen_apply \
	_builtin_worktree_list \
	_builtin_worktree_add \
	_builtin_worktree_update \
	_builtin_worktree_remove

SELF_MAKE := $(MAKE) --no-print-directory -f "$(SELF_MAKEFILE)"

define _dispatch
	@what="$(strip $(WHAT))"; \
	if [ -z "$$what" ]; then what="$(_DEFAULT_$(1))"; fi; \
	case "$$what" in \
		*[!a-z0-9_-]*|'') printf 'ERROR: invalid WHAT selector %s\n' "$$what" >&2; exit 2 ;; \
	esac; \
	builtin="_builtin_$(1)_$$what"; \
	custom="_custom_$(1)_$$what"; \
	for hook in "pre-$(1)" "pre-$(1)-$$what"; do \
		$(SELF_MAKE) -q "$$hook" >/dev/null 2>&1; rc=$$?; \
		if [ "$$rc" -ne 2 ]; then $(SELF_MAKE) "$$hook" || exit $$?; fi; \
	done; \
	case " $(_BUILTIN_HANDLERS) " in \
		*" $$builtin "*) $(SELF_MAKE) "$$builtin" || exit $$? ;; \
		*) $(SELF_MAKE) "$$custom" || exit $$? ;; \
	esac; \
	for hook in "post-$(1)-$$what" "post-$(1)"; do \
		$(SELF_MAKE) -q "$$hook" >/dev/null 2>&1; rc=$$?; \
		if [ "$$rc" -ne 2 ]; then $(SELF_MAKE) "$$hook" || exit $$?; fi; \
	done
endef

define _require_apply
	@if [ "$(APPLY)" != "Y" ]; then \
		printf 'ERROR: this action requires APPLY=Y\n' >&2; \
		exit 2; \
	fi
endef

define _run_for_selected_projects
	@set -eu; \
	selected="$(strip $(if $(PROJECT),$(PROJECT),$(PROJECTS)))"; \
	if [ -z "$$selected" ]; then selected="."; fi; \
	for project in $$selected; do \
		case " $(ALLOWED_PROJECTS) " in \
			*" $$project "*) ;; \
			*) printf 'ERROR: undeclared project %s\n' "$$project" >&2; exit 2 ;; \
		esac; \
		$(UV) lock --project "$(PROJECT_ROOT)/$$project" $(1); \
	done
endef

.PHONY: $(PUBLIC_VERBS) $(SERIALIZED_TARGETS) $(_BUILTIN_HANDLERS)

$(filter-out setup $(SERIALIZED_VERBS),$(PUBLIC_VERBS)):
	$(call _dispatch,$@)


check: _builtin_require_environment
	@$(PROJECT_FLEXT_INFRA) workspace serialize-make --workspace "$(PROJECT_ROOT)" --makefile "$(SELF_MAKEFILE)" --verb "check"

_serialized_check:
	$(call _dispatch,check)


test: _builtin_require_environment
	@$(PROJECT_FLEXT_INFRA) workspace serialize-make --workspace "$(PROJECT_ROOT)" --makefile "$(SELF_MAKEFILE)" --verb "test"

_serialized_test:
	$(call _dispatch,test)


codegen: _builtin_require_environment
	@$(PROJECT_FLEXT_INFRA) workspace serialize-make --workspace "$(PROJECT_ROOT)" --makefile "$(SELF_MAKEFILE)" --verb "codegen"

_serialized_codegen:
	$(call _dispatch,codegen)



setup:
	@if [ -n "$(strip $(WHAT))" ]; then printf 'ERROR: setup does not accept WHAT\n' >&2; exit 2; fi
	@$(SELF_MAKE) _builtin_setup_environment

_builtin_help_usage:
	@printf '%s\n' 'flext-core [workspace-member]' '';


	@printf '  %-10s WHAT=%s\n' 'help' 'usage';



	@printf '  %-10s\n' 'setup';



	@printf '  %-10s WHAT=%s\n' 'deps' 'check';



	@printf '  %-10s WHAT=%s\n' 'build' 'artifacts';



	@printf '  %-10s WHAT=%s\n' 'check' 'all';



	@printf '  %-10s WHAT=%s\n' 'test' 'all';



	@printf '  %-10s WHAT=%s APPLY=Y\n' 'fmt' 'check';



	@printf '  %-10s WHAT=%s\n' 'run' 'default';



	@printf '  %-10s WHAT=%s\n' 'status' 'diagnostics';



	@printf '  %-10s WHAT=%s\n' 'docs' 'all|generate|fix|audit|build|validate';



	@printf '  %-10s WHAT=%s\n' 'clean' 'generated';



	@printf '  %-10s WHAT=%s\n' 'release' 'status';



	@printf '  %-10s WHAT=%s APPLY=Y\n' 'codegen' 'check';



	@printf '  %-10s WHAT=%s\n' 'worktree' 'list';


	@printf '  %-10s %s\n' 'WORKSPACE' 'target repository (default: current project)';
	@printf '  %-10s %s\n' 'BASE' 'required for worktree add/update';
	@printf '\n%s\n' 'Custom hooks (custom.mk):';
	@printf '  %s\n' 'Define pre-<verb>, post-<verb>, pre-<verb>-<what>, post-<verb>-<what>';
	@printf '  %s\n' 'in custom.mk to run extra steps at the start or end of any verb,';
	@printf '  %s\n' 'for all or some WHATs. Add _custom_<verb>_<what> to define a new WHAT.';
	@if [ -f custom.mk ]; then \
		hooks=$$(grep -oE '^(pre|post)-[a-z][a-z0-9-]*|^_custom_[a-z][a-z0-9_-]*' custom.mk 2>/dev/null | sort -u); \
		if [ -n "$$hooks" ]; then \
			printf '  %s\n' 'Defined in this project:'; \
			for hook in $$hooks; do printf '    %s\n' "$$hook"; done; \
		fi; \
	fi

# A project owns the sources declared by its manifest. The generated setup
# reconciler validates every initialized checkout before mutation, initializes
# only missing modules, and preserves declared branches that fix forward beyond
# the recorded gitlink.
.PHONY: _builtin_setup_tools

_builtin_setup_tools:
	@set -eu; \
	tools_bin="$(TOOLS_BIN)"; \
	mise="$(MISE)"; \
	mise_data_dir="$(MISE_DATA_DIR)"; \
	mise_version="$(MISE_VERSION)"; \
	uv_version="$(UV_VERSION)"; \
	current=""; \
	if [ -x "$$mise" ]; then \
		current=$$("$$mise" --version 2>/dev/null | cut -d ' ' -f 1 || true); \
	fi; \
	if [ "$$current" != "$$mise_version" ]; then \
		command -v curl >/dev/null 2>&1 || { \
			printf 'ERROR: make setup requires curl to bootstrap project binaries\n' >&2; \
			exit 2; \
		}; \
		case "$$(uname -s)" in \
			Linux*) os=linux ;; \
			Darwin*) os=macos ;; \
			MINGW*|MSYS*|CYGWIN*|Windows_NT) os=windows ;; \
			*) printf 'ERROR: unsupported bootstrap OS: %s\n' "$$(uname -s)" >&2; exit 2 ;; \
		esac; \
		case "$$(uname -m)" in \
			x86_64|amd64) arch=x64 ;; \
			aarch64|arm64) arch=arm64 ;; \
			*) printf 'ERROR: unsupported bootstrap architecture: %s\n' "$$(uname -m)" >&2; exit 2 ;; \
		esac; \
		suffix="$$os-$$arch"; extension=""; \
		if [ "$$os" = linux ] && ldd --version 2>&1 | grep -qi musl; then \
			suffix="$$suffix-musl"; \
		fi; \
		if [ "$$os" = windows ]; then extension=.exe; fi; \
		asset="mise-v$$mise_version-$$suffix$$extension"; \
		url="https://github.com/jdx/mise/releases/download/v$$mise_version/$$asset"; \
		mkdir -p "$$tools_bin" "$$mise_data_dir"; \
		tmp="$$mise.download"; \
		if ! curl -fsSL "$$url" -o "$$tmp"; then rm -f "$$tmp"; exit 2; fi; \
		chmod +x "$$tmp"; \
		mv "$$tmp" "$$mise"; \
		current=$$("$$mise" --version | cut -d ' ' -f 1); \
		[ "$$current" = "$$mise_version" ] || { \
			printf 'ERROR: mise bootstrap version mismatch: expected %s, got %s\n' \
				"$$mise_version" "$$current" >&2; \
			exit 2; \
		}; \
	fi; \
	$(MISE) install --yes "uv@$$uv_version"

.PHONY: _builtin_setup_submodules

_builtin_setup_submodules:
	@set -eu; \
	root="$(PROJECT_ROOT)"; \
	selected=""; \
	if [ ! -f "$$root/.gitmodules" ]; then exit 0; fi; \
	setup_selected() { \
		candidate="$$1"; \
		if [ -z "$$selected" ]; then return 0; fi; \
		for project in $$selected; do \
			if [ "$$project" = "." ]; then continue; fi; \
			case "$$project" in \
				"$$candidate"|"$$candidate"/*) return 0 ;; \
			esac; \
		done; \
		return 1; \
	}; \
	preflight_managed_submodules() { \
		superproject="$$1"; \
		if [ ! -f "$$superproject/.gitmodules" ]; then return 0; fi; \
		git -C "$$superproject" config -f .gitmodules --get-regexp '^submodule\..*\.path$$' 2>/dev/null | \
		while IFS=' ' read -r path_key child_path; do \
			if [ "$$superproject" = "$$root" ] && ! setup_selected "$$child_path"; then continue; fi; \
			prefix=$${path_key%.path}; \
			managed=$$(git -C "$$superproject" config -f .gitmodules --bool --get --default false "$$prefix.flext-managed"); \
			if [ "$$managed" != true ]; then continue; fi; \
			state=$$(git -C "$$superproject" submodule status -- "$$child_path"); \
			case "$$state" in -*) continue ;; esac; \
			checkout="$$superproject/$$child_path"; \
			displaypath=$${checkout#"$$root"/}; \
			sha1=$$(git -C "$$superproject" rev-parse "HEAD:$$child_path"); \
			branch=$$(git -C "$$superproject" config -f .gitmodules --get --default "" "$$prefix.branch"); \
			current=$$(git -C "$$checkout" branch --show-current); \
			if [ -z "$$branch" ]; then \
				if [ -n "$$current" ]; then \
					printf "ERROR: %s: branch %s is checked out but .gitmodules declares no branch\n" "$$displaypath" "$$current" >&2; \
					exit 1; \
				fi; \
			elif [ "$$branch" = "." ]; then \
				branch=$$(git -C "$$superproject" branch --show-current); \
				if [ -z "$$branch" ]; then \
					printf "ERROR: %s: branch = . requires its superproject on a named branch\n" "$$displaypath" >&2; \
					exit 1; \
				fi; \
			fi; \
			if [ -n "$$branch" ]; then \
				git check-ref-format --branch "$$branch" >/dev/null || { \
					printf "ERROR: %s: invalid declared branch %s\n" "$$displaypath" "$$branch" >&2; \
					exit 1; \
				}; \
			fi; \
			if [ -n "$$current" ] && [ "$$current" != "$$branch" ]; then \
				printf "ERROR: %s: conflicting branch %s; expected %s\n" "$$displaypath" "$$current" "$$branch" >&2; \
				exit 1; \
			fi; \
			if [ -z "$$current" ] && [ "$$(git -C "$$checkout" rev-parse HEAD)" != "$$sha1" ]; then \
				printf "ERROR: %s: detached HEAD diverges from recorded gitlink %s\n" "$$displaypath" "$$(git -C "$$checkout" rev-parse --short "$$sha1")" >&2; \
				exit 1; \
			fi; \
			if [ -n "$$current" ] && ! git -C "$$checkout" merge-base --is-ancestor "$$sha1" HEAD; then \
				printf "ERROR: %s: branch %s diverges from recorded gitlink %s\n" "$$displaypath" "$$branch" "$$(git -C "$$checkout" rev-parse --short "$$sha1")" >&2; \
				exit 1; \
			fi; \
			preflight_managed_submodules "$$checkout"; \
		done; \
	}; \
	initialize_declared_submodules() { \
		superproject="$$1"; \
		if [ ! -f "$$superproject/.gitmodules" ]; then return 0; fi; \
		git -C "$$superproject" config -f .gitmodules --get-regexp '^submodule\..*\.path$$' 2>/dev/null | \
		while IFS=' ' read -r path_key child_path; do \
			if [ "$$superproject" = "$$root" ] && ! setup_selected "$$child_path"; then continue; fi; \
			prefix=$${path_key%.path}; \
			managed=$$(git -C "$$superproject" config -f .gitmodules --bool --get --default false "$$prefix.flext-managed"); \
			if [ "$$managed" != true ]; then continue; fi; \
			git -C "$$superproject" submodule sync --quiet -- "$$child_path"; \
			state=$$(git -C "$$superproject" submodule status -- "$$child_path"); \
			case "$$state" in \
				-*) git -C "$$superproject" submodule update --init --recursive -- "$$child_path" ;; \
			esac; \
			initialize_declared_submodules "$$superproject/$$child_path"; \
		done; \
	}; \
	reconcile_managed_submodules() { \
		superproject="$$1"; \
		if [ ! -f "$$superproject/.gitmodules" ]; then return 0; fi; \
		git -C "$$superproject" config -f .gitmodules --get-regexp '^submodule\..*\.path$$' 2>/dev/null | \
		while IFS=' ' read -r path_key child_path; do \
			if [ "$$superproject" = "$$root" ] && ! setup_selected "$$child_path"; then continue; fi; \
			prefix=$${path_key%.path}; \
			managed=$$(git -C "$$superproject" config -f .gitmodules --bool --get --default false "$$prefix.flext-managed"); \
			if [ "$$managed" != true ]; then continue; fi; \
			checkout="$$superproject/$$child_path"; \
			displaypath=$${checkout#"$$root"/}; \
			sha1=$$(git -C "$$superproject" rev-parse "HEAD:$$child_path"); \
			branch=$$(git -C "$$superproject" config -f .gitmodules --get --default "" "$$prefix.branch"); \
			if [ -n "$$branch" ]; then \
				if [ "$$branch" = "." ]; then \
					branch=$$(git -C "$$superproject" branch --show-current); \
					if [ -z "$$branch" ]; then \
						printf "ERROR: %s: branch = . requires its superproject on a named branch\n" "$$displaypath" >&2; \
						exit 1; \
					fi; \
				fi; \
				git check-ref-format --branch "$$branch" >/dev/null || { \
					printf "ERROR: %s: invalid declared branch %s\n" "$$displaypath" "$$branch" >&2; \
					exit 1; \
				}; \
				current=$$(git -C "$$checkout" branch --show-current); \
				if [ "$$current" = "$$branch" ]; then \
					:; \
				elif [ -n "$$current" ]; then \
					printf "ERROR: %s: conflicting branch %s; expected %s\n" "$$displaypath" "$$current" "$$branch" >&2; \
					exit 1; \
				elif ! git -C "$$checkout" rev-parse --verify --quiet "refs/heads/$$branch" >/dev/null; then \
					git -C "$$checkout" checkout --quiet -b "$$branch"; \
				elif git -C "$$checkout" merge-base --is-ancestor "$$sha1" "refs/heads/$$branch"; then \
					git -C "$$checkout" checkout --quiet "$$branch"; \
				else \
					printf "ERROR: %s: branch %s diverges from recorded gitlink %s\n" "$$displaypath" "$$branch" "$$(git -C "$$checkout" rev-parse --short "$$sha1")" >&2; \
					exit 1; \
				fi; \
			fi; \
			reconcile_managed_submodules "$$checkout"; \
		done; \
	}; \
	preflight_managed_submodules "$$root"; \
	initialize_declared_submodules "$$root"; \
	reconcile_managed_submodules "$$root"

_builtin_require_environment:
	@if [ ! -x "$(RUNTIME_PYTHON)" ]; then \
		printf 'ERROR: missing environment interpreter %s; make setup creates it\n' "$(RUNTIME_PYTHON)" >&2; \
		exit 2; \
	fi

ifeq ($(MAKE_PROFILE),workspace-root)
_builtin_setup_topology: _builtin_setup_tools
	@$(FLEXT_INFRA_BOOTSTRAP) codegen conform --root "$(PROJECT_ROOT)" --scope "self" --mode apply

_builtin_setup_conform: _builtin_setup_topology _builtin_setup_submodules
	@$(FLEXT_INFRA_BOOTSTRAP) codegen conform --root "$(PROJECT_ROOT)" --scope "$(CODEGEN_SCOPE)" --mode apply
else
_builtin_setup_conform: _builtin_setup_tools _builtin_setup_submodules
	@$(FLEXT_INFRA_BOOTSTRAP) codegen conform --root "$(PROJECT_ROOT)" --scope "$(CODEGEN_SCOPE)" --mode apply
endif

_builtin_setup_mise: _builtin_setup_conform
	@$(MISE) install --yes

ifeq ($(MAKE_PROFILE),workspace-root)
_builtin_setup_environment: _builtin_setup_mise
	@$(UV) sync --project "$(PROJECT_ROOT)" $(UV_SYNC_FLAGS) --link-mode "$(UV_LINK_MODE)"
	@$(UV) pip check --python "$(RUNTIME_VENV)"
else ifeq ($(MAKE_PROFILE),workspace-member)
ifeq ($(ATTACHED_MEMBER),Y)
_builtin_setup_environment: _builtin_setup_mise
	@$(SELF_MAKE) -C "$(RUNTIME_ROOT)" _builtin_setup_environment
else
_builtin_setup_environment: _builtin_setup_mise
	@$(UV) sync --project "$(PROJECT_ROOT)" $(UV_SYNC_FLAGS) --link-mode "$(UV_LINK_MODE)"
endif
else
_builtin_setup_environment: _builtin_setup_mise
	@$(UV) sync --project "$(PROJECT_ROOT)" $(UV_SYNC_FLAGS) --link-mode "$(UV_LINK_MODE)"
endif

_builtin_deps_check: _builtin_require_environment
	$(call _run_for_selected_projects,--check)

_builtin_deps_lock:
	$(call _require_apply)
	$(call _run_for_selected_projects,)

_builtin_deps_upgrade: _builtin_require_environment
	$(call _require_apply)
	$(call _run_for_selected_projects,--upgrade)
	@set -eu; \
	selected="$(strip $(PROJECTS))"; \
	if [ -z "$$selected" ]; then selected="."; fi; \
	set --; \
	for project in $$selected; do set -- "$$@" --projects "$$project"; done; \
	$(PROJECT_FLEXT_INFRA) deps modernize --workspace "$(PROJECT_ROOT)" \
		--apply --rewrite-constraints --skip-check "$$@"
	$(call _run_for_selected_projects,)


_builtin_build_artifacts:
	@$(UV) build --project "$(PROJECT_ROOT)"

_builtin_check_all: _builtin_require_environment
	@set -eu; \
	if [ "$(FIX)" = "1" ] && [ "$(APPLY)" != "Y" ]; then \
		printf 'ERROR: FIX=1 requires APPLY=Y\n' >&2; exit 2; \
	fi; \
	gates="$(strip $(CHECK_GATES))"; \
	if [ -z "$$gates" ]; then gates="$$(printf '%s' '$(CHECK_GATES_DEFAULT)' | tr ' ' ',')"; fi; \
	gates="$$(printf '%s' "$$gates" | tr -d '[:space:]')"; \
	for gate in $$(printf '%s' "$$gates" | tr ',' ' '); do \
		case " $(CHECK_GATES_ALLOWED) " in *" $$gate "*) ;; \
			*) printf 'ERROR: unknown CHECK_GATES value: %s (allowed: %s)\n' "$$gate" "$(CHECK_GATES_ALLOWED)" >&2; exit 2 ;; \
		esac; \
	done; \
	$(PROJECT_FLEXT_INFRA) check run --workspace "$(PROJECT_ROOT)" --gates "$$gates" --projects . $(if $(filter 1,$(FIX)),--fix)

_builtin_test_all: _builtin_require_environment

	@_files="$(strip $(FILES))"; \
	if [ -n "$(FILE)" ]; then \
		case "$(FILE)" in /*|..|../*|*/../*|*/..) \
			printf 'ERROR: FILE must be a repository-relative path\n' >&2; exit 2 ;; \
		esac; \
		if [ -n "$$_files" ]; then _files="$$_files $(FILE)"; else _files="$(FILE)"; fi; \
	fi; \
	_pytest_run="$(PYTEST_TARGETS)"; \
	if [ -n "$$_files" ]; then _pytest_run="$$_files"; fi; \
	for target in $$_pytest_run; do \
		if [ ! -e "$$target" ]; then \
			printf 'ERROR: test target does not exist: %s\n' "$$target" >&2; exit 2; \
		fi; \
	done; \
	_all_pytest_args="$(PYTEST_ARGS)"; \
	if [ -n "$(MATCH)" ]; then _all_pytest_args="$$_all_pytest_args -k $(MATCH)"; fi; \
	if [ "$(FAIL_FAST)" = "1" ]; then _all_pytest_args="$$_all_pytest_args -x"; fi; \
	if [ "$(VERBOSE)" = "1" ]; then _all_pytest_args="$$_all_pytest_args -vv -s"; fi; \
	run_id=$$(date -u +%Y%m%dT%H%M%SZ)-$$$$; \
	report_dir="$(PYTEST_REPORTS_DIR)/$$run_id"; \
	mkdir -p "$$report_dir"; \
	log_file="$$report_dir/pytest.log"; \
	junit_file="$$report_dir/junit.xml"; \
	coverage_file="$$report_dir/coverage.xml"; \
	summary_file="$$report_dir/summary.txt"; \
	failed_file="$$report_dir/failed-tests.txt"; \
	errors_file="$$report_dir/errors.txt"; \
	warnings_file="$$report_dir/warnings.txt"; \
	slowest_file="$$report_dir/slowest-tests.txt"; \
	skips_file="$$report_dir/skipped-tests.txt"; \
	command_file="$$report_dir/command.txt"; \
	_coverage_args="--cov --cov-report=xml:$$coverage_file"; \
	_coverage_required=1; \
	_coverage_value="$$coverage_file"; \
	if [ -n "$$_files" ] || [ -n "$(MATCH)" ] || \
		[ "$$_pytest_run" != "$(PROJECT_ROOT)/tests" ]; then \
		_coverage_args="--no-cov"; \
		_coverage_required=0; \
		_coverage_value="not-generated"; \
	fi; \
	printf '%s\n' '$(UV_RUN) python -m pytest' \
		"$$_pytest_run $(PYTEST_REPORT_ARGS) -p no:metadata --junitxml=$$junit_file" \
		"$$_coverage_args $$_all_pytest_args" > "$$command_file"; \
	$(UV_RUN) python -m pytest $$_pytest_run \
		$(PYTEST_REPORT_ARGS) \
		$(if $(filter 1,$(DIAG)),$(PYTEST_DIAG_ARGS),) \
		-p no:metadata \
		--junitxml="$$junit_file" \
		$$_coverage_args \
		$(if $(filter 1,$(DIAG)),-vv,-q) $$_all_pytest_args > "$$log_file" 2>&1; \
	rc=$$?; \
	cat "$$log_file"; \
	if [ "$$_coverage_required" -eq 1 ] && [ ! -s "$$coverage_file" ]; then \
		printf 'ERROR: coverage report was not generated or is empty: %s\n' \
			"$$coverage_file" >&2; \
		if [ "$$rc" -eq 0 ]; then rc=2; fi; \
	fi; \
	if [ -f "$$junit_file" ]; then \
		tests=$$(grep -Eo 'tests="[0-9]+"' "$$junit_file" | head -n 1 | tr -dc '0-9'); \
		failures=$$(grep -Eo 'failures="[0-9]+"' "$$junit_file" | head -n 1 | tr -dc '0-9'); \
		errors=$$(grep -Eo 'errors="[0-9]+"' "$$junit_file" | head -n 1 | tr -dc '0-9'); \
		skipped=$$(grep -Eo 'skipped="[0-9]+"' "$$junit_file" | head -n 1 | tr -dc '0-9'); \
		duration=$$(grep -Eo 'time="[0-9.]+"' "$$junit_file" | head -n 1 | sed -E 's/time="([0-9.]+)"/\1/'); \
		tests=$${tests:-0}; failures=$${failures:-0}; errors=$${errors:-0}; \
		skipped=$${skipped:-0}; duration=$${duration:-0}; \
		passed=$$((tests - failures - errors - skipped)); \
		if [ "$$passed" -lt 0 ]; then passed=0; fi; \
		printf 'junit=%s\ncoverage=%s\ntotal=%s\npassed=%s\nfailed=%s\nerrors=%s\nskipped=%s\nduration_seconds=%s\n' \
			"$$junit_file" "$$_coverage_value" "$$tests" "$$passed" "$$failures" \
			"$$errors" "$$skipped" "$$duration" > "$$summary_file"; \
	else \
		printf 'junit=not-generated\ncoverage=%s\ntotal=0\npassed=0\nfailed=0\nerrors=0\nskipped=0\nduration_seconds=0\n' \
			"$$_coverage_value" > "$$summary_file"; \
	fi; \
	counts_file="$$report_dir/counts.env"; \
	if $(PROJECT_FLEXT_INFRA) validate pytest-diag \
		--junit "$$junit_file" --log "$$log_file" \
		--failed "$$failed_file" --errors "$$errors_file" \
		--warnings "$$warnings_file" --slowest "$$slowest_file" \
		--skips "$$skips_file" > "$$counts_file"; then \
		:; \
	else \
		counts_status=$$?; \
		printf 'ERROR: pytest diagnostic extraction failed (exit=%s)\n' \
			"$$counts_status" >&2; \
		cat "$$counts_file" >&2; \
		exit "$$counts_status"; \
	fi; \
	if ! awk ' \
		BEGIN { required["failed_count"]; required["error_count"]; required["warning_count"]; required["skipped_count"] } \
		$$0 !~ /^(failed_count|error_count|warning_count|skipped_count)=[0-9]+$$/ { invalid=1; next } \
		{ split($$0, fields, "="); if (seen[fields[1]]++) invalid=1 } \
		END { if (NR != 4) invalid=1; for (key in required) if (seen[key] != 1) invalid=1; exit invalid } \
	' "$$counts_file"; then \
		echo "ERROR: invalid pytest diagnostic counts contract" >&2; \
		cat "$$counts_file" >&2; \
		exit 2; \
	fi; \
	. "$$counts_file"; \
	if [ "$${failed_count:-0}" -gt 0 ] || [ "$${error_count:-0}" -gt 0 ] || \
		[ "$${warning_count:-0}" -gt 0 ] || [ "$${skipped_count:-0}" -gt 0 ]; then \
		if [ "$$rc" -eq 0 ]; then rc=1; fi; \
	fi; \
	if [ "$(DIAG)" = "1" ]; then \
		run_state=COMPLETED; \
		if [ "$$rc" -eq 130 ]; then run_state=INTERRUPTED; fi; \
		printf 'DIAG %s | failed=%s errors=%s warnings=%s skipped=%s\n' \
			"$$run_state" "$$failed_count" "$$error_count" \
			"$$warning_count" "$$skipped_count" >&2; \
	fi; \
	ln -sfn "$$run_id" "$(PYTEST_REPORTS_DIR)/latest"; \
	printf 'Reports: %s (latest: %s/latest)\n' \
		"$$report_dir" "$(PYTEST_REPORTS_DIR)" >&2; \
	exit "$$rc"


_builtin_fmt_check: _builtin_require_environment
	@$(UV_RUN) ruff check --no-fix $(RUFF_PATHS)
	@$(UV_RUN) ruff format --check $(RUFF_PATHS)

_builtin_fmt_apply: _builtin_require_environment
	$(call _require_apply)
	@$(UV_RUN) ruff check --fix $(RUFF_PATHS)
	@$(UV_RUN) ruff format $(RUFF_PATHS)

_builtin_run_default: _builtin_require_environment
	@$(UV_RUN) $(PROJECT_NAME) $(ARGS)

_builtin_status_diagnostics: _builtin_require_environment
	@printf 'profile=%s\nattached=%s\nproject=%s\nruntime=%s\n' \
		'$(MAKE_PROFILE)' '$(ATTACHED_MEMBER)' '$(PROJECT_ROOT)' '$(RUNTIME_ROOT)'
	@$(UV) --version
	@$(UV) lock --project "$(PROJECT_ROOT)" --check
	@if [ -x "$(RUNTIME_PYTHON)" ]; then \
		$(UV) pip check --python "$(RUNTIME_VENV)"; \
	fi
	@git -C "$(PROJECT_ROOT)" status --short

_builtin_docs_all:
	@set -eu; \
	for action in $(DOCS_ACTIONS); do \
		case "$$action" in generate|fix) mode=$(if $(filter Y,$(APPLY)),--apply,--check) ;; *) mode= ;; esac; \
		$(PROJECT_FLEXT_INFRA) docs "$$action" --workspace "$(PROJECT_ROOT)" --output-dir "$(PROJECT_ROOT)/.reports/docs" $$mode $(DOCS_PROJECT_ARGS); \
	done


_builtin_docs_generate:
	@$(PROJECT_FLEXT_INFRA) docs generate --workspace "$(PROJECT_ROOT)" --output-dir "$(PROJECT_ROOT)/.reports/docs" $(if $(filter Y,$(APPLY)),--apply,--check) $(DOCS_PROJECT_ARGS)


_builtin_docs_fix:
	@$(PROJECT_FLEXT_INFRA) docs fix --workspace "$(PROJECT_ROOT)" --output-dir "$(PROJECT_ROOT)/.reports/docs" $(if $(filter Y,$(APPLY)),--apply,--check) $(DOCS_PROJECT_ARGS)


_builtin_docs_audit:
	@$(PROJECT_FLEXT_INFRA) docs audit --workspace "$(PROJECT_ROOT)" --output-dir "$(PROJECT_ROOT)/.reports/docs" $(DOCS_PROJECT_ARGS)


_builtin_docs_build:
	@$(PROJECT_FLEXT_INFRA) docs build --workspace "$(PROJECT_ROOT)" --output-dir "$(PROJECT_ROOT)/.reports/docs" $(DOCS_PROJECT_ARGS)


_builtin_docs_validate:
	@$(PROJECT_FLEXT_INFRA) docs validate --workspace "$(PROJECT_ROOT)" --output-dir "$(PROJECT_ROOT)/.reports/docs" $(DOCS_PROJECT_ARGS)



_builtin_clean_generated:
	$(call _require_apply)
	@find "$(PROJECT_ROOT)" -type d \
		\( -name __pycache__ -o -name .mypy_cache -o -name .pytest_cache -o -name .ruff_cache \) \
		-prune -exec rm -rf {} +
	@rm -rf "$(PROJECT_ROOT)/build" "$(PROJECT_ROOT)/dist" "$(PROJECT_ROOT)/htmlcov"
	@rm -f "$(PROJECT_ROOT)/.coverage"

_builtin_release_status: _builtin_require_environment
	@$(UV) lock --project "$(PROJECT_ROOT)" --check
	@git -C "$(PROJECT_ROOT)" diff --quiet
	@git -C "$(PROJECT_ROOT)" diff --cached --quiet

_builtin_codegen_check: _builtin_require_environment
	@$(PROJECT_FLEXT_INFRA) codegen conform --root "$(PROJECT_ROOT)" --scope "$(CODEGEN_SCOPE)" --mode check

_builtin_codegen_apply: _builtin_require_environment
	$(call _require_apply)
	@$(PROJECT_FLEXT_INFRA) codegen conform --root "$(PROJECT_ROOT)" --scope "$(CODEGEN_SCOPE)" --mode apply

_builtin_worktree_list:
	@$(PROJECT_FLEXT_INFRA) workspace worktree --workspace "$(WORKSPACE)" --operation list

_builtin_worktree_add:
	$(call _require_apply)
	@$(PROJECT_FLEXT_INFRA) workspace worktree --workspace "$(WORKSPACE)" --operation add --branch "$(BRANCH)" --base "$(BASE)" --apply

_builtin_worktree_update:
	$(call _require_apply)
	@$(PROJECT_FLEXT_INFRA) workspace worktree --workspace "$(WORKSPACE)" --operation update --branch "$(BRANCH)" --base "$(BASE)" --apply

_builtin_worktree_remove:
	$(call _require_apply)
	@$(PROJECT_FLEXT_INFRA) workspace worktree --workspace "$(WORKSPACE)" --operation remove --branch "$(BRANCH)" --apply
