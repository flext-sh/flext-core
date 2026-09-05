# @flext-generated: continuous
# @flext-owner: flext-infra/config/codegen.yaml + flext-infra/src/flext_infra/templates/project/base/Makefile.j2
# @flext-adjust: edit the owner configuration or template; never this projection
# @flext-regenerate: make gen APPLY=Y
# flext-core — selector-free generated project interface.

SHELL := /bin/sh
.DEFAULT_GOAL := help
.DELETE_ON_ERROR:
undefine PYTHONDONTWRITEBYTECODE
unexport PYTHONDONTWRITEBYTECODE

PROJECT_NAME := flext-core
MAKE_PROFILE := standalone
DECLARED_REPOSITORIES :=
MANAGED_GITLINKS :=
UV_LINK_MODE := copy
PUBLIC_VERBS := help audit status setup deps build check test fmt fix docs clean release-plan release-version release-tag release-build publication gen conform initialize mod waza duplication

# APPLY=Y is the sole public input. Any command-line assignment outside this
# closed contract is rejected before a recipe can partially execute.
PUBLIC_INPUTS := APPLY
COMMAND_LINE_INPUTS := $(foreach name,$(.VARIABLES),$(if $(filter command line override,$(origin $(name))),$(name)))
UNKNOWN_INPUTS := $(filter-out $(PUBLIC_INPUTS),$(COMMAND_LINE_INPUTS))
ifneq ($(strip $(UNKNOWN_INPUTS)),)
$(error Unsupported Make input(s): $(UNKNOWN_INPUTS); public operations accept only APPLY=Y)
endif
APPLY ?= N
ifneq ($(filter $(APPLY),N Y),$(APPLY))
$(error APPLY must be Y when enabled)
endif

SELF_MAKEFILE := $(abspath $(firstword $(MAKEFILE_LIST)))
PROJECT_ROOT := $(patsubst %/,%,$(dir $(SELF_MAKEFILE)))
REPOSITORY_ROOT := $(shell cd "$(PROJECT_ROOT)" && root=$$(git rev-parse --show-superproject-working-tree 2>/dev/null); if [ -n "$$root" ]; then printf '%s\n' "$$root"; else git rev-parse --show-toplevel 2>/dev/null || printf '%s\n' "$(PROJECT_ROOT)"; fi)
ifeq ($(OS),Windows_NT)
TRACKED_MISE := $(PROJECT_ROOT)/bin/mise.cmd
RUNTIME_BIN := $(PROJECT_ROOT)/.venv/Scripts
RUNTIME_PYTHON := $(RUNTIME_BIN)/python.exe
else
TRACKED_MISE := $(PROJECT_ROOT)/bin/mise
RUNTIME_BIN := $(PROJECT_ROOT)/.venv/bin
RUNTIME_PYTHON := $(RUNTIME_BIN)/python
endif
RUNTIME_VENV := $(PROJECT_ROOT)/.venv
override SETUP_MISE := $(TRACKED_MISE)
UV ?= uv
UV_REQUESTED := $(UV)
CALLER_PATH := $(PATH)
ifneq ($(filter help setup,$(MAKECMDGOALS)),)
SETUP_BOOTSTRAP_ONLY := Y
export SETUP_BOOTSTRAP_ONLY
RESOLVED_UV :=
else
RESOLVED_UV := $(shell PATH="$(CALLER_PATH)" command -v "$(UV_REQUESTED)")
ifneq ($(.SHELLSTATUS),0)
$(error Required uv executable not found: $(UV_REQUESTED))
endif
endif
override UV := $(if $(strip $(RESOLVED_UV)),$(RESOLVED_UV),$(UV_REQUESTED))
override UV_PROJECT := $(PROJECT_ROOT)
override UV_PROJECT_ENVIRONMENT := $(RUNTIME_VENV)
override VIRTUAL_ENV := $(RUNTIME_VENV)
override PATH := $(RUNTIME_BIN):$(CALLER_PATH)
override export FLEXT_INFRA_PYTHON := $(RUNTIME_PYTHON)
export FLEXT_INFRA_PYTHON UV UV_PROJECT UV_PROJECT_ENVIRONMENT VIRTUAL_ENV PATH

RUNTIME_STATE_ROOT := $(abspath $(dir $(REPOSITORY_ROOT))/.flext-runtime)
PROJECT_STATE_ROOT := $(RUNTIME_STATE_ROOT)/$(notdir $(PROJECT_ROOT))
PROJECT_SCRATCH_ROOT := $(PROJECT_STATE_ROOT)/scratch
override export TMPDIR := $(PROJECT_SCRATCH_ROOT)
override export GOTMPDIR := $(PROJECT_SCRATCH_ROOT)
TESTMON_STATE_ROOT := $(RUNTIME_STATE_ROOT)/testmon/$(notdir $(PROJECT_ROOT))
override export PYTHONPYCACHEPREFIX := $(PROJECT_STATE_ROOT)/pycache
override export TESTMON_DATAFILE := $(TESTMON_STATE_ROOT)/.testmondata
override export FLEXT_PYTEST_TARGET_RAW := tests
override export FLEXT_PYTEST_REPORTS_RAW := .reports/tests
PYTEST_PROCESS_TIMEOUT_SECONDS := 660
PYTEST_BOUNDED := timeout --signal=TERM --kill-after=5s "$(PYTEST_PROCESS_TIMEOUT_SECONDS)s"

UV_RUN := env -u MYPYPATH -u VIRTUAL_ENV -u UV_PROJECT -u UV_PROJECT_ENVIRONMENT PYTHONPATH="$(PROJECT_ROOT)/src" $(UV) run --project "$(PROJECT_ROOT)" --no-sync
PROJECT_FLEXT_INFRA := if [ ! -x "$(RUNTIME_PYTHON)" ]; then printf 'ERROR: missing managed Python %s\n' "$(RUNTIME_PYTHON)" >&2; exit 2; fi; env -u PYTHONPATH -u MYPYPATH -u VIRTUAL_ENV -u UV_PROJECT -u UV_PROJECT_ENVIRONMENT PYTHONPATH="$(PROJECT_ROOT)/src" "$(RUNTIME_PYTHON)" -m flext_infra
SELF_MAKE := $(MAKE) --no-print-directory -f "$(SELF_MAKEFILE)"
CUSTOM_MAKEFILE := $(PROJECT_ROOT)/custom.mk
CUSTOM_DECLARED_TARGETS :=
ifneq ($(wildcard $(CUSTOM_MAKEFILE)),)
CUSTOM_DECLARED_TARGETS := $(shell awk '/^(_custom-[a-z][a-z0-9-]*|(pre|post)-[a-z][a-z0-9-]*):/ { target=$$1; sub(/:.*/, "", target); if (!seen[target]++) printf "%s ", target }' "$(CUSTOM_MAKEFILE)")
ifneq ($(.SHELLSTATUS),0)
$(error Failed to inspect custom Make hooks in $(CUSTOM_MAKEFILE))
endif
endif
-include custom.mk

ifeq ($(MAKE_PROFILE),workspace)
CODEGEN_SCOPE := all
else ifeq ($(MAKE_PROFILE),standalone)
CODEGEN_SCOPE := self
else
$(error Invalid generated Make profile: $(MAKE_PROFILE))
endif
WORKSPACE_ORCHESTRATE := $(UV_RUN) python -m flext_infra workspace orchestrate
CANONICAL_GATES := lint pyrefly mypy pyright silent-failure deferred-self-reference security markdown loc-cap boundary canonical-alias runtime-census namespace layout tier-whitelist smells codemod direnv duplication
FIXABLE_GATES := lint markdown canonical-alias smells
empty :=
space := $(empty) $(empty)
comma := ,
CANONICAL_GATE_CSV := $(subst $(space),$(comma),$(strip $(CANONICAL_GATES)))
FIXABLE_GATE_CSV := $(subst $(space),$(comma),$(strip $(FIXABLE_GATES)))

define REQUIRE_APPLY
	@if [ "$(APPLY)" != "Y" ]; then printf 'ERROR: $@ requires APPLY=Y\n' >&2; exit 2; fi
endef

define REJECT_APPLY
	@if [ "$(APPLY)" = "Y" ]; then printf 'ERROR: $@ is declared read-only and rejects APPLY\n' >&2; exit 2; fi
endef

define RUN_PUBLIC
	$(if $(filter pre-$(1),$(CUSTOM_DECLARED_TARGETS)),@$(SELF_MAKE) pre-$(1))
	@$(SELF_MAKE) _builtin-$(1)
	$(if $(filter post-$(1),$(CUSTOM_DECLARED_TARGETS)),@$(SELF_MAKE) post-$(1))
endef

define RUN_WORKSPACE
	@APPLY="$(APPLY)" $(WORKSPACE_ORCHESTRATE) --verb $(1)
endef

.PHONY: _bootstrap-setup-tools

_bootstrap-setup-tools:
	@set -eu; \
	project_root="$(PROJECT_ROOT)"; \
	mise="$(SETUP_MISE)"; \
	uv_required="0.12"; \
	if [ ! -f "$$mise" ]; then \
		printf 'ERROR: missing generated mise launcher: %s; run make gen APPLY=Y\n' "$$mise" >&2; \
		exit 2; \
	fi; \
	if [ ! -f "$$project_root/mise.lock" ]; then \
		printf 'ERROR: missing generated mise.lock; run make gen APPLY=Y and commit it\n' >&2; \
		exit 2; \
	fi; \
	scratch_parent="$(PROJECT_SCRATCH_ROOT)"; \
	mkdir -p "$$scratch_parent"; \
	scratch=$$(mktemp -d "$$scratch_parent/mise-toolchain.XXXXXX"); \
	trap 'find "$$scratch" -depth -delete' EXIT; \
	mkdir -p "$$scratch/receipt/bin" "$$scratch/data" "$$scratch/cache" \
		"$$scratch/state" "$$scratch/tmp" "$$scratch/config" \
		"$$scratch/system-config"; \
	: > "$$scratch/global-config.toml"; \
	if MISE_GLOBAL_CONFIG_FILE="$$scratch/global-config.toml" \
		MISE_CONFIG_DIR="$$scratch/config" MISE_DATA_DIR="$$scratch/data" \
		MISE_CACHE_DIR="$$scratch/cache" MISE_STATE_DIR="$$scratch/state" \
		TMPDIR="$$scratch/tmp" MISE_CEILING_PATHS="$$scratch_parent" \
		MISE_TRUSTED_CONFIG_PATHS="$$scratch" \
		env -u MISE_INSTALL_PATH -u MISE_VERSION "$(SETUP_MISE)" \
		-C "$$scratch" generate install-script \
		--write "$$scratch/receipt/bin/mise" --windows \
		>"$$scratch/generate.log" 2>&1; then \
		cat "$$scratch/generate.log"; \
	else \
		status=$$?; cat "$$scratch/generate.log"; exit "$$status"; \
	fi; \
	if grep -Fq 'mise WARN' "$$scratch/generate.log"; then \
		printf 'ERROR: Mise launcher generation emitted warnings\n' >&2; exit 2; \
	else \
		status=$$?; if [ "$$status" -ne 1 ]; then exit "$$status"; fi; \
	fi; \
	chmod +x "$$scratch/receipt/bin/mise"; \
	latest_mise="$$scratch/receipt/bin/mise"; \
	mise_runtime=$$(env -u MISE_INSTALL_PATH -u MISE_VERSION \
		"$$latest_mise" --version); \
	mise_release=$${mise_runtime%% *}; \
	case "$$mise_release" in \
		''|*[!0-9.]*|.*|*.|*..*) \
			printf 'ERROR: generated Mise returned invalid version: %s\n' \
				"$$mise_runtime" >&2; exit 2 ;; \
	esac; \
	old_ifs=$$IFS; IFS=.; set -- $$mise_release; IFS=$$old_ifs; \
	if [ "$$#" -ne 3 ]; then \
		printf 'ERROR: generated Mise returned invalid version: %s\n' \
			"$$mise_runtime" >&2; exit 2; \
	fi; \
	printf 'mise setup runtime=%s\n' "$$mise_runtime"; \
	global_config="$$scratch/global-config.toml"; \
	config_dir="$$scratch/config"; \
	MISE_CONFIG_DIR="$$config_dir" MISE_GLOBAL_CONFIG_FILE="$$global_config" \
		env -u MISE_INSTALL_PATH -u MISE_VERSION "$$latest_mise" trust "$$project_root/.mise.toml"; \
	MISE_CONFIG_DIR="$$config_dir" MISE_GLOBAL_CONFIG_FILE="$$global_config" \
		env -u MISE_INSTALL_PATH -u MISE_VERSION "$$latest_mise" -C "$$project_root" install --locked --yes; \
	uv_output=$$(MISE_CONFIG_DIR="$$config_dir" \
		MISE_GLOBAL_CONFIG_FILE="$$global_config" \
		env -u MISE_INSTALL_PATH -u MISE_VERSION "$$latest_mise" -C "$$project_root" \
		exec -- uv --version); \
	case "$$uv_output" in \
		'uv '*) uv_actual=$${uv_output#uv }; uv_actual=$${uv_actual%% *} ;; \
		*) printf 'ERROR: uv --version returned an invalid value\n' >&2; exit 2 ;; \
	esac; \
	case "$$uv_actual" in \
		"$$uv_required"|"$$uv_required".*) ;; \
		*) printf 'ERROR: mise must install uv %s.x, found %s\n' \
			"$$uv_required" "$$uv_actual" >&2; exit 2 ;; \
	esac; \
	if [ -n "$${GITHUB_PATH:-}" ]; then \
		managed_path=$$(MISE_CONFIG_DIR="$$config_dir" \
			MISE_GLOBAL_CONFIG_FILE="$$global_config" \
			env -u MISE_INSTALL_PATH -u MISE_VERSION "$$latest_mise" \
			-C "$$project_root" exec -- sh -c 'printf %s "$$PATH"'); \
		printf '%s\n' "$$project_root/bin" >> "$$GITHUB_PATH"; \
		old_ifs=$$IFS; IFS=:; \
		for bin_dir in $$managed_path; do printf '%s\n' "$$bin_dir" >> "$$GITHUB_PATH"; done; \
		IFS=$$old_ifs; \
	fi; \
	MISE_CONFIG_DIR="$$config_dir" MISE_GLOBAL_CONFIG_FILE="$$global_config" \
		env -u MISE_INSTALL_PATH -u MISE_VERSION "$$latest_mise" \
		-C "$$project_root" exec -- $(SELF_MAKE) _setup-lifecycle
.PHONY: _setup-submodules

# === SECTION: submodule setup (managed) ===
# Source: template (submodule_setup_recipe.j2)
# Computed: workspace uses DECLARED_REPOSITORIES from config; standalone discovers
#           submodules with flext-managed=true from .gitmodules at runtime.
# Rule: setup PROVISIONS an absent governed gitlink and VERIFIES a present one.
#       An absent checkout holds no work, so setup initializes it at the recorded
#       gitlink. A present checkout is never destroyed: git checkout, git reset,
#       fetch, and branch attachment are forbidden. Pin validity is HEAD contains
#       gitlink. Declared branch is the named integration line;
#       legacy branch=. still resolves to the superproject named branch if present.
#       A checkout is also accepted on the superproject current branch (workspace
#       lane). Any third branch fails. Nested gitlinks belong to their own setup.
# Free: no
# End SECTION: submodule setup
_setup-submodules:
	@set -eu; \
	root="$(PROJECT_ROOT)"; \
	if [ ! -f "$$root/.gitmodules" ]; then exit 0; fi; \
	profile="$(MAKE_PROFILE)"; \
	if [ "$$profile" = "workspace" ]; then \
		managed="$(MANAGED_GITLINKS)"; \
	else \
		managed=""; \
		keys=""; \
		if keys=$$(git -C "$$root" config -f .gitmodules --name-only --get-regexp '^submodule\..*\.flext-managed$$'); then \
			keys_status=0; \
		else \
			keys_status=$$?; \
		fi; \
		if [ "$$keys_status" -ne 0 ] && [ "$$keys_status" -ne 1 ]; then \
			printf 'ERROR: cannot enumerate governed gitlinks\n' >&2; \
			exit "$$keys_status"; \
		fi; \
		for key in $$keys; do \
			value=$$(git -C "$$root" config -f .gitmodules --get "$$key"); \
			if [ "$$value" = "true" ]; then \
				section=$${key%.flext-managed}; \
				path=$$(git -C "$$root" config -f .gitmodules --get --default "" "$$section.path"); \
				if [ -n "$$path" ]; then \
					managed="$$managed $$path"; \
				fi; \
			fi; \
		done; \
	fi; \
	managed=$$(printf '%s' "$$managed" | tr ' ' '\n' | sort -u | tr '\n' ' '); \
	if [ -z "$$managed" ]; then exit 0; fi; \
	validate_submodule() { \
		superproject="$$1"; \
		child_path="$$2"; \
		child_root="$$superproject/$$child_path"; \
		keys=""; \
		if keys=$$(git -C "$$superproject" config -f .gitmodules --name-only --get-regexp '^submodule\..*\.path$$'); then \
			keys_status=0; \
		else \
			keys_status=$$?; \
		fi; \
		if [ "$$keys_status" -ne 0 ] && [ "$$keys_status" -ne 1 ]; then \
			printf 'ERROR: cannot enumerate submodule paths\n' >&2; \
			exit "$$keys_status"; \
		fi; \
		section=""; \
		for key in $$keys; do \
			declared=$$(git -C "$$superproject" config -f .gitmodules --get "$$key"); \
			if [ "$$declared" = "$$child_path" ]; then \
				if [ -n "$$section" ]; then \
					printf 'ERROR: governed gitlink path is duplicated: %s\n' "$$child_path" >&2; \
					exit 2; \
				fi; \
				section=$${key%.path}; \
			fi; \
		done; \
		if [ -z "$$section" ]; then \
			printf 'ERROR: governed gitlink is absent from .gitmodules: %s\n' "$$child_path" >&2; \
			exit 2; \
		fi; \
		branch=$$(git -C "$$superproject" config -f .gitmodules --get --default "" "$$section.branch"); \
		if [ -z "$$branch" ]; then \
			printf 'ERROR: governed gitlink has no declared branch: %s\n' "$$child_path" >&2; \
			exit 2; \
		fi; \
		super_branch=$$(git -C "$$superproject" branch --show-current); \
		if [ "$$branch" = "." ]; then \
			branch="$$super_branch"; \
			if [ -z "$$branch" ]; then \
				printf 'ERROR: %s: branch = . requires a named superproject branch\n' "$$child_path" >&2; \
				exit 1; \
			fi; \
		fi; \
		declared_branch="$$branch"; \
		accepted_branches="$$declared_branch"; \
		if [ -n "$$super_branch" ] && [ "$$super_branch" != "$$declared_branch" ]; then \
			accepted_branches="$$declared_branch or $$super_branch"; \
		fi; \
		validated_branch=$$(git check-ref-format --branch "$$branch"); \
		if [ "$$validated_branch" != "$$branch" ]; then \
			printf 'ERROR: branch validator changed %s to %s\n' "$$branch" "$$validated_branch" >&2; \
			exit 2; \
		fi; \
		gitlink_entry=$$(git -C "$$superproject" ls-files --stage -- "$$child_path"); \
		if [ -z "$$gitlink_entry" ]; then \
			printf 'ERROR: governed gitlink is absent from the index: %s\n' "$$child_path" >&2; \
			exit 2; \
		fi; \
		set -- $$gitlink_entry; \
		if [ "$$1" != 160000 ]; then \
			printf 'ERROR: governed path is not a gitlink: %s\n' "$$child_path" >&2; \
			exit 2; \
		fi; \
		gitlink="$$2"; \
		if [ ! -e "$$child_root/.git" ]; then \
			git -C "$$superproject" submodule update --init -- "$$child_path"; \
		fi; \
		current=$$(git -C "$$child_root" branch --show-current); \
		if [ -n "$$current" ] && [ "$$current" != "$$declared_branch" ] && \
		   [ -n "$$super_branch" ] && [ "$$current" = "$$super_branch" ]; then \
			branch="$$super_branch"; \
		fi; \
		head=$$(git -C "$$child_root" rev-parse HEAD); \
		if [ -n "$$current" ] && [ "$$current" != "$$branch" ]; then \
			printf 'ERROR: %s: conflicting branch %s; expected %s (setup never runs checkout/reset; switch it yourself while keeping dirty)\n' "$$child_path" "$$current" "$$accepted_branches" >&2; \
			exit 1; \
		fi; \
		if git -C "$$child_root" merge-base --is-ancestor "$$gitlink" HEAD; then \
			ancestor=Y; \
		else \
			status=$$?; if [ "$$status" -eq 1 ]; then ancestor=N; else exit "$$status"; fi; \
		fi; \
		if [ "$$ancestor" = N ]; then \
			if [ -z "$$current" ]; then \
				printf 'ERROR: %s: detached HEAD %s does not contain recorded gitlink %s\n' "$$child_path" "$$head" "$$gitlink" >&2; \
			else \
				printf 'ERROR: %s: branch %s does not contain recorded gitlink %s\n' "$$child_path" "$$branch" "$$gitlink" >&2; \
			fi; \
			exit 1; \
		fi; \
	}; \
	for child_path in $$managed; do \
		validate_submodule "$$root" "$$child_path"; \
	done

.PHONY: $(PUBLIC_VERBS) $(addprefix _builtin-,$(PUBLIC_VERBS)) $(addprefix _builtin-self-,build check test fmt fix) _require-environment _setup-lifecycle _setup-submodules _bootstrap-setup-tools


help:
	$(call REJECT_APPLY)
	$(call RUN_PUBLIC,help)

audit: _require-environment
	$(call REJECT_APPLY)
	$(call RUN_PUBLIC,audit)

status: _require-environment
	$(call REJECT_APPLY)
	$(call RUN_PUBLIC,status)

deps: _require-environment
	$(call REQUIRE_APPLY)
	$(call RUN_PUBLIC,deps)

build: _require-environment
	$(call REQUIRE_APPLY)
	$(call RUN_PUBLIC,build)

check: _require-environment
	$(call REQUIRE_APPLY)
	$(call RUN_PUBLIC,check)

test: _require-environment
	$(call REQUIRE_APPLY)
	$(call RUN_PUBLIC,test)

fmt: _require-environment
	$(call REQUIRE_APPLY)
	$(call RUN_PUBLIC,fmt)

fix: _require-environment
	$(call REQUIRE_APPLY)
	$(call RUN_PUBLIC,fix)

docs: _require-environment
	$(call REQUIRE_APPLY)
	$(call RUN_PUBLIC,docs)

clean: _require-environment
	$(call REQUIRE_APPLY)
	$(call RUN_PUBLIC,clean)

release-plan: _require-environment
	$(call REQUIRE_APPLY)
	$(call RUN_PUBLIC,release-plan)

release-version: _require-environment
	$(call REQUIRE_APPLY)
	$(call RUN_PUBLIC,release-version)

release-tag: _require-environment
	$(call REQUIRE_APPLY)
	$(call RUN_PUBLIC,release-tag)

release-build: _require-environment
	$(call REQUIRE_APPLY)
	$(call RUN_PUBLIC,release-build)

publication: _require-environment
	$(call REQUIRE_APPLY)
	$(call RUN_PUBLIC,publication)

gen: _require-environment
	$(call REQUIRE_APPLY)
	$(call RUN_PUBLIC,gen)

conform: _require-environment
	$(call REQUIRE_APPLY)
	$(call RUN_PUBLIC,conform)

initialize: _require-environment
	$(call REQUIRE_APPLY)
	$(call RUN_PUBLIC,initialize)

mod: _require-environment
	$(call REQUIRE_APPLY)
	$(call RUN_PUBLIC,mod)

waza: _require-environment
	$(call REQUIRE_APPLY)
	$(call RUN_PUBLIC,waza)

duplication: _require-environment
	$(call REQUIRE_APPLY)
	$(call RUN_PUBLIC,duplication)



setup:
	$(call REQUIRE_APPLY)
	@SETUP_BOOTSTRAP_ONLY=Y $(SELF_MAKE) _bootstrap-setup-tools

_require-environment:
	@if [ ! -x "$(RUNTIME_PYTHON)" ]; then printf 'ERROR: missing environment interpreter %s; run make setup APPLY=Y\n' "$(RUNTIME_PYTHON)" >&2; exit 2; fi

_setup-lifecycle: _setup-submodules
	$(if $(filter pre-setup,$(CUSTOM_DECLARED_TARGETS)),@$(SELF_MAKE) pre-setup)
	@set -eu; \
	$(UV) lock --project "$(PROJECT_ROOT)"; \
	if [ -L "$(RUNTIME_VENV)" ]; then \
		printf 'setup: borrowed environment %s\n' "$(RUNTIME_VENV)"; \
	else \
		if [ ! -x "$(RUNTIME_PYTHON)" ]; then $(UV) venv "$(RUNTIME_VENV)"; fi; \
		for metadata_dir in "$(RUNTIME_VENV)"/lib/python*/site-packages/*.dist-info; do \
			if [ -d "$$metadata_dir" ] && [ ! -s "$$metadata_dir/METADATA" ]; then \
				case "$$metadata_dir" in \
					"$(RUNTIME_VENV)"/lib/python*/site-packages/*.dist-info) ;; \
					*) printf 'ERROR: invalid distribution metadata path: %s\n' "$$metadata_dir" >&2; exit 2 ;; \
				esac; \
				printf 'setup: removing incomplete distribution metadata %s\n' "$$metadata_dir"; \
				rm -rf -- "$$metadata_dir"; \
			fi; \
		done; \
		$(UV) sync --frozen --project "$(PROJECT_ROOT)" --all-extras --all-groups --link-mode "$(UV_LINK_MODE)"; \
	fi; \
	direnv allow "$(PROJECT_ROOT)"; \
	$(UV) pip check --python "$(RUNTIME_VENV)"
	$(if $(filter post-setup,$(CUSTOM_DECLARED_TARGETS)),@$(SELF_MAKE) post-setup)

_builtin-help:
	@printf '%s\n\n' 'flext-core [standalone]';
	@printf '  %-16s %s\n' 'help' 'Show the complete selector-free public interface.';
	@printf '  %-16s %s\n' 'audit' 'Inspect ownership, dependency, and generated-state health.';
	@printf '  %-16s %s\n' 'status' 'Report the resolved runtime and repository state.';
	@printf '  %-16s %s (APPLY=Y)\n' 'setup' 'Provision the declared environment and hooks.';
	@printf '  %-16s %s (APPLY=Y)\n' 'deps' 'Upgrade, lock, and conform every declared dependency.';
	@printf '  %-16s %s (APPLY=Y)\n' 'build' 'Build the project distribution artifacts.';
	@printf '  %-16s %s (APPLY=Y)\n' 'check' 'Run every configured non-test gate.';
	@printf '  %-16s %s (APPLY=Y)\n' 'test' 'Run the complete suite through the persistent testmon cache.';
	@printf '  %-16s %s (APPLY=Y)\n' 'fmt' 'Apply canonical formatting.';
	@printf '  %-16s %s (APPLY=Y)\n' 'fix' 'Apply every configured safe correction.';
	@printf '  %-16s %s (APPLY=Y)\n' 'docs' 'Generate, repair, build, and validate documentation.';
	@printf '  %-16s %s (APPLY=Y)\n' 'clean' 'Remove every declared disposable artifact.';
	@printf '  %-16s %s (APPLY=Y)\n' 'release-plan' 'Resolve the release decision through the public protocol.';
	@printf '  %-16s %s (APPLY=Y)\n' 'release-version' 'Materialize the planned version.';
	@printf '  %-16s %s (APPLY=Y)\n' 'release-tag' 'Tag the verified release commit.';
	@printf '  %-16s %s (APPLY=Y)\n' 'release-build' 'Build the release receipt and artifacts.';
	@printf '  %-16s %s (APPLY=Y)\n' 'publication' 'Publish only receipt-attested release artifacts.';
	@printf '  %-16s %s (APPLY=Y)\n' 'gen' 'Regenerate every managed projection atomically.';
	@printf '  %-16s %s (APPLY=Y)\n' 'conform' 'Prove generated projections are at their fixed point.';
	@printf '  %-16s %s (APPLY=Y)\n' 'initialize' 'Materialize the declared package initializer graph.';
	@printf '  %-16s %s (APPLY=Y)\n' 'mod' 'Apply the declared structural codemods.';
	@printf '  %-16s %s (APPLY=Y)\n' 'waza' 'Validate provider-neutral governance semantics with Waza.';
	@printf '  %-16s %s (APPLY=Y)\n' 'duplication' 'Run the canonical jscpd duplicate-code gate.';


_builtin-audit:
	@$(UV) lock --project "$(PROJECT_ROOT)" --check
	@$(UV) pip check --python "$(RUNTIME_VENV)"
	@$(PROJECT_FLEXT_INFRA) codegen conform --root "$(PROJECT_ROOT)" --scope "$(CODEGEN_SCOPE)" --mode check

_builtin-status:
	@printf 'profile=%s\nproject=%s\nruntime=%s\ntestmon=%s\n' '$(MAKE_PROFILE)' '$(PROJECT_ROOT)' '$(RUNTIME_VENV)' '$(TESTMON_DATAFILE)'
	@env -u MISE_INSTALL_PATH -u MISE_VERSION "$(SETUP_MISE)" --version
	@$(UV) --version
	@$(UV) lock --project "$(PROJECT_ROOT)" --check
	@$(UV) pip check --python "$(RUNTIME_VENV)"
	@git -C "$(PROJECT_ROOT)" status --short

_builtin-deps:
	@$(UV) lock --project "$(PROJECT_ROOT)" --upgrade
	@$(PROJECT_FLEXT_INFRA) deps modernize --workspace "$(PROJECT_ROOT)" --apply --rewrite-constraints --skip-check
	@$(UV) lock --project "$(PROJECT_ROOT)"

_builtin-self-build:
	@$(UV) build --project "$(PROJECT_ROOT)"
_builtin-self-check:
	@$(PROJECT_FLEXT_INFRA) check run --workspace "$(PROJECT_ROOT)" --gates "$(CANONICAL_GATE_CSV)" --projects .
_builtin-self-test:
	@printf 'test: retained testmon cache=%s\n' '$(TESTMON_DATAFILE)'

	@set -eu; \
		test_tmp_parent="$(PROJECT_SCRATCH_ROOT)/pytest"; \
		mkdir -p "$$test_tmp_parent"; \
		test_tmp=$$(mktemp -d "$$test_tmp_parent/invocation.XXXXXX"); \
		cleanup_test_tmp() { rm -rf "$$test_tmp"; }; \
		trap cleanup_test_tmp EXIT INT TERM; \
		TMPDIR="$$test_tmp" GOTMPDIR="$$test_tmp" $(PYTEST_BOUNDED) $(UV_RUN) python -m flext_infra._pytest_entry
_builtin-self-fmt:
	@$(PROJECT_FLEXT_INFRA) check run --workspace "$(PROJECT_ROOT)" --gates format --projects . --fix
_builtin-self-fix:
	@$(PROJECT_FLEXT_INFRA) check run --workspace "$(PROJECT_ROOT)" --gates "$(FIXABLE_GATE_CSV)" --projects . --fix


_builtin-build: _builtin-self-build
_builtin-check: _builtin-self-check
_builtin-test: _builtin-self-test
_builtin-fmt: _builtin-self-fmt
_builtin-fix: _builtin-self-fix


_builtin-docs:
	@set -eu; \
	for action in generate fix audit build validate; do \
		case "$$action" in generate|fix) mode=--apply ;; *) mode= ;; esac; \
		$(PROJECT_FLEXT_INFRA) docs "$$action" --workspace "$(PROJECT_ROOT)" --output-dir "$(PROJECT_ROOT)/.reports/docs" $$mode; \
	done

_builtin-clean:
	@$(PROJECT_FLEXT_INFRA) maintenance clean --workspace "$(PROJECT_ROOT)" --apply

_builtin-release-plan:
	@$(PROJECT_FLEXT_INFRA) release run --workspace "$(PROJECT_ROOT)" --phase plan $(if $(strip $(PR_TITLE)),--pr-title "$(PR_TITLE)")
_builtin-release-version:
	@$(PROJECT_FLEXT_INFRA) release run --workspace "$(PROJECT_ROOT)" --phase version --apply
_builtin-release-tag:
	@$(PROJECT_FLEXT_INFRA) release run --workspace "$(PROJECT_ROOT)" --phase tag --apply
_builtin-release-build:
	@$(PROJECT_FLEXT_INFRA) release run --workspace "$(PROJECT_ROOT)" --phase build --apply
_builtin-publication:
	@$(PROJECT_FLEXT_INFRA) release run --workspace "$(PROJECT_ROOT)" --phase publish --apply

_builtin-gen:
	@$(PROJECT_FLEXT_INFRA) codegen layout --workspace "$(PROJECT_ROOT)" --apply
	@$(PROJECT_FLEXT_INFRA) codegen conform --root "$(PROJECT_ROOT)" --scope "$(CODEGEN_SCOPE)" --mode apply
	@$(PROJECT_FLEXT_INFRA) codegen py-typed --workspace "$(PROJECT_ROOT)" --apply
	@$(PROJECT_FLEXT_INFRA) codegen lazy-init --workspace "$(PROJECT_ROOT)" --apply
	@$(PROJECT_FLEXT_INFRA) docs generate --workspace "$(PROJECT_ROOT)" --output-dir "$(PROJECT_ROOT)/.reports/docs" --apply

_builtin-conform:
	@$(PROJECT_FLEXT_INFRA) codegen conform --root "$(PROJECT_ROOT)" --scope "$(CODEGEN_SCOPE)" --mode check
	@$(PROJECT_FLEXT_INFRA) codegen py-typed --workspace "$(PROJECT_ROOT)" --check
	@$(PROJECT_FLEXT_INFRA) codegen lazy-init --workspace "$(PROJECT_ROOT)" --check
	@$(PROJECT_FLEXT_INFRA) docs generate --workspace "$(PROJECT_ROOT)" --output-dir "$(PROJECT_ROOT)/.reports/docs" --check
	@$(PROJECT_FLEXT_INFRA) check run --workspace "$(PROJECT_ROOT)" --gates layout --projects .

_builtin-initialize:
	@$(PROJECT_FLEXT_INFRA) codegen lazy-init --workspace "$(PROJECT_ROOT)" --apply
	@$(PROJECT_FLEXT_INFRA) codegen lazy-init --workspace "$(PROJECT_ROOT)" --check

_builtin-mod:
	@$(PROJECT_FLEXT_INFRA) refactor mod --workspace "$(PROJECT_ROOT)" --apply

_builtin-waza:
	@cd "$(PROJECT_ROOT)" && "$(SETUP_MISE)" exec -- waza check --no-update-check

_builtin-duplication:
	@$(PROJECT_FLEXT_INFRA) check run --workspace "$(PROJECT_ROOT)" --gates duplication --projects .
