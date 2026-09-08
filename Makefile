# @flext-generated: continuous
# @flext-owner: flext-infra/config/codegen.yaml + flext-infra/src/flext_infra/templates/project/base/Makefile.j2
# @flext-adjust: edit the owner configuration or template; never this projection
# @flext-regenerate: make gen APPLY=Y
# flext-core — selector-free generated project interface.
# Managed by flext-infra codegen conform for new and existing repositories.
# === SECTION: header (managed) ===
# Source: template (base/Makefile.j2)
# Free: no
# End SECTION: header

SHELL := /bin/sh
.DEFAULT_GOAL := help
ifeq ($(filter command line override,$(origin SETUP_BOOTSTRAP_ONLY)),)
ifneq ($(filter setup,$(MAKECMDGOALS)),)
SETUP_BOOTSTRAP_ONLY := Y
export SETUP_BOOTSTRAP_ONLY
endif
endif

# === SECTION: project identity (managed) ===
# Source: config:dist / config:make_profile / config:repository_root_rel / config:uv_link_mode
PROJECT_NAME := flext-core
MAKE_PROFILE := standalone
REPOSITORY_ROOT_REL := .
# === SECTION: workspace subprojects (managed) ===
# Source: config:workspace_subprojects (list), config:workspace_repositories (list)
# Computed: MANAGED_GITLINKS mirrors the read-only local .gitmodules topology.
WORKSPACE_SUBPROJECTS :=
MANAGED_GITLINKS :=
WORKSPACE_EDITABLES := $(PROJECT_NAME):.
UV_LINK_MODE := copy
# End SECTION: project identity

# === SECTION: public boundary (managed) ===
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
APPLYING := $(if $(filter Y,$(APPLY)),Y)
PYTEST_DIAG_ARGS := -rA --durations=0 --tb=long --showlocals
PYTEST_REPORT_ARGS := -ra --durations=25 --durations-min=0.001 --tb=short
PYTEST_PROCESS_TIMEOUT_SECONDS := 660
# mro-99ae: the pytest process inherits a hard wall-clock boundary, so a hung
# run is terminated even if the runner itself stalls.
PYTEST_BOUNDED = timeout --signal=TERM --kill-after=5s "$(PYTEST_PROCESS_TIMEOUT_SECONDS)s"
PYTEST_REPORTS_DIR := .reports/tests
override PYTEST_CASE_TIMEOUT_SECONDS := 10
override PYTEST_RUN_TIMEOUT_SECONDS := 600
override PYTEST_TERMINATION_GRACE_SECONDS := 2
override PYTEST_TIMEOUT_EXIT_CODE := 124
override PYTEST_ENFORCEMENT_PLUGIN := flext_tests.enforcement_plugin
override PYTEST_PROGRESS_ARGS := --verbose
override PYTEST_REPORT_ARGS := -ra --durations=25 --durations-min=0.001 --tb=short
override PYTEST_DIAG_ARGS := -rA --durations=0 --tb=long --showlocals
override PYTEST_PARALLEL_WORKERS := 2
override PYTEST_PARALLEL_WORKER_MEMORY_GB := 2
override PYTEST_PARALLEL_DISTRIBUTION := load
override PYTEST_PROFILE_SORT := cumulative
override PYTEST_PROFILE_LIMIT := 50
override PROCESS_TIMEOUT_COMMAND := timeout
override export FLEXT_PYTEST_REPORTS_RAW := $(value PYTEST_REPORTS_DIR)
# End SECTION: public boundary

# === SECTION: derived paths (managed) ===
# Source: computed (git rev-parse, MAKEFILE_LIST, abspath)
# Rule: PROJECT_ROOT is the checkout that OWNS this Makefile, never the caller's
# CWD. Deriving it from `pwd -P` made a member validate whatever tree the
# caller happened to stand in: `make -f <member>/Makefile` invoked from the
# superproject resolved RUFF_PATHS to the SUPERPROJECT's src/tests, so the
# member linted files it does not even contain. With many shared worktrees that
# silently validates the wrong tree.
SELF_MAKEFILE := $(abspath $(firstword $(MAKEFILE_LIST)))
MAKEFILE_ROOT := $(patsubst %/,%,$(dir $(SELF_MAKEFILE)))
PROJECT_ROOT := $(MAKEFILE_ROOT)
SETUP_BIN := $(PROJECT_ROOT)/.bin
ifeq ($(OS),Windows_NT)
TRACKED_MISE := $(PROJECT_ROOT)/bin/mise.cmd
else
TRACKED_MISE := $(PROJECT_ROOT)/bin/mise
endif
override SETUP_MISE := $(TRACKED_MISE)
override export FLEXT_PYTEST_TARGET_RAW := tests
PROJECT_STATE_ROOT := $(abspath $(PROJECT_ROOT)/../.flext-runtime/$(notdir $(PROJECT_ROOT)))
PROJECT_SCRATCH_ROOT := $(PROJECT_STATE_ROOT)/scratch
TESTMON_DATAFILE := $(PROJECT_STATE_ROOT)/testmon/.testmondata
export TESTMON_DATAFILE
# === SECTION: REPOSITORY_ROOT isolation (managed) ===
# Source: computed (rule: derive from current checkout unless caller overrides)
# Rule: REPOSITORY_ROOT is always derived from the current checkout unless the
# caller passed it on the command line or via an override origin. An inherited
# environment REPOSITORY_ROOT (e.g. a leaked .envrc export from a foreign checkout)
# must never redirect verbs to another working tree. The git queries therefore
# run inside MAKEFILE_ROOT: run from a foreign CWD they would report THAT
# checkout's topology and redirect the verb to the wrong tree.
ifeq ($(filter command line override,$(origin REPOSITORY_ROOT)),)
REPOSITORY_ROOT := $(shell cd "$(MAKEFILE_ROOT)" && root=$$(git rev-parse --show-superproject-working-tree 2>/dev/null); if [ -n "$$root" ]; then printf '%s\n' "$$root"; else git rev-parse --show-toplevel 2>/dev/null || printf '%s\n' "$(MAKEFILE_ROOT)"; fi)
endif
# End SECTION: REPOSITORY_ROOT isolation
# === SECTION: verb dispatch (managed) ===
# Source: config:make.verbs and the canonical gate vocabulary.
PUBLIC_VERBS := help setup deps build check test fmt fix fix-enforcement audit status docs clean release-plan release-version release-tag release-build publication gen conform initialize mod waza duplication
BUILTIN_VERBS := help setup deps build check test fmt fix fix-enforcement audit status docs clean release-plan release-version release-tag release-build publication gen conform initialize mod waza duplication
SCRIPT_VERBS :=
CUSTOM_MAKEFILE := $(MAKEFILE_ROOT)/custom.mk
CUSTOM_DECLARED_TARGETS :=
ifneq ($(wildcard $(CUSTOM_MAKEFILE)),)
CUSTOM_DECLARED_TARGETS := $(shell awk '/^(_custom-[a-z][a-z0-9-]*|(pre|post)-[a-z][a-z0-9-]*):/ { target=$$1; sub(/:.*/, "", target); if (!seen[target]++) printf "%s ", target }' "$(CUSTOM_MAKEFILE)")
ifneq ($(.SHELLSTATUS),0)
$(error Failed to inspect custom Make targets in $(CUSTOM_MAKEFILE))
endif
endif
DOCS_ACTIONS := generate fix audit build validate
 # End SECTION: verb dispatch

# === SECTION: lint/type paths (managed) ===
# Source: template + computed (script_dispatch conditional)
RUFF_PATHS := $(strip $(foreach d,src tests examples,$(if $(wildcard $(PROJECT_ROOT)/$(d)/.),$(PROJECT_ROOT)/$(d),)))
MYPY_PATHS := $(strip $(foreach d,src tests examples,$(if $(wildcard $(PROJECT_ROOT)/$(d)/.),$(PROJECT_ROOT)/$(d),)))
# End SECTION: lint/type paths

# === SECTION: project tool owner (managed) ===
# Source: caller-selected uv command; make setup owns environment provisioning.
UV ?= uv
UV_REQUESTED := $(UV)
CALLER_PATH := $(PATH)
CALLER_VIRTUAL_ENV := $(patsubst %/,%,$(VIRTUAL_ENV))
# End SECTION: project tool owner

# === SECTION: profile routing (managed) ===
# Source: repository topology. workspace has .gitmodules; standalone does not.
# Both own their runtime in PROJECT_ROOT.
ifneq ($(filter $(MAKE_PROFILE),workspace standalone),$(MAKE_PROFILE))
$(error Invalid MAKE_PROFILE '$(MAKE_PROFILE)')
endif

RUNTIME_ROOT := $(PROJECT_ROOT)
# End SECTION: profile routing

RUNTIME_VENV := $(RUNTIME_ROOT)/.venv
PROJECT_VENV := $(PROJECT_ROOT)/.venv
FLEXT_INFRA_RUNTIME_ROOT := $(if $(filter $(MAKEFILE_ROOT),$(PROJECT_ROOT)),$(RUNTIME_ROOT),$(MAKEFILE_ROOT))
ifeq ($(OS),Windows_NT)
RUNTIME_BIN := $(RUNTIME_VENV)/Scripts
RUNTIME_PYTHON := $(RUNTIME_BIN)/python.exe
FLEXT_INFRA_RUNTIME_PYTHON := $(FLEXT_INFRA_RUNTIME_ROOT)/.venv/Scripts/python.exe
NORMALIZED_CALLER_PATH := $(shell cygpath --path "$(CALLER_PATH)")
ifneq ($(.SHELLSTATUS),0)
$(error cygpath failed to normalize PATH)
endif
NORMALIZED_CALLER_VIRTUAL_ENV := $(shell cygpath --unix "$(CALLER_VIRTUAL_ENV)")
ifneq ($(.SHELLSTATUS),0)
$(error cygpath failed to normalize VIRTUAL_ENV)
endif
CALLER_VIRTUAL_ENV_BIN := $(NORMALIZED_CALLER_VIRTUAL_ENV)/Scripts
else
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
ifneq ($(filter Y,$(GEN_INIT_ONLY) $(SETUP_BOOTSTRAP_ONLY)),)
RESOLVED_UV :=
else
RESOLVED_UV := $(shell PATH="$(SANITIZED_CALLER_PATH)" command -v "$(UV_REQUESTED)")
ifneq ($(.SHELLSTATUS),0)
$(error Required uv executable not found: $(UV_REQUESTED))
endif
endif
override UV := $(if $(strip $(RESOLVED_UV)),$(RESOLVED_UV),$(UV_REQUESTED))
override FLEXT_INFRA_PYTHON := $(FLEXT_INFRA_RUNTIME_PYTHON)
override UV_PROJECT := $(RUNTIME_ROOT)
override UV_PROJECT_ENVIRONMENT := $(RUNTIME_VENV)
override VIRTUAL_ENV := $(RUNTIME_VENV)
override PATH := $(RUNTIME_BIN):$(SANITIZED_CALLER_PATH)
export FLEXT_INFRA_PYTHON UV UV_PROJECT UV_PROJECT_ENVIRONMENT VIRTUAL_ENV PATH

.PHONY: _bootstrap_setup_tools

_bootstrap_setup_tools:
	@set -eu; \
	uv_selector="latest"; \
	if [ ! -f "$(SETUP_MISE)" ]; then \
		printf 'ERROR: missing generated mise launcher: %s; run make gen APPLY=Y\n' "$(SETUP_MISE)" >&2; \
		exit 2; \
	fi; \
	project_root="$(PROJECT_ROOT)"; \
	mise="$(SETUP_MISE)"; \
	mise_storage_root="$(MISE_DATA_DIR)"; \
	caller_home="$${HOME:-}"; \
	caller_xdg_data_home="$${XDG_DATA_HOME:-}"; \
	if [ -z "$$caller_xdg_data_home" ] && [ -n "$$caller_home" ]; then \
		caller_xdg_data_home="$$caller_home/.local/share"; \
	fi; \
	caller_path="$$PATH"; \
caller_comspec="$(COMSPEC)"; \
caller_pathext="$(PATHEXT)"; \
caller_systemroot="$(SYSTEMROOT)"; \
caller_windir="$(WINDIR)"; \
caller_github_token="$(GITHUB_TOKEN)"; \
caller_gh_token="$(GH_TOKEN)"; \
caller_mise_github_token="$(MISE_GITHUB_TOKEN)"; \
caller_mise_github_credential_command="$(MISE_GITHUB_CREDENTIAL_COMMAND)"; \
caller_mise_http_timeout="$(MISE_HTTP_TIMEOUT)"; \
caller_mise_version="$(MISE_VERSION)"; \
if [ -z "$$mise_storage_root" ]; then \
		if [ -n "$$caller_xdg_data_home" ]; then \
			mise_storage_root="$$caller_xdg_data_home/mise"; \
		elif [ -n "$$caller_home" ]; then \
			mise_storage_root="$$caller_home/.local/share/mise"; \
		else \
			printf 'ERROR: MISE_DATA_DIR, XDG_DATA_HOME, or HOME must identify persistent Mise storage\n' >&2; \
			exit 2; \
		fi; \
	fi; \
	case "$$mise_storage_root" in \
		/*) ;; \
		*) printf 'ERROR: MISE_DATA_DIR must be absolute: %s\n' "$$mise_storage_root" >&2; exit 2 ;; \
	esac; \
	case "$$mise_storage_root/" in \
		*'/../'*|*'/./'*|*'//'*) printf 'ERROR: MISE_DATA_DIR must be normalized: %s\n' "$$mise_storage_root" >&2; exit 2 ;; \
	esac; \
	project_root=$$(cd "$$project_root" && pwd -P); \
	project_parent=$${project_root%/*}; \
	if [ -z "$$project_parent" ]; then project_parent=/; fi; \
	if [ -L "$$mise_storage_root" ]; then \
		printf 'ERROR: MISE_DATA_DIR must not be a symlink: %s\n' "$$mise_storage_root" >&2; \
		exit 2; \
	fi; \
	umask 077; \
	mkdir -p "$$mise_storage_root"; \
	if [ -L "$$mise_storage_root" ]; then \
		printf 'ERROR: MISE_DATA_DIR became a symlink: %s\n' "$$mise_storage_root" >&2; \
		exit 2; \
	fi; \
	mise_storage_root=$$(cd "$$mise_storage_root" && pwd -P); \
	case "$$mise_storage_root/" in \
		/tmp/|/tmp/*) printf 'ERROR: persistent Mise storage must not live under /tmp: %s\n' "$$mise_storage_root" >&2; exit 2 ;; \
	esac; \
	case "$$mise_storage_root/" in \
		"$$project_root/"*) printf 'ERROR: persistent Mise storage must be outside the checkout: %s\n' "$$mise_storage_root" >&2; exit 2 ;; \
	esac; \
	case "$$project_root/" in \
		"$$mise_storage_root/"*) printf 'ERROR: persistent Mise storage must not contain the checkout: %s\n' "$$mise_storage_root" >&2; exit 2 ;; \
	esac; \
	for persistent_path in "$$mise_storage_root" "$$mise_storage_root/cache" "$$mise_storage_root/state" "$$mise_storage_root/installs" "$$mise_storage_root/shims" "$$mise_storage_root/bootstrap"; do \
		if [ -L "$$persistent_path" ]; then \
			printf 'ERROR: persistent Mise path must not be a symlink: %s\n' "$$persistent_path" >&2; exit 2; \
		fi; \
		mkdir -p "$$persistent_path"; \
		if [ -L "$$persistent_path" ]; then \
			printf 'ERROR: persistent Mise path became a symlink: %s\n' "$$persistent_path" >&2; exit 2; \
		fi; \
		persistent_physical=$$(cd "$$persistent_path" && pwd -P); \
		case "$$persistent_physical" in \
			"$$mise_storage_root"|"$$mise_storage_root"/*) ;; \
			*) printf 'ERROR: persistent Mise path escaped storage: %s\n' "$$persistent_physical" >&2; exit 2 ;; \
		esac; \
	done; \
	scratch_parent="$$project_root/.test-tmp"; \
	if [ -L "$$scratch_parent" ]; then \
		printf 'ERROR: Mise scratch parent must not be a symlink: %s\n' "$$scratch_parent" >&2; exit 2; \
	fi; \
	mkdir -p "$$scratch_parent"; \
	scratch=$$(mktemp -d "$$scratch_parent/mise-toolchain.XXXXXX"); \
	trap 'find "$$scratch" -depth -delete' EXIT; \
	mkdir -p "$$scratch/home" "$$scratch/home" "$$scratch/appdata" "$$scratch/appdata" "$$scratch/xdg-config" "$$scratch/xdg-data" "$$scratch/xdg-cache" "$$scratch/xdg-state" "$$scratch/config" "$$scratch/tmp" "$$scratch/." "$$scratch/system-config" "$$scratch/system-data" "$$scratch/system-installs" "$$scratch/system-shims" "$$scratch/tmp" "$$scratch/tmp" "$$scratch/tmp"; \
: > "$$scratch/global-config.toml"; chmod 600 "$$scratch/global-config.toml"; \
: > "$$scratch/system-config/config.toml"; chmod 600 "$$scratch/system-config/config.toml"; \
: > "$$scratch/gitconfig"; chmod 600 "$$scratch/gitconfig"; \
: > "$$scratch/netrc"; chmod 600 "$$scratch/netrc"; \
mise_exec() { \
		mise_config_mode="$$1"; shift; \
		case "$$mise_config_mode" in \
			no-config) mise_config_argument='MISE_NO_CONFIG=1' ;; \
			project) mise_config_argument= ;; \
			*) printf 'ERROR: invalid Mise config mode: %s\n' "$$mise_config_mode" >&2; return 2 ;; \
		esac; \
		env -i \
'GIT_CONFIG_NOSYSTEM=1' \
'GIT_TERMINAL_PROMPT=0' \
'LANG=C' \
'LC_ALL=C' \
'MISE_SAFE=1' \
'MISE_PARANOID=true' \
'MISE_QUIET=1' \
'MISE_NO_ENV=1' \
'MISE_NO_HOOKS=1' \
'MISE_AUTO_ENV=false' \
'MISE_AUTO_INSTALL=false' \
'MISE_EXEC_AUTO_INSTALL=false' \
'MISE_TASK_RUN_AUTO_INSTALL=false' \
'MISE_AUTO_UPDATE=false' \
'MISE_HTTP_RETRIES=0' \
'MISE_NETRC=false' \
'MISE_NOT_FOUND_AUTO_INSTALL=false' \
'MISE_NOT_FOUND_SYSTEM_FALLBACK=false' \
'MISE_OVERRIDE_CONFIG_FILENAMES=.mise.toml' \
'MISE_OVERRIDE_TOOL_VERSIONS_FILENAMES=none' \
'MISE_GITHUB_GH_CLI_TOKENS=false' \
'MISE_GITHUB_USE_GIT_CREDENTIALS=false' \
'MISE_GITHUB_OAUTH_CLIENT_ID=' \
'MISE_GITHUB_OAUTH_EXPORT_ENV=' \
'MISE_GITHUB_OAUTH_OPEN_BROWSER=false' \
"HOME=$$scratch/home" \
"USERPROFILE=$$scratch/home" \
"APPDATA=$$scratch/appdata" \
"LOCALAPPDATA=$$scratch/appdata" \
"XDG_CONFIG_HOME=$$scratch/xdg-config" \
"XDG_DATA_HOME=$$scratch/xdg-data" \
"XDG_CACHE_HOME=$$scratch/xdg-cache" \
"XDG_STATE_HOME=$$scratch/xdg-state" \
"NETRC=$$scratch/netrc" \
"GIT_CONFIG_GLOBAL=$$scratch/gitconfig" \
"MISE_NETRC_FILE=$$scratch/netrc" \
"MISE_GLOBAL_CONFIG_FILE=$$scratch/global-config.toml" \
"MISE_CONFIG_DIR=$$scratch/config" \
"MISE_TMP_DIR=$$scratch/tmp" \
"MISE_GLOBAL_CONFIG_ROOT=$$scratch/." \
"MISE_SYSTEM_CONFIG_DIR=$$scratch/system-config" \
"MISE_SYSTEM_CONFIG_FILE=$$scratch/system-config/config.toml" \
"MISE_SYSTEM_DATA_DIR=$$scratch/system-data" \
"MISE_SYSTEM_INSTALLS_DIR=$$scratch/system-installs" \
"MISE_SYSTEM_SHIMS_DIR=$$scratch/system-shims" \
"TMPDIR=$$scratch/tmp" \
"TMP=$$scratch/tmp" \
"TEMP=$$scratch/tmp" \
"MISE_DATA_DIR=$$mise_storage_root" \
"MISE_CACHE_DIR=$$mise_storage_root/cache" \
"MISE_STATE_DIR=$$mise_storage_root/state" \
"MISE_INSTALLS_DIR=$$mise_storage_root/installs" \
"MISE_SHIMS_DIR=$$mise_storage_root/shims" \
"GIT_CEILING_DIRECTORIES=$$project_parent" \
			"MISE_CEILING_PATHS=$$project_parent" \
			"MISE_TRUSTED_CONFIG_PATHS=$$project_root" \
$${caller_path:+"PATH=$$caller_path"} \
$${caller_comspec:+"COMSPEC=$$caller_comspec"} \
$${caller_pathext:+"PATHEXT=$$caller_pathext"} \
$${caller_systemroot:+"SYSTEMROOT=$$caller_systemroot"} \
$${caller_windir:+"WINDIR=$$caller_windir"} \
$${caller_github_token:+"GITHUB_TOKEN=$$caller_github_token"} \
$${caller_gh_token:+"GH_TOKEN=$$caller_gh_token"} \
$${caller_mise_github_token:+"MISE_GITHUB_TOKEN=$$caller_mise_github_token"} \
$${caller_mise_github_credential_command:+"MISE_GITHUB_CREDENTIAL_COMMAND=$$caller_mise_github_credential_command"} \
$${caller_mise_http_timeout:+"MISE_HTTP_TIMEOUT=$$caller_mise_http_timeout"} \
$${caller_mise_version:+"MISE_VERSION=$$caller_mise_version"} \
$${mise_config_argument:+"$$mise_config_argument"} \
			"$$@"; \
	}; \
	mise_checked() { \
		mise_log="$$1"; shift; \
		printf 'setup probe: begin stage=%s log=%s\n' "$${mise_log##*/}" "$$mise_log" >&2; \
		if "$$@" >"$$mise_log" 2>&1; then :; \
		else mise_status=$$?; cat "$$mise_log"; printf 'setup probe: failed stage=%s exit=%s\n' "$${mise_log##*/}" "$$mise_status" >&2; return "$$mise_status"; fi; \
		cat "$$mise_log"; \
		if grep -Fq 'mise WARN' "$$mise_log"; then \
			printf 'ERROR: Mise emitted a warning\n' >&2; return 2; \
		fi; \
		printf 'setup probe: end stage=%s exit=0\n' "$${mise_log##*/}" >&2; \
	}; \
	mise_checked_stdout() { \
		mise_stdout_log="$$1"; mise_stderr_log="$$2"; shift 2; \
		if "$$@" >"$$mise_stdout_log" 2>"$$mise_stderr_log"; then :; \
		else mise_status=$$?; cat "$$mise_stderr_log" >&2; cat "$$mise_stdout_log"; return "$$mise_status"; fi; \
		cat "$$mise_stderr_log" >&2; cat "$$mise_stdout_log"; \
		if grep -Fq 'mise WARN' "$$mise_stderr_log" || grep -Fq 'mise WARN' "$$mise_stdout_log"; then \
			printf 'ERROR: Mise emitted a warning\n' >&2; return 2; \
		fi; \
	}; \
	latest_mise="$$mise"; \
	mise_checked_stdout "$$scratch/runtime-version.stdout" "$$scratch/runtime-version.stderr" mise_exec no-config "$$latest_mise" --version; \
	receipt_runtime=$$(cat "$$scratch/runtime-version.stdout"); \
	case "$$receipt_runtime" in \
		'mise '*) runtime_release=$${receipt_runtime#mise }; runtime_release=$${runtime_release%% *} ;; \
		*) runtime_release=$${receipt_runtime%% *} ;; \
	esac; \
	case "$$runtime_release" in \
		''|*[!0-9.]*|.*|*.|*..*) printf 'ERROR: Mise receipt returned invalid version: %s\n' "$$receipt_runtime" >&2; exit 2 ;; \
	esac; \
	old_ifs=$$IFS; IFS=.; set -- $$runtime_release; IFS=$$old_ifs; \
	if [ "$$#" -ne 3 ]; then \
		printf 'ERROR: Mise receipt returned invalid version: %s\n' "$$receipt_runtime" >&2; exit 2; \
	fi; \
	printf 'mise setup receipt=%s storage=%s\n' "$$runtime_release" "$$mise_storage_root"; \
	mise_checked "$$scratch/install.log" mise_exec project "$$latest_mise" -C "$$project_root" install --yes; \
	mise_checked "$$scratch/uv-version.log" mise_exec project "$$latest_mise" -C "$$project_root" exec -- uv --version; \
	uv_output=$$(cat "$$scratch/uv-version.log"); \
	case "$$uv_output" in \
		'uv '*) uv_actual=$${uv_output#uv }; uv_actual=$${uv_actual%% *} ;; \
		*) printf 'ERROR: uv --version returned an invalid value\n' >&2; exit 2 ;; \
	esac; \
	case "$$uv_actual" in \
		''|*[!0-9.]*|.*|*.|*..*) printf 'ERROR: uv --version returned an invalid release: %s\n' "$$uv_actual" >&2; exit 2 ;; \
	esac; \
	old_ifs=$$IFS; IFS=.; set -- $$uv_actual; IFS=$$old_ifs; \
	if [ "$$#" -ne 3 ]; then \
		printf 'ERROR: uv --version returned an invalid release: %s\n' "$$uv_actual" >&2; exit 2; \
	fi; \
	printf 'uv setup selector=%s receipt=%s\n' "$$uv_selector" "$$uv_actual"; \
	mise_checked "$$scratch/direnv-path.log" mise_exec project "$$latest_mise" -C "$$project_root" which direnv; \
	direnv_executable=$$(cat "$$scratch/direnv-path.log"); \
	if [ ! -x "$$direnv_executable" ]; then \
		printf 'ERROR: Mise resolved a non-executable direnv path: %s\n' "$$direnv_executable" >&2; exit 2; \
	fi; \
	if [ -n "$${GITHUB_PATH:-}" ]; then \
		printf '%s\n' "$$project_root/bin" >> "$$GITHUB_PATH"; \
printf '%s\n' "$$mise_storage_root/shims" >> "$$GITHUB_PATH"; \
fi; \
	printf 'setup: entering lifecycle (submodules, environment, hooks)\n'; \
	mise_exec project "$$latest_mise" -C "$$project_root" exec -- env "SETUP_DIRENV=$$direnv_executable" "SETUP_DIRENV_XDG_DATA_HOME=$$caller_xdg_data_home" $(SELF_MAKE) _setup_lifecycle

ifeq ($(MAKE_PROFILE),workspace)
CODEGEN_SCOPE := all
ALLOWED_PROJECTS := . $(WORKSPACE_SUBPROJECTS)
else
CODEGEN_SCOPE := self
ALLOWED_PROJECTS := .
endif

# Workspace-root gate verbs fan out across declared members through the generic
# `flext-infra workspace orchestrate` primitive (verb allowlist + CLI group come
# from the constants SSOT, never hardcoded here). Members and standalone projects
# run the gate locally. FAIL_FAST forwards the stop-on-first-failure policy.
# Provisioning is declared once and shared by every profile. Creating a missing
# venv is provisioning; clearing a present one is destruction, so it never happens.
# A symlinked RUNTIME_VENV is a BORROWED environment: a linked worktree (a
# lane checkout) shares the primary checkout's environment so the two never
# diverge. Syncing it would rewrite the editable pointers the owner and every
# sibling lane resolve through, so the borrower provisions nothing and the owner
# stays the only writer.
SETUP_ENVIRONMENT_RECIPE = set -eu; \
	if [ -L "$(RUNTIME_VENV)" ]; then \
		printf 'setup: borrowed environment %s is owned by another checkout\n' "$(RUNTIME_VENV)"; \
	else \
		if [ ! -x "$(RUNTIME_PYTHON)" ]; then \
			$(UV) venv "$(RUNTIME_VENV)"; \
		fi; \
		$(UV) sync --project "$(PROJECT_ROOT)" $(UV_SYNC_FLAGS) --link-mode "$(UV_LINK_MODE)"; \
	fi; \
	XDG_DATA_HOME="$${SETUP_DIRENV_XDG_DATA_HOME:?missing persistent direnv data home}" \
		"$${SETUP_DIRENV:?missing Mise-resolved direnv executable}" allow "$(PROJECT_ROOT)"

# A delegated runtime lives in another checkout, so this project has no local
# environment of its own. Generated tooling still addresses the environment by
# its project-local name (`$${workspaceFolder}/.venv`), which must never be
# rewritten into a cross-project relative hop: the link makes that name resolve.
# Linking is provisioning, so a real local environment is never replaced.
BORROW_RUNTIME_VENV_RECIPE = set -eu; \
	if [ ! -e "$(PROJECT_VENV)" ] || [ -L "$(PROJECT_VENV)" ]; then \
		ln -sfn "$(RUNTIME_VENV)" "$(PROJECT_VENV)"; \
	fi

WORKSPACE_ORCHESTRATE = $(UV_RUN) python -m flext_infra workspace orchestrate
DEFAULT_PROJECTS := $(WORKSPACE_SUBPROJECTS) .
SELECTED_PROJECTS := $(DEFAULT_PROJECTS)
WORKSPACE_PROJECT_ARGS := $(foreach project,$(SELECTED_PROJECTS),--projects $(project))
DOCS_PROJECT_ARGS := $(foreach project,$(SELECTED_PROJECTS),--projects $(project))

# A borrowed RUNTIME_VENV keeps the primary editable install. Clearing
# PYTHONPATH would make `make test` in a linked worktree execute that primary
# tree instead of this checkout. Prefer PROJECT_ROOT/src so the Makefile owner
# always wins over the shared editable (terminus T4 / path-purity).
UV_RUN := env -u MYPYPATH -u VIRTUAL_ENV -u UV_PROJECT -u UV_PROJECT_ENVIRONMENT -u PROJECT_ROOT PYTHONPATH="$(PROJECT_ROOT)/src" $(UV) run --project "$(RUNTIME_ROOT)" --no-sync
PROJECT_INFRA_PYTHONPATH ?= $(MAKEFILE_ROOT)/src
PROJECT_FLEXT_INFRA := if [ ! -x "$(FLEXT_INFRA_PYTHON)" ]; then printf 'ERROR: FLEXT_INFRA_PYTHON must name an executable managed Python\n' >&2; exit 2; fi; env -u PYTHONPATH -u MYPYPATH -u VIRTUAL_ENV -u UV_PROJECT -u UV_PROJECT_ENVIRONMENT PATH="$(dir $(FLEXT_INFRA_PYTHON)):$(SANITIZED_CALLER_PATH)" PYTHONPATH="$(PROJECT_INFRA_PYTHONPATH)" $(FLEXT_INFRA_PYTHON) -m flext_infra
# Scaffold dev tools live in the validated optional dev
# profile; a fresh project must create its lock before later check-mode locks.
# Keyed on the environment's OWNER, not on the caller's profile. A member has
# no local venv -- RUNTIME_VENV is RUNTIME_ROOT/.venv -- so every checkout that
# provisions a shared environment must describe the same contents. A member
# syncing without --all-packages treats the siblings already installed there as
# surplus and uninstalls them, undoing the root's provisioning and leaving
# `uv sync --check` permanently divergent. A standalone project owns its venv
# alone and has no workspace packages to include.
SHARED_RUNTIME := $(if $(filter-out $(PROJECT_ROOT),$(RUNTIME_ROOT)),1,$(if $(strip $(WORKSPACE_SUBPROJECTS)),1,))
UV_SYNC_FLAGS := $(if $(SHARED_RUNTIME),--all-packages ,)--all-extras --all-groups

-include custom.mk
SELF_MAKE := $(MAKE) --no-print-directory -f "$(SELF_MAKEFILE)"

define RUN_PUBLIC
	$(if $(filter pre-$(1),$(CUSTOM_DECLARED_TARGETS)),+@$(SELF_MAKE) pre-$(1))
	$(if $(filter _custom-$(1),$(CUSTOM_DECLARED_TARGETS)),+@$(SELF_MAKE) _custom-$(1),+@$(SELF_MAKE) _builtin-$(1))
	$(if $(filter post-$(1),$(CUSTOM_DECLARED_TARGETS)),+@$(SELF_MAKE) post-$(1))
endef

define _require_apply
	@if [ "$(APPLY)" != "Y" ]; then \
		printf 'ERROR: this action requires APPLY=Y\n' >&2; \
		exit 2; \
	fi
endef

define _run_for_all_projects
	@set -eu; \
	for project in $(SELECTED_PROJECTS); do \
		if [ "$$project" = "." ]; then project_root="$(PROJECT_ROOT)"; \
		else project_root="$(PROJECT_ROOT)/$$project"; fi; \
		$(UV) lock --project "$$project_root" $(1); \
	done
endef

.PHONY: $(PUBLIC_VERBS) $(addprefix _builtin-,$(PUBLIC_VERBS))


help:
	$(call RUN_PUBLIC,help)

deps: _builtin_require_environment
	$(call _require_apply)
	$(call RUN_PUBLIC,deps)

build: _builtin_require_environment
	$(call _require_apply)
	$(call RUN_PUBLIC,build)

check: _builtin_require_environment
	$(call _require_apply)
	$(call RUN_PUBLIC,check)

test: _builtin_require_environment
	$(call _require_apply)
	$(call RUN_PUBLIC,test)

fmt: _builtin_require_environment
	$(call _require_apply)
	$(call RUN_PUBLIC,fmt)

fix: _builtin_require_environment
	$(call _require_apply)
	$(call RUN_PUBLIC,fix)

fix-enforcement: _builtin_require_environment
	$(call _require_apply)
	$(call RUN_PUBLIC,fix-enforcement)

audit: _builtin_require_environment
	$(call RUN_PUBLIC,audit)

status: _builtin_require_environment
	$(call RUN_PUBLIC,status)

docs: _builtin_require_environment
	$(call _require_apply)
	$(call RUN_PUBLIC,docs)

clean: _builtin_require_environment
	$(call _require_apply)
	$(call RUN_PUBLIC,clean)

release-plan: _builtin_require_environment
	$(call _require_apply)
	$(call RUN_PUBLIC,release-plan)

release-version: _builtin_require_environment
	$(call _require_apply)
	$(call RUN_PUBLIC,release-version)

release-tag: _builtin_require_environment
	$(call _require_apply)
	$(call RUN_PUBLIC,release-tag)

release-build: _builtin_require_environment
	$(call _require_apply)
	$(call RUN_PUBLIC,release-build)

publication: _builtin_require_environment
	$(call _require_apply)
	$(call RUN_PUBLIC,publication)

gen: _builtin_require_environment
	$(call _require_apply)
	$(call RUN_PUBLIC,gen)

conform: _builtin_require_environment
	$(call _require_apply)
	$(call RUN_PUBLIC,conform)

initialize: _builtin_require_environment
	$(call _require_apply)
	$(call RUN_PUBLIC,initialize)

mod: _builtin_require_environment
	$(call _require_apply)
	$(call RUN_PUBLIC,mod)

waza: _builtin_require_environment
	$(call _require_apply)
	$(call RUN_PUBLIC,waza)

duplication: _builtin_require_environment
	$(call _require_apply)
	$(call RUN_PUBLIC,duplication)


# Repository-owned extra verbs dispatch exactly like canonical ones: the
# project declares them (help, .PHONY) and must also be able to run them.


# `setup` keeps its own recipe (it must not require the environment it is about
# to build), but it still runs the pre-/post-setup lifecycle hooks so a project
# declaring them in the custom handler surface is actually honoured.
setup: _bootstrap_setup_tools

.PHONY: _setup_lifecycle
_setup_lifecycle:
	@set -eu; \
	case " $(CUSTOM_DECLARED_TARGETS) " in \
		*" pre-setup "*) $(SELF_MAKE) pre-setup ;; \
	esac
	@$(SELF_MAKE) _builtin_setup_environment
	@set -eu; \
	case " $(CUSTOM_DECLARED_TARGETS) " in \
		*" post-setup "*) $(SELF_MAKE) post-setup ;; \
	esac

_builtin-help:
	@printf '%s\n' 'flext-core [standalone]' '';

	@printf '  %-16s %s\n' 'help' 'Show the complete selector-free public interface.';

	@printf '  %-16s %s\n' 'setup' 'Provision the declared environment and hooks.';

	@printf '  %-16s %s (APPLY=Y)\n' 'deps' 'Upgrade, lock, and conform every declared dependency.';

	@printf '  %-16s %s (APPLY=Y)\n' 'build' 'Build the project distribution artifacts.';

	@printf '  %-16s %s (APPLY=Y)\n' 'check' 'Run every configured non-test gate.';

	@printf '  %-16s %s (APPLY=Y)\n' 'test' 'Run the complete suite through the persistent testmon cache.';

	@printf '  %-16s %s (APPLY=Y)\n' 'fmt' 'Apply ruff format --preview and ruff check --fix --unsafe-fixes --preview. Ruff is the rule; change code, never ruff.';

	@printf '  %-16s %s (APPLY=Y)\n' 'fix' 'Apply ruff check --fix --unsafe-fixes --preview plus every other configured safe correction. Ruff is the rule; change code, never ruff.';

	@printf '  %-16s %s (APPLY=Y)\n' 'fix-enforcement' 'Apply the safe fix actions declared by the enforcement catalog.';

	@printf '  %-16s %s\n' 'audit' 'Inspect ownership, dependency, and generated-state health.';

	@printf '  %-16s %s\n' 'status' 'Report the resolved runtime and repository state.';

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


# A project owns the sources declared by its manifest. The generated setup
# reconciler validates every initialized checkout before mutation, initializes
# only missing modules, and preserves declared branches that fix forward beyond
# the recorded gitlink.
.PHONY: _builtin_setup_submodules

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
#       A present checkout may be on its own named change lane. Its branch name is
#       not a safety boundary: exact containment of the recorded gitlink is.
#       Nested gitlinks belong to their own setup.
# Free: no
# End SECTION: submodule setup
_builtin_setup_submodules:
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
		head=$$(git -C "$$child_root" rev-parse HEAD); \
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

_builtin_require_environment:
# Documenting the interface (`make help`) must not require the interpreter it
# tells the operator how to provision. Only `make help` with no other goal
# skips the check; any combined goal still demands the environment.
ifneq ($(MAKECMDGOALS),help)
	@if [ ! -x "$(RUNTIME_PYTHON)" ]; then \
		printf 'ERROR: missing environment interpreter %s; make setup creates it\n' "$(RUNTIME_PYTHON)" >&2; \
		exit 2; \
	fi
endif

# === SECTION: setup environment (managed) ===
# Source: computed (MAKE_PROFILE routing) + operator contract (mro-e9j0.6 C7)
# Operator contract: setup PROVISIONS tooling only — mise, venv, dependencies.
# It never generates, conforms, or mutates project code; `make gen` (APPLY=Y)
# is the single public conformance/generation surface.
# Setup always reconciles directly from the lock. The venv is created when
# missing and is never cleared while present, because a concurrent lane may be
# running against it.
ifeq ($(MAKE_PROFILE),workspace)
_builtin_setup_environment: _builtin_setup_submodules
	@$(SETUP_ENVIRONMENT_RECIPE)
	@$(UV) pip check --python "$(RUNTIME_VENV)"
else
_builtin_setup_environment: _builtin_setup_submodules
	@$(SETUP_ENVIRONMENT_RECIPE)
endif
# End SECTION: setup environment

_builtin_deps_check: _builtin_require_environment
	$(call _run_for_all_projects,--check)

_builtin_deps_lock:
	$(call _require_apply)
	$(call _run_for_all_projects,)

_builtin_deps_upgrade: _builtin_require_environment
	$(call _require_apply)
	# Branch-tracked git dependencies are moving sources by declaration
	# (workspace.yaml owns the branch): --refresh re-reads their metadata so a
	# stale cached requires-dist can never block or skew the resolution
	# (flext-62fbu). The cooldown, not the cache, governs version movement.
	$(call _run_for_all_projects,--upgrade --refresh)
	@set -eu; \
	selected="$(strip $(SELECTED_PROJECTS))"; \
	if [ -z "$$selected" ]; then selected="."; fi; \
	set --; \
	for project in $$selected; do set -- "$$@" --projects "$$project"; done; \
	$(PROJECT_FLEXT_INFRA) deps modernize --repository-root "$(PROJECT_ROOT)" \
		--apply --rewrite-constraints --skip-check "$$@"
	$(call _run_for_all_projects,--check)


_builtin_build_artifacts:
	@$(UV) build --project "$(PROJECT_ROOT)"

# `check` is read-only by contract: it never mutates the tree. Fixing is owned
# by `make fix APPLY=Y` and formatting by `make fmt APPLY=Y`, both run BEFORE
# check. APPLY here made the same tools run twice with conflicting intents,
# so it is rejected instead of silently honoured; FIX=1 became the `fix` verb.
# CI=Y keeps make.ci.check_gates, the strict complement of
# make.ci.local_check_gates.
_builtin_check_all: _builtin_require_environment
	@set -eu; \
		gates="lint,pyrefly,mypy,pyright,silent-failure,deferred-self-reference,security,markdown,loc-cap,boundary,runtime-census,namespace,tier-whitelist,smells,codemod,layout,canonical-alias,direnv,duplication"; \
		if [ "$(strip $(CI))" = "Y" ]; then \
			gates="lint,pyright,silent-failure,deferred-self-reference,security,markdown,loc-cap,boundary,runtime-census,namespace,tier-whitelist,smells,codemod,layout,canonical-alias,direnv,duplication"; \
			printf 'INFO: CI=Y runs check gates: lint pyright silent-failure deferred-self-reference security markdown loc-cap boundary runtime-census namespace tier-whitelist smells codemod layout canonical-alias direnv duplication\n'; \
		fi; \
		if [ -z "$$gates" ]; then \
		printf 'ERROR: no check gates remain after CI=Y filtering\n' >&2; \
		exit 2; \
	fi; \
	$(PROJECT_FLEXT_INFRA) check run --repository-root "$(PROJECT_ROOT)" --gates "$$gates" --projects .

_builtin_test_all: _builtin_require_environment

	@set -eu; \
		test_tmp_parent="$(PROJECT_SCRATCH_ROOT)/pytest"; \
		mkdir -p "$$test_tmp_parent"; \
		test_tmp=$$(mktemp -d "$$test_tmp_parent/invocation.XXXXXX"); \
		cleanup_test_tmp() { rm -rf "$$test_tmp"; }; \
		trap cleanup_test_tmp EXIT INT TERM; \
		TMPDIR="$$test_tmp" GOTMPDIR="$$test_tmp" $(PYTEST_BOUNDED) $(UV_RUN) python -m flext_infra._pytest_entry

# Ruff is the style/autofix rule (make.ruff in codegen.yaml). Every
# invocation uses --preview. fmt APPLY also runs check --fix --unsafe-fixes
# --preview. Never weaken ruff to keep a file; change the code.
_builtin_fmt_check: _builtin_require_environment
	@$(UV_RUN) ruff format --preview --check $(RUFF_PATHS)

_builtin_fmt_all: _builtin_require_environment
	$(call _require_apply)
	@$(UV_RUN) ruff format --preview $(RUFF_PATHS)
	@$(UV_RUN) ruff check --preview --fix --unsafe-fixes $(RUFF_PATHS)

_builtin_fmt_apply: _builtin_fmt_all

_builtin_fix_check: _builtin_require_environment
	@$(UV_RUN) ruff check --preview --no-fix $(RUFF_PATHS)

_builtin_fix_all: _builtin_require_environment
	$(call _require_apply)
	@$(UV_RUN) ruff check --preview --fix --unsafe-fixes $(RUFF_PATHS)
	@$(PROJECT_FLEXT_INFRA) check run --repository-root "$(PROJECT_ROOT)" --gates "lint,markdown,canonical-alias,smells" --projects . --apply

_builtin_fix_apply: _builtin_fix_all

# Catalog-driven enforcement fixes: every ENFORCE rule whose fix action is
# declared safe, applied through its registered adapter.
_builtin_fix_enforcement: _builtin_require_environment
	$(call _require_apply)
	@$(PROJECT_FLEXT_INFRA) check fix-enforcement --repository-root "$(PROJECT_ROOT)" --safe-only --apply


_builtin_run_default: _builtin_require_environment
	@$(UV_RUN) $(PROJECT_NAME) $(ARGS)

_builtin_status_diagnostics: _builtin_require_environment
	@printf 'profile=%s\nproject=%s\nruntime=%s\n' \
		'$(MAKE_PROFILE)' '$(PROJECT_ROOT)' '$(RUNTIME_ROOT)'
	@$(UV) --version
	@$(UV) lock --project "$(PROJECT_ROOT)" --check
	@if [ -x "$(RUNTIME_PYTHON)" ]; then \
		$(UV) pip check --python "$(RUNTIME_VENV)"; \
	fi
	@git -C "$(PROJECT_ROOT)" status --short

_builtin_docs_all:
	@set -eu; \
	for action in $(DOCS_ACTIONS); do \
		case "$$action" in fix) mode=$(if $(filter Y,$(APPLY)),--apply,--check) ;; *) mode= ;; esac; \
		$(PROJECT_FLEXT_INFRA) docs "$$action" --repository-root "$(PROJECT_ROOT)" --output-dir "$(PROJECT_ROOT)/.reports/docs" $$mode $(DOCS_PROJECT_ARGS); \
	done

_builtin_clean_generated:
	$(call _require_apply)

	@find "$(PROJECT_ROOT)" -type d \
		\( -name __pycache__ -o -name .mypy_cache -o -name .pytest_cache -o -name .ruff_cache -o -name .pyrefly_cache -o -name .benchmarks -o -name .hypothesis \) \
		-prune -exec sh -eu -c 'for target do find "$$target" -depth -delete; done' sh {} +


	@set -eu; \
	for target in "$(PROJECT_ROOT)/.test-tmp" "$(PROJECT_ROOT)/.test-runtime" "$(PROJECT_ROOT)/build" "$(PROJECT_ROOT)/dist" "$(PROJECT_ROOT)/htmlcov" "$(PROJECT_ROOT)/.reports"; do \
		if [ -e "$$target" ]; then find "$$target" -depth -delete; \
		elif [ -L "$$target" ]; then find "$$target" -depth -delete; fi; \
	done


	@set -eu; \
	for target in "$(PROJECT_ROOT)/.coverage" "$(PROJECT_ROOT)/.testmondata"; do \
		if [ -e "$$target" ]; then rm -- "$$target"; \
		elif [ -L "$$target" ]; then rm -- "$$target"; fi; \
	done


	@find "$(PROJECT_ROOT)" -type f \
		\( -name '*.pstats' \) \
		-delete


# Release protocol. `plan` derives the next version from merged pull-request
# titles and guards against any version change made outside the protocol;
# `version` opens the release pull request; `tag` marks the merged release
# commit; `build` writes the artifact receipt; `publish` uploads exactly what
# the receipt attests (INDEX=Y adds the package index).
_builtin_release_plan: _builtin_require_environment
	@$(PROJECT_FLEXT_INFRA) release run --phase plan $(if $(strip $(PR_TITLE)),--pr-title "$(PR_TITLE)")

_builtin_release_version: _builtin_require_environment
	$(call _require_apply)
	@$(PROJECT_FLEXT_INFRA) release run --phase version --apply

_builtin_release_tag: _builtin_require_environment
	$(call _require_apply)
	@$(PROJECT_FLEXT_INFRA) release run --phase tag --apply

_builtin_release_build: _builtin_require_environment
	@$(PROJECT_FLEXT_INFRA) release run --phase build --apply

_builtin_release_publish: _builtin_require_environment
	$(call _require_apply)
	@$(PROJECT_FLEXT_INFRA) release run --phase publish --apply $(if $(filter Y,$(INDEX)),--index)

# Generation has one transaction owner. Conform preserves the caller's scope and
# journals ordinary, Mise, lazy-init, and documentation phases through one fixed
# point. Dependency upgrades remain a separate explicit verb because they rewrite
# lock floors; gen never runs another writer before or after conform's journal.
_builtin_gen_check: _builtin_require_environment
	@$(PROJECT_FLEXT_INFRA) codegen conform --root "$(PROJECT_ROOT)" --scope "$(CODEGEN_SCOPE)" --mode check

_builtin_gen_init:
	$(call _require_apply)
	@$(PROJECT_FLEXT_INFRA) codegen init --repository-root "$(PROJECT_ROOT)" --apply
	@$(PROJECT_FLEXT_INFRA) codegen init --repository-root "$(PROJECT_ROOT)" --check

_builtin_gen_all:
	$(call _require_apply)
	@$(PROJECT_FLEXT_INFRA) codegen conform --root "$(PROJECT_ROOT)" --scope "$(CODEGEN_SCOPE)" --mode apply

_builtin_gen_apply: _builtin_gen_all

# Structural rewrites have one selector-free public Make surface. The current
# directory defines scope; callers never address ast-grep, Rope, or LSP directly.
_builtin_mod_apply: _builtin_require_environment
	@$(PROJECT_FLEXT_INFRA) refactor mod --apply

# Selector-free public verbs map one-to-one to their canonical implementation.
_builtin-deps: _builtin_deps_upgrade
_builtin-build: _builtin_build_artifacts
_builtin-check: _builtin_check_all
_builtin-test: _builtin_test_all
_builtin-fmt: _builtin_fmt_all
_builtin-fix: _builtin_fix_all
_builtin-fix-enforcement: _builtin_fix_enforcement
_builtin-audit:
	@if [ "$(MAKE_PROFILE)" = "workspace" ]; then \
		$(UV) lock --project "$(PROJECT_ROOT)" --check; \
	else \
		printf '%s\n' "audit: dependency truth is owned by the fleet workspace lock (design B, flext-62fbu); run the audit at the fleet root."; \
	fi
	@$(UV) pip check --python "$(RUNTIME_VENV)"
	@$(PROJECT_FLEXT_INFRA) codegen conform --root "$(PROJECT_ROOT)" --scope "$(CODEGEN_SCOPE)" --mode check
_builtin-status: _builtin_status_diagnostics
_builtin-docs: _builtin_docs_all
_builtin-clean: _builtin_clean_generated
_builtin-release-plan: _builtin_release_plan
_builtin-release-version: _builtin_release_version
_builtin-release-tag: _builtin_release_tag
_builtin-release-build: _builtin_release_build
_builtin-publication: _builtin_release_publish
_builtin-gen: _builtin_gen_all
_builtin-conform: _builtin_gen_check
_builtin-initialize: _builtin_gen_init
_builtin-mod: _builtin_mod_apply
_builtin-waza:
	@cd "$(PROJECT_ROOT)" && "$(SETUP_MISE)" exec -- waza check --no-update-check
_builtin-duplication:
	@$(PROJECT_FLEXT_INFRA) check run --repository-root "$(PROJECT_ROOT)" --gates duplication --projects .
