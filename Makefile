

# @flext-generated: continuous
# @flext-owner: flext-infra/config/codegen.yaml + flext-infra/src/flext_infra/templates/project/base/Makefile.j2
# @flext-adjust: edit the owner configuration or template; never this projection
# @flext-regenerate: make gen
# flext-core — generated project interface.
# Managed by flext-infra codegen conform for new and existing repositories.
# === SECTION: header (managed) ===
# Source: template (base/Makefile.j2)
# Free: no
# End SECTION: header

override SHELL := /bin/sh
ifneq ($(strip $(MAKEFILES)),)
$(error MAKEFILES makes the selected project Makefile ambiguous; invoke one project Makefile directly)
endif
# GNU MAKE_COMMAND may be a bare name. Resolve it before changing PATH so
# recursive lifecycle calls keep this invoker instead of selecting a Mise shim.
ifneq ($(filter /%,$(MAKE_COMMAND)),)
override SELF_MAKE_EXECUTABLE := $(MAKE_COMMAND)
else
override SELF_MAKE_EXECUTABLE := $(shell command -v "$(MAKE_COMMAND)")
ifneq ($(.SHELLSTATUS),0)
$(error Cannot resolve current Make executable: $(MAKE_COMMAND))
endif
endif
override SELF_MAKE_EXECUTABLE := $(shell set -eu; \
physical_file="$(SELF_MAKE_EXECUTABLE)"; \
if [ -z "$${physical_file}" ]; then printf 'ERROR: physical file discovery requires one nonempty path\n' >&2; exit 2; fi; \
set --; \
while :; do \
	case "$${physical_file}" in (/*) ;; (*) physical_file="$$(pwd -P)/$${physical_file}" ;; esac; \
	physical_parent="$$(dirname "$${physical_file}")"; \
	physical_name="$${physical_file##*/}"; \
	physical_parent="$$(cd "$${physical_parent}" && pwd -P)"; \
	physical_file="$${physical_parent}/$${physical_name}"; \
	if [ ! -L "$${physical_file}" ]; then \
		if [ ! -f "$${physical_file}" ]; then printf 'ERROR: physical file is not an existing regular file: %s\n' "$${physical_file}" >&2; exit 2; fi; \
		printf '%s\n' "$${physical_file}"; exit 0; \
	fi; \
	for previous_file; do \
		if [ "$${previous_file}" = "$${physical_file}" ]; then printf 'ERROR: physical file link cycle: %s\n' "$${physical_file}" >&2; exit 2; fi; \
	done; \
	set -- "$$@" "$${physical_file}"; \
	link_target="$$(readlink "$${physical_file}")"; \
	case "$${link_target}" in (/*) physical_file="$${link_target}" ;; (*) physical_file="$${physical_parent}/$${link_target}" ;; esac; \
done)
ifneq ($(.SHELLSTATUS),0)
$(error Cannot resolve physical Make executable: $(MAKE_COMMAND))
endif
ifeq ($(strip $(SELF_MAKE_EXECUTABLE)),)
$(error Current Make executable has no physical path: $(MAKE_COMMAND))
endif
.DEFAULT_GOAL := help

# Contract display never provisions or requires native workspace tools.
VERB_CONTRACT := $(filter 1 TRUE Y YES true y yes True Yes,$(OPTIONS) $(HELP))

# Capture the selected approval mode before any project-owned include.
ifeq ($(strip $(CI)),Y)
override APPROVAL_CONTEXT := Y
ifneq ($(filter $(if $(VERB_CONTRACT),,upg lock) _upg% _lock% dep propagate,$(MAKECMDGOALS)),)
$(error Resolution and member propagation are forbidden in CI)
endif
endif

# One GitHub credential from declared sources, each selected once before any
# effect (config make.github_credential_command): the caller's exported token,
# else, outside CI, the declared command when its executable is installed, else
# anonymous. A selected command that fails or prints nothing stops the network
# verbs (_builtin_require_github_credential), never degrading to anonymous.
# The value stays in the environment and is never printed.
GITHUB_TOKEN := $(firstword $(GITHUB_TOKEN) $(GH_TOKEN) $(MISE_GITHUB_TOKEN))
GITHUB_CREDENTIAL_SOURCE := $(if $(GITHUB_TOKEN),environment,anonymous)
ifeq ($(GITHUB_TOKEN),)
ifneq ($(strip $(CI)),Y)
GITHUB_CREDENTIAL_EXECUTABLE := $(shell command -v gh)
ifneq ($(GITHUB_CREDENTIAL_EXECUTABLE),)
GITHUB_CREDENTIAL_SOURCE := command
GITHUB_TOKEN := $(shell "$(GITHUB_CREDENTIAL_EXECUTABLE)" auth token)
GITHUB_CREDENTIAL_STATUS := $(.SHELLSTATUS)
endif
endif
endif
ifneq ($(GITHUB_TOKEN),)
GH_TOKEN := $(GITHUB_TOKEN)
MISE_GITHUB_TOKEN := $(GITHUB_TOKEN)
export GITHUB_TOKEN GH_TOKEN MISE_GITHUB_TOKEN
else
unexport GITHUB_TOKEN GH_TOKEN MISE_GITHUB_TOKEN
endif
unexport GITHUB_API_TOKEN

# === SECTION: project identity (managed) ===
# Source: config:dist / config:make_profile / config:repository_root_rel / config:uv_link_mode
PROJECT_NAME := flext-core
MAKE_PROFILE := standalone
REPOSITORY_ROOT_REL := .
# === SECTION: workspace subprojects (managed) ===
# Source: config:workspace_subprojects (list), config:workspace_repositories (list)
# Computed: traversal projects observed members; provisioning reads local .gitmodules.
WORKSPACE_SUBPROJECTS :=
# End SECTION: project identity

# === SECTION: public boundary (managed) ===
# The Makefile validates no caller input. Every verb
# always applies; a mistyped variable fails where the verb consumes it, and an
# unconsumed variable is ignored.
export CHECK_PHASE
PYTEST_REPORTS_DIR := .reports/tests
PYTEST_CACHE_HOME = $(if $(strip $(XDG_CACHE_HOME)),$(XDG_CACHE_HOME),$(if $(strip $(HOME)),$(HOME)/.cache,))
# One persistent testmon database per project, shared by every checkout and
# worktree of it (like the Mypy cache): testmon tracks each checkout's source by
# file checksum, so a new lane starts from the project's measured selection
# instead of a cold full inventory.
override FLEXT_PYTEST_TESTMON_DATABASE = $(if $(strip $(PYTEST_CACHE_HOME)),$(PYTEST_CACHE_HOME)/flext/infra/testmon/$(PROJECT_NAME)/.testmondata)
# Storage law: the pytest scratch root sits under the user home, keyed by the
# absolute checkout path, never under /tmp and never inside a versioned tree.
override FLEXT_PYTEST_SCRATCH_ROOT = $(if $(strip $(HOME)),$(HOME)/.local/state/.flext-runtime$(PROJECT_ROOT)/scratch)
override PYTEST_RUN_TIMEOUT_SECONDS := 240
override PYTEST_TERMINATION_GRACE_SECONDS := 2
override PYTEST_TIMEOUT_EXIT_CODE := 124
override PYTEST_ENFORCEMENT_PLUGIN := flext_tests.enforcement_plugin
override PYTEST_PROGRESS_ARGS := --verbose --failed-first
override PYTEST_REPORT_ARGS := -ra --durations=25 --durations-min=0.001 --tb=short
override PYTEST_DIAG_ARGS := -rA --durations=0 --tb=long --showlocals
override PYTEST_PARALLEL_WORKERS := 1
override PYTEST_PARALLEL_WORKER_MEMORY_GB := 2
override PYTEST_PARALLEL_DISTRIBUTION := load
override PYTEST_LOADSCOPE_REORDER := true
override PYTEST_PARALLEL_SCHEDULE_CHUNK := 1
override PYTEST_PROFILE_SORT := cumulative
override PYTEST_PROFILE_LIMIT := 50
override PROCESS_TIMEOUT_COMMAND := timeout
override export FLEXT_PYTEST_REPORTS_RAW := $(value PYTEST_REPORTS_DIR)
# End SECTION: public boundary

# === SECTION: derived paths (managed) ===
# Source: computed (git rev-parse, MAKEFILE_LIST, physical path readers)
# Rule: PROJECT_ROOT is the checkout that OWNS this Makefile, never the caller's
# CWD. Deriving it from `pwd -P` made a member validate whatever tree the
# caller happened to stand in: `make -f <member>/Makefile` invoked from the
# superproject resolved RUFF_PATHS to the SUPERPROJECT's src/tests, so the
# member linted files it does not even contain. With many shared worktrees that
# silently validates the wrong tree.
# Capture the one selected Makefile before any project include. GNU Make's
# filename-list functions split whitespace inside a legitimate physical path.
override SELF_MAKEFILE := $(shell set -eu; \
physical_file="$(MAKEFILE_LIST)"; \
if [ -z "$${physical_file}" ]; then printf 'ERROR: physical file discovery requires one nonempty path\n' >&2; exit 2; fi; \
set --; \
while :; do \
	case "$${physical_file}" in (/*) ;; (*) physical_file="$$(pwd -P)/$${physical_file}" ;; esac; \
	physical_parent="$$(dirname "$${physical_file}")"; \
	physical_name="$${physical_file##*/}"; \
	physical_parent="$$(cd "$${physical_parent}" && pwd -P)"; \
	physical_file="$${physical_parent}/$${physical_name}"; \
	if [ ! -L "$${physical_file}" ]; then \
		if [ ! -f "$${physical_file}" ]; then printf 'ERROR: physical file is not an existing regular file: %s\n' "$${physical_file}" >&2; exit 2; fi; \
		printf '%s\n' "$${physical_file}"; exit 0; \
	fi; \
	for previous_file; do \
		if [ "$${previous_file}" = "$${physical_file}" ]; then printf 'ERROR: physical file link cycle: %s\n' "$${physical_file}" >&2; exit 2; fi; \
	done; \
	set -- "$$@" "$${physical_file}"; \
	link_target="$$(readlink "$${physical_file}")"; \
	case "$${link_target}" in (/*) physical_file="$${link_target}" ;; (*) physical_file="$${physical_parent}/$${link_target}" ;; esac; \
done)
ifneq ($(.SHELLSTATUS),0)
$(error Cannot authenticate selected Makefile: $(MAKEFILE_LIST))
endif
override MAKEFILE_ROOT := $(shell dirname -- "$(SELF_MAKEFILE)")
ifneq ($(.SHELLSTATUS),0)
$(error Cannot resolve selected Makefile directory: $(SELF_MAKEFILE))
endif
override export PROJECT_ROOT := $(MAKEFILE_ROOT)
# The common tests verb always runs the complete configured target. Its suite
# cannot be narrowed by a caller's environment or command-line assignment.
override export FLEXT_PYTEST_TARGET_RAW := tests
# === SECTION: REPOSITORY_ROOT isolation (managed) ===
# Source: authenticated physical Git boundary and governed native UV membership.
# Discovery clears repository selectors inherited from recursive Make before
# electing a new boundary. Git credential configuration remains unchanged.
override GIT_DISCOVERY_ROOT := $(shell set -eu; \
unset GIT_CEILING_DIRECTORIES; \
git_local_keys="$$(git rev-parse --local-env-vars)"; \
for git_local_key in $${git_local_keys}; do \
	case "$${git_local_key}" in (GIT_CONFIG|GIT_CONFIG_PARAMETERS|GIT_CONFIG_COUNT) ;; (*) unset "$${git_local_key}" ;; esac; \
done; \
project_root="$(MAKEFILE_ROOT)"; \
case "$${project_root}" in (/*) ;; (*) printf 'ERROR: workspace discovery requires an absolute project directory: %s\n' "$${project_root}" >&2; exit 2 ;; esac; \
project_root="$$(cd "$${project_root}" && pwd -P)"; \
git_parent="$${project_root}"; \
case "$${git_parent}" in (/*) ;; (*) printf 'ERROR: Git marker discovery requires an absolute physical directory: %s\n' "$${git_parent}" >&2; exit 2 ;; esac; \
while [ "$${git_parent}" != / ] && [ ! -e "$${git_parent}/.git" ] && [ ! -L "$${git_parent}/.git" ]; do \
	next_git_parent="$$(dirname "$${git_parent}")"; \
	if [ "$${next_git_parent}" = "$${git_parent}" ]; then printf 'ERROR: Git marker discovery cannot advance from: %s\n' "$${git_parent}" >&2; exit 2; fi; \
	git_parent="$${next_git_parent}"; \
done; \
if [ -e "$${git_parent}/.git" ] || [ -L "$${git_parent}/.git" ]; then git -C "$${project_root}" rev-parse --resolve-git-dir "$${git_parent}/.git" >/dev/null; fi; \
git_root="$$(git -C "$${project_root}" rev-parse --show-toplevel)"; \
git_root="$$(cd "$${git_root}" && pwd -P)"; \
case "$${project_root}/" in ("$${git_root}/"*) ;; (*) printf 'ERROR: physical Git root does not contain this project: %s\n' "$${git_root}" >&2; exit 2 ;; esac; \
printf '%s\n' "$${git_root}")
ifneq ($(.SHELLSTATUS),0)
$(error Cannot authenticate the physical Git boundary for $(MAKEFILE_ROOT))
endif
override BOOTSTRAP_ROOT := $(shell set -eu; \
unset GIT_CEILING_DIRECTORIES; \
git_local_keys="$$(git rev-parse --local-env-vars)"; \
for git_local_key in $${git_local_keys}; do \
	case "$${git_local_key}" in (GIT_CONFIG|GIT_CONFIG_PARAMETERS|GIT_CONFIG_COUNT) ;; (*) unset "$${git_local_key}" ;; esac; \
done; \
project_root="$(MAKEFILE_ROOT)"; \
git_root="$(GIT_DISCOVERY_ROOT)"; \
attached="$$(git -C "$${git_root}" rev-parse --show-superproject-working-tree)"; \
if [ -n "$${attached}" ] && [ -f "$${attached}/pyproject.toml" ]; then \
	attached="$$(cd "$${attached}" && pwd -P)"; \
	member_keys="$$(git -C "$${attached}" config --file .gitmodules --name-only --get-regexp '^submodule\..*\.path$$')"; \
	for member_key in $${member_keys}; do \
		member_path="$$(git -C "$${attached}" config --file .gitmodules --get "$${member_key}")"; \
		if [ "$${attached}/$${member_path}" -ef "$${git_root}" ]; then printf '%s\n' "$${attached}"; exit 0; fi; \
	done; \
	printf 'ERROR: attached runtime has no original member declaration: %s\n' "$${git_root}" >&2; exit 2; \
fi; \
printf '%s\n' "$${project_root}")
ifneq ($(.SHELLSTATUS),0)
$(error Cannot resolve the physical bootstrap scope for $(MAKEFILE_ROOT))
endif
# Setup and upg provision the toolchain the native workspace reader needs, so
# they elect the physical bootstrap scope without running that reader.
# When attached to a workspace (BOOTSTRAP_ROOT != PROJECT_ROOT), all verbs use
# the workspace root so they share the workspace's .venv. Standalone checkouts
# use the native uv workspace reader for non-setup/upg/help verbs.
ifneq ($(strip $(VERB_CONTRACT) $(filter setup upg,$(MAKECMDGOALS))),)
override REPOSITORY_ROOT := $(BOOTSTRAP_ROOT)
else ifeq ($(strip $(filter-out help _builtin-help _setup_native_prepare pre-upg post-upg,$(MAKECMDGOALS))),)
override REPOSITORY_ROOT := $(BOOTSTRAP_ROOT)
else ifneq ($(BOOTSTRAP_ROOT),$(PROJECT_ROOT))
override REPOSITORY_ROOT := $(BOOTSTRAP_ROOT)
else
override REPOSITORY_ROOT := $(shell set -eu; \
unset GIT_CEILING_DIRECTORIES; \
git_local_keys="$$(git rev-parse --local-env-vars)"; \
for git_local_key in $${git_local_keys}; do \
	case "$${git_local_key}" in (GIT_CONFIG|GIT_CONFIG_PARAMETERS|GIT_CONFIG_COUNT) ;; (*) unset "$${git_local_key}" ;; esac; \
done; \
project_root="$(MAKEFILE_ROOT)"; \
reader_root="$(BOOTSTRAP_ROOT)"; \
scope="$$(git -C "$${reader_root}" rev-parse --show-toplevel)"; \
scope="$$(cd "$${scope}" && pwd -P)"; \
unset MISE_SYSTEM_CONFIG_FILE MISE_ENV MISE_ENV_FILE MISE_PYTHON_UV_VENV_AUTO UV_PROJECT UV_WORKING_DIR UV_CONFIG_FILE UV_PROJECT_ENVIRONMENT UV_PYTHON UV_PYTHON_DOWNLOADS UV_NO_CONFIG UV_NO_SOURCES UV_FROZEN UV_LOCKED VIRTUAL_ENV PYTHONHOME PYTHONPATH; \
MISE_CONFIG_DIR="$${reader_root}/.mise"; \
MISE_SYSTEM_CONFIG_DIR="$${MISE_CONFIG_DIR}"; \
MISE_GLOBAL_CONFIG_FILE="$${MISE_CONFIG_DIR}/config.toml"; \
MISE_CEILING_PATHS="$$(dirname "$${reader_root}")"; \
MISE_AUTO_INSTALL=0; \
MISE_OVERRIDE_CONFIG_FILENAMES=".mise.toml"; \
export MISE_CONFIG_DIR MISE_SYSTEM_CONFIG_DIR MISE_GLOBAL_CONFIG_FILE MISE_CEILING_PATHS MISE_AUTO_INSTALL MISE_OVERRIDE_CONFIG_FILENAMES; \
case "$${OS:-}" in (Windows_NT) reader_mise="$${reader_root}/.bin/mise.exe" ;; (*) reader_mise="$${reader_root}/.bin/mise" ;; esac; \
if [ -L "$${reader_root}/.bin" ] || [ -L "$${reader_mise}" ]; then printf 'ERROR: Mise acquisition must be physical: %s\n' "$${reader_mise}" >&2; exit 2; fi; \
reader_mise="$$(MISE_LOCKED=1 "$${reader_mise}" -C "$${reader_root}" which mise)"; \
if [ "$${OS:-}" = Windows_NT ]; then reader_mise="$$(cygpath --unix "$${reader_mise}")"; fi; \
if [ "$${reader_root}" != "$${project_root}" ]; then \
	owner="$${reader_root}"; \
else \
	uv_binary="$$(MISE_LOCKED=1 "$${reader_mise}" -C "$${reader_root}" which "uv")"; \
	if [ "$${OS:-}" = Windows_NT ]; then uv_binary="$$(cygpath --unix "$${uv_binary}")"; fi; \
	if [ ! -x "$${uv_binary}" ]; then printf 'ERROR: native workspace reader is not provisioned; run make setup\n' >&2; exit 2; fi; \
	owner="$$("$${uv_binary}" workspace dir --project "$${project_root}")"; \
	if [ "$${OS:-}" = Windows_NT ]; then owner="$$(cygpath --unix "$${owner}")"; fi; \
fi; \
case "$${owner}" in (/*) ;; (*) printf 'ERROR: native UV workspace owner is not absolute: %s\n' "$${owner}" >&2; exit 2 ;; esac; \
owner="$$(cd "$${owner}" && pwd -P)"; \
case "$${owner}/" in ("$${scope}/"*) ;; (*) printf 'ERROR: native UV workspace owner leaves its Git scope: %s\n' "$${owner}" >&2; exit 2 ;; esac; \
case "$${project_root}/" in ("$${owner}/"*) ;; (*) printf 'ERROR: native UV workspace owner does not contain this project: %s\n' "$${owner}" >&2; exit 2 ;; esac; \
printf '%s\n' "$${owner}")
ifneq ($(.SHELLSTATUS),0)
$(error Cannot resolve the physical workspace for $(MAKEFILE_ROOT))
endif
endif
override SETUP_BIN := $(BOOTSTRAP_ROOT)/.bin
ifeq ($(OS),Windows_NT)
override BOOTSTRAP_MISE := $(SETUP_BIN)/mise.exe
else
override BOOTSTRAP_MISE := $(SETUP_BIN)/mise
endif
# End SECTION: REPOSITORY_ROOT isolation
# === SECTION: verb dispatch (managed) ===
# Source: config:make.verbs and the canonical gate vocabulary. A verb exists
# only in the profiles it declares (make.verbs[].profiles).
PUBLIC_VERBS := setup upg lock gen gen-scope mod fix fmt check tests tests-full help build clean release-plan release-version release-tag release-build release-publish
SCRIPT_VERBS :=

CUSTOM_MAKEFILE := $(MAKEFILE_ROOT)/custom.mk
override FLEXT_MAKE_EMPTY :=
override FLEXT_MAKE_SPACE := $(FLEXT_MAKE_EMPTY) $(FLEXT_MAKE_EMPTY)
override CUSTOM_MAKEFILE_ESCAPED := $(subst $(FLEXT_MAKE_SPACE),\$(FLEXT_MAKE_SPACE),$(CUSTOM_MAKEFILE))
CUSTOM_DECLARED_TARGETS :=
ifneq ($(shell if [ -f "$(CUSTOM_MAKEFILE)" ]; then printf 'present'; fi),)
CUSTOM_DECLARED_TARGETS := $(shell awk '/^[a-z_][a-z0-9_-]*:/ { target=$$1; sub(/:.*/, "", target); if (!seen[target]++) printf "%s ", target }' "$(CUSTOM_MAKEFILE)")
ifneq ($(.SHELLSTATUS),0)
$(error Failed to inspect custom Make targets in $(CUSTOM_MAKEFILE))
endif
ifeq ($(APPROVAL_CONTEXT),Y)
ifneq ($(filter setup check,$(CUSTOM_DECLARED_TARGETS)),)
$(error Approval stages cannot be replaced by custom targets)
endif

ifneq ($(filter _custom-setup,$(CUSTOM_DECLARED_TARGETS)),)
override CUSTOM_WRAPPER_REFERENCE_COUNT := $(shell awk -v reference="_builtin-setup" 'index($$0, reference) { count++ } END { print count + 0 }' "$(CUSTOM_MAKEFILE)")
ifneq ($(.SHELLSTATUS),0)
$(error Failed to inspect custom approval reference in $(CUSTOM_MAKEFILE))
endif
ifeq ($(CUSTOM_WRAPPER_REFERENCE_COUNT),0)
$(error Approval stage _custom-setup must chain _builtin-setup (wrapper parity; replacements are forbidden))
endif
endif

ifneq ($(filter _custom-check,$(CUSTOM_DECLARED_TARGETS)),)
override CUSTOM_WRAPPER_REFERENCE_COUNT := $(shell awk -v reference="_builtin-check" 'index($$0, reference) { count++ } END { print count + 0 }' "$(CUSTOM_MAKEFILE)")
ifneq ($(.SHELLSTATUS),0)
$(error Failed to inspect custom approval reference in $(CUSTOM_MAKEFILE))
endif
ifeq ($(CUSTOM_WRAPPER_REFERENCE_COUNT),0)
$(error Approval stage _custom-check must chain _builtin-check (wrapper parity; replacements are forbidden))
endif
endif

endif
endif
 # End SECTION: verb dispatch

# === SECTION: lint/type paths (managed) ===
# Source: template + computed (script_dispatch conditional)
RUFF_PATHS := $(shell for directory in src tests examples scripts; do if [ -e "$(PROJECT_ROOT)/$$directory" ]; then printf '"%s" ' "$(PROJECT_ROOT)/$$directory"; fi; done)
MYPY_PATHS := $(RUFF_PATHS)
# End SECTION: lint/type paths

# === SECTION: project tool owner (managed) ===
# Source: the repository's declared Mise toolchain, provisioned by make setup.
# Native Mise interprets the lock and resolves each declared executable. The
# owned acquisition binary, never host PATH, starts that read-only selection;
# activation and its authorization therefore never depend on a host direnv.
override MISE = "$$(set -eu; mise_path="$$("$(BOOTSTRAP_MISE)" -C "$(RUNTIME_ROOT)" which mise)"; if [ "$${OS:-}" = Windows_NT ]; then cygpath --unix "$$mise_path"; else printf '%s\n' "$$mise_path"; fi)"
override UV = "$$(set -eu; uv_mise=$(MISE); uv_path="$$("$$uv_mise" -C "$(RUNTIME_ROOT)" which "uv")"; if [ "$${OS:-}" = Windows_NT ]; then cygpath --unix "$$uv_path"; else printf '%s\n' "$$uv_path"; fi)"
override DIRENV = "$$(set -eu; direnv_mise=$(MISE); direnv_path="$$("$$direnv_mise" -C "$(RUNTIME_ROOT)" which "direnv")"; if [ "$${OS:-}" = Windows_NT ]; then cygpath --unix "$$direnv_path"; else printf '%s\n' "$$direnv_path"; fi)"
CALLER_PATH := $(PATH)
CALLER_VIRTUAL_ENV := $(patsubst %/,%,$(VIRTUAL_ENV))
# End SECTION: project tool owner

# === SECTION: profile routing (managed) ===
# Source: repository topology. workspace has .gitmodules; standalone does not.
# Native UV membership and Git attachment elect the physical runtime.
override RUNTIME_ROOT := $(REPOSITORY_ROOT)
override export GIT_CEILING_DIRECTORIES := $(shell cd "$(GIT_DISCOVERY_ROOT)/.." && pwd -P)
ifneq ($(.SHELLSTATUS),0)
$(error Cannot resolve Git ceiling for $(GIT_DISCOVERY_ROOT))
endif
override export MISE_CEILING_PATHS := $(shell cd "$(RUNTIME_ROOT)/.." && pwd -P)
ifneq ($(.SHELLSTATUS),0)
$(error Cannot resolve Mise ceiling for $(RUNTIME_ROOT))
endif
# The locked install reads only the declared project config: a machine-local
# override (mise.local.toml and siblings) never selects a tool outside the lock.
override export MISE_OVERRIDE_CONFIG_FILENAMES := .mise.toml
override export MISE_CONFIG_DIR := $(RUNTIME_ROOT)/.mise
# The project manifest remains in native project scope. An optional global
# configuration has its own explicit runtime-scoped path; when absent, Mise
# reads no global file. Reusing .mise.toml here classifies every project tool
# as global and leaves the project lock scope empty.
override export MISE_GLOBAL_CONFIG_FILE := $(MISE_CONFIG_DIR)/config.toml
override export MISE_SYSTEM_CONFIG_DIR := $(MISE_CONFIG_DIR)
# A file override wins over the system directory in native Mise discovery.
unexport MISE_SYSTEM_CONFIG_FILE MISE_ENV MISE_ENV_FILE
# No global locked mode: setup installs whatever mise.lock pins and never stops
# on a stale or incomplete lock (operator-ruling-2026-10-09-setup-resilient);
# only the runtime reader's discovery calls run locked, scoped to those calls.
override export MISE_AUTO_INSTALL := 0
# Project provisioning must never install or resolve tools from host config.
# The physical runtime owns both its environment and frozen tool identities.
# Attached members retain their own lock inputs for standalone consumption;
# the pinned mise release is the [tools] entry the committed mise.lock pins.
# End SECTION: profile routing

# D-VENV: the environment is <RUNTIME_ROOT>/.venv only. A member inside a
# workspace uses the workspace environment; a standalone checkout or a linked
# worktree owns its own physical environment inside the checkout. The path is
# law, never configuration, and is never shared with another checkout.
override RUNTIME_VENV := $(RUNTIME_ROOT)/.venv
# The upgrade executor (make.executor_environment) is a separate uv-created
# environment under the runtime root: it runs only `make upg`'s residual
# flext-infra step and never imports the project or the runtime environment.
override FLEXT_INFRA_EXECUTOR_VENV := $(RUNTIME_ROOT)/.flext/executor
ifeq ($(OS),Windows_NT)
override RUNTIME_BIN := $(RUNTIME_VENV)/Scripts
override RUNTIME_PYTHON := $(RUNTIME_BIN)/python.exe
override FLEXT_INFRA_EXECUTOR_PYTHON := $(FLEXT_INFRA_EXECUTOR_VENV)/Scripts/python.exe
NORMALIZED_CALLER_PATH := $(shell cygpath --path "$(CALLER_PATH)")
ifneq ($(.SHELLSTATUS),0)
$(error cygpath failed to normalize PATH)
endif
ifneq ($(strip $(CALLER_VIRTUAL_ENV)),)
NORMALIZED_CALLER_VIRTUAL_ENV := $(shell cygpath --unix "$(CALLER_VIRTUAL_ENV)")
ifneq ($(.SHELLSTATUS),0)
$(error cygpath failed to normalize VIRTUAL_ENV)
endif
endif
CALLER_VIRTUAL_ENV_BIN := $(NORMALIZED_CALLER_VIRTUAL_ENV)/Scripts
else
override RUNTIME_BIN := $(RUNTIME_VENV)/bin
override RUNTIME_PYTHON := $(RUNTIME_BIN)/python
override FLEXT_INFRA_EXECUTOR_PYTHON := $(FLEXT_INFRA_EXECUTOR_VENV)/bin/python
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
override FLEXT_INFRA_PYTHON := $(RUNTIME_PYTHON)
override UV_PROJECT_ENVIRONMENT := $(RUNTIME_VENV)
override VIRTUAL_ENV := $(RUNTIME_VENV)
override PATH := $(RUNTIME_BIN):$(SANITIZED_CALLER_PATH)
unexport UV MYPYPATH MISE_PYTHON_UV_VENV_AUTO UV_PROJECT UV_WORKING_DIR UV_CONFIG_FILE UV_PROJECT_ENVIRONMENT UV_PYTHON UV_PYTHON_DOWNLOADS UV_NO_CONFIG UV_NO_SOURCES UV_FROZEN UV_LOCKED VIRTUAL_ENV PYTHONHOME PYTHONPATH
export FLEXT_INFRA_PYTHON UV_PROJECT_ENVIRONMENT VIRTUAL_ENV PATH RUNTIME_ROOT

# Tool provisioning belongs to setup and upgrade, never to generation: upgrade
# always cures (tracker memory operator-ruling-2026-10-09-setup-always-upg-cures),
# so `make upg` acquires Mise, repairs and relocks the Mise locks, and installs
# the locked toolchain (_upg_toolchain) before any stage resolves through it.
TOOL_BOOTSTRAP_LIFECYCLE := _setup_native_prepare
.PHONY: _bootstrap_setup_tools

# The lifecycle invokes recursive make through mise, so preserve jobserver FDs.
_bootstrap_setup_tools:
	+@set -eu; \
	bootstrap_stage=source-identity; bootstrap_completed=false; bootstrap_sha=null; bootstrap_git_status=null; mise_stage=; \
	bootstrap_report="$(PROJECT_ROOT)/.reports/setup-operations/tool-bootstrap.json"; \
	bootstrap_publish_report() { \
		if [ -d "$$(dirname "$${bootstrap_report}")" ]; then :; else mkdir -p "$$(dirname "$${bootstrap_report}")" 2>/dev/null || true; fi; \
		if printf '%s\n' '{"root":".","operation":"tool-bootstrap","scope":["."],"receipts":[{"native_tool_qualifications":[],"runtime_context":null,"root":".","executed":true,"outcome":null,"native_exit_code":@@EXIT@@,"stage":"@@STAGE@@","completed":@@COMPLETE@@,"applicability":"","files":[],"issues":[]}],"candidate_read_exit_code":@@GIT_EXIT@@,"candidate_sha":@@SHA@@}' | sed \
			-e "s/@@SHA@@/$${bootstrap_sha}/" \
			-e "s/@@GIT_EXIT@@/$${bootstrap_git_status}/" \
			-e "s/@@EXIT@@/$${bootstrap_status}/" \
			-e "s/@@STAGE@@/$${bootstrap_stage}/" \
			-e "s/@@COMPLETE@@/$${bootstrap_completed}/" > "$${bootstrap_report}.pending"; then :; else return $$?; fi; \
		mv "$${bootstrap_report}.pending" "$${bootstrap_report}"; \
	}; \
	trap 'bootstrap_status=$$?; trap - EXIT; \
		if [ -n "$${mise_stage}" ]; then \
			if rm -rf "$${mise_stage}"; then :; else \
				mise_cleanup_status=$$?; \
				printf "ERROR: bootstrap staging cleanup failed (exit %s); original exit %s\n" "$${mise_cleanup_status}" "$${bootstrap_status}" >&2; \
				if [ "$${bootstrap_status}" -eq 0 ]; then bootstrap_status=$${mise_cleanup_status}; fi; \
			fi; \
		fi; \
		if bootstrap_publish_report; then :; else \
			bootstrap_report_status=$$?; \
			printf "ERROR: bootstrap evidence publication failed (exit %s); original exit %s\n" "$${bootstrap_report_status}" "$${bootstrap_status}" >&2; \
			if [ "$${bootstrap_status}" -eq 0 ]; then bootstrap_status=$${bootstrap_report_status}; fi; \
		fi; \
		exit "$${bootstrap_status}"' EXIT; \
	if bootstrap_candidate="$$(git -C "$(PROJECT_ROOT)" rev-parse --verify HEAD)"; then \
		bootstrap_git_status=0; \
		case "$${bootstrap_candidate}" in \
			*[!0-9a-f]*) printf 'ERROR: bootstrap candidate is not a Git object ID\n' >&2; exit 2 ;; \
		esac; \
		case "$${#bootstrap_candidate}" in \
			40|64) bootstrap_sha="\"$${bootstrap_candidate}\"" ;; \
			*) printf 'ERROR: bootstrap candidate has an invalid object ID length\n' >&2; exit 2 ;; \
		esac; \
	else bootstrap_git_status=$$?; fi; \
	bootstrap_stage=network-auth; \
	if git config --global --get-regexp '^url\.https?://[^ /]+:[^ /]+@.*\.insteadof$$' >/dev/null; then \
		printf 'ERROR: Global Git URL rewrite configuration embeds credentials and shadows the fleet SSH route; correct that configuration at its owner\n' >&2; \
		exit 2; \
	else \
		rewrite_status=$$?; \
		if [ "$$rewrite_status" -ne 1 ]; then \
			printf 'ERROR: Cannot inspect the global Git URL rewrite configuration\n' >&2; \
			exit "$$rewrite_status"; \
		fi; \
	fi; \
	bootstrap_stage=acquisition; \
	if [ -L "$(PROJECT_ROOT)/.bin" ] && [ ! -e "$(PROJECT_ROOT)/.bin" ]; then rm -f "$(PROJECT_ROOT)/.bin"; fi; mkdir -p "$(PROJECT_ROOT)/.bin"; \
	export MISE_INSTALL_FROM_GITHUB=1 MISE_INSTALL_SKIP_IF_EXISTS=1 MISE_INSTALL_HELP=0; \
	export MISE_VERSION="2026.10.3"; \
	mise_acquired=false; \
	if [ -x "$(BOOTSTRAP_MISE)" ]; then \
		mise_present="$$("$(BOOTSTRAP_MISE)" --version)"; \
		if [ "$${mise_present%% *}" = "$${MISE_VERSION}" ]; then mise_acquired=true; fi; \
	fi; \
	if [ "$${mise_acquired}" = true ]; then \
		printf 'setup: mise %s already acquired at %s\n' "$${MISE_VERSION}" "$(BOOTSTRAP_MISE)"; \
	else \
	case "$${OS:-}" in \
		(Windows_NT) \
			export MISE_INSTALL_PATH="$$(cygpath --windows "$(BOOTSTRAP_MISE)")"; \
			powershell.exe -NoProfile -Command '$$ErrorActionPreference = "Stop"; Invoke-RestMethod https://mise.run/powershell | Invoke-Expression' ;; \
		(*) \
			export MISE_INSTALL_PATH="$(BOOTSTRAP_MISE)"; \
			mise_stage="$$(mktemp -d "$(PROJECT_ROOT)/.bin/.mise-install.XXXXXXXX")"; \
			curl --proto '=https' --tlsv1.2 -fsSL -o "$${mise_stage}/install.sh" https://mise.run; \
			sh "$${mise_stage}/install.sh" ;; \
	esac; \
	fi; \
	export MISE_SHIMS_DIR="$(PROJECT_ROOT)/.bin/shims"; \
	export MISE_LOCKED=0; \
	bootstrap_stage=tool-install; \
	"$(BOOTSTRAP_MISE)" -C "$(PROJECT_ROOT)" install --yes "python" "github:jdx/mise" "uv" "github:direnv/direnv" "taplo" "aqua:ast-grep/ast-grep" "gitleaks" "aqua:boyter/scc" "node" "github:qltysh/qlty" "github:kucherenko/jscpd" "github:microsoft/waza" "aqua:j178/prek"; \
	mise_active="$$("$(BOOTSTRAP_MISE)" -C "$(PROJECT_ROOT)" tool "github:jdx/mise" --active)"; \
	mise_binary="$$("$(BOOTSTRAP_MISE)" -C "$(PROJECT_ROOT)" which mise)"; \
	if [ "$${OS:-}" = Windows_NT ]; then mise_binary="$$(cygpath --unix "$${mise_binary}")"; fi; \
	bootstrap_stage=tool-paths; \
	if [ -n "$${GITHUB_PATH:-}" ]; then "$${mise_binary}" -C "$(PROJECT_ROOT)" bin-paths >> "$${GITHUB_PATH}"; fi; \
	printf 'setup: mise %s provisioning complete\n' "$${mise_active}"; \
	printf 'setup: entering lifecycle (submodules, environment, hooks) make=%s\n' "$(SELF_MAKE_EXECUTABLE)"; \
	bootstrap_stage=lifecycle; \
	MISE_AUTO_INSTALL=0 "$${mise_binary}" -C "$(PROJECT_ROOT)" exec -- env "CI=$(CI)" $(SELF_MAKE) $(TOOL_BOOTSTRAP_LIFECYCLE); \
	bootstrap_stage=complete; bootstrap_completed=true
_bootstrap_setup_tools: _builtin_require_workspace

# CI evaluates only its own repository. Locally, a workspace root's fmt and fix
# process the root and every governed member together in one checker run
# (operator ruling 2026-10-25: never member by member); each multi-root tool
# runs once over their union.
GATE_SCOPE := self
# Local and CI consume the same native project/workspace declarations and lock.
# Setup always bootstraps: `uv sync --frozen` installs exactly the committed
# uv.lock without re-resolving or rewriting it, so manifest drift never blocks
# provisioning. Check reports drift through `uv lock --check`; only `make upg`
# writes the lock.
RESOLVE_DESIRED_PYTHON = desired_python="$$($(MISE) -C "$(RUNTIME_ROOT)" which python)"; \
	if [ "$${OS:-}" = Windows_NT ]; then desired_python="$$(cygpath --unix "$$desired_python")"; fi; \
	if [ ! -x "$$desired_python" ]; then \
		printf 'ERROR[setup] Mise-resolved Python is not executable: %s\n' "$$desired_python" >&2; \
		exit 2; \
	fi
# Every installation is prepared under the existing ignored runtime state.
# Its frozen packages, editable receipt and real consumers precede one atomic
# publication; the displaced physical environment remains recoverable.
TARGET_ENVIRONMENT = install_target="$(1)"; \
	install_state="$$(dirname "$(FLEXT_INFRA_EXECUTOR_VENV)")"; \
	if [ -L "$$install_target" ] || [ -L "$$install_state" ]; then \
		printf 'ERROR[setup] environment and its state directory must be physical paths\n' >&2; exit 2; \
	fi; \
	mkdir -p "$$install_state"
PREPARE_ENVIRONMENT = install_stage=; install_candidate_origin=; \
	mv --exchange --no-copy --help >/dev/null; \
	install_origin=; \
	if [ -e "$$install_target" ]; then install_origin="$$(stat -c '%d:%i' "$$install_target")"; fi; \
	install_stage="$$(mktemp -d "$$install_state/setup.XXXXXXXX")"; \
	install_environment="$$install_stage/.venv"; \
	trap 'install_status=$$?; trap - EXIT; if [ -n "$$install_stage" ] && { [ -z "$$install_candidate_origin" ] || [ ! -e "$$install_environment" ] || [ "$$(stat -c "%d:%i" "$$install_environment")" = "$$install_candidate_origin" ]; }; then rm -rf -- "$$install_stage"; fi; exit "$$install_status"' EXIT; \
	trap 'exit 129' HUP; trap 'exit 130' INT; trap 'exit 143' TERM; \
	printf 'setup: preparing %s for %s\n' "$$install_environment" "$$install_target"; \
	$(UV) venv --relocatable --project "$(RUNTIME_ROOT)" --python "$$desired_python" --no-python-downloads "$$install_environment"; \
	install_candidate_origin="$$(stat -c '%d:%i' "$$install_environment")"; \
	install_python="$$install_environment/$(patsubst $(RUNTIME_VENV)/%,%,$(RUNTIME_PYTHON))"
# The candidate's own flext-infra console script. Setup runs its workspace step
# through it, so one process does the work and proves the entry point.
INSTALL_FLEXT_INFRA = "$$install_environment/$(patsubst $(RUNTIME_VENV)/%,%,$(RUNTIME_BIN))/flext-infra$(if $(filter Windows_NT,$(OS)),.exe,)"
VERIFY_ENVIRONMENT = test -f "$$install_environment/pyvenv.cfg"; \
	"$$install_python" -I -c 'import pathlib, sys; sys.exit(0 if sys.prefix != sys.base_prefix and pathlib.Path(sys.prefix).resolve() == pathlib.Path(sys.argv[1]).resolve() else "candidate interpreter leaves its environment")' "$$install_environment"
PUBLISH_ENVIRONMENT = $(VERIFY_ENVIRONMENT); \
	if [ -n "$$install_origin" ]; then \
		if [ -L "$$install_target" ] || [ "$$(stat -c '%d:%i' "$$install_target")" != "$$install_origin" ]; then \
			printf 'ERROR[setup] environment changed during preparation: %s\n' "$$install_target" >&2; exit 2; \
		fi; \
		mv --exchange --no-copy -T "$$install_environment" "$$install_target"; \
		rm -rf -- "$$install_stage"; \
		printf 'setup: published %s; previous environment removed\n' "$$install_target"; \
	else \
		mv --no-copy --no-clobber -T "$$install_environment" "$$install_target"; \
		if [ -e "$$install_environment" ]; then printf 'ERROR[setup] environment appeared before publication\n' >&2; exit 2; fi; \
		rmdir "$$install_stage"; \
	fi; \
	install_stage=; trap - EXIT HUP INT TERM
# An environment that already satisfies the frozen lock is kept: uv's own
# --check is the proof, and the same receipt and interpreter verification as a
# publication applies. Only a differing environment (or a local FLEXT editable
# selection) is rebuilt and atomically exchanged, so a repeated setup costs a
# check, not a full reinstall and deletion.
# The check's exit status is its whole result: a non-zero status only selects
# the rebuild, so uv's native "environment is outdated" report is held in
# probe_report instead of surfacing as an error on a successful setup. A real
# failure recurs in the rebuild's own `uv sync --frozen`, which runs unfiltered
# and fails loud with uv's native output.
INSTALL_RUNTIME_RECIPE = set -eu; \
	$(REQUIRE_WORKSPACE_ENVIRONMENT); \
	$(RESOLVE_DESIRED_PYTHON); \
	$(call TARGET_ENVIRONMENT,$(RUNTIME_VENV)); \
	sync_args="$(SETUP_SYNC_ARGS)"; \
	if [ -z "$(strip $(FLEXT))" ] && [ -d "$$install_target" ] && probe_report="$$(env -u MISE_PYTHON_UV_VENV_AUTO -u UV_PROJECT -u UV_WORKING_DIR -u UV_CONFIG_FILE -u UV_PROJECT_ENVIRONMENT -u UV_PYTHON -u UV_PYTHON_DOWNLOADS -u UV_NO_CONFIG -u UV_NO_SOURCES -u UV_FROZEN -u UV_LOCKED -u VIRTUAL_ENV -u PYTHONHOME -u PYTHONPATH UV_PROJECT_ENVIRONMENT="$$install_target" timeout --signal=TERM --kill-after=5s 118s $(UV) sync --frozen --check --project "$(RUNTIME_ROOT)" --python "$$desired_python" --no-python-downloads $$sync_args 2>&1)"; then \
		install_environment="$$install_target"; \
		install_python="$$install_environment/$(patsubst $(RUNTIME_VENV)/%,%,$(RUNTIME_PYTHON))"; \
		env -u MYPYPATH -u MISE_PYTHON_UV_VENV_AUTO -u UV_PROJECT -u UV_WORKING_DIR -u UV_CONFIG_FILE -u UV_PROJECT_ENVIRONMENT -u UV_PYTHON -u UV_PYTHON_DOWNLOADS -u UV_NO_CONFIG -u UV_NO_SOURCES -u UV_FROZEN -u UV_LOCKED -u VIRTUAL_ENV -u PYTHONHOME -u PYTHONPATH $(INSTALL_FLEXT_INFRA) workspace flext-editable --repository-root "$(RUNTIME_ROOT)" --candidate-environment "$$install_environment"; \
		$(VERIFY_ENVIRONMENT); \
		printf 'setup: %s already satisfies uv.lock; kept\n' "$$install_target"; \
	else \
		$(PREPARE_ENVIRONMENT); \
		env -u MISE_PYTHON_UV_VENV_AUTO -u UV_PROJECT -u UV_WORKING_DIR -u UV_CONFIG_FILE -u UV_PROJECT_ENVIRONMENT -u UV_PYTHON -u UV_PYTHON_DOWNLOADS -u UV_NO_CONFIG -u UV_NO_SOURCES -u UV_FROZEN -u UV_LOCKED -u VIRTUAL_ENV -u PYTHONHOME -u PYTHONPATH UV_PROJECT_ENVIRONMENT="$$install_environment" timeout --signal=TERM --kill-after=5s 118s $(UV) sync --frozen --project "$(RUNTIME_ROOT)" --python "$$desired_python" --no-python-downloads $$sync_args; \
		if [ -n "$(strip $(FLEXT))" ]; then \
			if [ "$(CI)" = "Y" ]; then printf 'ERROR[setup] local editable sources are forbidden in CI\n' >&2; exit 2; fi; \
			(cd "$(FLEXT)"; $(UV) pip install --no-config --python "$$install_python" --no-deps --requirement "requirements-infra.txt"); \
			$(UV) pip install --python "$$install_python" --no-deps --editable "$(RUNTIME_ROOT)"; \
		fi; \
		env -u MYPYPATH -u MISE_PYTHON_UV_VENV_AUTO -u UV_PROJECT -u UV_WORKING_DIR -u UV_CONFIG_FILE -u UV_PROJECT_ENVIRONMENT -u UV_PYTHON -u UV_PYTHON_DOWNLOADS -u UV_NO_CONFIG -u UV_NO_SOURCES -u UV_FROZEN -u UV_LOCKED -u VIRTUAL_ENV -u PYTHONHOME -u PYTHONPATH $(INSTALL_FLEXT_INFRA) workspace flext-editable --repository-root "$(RUNTIME_ROOT)" --candidate-environment "$$install_environment"$(if $(strip $(FLEXT)), --flext-root "$(FLEXT)"); \
		$(PUBLISH_ENVIRONMENT); \
	fi

# The governed checkouts of this repository: itself (`.`) and every
# initialized gitlink its .gitmodules declares without flext-managed = false.
# Sets $$checkouts (space-separated, relative to $$root); callers run it
# under `set -f`.
GOVERNED_CHECKOUTS = root="$(PROJECT_ROOT)"; \
	checkouts="."; \
	if [ -f "$$root/.gitmodules" ]; then \
		if paths=$$(git -C "$$root" config -f .gitmodules --get-regexp '^submodule\..*\.path$$'); then :; else \
			status=$$?; [ "$$status" -eq 1 ] || exit "$$status"; paths=""; \
		fi; \
		key=""; \
		for word in $$paths; do \
			if [ -z "$$key" ]; then key="$$word"; continue; fi; \
			path="$$word"; \
			if managed=$$(git -C "$$root" config -f .gitmodules --type=bool --get "$${key%.path}.flext-managed"); then :; else \
				status=$$?; [ "$$status" -eq 1 ] || exit "$$status"; managed=true; \
			fi; \
			key=""; \
			[ "$$managed" = true ] || continue; \
			if [ ! -e "$$root/$$path/.git" ]; then \
				printf 'ERROR[setup] governed gitlink is not initialized: %s\n' "$$path" >&2; exit 2; \
			fi; \
			checkouts="$$checkouts $$path"; \
		done; \
		if [ -n "$$key" ]; then \
			printf 'ERROR[setup] gitlink declaration without a path: %s\n' "$$key" >&2; exit 2; \
		fi; \
	fi


# Local setup installs every declared Git hook stage into the effective hooks
# directory (core.hooksPath included, created when absent) of this repository
# and, in the same pass, of every governed gitlink declared in its
# .gitmodules (operator ruling 2026-10-25: workspace verbs process all members
# together, never member by member), so no configured hook path is ever
# silently skipped. Each hook runs the projected validator of the checkout
# that invokes it, resolved at hook runtime: linked worktrees share one hooks
# directory, so an absolute worktree path would bind every sibling to whichever
# worktree ran setup last and break them all once it is retired. Hooks use the
# provisioned prek runner; CI installs none. Setup owns only its marked
# section, placed first after the interpreter line: every other owner's
# content (the Beads integration section among them) is preserved and still
# runs after it. The pre-push section buffers Git's ref list once and hands
# the same stdin to the sections that follow. A managed stage that is no
# longer declared loses only setup's marked section; the hook file is
# removed when no other owner's content remains in it.
INSTALL_GIT_HOOKS_RECIPE = set -eu; \
	prek_binary="$$($(MISE) -C "$(RUNTIME_ROOT)" which prek)"; \
	if [ "$${OS:-}" = Windows_NT ]; then prek_binary="$$(cygpath --unix "$$prek_binary")"; fi; \
	if [ ! -x "$$prek_binary" ]; then \
		printf 'ERROR[setup] Mise-resolved prek is not executable: %s\n' "$$prek_binary" >&2; \
		exit 2; \
	fi; \
	set -f; \
	$(GOVERNED_CHECKOUTS); \
	for checkout in $$checkouts; do \
		hooks_dir="$$(git -C "$$root/$$checkout" rev-parse --path-format=absolute --git-path hooks)"; \
		mkdir -p "$$hooks_dir"; \
		for stage in pre-push; do \
			hook="$$hooks_dir/$$stage"; \
			if [ -e "$$hook" ] && grep -q 'Generated by make setup (flext-infra)' "$$hook"; then \
				others="$$(sed -e '/^\# BEGIN FLEXT-INFRA SETUP/,/^\# END FLEXT-INFRA SETUP/d' -e '1{/^\#!/d;}' "$$hook")"; \
				if printf '%s\n' "$$others" | grep -q 'Generated by make setup (flext-infra)'; then others=""; fi; \
				if [ -n "$$(printf '%s' "$$others" | tr -d '[:space:]')" ]; then \
					printf '%s\n' '\#!/bin/sh' "$$others" > "$$hook.pending"; \
					chmod 0755 "$$hook.pending"; \
					mv "$$hook.pending" "$$hook"; \
				else \
					rm -f "$$hook"; \
				fi; \
				printf 'setup: retired undeclared %s hook section at %s\n' "$$stage" "$$hook"; \
			fi; \
			if [ -e "$$hook.backup" ] && grep -q 'Generated by make setup (flext-infra)' "$$hook.backup"; then rm -f "$$hook.backup"; fi; \
		done; \
		validator_path="$$(git -C "$$root/$$checkout" rev-parse --show-toplevel)/.github/scripts/validate-git-hook.sh"; \
		if [ ! -x "$$validator_path" ]; then \
			printf 'setup: hooks skipped (no validator) checkout=%s\n' "$$checkout"; \
			continue; \
		fi; \
		for stage in pre-commit; do \
			hook="$$hooks_dir/$$stage"; \
			others=""; \
			if [ -e "$$hook" ]; then \
				others="$$(sed -e '/^\# BEGIN FLEXT-INFRA SETUP/,/^\# END FLEXT-INFRA SETUP/d' -e '1{/^\#!/d;}' "$$hook")"; \
				if printf '%s\n' "$$others" | grep -q 'Generated by make setup (flext-infra)'; then others=""; fi; \
			fi; \
			validator='"$$(git rev-parse --show-toplevel)/.github/scripts/validate-git-hook.sh"'; \
			{ \
				printf '%s\n' '\#!/bin/sh' "\# BEGIN FLEXT-INFRA SETUP: Generated by make setup (flext-infra); rerun make setup to refresh."; \
				if [ "$$stage" = pre-push ]; then \
					printf '%s\n' \
						'flext_hook_stdin="$$(mktemp "$$(git rev-parse --absolute-git-dir)/flext-pre-push.XXXXXX")"' \
						'cat > "$$flext_hook_stdin"' \
						"PREK=$$prek_binary $$validator $$stage \"\$$@\" < \"\$$flext_hook_stdin\" || { flext_hook_status=\$$?; rm -f \"\$$flext_hook_stdin\"; exit \"\$$flext_hook_status\"; }" \
						'exec < "$$flext_hook_stdin"' \
						'rm -f "$$flext_hook_stdin"'; \
				else \
					printf '%s\n' "PREK=$$prek_binary $$validator $$stage \"\$$@\" || exit \$$?"; \
				fi; \
				printf '%s\n' '\# END FLEXT-INFRA SETUP'; \
				if [ -n "$$others" ]; then printf '%s\n' "$$others"; fi; \
			} > "$$hook.pending"; \
			chmod 0755 "$$hook.pending"; \
			mv "$$hook.pending" "$$hook"; \
			if [ -e "$$hook.backup" ] && grep -q 'Generated by make setup (flext-infra)' "$$hook.backup"; then rm -f "$$hook.backup"; fi; \
			printf 'setup: installed %s hook section at %s\n' "$$stage" "$$hook"; \
		done; \
	done

# Reject borrowed environments before bootstrap, activation or custom hooks.
# Members may use their containing workspace, never an unrelated checkout.
REQUIRE_WORKSPACE_ENVIRONMENT = case "$(PROJECT_ROOT)/" in \
	"$(RUNTIME_ROOT)/"*) ;; \
	*) printf 'ERROR: runtime workspace does not contain this project: %s\n' "$(RUNTIME_ROOT)" >&2; exit 2 ;; \
	esac; \
	for environment_path in "$$(dirname "$(RUNTIME_VENV)")" "$(RUNTIME_VENV)" "$(RUNTIME_BIN)" "$(PROJECT_ROOT)/.venv" "$(PROJECT_ROOT)/.venv/bin" "$(PROJECT_ROOT)/.venv/Scripts" "$$(dirname "$(FLEXT_INFRA_EXECUTOR_VENV)")" "$(FLEXT_INFRA_EXECUTOR_VENV)"; do \
		if [ -L "$$environment_path" ]; then \
			printf 'ERROR: workspace environment must be physical, not a symlink: %s\n' "$$environment_path" >&2; exit 2; \
		fi; \
	done

.PHONY: _builtin_require_workspace
_builtin_require_workspace:
	@$(REQUIRE_WORKSPACE_ENVIRONMENT)

# Execute the interpreter provisioned by setup without discovering a project
# workspace or creating a dependency-resolution file during a runtime command.
override UV_RUN = env -u MYPYPATH -u VIRTUAL_ENV -u UV_PROJECT -u PROJECT_ROOT PATH="$(RUNTIME_BIN):$$PATH" PYTHONPATH="$(PROJECT_ROOT)/src" $(UV) run --directory "$(PROJECT_ROOT)" --no-project --python "$(RUNTIME_PYTHON)"
# Native locked workspace membership owns control imports, never caller overlays.
override PROJECT_INFRA_RUN = env -u MYPYPATH -u MISE_PYTHON_UV_VENV_AUTO -u UV_PROJECT -u UV_WORKING_DIR -u UV_CONFIG_FILE -u UV_PROJECT_ENVIRONMENT -u UV_PYTHON -u UV_PYTHON_DOWNLOADS -u UV_NO_CONFIG -u UV_NO_SOURCES -u UV_FROZEN -u UV_LOCKED -u VIRTUAL_ENV -u PYTHONHOME -u PYTHONPATH "$(FLEXT_INFRA_PYTHON)"
override PROJECT_FLEXT_INFRA := $(PROJECT_INFRA_RUN) -m flext_infra
# The upgrade executor (make.executor_environment) runs isolated (-I: no working
# directory, PYTHONPATH or user site on sys.path) from its own environment, so
# `make upg`'s residual flext-infra step imports neither the project's source nor
# the runtime environment.
override EXECUTOR_FLEXT_INFRA = env -u MYPYPATH -u MISE_PYTHON_UV_VENV_AUTO -u UV_PROJECT -u UV_WORKING_DIR -u UV_CONFIG_FILE -u UV_PROJECT_ENVIRONMENT -u UV_PYTHON -u UV_PYTHON_DOWNLOADS -u UV_NO_CONFIG -u UV_NO_SOURCES -u UV_FROZEN -u UV_LOCKED -u VIRTUAL_ENV -u PYTHONHOME -u PYTHONPATH "$(FLEXT_INFRA_EXECUTOR_PYTHON)" -I -m flext_infra
# Native uv syncs the executor from the project's frozen lock: the dependency
# group that pins flext-infra, without editable installs, into its own
# environment (recreated first when it is not a uv environment, and refreshed,
# so a sync never targets a Mise base interpreter). A consumer excludes the
# project, so the executor never imports its source. A project in flext-infra's
# own runtime closure (derived from the installed metadata) supplies part of the
# executor, so its locked non-editable build is installed, never the editable
# source; every lock is already written before it runs.
# Native `--compile-bytecode` writes the .pyc at install time, so the first import
# never compiles the stack (the host may export PYTHONDONTWRITEBYTECODE).
SYNC_UPGRADE_EXECUTOR_RECIPE = $(call TARGET_ENVIRONMENT,$(FLEXT_INFRA_EXECUTOR_VENV)); $(PREPARE_ENVIRONMENT); \
	env -u MISE_PYTHON_UV_VENV_AUTO -u UV_PROJECT -u UV_WORKING_DIR -u UV_CONFIG_FILE -u UV_PROJECT_ENVIRONMENT -u UV_PYTHON -u UV_PYTHON_DOWNLOADS -u UV_NO_CONFIG -u UV_NO_SOURCES -u UV_FROZEN -u UV_LOCKED -u VIRTUAL_ENV -u PYTHONHOME -u PYTHONPATH UV_PROJECT_ENVIRONMENT="$$install_environment" timeout --signal=TERM --kill-after=5s 118s $(UV) sync --frozen --project "$(RUNTIME_ROOT)" --python "$$desired_python" --no-python-downloads --no-default-groups --group "dev" --no-editable --compile-bytecode --link-mode clone; \
	$(INSTALL_FLEXT_INFRA) --help >/dev/null; \
	$(PUBLISH_ENVIRONMENT)

# Setup checks native lock freshness without writing the lock. Native uv selects
# every member's declared extras and groups, without copying dependency graphs.


override SETUP_SYNC_ARGS := --all-extras --all-groups --compile-bytecode --link-mode clone

ifeq ($(CI),Y)
override SETUP_SYNC_ARGS := --all-extras --all-groups --compile-bytecode --link-mode clone --no-editable
endif

# Custom hooks are declared in CI too; load their selected physical Makefile.
-include $(CUSTOM_MAKEFILE_ESCAPED)
ifeq ($(APPROVAL_CONTEXT),Y)
override CI := Y
export CI
override TOOL_BOOTSTRAP_LIFECYCLE := _setup_native_prepare
endif
override SELF_MAKE := "$(SELF_MAKE_EXECUTABLE)" --no-print-directory -f "$(SELF_MAKEFILE)"

define RUN_PUBLIC_POST
	$(if $(filter post-$(1),$(CUSTOM_DECLARED_TARGETS)),+@$(SELF_MAKE) post-$(1))
endef

# Every public operation invokes its declared hooks and handler once.
# Environment activation is an input prerequisite, never a second operation
# after a generator has published a new .envrc.
define RUN_PUBLIC_PRODUCE
	$(if $(filter pre-$(1),$(CUSTOM_DECLARED_TARGETS)),+@$(SELF_MAKE) pre-$(1))
	$(if $(filter _custom-$(1),$(CUSTOM_DECLARED_TARGETS)),+@$(SELF_MAKE) _custom-$(1),+@$(SELF_MAKE) _builtin-$(1))
endef

define RUN_PUBLIC
$(call RUN_PUBLIC_PRODUCE,$(1))
$(call RUN_PUBLIC_POST,$(1))
endef



.PHONY: $(PUBLIC_VERBS) $(addprefix _builtin-,$(PUBLIC_VERBS))
.PHONY: _builtin_gen_all


VERB_DESCRIPTION_setup := Install the declared frozen locks, synchronize the physical environment and configure local integrations; FLEXT=<path> reinstalls the FLEXT distributions that workspace provides as editable (local only).
VERB_ARGUMENTS_setup := FLEXT RECREATE

VERB_DESCRIPTION_upg := Resolve declared dependency and tool updates and publish their resolution metadata and locks.
VERB_ARGUMENTS_upg :=

VERB_DESCRIPTION_lock := Resolve newly declared dependencies into the locks and keep every existing pin; only upg moves pins.
VERB_ARGUMENTS_lock :=

VERB_DESCRIPTION_gen := Materialize explicitly declared new projects and publish every applicable generated projection.
VERB_ARGUMENTS_gen := ROOT

VERB_DESCRIPTION_gen-scope := Run the focal generation surface (WHAT, MODULE for facades, SCOPE) through the same engine, lock and budget as gen.
VERB_ARGUMENTS_gen-scope := ROOT WHAT MODULE SCOPE

VERB_DESCRIPTION_mod := Refactor declared source structure through the shared pipeline and publish the complete coherent change; LAWS (ENFORCE ids) or RULES (rule ids) diagnose or repair one law (ADR-028 Decision 18 item 2).
VERB_ARGUMENTS_mod := LAWS RULES

VERB_DESCRIPTION_fix := Apply declared repairs to authored inputs through their canonical owners.
VERB_ARGUMENTS_fix :=

VERB_DESCRIPTION_fmt := Format authored content through the configured formatters.
VERB_ARGUMENTS_fmt :=

VERB_DESCRIPTION_check := Validate declarations, projections, environment, documentation and configured gates without repairs or publication.
VERB_ARGUMENTS_check := CHECK_PHASE

VERB_DESCRIPTION_tests := Run native incremental tests through the persistent Testmon selection.
VERB_ARGUMENTS_tests :=

VERB_DESCRIPTION_tests-full := Run the complete selected inventory once without Testmon.
VERB_ARGUMENTS_tests-full :=

VERB_DESCRIPTION_help := Discover the declared public interface without effects.
VERB_ARGUMENTS_help :=

VERB_DESCRIPTION_build := Build local distribution and documentation artifacts without publishing them.
VERB_ARGUMENTS_build :=

VERB_DESCRIPTION_clean := Remove only declared disposable resources owned by the selected target.
VERB_ARGUMENTS_clean :=

VERB_DESCRIPTION_release-plan := Execute the declared release plan phase.
VERB_ARGUMENTS_release-plan := PR_TITLE

VERB_DESCRIPTION_release-version := Execute the declared release version phase.
VERB_ARGUMENTS_release-version :=

VERB_DESCRIPTION_release-tag := Execute the declared release tag phase.
VERB_ARGUMENTS_release-tag :=

VERB_DESCRIPTION_release-build := Execute the declared release build phase.
VERB_ARGUMENTS_release-build :=

VERB_DESCRIPTION_release-publish := Execute the declared release publish phase.
VERB_ARGUMENTS_release-publish := INDEX

VERB_HELP_LINE = printf '  %-16s %s\n' '$(1)' '$(VERB_DESCRIPTION_$(1))'$(if $(strip $(VERB_ARGUMENTS_$(1))),; printf '    Inputs: %s\n' '$(VERB_ARGUMENTS_$(1))')
# OPTIONS=Y (or HELP=Y) displays every public verb's catalog contract without
# running its prerequisites, hooks, handler or product dispatcher.
ifeq ($(VERB_CONTRACT),)
PROFILE_VERBS_PLAIN := gen gen-scope
PROFILE_VERBS_DISPATCH := help clean
PROFILE_VERBS_ACTIVATED := mod fix fmt check tests tests-full build release-plan release-version release-tag release-build release-publish
PROFILE_VERBS_INTERPRETER :=
PROFILE_VERBS_HOOKED := $(foreach v,$(filter-out $(PROFILE_VERBS_INTERPRETER),$(PROFILE_VERBS_ACTIVATED)),$(if $(filter pre-$(v) _custom-$(v) post-$(v),$(CUSTOM_DECLARED_TARGETS)),$(v)))
PROFILE_VERBS_DIRECT := $(filter-out $(PROFILE_VERBS_HOOKED) $(PROFILE_VERBS_INTERPRETER),$(PROFILE_VERBS_ACTIVATED))
PROFILE_VERBS_ROUTED := $(PROFILE_VERBS_HOOKED) $(PROFILE_VERBS_INTERPRETER)

$(PROFILE_VERBS_PLAIN): %: _builtin_require_workspace _builtin_require_interpreter
	@printf '==> %s | project=%s | profile=%s\n' '$@' "$(PROJECT_ROOT)" 'standalone'
	$(call RUN_PUBLIC,$@)


$(PROFILE_VERBS_DISPATCH):
	$(call RUN_PUBLIC,$@)


$(PROFILE_VERBS_ACTIVATED): %: _builtin_require_workspace
	@printf '==> %s | project=%s | profile=%s\n' '$@' "$(PROJECT_ROOT)" 'standalone'
	+@$(if $(filter Y,$(CI)),$(MISE) -C "$(RUNTIME_ROOT)" exec -- $(SELF_MAKE) _activated-$@,$(DIRENV) exec "$(PROJECT_ROOT)" $(SELF_MAKE) _activated-$@)
.PHONY: $(addprefix _activated-,$(PROFILE_VERBS_ACTIVATED))
ifneq ($(strip $(PROFILE_VERBS_DIRECT)),)
$(addprefix _activated-,$(PROFILE_VERBS_DIRECT)): _activated-%: _builtin-%
$(addprefix _builtin-,$(PROFILE_VERBS_DIRECT)): _builtin_require_environment
endif
ifneq ($(strip $(PROFILE_VERBS_HOOKED)),)
$(addprefix _activated-,$(PROFILE_VERBS_HOOKED)): _builtin_require_environment
endif
ifneq ($(strip $(PROFILE_VERBS_INTERPRETER)),)
$(addprefix _activated-,$(PROFILE_VERBS_INTERPRETER)): _builtin_require_interpreter
endif
ifneq ($(strip $(PROFILE_VERBS_ROUTED)),)
$(addprefix _activated-,$(PROFILE_VERBS_ROUTED)): _activated-%:
	$(call RUN_PUBLIC,$*)
endif


# `setup` keeps its own recipe (it must not require the environment it is about
# to build), but it still runs the pre-/post-setup lifecycle hooks so a project
# declaring them in the custom handler surface is actually honoured.
ifeq ($(PROJECT_ROOT),$(RUNTIME_ROOT))
setup: _builtin_require_github_credential _bootstrap_setup_tools
else
setup: _builtin_require_workspace
	+@"$(SELF_MAKE_EXECUTABLE)" --no-print-directory -C "$(RUNTIME_ROOT)" -f "$(RUNTIME_ROOT)/Makefile" setup
	@$(if $(filter Y,$(CI)),:,$(DIRENV) allow "$(PROJECT_ROOT)")
endif

# Upgrade relocks Mise and provisions the locked toolchain, then resolves
# dependencies and writes the UV lock; projections belong to gen. Its one Python step runs last, from
# the isolated executor (_upg_lifecycle). Attached members delegate the same
# operation to their physical workspace owner before resolving anything.
ifeq ($(PROJECT_ROOT),$(RUNTIME_ROOT))
upg: _builtin_require_workspace _builtin_require_github_credential _upg_toolchain
	+@$(SELF_MAKE) _upg_lifecycle
else
upg: _builtin_require_workspace
	+@"$(SELF_MAKE_EXECUTABLE)" --no-print-directory -C "$(RUNTIME_ROOT)" -f "$(RUNTIME_ROOT)/Makefile" upg
endif

# Attached members delegate the same operation to their physical workspace owner
# before resolving anything.
ifeq ($(PROJECT_ROOT),$(RUNTIME_ROOT))
lock: _builtin_require_workspace _builtin_require_github_credential
	+@$(SELF_MAKE) _lock_lifecycle
else
lock: _builtin_require_workspace
	+@"$(SELF_MAKE_EXECUTABLE)" --no-print-directory -C "$(RUNTIME_ROOT)" -f "$(RUNTIME_ROOT)/Makefile" lock
endif
SCRIPT_VERBS_PLAIN :=
SCRIPT_VERBS_DISPATCH :=
SCRIPT_VERBS_ACTIVATED :=
SCRIPT_VERBS_INTERPRETER :=
SCRIPT_VERBS_HOOKED := $(foreach v,$(filter-out $(SCRIPT_VERBS_INTERPRETER),$(SCRIPT_VERBS_ACTIVATED)),$(if $(filter pre-$(v) _custom-$(v) post-$(v),$(CUSTOM_DECLARED_TARGETS)),$(v)))
SCRIPT_VERBS_DIRECT := $(filter-out $(SCRIPT_VERBS_HOOKED) $(SCRIPT_VERBS_INTERPRETER),$(SCRIPT_VERBS_ACTIVATED))
SCRIPT_VERBS_ROUTED := $(SCRIPT_VERBS_HOOKED) $(SCRIPT_VERBS_INTERPRETER)



else
$(PUBLIC_VERBS):
	@$(call VERB_HELP_LINE,$@)
	@printf '%s\n' 'OPTIONS=Y displays this contract without effects; run make $@ to execute it.'
endif

.PHONY: _setup_native_prepare _setup_lifecycle
# Native workspace discovery requires the shipped member manifests. Prepare
# absent gitlinks with the existing WIP-preserving owner before recursive parsing.
_setup_native_prepare: _builtin_setup_submodules
	+@$(SELF_MAKE) _setup_lifecycle

_setup_lifecycle: _builtin_require_workspace
	+@$(INSTALL_RUNTIME_RECIPE); \
	case "$(strip $(CI)): $(CUSTOM_DECLARED_TARGETS) " in \
		Y:*) ;; \
		*" pre-setup "*) $(SELF_MAKE) pre-setup ;; \
	esac; \
	$(if $(filter Y,$(CI)),,$(DIRENV) allow "$(PROJECT_ROOT)";) \
	$(if $(filter Y,$(CI)),$(SELF_MAKE) _setup_activated,$(DIRENV) exec "$(PROJECT_ROOT)" $(SELF_MAKE) _setup_activated)

# Setup installs UV's frozen resolution; Mise completes missing native tool
# pins during provisioning. Required input and installation failures terminate
# immediately. Generation and upgrade own their respective repairs. Quality gates
# remain separate operations and no failure selects another producer or source.
.PHONY: _setup_activated
# Check owns quality verdicts after provisioning.
_setup_activated:
	+@set -eu; \
	case " $(CUSTOM_DECLARED_TARGETS) " in \
		*" post-setup "*) $(SELF_MAKE) post-setup ;; \
	esac
ifneq ($(CI),Y)
	@$(INSTALL_GIT_HOOKS_RECIPE)
endif

_builtin-help:
	@printf '%s\n' 'flext-core [standalone]' ''
	@$(foreach verb,$(PUBLIC_VERBS),$(call VERB_HELP_LINE,$(verb));)

# A project owns the sources declared by its manifest. Generated setup
# initializes only uninitialized governed gitlinks and never touches an
# initialized checkout.
.PHONY: _builtin_setup_submodules

# === SECTION: submodule setup (managed) ===
# Source: template (submodule_setup_recipe.j2)
# Computed: every checkout discovers governed paths from its own .gitmodules.
#           An absent management flag participates; an explicit false does not.
# Rule: setup INITIALIZES only governed gitlinks that `git submodule status`
#       reports as uninitialized (leading `-`). An initialized checkout is never
#       touched: its branch, HEAD, index, and local work stay as they are.
#       No fetch, no recursion; nested gitlinks belong to their own setup.
# Free: no
# End SECTION: submodule setup
# Why: a no-op setup costs a constant number of Git processes: one read of
# the declared paths, one read of every management flag, and one native
# `git submodule status`, whatever the number of submodules.
# Runners expose umask 002 and `submodule update --init` materializes tracked
# files as 0664; canonical Mise artifact gates demand exact modes, so
# provisioning normalizes the umask before checkout.
# An uninitialized gitlink is cloned at depth 1, the same flag private
# submodule init uses: setup's contract is the recorded commit, and a full
# history cannot finish inside submodule_timeout_seconds when the object
# database is large. Gitlinks clone submodule_jobs at a time; the deadline is
# one submodule_timeout_seconds per wave of concurrent clones, so a fresh
# checkout of the whole fleet gets the same per-clone budget as one member.
_builtin_setup_submodules:
	@set -euf; \
	umask 022; \
	root="$(PROJECT_ROOT)"; \
	if [ ! -f "$${root}/.gitmodules" ]; then exit 0; fi; \
	nl=$$(printf '\n_'); nl=$${nl%_}; \
	if paths=$$(git -C "$${root}" config -f .gitmodules --get-regexp '^submodule\..*\.path$$'); then :; else \
		status=$$?; [ "$${status}" -eq 1 ] || exit "$${status}"; paths=""; \
	fi; \
	if flags=$$(git -C "$${root}" config -f .gitmodules --type=bool --get-regexp '^submodule\..*\.flext-managed$$'); then :; else \
		status=$$?; [ "$${status}" -eq 1 ] || exit "$${status}"; flags=""; \
	fi; \
	unmanaged=""; \
	IFS="$${nl}"; \
	for line in $${flags}; do \
		case "$${line}" in \
			*" true") ;; \
			*" false") unmanaged="$${unmanaged} $${line% false}" ;; \
			*) printf 'ERROR: Git returned an invalid management boolean: %s\n' "$${line}" >&2; exit 2 ;; \
		esac; \
	done; \
	managed=""; \
	for line in $${paths}; do \
		key=$${line%% *}; \
		path=""; \
		[ "$${key}" = "$${line}" ] || path=$${line#* }; \
		case " $${unmanaged} " in *" $${key%.path}.flext-managed "*) continue ;; esac; \
		case "$${path}" in \
			""|/*|..|../*|*/../*|*/..|*[[:space:]]*) \
				printf 'ERROR: governed gitlink path is not a safe relative Make path: %s\n' "$${path}" >&2; \
				exit 2 ;; \
		esac; \
		case " $${managed} " in \
			*" $${path} "*) printf 'ERROR: governed gitlink path is duplicated: %s\n' "$${path}" >&2; exit 2 ;; \
		esac; \
		managed="$${managed} $${path}"; \
	done; \
	IFS=' '; \
	if [ -z "$${managed}" ]; then exit 0; fi; \
	if [ "$${GIT_INDEX_FILE+x}" = x ]; then \
		printf 'ERROR: submodule setup cannot authenticate a relocated Git index\n' >&2; \
		exit 2; \
	fi; \
	state=$$(git -C "$${root}" submodule status -- $${managed}); \
	absent=""; \
	IFS="$${nl}"; \
	for line in $${state}; do \
		case "$${line}" in -*) absent="$${absent} $${line#* }" ;; esac; \
	done; \
	IFS=' '; \
	if [ -z "$${absent}" ]; then exit 0; fi; \
	set -- $${absent}; \
	clone_waves=$$(( ($$# + 8 - 1) / 8 )); \
	credential_helper='!f() { if [ "$$1" = get ]; then printf "username=x-access-token\npassword=%s\n" "$${GITHUB_TOKEN}"; fi; }; f'; \
	GIT_TERMINAL_PROMPT=0 timeout --signal=TERM --kill-after=5s "$$(( clone_waves * 120 ))s" \
		git -C "$${root}" -c credential.helper= \
		-c "credential.https://$${GH_HOST:-github.com}.helper=$${credential_helper}" \
		submodule update --init --checkout --depth 1 --jobs "8" -- "$$@"
_builtin_setup_submodules: _builtin_require_workspace

# A provisioned local setup needs no GitHub credential. Mise and Git receive
# only the explicit process credential above; an operation that actually needs
# network access reports its own failure without a preflight or fallback.

# The network verbs refuse a selected credential command that failed or printed
# nothing; the caller's token and anonymous access need no further check.
.PHONY: _builtin_require_github_credential
_builtin_require_github_credential:
	@if [ "$(GITHUB_CREDENTIAL_SOURCE)" = command ] && { [ "$(GITHUB_CREDENTIAL_STATUS)" != 0 ] || [ -z "$${GITHUB_TOKEN:-}" ]; }; then \
		printf 'ERROR: the declared GitHub credential command %s exited %s without a token; refusing anonymous network access\n' 'gh auth token' '$(GITHUB_CREDENTIAL_STATUS)' >&2; \
		exit 2; \
	fi

# Native Mise, not a shell parser, resolves the declared locked self entry.
.PHONY: _builtin_require_mise
_builtin_require_mise:
	@set -eu; \
	if [ ! -f "$(RUNTIME_ROOT)/.mise.toml" ]; then \
		printf 'ERROR: missing project mise config %s; run make gen\n' "$(RUNTIME_ROOT)/.mise.toml" >&2; \
		exit 2; \
	fi; \
	if [ ! -f "$(RUNTIME_ROOT)/mise.lock" ]; then \
		printf 'ERROR: missing %s/mise.lock; run make upg\n' "$(RUNTIME_ROOT)" >&2; \
		exit 2; \
	fi; \
	$(MISE) --version

.PHONY: _builtin_require_interpreter
_builtin_require_interpreter: _builtin_require_workspace
# Documenting the interface (`make help`) must not require the interpreter it
# tells the user how to provision. Only `make help` with no other goal
# skips the check; any combined goal still demands the environment.
ifneq ($(MAKECMDGOALS),help)
	@if [ ! -x "$(RUNTIME_PYTHON)" ]; then \
		printf 'ERROR: missing environment interpreter %s; make setup creates it\n' "$(RUNTIME_PYTHON)" >&2; \
		exit 2; \
	fi
endif

# Operation prerequisites require only this checkout's physical interpreter.
# The check engine's runtime-integrity gate owns environment provenance.
_builtin_require_environment: _builtin_require_interpreter

# === SECTION: setup environment (managed) ===
# Source: computed (MAKE_PROFILE routing)
# Setup provisions tooling and the native project's declared dependencies.
# It never generates or mutates project source; `make gen` owns source generation.
# The lifecycle above owns the single native toolchain installation.
# Attached members delegate publicly before entering it; there is no second
# private installer or alternative source-selection pipeline.
# End SECTION: setup environment

# Upgrade always cures, even when the tree, environment or locks are broken
# (tracker memories operator-ruling-2026-10-02-repair-never-deadlocks and
# operator-ruling-2026-10-09-setup-always-upg-cures; design D1, operator
# approval 2026-10-31). Every stage is native and runs before any flext-infra
# import, each lock written by its own native writer. _upg_toolchain runs in
# the upg process itself, which elects the bootstrap scope without the native
# workspace reader, so neither a missing toolchain nor a corrupt lock can stop it:
#   1. acquire the pinned Mise through the official installer;
#   2. the acquired Mise's own TOML reader (`mise fmt --stdin`, which reads no
#      lock) classifies uv.lock and each governed mise.lock, and an invalid lock
#      is deleted so its native writer regenerates it;
#   3. per governed repository, `mise lock --bump` relocks only the tools whose
#      own dry-run preview is not `[]`, after one whole-lock `[]` gate (an absent
#      lock previews every tool, so it is regenerated in full);
#   4. the locked toolchain is installed from that lock, so a missing, bumped
#      or newly declared tool (Python and uv included) is the one used next.
# _upg_lifecycle then runs, in a fresh Make, the native `uv lock --upgrade`
# with that Python once per project (ADR-030): each governed member in
# .gitmodules order forms its own uv.lock, then this project, whose lock is
# their composition; then the frozen installation, and last the isolated executor
# (make.executor_environment) repairs the one gap Mise does not report: a
# locked tool missing a declared platform. A project whose source no longer
# imports still has every lock written. It generates no projection.
.PHONY: _upg_toolchain _upg_lifecycle
_upg_toolchain: _builtin_require_workspace _builtin_require_github_credential
	+@set -eu; \
	mise_stage=; bootstrap_status=0; \
	trap 'bootstrap_status=$$?; trap - EXIT; \
		if [ -n "$${mise_stage}" ]; then \
			if rm -rf "$${mise_stage}"; then :; else \
				mise_cleanup_status=$$?; \
				printf "ERROR: bootstrap staging cleanup failed (exit %s); original exit %s\n" "$${mise_cleanup_status}" "$${bootstrap_status}" >&2; \
				if [ "$${bootstrap_status}" -eq 0 ]; then bootstrap_status=$${mise_cleanup_status}; fi; \
			fi; \
		fi; \
		exit "$${bootstrap_status}"' EXIT; \
	bootstrap_stage=network-auth; \
	if git config --global --get-regexp '^url\.https?://[^ /]+:[^ /]+@.*\.insteadof$$' >/dev/null; then \
		printf 'ERROR: Global Git URL rewrite configuration embeds credentials and shadows the fleet SSH route; correct that configuration at its owner\n' >&2; \
		exit 2; \
	else \
		rewrite_status=$$?; \
		if [ "$$rewrite_status" -ne 1 ]; then \
			printf 'ERROR: Cannot inspect the global Git URL rewrite configuration\n' >&2; \
			exit "$$rewrite_status"; \
		fi; \
	fi; \
	case " $(CUSTOM_DECLARED_TARGETS) " in \
		*" pre-upg "*) $(SELF_MAKE) pre-upg ;; \
	esac; \
	bootstrap_stage=acquisition; \
	if [ -L "$(PROJECT_ROOT)/.bin" ] && [ ! -e "$(PROJECT_ROOT)/.bin" ]; then rm -f "$(PROJECT_ROOT)/.bin"; fi; mkdir -p "$(PROJECT_ROOT)/.bin"; \
	export MISE_INSTALL_FROM_GITHUB=1 MISE_INSTALL_SKIP_IF_EXISTS=1 MISE_INSTALL_HELP=0; \
	export MISE_VERSION="2026.10.3"; \
	mise_acquired=false; \
	if [ -x "$(BOOTSTRAP_MISE)" ]; then \
		mise_present="$$("$(BOOTSTRAP_MISE)" --version)"; \
		if [ "$${mise_present%% *}" = "$${MISE_VERSION}" ]; then mise_acquired=true; fi; \
	fi; \
	if [ "$${mise_acquired}" = true ]; then \
		printf 'setup: mise %s already acquired at %s\n' "$${MISE_VERSION}" "$(BOOTSTRAP_MISE)"; \
	else \
	case "$${OS:-}" in \
		(Windows_NT) \
			export MISE_INSTALL_PATH="$$(cygpath --windows "$(BOOTSTRAP_MISE)")"; \
			powershell.exe -NoProfile -Command '$$ErrorActionPreference = "Stop"; Invoke-RestMethod https://mise.run/powershell | Invoke-Expression' ;; \
		(*) \
			export MISE_INSTALL_PATH="$(BOOTSTRAP_MISE)"; \
			mise_stage="$$(mktemp -d "$(PROJECT_ROOT)/.bin/.mise-install.XXXXXXXX")"; \
			curl --proto '=https' --tlsv1.2 -fsSL -o "$${mise_stage}/install.sh" https://mise.run; \
			sh "$${mise_stage}/install.sh" ;; \
	esac; \
	fi; \
	uv_lock="$(RUNTIME_ROOT)/uv.lock"; \
	if [ -f "$$uv_lock" ]; then \
		"$(BOOTSTRAP_MISE)" fmt --stdin < /dev/null >/dev/null; \
		if "$(BOOTSTRAP_MISE)" fmt --stdin < "$$uv_lock" >/dev/null; then :; else \
			lint_status=$$?; \
			if [ "$$lint_status" -ne 1 ]; then exit "$$lint_status"; fi; \
			printf 'upg: %s is not valid TOML; deleting it so its native writer regenerates it\n' "$$uv_lock"; \
			rm -f "$$uv_lock"; \
		fi; \
	fi; \
	set -f; \
	$(GOVERNED_CHECKOUTS); \
	for checkout in $$checkouts; do \
		case "$$checkout" in .) lock_root="$$root" ;; *) lock_root="$$root/$$checkout" ;; esac; \
		mise_lock="$$lock_root/mise.lock"; \
		if [ -f "$$mise_lock" ]; then \
		"$(BOOTSTRAP_MISE)" fmt --stdin < /dev/null >/dev/null; \
		if "$(BOOTSTRAP_MISE)" fmt --stdin < "$$mise_lock" >/dev/null; then :; else \
			lint_status=$$?; \
			if [ "$$lint_status" -ne 1 ]; then exit "$$lint_status"; fi; \
			printf 'upg: %s is not valid TOML; deleting it so its native writer regenerates it\n' "$$mise_lock"; \
			rm -f "$$mise_lock"; \
		fi; \
	fi; \
		if [ "$$("$(BOOTSTRAP_MISE)" -C "$$lock_root" lock --bump --dry-run --json)" = "[]" ]; then \
		printf 'upg: %s: no locked version changes\n' "$$lock_root"; \
	else \
		changed_tools=""; \
		for tool in "python" "github:jdx/mise" "uv" "github:direnv/direnv" "taplo" "aqua:ast-grep/ast-grep" "gitleaks" "aqua:boyter/scc" "node" "github:qltysh/qlty" "github:kucherenko/jscpd" "github:microsoft/waza" "aqua:j178/prek"; do \
			if [ "$$("$(BOOTSTRAP_MISE)" -C "$$lock_root" lock --bump --dry-run --json "$$tool")" != "[]" ]; then changed_tools="$$changed_tools $$tool"; fi; \
		done; \
		if [ -z "$$changed_tools" ]; then \
			printf 'upg: %s: pruning lock entries no declared tool owns\n' "$$lock_root"; \
			"$(BOOTSTRAP_MISE)" -C "$$lock_root" lock; \
			if [ "$$("$(BOOTSTRAP_MISE)" -C "$$lock_root" lock --bump --dry-run --json)" != "[]" ]; then \
				printf 'ERROR[upg] %s: mise still previews a lock change that no declared tool owns\n' "$$lock_root" >&2; \
				exit 2; \
			fi; \
		else \
			printf 'upg: %s: locking%s\n' "$$lock_root" "$$changed_tools"; \
			"$(BOOTSTRAP_MISE)" -C "$$lock_root" lock --bump $$changed_tools; \
		fi; \
	fi; \
	done; \
	set +f; \
	export MISE_SHIMS_DIR="$(PROJECT_ROOT)/.bin/shims"; \
	export MISE_LOCKED=1; \
	bootstrap_stage=tool-install; \
	"$(BOOTSTRAP_MISE)" -C "$(PROJECT_ROOT)" install --yes "python" "github:jdx/mise" "uv" "github:direnv/direnv" "taplo" "aqua:ast-grep/ast-grep" "gitleaks" "aqua:boyter/scc" "node" "github:qltysh/qlty" "github:kucherenko/jscpd" "github:microsoft/waza" "aqua:j178/prek"; \
	printf 'upg: locks classified, Mise locks current, locked toolchain provisioned\n'

_upg_lifecycle:
	+@set -eu; \
	bootstrap_stage=network-auth; \
	if git config --global --get-regexp '^url\.https?://[^ /]+:[^ /]+@.*\.insteadof$$' >/dev/null; then \
		printf 'ERROR: Global Git URL rewrite configuration embeds credentials and shadows the fleet SSH route; correct that configuration at its owner\n' >&2; \
		exit 2; \
	else \
		rewrite_status=$$?; \
		if [ "$$rewrite_status" -ne 1 ]; then \
			printf 'ERROR: Cannot inspect the global Git URL rewrite configuration\n' >&2; \
			exit "$$rewrite_status"; \
		fi; \
	fi; \
	mise_bin="$(MISE)"; \
	$(RESOLVE_DESIRED_PYTHON); \
	set -f; \
	$(GOVERNED_CHECKOUTS); \
	for checkout in $$checkouts; do \
		case "$$checkout" in .) continue ;; esac; \
		[ ! -f "$$root/$$checkout/pyproject.toml" ] || \
			$(UV) lock --upgrade --project "$$root/$$checkout" --python "$$desired_python" --no-python-downloads; \
	done; \
	set +f; \
	$(UV) lock --upgrade --project "$(PROJECT_ROOT)" --python "$$desired_python" --no-python-downloads; \
	$(INSTALL_RUNTIME_RECIPE); \
	$(SYNC_UPGRADE_EXECUTOR_RECIPE); \
	PATH="$$(dirname "$$mise_bin"):$$PATH" $(EXECUTOR_FLEXT_INFRA) deps repair-mise-platforms --repository-root "$(RUNTIME_ROOT)"; \
	case " $(CUSTOM_DECLARED_TARGETS) " in \
		*" post-upg "*) $(SELF_MAKE) post-upg ;; \
	esac; \
	$(if $(filter Y,$(CI)),:,$(DIRENV) allow "$(PROJECT_ROOT)")

# Lock resolves a changed dependency declaration (a newly declared requirement)
# into the UV locks and keeps every pin they already hold; only upg moves a pin.
# It runs from the environment setup installed, provisions no tool, and ends with
# the same frozen installation, so the new requirement is installed.
.PHONY: _lock_lifecycle
_lock_lifecycle:
	+@set -eu; \
	bootstrap_stage=network-auth; \
	if git config --global --get-regexp '^url\.https?://[^ /]+:[^ /]+@.*\.insteadof$$' >/dev/null; then \
		printf 'ERROR: Global Git URL rewrite configuration embeds credentials and shadows the fleet SSH route; correct that configuration at its owner\n' >&2; \
		exit 2; \
	else \
		rewrite_status=$$?; \
		if [ "$$rewrite_status" -ne 1 ]; then \
			printf 'ERROR: Cannot inspect the global Git URL rewrite configuration\n' >&2; \
			exit "$$rewrite_status"; \
		fi; \
	fi; \
	mise_bin="$(MISE)"; \
	$(RESOLVE_DESIRED_PYTHON); \
	set -f; \
	$(GOVERNED_CHECKOUTS); \
	for checkout in $$checkouts; do \
		case "$$checkout" in .) continue ;; esac; \
		[ ! -f "$$root/$$checkout/pyproject.toml" ] || \
			$(UV) lock --project "$$root/$$checkout" --python "$$desired_python" --no-python-downloads; \
	done; \
	set +f; \
	$(UV) lock --project "$(PROJECT_ROOT)" --python "$$desired_python" --no-python-downloads; \
	$(INSTALL_RUNTIME_RECIPE)

# Gate, fix and build verbs act on this repository only, with one body per verb
# in every profile: a workspace root evaluates itself exactly as CI does.
_builtin_build_artifacts:

	@$(UV) build --project "$(PROJECT_ROOT)"


# Check is read-only: it runs the gates without --apply, so the tree is left
# unchanged; fix applies the declared repairs of the fixable gates.
# Each gate's declared stage selects its partition: CI=Y runs
# make.check_gates_ci, CI=C (the commit hook) runs
# make.pre_commit_check_gates, CI=N runs
# make.check_gates_local, and an absent CI token runs every active gate.
_builtin_check_all: _builtin_require_environment
	@$(if $(CHECK_PROFILE),FLEXT_INFRA_PROFILE=$(CHECK_PROFILE) ,)FLEXT_INFRA_PROFILE_OUTPUT="$(PROJECT_ROOT)/.reports/check/route.pstats" timeout --signal=TERM --kill-after=5s 605s $(PROJECT_FLEXT_INFRA) workspace check --repository-root "$(if $(filter Y,$(CI)),$(PROJECT_ROOT),$(if $(filter primary,$(CHECK_PHASE)),$(RUNTIME_ROOT),$(PROJECT_ROOT)))"$(if $(CHECK_PHASE), --phase "$$CHECK_PHASE")

_builtin_tests_all: _builtin_require_environment
	@$(PROJECT_FLEXT_INFRA) workspace tests --repository-root "$(PROJECT_ROOT)" --scope "$(GATE_SCOPE)"

_builtin_tests_full_all: _builtin_require_environment
	@$(PROJECT_FLEXT_INFRA) workspace tests-full --repository-root "$(PROJECT_ROOT)" --scope "$(GATE_SCOPE)"

# fmt is format-only (single-pass verb law): ruff formats Python, the
# fmt_gates formatters run once through the checker's apply mode, and every
# lint repair belongs to `make fix`. Every repair requires read-only native
# verification; remaining findings and tool failures both fail the verb.
_builtin_fmt_all: _builtin_require_environment
	@$(PROJECT_FLEXT_INFRA) check run --repository-root "$(PROJECT_ROOT)" --gate-set fmt --scope $(GATE_SCOPE) --apply

_builtin_fix_all: _builtin_require_environment
	@$(PROJECT_FLEXT_INFRA) check run --repository-root "$(PROJECT_ROOT)" --gate-set fix --scope $(GATE_SCOPE) --apply
	@$(PROJECT_FLEXT_INFRA) docs fix --repository-root "$(PROJECT_ROOT)" --output-dir ".reports/docs" --projects . --apply

_builtin_clean_generated: _builtin_require_interpreter
	@$(PROJECT_FLEXT_INFRA) maintenance clean --repository-root "$(PROJECT_ROOT)" --apply

# Generation is a fenced, budgeted operation: `codegen generate` holds the
# tree's mutation lock for the whole run (a concurrent writer waits visibly,
# bounded by config Infra.mutation_lock), terminated inside the declared budget
# with its kill reserve (the profile checkpoint survives a budget kill).
GEN_BUDGET_SECONDS ?= 900
GEN_KILL_RESERVE_SECONDS ?= 30
GEN_PROFILE ?=
# Modification is a budgeted operation too: a stuck catalog run is killed
# inside the declared budget, and MOD_PROFILE=1 captures its cProfile.
MOD_BUDGET_SECONDS ?= 120
MOD_KILL_RESERVE_SECONDS ?= 10
MOD_PROFILE ?=
# Check profiling is opt-in like generation (CHECK_PROFILE=1 make check).
CHECK_PROFILE ?=

_builtin_gen_all: _builtin_require_interpreter
	@set -eu; \
	root="$(abspath $(or $(strip $(ROOT)),$(PROJECT_ROOT)))"; \
	scope=all; \
	if [ "$$root" != "$(PROJECT_ROOT)" ] || [ "$(strip $(CI))" = "Y" ]; then scope=self; fi; \
	$(if $(GEN_PROFILE),FLEXT_INFRA_PROFILE=$(GEN_PROFILE) ,)FLEXT_INFRA_PROFILE_OUTPUT="$$root/.reports/codegen/gen.pstats" timeout --signal=TERM --kill-after=$(GEN_KILL_RESERVE_SECONDS)s $(GEN_BUDGET_SECONDS)s $(PROJECT_FLEXT_INFRA) codegen generate --root "$$root" --scope "$$scope"

# Focal generation is the bounded diagnosis vehicle: one surface (and, for
# facades, one module) through the identical engine, lock and budget.
_builtin_gen_scope_all: _builtin_require_interpreter
	@set -eu; \
	root="$(abspath $(or $(strip $(ROOT)),$(PROJECT_ROOT)))"; \
	scope="$(or $(strip $(SCOPE)),self)"; \
	module_args=""; \
	if [ -n "$(strip $(MODULE))" ]; then module_args=" --module $(strip $(MODULE))"; fi; \
	$(if $(GEN_PROFILE),FLEXT_INFRA_PROFILE=$(GEN_PROFILE) ,)FLEXT_INFRA_PROFILE_OUTPUT="$$root/.reports/codegen/gen-scope.pstats" timeout --signal=TERM --kill-after=$(GEN_KILL_RESERVE_SECONDS)s $(GEN_BUDGET_SECONDS)s $(PROJECT_FLEXT_INFRA) codegen generate --root "$$root" --what "$(strip $(WHAT))" --scope "$$scope"$$module_args

# Structural rewrites run one theme at a time (ADR-028 Decision 18 item 2):
# LAWS selects ENFORCE law ids, RULES selects exact rule ids; the engine refuses
# a run without either.
_builtin_mod_apply: _builtin_require_environment
	@set -eu; \
	selection=""; \
	for law in $(strip $(LAWS)); do selection="$$selection --laws $$law"; done; \
	for rule in $(strip $(RULES)); do selection="$$selection --rule-ids $$rule"; done; \
	$(if $(MOD_PROFILE),FLEXT_INFRA_PROFILE=$(MOD_PROFILE) ,)FLEXT_INFRA_PROFILE_OUTPUT="$(PROJECT_ROOT)/.reports/refactor/mod.pstats" timeout --signal=TERM --kill-after=$(MOD_KILL_RESERVE_SECONDS)s $(MOD_BUDGET_SECONDS)s $(PROJECT_FLEXT_INFRA) refactor mod --repository-root "$(PROJECT_ROOT)"$$selection

# Public operations invoke only their own canonical implementation.
_builtin-build: _builtin_build_artifacts
	@$(PROJECT_FLEXT_INFRA) docs build --repository-root "$(PROJECT_ROOT)" --output-dir ".reports/docs" --projects .

_builtin-check: _builtin_check_all

_builtin-tests: _builtin_tests_all
_builtin-tests-full: _builtin_tests_full_all
_builtin-fmt: _builtin_fmt_all
_builtin-fix: _builtin_fix_all
_builtin-clean: _builtin_clean_generated
_builtin-gen: _builtin_gen_all
_builtin-gen-scope: _builtin_gen_scope_all
_builtin-mod: _builtin_mod_apply
# The repository's release capability elects these product operations.
# Operator values cross the shell boundary as environment data.


export FLEXT_RELEASE_PR_TITLE := $(value PR_TITLE)









export FLEXT_RELEASE_INDEX := $(value INDEX)



_builtin-release-plan: _builtin_require_environment
	@set -eu; set --; \
	if [ -n "$$FLEXT_RELEASE_PR_TITLE" ]; then set -- --pr-title "$$FLEXT_RELEASE_PR_TITLE"; fi; \
	$(PROJECT_FLEXT_INFRA) release run --repository-root "$(PROJECT_ROOT)" --phase plan "$$@"

_builtin-release-version: _builtin_require_environment
	@set -eu; set --; \
	$(PROJECT_FLEXT_INFRA) release run --repository-root "$(PROJECT_ROOT)" --phase version --apply "$$@"

_builtin-release-tag: _builtin_require_environment
	@set -eu; set --; \
	$(PROJECT_FLEXT_INFRA) release run --repository-root "$(PROJECT_ROOT)" --phase tag --apply "$$@"

_builtin-release-build: _builtin_require_environment
	@set -eu; set --; \
	$(PROJECT_FLEXT_INFRA) release run --repository-root "$(PROJECT_ROOT)" --phase build --apply "$$@"

_builtin-release-publish: _builtin_require_environment
	@set -eu; set --; \
	if [ "$$FLEXT_RELEASE_INDEX" = Y ]; then set -- --index; fi; \
	$(PROJECT_FLEXT_INFRA) release run --repository-root "$(PROJECT_ROOT)" --phase publish --apply "$$@"
