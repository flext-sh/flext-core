
# @flext-generated: continuous
# @flext-owner: flext-infra/config/codegen.yaml + flext-infra/src/flext_infra/templates/project/base/Makefile.j2
# @flext-adjust: edit the owner configuration or template; never this projection
# @flext-regenerate: make gen
# flext-core — selector-free generated project interface.
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
ifneq ($(filter $(if $(VERB_CONTRACT),,upg) _upg% dep propagate,$(MAKECMDGOALS)),)
$(error Resolution and member propagation are forbidden in CI)
endif
endif

# Native consumers receive only the caller's explicit GitHub token. Credentials
# stay in the environment; Make never queries another producer or prints them.
GITHUB_TOKEN := $(firstword $(GITHUB_TOKEN) $(GH_TOKEN) $(MISE_GITHUB_TOKEN))
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
override PYTEST_RUN_TIMEOUT_SECONDS := 120
override PYTEST_TERMINATION_GRACE_SECONDS := 2
override PYTEST_TIMEOUT_EXIT_CODE := 124
override PYTEST_ENFORCEMENT_PLUGIN := flext_tests_enforcement
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
ifneq ($(strip $(VERB_CONTRACT) $(filter setup,$(MAKECMDGOALS))),)
override REPOSITORY_ROOT := $(BOOTSTRAP_ROOT)
else ifeq ($(strip $(filter-out help _builtin-help _setup_native_prepare post-upg,$(MAKECMDGOALS))),)
override REPOSITORY_ROOT := $(BOOTSTRAP_ROOT)
else
override REPOSITORY_ROOT := $(shell set -eu; \
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
reader_root="$$( set -eu; \
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
printf '%s\n' "$${project_root}" )"; \
scope="$$(git -C "$${reader_root}" rev-parse --show-toplevel)"; \
scope="$$(cd "$${scope}" && pwd -P)"; \
unset MISE_SYSTEM_CONFIG_FILE MISE_ENV MISE_ENV_FILE UV_PROJECT UV_WORKING_DIR UV_CONFIG_FILE UV_PROJECT_ENVIRONMENT UV_PYTHON UV_PYTHON_DOWNLOADS UV_NO_CONFIG UV_NO_SOURCES UV_FROZEN UV_LOCKED VIRTUAL_ENV PYTHONHOME PYTHONPATH; \
MISE_CONFIG_DIR="$${reader_root}/.mise"; \
MISE_SYSTEM_CONFIG_DIR="$${MISE_CONFIG_DIR}"; \
MISE_GLOBAL_CONFIG_FILE="$${MISE_CONFIG_DIR}/config.toml"; \
MISE_CEILING_PATHS="$$(dirname "$${reader_root}")"; \
MISE_LOCKED=1; MISE_AUTO_INSTALL=0; \
export MISE_CONFIG_DIR MISE_SYSTEM_CONFIG_DIR MISE_GLOBAL_CONFIG_FILE MISE_CEILING_PATHS MISE_LOCKED MISE_AUTO_INSTALL; \
case "$${OS:-}" in (Windows_NT) reader_mise="$${reader_root}/.bin/mise.exe" ;; (*) reader_mise="$${reader_root}/.bin/mise" ;; esac; \
if [ -L "$${reader_root}/.bin" ] || [ -L "$${reader_mise}" ]; then printf 'ERROR: Mise acquisition must be physical: %s\n' "$${reader_mise}" >&2; exit 2; fi; \
reader_mise="$$("$${reader_mise}" -C "$${reader_root}" which mise)"; \
if [ "$${OS:-}" = Windows_NT ]; then reader_mise="$$(cygpath --unix "$${reader_mise}")"; fi; \
uv_binary="$$("$${reader_mise}" -C "$${reader_root}" which "uv")"; \
if [ "$${OS:-}" = Windows_NT ]; then uv_binary="$$(cygpath --unix "$${uv_binary}")"; fi; \
if [ ! -x "$${uv_binary}" ]; then printf 'ERROR: native workspace reader is not provisioned; run make setup\n' >&2; exit 2; fi; \
owner="$$("$${uv_binary}" workspace dir --project "$${project_root}")"; \
if [ "$${OS:-}" = Windows_NT ]; then owner="$$(cygpath --unix "$${owner}")"; fi; \
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
PUBLIC_VERBS := setup upg gen mod fix fmt check tests help build clean
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
ifneq ($(filter setup gen check tests,$(CUSTOM_DECLARED_TARGETS)),)
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

ifneq ($(filter _custom-gen,$(CUSTOM_DECLARED_TARGETS)),)
override CUSTOM_WRAPPER_REFERENCE_COUNT := $(shell awk -v reference="_builtin-gen" 'index($$0, reference) { count++ } END { print count + 0 }' "$(CUSTOM_MAKEFILE)")
ifneq ($(.SHELLSTATUS),0)
$(error Failed to inspect custom approval reference in $(CUSTOM_MAKEFILE))
endif
ifeq ($(CUSTOM_WRAPPER_REFERENCE_COUNT),0)
$(error Approval stage _custom-gen must chain _builtin-gen (wrapper parity; replacements are forbidden))
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

ifneq ($(filter _custom-tests,$(CUSTOM_DECLARED_TARGETS)),)
override CUSTOM_WRAPPER_REFERENCE_COUNT := $(shell awk -v reference="_builtin-tests" 'index($$0, reference) { count++ } END { print count + 0 }' "$(CUSTOM_MAKEFILE)")
ifneq ($(.SHELLSTATUS),0)
$(error Failed to inspect custom approval reference in $(CUSTOM_MAKEFILE))
endif
ifeq ($(CUSTOM_WRAPPER_REFERENCE_COUNT),0)
$(error Approval stage _custom-tests must chain _builtin-tests (wrapper parity; replacements are forbidden))
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
# owned acquisition binary, never host PATH, starts that read-only selection.
override MISE = "$$(set -eu; mise_path="$$("$(BOOTSTRAP_MISE)" -C "$(RUNTIME_ROOT)" which mise)"; if [ "$${OS:-}" = Windows_NT ]; then cygpath --unix "$$mise_path"; else printf '%s\n' "$$mise_path"; fi)"
override UV = "$$(set -eu; uv_mise=$(MISE); uv_path="$$("$$uv_mise" -C "$(RUNTIME_ROOT)" which "uv")"; if [ "$${OS:-}" = Windows_NT ]; then cygpath --unix "$$uv_path"; else printf '%s\n' "$$uv_path"; fi)"
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
override export MISE_CONFIG_DIR := $(RUNTIME_ROOT)/.mise
# The project manifest remains in native project scope. An optional global
# configuration has its own explicit runtime-scoped path; when absent, Mise
# reads no global file. Reusing .mise.toml here classifies every project tool
# as global and leaves the project lock scope empty.
override export MISE_GLOBAL_CONFIG_FILE := $(MISE_CONFIG_DIR)/config.toml
override export MISE_SYSTEM_CONFIG_DIR := $(MISE_CONFIG_DIR)
# A file override wins over the system directory in native Mise discovery.
unexport MISE_SYSTEM_CONFIG_FILE MISE_ENV MISE_ENV_FILE
override export MISE_LOCKED := 1
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
ifeq ($(OS),Windows_NT)
override RUNTIME_BIN := $(RUNTIME_VENV)/Scripts
override RUNTIME_PYTHON := $(RUNTIME_BIN)/python.exe
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
unexport UV MYPYPATH UV_PROJECT UV_WORKING_DIR UV_CONFIG_FILE UV_PROJECT_ENVIRONMENT UV_PYTHON UV_PYTHON_DOWNLOADS UV_NO_CONFIG UV_NO_SOURCES UV_FROZEN UV_LOCKED VIRTUAL_ENV PYTHONHOME PYTHONPATH
export FLEXT_INFRA_PYTHON UV_PROJECT_ENVIRONMENT VIRTUAL_ENV PATH RUNTIME_ROOT

# Tool provisioning belongs to setup, never to upgrade or generation.
TOOL_BOOTSTRAP_LIFECYCLE := _setup_native_prepare
.PHONY: _bootstrap_setup_tools

_bootstrap_setup_tools:
	# The lifecycle invokes recursive make through mise, so preserve jobserver FDs.
	+@set -eu; \
	bootstrap_stage=source-identity; bootstrap_completed=false; bootstrap_sha=null; bootstrap_git_status=null; mise_stage=; \
	bootstrap_report="$(PROJECT_ROOT)/.reports/setup-operations/tool-bootstrap.json"; \
	bootstrap_publish_report() { \
		if mkdir -p "$$(dirname "$${bootstrap_report}")"; then :; else return $$?; fi; \
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
	mkdir -p "$(PROJECT_ROOT)/.bin"; \
	export MISE_INSTALL_FROM_GITHUB=1 MISE_INSTALL_SKIP_IF_EXISTS=1 MISE_INSTALL_MODIFY_PATH=0 MISE_INSTALL_HELP=0; \
	export MISE_VERSION="2026.10.3"; \
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
	export MISE_SHIMS_DIR="$(PROJECT_ROOT)/.bin/shims"; \
	bootstrap_stage=tool-install; \
	"$(BOOTSTRAP_MISE)" -C "$(PROJECT_ROOT)" install --locked --yes "python" "github:jdx/mise" "uv" "github:direnv/direnv" "taplo" "aqua:ast-grep/ast-grep" "gitleaks" "aqua:boyter/scc" "node" "github:qltysh/qlty" "github:kucherenko/jscpd" "github:microsoft/waza"; \
	mise_active="$$("$(BOOTSTRAP_MISE)" -C "$(PROJECT_ROOT)" tool "github:jdx/mise" --active)"; \
	mise_binary="$$("$(BOOTSTRAP_MISE)" -C "$(PROJECT_ROOT)" which mise)"; \
	if [ "$${OS:-}" = Windows_NT ]; then mise_binary="$$(cygpath --unix "$${mise_binary}")"; fi; \
	bootstrap_stage=shims; \
	"$${mise_binary}" -C "$(PROJECT_ROOT)" reshim; \
	if [ -n "$${GITHUB_PATH:-}" ]; then printf '%s\n' "$$(dirname "$${mise_binary}")" "$${MISE_SHIMS_DIR}" >> "$${GITHUB_PATH}"; fi; \
	printf 'setup: mise %s provisioning complete\n' "$${mise_active}"; \
	printf 'setup: entering lifecycle (submodules, environment, hooks) make=%s\n' "$(SELF_MAKE_EXECUTABLE)"; \
	bootstrap_stage=lifecycle; \
	MISE_AUTO_INSTALL=0 "$${mise_binary}" -C "$(PROJECT_ROOT)" exec -- env "PATH=$$(dirname "$${mise_binary}"):$${MISE_SHIMS_DIR}:$${PATH}" "CI=$(CI)" $(SELF_MAKE) $(TOOL_BOOTSTRAP_LIFECYCLE); \
	bootstrap_stage=complete; bootstrap_completed=true
_bootstrap_setup_tools: _builtin_require_workspace

# Every repository evaluates only itself: a workspace root consumes its members
# as installed libraries and each member runs its own lifecycle in its own
# repository. Locally, the mutating verbs fmt/fix/mod then carry that same
# member-own verb to every governed member (fleet_fanout); CI never fans out.
# Local and CI consume the same native project/workspace declarations and lock.
# Only installation mode changes in CI; no resolver or source is substituted.
INSTALL_RUNTIME_RECIPE = set -eu; \
	$(REQUIRE_WORKSPACE_ENVIRONMENT); \
	desired_python="$$($(MISE) -C "$(RUNTIME_ROOT)" which python)"; \
	if [ "$${OS:-}" = Windows_NT ]; then desired_python="$$(cygpath --unix "$$desired_python")"; fi; \
	if [ ! -x "$$desired_python" ]; then \
		printf 'ERROR[setup] Mise-resolved Python is not executable: %s\n' "$$desired_python" >&2; \
		exit 2; \
	fi; \
	UV_PROJECT_ENVIRONMENT="$(RUNTIME_VENV)" timeout --signal=TERM --kill-after=5s 118s $(UV) sync --locked --project "$(RUNTIME_ROOT)" --python "$$desired_python" --no-python-downloads $(SETUP_SYNC_ARGS)

# Reject borrowed environments before bootstrap, activation or custom hooks.
# Members may use their containing workspace, never an unrelated checkout.
REQUIRE_WORKSPACE_ENVIRONMENT = case "$(PROJECT_ROOT)/" in \
	"$(RUNTIME_ROOT)/"*) ;; \
	*) printf 'ERROR: runtime workspace does not contain this project: %s\n' "$(RUNTIME_ROOT)" >&2; exit 2 ;; \
	esac; \
	for environment_path in "$$(dirname "$(RUNTIME_VENV)")" "$(RUNTIME_VENV)" "$(RUNTIME_BIN)" "$(PROJECT_ROOT)/.venv" "$(PROJECT_ROOT)/.venv/bin" "$(PROJECT_ROOT)/.venv/Scripts" "$(SETUP_BIN)" "$(BOOTSTRAP_MISE)"; do \
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
override PROJECT_INFRA_RUN = env -u MYPYPATH -u UV_PROJECT -u UV_WORKING_DIR -u UV_CONFIG_FILE -u UV_PROJECT_ENVIRONMENT -u UV_PYTHON -u UV_PYTHON_DOWNLOADS -u UV_NO_CONFIG -u UV_NO_SOURCES -u UV_FROZEN -u UV_LOCKED -u VIRTUAL_ENV -u PYTHONHOME -u PYTHONPATH "$(FLEXT_INFRA_PYTHON)"
override PROJECT_FLEXT_INFRA := $(PROJECT_INFRA_RUN) -m flext_infra

# Setup checks native lock freshness without writing the lock. Native uv selects
# every member's declared extras and groups, without copying dependency graphs.


override SETUP_SYNC_ARGS := --all-extras --all-groups --link-mode copy

ifeq ($(CI),Y)
override SETUP_SYNC_ARGS := --all-extras --all-groups --link-mode copy --no-editable
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



# `make upg` is the only verb that writes uv.lock (`uv lock --upgrade
# --refresh`, then `uv lock` of the manifest `gen` projected and `uv lock
# --check`). Setup never writes it (lock law above).

.PHONY: $(PUBLIC_VERBS) $(addprefix _builtin-,$(PUBLIC_VERBS))
.PHONY: _builtin_gen_all


VERB_DESCRIPTION_setup := Install the declared frozen locks, synchronize the physical environment and configure local integrations.
VERB_ARGUMENTS_setup :=

VERB_DESCRIPTION_upg := Resolve declared dependency and tool updates and publish their resolution metadata and locks.
VERB_ARGUMENTS_upg :=

VERB_DESCRIPTION_gen := Materialize explicitly declared new projects and publish every applicable generated projection.
VERB_ARGUMENTS_gen :=

VERB_DESCRIPTION_mod := Transform declared source structure and publish the complete coherent semantic change through the shared engines.
VERB_ARGUMENTS_mod :=

VERB_DESCRIPTION_fix := Apply declared repairs to authored inputs through their canonical owners.
VERB_ARGUMENTS_fix :=

VERB_DESCRIPTION_fmt := Format authored content through the configured formatters.
VERB_ARGUMENTS_fmt :=

VERB_DESCRIPTION_check := Validate declarations, projections, environment, documentation and configured gates without repairs or publication.
VERB_ARGUMENTS_check :=

VERB_DESCRIPTION_tests := Execute the complete applicable suite once with the persistent testmon database.
VERB_ARGUMENTS_tests :=

VERB_DESCRIPTION_help := Discover the declared public interface without effects.
VERB_ARGUMENTS_help :=

VERB_DESCRIPTION_build := Build local distribution and documentation artifacts without publishing them.
VERB_ARGUMENTS_build :=

VERB_DESCRIPTION_clean := Remove only declared disposable resources owned by the selected target.
VERB_ARGUMENTS_clean :=

VERB_HELP_LINE = printf '  %-16s %s\n' '$(1)' '$(VERB_DESCRIPTION_$(1))'$(if $(strip $(VERB_ARGUMENTS_$(1))),; printf '    Inputs: %s\n' '$(VERB_ARGUMENTS_$(1))')
# OPTIONS=Y (or HELP=Y) displays every public verb's catalog contract without
# running its prerequisites, hooks, handler or product dispatcher.
ifeq ($(VERB_CONTRACT),)
PROFILE_VERBS_PLAIN := gen
PROFILE_VERBS_DISPATCH := help clean
PROFILE_VERBS_ACTIVATED := mod fix fmt check tests build
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
	+@$(if $(filter Y,$(CI)),$(SELF_MAKE) _activated-$@,direnv exec "$(PROJECT_ROOT)" $(SELF_MAKE) _activated-$@)
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
setup: _bootstrap_setup_tools
else
setup: _builtin_require_workspace
	+@"$(SELF_MAKE_EXECUTABLE)" --no-print-directory -C "$(RUNTIME_ROOT)" -f "$(RUNTIME_ROOT)/Makefile" setup
endif

# Upgrade only resolves dependencies and writes locks. Provisioning belongs
# to setup and projections belong to gen. Attached members delegate the same
# operation to their physical workspace owner before resolving anything.
ifeq ($(PROJECT_ROOT),$(RUNTIME_ROOT))
upg: _builtin_require_workspace
	+@$(SELF_MAKE) _upg_lifecycle
else
upg: _builtin_require_workspace
	+@"$(SELF_MAKE_EXECUTABLE)" --no-print-directory -C "$(RUNTIME_ROOT)" -f "$(RUNTIME_ROOT)/Makefile" upg
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
	@$(INSTALL_RUNTIME_RECIPE); \
	$(PROJECT_FLEXT_INFRA) workspace verify-environment --repository-root "$(PROJECT_ROOT)"; \
	case "$(strip $(CI)): $(CUSTOM_DECLARED_TARGETS) " in \
		Y:*) ;; \
		*" pre-setup "*) $(SELF_MAKE) pre-setup ;; \
	esac; \
	$(PROJECT_FLEXT_INFRA) workspace sync-environment --repository-root "$(PROJECT_ROOT)" --scope self $(if $(filter Y,$(CI)),--no-allow-direnv,--allow-direnv); \
	$(if $(filter Y,$(CI)),$(SELF_MAKE) _setup_activated,direnv exec "$(PROJECT_ROOT)" $(SELF_MAKE) _setup_activated)

# Setup consumes the native locked inputs without resolving versions. Required
# input and native installation failures terminate immediately;
# generation and explicit upgrade own their respective repairs. Quality gates
# remain separate operations and no failure selects another producer or source.
.PHONY: _setup_activated
# Check owns quality verdicts after provisioning.
_setup_activated:
	@set -eu; \
	case "$(strip $(CI)): $(CUSTOM_DECLARED_TARGETS) " in \
		Y:*) ;; \
		*" post-setup "*) $(SELF_MAKE) post-setup ;; \
	esac

_builtin-help:
	@printf '%s\n' 'flext-core [standalone]' ''
	@$(foreach verb,$(PUBLIC_VERBS),$(call VERB_HELP_LINE,$(verb));)

# A project owns the sources declared by its manifest. The generated setup
# reconciler validates every initialized checkout before mutation, initializes
# only missing modules, and preserves declared branches that fix forward beyond
# the recorded gitlink.
.PHONY: _builtin_setup_submodules

# === SECTION: submodule setup (managed) ===
# Source: template (submodule_setup_recipe.j2)
# Computed: every checkout discovers governed paths from its own .gitmodules.
#           An absent management flag participates; an explicit false does not.
# Rule: setup PROVISIONS an absent or proven unfinished initial clone and
#       VERIFIES an established checkout. An initial clone has no physical index,
#       no worktree content, and only its clone reflog entry. Established work
#       is never destroyed: git checkout, git reset,
#       fetch, and branch attachment are forbidden. Pin validity is HEAD contains
#       gitlink. Declared branch is the named integration line;
#       legacy branch=. still resolves to the superproject named branch if present.
#       A present checkout may be on its own named change lane. Its branch name is
#       not a safety boundary: exact containment of the recorded gitlink is.
#       Nested gitlinks belong to their own setup.
# Free: no
# End SECTION: submodule setup
# Why: runners expose umask 002 and `submodule update --init`
# materializes tracked files as 0664; canonical Mise artifact gates demand
# exact modes, so provisioning normalizes the umask before checkout.
# An absent gitlink is cloned at depth 1, the same flag private submodule
# init uses. Setup's contract is the recorded commit, and a full history
# cannot finish inside submodule_timeout_seconds when the object database
# is large. Absent gitlinks clone submodule_jobs at a time; the deadline is
# one submodule_timeout_seconds per wave of concurrent clones, so a fresh
# checkout of the whole fleet gets the same per-clone budget as one member.
# Derive the physical index from its Git directory: --git-path resolves
# a final symlink and cannot prove that the index entry itself is absent.
_builtin_setup_submodules:
	@set -eu; \
	umask 022; \
	root="$(PROJECT_ROOT)"; \
	if [ ! -f "$${root}/.gitmodules" ]; then exit 0; fi; \
	managed=""; \
	keys=""; \
	if keys=$$(git -C "$${root}" config -f .gitmodules --name-only --get-regexp '^submodule\..*\.path$$'); then \
		keys_status=0; \
	else \
		keys_status=$$?; \
	fi; \
	if [ "$${keys_status}" -ne 0 ] && [ "$${keys_status}" -ne 1 ]; then \
		printf 'ERROR: cannot enumerate governed gitlinks\n' >&2; \
		exit "$${keys_status}"; \
	fi; \
	for key in $${keys}; do \
		section=$${key%.path}; \
		value=$$(git -C "$${root}" config -f .gitmodules --type=bool --get --default true "$${section}.flext-managed"); \
		case "$${value}" in \
			true) ;; \
			false) continue ;; \
			*) printf 'ERROR: Git returned an invalid management boolean for %s\n' "$${section}" >&2; exit 2 ;; \
		esac; \
		path=$$(git -C "$${root}" config -f .gitmodules --get "$${key}"); \
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
	if [ -z "$${managed}" ]; then exit 0; fi; \
	if [ "$${GIT_INDEX_FILE+x}" = x ]; then \
		printf 'ERROR: submodule setup cannot authenticate a relocated Git index\n' >&2; \
		exit 2; \
	fi; \
	absent=""; \
	for path in $${managed}; do \
		child="$${root}/$${path}"; \
		if [ ! -e "$${child}/.git" ]; then \
			absent="$${absent} $${path}"; \
			continue; \
		fi; \
		owner=$$(git -C "$${child}" rev-parse --show-superproject-working-tree); \
		checkout_root=$$(git -C "$${child}" rev-parse --show-toplevel); \
		if [ "$${owner}" != "$${root}" ] || [ "$${checkout_root}" != "$${child}" ]; then \
			printf 'ERROR: %s: checkout identity does not match the governed child\n' "$${path}" >&2; \
			exit 2; \
		fi; \
		git_dir=$$(git -C "$${child}" rev-parse --absolute-git-dir); \
		index="$${git_dir}/index"; \
		if [ -e "$${index}" ] || [ -L "$${index}" ]; then continue; fi; \
		content=$$(git -C "$${child}" ls-files --others --directory); \
		reflog=$$(git -C "$${child}" reflog show --format=%gs HEAD); \
		initial=$$(git -C "$${child}" reflog show --format=%gs -1 HEAD); \
		if [ -n "$${content}" ] || [ "$${reflog}" != "$${initial}" ]; then \
			printf 'ERROR: %s: missing index with unproven initial-clone state; preserve and review\n' "$${path}" >&2; \
			exit 2; \
		fi; \
		case "$${initial}" in \
			'clone: from '*) ;; \
			*) printf 'ERROR: %s: missing index without initial clone receipt\n' "$${path}" >&2; exit 2 ;; \
		esac; \
		local_refs=$$(git -C "$${child}" for-each-ref --format='%(refname)' refs/heads refs/stash); \
		if active_ref=$$(git -C "$${child}" symbolic-ref -q HEAD); then \
			:; \
		else \
			ref_status=$$?; \
			[ "$${ref_status}" -eq 1 ] || exit "$${ref_status}"; \
		fi; \
		if [ "$${local_refs}" != "$${active_ref}" ]; then \
			printf 'ERROR: %s: missing index with local branch or stash work; preserve and review\n' "$${path}" >&2; \
			exit 2; \
		fi; \
		entry=$$(git -C "$${root}" ls-files --stage -- "$${path}"); \
		set -- $${entry}; \
		if [ "$$#" -ne 4 ] || [ "$$1" != 160000 ] || [ "$$3" != 0 ] || [ "$$4" != "$${path}" ]; then \
			printf 'ERROR: governed path is not a gitlink: %s\n' "$${path}" >&2; \
			exit 2; \
		fi; \
		head=$$(git -C "$${child}" rev-parse HEAD); \
		if [ "$${head}" = "$$2" ]; then \
			printf 'setup: materializing unfinished initial clone at recorded pin: %s\n' "$${path}"; \
			GIT_TERMINAL_PROMPT=0 timeout --signal=TERM --kill-after=5s "120s" \
				git -C "$${child}" -c submodule.recurse=false checkout --detach --no-overwrite-ignore "$${head}"; \
			continue; \
		fi; \
		printf 'setup: resuming unfinished initial clone: %s\n' "$${path}"; \
		absent="$${absent} $${path}"; \
	done; \
	if [ -n "$${absent}" ]; then \
		credential_helper='!f() { if [ "$$1" = get ]; then printf "username=x-access-token\npassword=%s\n" "$${GITHUB_TOKEN}"; fi; }; f'; \
		absent_count=$$(printf '%s\n' $${absent} | wc -l); \
		clone_waves=$$(( (absent_count + 8 - 1) / 8 )); \
		GIT_TERMINAL_PROMPT=0 timeout --signal=TERM --kill-after=5s "$$(( clone_waves * 120 ))s" \
			git -C "$${root}" -c credential.helper= \
			-c "credential.https://$${GH_HOST:-github.com}.helper=$${credential_helper}" \
			submodule update --init --checkout --depth 1 --jobs "8" -- $${absent}; \
	fi; \
	validate_submodule() { \
		superproject="$$1"; \
		child_path="$$2"; \
		child_root="$${superproject}/$${child_path}"; \
		keys=""; \
		if keys=$$(git -C "$${superproject}" config -f .gitmodules --name-only --get-regexp '^submodule\..*\.path$$'); then \
			keys_status=0; \
		else \
			keys_status=$$?; \
		fi; \
		if [ "$${keys_status}" -ne 0 ] && [ "$${keys_status}" -ne 1 ]; then \
			printf 'ERROR: cannot enumerate submodule paths\n' >&2; \
			exit "$${keys_status}"; \
		fi; \
		section=""; \
		for key in $${keys}; do \
			declared=$$(git -C "$${superproject}" config -f .gitmodules --get "$${key}"); \
			if [ "$${declared}" = "$${child_path}" ]; then \
				if [ -n "$${section}" ]; then \
					printf 'ERROR: governed gitlink path is duplicated: %s\n' "$${child_path}" >&2; \
					exit 2; \
				fi; \
				section=$${key%.path}; \
			fi; \
		done; \
		if [ -z "$${section}" ]; then \
			printf 'ERROR: governed gitlink is absent from .gitmodules: %s\n' "$${child_path}" >&2; \
			exit 2; \
		fi; \
		branch=$$(git -C "$${superproject}" config -f .gitmodules --get --default "" "$${section}.branch"); \
		if [ -z "$${branch}" ]; then \
			printf 'ERROR: governed gitlink has no declared branch: %s\n' "$${child_path}" >&2; \
			exit 2; \
		fi; \
		super_branch=$$(git -C "$${superproject}" branch --show-current); \
		if [ "$${branch}" = "." ]; then \
			branch="$${super_branch}"; \
			if [ -z "$${branch}" ]; then \
				printf 'ERROR: %s: branch = . requires a named superproject branch\n' "$${child_path}" >&2; \
				exit 1; \
			fi; \
		fi; \
		validated_branch=$$(git check-ref-format --branch "$${branch}"); \
		if [ "$${validated_branch}" != "$${branch}" ]; then \
			printf 'ERROR: branch validator changed %s to %s\n' "$${branch}" "$${validated_branch}" >&2; \
			exit 2; \
		fi; \
		gitlink_entry=$$(git -C "$${superproject}" ls-files --stage -- "$${child_path}"); \
		if [ -z "$${gitlink_entry}" ]; then \
			printf 'ERROR: governed gitlink is absent from the index: %s\n' "$${child_path}" >&2; \
			exit 2; \
		fi; \
		set -- $${gitlink_entry}; \
		if [ "$$1" != 160000 ]; then \
			printf 'ERROR: governed path is not a gitlink: %s\n' "$${child_path}" >&2; \
			exit 2; \
		fi; \
		gitlink="$$2"; \
		if [ ! -e "$${child_root}/.git" ]; then \
			printf 'ERROR: governed gitlink is not materialized: %s\n' "$${child_path}" >&2; \
			exit 2; \
		fi; \
		current=$$(git -C "$${child_root}" branch --show-current); \
		head=$$(git -C "$${child_root}" rev-parse HEAD); \
		if git -C "$${child_root}" merge-base --is-ancestor "$${gitlink}" HEAD; then \
			ancestor=Y; \
		else \
			status=$$?; if [ "$${status}" -eq 1 ]; then ancestor=N; else exit "$${status}"; fi; \
		fi; \
		if [ "$${ancestor}" = N ]; then \
			if [ -z "$${current}" ]; then \
				printf 'ERROR: %s: detached HEAD %s does not contain recorded gitlink %s; merge the superproject gitlink into this head\n' "$${child_path}" "$${head}" "$${gitlink}" >&2; \
			else \
				printf 'ERROR: %s: checked-out branch %s at %s does not contain recorded gitlink %s; merge the superproject gitlink into this branch\n' "$${child_path}" "$${current}" "$${head}" "$${gitlink}" >&2; \
			fi; \
			exit 1; \
		fi; \
	}; \
	for child_path in $${managed}; do \
		validate_submodule "$${root}" "$${child_path}"; \
	done
_builtin_setup_submodules: _builtin_require_workspace

# A provisioned local setup needs no GitHub credential. Mise and Git receive
# only the explicit process credential above; an operation that actually needs
# network access reports its own failure without a preflight or fallback.

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
# The lifecycle above owns the single native locked installation.
# Attached members delegate publicly before entering it; there is no second
# private installer or alternative source-selection pipeline.
# End SECTION: setup environment

# Upgrade consumes the native declarations without installing packages,
# generating projections, or auditing the running environment.
.PHONY: _upg_lifecycle
_upg_lifecycle:
	@set -eu; \
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
	$(PROJECT_FLEXT_INFRA) deps upgrade-lock --repository-root "$(RUNTIME_ROOT)"; \
	env -u MISE_SYSTEM_CONFIG_FILE \
	MISE_CEILING_PATHS="$$(dirname "$(PROJECT_ROOT)")" \
	MISE_CONFIG_DIR="$(PROJECT_ROOT)/.mise" \
	MISE_GLOBAL_CONFIG_FILE="$(PROJECT_ROOT)/.mise/config.toml" \
	MISE_SYSTEM_CONFIG_DIR="$(PROJECT_ROOT)/.mise" \
	$(MISE) -C "$(PROJECT_ROOT)" lock --bump; \
	case " $(CUSTOM_DECLARED_TARGETS) " in \
		*" post-upg "*) $(SELF_MAKE) post-upg ;; \
	esac

# Gate, fix and build verbs act on this repository only, with one body per verb
# in every profile: a workspace root evaluates itself exactly as CI does.
_builtin_build_artifacts:

	@$(UV) build --project "$(PROJECT_ROOT)"


# Check is read-only: it runs the gates without --apply, so the tree is left
# unchanged; fix applies the declared repairs of the fixable gates.
# CI=Y keeps make.check_gates_ci, the strict complement of
# make.check_gates_local; CI=N runs that local partition.
# An absent CI token runs every active default gate.
_builtin_check_all: _builtin_require_environment
	@set -eu; \
		gates="runtime-integrity,lint,format,security,markdown,markdown-format,markdown-code,duplication,pyrefly,mypy,pyright,conflict-markers,loc-cap,runtime-census,fresh-import,deps,index-declarations,smells,codemod,layout,direnv"; \
		if [ "$(strip $(CI))" = "Y" ]; then \
			gates="runtime-integrity,lint,format,security,markdown,markdown-format,markdown-code,duplication,pyrefly,conflict-markers,loc-cap,runtime-census,fresh-import,deps,index-declarations,layout"; \
			printf 'INFO: CI=Y runs check gates: runtime-integrity lint format security markdown markdown-format markdown-code duplication pyrefly conflict-markers loc-cap runtime-census fresh-import deps index-declarations layout\n'; \
		elif [ "$(strip $(CI))" = "N" ]; then \
			gates="runtime-integrity,lint,format,security,markdown,markdown-format,markdown-code,duplication,pyrefly,mypy,pyright,conflict-markers,loc-cap,runtime-census,fresh-import,deps,index-declarations,smells,codemod,layout,direnv"; \
			printf 'INFO: CI=N runs check gates: runtime-integrity lint format security markdown markdown-format markdown-code duplication pyrefly mypy pyright conflict-markers loc-cap runtime-census fresh-import deps index-declarations smells codemod layout direnv\n'; \
		else \
			printf 'INFO: default context runs check gates: runtime-integrity lint format security markdown markdown-format markdown-code duplication pyrefly mypy pyright conflict-markers loc-cap runtime-census fresh-import deps index-declarations smells codemod layout direnv\n'; \
		fi; \
		if [ -z "$$gates" ]; then \
			printf 'ERROR: no active check gates remain in the selected context\n' >&2; \
			exit 2; \
		fi; \
		$(PROJECT_FLEXT_INFRA) workspace capture-gate --repository-root "$(PROJECT_ROOT)" --gates "$$gates"; \
		$(PROJECT_FLEXT_INFRA) codegen conform --root "$(PROJECT_ROOT)" --scope self --mode check

_builtin_tests_all: _builtin_require_environment
	@set -eu; \
database="$(FLEXT_PYTEST_TESTMON_DATABASE)"; \
case "$$database" in /*) ;; *) printf 'ERROR: persistent testmon database requires XDG_CACHE_HOME or HOME\n' >&2; exit 2 ;; esac; \
case "$$database" in "$(PROJECT_ROOT)"/*) printf 'ERROR: persistent testmon database must be outside the checkout: %s\n' "$$database" >&2; exit 2 ;; esac; \
case "$$database" in "$${TMPDIR:-/tmp}"/*|/tmp/*) printf 'ERROR: persistent testmon database must not live under the temporary directory: %s\n' "$$database" >&2; exit 2 ;; esac; \
mkdir -p "$$(dirname "$$database")"; \
scratch_root="$(FLEXT_PYTEST_SCRATCH_ROOT)"; \
case "$$scratch_root" in /*) ;; *) printf 'ERROR: pytest scratch root requires HOME: %s\n' "$$scratch_root" >&2; exit 2 ;; esac; \
mkdir -p "$$scratch_root"; \
scratch="$$(mktemp -d "$$scratch_root/pytest.XXXXXX")"; \
trap 'find "$$scratch" -depth -delete' EXIT; \
mkdir -p "$$scratch/tmp"; \
scratch_tmp="$$(cd "$$scratch/tmp" && pwd -P)"; \
TMPDIR="$$scratch_tmp"; TMP="$$scratch_tmp"; TEMP="$$scratch_tmp"; \
export TMPDIR TMP TEMP; \
unset FLEXT_PYTEST_TARGET_FILE; \
TESTMON_DATAFILE="$$database" $(PROJECT_INFRA_RUN) -m flext_infra._pytest_entry full

# fmt is format-only (single-pass verb law): ruff formats Python, the
# fmt_gates formatters run once through the checker's apply mode, and every
# lint repair belongs to `make fix`. Every repair requires read-only native
# verification; remaining findings and tool failures both fail the verb.
_builtin_fmt_all: _builtin_require_environment
	@$(PROJECT_FLEXT_INFRA) check run --repository-root "$(PROJECT_ROOT)" --gates "format,markdown-format,markdown-code" --apply

_builtin_fix_all: _builtin_require_environment
	@$(PROJECT_FLEXT_INFRA) check run --repository-root "$(PROJECT_ROOT)" --gates "lint,markdown,deps" --apply
	@$(PROJECT_FLEXT_INFRA) docs fix --repository-root "$(PROJECT_ROOT)" --output-dir ".reports/docs" --projects . --apply

_builtin_clean_generated: _builtin_require_interpreter
	@$(PROJECT_FLEXT_INFRA) maintenance clean --repository-root "$(PROJECT_ROOT)" --apply

_builtin_gen_all: _builtin_require_interpreter
	@set -eu; \
	scope=all; \
	if [ "$(strip $(CI))" = "Y" ]; then scope=self; fi; \
	$(PROJECT_INFRA_RUN) -m flext_infra._profile_entry "$(PROJECT_ROOT)/.reports/codegen/gen.pstats" codegen generate --root "$(PROJECT_ROOT)" --scope "$$scope"

# Structural rewrites use the target's existing semantic operation.
_builtin_mod_apply: _builtin_require_environment
	@$(PROJECT_INFRA_RUN) -m flext_infra._profile_entry "$(PROJECT_ROOT)/.reports/refactor/mod.pstats" refactor mod --repository-root "$(PROJECT_ROOT)" --apply

# Public operations invoke only their own canonical implementation.
_builtin-build: _builtin_build_artifacts
	@$(PROJECT_FLEXT_INFRA) docs build --repository-root "$(PROJECT_ROOT)" --output-dir ".reports/docs" --projects .

_builtin-check: _builtin_check_all
ifneq ($(CI),Y)
	@$(PROJECT_FLEXT_INFRA) workspace verify-lanes --repo-root "$(PROJECT_ROOT)"
endif
	@$(PROJECT_FLEXT_INFRA) docs validate --repository-root "$(PROJECT_ROOT)" --output-dir ".reports/docs" --projects .
	@$(PROJECT_FLEXT_INFRA) docs audit --repository-root "$(PROJECT_ROOT)" --output-dir ".reports/docs" --projects .
	@$(PROJECT_FLEXT_INFRA) workspace verify-clean --repo-root "$(PROJECT_ROOT)"

_builtin-tests: _builtin_tests_all
_builtin-fmt: _builtin_fmt_all
_builtin-fix: _builtin_fix_all
_builtin-clean: _builtin_clean_generated
_builtin-gen: _builtin_gen_all
_builtin-mod: _builtin_mod_apply
