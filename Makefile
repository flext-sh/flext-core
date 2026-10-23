
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

# Approval fixes its context before any topology, activation or credential read.
ifneq ($(filter pre-commit,$(MAKECMDGOALS)),)
override CI := Y
export CI
ifneq ($(strip $(HELP) $(OPTIONS)),)
$(error Approval cannot run with HELP or OPTIONS)
endif
ifneq ($(strip $(foreach flag,n q t,$(findstring $(flag),$(firstword $(MAKEFLAGS))))),)
$(error Approval requires execution; dry-run, question and touch are forbidden)
endif
ifneq ($(strip $(MAKEFILES)),)
$(error Approval cannot load caller-supplied Makefiles)
endif
ifneq ($(filter command line override,$(origin MAKE_COMMAND)),)
$(error Approval cannot replace the native Make invoker)
endif
endif

ifeq ($(strip $(GITHUB_ACTIONS)),true)
ifneq ($(filter upg _upg% dep propagate,$(MAKECMDGOALS)),)
$(error Resolution and member propagation are forbidden in hosted CI)
endif
endif

# Capture the selected approval mode before any project-owned include.
ifeq ($(strip $(CI)),Y)
override APPROVAL_CONTEXT := Y
ifneq ($(filter upg _upg% dep propagate,$(MAKECMDGOALS)),)
$(error Resolution and member propagation are forbidden in CI)
endif
endif

ifeq ($(filter command line override,$(origin SETUP_BOOTSTRAP_ONLY)),)
ifneq ($(filter setup,$(MAKECMDGOALS)),)
SETUP_BOOTSTRAP_ONLY := Y
export SETUP_BOOTSTRAP_ONLY
endif
endif
ifeq ($(filter command line override,$(origin GEN_INIT_ONLY)),)
ifneq ($(filter initialize,$(MAKECMDGOALS)),)
GEN_INIT_ONLY := Y
export GEN_INIT_ONLY
endif
endif

# One GitHub credential for every verb (operator 2026-10-03): the first
# non-empty of the caller's GITHUB_TOKEN, GH_TOKEN, MISE_GITHUB_TOKEN, then the
# declared credential command when it is installed. Every name gh, uv and mise
# read carries that same value, so no inherited alias can shadow it. The value
# stays in the environment and is never printed.
GITHUB_AUTH_SOURCE := $(if $(strip $(GITHUB_TOKEN)),GITHUB_TOKEN,$(if $(strip $(GH_TOKEN)),GH_TOKEN,$(if $(strip $(MISE_GITHUB_TOKEN)),MISE_GITHUB_TOKEN,none)))
GITHUB_AUTH_STATUS := not-selected
override GITHUB_AUTH_COMMAND := not-selected
override GITHUB_AUTH_COMMAND_STATUS := not-selected
GITHUB_AUTH_HOST_SOURCE := $(if $(strip $(GH_HOST)),GH_HOST,native-default)
GITHUB_AUTH_CONFIG_OVERRIDE := $(if $(strip $(GH_CONFIG_DIR)),yes,no)
GITHUB_AUTH_XDG_OVERRIDE := $(if $(strip $(XDG_CONFIG_HOME)),yes,no)
GITHUB_AUTH_SESSION_BUS := $(if $(strip $(DBUS_SESSION_BUS_ADDRESS)),yes,no)
GITHUB_AUTH_CI := other
ifeq ($(strip $(CI)),Y)
GITHUB_AUTH_CI := ci
else ifeq ($(strip $(CI)),N)
GITHUB_AUTH_CI := local
else ifeq ($(strip $(CI)),)
GITHUB_AUTH_CI := unset
endif
GITHUB_TOKEN := $(firstword $(GITHUB_TOKEN) $(GH_TOKEN) $(MISE_GITHUB_TOKEN))
ifeq ($(GITHUB_TOKEN),)
ifneq ($(filter local unset,$(GITHUB_AUTH_CI)),)
override GITHUB_AUTH_COMMAND := $(shell command -v gh 2>/dev/null)
override GITHUB_AUTH_COMMAND_STATUS := $(.SHELLSTATUS)
ifeq ($(GITHUB_AUTH_COMMAND_STATUS),0)
GITHUB_AUTH_SOURCE := gh
GITHUB_TOKEN := $(shell "$(GITHUB_AUTH_COMMAND)" auth token 2>/dev/null)
GITHUB_AUTH_STATUS := $(.SHELLSTATUS)
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
GITHUB_AUTH_PRESENT := $(if $(strip $(GITHUB_TOKEN)),yes,no)
GITHUB_AUTH_PHYSICAL_COMMAND := $(if $(filter gh,$(GITHUB_AUTH_SOURCE)),$(realpath $(GITHUB_AUTH_COMMAND)),not-selected)
# Diagnostics expose only producer metadata, never credentials or command stderr.
GITHUB_AUTH_DIAGNOSTICS = printf 'github-auth source=%s extraction-exit=%s present=%s ci=%s command-exit=%s command=%s physical-command=%s host-source=%s gh-config-override=%s xdg-config-override=%s session-bus=%s\n' '$(GITHUB_AUTH_SOURCE)' '$(GITHUB_AUTH_STATUS)' '$(GITHUB_AUTH_PRESENT)' '$(GITHUB_AUTH_CI)' '$(GITHUB_AUTH_COMMAND_STATUS)' '$(GITHUB_AUTH_COMMAND)' '$(GITHUB_AUTH_PHYSICAL_COMMAND)' '$(GITHUB_AUTH_HOST_SOURCE)' '$(GITHUB_AUTH_CONFIG_OVERRIDE)' '$(GITHUB_AUTH_XDG_OVERRIDE)' '$(GITHUB_AUTH_SESSION_BUS)'
# A global url.<https-url>.insteadOf rewrite that embeds credentials (a
# userinfo with a separator colon) shadows the fleet SSH route and poisons
# every resolver that shells out to git (uv, mise, cargo): their fetches
# authenticate with the baked-in value and fail in confusing ways far from
# the cause. The rewrite itself is operator territory; this gate only
# refuses to run under it and names the configuration owner. The SSH
# form (url.git@host:.insteadOf) is the fleet's own route and stays legal.
GITHUB_AUTH_REWRITE_PRESENT := $(shell if git config --global --get-regexp '^url\.https?://[^ /]+:[^ /]+@.*\.insteadof$$' >/dev/null; then printf yes; else status=$$?; if [ "$$status" -eq 1 ]; then printf no; else exit "$$status"; fi; fi)
ifneq ($(.SHELLSTATUS),0)
$(error Cannot inspect the global Git URL rewrite configuration)
endif
ifeq ($(GITHUB_AUTH_REWRITE_PRESENT),yes)
ifneq ($(MAKECMDGOALS),help)
$(error Global Git URL rewrite configuration embeds credentials and shadows the fleet SSH route; correct that configuration at its owner)
endif
endif

# === SECTION: project identity (managed) ===
# Source: config:dist / config:make_profile / config:repository_root_rel / config:uv_link_mode
PROJECT_NAME := flext-core
MAKE_PROFILE := standalone
REPOSITORY_ROOT_REL := .
# === SECTION: workspace subprojects (managed) ===
# Source: config:workspace_subprojects (list), config:workspace_repositories (list)
# Computed: traversal projects observed members; provisioning reads local .gitmodules.
WORKSPACE_SUBPROJECTS :=
UV_LINK_MODE := copy
# End SECTION: project identity

# === SECTION: public boundary (managed) ===
# The Makefile validates no caller input. Every verb
# always applies; a mistyped variable fails where the verb consumes it, and an
# unconsumed variable is ignored.
PYTEST_DIAG_ARGS := -rA --durations=0 --tb=long --showlocals
PYTEST_REPORT_ARGS := -ra --durations=25 --durations-min=0.001 --tb=short
PYTEST_PROCESS_TIMEOUT_SECONDS := 124
# The pytest process inherits a hard wall-clock boundary, so a hung
# run is terminated even if the runner itself stalls.
override PYTEST_BOUNDED = timeout --signal=TERM --kill-after=5s "$(PYTEST_PROCESS_TIMEOUT_SECONDS)s"
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
# Profiles sit beside the other reports of this checkout (.reports is ignored).
PROFILE_REPORTS_DIR = $(PROJECT_ROOT)/$(shell dirname -- "$(PYTEST_REPORTS_DIR)")/profiles
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
SETUP_BIN := $(PROJECT_ROOT)/.bin
# Caller-overridable test target (a directory or a single test file under
# it): `make test FLEXT_PYTEST_TARGET_RAW=tests/unit/foo_test.py` runs one
# file through the same bounded runner. The default stays the declared
# target_directory; the runner validates the path's shape.
export FLEXT_PYTEST_TARGET_RAW := tests
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
if [ "$${GITHUB_ACTIONS:-}" = "true" ]; then printf '%s\n' "$${project_root}"; exit 0; fi; \
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
reader_root=; \
parent="$$(dirname "$${project_root}")"; \
while [ "$${parent}" != / ]; do \
	if [ -f "$${parent}/pyproject.toml" ]; then reader_root="$${parent}"; break; fi; \
	if [ "$${parent}" = "$${git_root}" ]; then break; fi; \
	parent="$$(dirname "$${parent}")"; \
done; \
if [ -z "$${reader_root}" ]; then printf '%s\n' "$${project_root}"; exit 0; fi; \
git_parent="$${reader_root}"; \
case "$${git_parent}" in (/*) ;; (*) printf 'ERROR: Git marker discovery requires an absolute physical directory: %s\n' "$${git_parent}" >&2; exit 2 ;; esac; \
while [ "$${git_parent}" != / ] && [ ! -e "$${git_parent}/.git" ] && [ ! -L "$${git_parent}/.git" ]; do \
	next_git_parent="$$(dirname "$${git_parent}")"; \
	if [ "$${next_git_parent}" = "$${git_parent}" ]; then printf 'ERROR: Git marker discovery cannot advance from: %s\n' "$${git_parent}" >&2; exit 2; fi; \
	git_parent="$${next_git_parent}"; \
done; \
if [ -e "$${git_parent}/.git" ] || [ -L "$${git_parent}/.git" ]; then git -C "$${reader_root}" rev-parse --resolve-git-dir "$${git_parent}/.git" >/dev/null; fi; \
if [ ! -e "$${git_parent}/.git" ] && [ ! -L "$${git_parent}/.git" ]; then printf '%s\n' "$${project_root}"; exit 0; fi; \
scope="$$(git -C "$${reader_root}" rev-parse --show-toplevel)"; \
scope="$$(cd "$${scope}" && pwd -P)"; \
case "$${project_root}/" in ("$${scope}/"*) ;; (*) printf 'ERROR: native workspace discovery scope does not contain this project: %s\n' "$${scope}" >&2; exit 2 ;; esac; \
if [ -f "$${scope}/pyproject.toml" ]; then reader_root="$${scope}"; fi; \
unset MISE_SYSTEM_CONFIG_FILE MISE_ENV MISE_ENV_FILE UV_PROJECT UV_PROJECT_ENVIRONMENT UV_WORKING_DIRECTORY UV_NO_CONFIG; \
MISE_CONFIG_DIR="$${reader_root}/.mise"; \
MISE_SYSTEM_CONFIG_DIR="$${MISE_CONFIG_DIR}"; \
MISE_GLOBAL_CONFIG_FILE="$${MISE_CONFIG_DIR}/config.toml"; \
MISE_CEILING_PATHS="$$(dirname "$${reader_root}")"; \
export MISE_CONFIG_DIR MISE_SYSTEM_CONFIG_DIR MISE_GLOBAL_CONFIG_FILE MISE_CEILING_PATHS; \
reader_mise=mise; \
case "$(strip $(MAKECMDGOALS))" in (setup|upg) \
	mise_stage=; \
	trap 'bootstrap_status=$$?; trap - EXIT; if [ -n "$${mise_stage}" ]; then \
			if rm -rf "$${mise_stage}"; then :; else \
				mise_cleanup_status=$$?; \
				printf "ERROR: bootstrap staging cleanup failed (exit %s); original exit %s\n" "$${mise_cleanup_status}" "$${bootstrap_status}" >&2; \
				if [ "$${bootstrap_status}" -eq 0 ]; then bootstrap_status=$${mise_cleanup_status}; fi; \
			fi; \
		fi; exit "$${bootstrap_status}"' EXIT; \
	{ \
		bootstrap_stage=network-auth; \
	$(GITHUB_AUTH_DIAGNOSTICS); \
	if [ "$(GITHUB_AUTH_SOURCE)" = gh ]; then \
		if [ "$(GITHUB_AUTH_STATUS)" != 0 ]; then \
			printf 'ERROR: selected gh auth token failed (exit %s); refusing anonymous provisioning\n' '$(GITHUB_AUTH_STATUS)' >&2; \
			exit "$(GITHUB_AUTH_STATUS)"; \
		fi; \
		if [ "$(GITHUB_AUTH_PRESENT)" != yes ]; then \
			printf 'ERROR: selected gh auth token returned no credential; refusing anonymous provisioning\n' >&2; \
			exit 2; \
		fi; \
	fi; \
		mise_bootstrap_root="$${XDG_CACHE_HOME:-$${HOME}/.cache}/flext/infra/mise-bootstrap"; \
	mise_probe() { \
		if mise_version_output="$$("$$1" --version)"; then :; else return $$?; fi; \
		mise_receipt="$${mise_version_output%% *}"; \
	}; \
	mise_qualify() { \
	mise_pin=; \
	if [ -f "$${reader_root}/mise.lock" ]; then \
		mise_pin="$$( awk 'index($$0, "[[tools.\"github:jdx/mise\"]]") == 1 { inside = 1; next } inside && substr($$0, 1, 1) == "[" { exit } inside && $$1 == "version" { gsub(/[",]/, "", $$3); print $$3; exit }' "$${reader_root}/mise.lock" )"; \
	fi; \
	if [ -z "$${mise_pin}" ]; then \
		bootstrap_stage=lock-pin; \
		printf 'ERROR: mise.lock pins no github:jdx/mise release\n' >&2; \
		exit 2; \
	fi; \
	bootstrap_stage=platform; \
	case "$$(uname -s)/$$(uname -m)" in \
		(Linux/x86_64) mise_platform=linux-x64 ;; \
		(Linux/aarch64) mise_platform=linux-arm64 ;; \
		(Darwin/x86_64) mise_platform=macos-x64 ;; \
		(Darwin/arm64) mise_platform=macos-arm64 ;; \
		(Windows/AMD64) mise_platform=windows-x64 ;; \
		(*) printf 'ERROR: no github:jdx/mise lock platform maps to %s/%s; install mise %s (https://mise.run) and retry\n' "$$(uname -s)" "$$(uname -m)" "$${mise_pin}" >&2; exit 2 ;; \
	esac; \
	bootstrap_stage=release-asset; \
	mise_key=url; \
	mise_url="$$( awk -v section="[tools.\"github:jdx/mise\".\"platforms.$${mise_platform}\"]" -v key="$${mise_key}" \
	'$$0 == section { inside = 1; next } inside && substr($$0, 1, 1) == "[" { exit } inside && $$1 == key { gsub(/"/, "", $$3); print $$3; exit }' "$${reader_root}/mise.lock" )"; \
	case "$${mise_url}" in \
		(https://github.com/*/releases/download/*) ;; \
		(*) printf 'ERROR: mise.lock has no release URL for github:jdx/mise on %s\n' "$${mise_platform}" >&2; exit 2 ;; \
	esac; \
	mise_key=checksum; \
	mise_checksum="$$( awk -v section="[tools.\"github:jdx/mise\".\"platforms.$${mise_platform}\"]" -v key="$${mise_key}" \
	'$$0 == section { inside = 1; next } inside && substr($$0, 1, 1) == "[" { exit } inside && $$1 == key { gsub(/"/, "", $$3); print $$3; exit }' "$${reader_root}/mise.lock" )"; \
	case "$${mise_checksum}" in \
		(sha256:*) mise_sha256="$${mise_checksum#sha256:}" ;; \
		(*) printf 'ERROR: mise.lock has no sha256 for github:jdx/mise on %s\n' "$${mise_platform}" >&2; exit 2 ;; \
	esac; \
	mise_bootstrap_bin="$${mise_bootstrap_root}/$${mise_pin}/mise"; \
	mise_recover=true; \
	if [ -x "$${mise_bootstrap_bin}" ]; then \
		bootstrap_stage=release-verification; \
		mise_probe "$${mise_bootstrap_bin}"; \
		if [ "$${mise_receipt}" = "$${mise_pin}" ]; then mise_recover=false; fi; \
	fi; \
	if [ "$${mise_recover}" = true ]; then \
		printf 'setup: recovering github:jdx/mise %s from the mise.lock release asset for %s\n' "$${mise_pin}" "$${mise_platform}"; \
		mkdir -p "$$(dirname "$${mise_bootstrap_bin}")"; \
		mise_stage="$$(mktemp -d "$$(dirname "$${mise_bootstrap_bin}")/.stage.XXXXXXXX")"; \
		curl --proto '=https' --tlsv1.2 -fsSL -o "$${mise_stage}/archive" "$${mise_url}"; \
		if command -v sha256sum >/dev/null 2>&1; then \
			echo "$${mise_sha256}  $${mise_stage}/archive" | sha256sum -c -; \
		else \
			echo "$${mise_sha256}  $${mise_stage}/archive" | shasum -a 256 -c -; \
		fi; \
		tar -xf "$${mise_stage}/archive" -C "$${mise_stage}"; \
		bootstrap_stage=release-verification; \
		chmod +x "$${mise_stage}/mise/bin/mise"; \
		mise_probe "$${mise_stage}/mise/bin/mise"; \
		if [ "$${mise_receipt}" != "$${mise_pin}" ]; then \
			printf 'ERROR: recovered Mise %s differs from the mise.lock pin %s\n' "$${mise_receipt}" "$${mise_pin}" >&2; \
			exit 2; \
		fi; \
		mv "$${mise_stage}/mise/bin/mise" "$${mise_bootstrap_bin}"; \
		rm -rf "$${mise_stage}"; \
		mise_stage=; \
	fi; \
	export MISE_SHIMS_DIR="$${mise_bootstrap_root}/$${mise_receipt}/shims"; \
	}; \
		mise_qualify; \
	} >&2; \
	reader_mise="$${mise_bootstrap_bin}"; \
;; esac; \
uv_requested="$$("$${reader_mise}" -C "$${reader_root}" tool "uv" --requested)"; \
case "$(strip $(MAKECMDGOALS))" in (setup|upg) \
	uv_active="$$("$${reader_mise}" -C "$${reader_root}" tool "uv" --active)"; \
	uv_installed="$$("$${reader_mise}" -C "$${reader_root}" tool "uv" --installed)"; \
	uv_present=; \
	for uv_version in $${uv_installed}; do \
		if [ "$${uv_version}" = "$${uv_active}" ]; then uv_present=Y; fi; \
	done; \
	if [ -z "$${uv_present}" ]; then \
		"$${reader_mise}" -C "$${reader_root}" install --locked --yes "uv" >&2; \
	fi; \
;; esac; \
uv_binary="$$("$${reader_mise}" -C "$${reader_root}" which "uv" --tool "uv@$${uv_requested}")"; \
if [ ! -x "$${uv_binary}" ]; then printf 'ERROR: native workspace reader is not provisioned; run make setup\n' >&2; exit 2; fi; \
owner="$$("$${uv_binary}" workspace dir --project "$${project_root}")"; \
case "$${owner}" in (/*) ;; (*) printf 'ERROR: native UV workspace owner is not absolute: %s\n' "$${owner}" >&2; exit 2 ;; esac; \
owner="$$(cd "$${owner}" && pwd -P)"; \
case "$${owner}/" in ("$${scope}/"*) ;; (*) printf 'ERROR: native UV workspace owner leaves its Git scope: %s\n' "$${owner}" >&2; exit 2 ;; esac; \
case "$${project_root}/" in ("$${owner}/"*) ;; (*) printf 'ERROR: native UV workspace owner does not contain this project: %s\n' "$${owner}" >&2; exit 2 ;; esac; \
printf '%s\n' "$${owner}")
ifneq ($(.SHELLSTATUS),0)
$(error Cannot resolve the physical workspace for $(MAKEFILE_ROOT))
endif
# End SECTION: REPOSITORY_ROOT isolation
# === SECTION: verb dispatch (managed) ===
# Source: config:make.verbs and the canonical gate vocabulary. A verb exists
# only in the profiles it declares (make.verbs[].profiles).
PUBLIC_VERBS := help setup pre-commit upg upg-sources validate-lifecycle build examples check check-pyright census census-constants smells test tests test-full test-file file-gate profile-test profile-test-report profile-docs profile-docs-report profile-environment profile-environment-report profile-lanes profile-lanes-report profile-mod profile-mod-report profile-mod-rules profile-mod-rules-report profile-codemod profile-codemod-report fmt fix fix-namespace fix-accessors fix-toolchain audit status verify-clean docs clean bootstrap-candidate release-plan release-version release-tag release-build publication gen gen-footprint gen-pyproject gen-editor initialize project-new mod mod-types mod-rules mod-text mod-text-candidate mod-snapshots waza duplication sonarcloud-sync sonarcloud-issues
BUILTIN_VERBS := help setup pre-commit upg upg-sources validate-lifecycle build examples check check-pyright census census-constants smells test tests test-full test-file file-gate profile-test profile-test-report profile-docs profile-docs-report profile-environment profile-environment-report profile-lanes profile-lanes-report profile-mod profile-mod-report profile-mod-rules profile-mod-rules-report profile-codemod profile-codemod-report fmt fix fix-namespace fix-accessors fix-toolchain audit status verify-clean docs clean bootstrap-candidate release-plan release-version release-tag release-build publication gen gen-footprint gen-pyproject gen-editor initialize project-new mod mod-types mod-rules mod-text mod-text-candidate mod-snapshots waza duplication sonarcloud-sync sonarcloud-issues
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
ifneq ($(filter pre-commit,$(CUSTOM_DECLARED_TARGETS)),)
$(error Mandatory approval cannot be replaced by custom targets)
endif
ifeq ($(APPROVAL_CONTEXT),Y)
ifneq ($(filter setup gen audit check test verify-clean,$(CUSTOM_DECLARED_TARGETS)),)
$(error Approval stages cannot be replaced by custom targets)
endif
# Wrapper parity: a custom approval-stage hook is legitimate only while it
# chains the canonical builtin inside its recipe (the host-service harness
# pattern). A declared hook without the builtin reference is a replacement
# and stays forbidden.
ifneq ($(filter _custom-pre-commit,$(CUSTOM_DECLARED_TARGETS)),)
override CUSTOM_WRAPPER_REFERENCE_COUNT := $(shell awk -v reference="_builtin-pre-commit" 'index($$0, reference) { count++ } END { print count + 0 }' "$(CUSTOM_MAKEFILE)")
ifneq ($(.SHELLSTATUS),0)
$(error Failed to inspect custom approval reference in $(CUSTOM_MAKEFILE))
endif
ifeq ($(CUSTOM_WRAPPER_REFERENCE_COUNT),0)
$(error Approval stage _custom-pre-commit must chain _builtin-pre-commit (wrapper parity; replacements are forbidden))
endif
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

ifneq ($(filter _custom-audit,$(CUSTOM_DECLARED_TARGETS)),)
override CUSTOM_WRAPPER_REFERENCE_COUNT := $(shell awk -v reference="_builtin-audit" 'index($$0, reference) { count++ } END { print count + 0 }' "$(CUSTOM_MAKEFILE)")
ifneq ($(.SHELLSTATUS),0)
$(error Failed to inspect custom approval reference in $(CUSTOM_MAKEFILE))
endif
ifeq ($(CUSTOM_WRAPPER_REFERENCE_COUNT),0)
$(error Approval stage _custom-audit must chain _builtin-audit (wrapper parity; replacements are forbidden))
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

ifneq ($(filter _custom-test,$(CUSTOM_DECLARED_TARGETS)),)
override CUSTOM_WRAPPER_REFERENCE_COUNT := $(shell awk -v reference="_builtin-test" 'index($$0, reference) { count++ } END { print count + 0 }' "$(CUSTOM_MAKEFILE)")
ifneq ($(.SHELLSTATUS),0)
$(error Failed to inspect custom approval reference in $(CUSTOM_MAKEFILE))
endif
ifeq ($(CUSTOM_WRAPPER_REFERENCE_COUNT),0)
$(error Approval stage _custom-test must chain _builtin-test (wrapper parity; replacements are forbidden))
endif
endif

ifneq ($(filter _custom-verify-clean,$(CUSTOM_DECLARED_TARGETS)),)
override CUSTOM_WRAPPER_REFERENCE_COUNT := $(shell awk -v reference="_builtin-verify-clean" 'index($$0, reference) { count++ } END { print count + 0 }' "$(CUSTOM_MAKEFILE)")
ifneq ($(.SHELLSTATUS),0)
$(error Failed to inspect custom approval reference in $(CUSTOM_MAKEFILE))
endif
ifeq ($(CUSTOM_WRAPPER_REFERENCE_COUNT),0)
$(error Approval stage _custom-verify-clean must chain _builtin-verify-clean (wrapper parity; replacements are forbidden))
endif
endif

endif
endif
DOCS_ACTIONS := generate fix fmt build validate audit
 # End SECTION: verb dispatch

# === SECTION: lint/type paths (managed) ===
# Source: template + computed (script_dispatch conditional)
RUFF_PATHS := $(shell for directory in src tests examples scripts; do if [ -e "$(PROJECT_ROOT)/$$directory" ]; then printf '"%s" ' "$(PROJECT_ROOT)/$$directory"; fi; done)
MYPY_PATHS := $(RUFF_PATHS)
# End SECTION: lint/type paths

# === SECTION: project tool owner (managed) ===
# Source: the repository's declared Mise toolchain, provisioned by make setup.
# Every tool resolves from the mise-managed PATH (direnv activation shims, or
# the CI provisioned PATH); _builtin_require_mise guards the running mise
# against the committed mise.lock pin.
override UV = "$$(mise -C "$(RUNTIME_ROOT)" which "uv" --tool "uv@$$(mise -C "$(RUNTIME_ROOT)" tool "uv" --requested)")"
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
unexport MISE_SYSTEM_CONFIG_FILE
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
override UV_LOCAL_PROJECT := $(RUNTIME_ROOT)/.flext/uv/local
override UV_STANDALONE_PROJECT := $(RUNTIME_ROOT)/.flext/uv/standalone
override EDITABLE_SUPPLIER_ACTIVE :=
ifneq ($(strip $(FLEXT_INFRA_EDITABLE_SUPPLIER_ROOT)),)
ifneq ($(realpath $(FLEXT_INFRA_EDITABLE_SUPPLIER_ROOT)),$(RUNTIME_ROOT))
ifneq ($(strip $(GITHUB_ACTIONS)),true)
override EDITABLE_SUPPLIER_ACTIVE := Y
endif
endif
endif
override UPG_RESOLUTION_PROJECTS := "$(PROJECT_ROOT)/.flext/uv/local" "$(PROJECT_ROOT)/.flext/uv/standalone"
ifneq ($(EDITABLE_SUPPLIER_ACTIVE),)
override UPG_RESOLUTION_PROJECTS := "$(RUNTIME_ROOT)/.flext/uv/editable"
endif
ifeq ($(strip $(GITHUB_ACTIONS)),true)
override UV_PROJECT := $(UV_STANDALONE_PROJECT)
else ifneq ($(EDITABLE_SUPPLIER_ACTIVE),)
override UV_PROJECT := $(RUNTIME_ROOT)/.flext/uv/editable
else
override UV_PROJECT := $(UV_LOCAL_PROJECT)
endif
override UV_PROJECT_ENVIRONMENT := $(RUNTIME_VENV)
override VIRTUAL_ENV := $(RUNTIME_VENV)
override PATH := $(RUNTIME_BIN):$(SANITIZED_CALLER_PATH)
unexport UV
export FLEXT_INFRA_PYTHON UV_PROJECT UV_PROJECT_ENVIRONMENT VIRTUAL_ENV PATH RUNTIME_ROOT

# One bootstrap serves `setup` (frozen install) and `upg` (resolve + install);
# the public verb selects its lifecycle and resolution through target-specific
# variables.
TOOL_BOOTSTRAP_LIFECYCLE := _setup_lifecycle
TOOL_BOOTSTRAP_RESOLVE :=
.PHONY: _bootstrap_setup_tools

_bootstrap_setup_tools:
	# The lifecycle invokes recursive make through mise, so preserve jobserver FDs.
	+@set -eu; \
	bootstrap_stage=host-mise; bootstrap_completed=false; bootstrap_sha=null; bootstrap_git_status=null; mise_stage=; \
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
	$(GITHUB_AUTH_DIAGNOSTICS); \
	if [ "$(GITHUB_AUTH_SOURCE)" = gh ]; then \
		if [ "$(GITHUB_AUTH_STATUS)" != 0 ]; then \
			printf 'ERROR: selected gh auth token failed (exit %s); refusing anonymous provisioning\n' '$(GITHUB_AUTH_STATUS)' >&2; \
			exit "$(GITHUB_AUTH_STATUS)"; \
		fi; \
		if [ "$(GITHUB_AUTH_PRESENT)" != yes ]; then \
			printf 'ERROR: selected gh auth token returned no credential; refusing anonymous provisioning\n' >&2; \
			exit 2; \
		fi; \
	fi; \
	bootstrap_stage=lock-pin; \
	mise_pin=; mise_resolved=0; \
	bootstrap_stage=lock-pin; \
	if [ -f "$(PROJECT_ROOT)/mise.lock" ]; then \
		mise_pin="$$( awk 'index($$0, "[[tools.\"github:jdx/mise\"]]") == 1 { inside = 1; next } inside && substr($$0, 1, 1) == "[" { exit } inside && $$1 == "version" { gsub(/[",]/, "", $$3); print $$3; exit }' "$(PROJECT_ROOT)/mise.lock" )"; \
	fi; \
	if [ -z "$${mise_pin}" ] && [ "$(TOOL_BOOTSTRAP_RESOLVE)" = "1" ]; then \
		bootstrap_stage=host-mise; \
		if ! command -v mise >/dev/null 2>&1; then \
			printf 'ERROR: resolving a missing Mise self-pin requires an available native Mise\n' >&2; \
			exit 2; \
		fi; \
		bootstrap_stage=lock-pin; \
		mise -C "$(PROJECT_ROOT)" lock --upgrade --bump; \
		mise_resolved=1; \
		mise_pin="$$( awk 'index($$0, "[[tools.\"github:jdx/mise\"]]") == 1 { inside = 1; next } inside && substr($$0, 1, 1) == "[" { exit } inside && $$1 == "version" { gsub(/[",]/, "", $$3); print $$3; exit }' "$(PROJECT_ROOT)/mise.lock" )"; \
	fi; \
	mise_bootstrap_root="$${XDG_CACHE_HOME:-$${HOME}/.cache}/flext/infra/mise-bootstrap"; \
	mise_probe() { \
		if mise_version_output="$$("$$1" --version)"; then :; else return $$?; fi; \
		mise_receipt="$${mise_version_output%% *}"; \
	}; \
	mise_qualify() { \
	mise_pin=; \
	if [ -f "$(PROJECT_ROOT)/mise.lock" ]; then \
		mise_pin="$$( awk 'index($$0, "[[tools.\"github:jdx/mise\"]]") == 1 { inside = 1; next } inside && substr($$0, 1, 1) == "[" { exit } inside && $$1 == "version" { gsub(/[",]/, "", $$3); print $$3; exit }' "$(PROJECT_ROOT)/mise.lock" )"; \
	fi; \
	if [ -z "$${mise_pin}" ]; then \
		bootstrap_stage=lock-pin; \
		printf 'ERROR: mise.lock pins no github:jdx/mise release\n' >&2; \
		exit 2; \
	fi; \
	bootstrap_stage=platform; \
	case "$$(uname -s)/$$(uname -m)" in \
		(Linux/x86_64) mise_platform=linux-x64 ;; \
		(Linux/aarch64) mise_platform=linux-arm64 ;; \
		(Darwin/x86_64) mise_platform=macos-x64 ;; \
		(Darwin/arm64) mise_platform=macos-arm64 ;; \
		(Windows/AMD64) mise_platform=windows-x64 ;; \
		(*) printf 'ERROR: no github:jdx/mise lock platform maps to %s/%s; install mise %s (https://mise.run) and retry\n' "$$(uname -s)" "$$(uname -m)" "$${mise_pin}" >&2; exit 2 ;; \
	esac; \
	bootstrap_stage=release-asset; \
	mise_key=url; \
	mise_url="$$( awk -v section="[tools.\"github:jdx/mise\".\"platforms.$${mise_platform}\"]" -v key="$${mise_key}" \
	'$$0 == section { inside = 1; next } inside && substr($$0, 1, 1) == "[" { exit } inside && $$1 == key { gsub(/"/, "", $$3); print $$3; exit }' "$(PROJECT_ROOT)/mise.lock" )"; \
	case "$${mise_url}" in \
		(https://github.com/*/releases/download/*) ;; \
		(*) printf 'ERROR: mise.lock has no release URL for github:jdx/mise on %s\n' "$${mise_platform}" >&2; exit 2 ;; \
	esac; \
	mise_key=checksum; \
	mise_checksum="$$( awk -v section="[tools.\"github:jdx/mise\".\"platforms.$${mise_platform}\"]" -v key="$${mise_key}" \
	'$$0 == section { inside = 1; next } inside && substr($$0, 1, 1) == "[" { exit } inside && $$1 == key { gsub(/"/, "", $$3); print $$3; exit }' "$(PROJECT_ROOT)/mise.lock" )"; \
	case "$${mise_checksum}" in \
		(sha256:*) mise_sha256="$${mise_checksum#sha256:}" ;; \
		(*) printf 'ERROR: mise.lock has no sha256 for github:jdx/mise on %s\n' "$${mise_platform}" >&2; exit 2 ;; \
	esac; \
	mise_bootstrap_bin="$${mise_bootstrap_root}/$${mise_pin}/mise"; \
	mise_recover=true; \
	if [ -x "$${mise_bootstrap_bin}" ]; then \
		bootstrap_stage=release-verification; \
		mise_probe "$${mise_bootstrap_bin}"; \
		if [ "$${mise_receipt}" = "$${mise_pin}" ]; then mise_recover=false; fi; \
	fi; \
	if [ "$${mise_recover}" = true ]; then \
		printf 'setup: recovering github:jdx/mise %s from the mise.lock release asset for %s\n' "$${mise_pin}" "$${mise_platform}"; \
		mkdir -p "$$(dirname "$${mise_bootstrap_bin}")"; \
		mise_stage="$$(mktemp -d "$$(dirname "$${mise_bootstrap_bin}")/.stage.XXXXXXXX")"; \
		curl --proto '=https' --tlsv1.2 -fsSL -o "$${mise_stage}/archive" "$${mise_url}"; \
		if command -v sha256sum >/dev/null 2>&1; then \
			echo "$${mise_sha256}  $${mise_stage}/archive" | sha256sum -c -; \
		else \
			echo "$${mise_sha256}  $${mise_stage}/archive" | shasum -a 256 -c -; \
		fi; \
		tar -xf "$${mise_stage}/archive" -C "$${mise_stage}"; \
		bootstrap_stage=release-verification; \
		chmod +x "$${mise_stage}/mise/bin/mise"; \
		mise_probe "$${mise_stage}/mise/bin/mise"; \
		if [ "$${mise_receipt}" != "$${mise_pin}" ]; then \
			printf 'ERROR: recovered Mise %s differs from the mise.lock pin %s\n' "$${mise_receipt}" "$${mise_pin}" >&2; \
			exit 2; \
		fi; \
		mv "$${mise_stage}/mise/bin/mise" "$${mise_bootstrap_bin}"; \
		rm -rf "$${mise_stage}"; \
		mise_stage=; \
	fi; \
	export MISE_SHIMS_DIR="$${mise_bootstrap_root}/$${mise_receipt}/shims"; \
	}; \
	mise_qualify; \
	bootstrap_stage=tool-lock; \
	if [ "$(TOOL_BOOTSTRAP_RESOLVE)" = "1" ] && [ "$${mise_resolved}" = "0" ]; then \
		"$${mise_bootstrap_bin}" -C "$(PROJECT_ROOT)" lock --upgrade --bump; \
		mise_qualify; \
	fi; \
	bootstrap_stage=tool-install; \
	"$${mise_bootstrap_bin}" -C "$(PROJECT_ROOT)" install --locked --yes "python" "github:jdx/mise" "uv" "github:direnv/direnv" "taplo" "aqua:ast-grep/ast-grep" "gitleaks" "aqua:boyter/scc" "node" "github:qltysh/qlty" "github:kucherenko/jscpd" "github:microsoft/waza"; \
	bootstrap_stage=shims; \
	"$${mise_bootstrap_bin}" reshim; \
	if [ -n "$${GITHUB_PATH:-}" ]; then printf '%s\n' "$${MISE_SHIMS_DIR}" >> "$${GITHUB_PATH}"; fi; \
	printf 'setup: mise %s provisioning complete\n' "$${mise_receipt}"; \
	printf 'setup: entering lifecycle (submodules, environment, hooks) make=%s\n' "$(SELF_MAKE_EXECUTABLE)"; \
	bootstrap_stage=lifecycle; \
	"$${mise_bootstrap_bin}" -C "$(PROJECT_ROOT)" exec -- env "PATH=$$(dirname "$${mise_bootstrap_bin}"):$${MISE_SHIMS_DIR}:$${PATH}" "CI=$(CI)" $(SELF_MAKE) $(TOOL_BOOTSTRAP_LIFECYCLE); \
	bootstrap_stage=complete; bootstrap_completed=true
_bootstrap_setup_tools: _builtin_require_workspace
ifneq ($(EDITABLE_SUPPLIER_ACTIVE),)
_bootstrap_setup_tools: _builtin_require_editable_supplier
endif

# UV, not Git attachment or runtime environment selection, owns lock discovery.
.PHONY: _builtin_require_upg_lock_owner
_builtin_require_upg_lock_owner:
	@set -eu; \
	for resolution in $(UPG_RESOLUTION_PROJECTS); do \
		owner="$$($(UV) workspace dir --project "$$resolution")"; \
		if [ "$${owner}" != "$$resolution" ]; then \
			printf 'ERROR[upg] UV lock owner is %s, not %s; regenerate the native resolution inputs before upgrading.\n' "$${owner}" "$$resolution" >&2; \
			exit 2; \
		fi; \
	done

# Every repository evaluates only itself: a workspace root consumes its members
# as installed libraries and each member runs its own lifecycle in its own
# repository. Locally, the mutating verbs fmt/fix/mod then carry that same
# member-own verb to every governed member (fleet_fanout); CI never fans out.
# Provisioning is declared once and shared by every profile. Dev environments
# consume present members as LIVE editable installs natively (operator law
# 2026-10-06: a commit never influences dev behavior — the worktree is the
# behavior): the root dev group requests every declared package, member group
# and extra. The topology-derived tool.uv.sources selects each member's live
# editable path in the committed lock; CI's --no-editable keeps frozen builds.
# Attached projects inherit this root's lock and never install over its graph.
EDITABLE_SUPPLIER_PREFLIGHT = set -eu; \
	supplier="$(FLEXT_INFRA_EDITABLE_SUPPLIER_ROOT)"; \
	case "$$supplier" in (/*) ;; (*) printf 'ERROR: editable supplier root must be an absolute path\n' >&2; exit 2 ;; esac; \
	if [ ! -d "$$supplier" ]; then printf 'ERROR: editable supplier root is not a directory: %s\n' "$$supplier" >&2; exit 2; fi; \
	supplier_resolution="$$supplier/.flext/uv/local"; \
	if [ ! -f "$$supplier_resolution/uv.lock" ]; then printf 'ERROR: selected supplier has no native bootstrap lock: %s\n' "$$supplier_resolution" >&2; exit 2; fi
.PHONY: _builtin_require_editable_supplier
_builtin_require_editable_supplier:
	@$(EDITABLE_SUPPLIER_PREFLIGHT)
EDITABLE_RESOLUTION_RECIPE = $(EDITABLE_SUPPLIER_PREFLIGHT); \
	desired_python="$$(mise -C "$(RUNTIME_ROOT)" which python)"; \
	bootstrap_infra() { env -u PYTHONPATH -u MYPYPATH -u VIRTUAL_ENV -u UV_PROJECT -u UV_PROJECT_ENVIRONMENT $(UV) run --isolated --frozen --all-groups --project "$$supplier_resolution" --python "$$desired_python" -m flext_infra "$$@"; }; \
	bootstrap_infra codegen generate --root "$(PROJECT_ROOT)" --what pyproject --scope self
INSTALL_RUNTIME_RECIPE = set -eu; \
	trap 'if [ -n "$${FLEXT_SETUP_CREDENTIAL_STORE:-}" ]; then rm -f "$$FLEXT_SETUP_CREDENTIAL_STORE"; fi' EXIT; \
	$(REQUIRE_WORKSPACE_ENVIRONMENT); \
	desired_python="$$(mise -C "$(RUNTIME_ROOT)" which python)"; \
	if [ ! -x "$$desired_python" ]; then \
		printf 'ERROR[setup] Mise-resolved Python is not executable: %s\n' "$$desired_python" >&2; \
		exit 2; \
	fi; \
	credential_env=; \
	if [ -n "$${FLEXT_SETUP_CREDENTIAL_STORE:-}" ]; then \
		credential_env="env GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=credential.helper GIT_CONFIG_VALUE_0=store --file=$$FLEXT_SETUP_CREDENTIAL_STORE"; \
	fi; \
	if [ ! -f "$(UV_PROJECT)/uv.lock" ]; then \
		printf 'ERROR[setup] uv.lock is missing: %s/uv.lock\n  Right way: only `make upg` writes uv.lock; setup installs the committed lock and never creates one.\n  How: run `make upg` in %s, then commit uv.lock.\n' "$(UV_PROJECT)" "$(PROJECT_ROOT)" >&2; \
		exit 2; \
	fi; \
	mise -C "$(RUNTIME_ROOT)" config ls --json; \
	if [ -n "$(EDITABLE_SUPPLIER_ACTIVE)" ]; then \
		$$credential_env $(UV) lock --check --project "$(UV_PROJECT)" --python "$$desired_python"; \
	fi; \
	$$credential_env $(UV) sync --project "$(UV_PROJECT)" --python "$$desired_python" $(UV_SYNC_FLAGS) --frozen --link-mode "$(UV_LINK_MODE)"
SETUP_ENVIRONMENT_RECIPE = $(INSTALL_RUNTIME_RECIPE); \
	setup_scope="self"; \
	if [ "$(strip $(GITHUB_ACTIONS))" = "true" ]; then setup_scope=self; fi; \
	$(PROJECT_FLEXT_INFRA) codegen mise-artifacts --repository-root "$(PROJECT_ROOT)" --config-only --scope "$$setup_scope"; \
	$(PROJECT_FLEXT_INFRA) workspace sync-environment --repository-root "$(PROJECT_ROOT)" --scope "$$setup_scope" $(if $(filter Y,$(strip $(CI))),--no-allow-direnv)

# Reject borrowed environments before bootstrap, activation or custom hooks.
# Members may use their containing workspace, never an unrelated checkout.
REQUIRE_WORKSPACE_ENVIRONMENT = case "$(PROJECT_ROOT)/" in \
	"$(RUNTIME_ROOT)/"*) ;; \
	*) printf 'ERROR: runtime workspace does not contain this project: %s\n' "$(RUNTIME_ROOT)" >&2; exit 2 ;; \
	esac; \
	if [ "$(strip $(GITHUB_ACTIONS))" = "true" ] && [ "$(PROJECT_ROOT)" != "$(RUNTIME_ROOT)" ]; then \
		printf 'ERROR: hosted CI requires an independent checkout and environment\n' >&2; exit 2; \
	fi; \
	for environment_path in "$$(dirname "$(RUNTIME_VENV)")" "$(RUNTIME_VENV)" "$(RUNTIME_BIN)" "$(PROJECT_ROOT)/.venv" "$(PROJECT_ROOT)/.venv/bin"; do \
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
override PROJECT_INFRA_RUN = env -u PYTHONPATH -u MYPYPATH -u VIRTUAL_ENV -u UV_PROJECT -u UV_PROJECT_ENVIRONMENT "$(FLEXT_INFRA_PYTHON)"
override PROJECT_FLEXT_INFRA := $(PROJECT_INFRA_RUN) -m flext_infra

# Scaffold dev tools live in the validated optional dev
# Lock law (operator 2026-10-03): only `make upg` writes uv.lock. Setup installs
# the committed lock and never deletes, creates, or relocks it. Frozen sync
# provisions its recorded versions; make audit owns the consistency verdict
# and make upg owns repair. A missing lock remains a native error.
# Native workspace selection installs every declared package.
UV_SYNC_FLAGS := --all-extras --all-groups

ifeq ($(strip $(GITHUB_ACTIONS)),true)
override UV_SYNC_FLAGS := $(UV_SYNC_FLAGS) --no-editable
endif

ifeq ($(GEN_INIT_ONLY),)
# Custom hooks are declared in CI too; load their selected physical Makefile.
-include $(CUSTOM_MAKEFILE_ESCAPED)
endif
ifeq ($(APPROVAL_CONTEXT),Y)
override CI := Y
export CI
override TOOL_BOOTSTRAP_LIFECYCLE := _setup_lifecycle
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
.PHONY: _builtin_gen_init _builtin_gen_all


VERB_DESCRIPTION_help := Show the complete selector-free public interface.

VERB_DESCRIPTION_setup := Install the declared locks, publish local runtime bindings, and provision hooks.

VERB_DESCRIPTION_pre-commit := Approve a commit through the CI check partition and incremental tests with the persistent testmon database.

VERB_DESCRIPTION_upg := Resolve the newest declared releases, write the uv and mise locks, then prove the upgraded tree still converges; gates stay with make check.

VERB_DESCRIPTION_upg-sources := Migrate an explicitly selected dependency Git reference in the root and its declared member manifests before resolving their locks.

VERB_DESCRIPTION_validate-lifecycle := Execute the declared public runtime acceptance steps serially, with root-owned mutation fanout and typed first-failure receipts.

VERB_DESCRIPTION_build := Build the project distribution artifacts.

VERB_DESCRIPTION_examples := Run this repository'"'"'s explicitly declared public runtime examples module.

VERB_DESCRIPTION_check := Check the configured quality gates and projection conformance without writes.

VERB_DESCRIPTION_check-pyright := Run the complete configured Pyright analysis and retain its native gate receipt.

VERB_DESCRIPTION_census := Inventory constants, duplicate declarations and placement across every declared first-party project.

VERB_DESCRIPTION_census-constants := Apply identity-proven constants consumer rewrites through the workspace census and publish its complete report.

VERB_DESCRIPTION_smells := Run the strict code-smell audit as a dedicated gate.

VERB_DESCRIPTION_test := Run incremental tests through the persistent testmon cache.

VERB_DESCRIPTION_tests := Run incremental tests first, then the complete eligible suite using the same persistent testmon database, without a time limit.

VERB_DESCRIPTION_test-full := Run incremental tests first, then the complete eligible suite using the same persistent testmon database, without a time limit.

VERB_DESCRIPTION_test-file := Run one complete declared test file once, slow items included, with the same persistent testmon cache (FILE=<repository-relative path>).

VERB_DESCRIPTION_file-gate := Run configured canonical read-only gates on one literal FILE=<repository-relative path>; invalid selection and missing gate owners fail loud.

VERB_DESCRIPTION_profile-test := Profile the canonical pytest entry and its collection children on the same persistent testmon database, without the outer bounded-gate wrapper.

VERB_DESCRIPTION_profile-test-report := Render the parent pytest profile and the aggregated child profiles from that run.

VERB_DESCRIPTION_profile-docs := Profile the complete configured mutating documentation lifecycle and retain reached phase profiles on native failure.

VERB_DESCRIPTION_profile-docs-report := Render only the latest documentation profiling run, including explicit lifecycle and phase completion states.

VERB_DESCRIPTION_profile-environment := Profile the canonical environment provenance guard without invoking that guard as a prerequisite.

VERB_DESCRIPTION_profile-environment-report := Render the profile retained by the canonical environment provenance guard.

VERB_DESCRIPTION_profile-lanes := Profile the complete read-only workspace lane audit and retain its native failure outcome.

VERB_DESCRIPTION_profile-lanes-report := Render the retained profile of the complete workspace lane audit.

VERB_DESCRIPTION_profile-mod := Profile the canonical mutating codemod lifecycle and retain the profile on failure.

VERB_DESCRIPTION_profile-mod-report := Render the canonical codemod profile from the last profiling invocation.

VERB_DESCRIPTION_profile-mod-rules := Profile the complete read-only rule diagnostic and retain its native failure outcome.

VERB_DESCRIPTION_profile-mod-rules-report := Render the profile of the complete read-only rule diagnostic.

VERB_DESCRIPTION_profile-codemod := Profile the canonical read-only codemod gate for one literal FILE, preserving native findings and interruption evidence.

VERB_DESCRIPTION_profile-codemod-report := Render the retained profile of the canonical read-only codemod gate.

VERB_DESCRIPTION_fmt := Apply ruff format --preview and every declared formatter gate. Ruff is the rule; change code, never ruff.

VERB_DESCRIPTION_fix := Apply conformance repairs and the configured fixable gates.

VERB_DESCRIPTION_fix-namespace := Apply the canonical namespace enforcer to the selected workspace; the same relocation cascade runs as a callback phase of make mod.

VERB_DESCRIPTION_fix-accessors := Migrate accessor names owned by the rename catalog'"'"'s origin package and every resolved consumer; homonyms are skipped with a warning; the same origin-aware rewrite runs as a callback phase of make mod.

VERB_DESCRIPTION_fix-toolchain := Reinstall one explicitly selected declared tool from its current lock, then prove the complete native toolchain.

VERB_DESCRIPTION_audit := Inspect environment, dependency, and workspace ownership health.

VERB_DESCRIPTION_status := Report the resolved runtime and repository state.

VERB_DESCRIPTION_verify-clean := Audit documentation consistency and report the current Git state without rejecting concurrent work or requiring cleanup.

VERB_DESCRIPTION_docs := Generate, fix, format, and check documentation.

VERB_DESCRIPTION_clean := Remove every declared disposable artifact.

VERB_DESCRIPTION_bootstrap-candidate := Bootstrap declared candidate worktrees with this branch-matched generator.

VERB_DESCRIPTION_release-plan := Resolve the release decision through the public protocol.

VERB_DESCRIPTION_release-version := Materialize the planned version.

VERB_DESCRIPTION_release-tag := Tag the verified release commit.

VERB_DESCRIPTION_release-build := Build the release receipt and artifacts.

VERB_DESCRIPTION_publication := Publish only receipt-attested release artifacts.

VERB_DESCRIPTION_gen := Regenerate declared projections with guarded publication per participant, without repairing authored inputs or resolving locks.

VERB_DESCRIPTION_gen-footprint := Inspect the pending generation journal and physical effect footprint without generation planning, participant leases, recovery, or publication.

VERB_DESCRIPTION_gen-pyproject := Regenerate this project'"'"'s analyzer and dependency metadata through guarded publication.

VERB_DESCRIPTION_gen-editor := Project editor settings, analyzer configuration, and Git ignores without generating Python facades.

VERB_DESCRIPTION_initialize := Bootstrap this target'"'"'s own declaration and canonical Makefile.

VERB_DESCRIPTION_project-new := Create a new project through the canonical scaffold service using explicitly declared identity inputs.

VERB_DESCRIPTION_mod := Apply the declared structural codemods; committed rule-test snapshots are verified, never rewritten.

VERB_DESCRIPTION_mod-types := Apply the declared source type-contract migrations through the native AST fixed point.

VERB_DESCRIPTION_mod-rules := Evaluate the declared AST and text rule catalogs across every governed project without rewriting sources; publish occurrence evidence by rule and repository.

VERB_DESCRIPTION_mod-text := Apply the declared Sed text rules with exact receipts and guarded publication.

VERB_DESCRIPTION_mod-text-candidate := Apply declared Sed text rules to the configured candidate workspace.

VERB_DESCRIPTION_mod-snapshots := Regenerate the owned ast-grep rule-test snapshots from their tests for a reviewed commit.

VERB_DESCRIPTION_waza := Validate provider-neutral governance semantics with Waza.

VERB_DESCRIPTION_duplication := Run the canonical jscpd duplicate-code gate.

VERB_DESCRIPTION_sonarcloud-sync := Write the SSOT SonarCloud issue exclusions to the server-side project settings (requires SONAR_TOKEN).

VERB_DESCRIPTION_sonarcloud-issues := Read unresolved new-code SonarCloud issues on the published integration branch (requires SONAR_TOKEN).

VERB_HELP_LINE = printf '  %-16s %s\n' '$(1)' '$(VERB_DESCRIPTION_$(1))'
# OPTIONS=Y (or HELP=Y) displays a built-in verb's contract without effects:
# no prerequisite, hook or handler of that verb runs. A script verb keeps its
# entry, and the promoted dispatcher renders its contract per WHAT.
VERB_CONTRACT := $(filter 1 TRUE Y YES true y yes True Yes,$(OPTIONS) $(HELP))
ifeq ($(VERB_CONTRACT),)
PROFILE_VERBS_PLAIN := gen gen-footprint gen-pyproject gen-editor project-new
PROFILE_VERBS_DISPATCH := help clean
PROFILE_VERBS_ACTIVATED := pre-commit upg-sources validate-lifecycle build examples check check-pyright census census-constants smells test tests test-full test-file file-gate profile-test profile-test-report profile-docs profile-docs-report profile-environment profile-environment-report profile-lanes profile-lanes-report profile-mod profile-mod-report profile-mod-rules profile-mod-rules-report profile-codemod profile-codemod-report fmt fix fix-namespace fix-accessors fix-toolchain audit status verify-clean docs bootstrap-candidate release-plan release-version release-tag release-build publication initialize mod mod-types mod-rules mod-text mod-text-candidate mod-snapshots waza duplication sonarcloud-sync sonarcloud-issues
PROFILE_VERBS_INTERPRETER := profile-environment profile-environment-report status initialize
PROFILE_VERBS_HOOKED := $(foreach v,$(filter-out $(PROFILE_VERBS_INTERPRETER),$(PROFILE_VERBS_ACTIVATED)),$(if $(filter pre-$(v) _custom-$(v) post-$(v),$(CUSTOM_DECLARED_TARGETS)),$(v)))
PROFILE_VERBS_DIRECT := $(filter-out $(PROFILE_VERBS_HOOKED) $(PROFILE_VERBS_INTERPRETER),$(PROFILE_VERBS_ACTIVATED))
PROFILE_VERBS_ROUTED := $(PROFILE_VERBS_HOOKED) $(PROFILE_VERBS_INTERPRETER)

$(PROFILE_VERBS_PLAIN): %: _builtin_require_workspace _builtin_require_interpreter
	@printf '==> %s | project=%s | profile=%s\n' '$@' "$(PROJECT_ROOT)" 'standalone'
	$(call RUN_PUBLIC,$@)


$(PROFILE_VERBS_DISPATCH):
	$(if $(filter Y,$(CI)),+@$(SELF_MAKE) _builtin-$@,$(call RUN_PUBLIC,$@))


$(PROFILE_VERBS_ACTIVATED): %: _builtin_require_workspace
	@printf '==> %s | project=%s | profile=%s\n' '$@' "$(PROJECT_ROOT)" 'standalone'
	$(if $(filter status,$@),@$(GITHUB_AUTH_DIAGNOSTICS))
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
	$(if $(filter status,$*),+@$(SELF_MAKE) _builtin_status_import_observation)
	$(if $(filter status,$*),+@$(SELF_MAKE) _builtin_require_environment)
	$(if $(filter Y,$(CI)),+@$(SELF_MAKE) _builtin-$*,$(call RUN_PUBLIC,$*))
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

_builtin-pre-commit: _builtin_require_environment
	+@CI=Y $(SELF_MAKE) check
	+@CI=Y $(SELF_MAKE) test

# `upg` builds the environment from the locks it writes, so like `setup` it
# must not require an existing environment, and as the only resolver it must
# not require a satisfied committed mise.lock either. Its bootstrap half runs
# `mise lock --bump` first (native resolution; no stage and no prior install),
# then installs from the fresh lock. The lifecycle verifies native UV ownership
# before resolving its two physical view locks. Attached members delegate
# before bootstrap, installation, hooks or lock resolution.
ifeq ($(PROJECT_ROOT),$(RUNTIME_ROOT))
upg: TOOL_BOOTSTRAP_LIFECYCLE := _upg_lifecycle
ifeq ($(EDITABLE_SUPPLIER_ACTIVE),)
upg: TOOL_BOOTSTRAP_RESOLVE := 1
endif
upg: _bootstrap_setup_tools
else
upg: _builtin_require_workspace
	+@"$(SELF_MAKE_EXECUTABLE)" --no-print-directory -C "$(RUNTIME_ROOT)" -f "$(RUNTIME_ROOT)/Makefile" upg
endif
else
$(BUILTIN_VERBS):
	@$(call VERB_HELP_LINE,$@)
	@printf '%s\n' 'OPTIONS=Y displays this contract without effects; run make $@ to execute it.'
endif
SCRIPT_VERBS_PLAIN :=
SCRIPT_VERBS_DISPATCH :=
SCRIPT_VERBS_ACTIVATED :=
SCRIPT_VERBS_INTERPRETER :=
SCRIPT_VERBS_HOOKED := $(foreach v,$(filter-out $(SCRIPT_VERBS_INTERPRETER),$(SCRIPT_VERBS_ACTIVATED)),$(if $(filter pre-$(v) _custom-$(v) post-$(v),$(CUSTOM_DECLARED_TARGETS)),$(v)))
SCRIPT_VERBS_DIRECT := $(filter-out $(SCRIPT_VERBS_HOOKED) $(SCRIPT_VERBS_INTERPRETER),$(SCRIPT_VERBS_ACTIVATED))
SCRIPT_VERBS_ROUTED := $(SCRIPT_VERBS_HOOKED) $(SCRIPT_VERBS_INTERPRETER)




.PHONY: _setup_lifecycle
_setup_lifecycle:
	@set -eu; \
	case "$(strip $(CI)): $(CUSTOM_DECLARED_TARGETS) " in \
		Y:*) ;; \
		*" pre-setup "*) $(SELF_MAKE) pre-setup ;; \
	esac
	@$(SELF_MAKE) _builtin_setup_environment
	+@$(if $(filter Y,$(CI)),$(SELF_MAKE) _setup_activated,direnv exec "$(PROJECT_ROOT)" $(SELF_MAKE) _setup_activated)

# Setup only provisions, in every context, CI included: whatever the committed
# locks pin is installed (a drifted uv.lock installs --frozen; a mise.lock
# without the self-managed release provisions through the host Mise) and a
# lock defect is reported with its cure, never a stop
# (operator-ruling-2026-10-09-setup-resilient). The verdicts - lock drift, lock
# integrity, the toolchain reality proof - belong to `make audit`, which CI runs
# right after setup; `make upg` cures them and never depends on setup.
.PHONY: _setup_activated
# Audit owns the final pin, completeness and runtime verdicts after provisioning.
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

# A provisioned local setup needs no GitHub credential. Mise and Git receive
# only the explicit process credential above; an operation that actually needs
# network access reports its own failure without a preflight or fallback.

# Every operational verb runs mise exactly as the committed mise.lock pins it:
# the version guard replaces the retired mise.version pin file. Direnv
# activation (or the CI-provisioned PATH) resolves `mise` through the
# mise-managed shims; a stale or absent toolchain fails loud and names the
# native repair verb.
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
	mise_pin="$$( awk 'index($$0, "[[tools.\"github:jdx/mise\"]]") == 1 { inside = 1; next } inside && substr($$0, 1, 1) == "[" { exit } inside && $$1 == "version" { gsub(/[",]/, "", $$3); print $$3; exit }' "$(RUNTIME_ROOT)/mise.lock" )"; \
	if [ -z "$$mise_pin" ]; then \
		printf 'ERROR: mise.lock pins no github:jdx/mise release; run make upg\n' >&2; \
		exit 2; \
	fi; \
	mise_version_output="$$(mise --version)"; \
	mise_actual="$${mise_version_output%% *}"; \
	if [ "$$mise_actual" != "$$mise_pin" ]; then \
		printf 'ERROR: mise %s differs from the mise.lock pin %s; run make setup\n' "$$mise_actual" "$$mise_pin" >&2; \
		exit 2; \
	fi

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

# Runtime commands authenticate the graph setup provisioned; only setup installs.
# Generation guards its physical runtime and consumed inputs at publication;
# the complete installed graph remains the explicit audit's responsibility.
_builtin_require_environment: _builtin_require_interpreter
ifneq ($(MAKECMDGOALS),help)
	@$(PROJECT_FLEXT_INFRA) workspace verify-environment --repository-root "$(RUNTIME_ROOT)"
endif

# === SECTION: setup environment (managed) ===
# Source: computed (MAKE_PROFILE routing)
# Setup provisions tooling, dependencies and declared runtime-local bindings.
# It never generates or mutates project source; `make gen` owns source generation.
# Setup installs the committed uv.lock and never writes it (lock law: only
# `make upg` writes locks); uv owns the venv: `uv sync --python` creates or
# replaces it against the declared interpreter.
# Local setup provisions declared gitlinks. CI consumes the committed lock
# without initializing member checkouts.
_builtin_setup_environment: $(if $(filter true,$(GITHUB_ACTIONS)),,_builtin_setup_submodules)
	+@set -eu; \
	if [ "$(PROJECT_ROOT)" != "$(RUNTIME_ROOT)" ]; then \
		"$(SELF_MAKE_EXECUTABLE)" --no-print-directory -f "$(RUNTIME_ROOT)/Makefile" setup; \
	else \
		if [ -n "$(EDITABLE_SUPPLIER_ACTIVE)" ]; then \
			printf 'setup: selected live supplier requires native view reconciliation; delegating lock writes to make upg\n'; \
			$(SELF_MAKE) upg; \
		fi; \
		$(SETUP_ENVIRONMENT_RECIPE); \
	fi
# End SECTION: setup environment

# Upgrade provisions its trusted locked tool closure before its producer runs.
# The complete setup acceptance follows projection and native lock convergence.
.PHONY: _upg_install_runtime
_upg_install_runtime:
	@$(INSTALL_RUNTIME_RECIPE)

# Resolve both native views before the first install, so the producer that
# runs the corrective projection is the one the declarations resolve to: a
# consumer whose lock still pins an older producer (one without the routes
# this template calls), or that has no view lock yet, upgrades in one run.
# The corrective projection is then re-locked by _upg_reconcile_locks.
# Only this elected runtime owner installs the resolved environment. Attached
# members delegate before bootstrap; this owner also refreshes their own native
# UV and tool locks for independent setup without installing member runtimes.
# Reparse a changed dispatcher once; a second change fails the fixed point.
.PHONY: _upg_lifecycle _upg_converge

_builtin-upg-sources: _builtin_require_environment

	@if [ -z "$(strip $(SOURCE_ROOT))" ]; then \
		printf 'ERROR[upg-sources]: SOURCE_ROOT is required\n' >&2; \
		exit 2; \
	fi

	@if [ -z "$(strip $(SOURCE_URL_PREFIX))" ]; then \
		printf 'ERROR[upg-sources]: SOURCE_URL_PREFIX is required\n' >&2; \
		exit 2; \
	fi

	@if [ -z "$(strip $(SOURCE_REF_FROM))" ]; then \
		printf 'ERROR[upg-sources]: SOURCE_REF_FROM is required\n' >&2; \
		exit 2; \
	fi

	@if [ -z "$(strip $(SOURCE_REF_TO))" ]; then \
		printf 'ERROR[upg-sources]: SOURCE_REF_TO is required\n' >&2; \
		exit 2; \
	fi

	@$(PROJECT_FLEXT_INFRA) deps modernize --repository-root "$(SOURCE_ROOT)" \
		--apply --git-source-url-prefix "$(SOURCE_URL_PREFIX)" \
		--git-source-ref-from "$(SOURCE_REF_FROM)" --git-source-ref-to "$(SOURCE_REF_TO)"

_upg_lifecycle: _builtin_setup_submodules
ifneq ($(EDITABLE_SUPPLIER_ACTIVE),)
	@$(EDITABLE_RESOLUTION_RECIPE)
endif
	@set -eu; \
	case " $(CUSTOM_DECLARED_TARGETS) " in \
		*" pre-upg "*) $(SELF_MAKE) pre-upg ;; \
	esac
ifneq ($(EDITABLE_SUPPLIER_ACTIVE),)
	+@$(SELF_MAKE) _upg_resolve_locks
	@$(SELF_MAKE) _upg_install_runtime
	@$(PROJECT_FLEXT_INFRA) codegen generate --root "$(PROJECT_ROOT)" --what pyproject --scope self
	@$(SELF_MAKE) _upg_reconcile_locks
	@$(SELF_MAKE) _upg_install_runtime
	@$(PROJECT_FLEXT_INFRA) workspace verify-environment --repository-root "$(RUNTIME_ROOT)"
	+@$(SELF_MAKE) _upg_refresh_member_locks
	$(call RUN_PUBLIC_POST,upg)
else
	+@$(SELF_MAKE) _upg_resolve_locks
	@$(SELF_MAKE) _upg_install_runtime
	@$(PROJECT_FLEXT_INFRA) codegen generate --root "$(PROJECT_ROOT)" --what pyproject --scope all
	@$(SELF_MAKE) _upg_reconcile_locks
	@$(SELF_MAKE) _upg_install_runtime
	@$(PROJECT_FLEXT_INFRA) workspace verify-environment --repository-root "$(RUNTIME_ROOT)"
	@$(PROJECT_FLEXT_INFRA) codegen generate --root "$(PROJECT_ROOT)" --what mise-config --scope self
	@$(PROJECT_FLEXT_INFRA) codegen generate --root "$(PROJECT_ROOT)" --what makefile --scope all
	+@$(SELF_MAKE) _upg_provision_tools
	+@$(SELF_MAKE) _upg_refresh_member_locks
	@$(PROJECT_FLEXT_INFRA) deps modernize --repository-root "$(PROJECT_ROOT)" \
		--apply --rewrite-constraints --projects .
	$(call RUN_PUBLIC_PRODUCE,gen)
	+@$(SELF_MAKE) _upg_converge
endif

# Reparse the projected dispatcher before resolving the declared native tool
# locks. Each writable FLEXT member retains its own lock for independent setup;
# only the elected runtime receives an installation.
.PHONY: _upg_provision_tools
_upg_provision_tools:
	@mise -C "$(PROJECT_ROOT)" lock --bump
	@mise -C "$(PROJECT_ROOT)" install --locked --yes "python" "github:jdx/mise" "uv" "github:direnv/direnv" "taplo" "aqua:ast-grep/ast-grep" "gitleaks" "aqua:boyter/scc" "node" "github:qltysh/qlty" "github:kucherenko/jscpd" "github:microsoft/waza"

# Writable FLEXT members keep their own native locks for independent setup.
UPG_LOCK_MEMBERS :=

# Every member .mise.toml is the root's toolchain projection, so its mise.lock
# is the root resolution, published once instead of resolved again per member;
# a member manifest that differs fails loud. An editable supplier keeps the
# runtime's pins while members refresh their own UV views.
.PHONY: _upg_refresh_member_locks
_upg_refresh_member_locks:
	+@set -eu; \
	for member in $(UPG_LOCK_MEMBERS); do \
		if ! cmp -s "$(PROJECT_ROOT)/.mise.toml" "$(PROJECT_ROOT)/$$member/.mise.toml"; then \
			printf 'ERROR: %s/.mise.toml differs from the root toolchain projection; run make gen\n' "$$member" >&2; \
			exit 2; \
		fi; \
		cp "$(PROJECT_ROOT)/mise.lock" "$(PROJECT_ROOT)/$$member/mise.lock.upg"; \
		mv "$(PROJECT_ROOT)/$$member/mise.lock.upg" "$(PROJECT_ROOT)/$$member/mise.lock"; \
		"$(SELF_MAKE_EXECUTABLE)" --no-print-directory -C "$(PROJECT_ROOT)" -f "$(PROJECT_ROOT)/$$member/Makefile" _upg_resolve_locks; \
	done

# Complete the upgrade against the projections just published. Re-reading the
# current Makefile selects the new recipes without restarting the upgrade or
# repeating generation and installation.
_upg_converge:
	@$(PROJECT_FLEXT_INFRA) codegen generate --root "$(PROJECT_ROOT)" --what pyproject --scope all
	@$(SELF_MAKE) _upg_reconcile_locks
	+@set -eu; \
	for member in $(UPG_LOCK_MEMBERS); do \
		"$(SELF_MAKE_EXECUTABLE)" --no-print-directory -C "$(PROJECT_ROOT)" -f "$(PROJECT_ROOT)/$$member/Makefile" _upg_reconcile_locks; \
	done
	@$(SELF_MAKE) _builtin_setup_environment
	@$(SELF_MAKE) _builtin_require_mise
	@$(PROJECT_FLEXT_INFRA) deps verify-locks --repository-root "$(PROJECT_ROOT)"
	+@$(if $(filter Y,$(CI)),$(SELF_MAKE) _upg_activated,direnv exec "$(PROJECT_ROOT)" $(SELF_MAKE) _upg_activated)

.PHONY: _upg_resolve_locks _upg_reconcile_locks
_upg_resolve_locks: _builtin_require_upg_lock_owner
	@set -eu; \
	desired_python="$$(mise -C "$(RUNTIME_ROOT)" which python)"; \
	for resolution in $(UPG_RESOLUTION_PROJECTS); do \
		printf 'upg: upgrading UV lock for %s\n' "$$resolution"; \
		$(UV) lock --project "$$resolution" --python "$$desired_python" $(if $(EDITABLE_SUPPLIER_ACTIVE),,--upgrade --refresh); \
		$(UV) lock --check --project "$$resolution" --python "$$desired_python"; \
	done

_upg_reconcile_locks: _builtin_require_upg_lock_owner
	@set -eu; \
	desired_python="$$(mise -C "$(RUNTIME_ROOT)" which python)"; \
	for resolution in $(UPG_RESOLUTION_PROJECTS); do \
		printf 'upg: reconciling UV lock for %s\n' "$$resolution"; \
		$(UV) lock --project "$$resolution" --python "$$desired_python"; \
		$(UV) lock --check --project "$$resolution" --python "$$desired_python"; \
	done

.PHONY: _upg_activated
_upg_activated:
	$(call RUN_PUBLIC_POST,gen)
	@set -eu; \
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
		gates="lint,format,security,markdown,markdown-format,markdown-code,duplication,pyrefly,mypy,pyright,conflict-markers,loc-cap,runtime-census,fresh-import,deps,index-declarations,codemod,layout,direnv"; \
		if [ "$(strip $(CI))" = "Y" ]; then \
			gates="lint,format,security,markdown,markdown-format,markdown-code,duplication,pyrefly,conflict-markers,loc-cap,runtime-census,fresh-import,deps,index-declarations,layout"; \
			printf 'INFO: CI=Y runs check gates: lint format security markdown markdown-format markdown-code duplication pyrefly conflict-markers loc-cap runtime-census fresh-import deps index-declarations layout\n'; \
		elif [ "$(strip $(CI))" = "N" ]; then \
			gates="lint,format,security,markdown,markdown-format,markdown-code,duplication,pyrefly,mypy,pyright,conflict-markers,loc-cap,runtime-census,fresh-import,deps,index-declarations,codemod,layout,direnv"; \
			printf 'INFO: CI=N runs check gates: lint format security markdown markdown-format markdown-code duplication pyrefly mypy pyright conflict-markers loc-cap runtime-census fresh-import deps index-declarations codemod layout direnv\n'; \
		else \
			printf 'INFO: default context runs check gates: lint format security markdown markdown-format markdown-code duplication pyrefly mypy pyright conflict-markers loc-cap runtime-census fresh-import deps index-declarations codemod layout direnv\n'; \
		fi; \
		if [ -z "$$gates" ]; then \
			printf 'ERROR: no active check gates remain in the selected context\n' >&2; \
			exit 2; \
		fi; \
		$(PROJECT_FLEXT_INFRA) workspace capture-gate --repository-root "$(PROJECT_ROOT)" --gates "$$gates"; \
		$(PROJECT_FLEXT_INFRA) codegen conform --root "$(PROJECT_ROOT)" --scope self --mode check

_builtin_test_all: _builtin_require_environment
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
	TESTMON_DATAFILE="$$database" $(PYTEST_BOUNDED) $(PROJECT_INFRA_RUN) -m flext_infra._pytest_entry

_builtin_test_full_all: _builtin_require_environment
	@if [ -n "$(strip $(FILE))" ]; then \
	if [ -z "$(strip $(FILE))" ]; then printf 'ERROR: test-full requires FILE=<repository-relative test file path>\n' >&2; exit 2; fi; \
	case "$(FILE)" in /*|*..*) printf 'ERROR: FILE must stay a repository-relative path: %s\n' "$(FILE)" >&2; exit 2 ;; esac; \
	if [ ! -f "$(PROJECT_ROOT)/$(FILE)" ]; then printf 'ERROR: FILE is not an existing repository file: %s\n' "$(FILE)" >&2; exit 2; fi; \
	export FLEXT_PYTEST_TARGET_FILE="$(FILE)"; \
	fi; \
	set -eu; \
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
	TESTMON_DATAFILE="$$database" $(PROJECT_INFRA_RUN) -m flext_infra._pytest_entry full

_builtin_test_file_all: _builtin_require_environment
	@if [ -z "$(strip $(FILE))" ]; then printf 'ERROR: test-file requires FILE=<repository-relative test file path>\n' >&2; exit 2; fi; \
	case "$(FILE)" in /*|*..*) printf 'ERROR: FILE must stay a repository-relative path: %s\n' "$(FILE)" >&2; exit 2 ;; esac; \
	if [ ! -f "$(PROJECT_ROOT)/$(FILE)" ]; then printf 'ERROR: FILE is not an existing repository file: %s\n' "$(FILE)" >&2; exit 2; fi; \
	export FLEXT_PYTEST_TARGET_FILE="$(FILE)"; \
	set -eu; \
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
	TESTMON_DATAFILE="$$database" $(PROJECT_INFRA_RUN) -m flext_infra._pytest_entry file

# Literal-file selection and verdicts belong to the existing canonical checker.
# Export the raw Make value instead of interpolating operator input into shell code.
export FLEXT_FILE_GATE_FILE := $(value FILE)
_builtin_file_gate_all: _builtin_require_environment
	@$(PROJECT_FLEXT_INFRA) check run --repository-root "$(PROJECT_ROOT)" --gates "lint,format,pyrefly,mypy,pyright,codemod" --file "$$FLEXT_FILE_GATE_FILE"

# Isolate the same read-only codemod gate when diagnosing file-gate preparation.
_builtin-profile-codemod: _builtin_require_environment
	@mkdir -p "$(PROFILE_REPORTS_DIR)"
	@$(PROJECT_INFRA_RUN) -m flext_infra._profile_entry \
		"$(PROFILE_REPORTS_DIR)/codemod.pstats" check run \
		--repository-root "$(PROJECT_ROOT)" --gates codemod \
		--file "$$FLEXT_FILE_GATE_FILE"

_builtin-profile-codemod-report: _builtin_require_environment
	@cd "$(PROJECT_ROOT)" || exit $$?; \
		$(PROJECT_INFRA_RUN) -m flext_infra._cprofile_entry \
		"$(PROFILE_REPORTS_DIR)/codemod.pstats"

# fmt is format-only (single-pass verb law): ruff formats Python, the
# fmt_gates formatters run once through the checker's apply mode, and every
# lint repair belongs to `make fix`. Every repair requires read-only native
# verification; remaining findings and tool failures both fail the verb.
_builtin_fmt_all: _builtin_require_environment
	@$(PROJECT_FLEXT_INFRA) check run --repository-root "$(PROJECT_ROOT)" --gates "format,markdown-format" --apply
	@set -eu; \
		gates="lint,format,security,markdown,markdown-format,markdown-code,duplication,pyrefly,mypy,pyright,conflict-markers,loc-cap,runtime-census,fresh-import,deps,index-declarations,codemod,layout,direnv"; \
		if [ "$(strip $(CI))" = "Y" ]; then \
			gates="lint,format,security,markdown,markdown-format,markdown-code,duplication,pyrefly,conflict-markers,loc-cap,runtime-census,fresh-import,deps,index-declarations,layout"; \
			printf 'INFO: CI=Y runs check gates: lint format security markdown markdown-format markdown-code duplication pyrefly conflict-markers loc-cap runtime-census fresh-import deps index-declarations layout\n'; \
		elif [ "$(strip $(CI))" = "N" ]; then \
			gates="lint,format,security,markdown,markdown-format,markdown-code,duplication,pyrefly,mypy,pyright,conflict-markers,loc-cap,runtime-census,fresh-import,deps,index-declarations,codemod,layout,direnv"; \
			printf 'INFO: CI=N runs check gates: lint format security markdown markdown-format markdown-code duplication pyrefly mypy pyright conflict-markers loc-cap runtime-census fresh-import deps index-declarations codemod layout direnv\n'; \
		else \
			printf 'INFO: default context runs check gates: lint format security markdown markdown-format markdown-code duplication pyrefly mypy pyright conflict-markers loc-cap runtime-census fresh-import deps index-declarations codemod layout direnv\n'; \
		fi; \
		if [ -z "$$gates" ]; then \
			printf 'ERROR: no active check gates remain in the selected context\n' >&2; \
			exit 2; \
		fi; \
		$(PROJECT_FLEXT_INFRA) workspace capture-gate --repository-root "$(PROJECT_ROOT)" --gates "$$gates"; \
		$(PROJECT_FLEXT_INFRA) codegen conform --root "$(PROJECT_ROOT)" --scope self --mode check

_builtin_fix_all: _builtin_require_environment
	@$(PROJECT_FLEXT_INFRA) codegen conform --root "$(PROJECT_ROOT)" --scope self --mode apply
	@$(PROJECT_FLEXT_INFRA) check run --repository-root "$(PROJECT_ROOT)" --gates "lint,markdown,markdown-code,deps" --apply
	@set -eu; \
		gates="lint,format,security,markdown,markdown-format,markdown-code,duplication,pyrefly,mypy,pyright,conflict-markers,loc-cap,runtime-census,fresh-import,deps,index-declarations,codemod,layout,direnv"; \
		if [ "$(strip $(CI))" = "Y" ]; then \
			gates="lint,format,security,markdown,markdown-format,markdown-code,duplication,pyrefly,conflict-markers,loc-cap,runtime-census,fresh-import,deps,index-declarations,layout"; \
			printf 'INFO: CI=Y runs check gates: lint format security markdown markdown-format markdown-code duplication pyrefly conflict-markers loc-cap runtime-census fresh-import deps index-declarations layout\n'; \
		elif [ "$(strip $(CI))" = "N" ]; then \
			gates="lint,format,security,markdown,markdown-format,markdown-code,duplication,pyrefly,mypy,pyright,conflict-markers,loc-cap,runtime-census,fresh-import,deps,index-declarations,codemod,layout,direnv"; \
			printf 'INFO: CI=N runs check gates: lint format security markdown markdown-format markdown-code duplication pyrefly mypy pyright conflict-markers loc-cap runtime-census fresh-import deps index-declarations codemod layout direnv\n'; \
		else \
			printf 'INFO: default context runs check gates: lint format security markdown markdown-format markdown-code duplication pyrefly mypy pyright conflict-markers loc-cap runtime-census fresh-import deps index-declarations codemod layout direnv\n'; \
		fi; \
		if [ -z "$$gates" ]; then \
			printf 'ERROR: no active check gates remain in the selected context\n' >&2; \
			exit 2; \
		fi; \
		$(PROJECT_FLEXT_INFRA) workspace capture-gate --repository-root "$(PROJECT_ROOT)" --gates "$$gates"; \
		$(PROJECT_FLEXT_INFRA) codegen conform --root "$(PROJECT_ROOT)" --scope self --mode check

# SonarCloud server-side issue exclusions (SSOT: codegen.sonarcloud). The verb
# writes an external service with SONAR_TOKEN from the environment; it belongs
# to no setup/gen/check/test workflow row and never runs implicitly.
_builtin_sonarcloud_sync_all: _builtin_require_environment
	@$(PROJECT_FLEXT_INFRA) maintenance sonarcloud-sync --repository-root "$(PROJECT_ROOT)"

# Read the complete unresolved new-code issue set on the published integration
# branch. This diagnostic never writes SonarCloud settings or issue state.
_builtin_sonarcloud_issues_all: _builtin_require_environment
	@$(PROJECT_FLEXT_INFRA) maintenance sonarcloud-issues --repository-root "$(PROJECT_ROOT)"


_builtin_run_default: _builtin_require_environment
	@$(UV_RUN) $(PROJECT_NAME) $(ARGS)

# Profile the real runtime-census gate through the same installed CLI selected
# by the root dispatcher. The report stays in this checkout's .reports tree.
.PHONY: profile-census
profile-census: _builtin_require_environment
	@mkdir -p "$(PROFILE_REPORTS_DIR)"
	@$(PROJECT_INFRA_RUN) -m flext_infra._profile_entry \
		"$(PROFILE_REPORTS_DIR)/runtime-census.pstats" check run \
		--repository-root "$(PROJECT_ROOT)" --gates runtime-census

.PHONY: profile-census-report
profile-census-report: _builtin_require_environment
	@$(RUNTIME_PYTHON) -c \
		'import pstats, sys; pstats.Stats(sys.argv[1]).sort_stats("cumtime").print_stats(35)' \
		"$(PROFILE_REPORTS_DIR)/runtime-census.pstats"

# Profile the checker process itself while retaining the canonical Mypy gate,
# its resource limit, native source inventory, and report directory.
.PHONY: profile-mypy
profile-mypy: _builtin_require_environment
	@mkdir -p "$(PROFILE_REPORTS_DIR)"
	@export FLEXT_MYPY_PROFILE_OUTPUT="$(PROFILE_REPORTS_DIR)/mypy.pstats"; \
		$(PROJECT_FLEXT_INFRA) check run --repository-root "$(PROJECT_ROOT)" \
		--gates mypy

.PHONY: profile-mypy-report
profile-mypy-report: _builtin_require_environment
	@$(RUNTIME_PYTHON) -c \
		'import pstats, sys; pstats.Stats(sys.argv[1]).sort_stats("cumtime").print_stats(50)' \
		"$(PROFILE_REPORTS_DIR)/mypy.pstats"

# Profile the complete generation operation, including its publication.
.PHONY: profile-gen
profile-gen: _builtin_require_interpreter
	@mkdir -p "$(PROFILE_REPORTS_DIR)"
	@set -eu; \
	scope=all; \
	if [ "$(strip $(CI))" = "Y" ]; then scope=self; fi; \
	$(PROJECT_INFRA_RUN) -m flext_infra._profile_entry \
		"$(PROFILE_REPORTS_DIR)/conform.pstats" codegen generate \
		--root "$(PROJECT_ROOT)" --scope "$$scope"

.PHONY: profile-gen-report
profile-gen-report: _builtin_require_environment
	@$(RUNTIME_PYTHON) -c \
		'import pstats, sys; stats = pstats.Stats(sys.argv[1]); stats.sort_stats("cumtime").print_stats(50); stats.sort_stats("tottime").print_callers(15)' \
		"$(PROFILE_REPORTS_DIR)/conform.pstats"

# Measure the environment guard itself before changing its provenance checks.
_builtin-profile-environment: _builtin_require_interpreter
	@mkdir -p "$(PROFILE_REPORTS_DIR)"
	@$(PROJECT_INFRA_RUN) -m flext_infra._profile_entry \
		"$(PROFILE_REPORTS_DIR)/environment.pstats" workspace verify-environment \
		--repository-root "$(RUNTIME_ROOT)"

_builtin-profile-environment-report: _builtin_require_interpreter
	@cd "$(PROJECT_ROOT)" || exit $$?; \
		$(PROJECT_INFRA_RUN) -m flext_infra._cprofile_entry \
		"$(PROFILE_REPORTS_DIR)/environment.pstats"

# Measure the same complete lane audit used by the public audit verb.
_builtin-profile-lanes: _builtin_require_environment
	@mkdir -p "$(PROFILE_REPORTS_DIR)"
	@$(PROJECT_INFRA_RUN) -m flext_infra._profile_entry \
		"$(PROFILE_REPORTS_DIR)/lanes.pstats" \
		workspace verify-lanes --repo-root "$(PROJECT_ROOT)"

_builtin-profile-lanes-report: _builtin_require_environment
	@cd "$(PROJECT_ROOT)" || exit $$?; \
		$(PROJECT_INFRA_RUN) -m flext_infra._cprofile_entry \
		"$(PROFILE_REPORTS_DIR)/lanes.pstats"

# Measure the same mutating codemod lifecycle, retaining profiles on failure.
_builtin-profile-mod: _builtin_require_environment
	@mkdir -p "$(PROFILE_REPORTS_DIR)"
	@$(PROJECT_INFRA_RUN) -m flext_infra._profile_entry \
		"$(PROFILE_REPORTS_DIR)/mod.pstats" refactor mod \
		--repository-root "$(PROJECT_ROOT)" --apply

_builtin-profile-mod-report: _builtin_require_environment
	@cd "$(PROJECT_ROOT)" || exit $$?; \
		$(PROJECT_INFRA_RUN) -m flext_infra._cprofile_entry \
		"$(PROFILE_REPORTS_DIR)/mod.pstats"

# Measure the same read-only diagnostic used by the public mod-rules verb.
_builtin-profile-mod-rules: _builtin_require_environment
	@mkdir -p "$(PROFILE_REPORTS_DIR)"
	@$(PROJECT_INFRA_RUN) -m flext_infra._profile_entry \
		"$(PROFILE_REPORTS_DIR)/mod-rules.pstats" refactor ast \
		--repository-root "$(PROJECT_ROOT)"

_builtin-profile-mod-rules-report: _builtin_require_environment
	@cd "$(PROJECT_ROOT)" || exit $$?; \
		$(PROJECT_INFRA_RUN) -m flext_infra._cprofile_entry \
		"$(PROFILE_REPORTS_DIR)/mod-rules.pstats"

# Profile the cold canonical pytest execution through its thin entrypoint
# for startup diagnosis: the same persistent testmon database
# guard and environment as the bounded gate, but deliberately NOT wrapped in
# PYTEST_BOUNDED. The runner's own deadline still applies. Central collection
# children also write profiles beside their manifests and print their paths.
# The parent adapter starts profiling before runner/model/pytest imports; each
# child runs under a stdlib-only launcher, so pytest imports before any plugin
# package. The runner binds every child profile to the exact run;
# reports never combine a parent profile with the mutable latest.txt pointer.
# Public names come from make.verbs; these targets are the implementations.
_builtin-profile-test: _builtin_require_environment
	@mkdir -p "$(PROFILE_REPORTS_DIR)"
	@if [ -n "$(strip $(FILE))" ]; then \
		if [ -z "$(strip $(FILE))" ]; then printf 'ERROR: profile-test requires FILE=<repository-relative test file path>\n' >&2; exit 2; fi; \
case "$(FILE)" in /*|*..*) printf 'ERROR: FILE must stay a repository-relative path: %s\n' "$(FILE)" >&2; exit 2 ;; esac; \
if [ ! -f "$(PROJECT_ROOT)/$(FILE)" ]; then printf 'ERROR: FILE is not an existing repository file: %s\n' "$(FILE)" >&2; exit 2; fi; \
export FLEXT_PYTEST_TARGET_FILE="$(FILE)"; \
	else unset FLEXT_PYTEST_TARGET_FILE; fi; \
	set -eu; \
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
	TESTMON_DATAFILE="$$database" $(PROJECT_INFRA_RUN) -m flext_infra._pytest_entry profile \
		"$(PROFILE_REPORTS_DIR)/pytest.pstats"

_builtin-profile-test-report: _builtin_require_environment
	@cd "$(PROJECT_ROOT)" || exit $$?; \
		$(PROJECT_INFRA_RUN) -m flext_infra._cprofile_entry \
		"$(PROFILE_REPORTS_DIR)/pytest.pstats" "$(PROFILE_REPORTS_DIR)/pytest.pstats.json"


_builtin_status_import_observation: _builtin_require_interpreter
	@$(RUNTIME_PYTHON) -c 'import json, sys; exec(json.loads(sys.argv[1]))' \
		'"\nimport importlib\nimport inspect\nimport json\nimport sys\n\ndeclared = json.loads(sys.argv[2])\nnames = tuple(name for name, root in declared)\nprint(f\"runtime-import-observation executable={sys.executable!r} prefix={sys.prefix!r}\", flush=True)\ntry:\n    for name, root in declared:\n        print(f\"runtime-import-declaration module={name!r} source={root!r}\", flush=True)\n        importlib.import_module(name)\nfinally:\n    for name, module in tuple(sys.modules.items()):\n        if not any(name == package or name.startswith(package + \".\") for package in names):\n            continue\n        spec = getattr(module, \"__spec__\", None)\n        print(\n            f\"runtime-import-module name={name!r} \"\n            f\"origin={getattr(spec, \u0027origin\u0027, None)!r} \"\n            f\"file={getattr(module, \u0027__file__\u0027, None)!r} \"\n            f\"path={getattr(module, \u0027__path__\u0027, None)!r} \"\n            f\"cached={getattr(module, \u0027__cached__\u0027, None)!r}\",\n            flush=True,\n        )\n        for symbol, value in tuple(vars(module).items()):\n            function = value.__func__ if inspect.ismethod(value) else value\n            if inspect.isfunction(function):\n                print(\n                    f\"runtime-import-code module={name!r} symbol={symbol!r} \"\n                    f\"co_filename={function.__code__.co_filename!r}\",\n                    flush=True,\n                )\n"' '[["flext_core", "."]]'

_builtin_status_diagnostics: _builtin_require_environment
	@$(GITHUB_AUTH_DIAGNOSTICS)
	@printf 'profile=%s\nproject=%s\nruntime=%s\n' \
		'$(MAKE_PROFILE)' '$(PROJECT_ROOT)' '$(RUNTIME_ROOT)'
	@$(SHELL) -c 'command -v uv'
	@$(UV) --version
	@if [ -x "$(RUNTIME_PYTHON)" ]; then \
		$(UV) pip check --python "$(RUNTIME_VENV)"; \
	fi
	@git -C "$(PROJECT_ROOT)" status --short

# `docs` is a docs-only lifecycle; the actions render from make.docs.actions and never
# depend on the workspace-wide codegen conform — docs actions render their
# own outputs from the live configuration and sources.
_builtin_docs_all:
	@set -eu; \
	for action in $(DOCS_ACTIONS); do \
		mode=; \
		case "$$action" in fix|fmt) mode=--apply ;; esac; \
		$(PROJECT_FLEXT_INFRA) docs "$$action" --repository-root "$(PROJECT_ROOT)" --output-dir ".reports/docs" $$mode --projects .; \
	done

# The same lifecycle and guarded effects, with one retained profile per phase.
# Unique run directories and explicit completion markers exclude stale evidence.
_builtin-profile-docs: _builtin_require_environment
	@set -eu; \
	mkdir -p "$(PROFILE_REPORTS_DIR)"; \
	profile_dir="$$(mktemp -d "$(PROFILE_REPORTS_DIR)/docs.XXXXXX")"; \
	printf '%s\n' incomplete > "$$profile_dir/status"; \
	printf '%s\n' $(DOCS_ACTIONS) > "$$profile_dir/actions"; \
	printf '%s\n' "$$profile_dir" > "$(PROFILE_REPORTS_DIR)/docs.latest"; \
	printf 'docs profile run=%s\n' "$$profile_dir"; \
	for action in $(DOCS_ACTIONS); do \
		mode=; \
		case "$$action" in fix|fmt) mode=--apply ;; esac; \
		printf '%s\n' started > "$$profile_dir/$$action.status"; \
		$(PROJECT_INFRA_RUN) -m flext_infra._profile_entry "$$profile_dir/$$action.pstats" docs "$$action" --repository-root "$(PROJECT_ROOT)" --output-dir ".reports/docs" $$mode --projects .; \
		printf '%s\n' complete > "$$profile_dir/$$action.status"; \
	done; \
	printf '%s\n' complete > "$$profile_dir/status"

_builtin-profile-docs-report: _builtin_require_environment
	@set -eu; \
	cd "$(PROJECT_ROOT)"; \
	IFS= read -r profile_dir < "$(PROFILE_REPORTS_DIR)/docs.latest"; \
	IFS= read -r status < "$$profile_dir/status"; \
	printf 'docs profile run=%s status=%s\n' "$$profile_dir" "$$status"; \
	while IFS= read -r action; do \
		if [ ! -f "$$profile_dir/$$action.status" ]; then \
			printf 'docs phase=%s status=not-reached\n' "$$action"; \
			continue; \
		fi; \
		IFS= read -r status < "$$profile_dir/$$action.status"; \
		printf 'docs phase=%s status=%s\n' "$$action" "$$status"; \
		$(PROJECT_INFRA_RUN) -m flext_infra._cprofile_entry "$$profile_dir/$$action.pstats"; \
	done < "$$profile_dir/actions"

_builtin_clean_generated:

	@find "$(PROJECT_ROOT)" -type d \
		\( -name __pycache__ -o -name .mypy_cache -o -name .pytest_cache -o -name .ruff_cache -o -name .pyrefly_cache -o -name .benchmarks -o -name .hypothesis \) \
		-prune -exec sh -eu -c 'for target do find "$$target" -depth -delete; done' sh {} +


	@set -eu; \
	for target in "$(PROJECT_ROOT)/build" "$(PROJECT_ROOT)/dist" "$(PROJECT_ROOT)/htmlcov" "$(PROJECT_ROOT)/.reports"; do \
		if [ -e "$$target" ]; then find "$$target" -depth -delete; \
		elif [ -L "$$target" ]; then find "$$target" -depth -delete; fi; \
	done


	@set -eu; \
	for target in "$(PROJECT_ROOT)/.coverage" "$(PROJECT_ROOT)/flext-infra-codegen-transaction-journal.json.lock"; do \
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
	@$(PROJECT_FLEXT_INFRA) release run --repository-root "$(PROJECT_ROOT)" --phase plan $(if $(strip $(PR_TITLE)),--pr-title "$(PR_TITLE)")

export FLEXT_LANE_CHECKPOINT_REF := $(value LANE_CHECKPOINT_REF)
_builtin_release_version: _builtin_require_environment
	@set -eu; set --; \
	if [ -n "$$FLEXT_LANE_CHECKPOINT_REF" ]; then \
		set -- --checkpoint-ref "$$FLEXT_LANE_CHECKPOINT_REF"; \
	fi; \
	$(PROJECT_FLEXT_INFRA) release run --repository-root "$(PROJECT_ROOT)" --phase version --apply "$$@"

_builtin_release_tag: _builtin_require_environment
	@$(PROJECT_FLEXT_INFRA) release run --repository-root "$(PROJECT_ROOT)" --phase tag --apply

_builtin_release_build: _builtin_require_environment
	@$(PROJECT_FLEXT_INFRA) release run --repository-root "$(PROJECT_ROOT)" --phase build --apply

_builtin_release_publish: _builtin_require_environment
	@$(PROJECT_FLEXT_INFRA) release run --repository-root "$(PROJECT_ROOT)" --phase publish --apply $(if $(filter Y,$(INDEX)),--index)

# Generation uses the shared planner and transaction publisher for the declared
# projections. It never repairs authored declarations or resolves dependencies.
# Only `upg` resolves and rewrites locks; `fix` owns conformance repairs.
export KIND
_builtin_gen_init:
	@set -eu; set --; \
	if [ -n "$${KIND-}" ]; then set -- --kind "$$KIND"; fi; \
	$(PROJECT_FLEXT_INFRA) codegen init --repository-root "$(PROJECT_ROOT)" "$$@"


# Identity inputs are environment data, never interpolated into shell code.

export FLEXT_NEW_NAME := $(value NAME)

export FLEXT_NEW_KIND := $(value KIND)

export FLEXT_NEW_OUTPUT_ROOT := $(value OUTPUT_ROOT)

export FLEXT_NEW_PROVIDER := $(value PROVIDER)

export FLEXT_NEW_REPOSITORY_URL := $(value REPOSITORY_URL)

export FLEXT_NEW_REPOSITORY_BRANCH := $(value REPOSITORY_BRANCH)

export FLEXT_NEW_FLEXT_REPOSITORY_URL := $(value FLEXT_REPOSITORY_URL)

export FLEXT_NEW_FLEXT_REPOSITORY_REF := $(value FLEXT_REPOSITORY_REF)

export FLEXT_NEW_LICENSE := $(value LICENSE)

export FLEXT_NEW_AUTHOR_NAME := $(value AUTHOR_NAME)

export FLEXT_NEW_AUTHOR_EMAIL := $(value AUTHOR_EMAIL)

export FLEXT_NEW_UPSTREAM := $(value UPSTREAM)

export FLEXT_NEW_FLEXT_SOURCE := $(value FLEXT_SOURCE)

export FLEXT_NEW_YEAR := $(value YEAR)


_builtin-project-new: _builtin_require_interpreter
	@$(PROJECT_FLEXT_INFRA) codegen new --repository-root "$(PROJECT_ROOT)" --apply-changes --name "$$FLEXT_NEW_NAME" --kind "$$FLEXT_NEW_KIND" --output-root "$$FLEXT_NEW_OUTPUT_ROOT" --provider "$$FLEXT_NEW_PROVIDER" --repository-url "$$FLEXT_NEW_REPOSITORY_URL" --repository-branch "$$FLEXT_NEW_REPOSITORY_BRANCH" --flext-repository-url "$$FLEXT_NEW_FLEXT_REPOSITORY_URL" --flext-repository-ref "$$FLEXT_NEW_FLEXT_REPOSITORY_REF" --license "$$FLEXT_NEW_LICENSE" --author-name "$$FLEXT_NEW_AUTHOR_NAME" --author-email "$$FLEXT_NEW_AUTHOR_EMAIL" --upstream "$$FLEXT_NEW_UPSTREAM" --flext-source "$$FLEXT_NEW_FLEXT_SOURCE" --year "$$FLEXT_NEW_YEAR"


_builtin_gen_all: _builtin_require_interpreter
	@set -eu; \
	scope=all; \
	if [ "$(strip $(CI))" = "Y" ]; then scope=self; fi; \
	$(PROJECT_FLEXT_INFRA) codegen generate --root "$(PROJECT_ROOT)" --scope "$$scope"

_builtin-gen-footprint: _builtin_require_interpreter
	@$(PROJECT_FLEXT_INFRA) codegen footprint --root "$(PROJECT_ROOT)"

_builtin-gen-pyproject: _builtin_require_interpreter
	@$(PROJECT_FLEXT_INFRA) codegen generate --root "$(PROJECT_ROOT)" --what pyproject --scope self

_builtin-gen-editor: _builtin_require_interpreter
	@set -eu; \
	scope=all; \
	if [ "$(strip $(CI))" = "Y" ]; then scope=self; fi; \
	$(PROJECT_FLEXT_INFRA) codegen generate --root "$(PROJECT_ROOT)" --what editor --scope "$$scope"

_builtin-bootstrap-candidate: _builtin_require_environment
	@$(PROJECT_FLEXT_INFRA) codegen candidate-bootstrap --repository-root "$(PROJECT_ROOT)"

# Structural rewrites use mod; mod-text replays the same declared text phase
# independently when malformed Python prevents the Rope phases from loading.
# The current directory defines scope; callers never address tools directly.
_builtin_mod_apply: _builtin_require_environment
	@$(PROJECT_FLEXT_INFRA) refactor mod --repository-root "$(PROJECT_ROOT)" --apply

_builtin-census: _builtin_require_environment
	@$(PROJECT_FLEXT_INFRA) refactor census --repository-root "$(PROJECT_ROOT)" \
		--rules duplicate --rules wrong_tier --rules constant-consumers \
		--json-output "$(PROJECT_ROOT)/.reports/constants-census.json"

_builtin-census-constants: _builtin_require_environment
	@$(PROJECT_FLEXT_INFRA) refactor census --repository-root "$(PROJECT_ROOT)" \
		--families c --rules duplicate --rules wrong_tier --rules constant-consumers \
		--json-output "$(PROJECT_ROOT)/.reports/constants-census.json" --apply

_builtin_mod_text: _builtin_require_environment
	@$(PROJECT_FLEXT_INFRA) refactor mod-text --repository-root "$(PROJECT_ROOT)" --apply

_builtin_mod_text_candidate: _builtin_require_environment
	@$(PROJECT_FLEXT_INFRA) refactor mod-text-candidate --repository-root "$(PROJECT_ROOT)" --apply

# `mod` verifies committed rule-test snapshots and never rewrites them; this is
# the one explicit regeneration, whose diff is reviewed and committed.
_builtin_mod_snapshots: _builtin_require_environment
	@$(PROJECT_FLEXT_INFRA) refactor mod-snapshots --repository-root "$(PROJECT_ROOT)" --apply

# Namespace and accessor migration are the same selector-free refactor surface
# as `mod`: each public verb owns one fixed rewrite of every resolved consumer.
# The workspace profile sweeps every namespace-enabled project of the topology
# (the root repository and each declared member) in one process: the report
# aggregates per project and one project's findings never stop the sweep. A
# member profile enforces only itself.

_builtin_fix_namespace: _builtin_require_environment
	@$(PROJECT_FLEXT_INFRA) refactor namespace-enforce --repository-root "$(PROJECT_ROOT)" --projects . --apply


_builtin_fix_accessors: _builtin_require_environment
	@$(PROJECT_FLEXT_INFRA) refactor accessor-migrate --repository-root "$(PROJECT_ROOT)" --projects . --apply

# Selector-free public verbs map one-to-one to their canonical implementation;
# each implementation owns one fixed operation.
_builtin-build: _builtin_build_artifacts
_builtin-validate-lifecycle: _builtin_require_environment
	@$(PROJECT_FLEXT_INFRA) workspace validate-lifecycle --repository-root "$(PROJECT_ROOT)"
.PHONY: _builtin_require_examples
_builtin_require_examples:

	@if [ ! -f "$(PROJECT_ROOT)/examples/ex_12_flext_registry.py" ]; then \
		printf 'ERROR: examples: declared module source is missing: %s\n' "$(PROJECT_ROOT)/examples/ex_12_flext_registry.py" >&2; \
		exit 2; \
	fi


_builtin-examples: _builtin_require_environment

	@timeout --signal=TERM --kill-after=5s "120s" $(UV_RUN) python -m examples.ex_12_flext_registry

_builtin-check: _builtin_check_all
_builtin-check-pyright: _builtin_require_environment
	@$(PROJECT_FLEXT_INFRA) workspace capture-gate --repository-root "$(PROJECT_ROOT)" --gates "pyright"
_builtin-test: _builtin_test_all
_builtin-tests: _builtin_test_full_all
_builtin-test-full: _builtin_test_full_all
_builtin-test-file: _builtin_test_file_all
_builtin-file-gate: _builtin_file_gate_all
_builtin-fmt: _builtin_fmt_all
_builtin-fix: _builtin_fix_all
_builtin-fix-namespace: _builtin_fix_namespace
_builtin-fix-accessors: _builtin_fix_accessors
_builtin-fix-toolchain:
	@$(PROJECT_FLEXT_INFRA) codegen mise-repair --repository-root "$(RUNTIME_ROOT)"
_builtin-audit:
ifneq ($(CI),Y)
	@$(PROJECT_FLEXT_INFRA) workspace verify-lanes --repo-root "$(PROJECT_ROOT)"
endif
	@$(UV) pip check --python "$(RUNTIME_VENV)"
	@$(PROJECT_FLEXT_INFRA) deps verify-locks --repository-root "$(PROJECT_ROOT)"
	@$(PROJECT_FLEXT_INFRA) workspace verify-environment --repository-root "$(RUNTIME_ROOT)"
	@$(PROJECT_FLEXT_INFRA) codegen mise-proof --repository-root "$(RUNTIME_ROOT)" --uv-executable "$$(command -v $(UV))"
_builtin-status: _builtin_status_diagnostics
	@$(PROJECT_FLEXT_INFRA) workspace signing-identity --repo-root "$(PROJECT_ROOT)"
_builtin-verify-clean: _builtin_require_environment
	@$(PROJECT_FLEXT_INFRA) docs audit --repository-root "$(PROJECT_ROOT)" --output-dir ".reports/docs" --projects .
	@$(PROJECT_FLEXT_INFRA) workspace verify-clean --repo-root "$(PROJECT_ROOT)"
_builtin-docs: _builtin_docs_all
_builtin-clean: _builtin_clean_generated
_builtin-release-plan: _builtin_release_plan
_builtin-release-version: _builtin_release_version
_builtin-release-tag: _builtin_release_tag
_builtin-release-build: _builtin_release_build
_builtin-publication: _builtin_release_publish
_builtin-gen: _builtin_gen_all
_builtin-initialize: _builtin_gen_init
_builtin-mod: _builtin_mod_apply
_builtin-mod-types: _builtin_require_environment
	@$(PROJECT_FLEXT_INFRA) refactor ast --repository-root "$(PROJECT_ROOT)" --rules "bind-typing-adapter-utilities,rewire-cli-json-mapping-adapter,rewire-cli-json-list-adapter,pydantic-field-positional-default,pydantic-unset-private-attribute,codegen-conform-state-request-workspace,codegen-conform-state-workspace-request,codegen-conform-state-request,codegen-conform-state-workspace,codegen-conform-state-workspace-only,codegen-conform-state-ports" --apply
_builtin-mod-rules: _builtin_require_environment
	@$(PROJECT_FLEXT_INFRA) refactor ast --repository-root "$(PROJECT_ROOT)"
_builtin-mod-text: _builtin_mod_text
_builtin-mod-text-candidate: _builtin_mod_text_candidate
_builtin-mod-snapshots: _builtin_mod_snapshots
_builtin-waza:
	@cd "$(PROJECT_ROOT)" && waza check --no-update-check

_builtin-smells:
	@$(PROJECT_FLEXT_INFRA) check run --repository-root "$(PROJECT_ROOT)" --gates "smells"

_builtin-duplication:
	@$(PROJECT_FLEXT_INFRA) check run --repository-root "$(PROJECT_ROOT)" --gates "duplication"
_builtin-sonarcloud-sync: _builtin_sonarcloud_sync_all
_builtin-sonarcloud-issues: _builtin_sonarcloud_issues_all
