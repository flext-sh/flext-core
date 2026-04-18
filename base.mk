# =============================================================================
# FLEXT BASE MAKEFILE - Shared patterns for all FLEXT projects
# =============================================================================
# Usage: Set PROJECT_NAME before including: include ../base.mk
# Silent by default. Use VERBOSE=1 for detailed output.
# =============================================================================

# === CONFIGURATION (override before include) ===
PROJECT_NAME ?= unnamed
PYTHON_VERSION ?= 3.13
SRC_DIR ?= src
TESTS_DIR ?= tests
DOCSTRING_MIN ?= 80
COMPLEXITY_MAX ?= 10
CORE_STACK ?= python
PYTEST_ARGS ?= 
DIAG ?= 0
CHECK_GATES ?= 
VALIDATE_GATES ?= 
DOCS_PHASE ?= all
FIX ?= 
PR_ACTION ?= status
PR_BASE ?= main
PR_HEAD ?= 
PR_NUMBER ?= 
PR_TITLE ?= 
PR_BODY ?= 
PR_DRAFT ?= 0
PR_MERGE_METHOD ?= squash
PR_AUTO ?= 0
PR_DELETE_BRANCH ?= 0
PR_CHECKS_STRICT ?= 0
PR_RELEASE_ON_MERGE ?= 1
FILE ?= 
FILES ?= 
CHANGED_ONLY ?= 
MATCH ?= 
RUFF_ARGS ?= 
PYRIGHT_ARGS ?= 
CHECK_ONLY ?= 
FAIL_FAST ?= 
VERBOSE ?= 


PYTEST_REPORT_ARGS := -ra --durations=25 --durations-min=0.001 --tb=short
PYTEST_DIAG_ARGS := -rA --durations=0 --tb=long --showlocals
PYTEST_REPORTS_DIR ?= .reports/tests

# === WORKSPACE/STANDALONE DETECTION ===
BASE_MK_DIR := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
PROJECT_ROOT := $(CURDIR)

ifeq ($(FLEXT_STANDALONE),1)
FLEXT_MODE := standalone
else
# Pure Make detection: if base.mk lives in a parent dir, we are inside a workspace.
# No Python dependency — shell/Make only until venv is ready.
ifneq ($(BASE_MK_DIR),$(PROJECT_ROOT))
FLEXT_MODE := workspace
else
FLEXT_MODE := standalone
endif
endif

ifeq ($(FLEXT_MODE),workspace)
WORKSPACE_ROOT := $(BASE_MK_DIR)
WORKSPACE_VENV := $(WORKSPACE_ROOT)/.venv
ifeq ($(wildcard $(WORKSPACE_VENV)),)
ACTIVE_VENV := $(PROJECT_ROOT)/.venv
export POETRY_VIRTUALENVS_PATH := $(PROJECT_ROOT)
export POETRY_VIRTUALENVS_IN_PROJECT := true
export POETRY_VIRTUALENVS_CREATE := true
else
ACTIVE_VENV := $(WORKSPACE_VENV)
export POETRY_VIRTUALENVS_PATH := $(WORKSPACE_ROOT)
export POETRY_VIRTUALENVS_IN_PROJECT := false
export POETRY_VIRTUALENVS_CREATE := false
endif
else
WORKSPACE_ROOT := $(PROJECT_ROOT)
ACTIVE_VENV := $(PROJECT_ROOT)/.venv
export POETRY_VIRTUALENVS_PATH := $(PROJECT_ROOT)
export POETRY_VIRTUALENVS_IN_PROJECT := true
export POETRY_VIRTUALENVS_CREATE := true
endif

export PYTHON_KEYRING_BACKEND := keyring.backends.null.Keyring

VENV_PYTHON := $(ACTIVE_VENV)/bin/python
VENV_ACTIVATE := source $(ACTIVE_VENV)/bin/activate
export VIRTUAL_ENV := $(ACTIVE_VENV)

# Go tooling/caches (zero-settings defaults for make targets).
GO_TOOLS_BIN ?= $(WORKSPACE_ROOT)/.tools/bin
GO_CACHE_ROOT ?= $(if $(XDG_CACHE_HOME),$(XDG_CACHE_HOME)/flext/go,$(WORKSPACE_ROOT)/.cache/go)
GO_BUILD_CACHE ?= $(GO_CACHE_ROOT)/build
GO_MOD_CACHE ?= $(GO_CACHE_ROOT)/mod
GO_LINT_CACHE ?= $(GO_CACHE_ROOT)/golangci-lint
GOLANGCI_LINT_CMD ?= golangci-lint
GOLANGCI_LINT_INSTALL ?= go install github.com/golangci/golangci-lint/v2/cmd/golangci-lint@latest

export GOBIN ?= $(GO_TOOLS_BIN)
export GOCACHE ?= $(GO_BUILD_CACHE)
export GOMODCACHE ?= $(GO_MOD_CACHE)
export GOLANGCI_LINT_CACHE ?= $(GO_LINT_CACHE)
export PATH := $(GO_TOOLS_BIN):$(ACTIVE_VENV)/bin:$(PATH)

# Poetry command (uses workspace venv automatically)
POETRY := poetry

# Quality tool (flext-quality with fallback)
QUALITY_CMD ?= flext-quality
QUALITY_AVAILABLE := $(shell command -v $(QUALITY_CMD) 2>/dev/null)
DMPY_SOCKET := .dmypy/socket.$(PROJECT_NAME)
PYRIGHT_PIDFILE := .pyright/daemon.pid
PYRIGHT_LOG := .pyright/daemon.log

# Export for subprocesses
export PROJECT_NAME PYTHON_VERSION
export FLEXT_ROOT := $(WORKSPACE_ROOT)

# === SILENT MODE ===
Q := @
ifdef VERBOSE
Q :=
endif

# === CACHE ===
LINT_CACHE_DIR := .lint-cache
CACHE_TIMEOUT := 300
BASE_INFRA_WORKSPACE := env -u PYTHONPATH -u MYPYPATH PYTHONPATH="$(WORKSPACE_ROOT)/flext-infra/src" $(if $(wildcard $(VENV_PYTHON)),$(VENV_PYTHON),python) -m flext_infra workspace

$(LINT_CACHE_DIR):
	$(Q)mkdir -p $(LINT_CACHE_DIR)

# === SIMPLE VERB SURFACE ===
.PHONY: help boot build check scan fmt docs test val clean pr _preflight daemon-start-mypy daemon-stop-mypy daemon-status-mypy daemon-start-pyright daemon-stop-pyright daemon-status-pyright daemon-start daemon-stop daemon-status daemon-restart
STANDARD_VERBS := boot build check scan fmt docs test val clean pr
$(STANDARD_VERBS): _preflight

define ENFORCE_WORKSPACE_VENV
if [ "$(FLEXT_MODE)" = "workspace" ]; then \
	if [ -d "$(WORKSPACE_ROOT)/.venv" ]; then \
		if [ -d ".venv" ] && [ "$(CURDIR)" != "$(WORKSPACE_ROOT)" ]; then \
			echo "[preflight] Removing local .venv in $(CURDIR) (workspace venv enforced)"; \
			rm -rf .venv; \
			if [ -d ".venv" ]; then \
				echo "ERROR: [preflight] Unable to remove local .venv in $(CURDIR)"; \
				exit 1; \
			fi; \
		fi; \
	elif [ "$(CURDIR)" = "$(WORKSPACE_ROOT)" ]; then \
		echo "ERROR: [preflight] Workspace venv not found. Run 'make boot' at workspace root."; \
		exit 1; \
	elif [ "$(filter boot,$(MAKECMDGOALS))" != "boot" ] && [ ! -d "$(ACTIVE_VENV)" ]; then \
		echo "ERROR: [preflight] No venv found (workspace or local). Run 'make boot' in $(PROJECT_NAME)."; \
		exit 1; \
	else \
		echo "INFO: [preflight] Using project-local venv for $(PROJECT_NAME) (workspace venv not found)."; \
	fi; \
elif [ "$(FLEXT_MODE)" = "standalone" ]; then \
	echo "INFO: [preflight] Running in standalone mode (workspace features unavailable)."; \
elif [ "$(filter boot,$(MAKECMDGOALS))" != "boot" ] && [ ! -d "$(ACTIVE_VENV)" ]; then \
	echo "ERROR: [preflight] No venv found at $(ACTIVE_VENV). Run 'make boot' in $(PROJECT_NAME)."; \
	exit 1; \
fi
endef

define AUTO_SYNC_BASE_AND_SCRIPTS
if [ "$(FLEXT_MODE)" = "workspace" ] && [ "$(CURDIR)" != "$(WORKSPACE_ROOT)" ]; then \
	$(BASE_INFRA_WORKSPACE) sync \
		--workspace "$(CURDIR)" --canonical-root "$(WORKSPACE_ROOT)" --apply; \
elif [ "$(FLEXT_MODE)" = "standalone" ]; then \
	echo "INFO: [preflight] Standalone mode: skipping workspace dependency sync."; \
fi
endef

_preflight: ## Preflight: sync base.mk and enforce venv contract
	$(Q)$(AUTO_SYNC_BASE_AND_SCRIPTS)
	$(Q)$(ENFORCE_WORKSPACE_VENV)

PROJECT_INFRA_HOME := $(WORKSPACE_ROOT)/flext-infra
ifeq ($(wildcard $(PROJECT_INFRA_HOME)/src/flext_infra),)
PROJECT_INFRA_HOME := $(PROJECT_ROOT)
endif
PROJECT_INFRA_SRC := $(PROJECT_INFRA_HOME)/src
PROJECT_INFRA_BOOT := env -u PYTHONPATH -u MYPYPATH PYTHONPATH="$(PROJECT_INFRA_SRC)" $(POETRY) run python -m flext_infra
PROJECT_INFRA_ROOT := env -u PYTHONPATH -u MYPYPATH PYTHONPATH="$(PROJECT_INFRA_SRC)" $(VENV_PYTHON) -m flext_infra
PROJECT_INFRA_CHECK := FLEXT_WORKSPACE_ROOT="$(WORKSPACE_ROOT)" $(PROJECT_INFRA_ROOT) check
PROJECT_INFRA_DEPS := FLEXT_WORKSPACE_ROOT="$(WORKSPACE_ROOT)" $(PROJECT_INFRA_BOOT) deps
PROJECT_INFRA_DOCS := FLEXT_WORKSPACE_ROOT="$(WORKSPACE_ROOT)" $(PROJECT_INFRA_ROOT) docs
PROJECT_INFRA_GITHUB := FLEXT_WORKSPACE_ROOT="$(WORKSPACE_ROOT)" $(PROJECT_INFRA_ROOT) github
PROJECT_INFRA_VALIDATE := FLEXT_WORKSPACE_ROOT="$(WORKSPACE_ROOT)" $(PROJECT_INFRA_ROOT) validate

help: ## Show commands
	$(Q)echo "================================================"
	$(Q)echo "  $(PROJECT_NAME)"
	$(Q)echo "================================================"
	$(Q)echo ""
	$(Q)echo "Core verbs:"

	$(Q)printf "  %-14s %s\n" "boot" "Install dependencies and hooks"

	$(Q)printf "  %-14s %s\n" "build" "Build distributable artifacts"

	$(Q)printf "  %-14s %s\n" "check" "Run lint gates (CHECK_GATES= to select)"

	$(Q)printf "  %-14s %s\n" "scan" "Run all security checks"

	$(Q)printf "  %-14s %s\n" "fmt" "Run all formatting"

	$(Q)printf "  %-14s %s\n" "docs" "Build docs (DOCS_PHASE= to select)"

	$(Q)printf "  %-14s %s\n" "test" "Run pytest (PYTEST_ARGS= for options)"

	$(Q)printf "  %-14s %s\n" "val" "Run validate gates (FIX=1 to auto-fix)"

	$(Q)printf "  %-14s %s\n" "clean" "Clean build/test/type artifacts"

	$(Q)echo ""
	$(Q)echo "Daemon management:"

	$(Q)printf "  %-16s %s\n" "daemon-start" "Start all daemons (mypy + pyright)"

	$(Q)printf "  %-16s %s\n" "daemon-stop" "Stop all daemons"

	$(Q)printf "  %-16s %s\n" "daemon-status" "Show status of all daemons"

	$(Q)printf "  %-16s %s\n" "daemon-restart" "Restart all daemons"

	$(Q)echo "  Also: daemon-{start,stop,status}-{mypy,pyright}"
	$(Q)echo ""
	$(Q)echo "Selectors and options:"

	$(Q)echo "  CHECK_GATES=lint,format,pyrefly,mypy,pyright,security,markdown,go,type"

	$(Q)echo "  VALIDATE_GATES=complexity,docstring"

	$(Q)echo "  FILE=src/foo.py             Single file for check/fmt/test"

	$(Q)echo "  FILES=\"a.py b.py\"          Multiple files for check/fmt/test"

	$(Q)echo "  CHANGED_ONLY=1              Git-changed Python files for check"

	$(Q)echo "  CHECK_ONLY=1                Dry-run format/check (no writes)"

	$(Q)echo "  RUFF_ARGS=\"--select E501\"   Extra args for ruff check"

	$(Q)echo "  PYRIGHT_ARGS=\"--level basic\" Extra args for pyright"

	$(Q)echo "  PYTEST_ARGS=\"-k expr\"       Extra pytest args"

	$(Q)echo "  MATCH=test_name             Alias for pytest -k"

	$(Q)echo "  FAIL_FAST=1                 Add -x to pytest"

	$(Q)echo "  DIAG=1                      Emit extended pytest diagnostics"

	$(Q)echo "  DOCS_PHASE=all|generate|fix|audit|build|validate"

	$(Q)echo "  FIX=1                       Auto-fix supported gates"

	$(Q)echo "  VERBOSE=1                   Show executed commands"

	$(Q)echo ""
	$(Q)echo "PR variables:"

	$(Q)echo "  PR_ACTION=status|create|view|checks|merge|close"

	$(Q)echo "  PR_BASE=main  PR_HEAD=<branch>  PR_NUMBER=<id>"

	$(Q)echo "  PR_TITLE='...'  PR_BODY='...'  PR_DRAFT=0|1"

	$(Q)echo "  PR_MERGE_METHOD=squash|merge|rebase  PR_AUTO=0|1"

	$(Q)echo "  PR_DELETE_BRANCH=0|1  PR_CHECKS_STRICT=0|1"

	$(Q)echo "  PR_RELEASE_ON_MERGE=0|1"


boot: ## Complete setup
	$(Q)if [ "$(CORE_STACK)" = "go" ]; then \
		go mod download; \
		go mod tidy; \
		exit 0; \
	fi
	$(Q)$(PROJECT_INFRA_DEPS) path-sync --mode auto --apply --workspace "$(CURDIR)"
	$(Q)$(PROJECT_INFRA_DEPS) internal-sync --workspace "$(CURDIR)"
	$(Q)$(POETRY) lock
	$(Q)$(POETRY) install --all-extras --all-groups
	$(Q)if git rev-parse --git-dir >/dev/null 2>&1; then \
		$(POETRY) run pre-commit install; \
	else \
		echo "INFO: skipping pre-commit install (no git repository)"; \
	fi

build: ## Build distributable artifacts
	$(Q)if [ "$(CORE_STACK)" = "go" ]; then \
		mkdir -p .reports/build; \
		build_start=$$(date +%s); \
		go build -o .reports/build/$(PROJECT_NAME) ./...; \
		echo "Build complete: $(PROJECT_NAME) ($$(($$(date +%s) - $$build_start))s)"; \
		exit 0; \
	fi
	$(Q)build_start=$$(date +%s); \
	$(POETRY) build; \
	echo "Build complete: $(PROJECT_NAME) ($$(($$(date +%s) - $$build_start))s)"

check: ## Run lint gates (CHECK_GATES=lint,format,pyrefly,mypy,pyright,security,markdown,go,type to select)
	$(Q)if [ "$(CORE_STACK)" = "go" ]; then \
		gates="$(CHECK_GATES)"; \
		if [ -n "$$gates" ]; then \
			for g in $$(echo "$$gates" | tr ',' ' '); do \
				case "$$g" in \
					lint|format|pyrefly|mypy|pyright|security|markdown|go|type) ;; \
					*) echo "ERROR: unknown CHECK_GATES value '$$g' (allowed: lint,format,pyrefly,mypy,pyright,security,markdown,go,type)"; exit 2;; \
				esac; \
			done; \
		else \
			gates="lint,format,security,markdown,go"; \
		fi; \
		gates=$$(echo "$$gates" | tr ',' ' ' | sed 's/\btype\b/go/g' | tr ' ' ','); \
		if echo "$$gates" | grep -qw lint; then \
			mkdir -p "$(GO_TOOLS_BIN)" "$(GO_BUILD_CACHE)" "$(GO_MOD_CACHE)" "$(GO_LINT_CACHE)"; \
			if ! command -v $(GOLANGCI_LINT_CMD) >/dev/null 2>&1; then \
				echo "INFO: installing golangci-lint into $(GO_TOOLS_BIN)"; \
				$(GOLANGCI_LINT_INSTALL) || { echo "FAIL: lint (golangci-lint install)"; exit 1; }; \
			fi; \
			lint_log=$$(mktemp); \
			if ! $(GOLANGCI_LINT_CMD) run >"$$lint_log" 2>&1; then \
				if grep -q "used to build golangci-lint is lower than the targeted Go version" "$$lint_log"; then \
					echo "INFO: refreshing golangci-lint for current Go toolchain"; \
					$(GOLANGCI_LINT_INSTALL) || { cat "$$lint_log"; rm -f "$$lint_log"; echo "FAIL: lint"; exit 1; }; \
					if ! $(GOLANGCI_LINT_CMD) run; then rm -f "$$lint_log"; echo "FAIL: lint"; exit 1; fi; \
				else \
					cat "$$lint_log"; rm -f "$$lint_log"; echo "FAIL: lint"; exit 1; \
				fi; \
			fi; \
			rm -f "$$lint_log"; \
		fi; \
		if echo "$$gates" | grep -qw format; then \
			if [ -n "$$(find . -type f -name '*.go' ! -path './.git/*' ! -path './vendor/*' ! -path './.cache/*')" ]; then \
				gofmt_diff=$$(find . -type f -name '*.go' ! -path './.git/*' ! -path './vendor/*' ! -path './.cache/*' -print0 | xargs -0 gofmt -l); \
				if [ -n "$$gofmt_diff" ]; then \
					echo "FAIL: gofmt"; \
					printf '%s\n' "$$gofmt_diff"; \
					exit 1; \
				fi; \
			fi; \
		fi; \
		if echo "$$gates" | grep -qw security; then \
			gosec ./... || { echo "FAIL: security"; exit 1; }; \
		fi; \
		if echo "$$gates" | grep -qw markdown; then \
			if [ "$(CURDIR)" = "$(WORKSPACE_ROOT)" ] && [ -n "$(ALL_PROJECTS)" ]; then \
				md_roots=". $(ALL_PROJECTS)"; \
			else \
				md_roots="."; \
			fi; \
			md_files=$$(for md_root in $$md_roots; do \
				[ -d "$$md_root" ] || continue; \
				if git -C "$$md_root" rev-parse --git-dir >/dev/null 2>&1; then \
					md_prefix=""; \
					if [ "$$md_root" != "." ]; then md_prefix="$$md_root/"; fi; \
					git -C "$$md_root" ls-files -- '*.md' ':!vendor/' | sed "s#^#$$md_prefix#"; \
					git -C "$$md_root" ls-files --others --exclude-standard -- '*.md' ':!vendor/' | sed "s#^#$$md_prefix#"; \
				else \
					find "$$md_root" -type f -name '*.md' ! -path '*/.git/*' ! -path '*/.reports/*' ! -path '*/reports/*' ! -path '*/.venv/*' ! -path '*/vendor/*' ! -path '*/node_modules/*' ! -path '*/dist/*' ! -path '*/build/*'; \
				fi; \
			done); \
			md_files=$$(printf '%s\n' "$$md_files" | awk 'NF' | while IFS= read -r f; do [ -f "$$f" ] && printf '%s\n' "$$f"; done | sort -u); \
			md_config=""; \
			if [ -f "$(WORKSPACE_ROOT)/.markdownlint.json" ]; then \
				md_config="--config $(WORKSPACE_ROOT)/.markdownlint.json"; \
			elif [ -f ".markdownlint.json" ]; then \
				md_config="--config .markdownlint.json"; \
			fi; \
			if [ -n "$$md_files" ]; then echo "$$md_files" | xargs -r markdownlint $$md_config || { echo "FAIL: markdown"; exit 1; }; fi; \
		fi; \
		if echo "$$gates" | grep -qw go; then \
			go vet ./... || { echo "FAIL: go"; exit 1; }; \
		fi; \
		python_gates=$$(printf '%s\n' "$$gates" | tr ',' '\n' | grep -E '^(pyrefly|mypy|pyright)$$' | tr '\n' ',' | sed 's/,$$//'); \
		if [ -n "$$python_gates" ]; then \
			project_key="$(PROJECT_NAME)"; \
			if [ "$(CURDIR)" = "$(WORKSPACE_ROOT)" ]; then \
				project_key="."; \
			fi; \
			$(PROJECT_INFRA_CHECK) run --gates "$$python_gates" --reports-dir "$(CURDIR)/.reports/check" --projects "$$project_key" $(if $(filter 1,$(FIX)),$(if $(filter 1,$(CHECK_ONLY)),,--fix),) $(if $(filter 1,$(CHECK_ONLY)),--check-only,) $(if $(RUFF_ARGS),--ruff-args "$(RUFF_ARGS)",) $(if $(PYRIGHT_ARGS),--pyright-args "$(PYRIGHT_ARGS)",) || exit $$?; \
		fi; \
		exit 0; \
	fi
	$(Q)gates="$(CHECK_GATES)"; \
	if [ -n "$$gates" ]; then \
		for g in $$(echo "$$gates" | tr ',' ' '); do \
			case "$$g" in \
				lint|format|pyrefly|mypy|pyright|security|markdown|go|type) ;; \
				*) echo "ERROR: unknown CHECK_GATES value '$$g' (allowed: lint,format,pyrefly,mypy,pyright,security,markdown,go,type)"; exit 2;; \
			esac; \
		done; \
	else \
		gates="lint,format,pyrefly,mypy,pyright,security,markdown,go"; \
	fi; \
	gates=$$(echo "$$gates" | tr ',' ' ' | sed 's/\btype\b/pyrefly/g' | tr ' ' ','); \
	_files=""; \
	if [ -n "$(FILES)" ]; then _files="$(FILES)"; fi; \
	if [ -n "$(FILE)" ]; then \
		if [ -n "$$_files" ]; then _files="$$_files $(FILE)"; \
		else _files="$(FILE)"; fi; \
	fi; \
	if [ "$(CHANGED_ONLY)" = "1" ]; then \
		_files=$$(git diff --name-only HEAD -- '*.py' 2>/dev/null | tr '\n' ' '); \
	fi; \
	if [ -n "$$_files" ]; then \
		if [ -z "$(CHECK_GATES)" ]; then gates="lint,format,pyrefly,mypy,pyright"; fi; \
		unsupported_gates=$$(printf '%s\n' "$$gates" | tr ',' '\n' | grep -E '^(security|markdown|go)$$' || true); \
		if [ -n "$$unsupported_gates" ]; then \
			echo "ERROR: FILE/FILES/CHANGED_ONLY fast-path only supports lint,format,pyrefly,mypy,pyright"; \
			exit 2; \
		fi; \
		echo "Fast-path check: $$_files"; \
		status=0; \
		case ",$$gates," in \
			*,lint,*) env -u PYTHONPATH -u MYPYPATH $(POETRY) run ruff check $$_files $(RUFF_ARGS) $(if $(filter 1,$(FIX)),$(if $(filter 1,$(CHECK_ONLY)),,--fix),) || status=$$?;; \
		esac; \
		case ",$$gates," in \
			*,format,*) env -u PYTHONPATH -u MYPYPATH $(POETRY) run ruff format $$_files $(if $(filter 1,$(CHECK_ONLY)),--check,--quiet) || status=$$?;; \
		esac; \
		case ",$$gates," in \
			*,pyright,*) env -u PYTHONPATH -u MYPYPATH $(POETRY) run pyright $$_files $(PYRIGHT_ARGS) || status=$$?;; \
		esac; \
		case ",$$gates," in \
			*,pyrefly,*) env -u PYTHONPATH -u MYPYPATH $(POETRY) run pyrefly check $$_files || status=$$?;; \
		esac; \
		case ",$$gates," in \
			*,mypy,*) env -u PYTHONPATH -u MYPYPATH $(POETRY) run mypy $$_files || status=$$?;; \
		esac; \
		exit $$status; \
	fi; \
	project_key="$(PROJECT_NAME)"; \
	if [ "$(CURDIR)" = "$(WORKSPACE_ROOT)" ]; then \
		project_key="."; \
	fi; \
	$(PROJECT_INFRA_CHECK) run --gates "$$gates" --reports-dir "$(CURDIR)/.reports/check" --projects "$$project_key" $(if $(filter 1,$(FIX)),$(if $(filter 1,$(CHECK_ONLY)),,--fix),) $(if $(filter 1,$(CHECK_ONLY)),--check-only,) $(if $(RUFF_ARGS),--ruff-args "$(RUFF_ARGS)",) $(if $(PYRIGHT_ARGS),--pyright-args "$(PYRIGHT_ARGS)",); \
	exit $$?

scan: ## Run all security checks
	$(Q)if [ "$(CORE_STACK)" = "go" ]; then \
		gosec ./...; \
		exit 0; \
	fi; \
	project_key="$(PROJECT_NAME)"; \
	if [ "$(CURDIR)" = "$(WORKSPACE_ROOT)" ]; then \
		project_key="."; \
	fi; \
	$(PROJECT_INFRA_CHECK) run \
		--workspace "$(WORKSPACE_ROOT)" \
		--gates "security" \
		--reports-dir "$(CURDIR)/.reports/scan" \
		--projects "$$project_key"; \
	exit $$?

fmt: ## Run code formatting (ruff/gofmt + markdownlint on tracked files)
	$(Q)if [ "$(CORE_STACK)" = "go" ]; then \
		go_files=$$(find . -type f -name '*.go' ! -path './.git/*' ! -path './vendor/*' ! -path './.cache/*'); \
		if [ -n "$$go_files" ]; then \
			printf '%s\n' "$$go_files" | xargs gofmt -w; \
			if command -v goimports >/dev/null 2>&1; then \
				printf '%s\n' "$$go_files" | xargs goimports -w; \
			fi; \
		fi; \
	else \
		_fmt_target="."; \
		_fmt_files=""; \
		if [ -n "$(FILES)" ]; then _fmt_files="$(FILES)"; fi; \
		if [ -n "$(FILE)" ]; then \
			if [ -n "$$_fmt_files" ]; then _fmt_files="$$_fmt_files $(FILE)"; \
			else _fmt_files="$(FILE)"; fi; \
		fi; \
		if [ -n "$$_fmt_files" ]; then _fmt_target="$$_fmt_files"; fi; \
		if [ "$(CHECK_ONLY)" = "1" ]; then \
			$(POETRY) run ruff format $$_fmt_target --check; \
		else \
			$(POETRY) run ruff format $$_fmt_target --quiet; \
		fi; \
		if [ -f go.mod ]; then \
			go_files=$$(find . -type f -name '*.go' ! -path './.git/*' ! -path './vendor/*' ! -path './.cache/*'); \
			if [ -n "$$go_files" ]; then \
				printf '%s\n' "$$go_files" | xargs gofmt -w; \
			fi; \
		fi; \
	fi
	$(Q)if [ "$(CURDIR)" = "$(WORKSPACE_ROOT)" ] && [ -n "$(ALL_PROJECTS)" ]; then \
		md_roots=". $(ALL_PROJECTS)"; \
	else \
		md_roots="."; \
	fi; \
	md_files=$$(for md_root in $$md_roots; do \
		[ -d "$$md_root" ] || continue; \
		if git -C "$$md_root" rev-parse --git-dir >/dev/null 2>&1; then \
			md_prefix=""; \
			if [ "$$md_root" != "." ]; then md_prefix="$$md_root/"; fi; \
			git -C "$$md_root" ls-files -- '*.md' ':!vendor/' | sed "s#^#$$md_prefix#"; \
			git -C "$$md_root" ls-files --others --exclude-standard -- '*.md' ':!vendor/' | sed "s#^#$$md_prefix#"; \
		else \
			find "$$md_root" -type f -name '*.md' ! -path '*/.git/*' ! -path '*/.reports/*' ! -path '*/.venv/*' ! -path '*/vendor/*' ! -path '*/node_modules/*' ! -path '*/dist/*' ! -path '*/build/*'; \
		fi; \
	done); \
	md_files=$$(printf '%s\n' "$$md_files" | awk 'NF' | while IFS= read -r f; do [ -f "$$f" ] && printf '%s\n' "$$f"; done | sort -u); \
	if [ -n "$$md_files" ]; then \
		md_config=""; \
		if [ -f "$(WORKSPACE_ROOT)/.markdownlint.json" ]; then \
			md_config="--config $(WORKSPACE_ROOT)/.markdownlint.json"; \
		elif [ -f ".markdownlint.json" ]; then \
			md_config="--config .markdownlint.json"; \
		fi; \
		echo "$$md_files" | xargs -r markdownlint --fix $$md_config; \
	fi
	$(Q)echo "Format complete: $(PROJECT_NAME)"

docs: ## Build docs
	$(Q)if python3 -c "import flext_infra.docs" >/dev/null 2>&1; then \
		echo "PROJECT=$(PROJECT_NAME) PHASE=sync RESULT=OK REASON=docs-module-available"; \
	else \
		echo "PROJECT=$(PROJECT_NAME) PHASE=sync RESULT=FAIL REASON=docs-module-missing"; \
		exit 1; \
	fi
	$(Q)if [ "$(DOCS_PHASE)" = "all" ]; then \
		phases="generate fix audit build validate"; \
		all_mode=1; \
	else \
		phases="$(DOCS_PHASE)"; \
		all_mode=0; \
	fi; \
	for phase in $$phases; do \
		case "$$phase" in \
			audit) subcmd="$(PROJECT_INFRA_DOCS) audit"; extra="--strict 1" ;; \
			fix) subcmd="$(PROJECT_INFRA_DOCS) fix"; extra="$(if $(filter 1,$(FIX)),--apply,)" ;; \
			build) subcmd="$(PROJECT_INFRA_DOCS) build"; extra="" ;; \
			generate) subcmd="$(PROJECT_INFRA_DOCS) generate"; extra="--apply" ;; \
			validate) subcmd="$(PROJECT_INFRA_DOCS) validate"; extra="$(if $(filter 1,$(FIX)),--apply,)" ;; \
				*) echo "ERROR: invalid DOCS_PHASE=$$phase (allowed: all|generate|fix|audit|build|validate)"; exit 2 ;; \
			esac; \
		if [ "$$phase" = "fix" ] && [ "$$all_mode" = "1" ]; then extra="--apply"; fi; \
		cmd="$$subcmd --workspace . --output-dir .reports/docs"; \
		if [ -n "$$extra" ]; then cmd="$$cmd $$extra"; fi; \
		eval $$cmd || exit $$?; \
	done

test: ## Run pytest only
	$(Q)if [ "$(CORE_STACK)" = "go" ]; then \
		go test -v -race -coverprofile=coverage.out -covermode=atomic ./...; \
		go tool cover -func=coverage.out; \
	else \
		_files=""; \
		if [ -n "$(FILES)" ]; then _files="$(FILES)"; fi; \
		if [ -n "$(FILE)" ]; then \
			if [ -n "$$_files" ]; then _files="$$_files $(FILE)"; \
			else _files="$(FILE)"; fi; \
		fi; \
		_pytest_run="$(TESTS_DIR)"; \
		if [ -n "$$_files" ]; then _pytest_run="$$_files"; fi; \
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
	interrupted=0; \
	echo "$(POETRY) run pytest $$_pytest_run $(PYTEST_REPORT_ARGS) $(if $(filter 1,$(DIAG)),$(PYTEST_DIAG_ARGS),) -p no:metadata --junitxml=$$junit_file --cov --cov-report=xml:$$coverage_file $(if $(filter 1,$(DIAG)),-vv,-q) $$_all_pytest_args" > "$$command_file"; \
	trap 'interrupted=1; trap "" INT TERM' INT TERM; \
	$(POETRY) run pytest $$_pytest_run \
		$(PYTEST_REPORT_ARGS) \
		$(if $(filter 1,$(DIAG)),$(PYTEST_DIAG_ARGS),) \
		-p no:metadata \
		--junitxml="$$junit_file" \
		--cov --cov-report=xml:$$coverage_file \
		$(if $(filter 1,$(DIAG)),-vv,-q) $$_all_pytest_args 2>&1 | tee "$$log_file"; \
	rc=$${PIPESTATUS[0]}; \
	if [ "$$interrupted" = "1" ]; then rc=130; fi; \
	if [ -f "$$junit_file" ]; then \
		tests=$$(grep -Eo 'tests="[0-9]+"' "$$junit_file" | head -n 1 | tr -dc '0-9'); \
		failures=$$(grep -Eo 'failures="[0-9]+"' "$$junit_file" | head -n 1 | tr -dc '0-9'); \
		errors=$$(grep -Eo 'errors="[0-9]+"' "$$junit_file" | head -n 1 | tr -dc '0-9'); \
		skipped=$$(grep -Eo 'skipped="[0-9]+"' "$$junit_file" | head -n 1 | tr -dc '0-9'); \
		duration=$$(grep -Eo 'time="[0-9.]+"' "$$junit_file" | head -n 1 | sed -E 's/time="([0-9.]+)"/\1/'); \
		tests=$${tests:-0}; failures=$${failures:-0}; errors=$${errors:-0}; skipped=$${skipped:-0}; duration=$${duration:-0}; \
		passed=$$((tests - failures - errors - skipped)); \
		if [ $$passed -lt 0 ]; then passed=0; fi; \
		printf 'junit=%s\ncoverage=%s\ntotal=%s\npassed=%s\nfailed=%s\nerrors=%s\nskipped=%s\nduration_seconds=%s\n' \
			"$$junit_file" "$$coverage_file" "$$tests" "$$passed" "$$failures" "$$errors" "$$skipped" "$$duration" > "$$summary_file"; \
	else \
		echo "junit=not-generated" > "$$summary_file"; \
		echo "coverage=$$coverage_file" >> "$$summary_file"; \
		echo "total=0" >> "$$summary_file"; \
		echo "passed=0" >> "$$summary_file"; \
		echo "failed=0" >> "$$summary_file"; \
		echo "errors=0" >> "$$summary_file"; \
		echo "skipped=0" >> "$$summary_file"; \
		echo "duration_seconds=0" >> "$$summary_file"; \
	fi; \
	counts_file="$$report_dir/counts.env"; \
	$(PROJECT_INFRA_VALIDATE) pytest-diag \
		--junit "$$junit_file" --log "$$log_file" \
		--failed "$$failed_file" --errors "$$errors_file" \
		--warnings "$$warnings_file" --slowest "$$slowest_file" \
		--skips "$$skips_file" > "$$counts_file"; \
	. "$$counts_file"; \
	if [ "$$rc" -eq 130 ] || [ "$$interrupted" = "1" ]; then run_state="INTERRUPTED"; else run_state="COMPLETED"; fi; \
	echo "================================================" >&2; \
	echo "DIAG $$run_state | failed=$$failed_count errors=$$error_count warnings=$$warning_count skipped=$$skipped_count" >&2; \
	echo "================================================" >&2; \
	echo "Top test durations (from $$slowest_file):" >&2; \
	if [ -s "$$slowest_file" ]; then awk 'NR<=10 {print}' "$$slowest_file" >&2; \
	else echo "(none)" >&2; fi; \
	echo "Error trace excerpt (from $$errors_file):" >&2; \
	if [ -s "$$errors_file" ]; then awk 'NR<=40 {print}' "$$errors_file" >&2; \
	else echo "(none)" >&2; fi; \
	ln -sfn "$$run_id" "$(PYTEST_REPORTS_DIR)/latest"; \
	echo "Reports: $$report_dir (latest: $(PYTEST_REPORTS_DIR)/latest)" >&2; \
	echo "Details: $$summary_file | $$failed_file | $$errors_file | $$warnings_file | $$slowest_file | $$skips_file | $$log_file" >&2; \
	exit $$rc; \
	fi

val: ## Run validate gates (VALIDATE_GATES=complexity,docstring to select, FIX=1)
	$(Q)if [ "$(CORE_STACK)" = "go" ]; then \
		if [ "$(FIX)" = "1" ]; then \
			$(MAKE) fmt; \
		fi; \
		go mod verify; \
		exit 0; \
	fi
	$(Q)if [ -n "$(FIX)" ] && [ "$(FIX)" != "1" ]; then \
		echo "ERROR: FIX must be empty or 1, got '$(FIX)'"; \
		exit 1; \
	fi
	$(Q)if [ "$(FIX)" = "1" ]; then $(POETRY) run ruff check --fix . --quiet; fi
	$(Q)gates="$(VALIDATE_GATES)"; \
	if [ -n "$$gates" ]; then \
		for g in $$(echo "$$gates" | tr ',' ' '); do \
			case "$$g" in \
				complexity|docstring) ;; \
				*) echo "ERROR: unknown VALIDATE_GATES value '$$g' (allowed: complexity,docstring)"; exit 2;; \
			esac; \
		done; \
	else \
		gates="complexity,docstring"; \
	fi; \
	if echo "$$gates" | grep -qw complexity; then \
		$(POETRY) run radon cc $(SRC_DIR) -n E -a --total-average; \
		$(POETRY) run radon mi $(SRC_DIR) -n C -s --sort; \
	fi; \
	if echo "$$gates" | grep -qw docstring; then \
		$(POETRY) run interrogate $(SRC_DIR) --fail-under=$(DOCSTRING_MIN) --ignore-init-method --ignore-magic -q; \
	fi

daemon-start-mypy: ## Start dmypy daemon for this project
	$(Q)mkdir -p .dmypy
	$(Q)if $(VENV_PYTHON) -m mypy.dmypy --status-file "$(DMPY_SOCKET)" status >/dev/null 2>&1; then \
		echo "dmypy already running for $(PROJECT_NAME) at $(DMPY_SOCKET)"; \
	else \
		$(VENV_PYTHON) -m mypy.dmypy --status-file "$(DMPY_SOCKET)" start -- --config-file "$(WORKSPACE_ROOT)/pyproject.toml"; \
	fi

daemon-stop-mypy: ## Stop dmypy daemon for this project
	$(Q)$(VENV_PYTHON) -m mypy.dmypy --status-file "$(DMPY_SOCKET)" stop >/dev/null 2>&1 || true
	$(Q)rm -f "$(DMPY_SOCKET)"

daemon-status-mypy: ## Show dmypy daemon status for this project
	$(Q)if $(VENV_PYTHON) -m mypy.dmypy --status-file "$(DMPY_SOCKET)" status 2>/dev/null; then \
		: ; \
	else \
		echo "dmypy daemon is not running"; \
	fi

daemon-start-pyright: ## Start pyright daemon in watch mode
	$(Q)mkdir -p .pyright
	$(Q)if [ -f "$(PYRIGHT_PIDFILE)" ]; then \
		pid=$$(cat "$(PYRIGHT_PIDFILE)"); \
		if [ -n "$$pid" ] && kill -0 "$$pid" >/dev/null 2>&1; then \
			echo "Pyright daemon already running (PID $$pid)"; \
			exit 0; \
		fi; \
		rm -f "$(PYRIGHT_PIDFILE)"; \
	fi
	$(Q)nohup pyright --watch --threads > "$(PYRIGHT_LOG)" 2>&1 & \
		pid=$$!; \
		echo "$$pid" > "$(PYRIGHT_PIDFILE)"; \
		echo "Pyright daemon started (PID $$pid), log: $(PYRIGHT_LOG)"

daemon-stop-pyright: ## Stop pyright daemon
	$(Q)if [ ! -f "$(PYRIGHT_PIDFILE)" ]; then \
		echo "Pyright daemon is not running"; \
		exit 0; \
	fi
	$(Q)pid=$$(cat "$(PYRIGHT_PIDFILE)"); \
	if [ -n "$$pid" ] && kill -0 "$$pid" >/dev/null 2>&1; then \
		kill "$$pid" >/dev/null 2>&1 || true; \
		echo "Stopped pyright daemon (PID $$pid)"; \
	else \
		echo "Pyright daemon PID file was stale"; \
	fi; \
	rm -f "$(PYRIGHT_PIDFILE)"

daemon-status-pyright: ## Show pyright daemon status
	$(Q)if [ ! -f "$(PYRIGHT_PIDFILE)" ]; then \
		echo "Pyright daemon is not running"; \
	else \
		pid=$$(cat "$(PYRIGHT_PIDFILE)"); \
		if [ -n "$$pid" ] && kill -0 "$$pid" >/dev/null 2>&1; then \
			echo "Pyright daemon running (PID $$pid), log: $(PYRIGHT_LOG)"; \
		else \
			echo "Pyright daemon not running (stale PID file cleaned)"; \
			rm -f "$(PYRIGHT_PIDFILE)"; \
		fi; \
	fi

daemon-start: daemon-start-mypy daemon-start-pyright ## Start all daemons

daemon-stop: daemon-stop-mypy daemon-stop-pyright ## Stop all daemons

daemon-status: ## Show status of all daemons
	$(Q)echo "== dmypy =="; \
	$(MAKE) daemon-status-mypy; \
	echo "== pyright =="; \
	$(MAKE) daemon-status-pyright

daemon-restart: daemon-stop daemon-start ## Restart all daemons

pr: ## Manage pull requests for this repository
	$(Q)$(PROJECT_INFRA_GITHUB) pr \
		--repo-root "$(CURDIR)" \
		--action "$(PR_ACTION)" \
		--base "$(PR_BASE)" \
		$(if $(PR_HEAD),--head "$(PR_HEAD)",) \
		$(if $(PR_NUMBER),--number "$(PR_NUMBER)",) \
		$(if $(PR_TITLE),--title "$(PR_TITLE)",) \
		$(if $(PR_BODY),--body "$(PR_BODY)",) \
		--draft "$(PR_DRAFT)" \
		--merge-method "$(PR_MERGE_METHOD)" \
		--auto "$(PR_AUTO)" \
		--delete-branch "$(PR_DELETE_BRANCH)" \
		--checks-strict "$(PR_CHECKS_STRICT)" \
		--release-on-merge "$(PR_RELEASE_ON_MERGE)"

clean: ## Clean artifacts
	$(Q)if [ "$(CORE_STACK)" = "go" ]; then \
		rm -f coverage.out coverage.html; \
		go clean; \
	fi
	$(Q)rm -rf build/ dist/ *.egg-info/ .pytest_cache/ htmlcov/ .coverage* \
		.mypy_cache/ .pyrefly_cache/ .ruff_cache/ $(LINT_CACHE_DIR)/ \
		.pyright/ .pytype/ .pyrefly-report.json .pyrefly-output.txt
	$(Q)find . -type d -name __pycache__ -exec rm -rf {} +
	$(Q)find . -type f -name "*.pyc" -delete
	$(Q)echo "Clean complete: $(PROJECT_NAME)"
