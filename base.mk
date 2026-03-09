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
AUTO_ADJUST ?= 1
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
export PATH := $(ACTIVE_VENV)/bin:$(PATH)

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

$(LINT_CACHE_DIR):
	$(Q)mkdir -p $(LINT_CACHE_DIR)

# === SIMPLE VERB SURFACE ===
.PHONY: help setup build check security format docs docs-base docs-sync-scripts test validate clean pr _preflight daemon-start-mypy daemon-stop-mypy daemon-status-mypy daemon-start-pyright daemon-stop-pyright daemon-status-pyright daemon-start daemon-stop daemon-status daemon-restart
STANDARD_VERBS := setup build check security format docs test validate clean pr
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
		echo "ERROR: [preflight] Workspace venv not found. Run 'make setup' at workspace root."; \
		exit 1; \
	elif [ "$(filter setup,$(MAKECMDGOALS))" != "setup" ] && [ ! -d "$(ACTIVE_VENV)" ]; then \
		echo "ERROR: [preflight] No venv found (workspace or local). Run 'make setup' in $(PROJECT_NAME)."; \
		exit 1; \
	else \
		echo "INFO: [preflight] Using project-local venv for $(PROJECT_NAME) (workspace venv not found)."; \
	fi; \
elif [ "$(FLEXT_MODE)" = "standalone" ]; then \
	echo "INFO: [preflight] Running in standalone mode (workspace features unavailable)."; \
elif [ "$(filter setup,$(MAKECMDGOALS))" != "setup" ] && [ ! -d "$(ACTIVE_VENV)" ]; then \
	echo "ERROR: [preflight] No venv found at $(ACTIVE_VENV). Run 'make setup' in $(PROJECT_NAME)."; \
	exit 1; \
fi
endef

define AUTO_ADJUST_PROJECT
endef

define AUTO_SYNC_BASE_AND_SCRIPTS
if [ "$(FLEXT_MODE)" = "workspace" ] && [ "$(CURDIR)" != "$(WORKSPACE_ROOT)" ]; then \
	python -m flext_infra workspace sync \
		--project-root "$(CURDIR)"; \
elif [ "$(FLEXT_MODE)" = "standalone" ]; then \
	echo "INFO: [preflight] Standalone mode: skipping workspace dependency sync."; \
fi
endef

_preflight: ## Preflight: sync base.mk, check venv, auto-adjust files
	$(Q)$(AUTO_SYNC_BASE_AND_SCRIPTS)
	$(Q)$(ENFORCE_WORKSPACE_VENV)
	$(Q)$(AUTO_ADJUST_PROJECT)

help: ## Show commands
	$(Q)echo "================================================"
	$(Q)echo "  $(PROJECT_NAME)"
	$(Q)echo "================================================"
	$(Q)echo ""
	$(Q)echo "Core verbs:"
	$(Q)printf "  %-14s %s\n" "setup"     "Install dependencies and hooks"
	$(Q)printf "  %-14s %s\n" "build"     "Build distributable artifacts"
	$(Q)printf "  %-14s %s\n" "check"     "Run lint gates (CHECK_GATES= to select)"
	$(Q)printf "  %-14s %s\n" "security"  "Run all security checks"
	$(Q)printf "  %-14s %s\n" "format"    "Run all formatting"
	$(Q)printf "  %-14s %s\n" "docs"      "Build docs (DOCS_PHASE= to select)"
	$(Q)printf "  %-14s %s\n" "test"      "Run pytest (PYTEST_ARGS= for options)"
	$(Q)printf "  %-14s %s\n" "validate"  "Run validate gates (FIX=1 to auto-fix)"
	$(Q)printf "  %-14s %s\n" "clean"     "Clean build/test/type artifacts"
	$(Q)echo ""
	$(Q)echo "Daemon management:"
	$(Q)printf "  %-16s %s\n" "daemon-start"   "Start all daemons (mypy + pyright)"
	$(Q)printf "  %-16s %s\n" "daemon-stop"    "Stop all daemons"
	$(Q)printf "  %-16s %s\n" "daemon-status"  "Show status of all daemons"
	$(Q)printf "  %-16s %s\n" "daemon-restart" "Restart all daemons"
	$(Q)echo "  Also: daemon-{start,stop,status}-{mypy,pyright}"
	$(Q)echo ""
	$(Q)echo "Other:"
	$(Q)printf "  %-14s %s\n" "check-fast" "Parallel lint (make -j4 check-fast)"
	$(Q)printf "  %-14s %s\n" "pr"         "Manage PRs (PR_ACTION=status)"
	$(Q)echo ""
	$(Q)echo "PR variables:"
	$(Q)echo "  PR_ACTION=status|create|view|checks|merge|close"
	$(Q)echo "  PR_BASE=main  PR_HEAD=<branch>  PR_NUMBER=<id>"
	$(Q)echo "  PR_TITLE='...'  PR_BODY='...'  PR_DRAFT=0|1"
	$(Q)echo "  PR_MERGE_METHOD=squash|merge|rebase  PR_AUTO=0|1"
	$(Q)echo "  PR_DELETE_BRANCH=0|1  PR_CHECKS_STRICT=0|1"
	$(Q)echo "  PR_RELEASE_ON_MERGE=0|1"

setup: ## Complete setup
	$(Q)if [ "$(CORE_STACK)" = "go" ]; then \
		go mod download; \
		go mod tidy; \
		exit 0; \
	fi
	$(Q)python -m flext_infra deps internal-sync
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
					lint|format|security|markdown|go|type) ;; \
					*) echo "ERROR: unknown CHECK_GATES value '$$g' (allowed: lint,format,security,markdown,go,type)"; exit 2;; \
				esac; \
			done; \
		else \
			gates="lint,format,security,markdown,go"; \
		fi; \
		gates=$$(echo "$$gates" | tr ',' ' ' | sed 's/\btype\b/go/g' | tr ' ' ','); \
		if echo "$$gates" | grep -qw lint; then \
			golangci-lint run || { echo "FAIL: lint"; exit 1; }; \
		fi; \
		if echo "$$gates" | grep -qw format; then \
			if [ -n "$$(find . -type f -name '*.go' ! -path './.git/*' ! -path './vendor/*')" ]; then \
				gofmt_diff=$$(find . -type f -name '*.go' ! -path './.git/*' ! -path './vendor/*' -print0 | xargs -0 gofmt -l); \
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
			if git rev-parse --git-dir >/dev/null 2>&1; then \
				md_files=$$(git ls-files -- '*.md' ':!vendor/'); \
			else \
				md_files=$$(find . -type f -name '*.md' ! -path './.git/*' ! -path './.reports/*' ! -path './reports/*' ! -path './.venv/*' ! -path './vendor/*' ! -path './node_modules/*' ! -path './dist/*' ! -path './build/*'); \
			fi; \
			md_config=""; \
			if [ -f "$(WORKSPACE_ROOT)/.markdownlint.json" ]; then \
				md_config="--config $(WORKSPACE_ROOT)/.markdownlint.json"; \
			elif [ -f ".markdownlint.json" ]; then \
				md_config="--config .markdownlint.json"; \
			fi; \
			if [ -n "$$md_files" ]; then echo "$$md_files" | xargs markdownlint $$md_config || { echo "FAIL: markdown"; exit 1; }; fi; \
		fi; \
		if echo "$$gates" | grep -qw go; then \
			go vet ./... || { echo "FAIL: go"; exit 1; }; \
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
	project_key="$(PROJECT_NAME)"; \
	if [ "$(CURDIR)" = "$(WORKSPACE_ROOT)" ]; then \
		project_key="."; \
	fi; \
	$(POETRY) run python -m flext_infra check fix-pyrefly-config "$$project_key"; \
	$(POETRY) run python -m flext_infra check run --gates "$$gates" --reports-dir "$(CURDIR)/.reports/check" --project "$$project_key"; \
	exit $$?; \
	if echo "$$gates" | grep -qw lint; then \
		$(POETRY) run ruff check . --quiet || { echo "FAIL: lint"; exit 1; }; \
	fi; \
	if echo "$$gates" | grep -qw format; then \
		$(POETRY) run ruff format --check . --quiet || { echo "FAIL: format"; exit 1; }; \
	fi; \
	check_dirs=""; \
	for d in src tests examples scripts; do \
		if [ -d "$$d" ]; then check_dirs="$$check_dirs $$d"; fi; \
	done; \
	check_dirs=$${check_dirs:-$(SRC_DIR)}; \
	if echo "$$gates" | grep -qw pyrefly; then \
		$(POETRY) run pyrefly check $$check_dirs --config pyproject.toml \
			--count-errors=0 --summarize-errors=1 --summary full || { echo "FAIL: pyrefly"; exit 1; }; \
	fi; \
	if echo "$$gates" | grep -qw mypy; then \
		$(POETRY) run mypy $$check_dirs --config-file "$(WORKSPACE_ROOT)/pyproject.toml" || { echo "FAIL: mypy"; exit 1; }; \
	fi; \
	if echo "$$gates" | grep -qw pyright; then \
		$(POETRY) run pyright $$check_dirs || { echo "FAIL: pyright"; exit 1; }; \
	fi; \
	if echo "$$gates" | grep -qw security; then \
		$(POETRY) run bandit -r $(SRC_DIR) -q -ll || { echo "FAIL: security"; exit 1; }; \
	fi; \
	if echo "$$gates" | grep -qw markdown; then \
		md_files=$$(find . -type f -name '*.md' ! -path './.git/*' ! -path './.reports/*' ! -path './reports/*' ! -path './.venv/*' ! -path './node_modules/*' ! -path './.flext-deps/*' ! -path './.mypy_cache/*' ! -path './.pytest_cache/*' ! -path './.ruff_cache/*' ! -path './dist/*' ! -path './build/*'); \
		md_config=""; \
		if [ -f "$(WORKSPACE_ROOT)/.markdownlint.json" ]; then \
			md_config="--config $(WORKSPACE_ROOT)/.markdownlint.json"; \
		elif [ -f ".markdownlint.json" ]; then \
			md_config="--config .markdownlint.json"; \
		fi; \
		if [ -n "$$md_files" ]; then \
			markdownlint $$md_config $$md_files || { echo "FAIL: markdown"; exit 1; }; \
		fi; \
	fi; \
	if echo "$$gates" | grep -qw go; then \
		if [ -f go.mod ]; then \
			go vet ./... || { echo "FAIL: go"; exit 1; }; \
			if [ -n "$$(find . -type f -name '*.go' ! -path './.git/*')" ]; then \
				gofmt_diff=$$(find . -type f -name '*.go' ! -path './.git/*' -print0 | xargs -0 gofmt -l); \
				if [ -n "$$gofmt_diff" ]; then \
					echo "FAIL: gofmt"; \
					printf '%s\n' "$$gofmt_diff"; \
					exit 1; \
				fi; \
			fi; \
		fi; \
	fi

security: ## Run all security checks
	$(Q)if [ "$(CORE_STACK)" = "go" ]; then \
		gosec ./...; \
		exit 0; \
	fi
	$(Q)$(POETRY) run bandit -r $(SRC_DIR) -q -ll

format: ## Run code formatting (ruff/gofmt + markdownlint on tracked files)
	$(Q)if [ "$(CORE_STACK)" = "go" ]; then \
		go_files=$$(find . -type f -name '*.go' ! -path './.git/*' ! -path './vendor/*'); \
		if [ -n "$$go_files" ]; then \
			printf '%s\n' "$$go_files" | xargs gofmt -w; \
			if command -v goimports >/dev/null 2>&1; then \
				printf '%s\n' "$$go_files" | xargs goimports -w; \
			fi; \
		fi; \
	else \
		$(POETRY) run ruff format . --quiet; \
		if [ -f go.mod ]; then \
			go_files=$$(find . -type f -name '*.go' ! -path './.git/*' ! -path './vendor/*'); \
			if [ -n "$$go_files" ]; then \
				printf '%s\n' "$$go_files" | xargs gofmt -w; \
			fi; \
		fi; \
	fi
	$(Q)if git rev-parse --git-dir >/dev/null 2>&1; then \
		md_files=$$(git ls-files -- '*.md' ':!vendor/' && git ls-files --others --exclude-standard -- '*.md' ':!vendor/'); \
	else \
		md_files=$$(find . -type f -name '*.md' ! -path './.git/*' ! -path './.reports/*' ! -path './.venv/*' ! -path './vendor/*' ! -path './node_modules/*' ! -path './dist/*' ! -path './build/*'); \
	fi; \
	if [ -n "$$md_files" ]; then \
		md_config=""; \
		if [ -f "$(WORKSPACE_ROOT)/.markdownlint.json" ]; then \
			md_config="--config $(WORKSPACE_ROOT)/.markdownlint.json"; \
		elif [ -f ".markdownlint.json" ]; then \
			md_config="--config .markdownlint.json"; \
		fi; \
		echo "$$md_files" | xargs markdownlint --fix $$md_config 2>/dev/null || true; \
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
			audit) subcmd="docs audit"; extra="--strict 1" ;; \
			fix) subcmd="docs fix"; extra="$(if $(filter 1,$(FIX)),--apply,)" ;; \
			build) subcmd="docs build"; extra="" ;; \
			generate) subcmd="docs generate"; extra="--apply" ;; \
			validate) subcmd="docs validate"; extra="$(if $(filter 1,$(FIX)),--apply,)" ;; \
			*) echo "ERROR: invalid DOCS_PHASE=$$phase"; exit 2 ;; \
		esac; \
		if [ "$$phase" = "fix" ] && [ "$$all_mode" = "1" ]; then extra="--apply"; fi; \
		cmd="python -m flext_infra $$subcmd --root . --output-dir .reports/docs"; \
		if [ -n "$$extra" ]; then cmd="$$cmd $$extra"; fi; \
		eval $$cmd || exit $$?; \
	done

test: ## Run pytest only
	$(Q)if [ "$(CORE_STACK)" = "go" ]; then \
		go test -v -race -coverprofile=coverage.out -covermode=atomic ./...; \
		go tool cover -func=coverage.out; \
	else \
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
	echo "$(POETRY) run pytest $(TESTS_DIR) $(PYTEST_REPORT_ARGS) $(if $(filter 1,$(DIAG)),$(PYTEST_DIAG_ARGS),) -p no:metadata --junitxml=$$junit_file --cov --cov-report=xml:$$coverage_file $(if $(filter 1,$(DIAG)),-vv,-q) $(PYTEST_ARGS)" > "$$command_file"; \
	trap 'interrupted=1; trap "" INT TERM' INT TERM; \
	$(POETRY) run pytest $(TESTS_DIR) \
		$(PYTEST_REPORT_ARGS) \
		$(if $(filter 1,$(DIAG)),$(PYTEST_DIAG_ARGS),) \
		-p no:metadata \
		--junitxml="$$junit_file" \
		--cov --cov-report=xml:$$coverage_file \
		$(if $(filter 1,$(DIAG)),-vv,-q) $(PYTEST_ARGS) 2>&1 | tee "$$log_file"; \
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
	$(VENV_PYTHON) -m flext_infra core pytest-diag \
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
	awk 'NR<=10 {print}' "$$slowest_file" >&2; \
	echo "Error trace excerpt (from $$errors_file):" >&2; \
	awk 'NR<=40 {print}' "$$errors_file" >&2; \
	ln -sfn "$$run_id" "$(PYTEST_REPORTS_DIR)/latest"; \
	echo "Reports: $$report_dir (latest: $(PYTEST_REPORTS_DIR)/latest)" >&2; \
	echo "Details: $$summary_file | $$failed_file | $$errors_file | $$warnings_file | $$slowest_file | $$skips_file | $$log_file" >&2; \
	exit $$rc; \
	fi

validate: ## Run validate gates (VALIDATE_GATES=complexity,docstring to select, FIX=1)
	$(Q)if [ "$(CORE_STACK)" = "go" ]; then \
		if [ "$(FIX)" = "1" ]; then \
			$(MAKE) format; \
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
	$(Q)python3 -m flext_infra github pr \
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
