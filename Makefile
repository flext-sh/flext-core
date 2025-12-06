# =============================================================================
# FLEXT-CORE - Foundation Library Makefile
# =============================================================================
# Python 3.13+ Foundation Framework - Clean Architecture + DDD + Zero Tolerance
# =============================================================================

# Project Configuration
PROJECT_NAME := flext-core
PYTHON_VERSION := 3.13
POETRY := poetry
SRC_DIR := src
COV_DIR := flext_core
TESTS_DIR := tests

# Quality Standards (MANDATORY - HIGH COVERAGE)
MIN_COVERAGE := 80

# Export Configuration
export PROJECT_NAME PYTHON_VERSION MIN_COVERAGE

# =============================================================================
# HELP & INFORMATION
# =============================================================================

.PHONY: help
help: ## Show available commands
	@echo "FLEXT-CORE - Foundation Library"
	@echo "================================"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

.PHONY: info
info: ## Show project information
	@echo "Project: $(PROJECT_NAME)"
	@echo "Python: $(PYTHON_VERSION)+"
	@echo "Poetry: $(POETRY)"
	@echo "Coverage: $(MIN_COVERAGE)% minimum (MANDATORY)"
	@echo "Architecture: Clean Architecture + DDD + Foundation"

# =============================================================================
# SETUP & INSTALLATION
# =============================================================================

.PHONY: install
install: ## Install dependencies
	$(POETRY) install

.PHONY: install-dev
install-dev: ## Install dev dependencies
	$(POETRY) install --extras "dev test typings security"

.PHONY: setup
setup: install-dev ## Complete project setup
	$(POETRY) run pre-commit install

# =============================================================================
# QUALITY GATES (MANDATORY - ZERO TOLERANCE)
# =============================================================================

.PHONY: validate
validate: lint format-check type-check complexity docstring-check security test ## Run all quality gates (MANDATORY ORDER)

.PHONY: check
check: lint type-check ## Quick health check

# audit-pydantic-v2 target REMOVED - use flext-quality instead:
#   poetry run flext-quality analyze .
#   Legacy scripts renamed to *.py.bak

.PHONY: lint
lint: ## Run linting (ZERO TOLERANCE)
	$(POETRY) run ruff check .

.PHONY: format
format: ## Format code
	$(POETRY) run ruff format .

.PHONY: type-check
type-check: ## Run type checking with Pyrefly (src only)
	@echo "🔍 Type checking with Pyrefly (src/ only)..."
	$(POETRY) run pyrefly check src/

.PHONY: type-check-all
type-check-all: ## Run type checking including examples
	@echo "🔍 Type checking with Pyrefly (src/ + examples/)..."
	$(POETRY) run pyrefly check src/ examples/

.PHONY: security
security: ## Run security scanning
	$(POETRY) run bandit -r $(SRC_DIR) --exclude $(SRC_DIR)/flext_tests
	$(POETRY) run pip-audit --ignore-vuln PYSEC-2022-42969  # py 1.11.0 - dev-only dependency from interrogate

.PHONY: fix
fix: ## Auto-fix issues
	$(POETRY) run ruff check . --fix
	$(POETRY) run ruff format .

.PHONY: format-check
format-check: ## Check code formatting without modifying
	@echo "🔍 Checking code formatting..."
	$(POETRY) run ruff format . --check --diff

.PHONY: complexity
complexity: ## Run complexity analysis (radon)
	@echo "🔍 Running complexity analysis..."
	$(POETRY) run radon cc $(SRC_DIR) -a -nb --total-average
	$(POETRY) run radon mi $(SRC_DIR) -nb

.PHONY: docstring-check
docstring-check: ## Check docstring coverage
	@echo "🔍 Checking docstring coverage..."
	$(POETRY) run interrogate $(SRC_DIR) --fail-under=80 --ignore-init-method --ignore-magic

# =============================================================================
# TESTING (MANDATORY - HIGH COVERAGE)
# =============================================================================

.PHONY: test
test: ## Run tests with high coverage (MANDATORY)
	PYTHONPATH=$(SRC_DIR) $(POETRY) run pytest -q --maxfail=10000 --cov=$(COV_DIR) --cov-report=term-missing --cov-fail-under=$(MIN_COVERAGE)

.PHONY: test-unit
test-unit: ## Run unit tests
	PYTHONPATH=$(SRC_DIR) $(POETRY) run pytest -m "not integration" -v

.PHONY: test-integration
test-integration: ## Run integration tests with Docker
	PYTHONPATH=$(SRC_DIR) $(POETRY) run pytest -m integration -v

.PHONY: test-fast
test-fast: ## Run tests without coverage
	PYTHONPATH=$(SRC_DIR) $(POETRY) run pytest -v

.PHONY: coverage-html
coverage-html: ## Generate HTML coverage report
	PYTHONPATH=$(SRC_DIR) $(POETRY) run pytest --cov=$(COV_DIR) --cov-report=html

# =============================================================================
# BUILD & DISTRIBUTION
# =============================================================================

.PHONY: build
build: ## Build package
	$(POETRY) build

.PHONY: build-clean
build-clean: clean build ## Clean and build

# =============================================================================
# DOCUMENTATION
# =============================================================================

.PHONY: docs
docs: ## Build documentation
	$(POETRY) run mkdocs build

.PHONY: docs-serve
docs-serve: ## Serve documentation
	$(POETRY) run mkdocs serve

# =============================================================================
# DEPENDENCIES
# =============================================================================

.PHONY: deps-update
deps-update: ## Update dependencies
	$(POETRY) update

.PHONY: deps-show
deps-show: ## Show dependency tree
	$(POETRY) show --tree

.PHONY: deps-audit
deps-audit: ## Audit dependencies
	$(POETRY) run pip-audit

# =============================================================================
# DEVELOPMENT
# =============================================================================

.PHONY: shell
shell: ## Open Python shell
	PYTHONPATH=$(SRC_DIR) $(POETRY) run python

.PHONY: pre-commit
pre-commit: ## Run pre-commit hooks
	$(POETRY) run pre-commit run --all-files

# =============================================================================
# MAINTENANCE
# =============================================================================

.PHONY: clean
clean: ## Clean build artifacts and cruft
	@echo "🧹 Cleaning $(PROJECT_NAME) - removing build artifacts, cache files, and cruft..."

	# Build artifacts
	rm -rf build/ dist/ *.egg-info/

	# Test artifacts
	rm -rf .pytest_cache/ htmlcov/ .coverage .coverage.* coverage.xml

	# Python cache directories
	rm -rf .mypy_cache/ .pyrefly_cache/ .ruff_cache/

	# Python bytecode
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true

	# Temporary files
	find . -type f -name "*.tmp" -delete 2>/dev/null || true
	find . -type f -name "*.temp" -delete 2>/dev/null || true
	find . -type f -name ".DS_Store" -delete 2>/dev/null || true



	# Editor files
	find . -type f -name ".vscode/settings.json" -delete 2>/dev/null || true
	find . -type f -name ".idea/" -type d -exec rm -rf {} + 2>/dev/null || true

	@echo "✅ $(PROJECT_NAME) cleanup complete"

.PHONY: clean-all
clean-all: clean ## Deep clean including venv
	rm -rf .venv/

.PHONY: reset
reset: clean-all setup ## Reset project

# =============================================================================
# DIAGNOSTICS
# =============================================================================

.PHONY: diagnose
diagnose: ## Project diagnostics
	@echo "Python: $$(python --version)"
	@echo "Poetry: $$($(POETRY) --version)"
	@$(POETRY) env info

.PHONY: doctor
doctor: diagnose check ## Health check

# =============================================================================

# =============================================================================

.PHONY: t l f fc tc cx dc c i v sec
t: test
l: lint
f: format
fc: format-check
tc: type-check
cx: complexity
dc: docstring-check
c: clean
i: install
v: validate
sec: security

# =============================================================================
# GIT OPERATIONS
# =============================================================================

# Optional: MESSAGE="your commit message" make commit
MESSAGE ?= "chore: update project files"

.PHONY: status
status: ## Show git status
	@git status --short

.PHONY: diff
diff: ## Show git diff
	@git diff --stat

.PHONY: add
add: ## Stage all changes
	@git add -A

.PHONY: commit
commit: add ## Commit changes (use MESSAGE="..." to customize)
	@git commit -m "$(MESSAGE)"

.PHONY: push
push: ## Push to remote
	@git push

.PHONY: save
save: commit push ## Add, commit and push (use MESSAGE="...")

.PHONY: amend
amend: add ## Amend last commit
	@git commit --amend --no-edit

.PHONY: commit-force
commit-force: add ## Commit without pre-commit hooks (use MESSAGE="...")
	@git commit --no-verify -m "$(MESSAGE)"

.PHONY: save-force
save-force: commit-force push ## Add, commit (no hooks) and push (use MESSAGE="...")

# =============================================================================
# GITHUB ACTIONS (gh CLI required)
# =============================================================================

.PHONY: gh-runs
gh-runs: ## List recent workflow runs
	@gh run list --limit 10

.PHONY: gh-watch
gh-watch: ## Watch latest workflow run
	@gh run watch

.PHONY: gh-status
gh-status: ## Show status of latest run
	@gh run list --limit 1

.PHONY: gh-logs
gh-logs: ## View logs of latest run
	@gh run view --log

.PHONY: gh-failed
gh-failed: ## List failed runs
	@gh run list --status failure --limit 5

.PHONY: gh-pr
gh-pr: ## Create pull request
	@gh pr create --fill

.PHONY: gh-pr-status
gh-pr-status: ## Check PR status
	@gh pr status

.PHONY: gh-checks
gh-checks: ## View PR checks
	@gh pr checks

# Shortcuts for git/gh
.PHONY: s d cm p sv ghr ghw ghs ghl
s: status
d: diff
cm: commit
p: push
sv: save
ghr: gh-runs
ghw: gh-watch
ghs: gh-status
ghl: gh-logs

# =============================================================================
# CONFIGURATION
# =============================================================================

.DEFAULT_GOAL := help
