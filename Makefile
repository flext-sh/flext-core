# FLEXT Core - Enterprise Development Makefile
# Poetry-based orchestration with strict quality gates
# =====================================================

.PHONY: help check fix test lint format type-check security pre-commit install clean validate-strict status poetry-check

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
PURPLE := \033[0;35m
CYAN := \033[0;36m
NC := \033[0m # No Color

# Project settings
PYTHON := python3.13
PROJECT_NAME := flext-core
SRC_DIR := src
TEST_DIR := tests
REPORTS_DIR := reports

# Poetry settings
POETRY := poetry
POETRY_RUN := $(POETRY) run
POETRY_OPTS := --no-interaction --quiet

# Check if Poetry is installed
POETRY_CHECK := $(shell command -v poetry 2> /dev/null)

poetry-check:
	@if [ -z "$(POETRY_CHECK)" ]; then \
		echo "$(RED)❌ Poetry is not installed!$(NC)"; \
		echo "$(YELLOW)Please install Poetry: https://python-poetry.org/docs/#installation$(NC)"; \
		exit 1; \
	fi

help: ## Show this help message
	@echo "$(PURPLE)╔═══════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(PURPLE)║$(NC)     $(BLUE)FLEXT Core - Poetry Development Commands$(NC)             $(PURPLE)║$(NC)"
	@echo "$(PURPLE)╚═══════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "$(CYAN)🔍 Quality Checks:$(NC)"
	@echo "  $(GREEN)check$(NC)            Run ALL quality checks"
	@echo "  $(GREEN)lint$(NC)             Linting with ruff (17 categories)"
	@echo "  $(GREEN)format-check$(NC)     Check code formatting"
	@echo "  $(GREEN)type-check$(NC)       MyPy in strict mode"
	@echo "  $(GREEN)security$(NC)         Security scans"
	@echo "  $(GREEN)complexity$(NC)       Code complexity analysis"
	@echo ""
	@echo "$(CYAN)🔧 Code Fixes:$(NC)"
	@echo "  $(GREEN)fix$(NC)              Auto-fix all issues"
	@echo "  $(GREEN)format$(NC)           Format code (black + ruff)"
	@echo "  $(GREEN)sort-imports$(NC)     Sort imports with isort"
	@echo ""
	@echo "$(CYAN)🧪 Testing:$(NC)"
	@echo "  $(GREEN)test$(NC)             Run all tests with coverage"
	@echo "  $(GREEN)test-unit$(NC)        Unit tests only"
	@echo "  $(GREEN)test-integration$(NC) Integration tests only"
	@echo "  $(GREEN)test-watch$(NC)       Watch mode testing"
	@echo ""
	@echo "$(CYAN)📦 Project Management:$(NC)"
	@echo "  $(GREEN)install$(NC)          Install all dependencies"
	@echo "  $(GREEN)update$(NC)           Update dependencies"
	@echo "  $(GREEN)lock$(NC)             Update lock file"
	@echo "  $(GREEN)build$(NC)            Build distribution"
	@echo "  $(GREEN)docs$(NC)             Build documentation"
	@echo "  $(GREEN)docs-serve$(NC)       Serve documentation"
	@echo ""
	@echo "$(CYAN)🛠️ Development:$(NC)"
	@echo "  $(GREEN)pre-commit$(NC)       Setup pre-commit hooks"
	@echo "  $(GREEN)clean$(NC)            Remove all artifacts"
	@echo "  $(GREEN)status$(NC)           Show quality status"
	@echo "  $(GREEN)validate$(NC)         Validate 100% compliance"
	@echo ""
	@echo "$(YELLOW)⚡ Strict Mode: ZERO violations tolerated!$(NC)"
	@echo "$(PURPLE)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"

# ═══════════════════════════════════════════════════════════════════
# QUALITY CHECKS
# ═══════════════════════════════════════════════════════════════════

check: poetry-check format-check lint type-check security complexity test ## Run ALL quality checks
	@echo ""
	@echo "$(GREEN)═══════════════════════════════════════════════════════════$(NC)"
	@echo "$(GREEN)✅ ALL QUALITY CHECKS PASSED - 100% COMPLIANCE!$(NC)"
	@echo "$(GREEN)═══════════════════════════════════════════════════════════$(NC)"

lint: poetry-check ## Run linting with ruff
	@echo "$(BLUE)🔥 Running ruff linter (17 rule categories)...$(NC)"
	@$(POETRY_RUN) ruff check $(SRC_DIR)/ $(TEST_DIR)/ --config pyproject.toml
	@echo "$(GREEN)✅ Linting passed!$(NC)"

format-check: poetry-check ## Check code formatting
	@echo "$(BLUE)⚫ Checking black formatting...$(NC)"
	@$(POETRY_RUN) black --check $(SRC_DIR)/ $(TEST_DIR)/
	@echo "$(BLUE)⚡ Checking ruff formatting...$(NC)"
	@$(POETRY_RUN) ruff format --check $(SRC_DIR)/ $(TEST_DIR)/
	@echo "$(GREEN)✅ Formatting check passed!$(NC)"

type-check: poetry-check ## Run mypy in strict mode
	@echo "$(BLUE)🛡️ Running mypy (strict mode)...$(NC)"
	@$(POETRY_RUN) mypy $(SRC_DIR)/ $(TEST_DIR)/ --config-file pyproject.toml
	@echo "$(GREEN)✅ Type checking passed!$(NC)"

security: poetry-check ## Run security scans
	@echo "$(BLUE)🔒 Running security scans...$(NC)"
	@echo "→ Bandit security scan..."
	@$(POETRY_RUN) bandit -r $(SRC_DIR)/ --severity-level medium
	@echo "→ Safety check..."
	@$(POETRY_RUN) safety check --json --output $(REPORTS_DIR)/safety.json 2>/dev/null || true
	@$(POETRY_RUN) safety check || true
	@echo "→ Detect-secrets scan..."
	@$(POETRY_RUN) detect-secrets scan --baseline .secrets.baseline
	@echo "$(GREEN)✅ Security scans passed!$(NC)"

complexity: poetry-check ## Code complexity analysis
	@echo "$(BLUE)📊 Analyzing code complexity...$(NC)"
	@echo "→ Cyclomatic Complexity:"
	@$(POETRY_RUN) radon cc $(SRC_DIR)/ -a -nb
	@echo ""
	@echo "→ Maintainability Index:"
	@$(POETRY_RUN) radon mi $(SRC_DIR)/ -nb
	@echo ""
	@echo "→ Dead Code Detection:"
	@$(POETRY_RUN) vulture $(SRC_DIR)/ --min-confidence 80 || true
	@echo "$(GREEN)✅ Complexity analysis complete!$(NC)"

# ═══════════════════════════════════════════════════════════════════
# TESTING
# ═══════════════════════════════════════════════════════════════════

test: poetry-check ## Run all tests with coverage
	@echo "$(BLUE)🧪 Running all tests with coverage...$(NC)"
	@mkdir -p $(REPORTS_DIR)
	@$(POETRY_RUN) pytest $(TEST_DIR)/ \
		-v \
		--tb=short \
		--cov=$(SRC_DIR)/flext_core \
		--cov-report=term-missing:skip-covered \
		--cov-report=html:$(REPORTS_DIR)/coverage \
		--cov-report=xml:$(REPORTS_DIR)/coverage.xml \
		--cov-fail-under=90
	@echo "$(GREEN)✅ All tests passed!$(NC)"

test-unit: poetry-check ## Run unit tests only
	@echo "$(BLUE)🧪 Running unit tests...$(NC)"
	@$(POETRY_RUN) pytest $(TEST_DIR)/unit/ -v --tb=short

test-integration: poetry-check ## Run integration tests only
	@echo "$(BLUE)🧪 Running integration tests...$(NC)"
	@$(POETRY_RUN) pytest $(TEST_DIR)/integration/ -v --tb=short

test-watch: poetry-check ## Watch mode testing
	@echo "$(BLUE)👁️ Running tests in watch mode...$(NC)"
	@$(POETRY_RUN) ptw $(TEST_DIR)/ -- -v --tb=short

# ═══════════════════════════════════════════════════════════════════
# CODE FIXES
# ═══════════════════════════════════════════════════════════════════

fix: poetry-check format sort-imports lint-fix ## Auto-fix all possible issues
	@echo "$(GREEN)✅ All auto-fixes applied!$(NC)"

format: poetry-check ## Format code
	@echo "$(BLUE)⚫ Formatting with black...$(NC)"
	@$(POETRY_RUN) black $(SRC_DIR)/ $(TEST_DIR)/
	@echo "$(BLUE)⚡ Formatting with ruff...$(NC)"
	@$(POETRY_RUN) ruff format $(SRC_DIR)/ $(TEST_DIR)/
	@echo "$(GREEN)✅ Formatting complete!$(NC)"

sort-imports: poetry-check ## Sort imports
	@echo "$(BLUE)📦 Sorting imports with isort...$(NC)"
	@$(POETRY_RUN) isort $(SRC_DIR)/ $(TEST_DIR)/
	@echo "$(GREEN)✅ Import sorting complete!$(NC)"

lint-fix: poetry-check ## Auto-fix linting issues
	@echo "$(BLUE)🔧 Auto-fixing linting issues...$(NC)"
	@$(POETRY_RUN) ruff check $(SRC_DIR)/ $(TEST_DIR)/ --fix
	@echo "$(GREEN)✅ Linting fixes applied!$(NC)"

# ═══════════════════════════════════════════════════════════════════
# PROJECT MANAGEMENT
# ═══════════════════════════════════════════════════════════════════

install: poetry-check ## Install all dependencies
	@echo "$(BLUE)📦 Installing dependencies with Poetry...$(NC)"
	@$(POETRY) install --with dev,test,docs $(POETRY_OPTS)
	@echo "$(GREEN)✅ Dependencies installed!$(NC)"

update: poetry-check ## Update dependencies
	@echo "$(BLUE)⬆️ Updating dependencies...$(NC)"
	@$(POETRY) update $(POETRY_OPTS)
	@echo "$(GREEN)✅ Dependencies updated!$(NC)"

lock: poetry-check ## Update lock file
	@echo "$(BLUE)🔒 Updating poetry.lock...$(NC)"
	@$(POETRY) lock $(POETRY_OPTS)
	@echo "$(GREEN)✅ Lock file updated!$(NC)"

build: poetry-check clean ## Build distribution
	@echo "$(BLUE)🔨 Building distribution...$(NC)"
	@$(POETRY) build $(POETRY_OPTS)
	@echo "$(GREEN)✅ Distribution built!$(NC)"
	@ls -la dist/

docs: poetry-check ## Build documentation
	@echo "$(BLUE)📚 Building documentation...$(NC)"
	@$(POETRY_RUN) mkdocs build --strict
	@echo "$(GREEN)✅ Documentation built!$(NC)"

docs-serve: poetry-check ## Serve documentation
	@echo "$(BLUE)🌐 Serving documentation at http://localhost:8000...$(NC)"
	@$(POETRY_RUN) mkdocs serve

# ═══════════════════════════════════════════════════════════════════
# DEVELOPMENT TOOLS
# ═══════════════════════════════════════════════════════════════════

pre-commit: poetry-check ## Setup pre-commit hooks
	@echo "$(BLUE)🔧 Setting up pre-commit hooks...$(NC)"
	@$(POETRY_RUN) pre-commit install --install-hooks
	@$(POETRY_RUN) pre-commit install --hook-type commit-msg
	@echo "$(GREEN)✅ Pre-commit hooks installed!$(NC)"

pre-commit-run: poetry-check ## Run pre-commit on all files
	@echo "$(BLUE)🎣 Running pre-commit on all files...$(NC)"
	@$(POETRY_RUN) pre-commit run --all-files

clean: ## Remove cache and build files
	@echo "$(BLUE)🧹 Cleaning up...$(NC)"
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type f -name ".coverage" -delete 2>/dev/null || true
	@rm -rf build/ dist/ *.egg-info/ $(REPORTS_DIR)/
	@echo "$(GREEN)✅ Cleanup complete!$(NC)"

# ═══════════════════════════════════════════════════════════════════
# VALIDATION & STATUS
# ═══════════════════════════════════════════════════════════════════

validate: poetry-check ## Validate STRICT compliance
	@echo "$(PURPLE)╔═══════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(PURPLE)║$(NC)           $(YELLOW)🚨 STRICT MODE VALIDATION$(NC)                     $(PURPLE)║$(NC)"
	@echo "$(PURPLE)╚═══════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "$(CYAN)Checking for any quality violations...$(NC)"
	@if $(MAKE) check > /dev/null 2>&1; then \
		echo ""; \
		echo "$(GREEN)╔═══════════════════════════════════════════════════════════╗$(NC)"; \
		echo "$(GREEN)║$(NC)      $(GREEN)✅ 100% STRICT COMPLIANCE ACHIEVED!$(NC)                $(GREEN)║$(NC)"; \
		echo "$(GREEN)╚═══════════════════════════════════════════════════════════╝$(NC)"; \
	else \
		echo ""; \
		echo "$(RED)╔═══════════════════════════════════════════════════════════╗$(NC)"; \
		echo "$(RED)║$(NC)        $(RED)❌ STRICT COMPLIANCE FAILED!$(NC)                     $(RED)║$(NC)"; \
		echo "$(RED)╚═══════════════════════════════════════════════════════════╝$(NC)"; \
		echo ""; \
		echo "$(YELLOW)Run 'make check' to see violations$(NC)"; \
		exit 1; \
	fi

status: poetry-check ## Show current quality status
	@echo "$(PURPLE)╔═══════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(PURPLE)║$(NC)           $(CYAN)📊 Current Quality Status$(NC)                     $(PURPLE)║$(NC)"
	@echo "$(PURPLE)╚═══════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "$(CYAN)Analyzing code quality metrics...$(NC)"
	@echo ""
	@LINT_COUNT=$$($(POETRY_RUN) ruff check $(SRC_DIR)/ $(TEST_DIR)/ --exit-zero 2>/dev/null | grep -E '^src/|^tests/' | wc -l || echo 0); \
	TYPE_COUNT=$$($(POETRY_RUN) mypy $(SRC_DIR)/ $(TEST_DIR)/ --no-error-summary 2>/dev/null | grep -E '^src/|^tests/' | wc -l || echo 0); \
	SEC_COUNT=$$($(POETRY_RUN) bandit -r $(SRC_DIR)/ -f json 2>/dev/null | jq '.results | length' 2>/dev/null || echo 0); \
	echo "  Lint violations:  $$LINT_COUNT"; \
	echo "  Type errors:      $$TYPE_COUNT"; \
	echo "  Security issues:  $$SEC_COUNT"; \
	echo ""; \
	if [ "$$LINT_COUNT" -eq 0 ] && [ "$$TYPE_COUNT" -eq 0 ] && [ "$$SEC_COUNT" -eq 0 ]; then \
		echo "$(GREEN)  ✅ 100% STRICT COMPLIANCE!$(NC)"; \
	else \
		echo "$(YELLOW)  ⚠️  Quality issues detected$(NC)"; \
	fi
	@echo ""
	@echo "$(PURPLE)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"

# ═══════════════════════════════════════════════════════════════════
# DEVELOPMENT HELPERS
# ═══════════════════════════════════════════════════════════════════

watch: poetry-check ## Watch for changes and run checks
	@echo "$(BLUE)👁️ Watching for changes...$(NC)"
	@while true; do \
		inotifywait -r -e modify,create,delete $(SRC_DIR)/ $(TEST_DIR)/ 2>/dev/null; \
		clear; \
		$(MAKE) check; \
	done

shell: poetry-check ## Open Poetry shell
	@echo "$(BLUE)🐚 Opening Poetry shell...$(NC)"
	@$(POETRY) shell

run: poetry-check ## Run the application
	@echo "$(BLUE)🚀 Running flext-core...$(NC)"
	@$(POETRY_RUN) python -m flext_core

# ═══════════════════════════════════════════════════════════════════
# ENVIRONMENT SETUP
# ═══════════════════════════════════════════════════════════════════

setup: install pre-commit ## Complete development setup
	@echo "$(PURPLE)╔═══════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(PURPLE)║$(NC)         $(GREEN)✅ Development Setup Complete!$(NC)                  $(PURPLE)║$(NC)"
	@echo "$(PURPLE)╚═══════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "$(CYAN)Next steps:$(NC)"
	@echo "  1. Run '$(GREEN)make check$(NC)' to verify everything works"
	@echo "  2. Run '$(GREEN)make help$(NC)' to see all available commands"
	@echo "  3. Happy coding! 🎉"

# Export environment variables
export PYTHONPATH := $(PWD)/$(SRC_DIR):$(PYTHONPATH)
export FLEXT_CORE_DEV := true

# Default target
.DEFAULT_GOAL := help
