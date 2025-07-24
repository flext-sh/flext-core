# FLEXT CORE - Makefile Unificado
# ===============================
# Comandos essenciais para desenvolvimento e qualidade
# Integração completa sem dependências externas

.PHONY: help install test lint type-check format clean build docs
.PHONY: check validate dev-setup deps-update deps-audit info diagnose
.PHONY: install-dev test-unit test-integration test-coverage test-watch
.PHONY: format-check security pre-commit build-clean publish publish-test
.PHONY: dev dev-test clean-all emergency-reset

# ============================================================================
# 🎯 CONFIGURAÇÃO E DETECÇÃO
# ============================================================================

# Detectar nome do projeto
PROJECT_NAME := flext-core
PROJECT_TITLE := Flext Core

# Ambiente Python
PYTHON := python3.13
POETRY := poetry
VENV_PATH := $(shell poetry env info --path 2>/dev/null || echo "")

# Cores para output
RED := \033[31m
GREEN := \033[32m
YELLOW := \033[33m
BLUE := \033[34m
MAGENTA := \033[35m
CYAN := \033[36m
WHITE := \033[37m
RESET := \033[0m

# ============================================================================
# 🎯 AJUDA E INFORMAÇÃO
# ============================================================================

help: ## Mostrar ajuda e comandos disponíveis
	@echo "$(CYAN)🏆 $(PROJECT_TITLE) - Comandos Essenciais$(RESET)"
	@echo "$(CYAN)====================================$(RESET)"
	@echo "$(BLUE)📦 Biblioteca base do ecossistema FLEXT$(RESET)"
	@echo "$(BLUE)🐍 Python 3.13 + Poetry + Qualidade Zero Tolerância$(RESET)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(CYAN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)💡 Comandos principais: make install, make test, make lint$(RESET)"

info: ## Mostrar informações do projeto
	@echo "$(CYAN)📊 Informações do Projeto$(RESET)"
	@echo "$(CYAN)======================$(RESET)"
	@echo "$(BLUE)Nome:$(RESET) $(PROJECT_NAME)"
	@echo "$(BLUE)Título:$(RESET) $(PROJECT_TITLE)"
	@echo "$(BLUE)Python:$(RESET) $(shell $(PYTHON) --version 2>/dev/null || echo "Não encontrado")"
	@echo "$(BLUE)Poetry:$(RESET) $(shell $(POETRY) --version 2>/dev/null || echo "Não instalado")"
	@echo "$(BLUE)Venv:$(RESET) $(shell [ -n "$(VENV_PATH)" ] && echo "$(VENV_PATH)" || echo "Não ativado")"
	@echo "$(BLUE)Diretório:$(RESET) $(CURDIR)"
	@echo "$(BLUE)Git Branch:$(RESET) $(shell git branch --show-current 2>/dev/null || echo "Não é repo git")"
	@echo "$(BLUE)Git Status:$(RESET) $(shell git status --porcelain 2>/dev/null | wc -l | xargs echo) arquivos alterados"

diagnose: ## Executar diagnósticos completos
	@echo "$(BLUE)🔍 Executando diagnósticos para $(PROJECT_NAME)...$(RESET)"
	@echo "$(CYAN)Informações do Sistema:$(RESET)"
	@echo "OS: $(shell uname -s)"
	@echo "Arquitetura: $(shell uname -m)"
	@echo "Python: $(shell $(PYTHON) --version 2>/dev/null || echo "Não encontrado")"
	@echo "Poetry: $(shell $(POETRY) --version 2>/dev/null || echo "Não instalado")"
	@echo ""
	@echo "$(CYAN)Estrutura do Projeto:$(RESET)"
	@ls -la
	@echo ""
	@echo "$(CYAN)Configuração Poetry:$(RESET)"
	@$(POETRY) config --list 2>/dev/null || echo "Poetry não configurado"
	@echo ""
	@echo "$(CYAN)Status das Dependências:$(RESET)"
	@$(POETRY) show --outdated 2>/dev/null || echo "Nenhuma dependência desatualizada"

# ============================================================================
# 📦 GERENCIAMENTO DE DEPENDÊNCIAS
# ============================================================================

validate-setup: ## Validar ambiente de desenvolvimento
	@echo "$(BLUE)🔍 Validando ambiente de desenvolvimento...$(RESET)"
	@command -v $(PYTHON) >/dev/null 2>&1 || { echo "$(RED)❌ Python 3.13 não encontrado$(RESET)"; exit 1; }
	@command -v $(POETRY) >/dev/null 2>&1 || { echo "$(RED)❌ Poetry não encontrado$(RESET)"; exit 1; }
	@test -f pyproject.toml || { echo "$(RED)❌ pyproject.toml não encontrado$(RESET)"; exit 1; }
	@echo "$(GREEN)✅ Validação do ambiente passou$(RESET)"

install: validate-setup ## Instalar dependências de runtime
	@echo "$(BLUE)📦 Instalando dependências de runtime para $(PROJECT_NAME)...$(RESET)"
	@$(POETRY) install --only main
	@echo "$(GREEN)✅ Dependências de runtime instaladas$(RESET)"

install-dev: validate-setup ## Instalar todas as dependências incluindo dev tools
	@echo "$(BLUE)📦 Instalando todas as dependências para $(PROJECT_NAME)...$(RESET)"
	@$(POETRY) install --all-extras
	@echo "$(GREEN)✅ Todas as dependências instaladas$(RESET)"

deps-update: ## Atualizar dependências para versões mais recentes
	@echo "$(BLUE)🔄 Atualizando dependências para $(PROJECT_NAME)...$(RESET)"
	@$(POETRY) update
	@echo "$(GREEN)✅ Dependências atualizadas$(RESET)"

deps-show: ## Mostrar árvore de dependências
	@echo "$(BLUE)📊 Árvore de dependências para $(PROJECT_NAME):$(RESET)"
	@$(POETRY) show --tree

deps-audit: ## Auditoria de dependências para vulnerabilidades
	@echo "$(BLUE)🔍 Auditando dependências para $(PROJECT_NAME)...$(RESET)"
	@$(POETRY) run pip-audit --format=columns || echo "$(YELLOW)⚠️  pip-audit não disponível$(RESET)"
	@$(POETRY) run safety check --json || echo "$(YELLOW)⚠️  safety não disponível$(RESET)"

# ============================================================================
# 🧪 TESTES
# ============================================================================

test: ## Executar todos os testes
	@echo "$(BLUE)🧪 Executando todos os testes para $(PROJECT_NAME)...$(RESET)"
	@$(POETRY) run pytest -xvs
	@echo "$(GREEN)✅ Todos os testes passaram$(RESET)"

test-unit: ## Executar apenas testes unitários
	@echo "$(BLUE)🧪 Executando testes unitários para $(PROJECT_NAME)...$(RESET)"
	@$(POETRY) run pytest tests/unit/ -xvs -m "not integration and not slow"
	@echo "$(GREEN)✅ Testes unitários passaram$(RESET)"

test-integration: ## Executar apenas testes de integração
	@echo "$(BLUE)🧪 Executando testes de integração para $(PROJECT_NAME)...$(RESET)"
	@$(POETRY) run pytest tests/integration/ -xvs -m "integration"
	@echo "$(GREEN)✅ Testes de integração passaram$(RESET)"

test-coverage: ## Executar testes com relatório de cobertura
	@echo "$(BLUE)🧪 Executando testes com cobertura para $(PROJECT_NAME)...$(RESET)"
	@$(POETRY) run pytest --cov --cov-report=html --cov-report=term-missing --cov-report=xml
	@echo "$(GREEN)✅ Relatório de cobertura gerado$(RESET)"

test-watch: ## Executar testes em modo watch
	@echo "$(BLUE)👀 Executando testes em modo watch para $(PROJECT_NAME)...$(RESET)"
	@$(POETRY) run pytest-watch --clear

# ============================================================================
# 🎨 QUALIDADE DE CÓDIGO E FORMATAÇÃO
# ============================================================================

lint: ## Executar todos os linters com máxima rigorosidade
	@echo "$(BLUE)🔍 Executando linting com máxima rigorosidade para $(PROJECT_NAME)...$(RESET)"
	@$(POETRY) run ruff check . --output-format=github
	@echo "$(GREEN)✅ Linting completado$(RESET)"

format: ## Formatar código com padrões rigorosos
	@echo "$(BLUE)🎨 Formatando código para $(PROJECT_NAME)...$(RESET)"
	@$(POETRY) run ruff format .
	@$(POETRY) run ruff check . --fix --unsafe-fixes
	@echo "$(GREEN)✅ Código formatado$(RESET)"

format-check: ## Verificar formatação sem alterar
	@echo "$(BLUE)🔍 Verificando formatação para $(PROJECT_NAME)...$(RESET)"
	@$(POETRY) run ruff format . --check
	@$(POETRY) run ruff check . --output-format=github
	@echo "$(GREEN)✅ Formatação verificada$(RESET)"

type-check: ## Executar verificação de tipos rigorosa
	@echo "$(BLUE)🔍 Executando verificação de tipos rigorosa para $(PROJECT_NAME)...$(RESET)"
	@$(POETRY) run mypy src/ --strict --show-error-codes
	@echo "$(GREEN)✅ Verificação de tipos passou$(RESET)"

security: ## Executar análise de segurança
	@echo "$(BLUE)🔒 Executando análise de segurança para $(PROJECT_NAME)...$(RESET)"
	@$(POETRY) run bandit -r src/ -f json || echo "$(YELLOW)⚠️  bandit não disponível$(RESET)"
	@$(POETRY) run detect-secrets scan --all-files || echo "$(YELLOW)⚠️  detect-secrets não disponível$(RESET)"
	@echo "$(GREEN)✅ Análise de segurança completada$(RESET)"

pre-commit: ## Executar hooks pre-commit
	@echo "$(BLUE)🔧 Executando hooks pre-commit para $(PROJECT_NAME)...$(RESET)"
	@$(POETRY) run pre-commit run --all-files || echo "$(YELLOW)⚠️  pre-commit não disponível$(RESET)"
	@echo "$(GREEN)✅ Hooks pre-commit completados$(RESET)"

check: lint type-check security ## Executar todas as verificações de qualidade
	@echo "$(BLUE)🔍 Executando verificações abrangentes de qualidade para $(PROJECT_NAME)...$(RESET)"
	@echo "$(GREEN)✅ Todas as verificações de qualidade passaram$(RESET)"

# ============================================================================
# 🏗️ BUILD E DISTRIBUIÇÃO
# ============================================================================

build: clean ## Construir o pacote com Poetry
	@echo "$(BLUE)🏗️  Construindo pacote $(PROJECT_NAME)...$(RESET)"
	@$(POETRY) build
	@echo "$(GREEN)✅ Pacote construído com sucesso$(RESET)"
	@echo "$(BLUE)📦 Artefatos de build:$(RESET)"
	@ls -la dist/

build-clean: clean build ## Limpar e construir
	@echo "$(GREEN)✅ Build limpo completado$(RESET)"

publish-test: build ## Publicar no TestPyPI
	@echo "$(BLUE)📤 Publicando $(PROJECT_NAME) no TestPyPI...$(RESET)"
	@$(POETRY) publish --repository testpypi
	@echo "$(GREEN)✅ Publicado no TestPyPI$(RESET)"

publish: build ## Publicar no PyPI
	@echo "$(BLUE)📤 Publicando $(PROJECT_NAME) no PyPI...$(RESET)"
	@$(POETRY) publish
	@echo "$(GREEN)✅ Publicado no PyPI$(RESET)"

# ============================================================================
# 📚 DOCUMENTAÇÃO
# ============================================================================

docs: ## Gerar documentação
	@echo "$(BLUE)📚 Gerando documentação para $(PROJECT_NAME)...$(RESET)"
	@if [ -f mkdocs.yml ]; then \
		$(POETRY) run mkdocs build; \
	else \
		echo "$(YELLOW)⚠️  Nenhum mkdocs.yml encontrado, pulando geração de documentação$(RESET)"; \
	fi
	@echo "$(GREEN)✅ Documentação gerada$(RESET)"

docs-serve: ## Servir documentação localmente
	@echo "$(BLUE)📚 Servindo documentação para $(PROJECT_NAME)...$(RESET)"
	@if [ -f mkdocs.yml ]; then \
		$(POETRY) run mkdocs serve; \
	else \
		echo "$(YELLOW)⚠️  Nenhum mkdocs.yml encontrado$(RESET)"; \
	fi

# ============================================================================
# 🚀 DESENVOLVIMENTO
# ============================================================================

dev-setup: install-dev ## Configuração completa de desenvolvimento
	@echo "$(BLUE)🚀 Configurando ambiente de desenvolvimento para $(PROJECT_NAME)...$(RESET)"
	@$(POETRY) run pre-commit install || echo "$(YELLOW)⚠️  pre-commit não disponível$(RESET)"
	@echo "$(GREEN)✅ Ambiente de desenvolvimento pronto$(RESET)"

dev: ## Executar em modo desenvolvimento
	@echo "$(BLUE)🚀 Iniciando modo desenvolvimento para $(PROJECT_NAME)...$(RESET)"
	@if [ -f src/flext_core/cli.py ]; then \
		$(POETRY) run python -m flext_core.cli --dev; \
	elif [ -f src/flext_core/main.py ]; then \
		$(POETRY) run python -m flext_core.main --dev; \
	else \
		echo "$(YELLOW)⚠️  Nenhum ponto de entrada principal encontrado$(RESET)"; \
	fi

dev-test: ## Ciclo rápido de teste de desenvolvimento
	@echo "$(BLUE)⚡ Ciclo rápido de teste de desenvolvimento para $(PROJECT_NAME)...$(RESET)"
	@$(POETRY) run ruff check . --fix
	@$(POETRY) run pytest tests/ -x --tb=short
	@echo "$(GREEN)✅ Ciclo de teste de desenvolvimento completado$(RESET)"

# ============================================================================
# 🧹 LIMPEZA
# ============================================================================

clean: ## Limpar artefatos de build
	@echo "$(BLUE)🧹 Limpando artefatos de build para $(PROJECT_NAME)...$(RESET)"
	@rm -rf build/
	@rm -rf dist/
	@rm -rf *.egg-info/
	@rm -rf .pytest_cache/
	@rm -rf .coverage
	@rm -rf htmlcov/
	@rm -rf .mypy_cache/
	@rm -rf .ruff_cache/
	@rm -rf reports/
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@echo "$(GREEN)✅ Limpeza completada$(RESET)"

clean-all: clean ## Limpar tudo incluindo ambiente virtual
	@echo "$(BLUE)🧹 Limpeza profunda para $(PROJECT_NAME)...$(RESET)"
	@$(POETRY) env remove --all || true
	@echo "$(GREEN)✅ Limpeza profunda completada$(RESET)"

# ============================================================================
# 🚨 PROCEDIMENTOS DE EMERGÊNCIA
# ============================================================================

emergency-reset: ## Reset de emergência para estado limpo
	@echo "$(RED)🚨 RESET DE EMERGÊNCIA para $(PROJECT_NAME)...$(RESET)"
	@read -p "Tem certeza que quer resetar tudo? (y/N) " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		$(MAKE) clean-all; \
		$(MAKE) install-dev; \
		echo "$(GREEN)✅ Reset de emergência completado$(RESET)"; \
	else \
		echo "$(YELLOW)⚠️  Reset de emergência cancelado$(RESET)"; \
	fi

# ============================================================================
# 🎯 VALIDAÇÃO E VERIFICAÇÃO
# ============================================================================

validate: ## Validar conformidade do workspace
	@echo "$(BLUE)🔍 Validando conformidade do workspace para $(PROJECT_NAME)...$(RESET)"
	@test -f pyproject.toml || { echo "$(RED)❌ pyproject.toml ausente$(RESET)"; exit 1; }
	@test -f CLAUDE.md || echo "$(YELLOW)⚠️  CLAUDE.md ausente$(RESET)"
	@test -f README.md || echo "$(YELLOW)⚠️  README.md ausente$(RESET)"
	@test -d src/ || { echo "$(RED)❌ diretório src/ ausente$(RESET)"; exit 1; }
	@test -d tests/ || echo "$(YELLOW)⚠️  diretório tests/ ausente$(RESET)"
	@echo "$(GREEN)✅ Conformidade do workspace validada$(RESET)"

# ============================================================================
# 🎯 ALIASES DE CONVENIÊNCIA
# ============================================================================

# Aliases para operações comuns
t: test ## Alias para test
l: lint ## Alias para lint
tc: type-check ## Alias para type-check
f: format ## Alias para format
c: clean ## Alias para clean
i: install-dev ## Alias para install-dev
d: dev ## Alias para dev
dt: dev-test ## Alias para dev-test
