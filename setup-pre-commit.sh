#!/bin/bash
# FLEXT Core - Strict Pre-commit Setup
# This script installs and configures pre-commit hooks for enterprise-grade quality

set -euo pipefail

echo "🚀 FLEXT Core - Setting up STRICT pre-commit hooks..."
echo "=================================================="

# Check if we're in a git repository
if ! git rev-parse --git-dir >/dev/null 2>&1; then
    echo "❌ Error: Not in a git repository!"
    echo "Please run this script from the project root."
    exit 1
fi

# Activate virtual environment if available
if [ -f "/home/marlonsc/flext/.venv/bin/activate" ]; then
    echo "✅ Activating virtual environment..."
    source /home/marlonsc/flext/.venv/bin/activate
elif [ -f ".venv/bin/activate" ]; then
    echo "✅ Activating local virtual environment..."
    source .venv/bin/activate
else
    echo "⚠️  Warning: No virtual environment found"
fi

# Install pre-commit if not already installed
if ! command -v pre-commit &>/dev/null; then
    echo "📦 Installing pre-commit..."
    pip install pre-commit
else
    echo "✅ pre-commit already installed"
fi

# Install pre-commit hooks
echo "🔧 Installing pre-commit hooks..."
pre-commit install --install-hooks
pre-commit install --hook-type commit-msg

# Update all hooks to latest versions
echo "⬆️  Updating pre-commit hooks to latest versions..."
pre-commit autoupdate

# Run pre-commit on all files to check current status
echo ""
echo "🔍 Running pre-commit checks on all files..."
echo "=================================================="

# Run all hooks
if pre-commit run --all-files; then
    echo ""
    echo "✅ SUCCESS! All pre-commit checks passed!"
    echo "=================================================="
    echo "🎉 Your repository is now protected by STRICT quality gates:"
    echo ""
    echo "   🔒 Security scanning (detect-secrets, bandit)"
    echo "   ⚡ Code formatting (ruff-format)"
    echo "   🔥 Linting with 17 rule categories (ruff)"
    echo "   🛡️  Type checking in --strict mode (mypy)"
    echo "   📦 Import sorting (isort)"
    echo "   📋 Config file validation (YAML/TOML/JSON)"
    echo "   🚫 Python best practices enforcement"
    echo "   💬 Commit message standards (commitizen)"
    echo ""
    echo "Pre-commit will now run automatically on every commit!"
else
    echo ""
    echo "⚠️  Some checks failed. Please fix the issues above."
    echo "Run 'pre-commit run --all-files' to re-check."
fi

echo ""
echo "📝 Additional commands:"
echo "   pre-commit run --all-files    # Run all checks manually"
echo "   pre-commit run <hook-id>      # Run specific hook"
echo "   SKIP=<hook-id> git commit     # Skip specific hook (emergency only!)"
echo ""
