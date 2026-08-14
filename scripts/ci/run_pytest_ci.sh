#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
PYTHON_BIN="${VENV_DIR}/bin/python"
PIP_BIN="${VENV_DIR}/bin/pip"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Virtualenv not found. Run 'make venv' or 'make setup' first."
  exit 1
fi

mkdir -p "${ROOT_DIR}/artifacts"

"${PIP_BIN}" install -q -U pip poetry
"${VENV_DIR}/bin/poetry" install --all-extras --all-groups

# Keep local editable ecosystem deps aligned in CI too.
chmod +x "${ROOT_DIR}/scripts/setup_org_libs.sh"
"${ROOT_DIR}/scripts/setup_org_libs.sh"

"${VENV_DIR}/bin/poetry" run pytest \
  -q \
  --maxfail=1 \
  --junitxml="${ROOT_DIR}/artifacts/pytest.xml" \
  --cov=src/flext_core \
  --cov-report=term-missing \
  --cov-report=xml:"${ROOT_DIR}/artifacts/coverage.xml"
