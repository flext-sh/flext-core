#!/usr/bin/env bash
set -euo pipefail

# Sync FLEXT ecosystem repositories from a GitHub org on the same branch as this repo.
# Defaults can be overridden via environment variables:
# - FLEXT_GH_ORG (default: flext-sh)
# - FLEXT_ORG_REPOS (space-separated repo names)
# - FLEXT_ORG_LIBS_DIR (default: .deps/org)
# - FLEXT_ORG_BRANCH (default: current git branch)

ROOT_DIR="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
ORG="${FLEXT_GH_ORG:-flext-sh}"
REPOS="${FLEXT_ORG_REPOS:-flext-cli flext-infra flext-tests}"
LIBS_DIR="${FLEXT_ORG_LIBS_DIR:-${ROOT_DIR}/.deps/org}"
CURRENT_BRANCH="${FLEXT_ORG_BRANCH:-$(git -C "${ROOT_DIR}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo main)}"
VENV_PIP="${ROOT_DIR}/.venv/bin/pip"

mkdir -p "${LIBS_DIR}"

if [[ ! -x "${VENV_PIP}" ]]; then
  echo "Virtualenv pip not found at ${VENV_PIP}. Run 'make venv' first."
  exit 1
fi

for repo in ${REPOS}; do
  repo_dir="${LIBS_DIR}/${repo}"
  remote_url="https://github.com/${ORG}/${repo}.git"

  if [[ -d "${repo_dir}/.git" ]]; then
    echo "==> Updating ${repo}..."
    git -C "${repo_dir}" fetch --all --prune
  else
    echo "==> Cloning ${remote_url}..."
    git clone "${remote_url}" "${repo_dir}"
  fi

  if git -C "${repo_dir}" show-ref --verify --quiet "refs/remotes/origin/${CURRENT_BRANCH}"; then
    target_branch="${CURRENT_BRANCH}"
  else
    target_branch="main"
  fi

  echo "==> Using branch '${target_branch}' for ${repo}"
  git -C "${repo_dir}" checkout "${target_branch}"
  git -C "${repo_dir}" pull --ff-only origin "${target_branch}"

  echo "==> Installing ${repo} in editable mode"
  "${VENV_PIP}" install -q -e "${repo_dir}"
done

echo "==> FLEXT org dependencies synced in ${LIBS_DIR}"
