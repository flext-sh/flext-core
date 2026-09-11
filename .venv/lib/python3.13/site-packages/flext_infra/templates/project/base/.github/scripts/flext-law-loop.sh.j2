#!/usr/bin/env bash
# flext-law-loop.sh — recurring strict-FLEXT rule enforcer.
#
# Every INTERVAL seconds this runs the flext-infra codegen/refactor strict-rule
# fixers inside a THROWAWAY git worktree, validates that worktree green
# (ruff -> pyrefly -> mypy[memory-capped] -> pytest), and only then applies the
# vetted changes to the real workspace and commits. If the worktree is not
# green the cycle is discarded and the workspace is left untouched.
#
# Operator rules honoured:
#   * validate in a temporary worktree BEFORE applying to the whole workspace
#   * mypy ALWAYS inherits the canonical Make memory and wall-time ceilings
#   * never mutate the workspace on a red cycle (fix-forward, no bypass)
#
# Usage:
#   scripts/flext-law-loop.sh [--once] [--interval SECONDS] [--apply]
#
#   --once            run a single cycle and exit (default: loop forever)
#   --interval N      seconds between cycles (default: 1200 = 20 min)
#   --apply           sync+commit vetted changes to the workspace
#                     (default: dry-run — validate only, never touch workspace)
set -euo pipefail

# This script lives at .github/scripts/flext-law-loop.sh; the composition root
# is therefore two levels up.
WORKSPACE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." >/dev/null 2>&1 && pwd)"
cd "${WORKSPACE_ROOT}"

# A loop owns the entire disposable-worktree namespace.  A non-blocking lock
# prevents a second invocation from deleting an active cycle during startup
# cleanup.
GIT_COMMON_DIR="$(git rev-parse --git-common-dir)"
exec 9>"${GIT_COMMON_DIR}/flext-law-loop.lock"
if ! flock -n 9; then
  printf 'flext-law-loop: another loop instance is already active\n' >&2
  exit 1
fi

INTERVAL="${FLEXT_LAW_INTERVAL:-1200}"
APPLY=0
ONCE=0
MYPY_MEMORY_LIMIT_MB="${MYPY_MEMORY_LIMIT_MB:-6144}"
MYPY_TIMEOUT_SECONDS="${MYPY_TIMEOUT_SECONDS:-600}"
FLEXT_INFRA=(uv run --all-packages flext-infra)
REPORT_DIR="${WORKSPACE_ROOT}/.reports/flext-law"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --once) ONCE=1 ;;
    --apply) APPLY=1 ;;
    --interval) shift; INTERVAL="$1" ;;
    --interval=*) INTERVAL="${1#*=}" ;;
    *) printf 'flext-law-loop: unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
  shift
done

export MYPY_MEMORY_LIMIT_MB MYPY_TIMEOUT_SECONDS

log() { printf '[flext-law %(%Y-%m-%dT%H:%M:%SZ)T] %s\n' -1 "$*"; }

# Strict-rule fixers to sweep, in dependency-safe order. Each is applied inside
# the worktree; flext-infra's own worktree_transaction validates every micro
# edit (ruff+pyrefly) before it is materialised.
FIXER_SPECS=(
  "check fix-enforcement --safe-only --check-after"
  "refactor modernize-patterns"
  "refactor modernize-pydantic"
  "refactor modernize-logging"
  "refactor modernize-result-di"
  "refactor namespace-enforce"
)

validate_worktree() {
  local root="$1"
  log "validate: ruff"
  make -C "${root}" check WHAT=lint || return 1
  log "validate: pyrefly"
  make -C "${root}" check WHAT=pyrefly || return 1
  log "validate: mypy (bounded ${MYPY_MEMORY_LIMIT_MB}MB/${MYPY_TIMEOUT_SECONDS}s)"
  make -C "${root}" check WHAT=mypy \
    MYPY_MEMORY_LIMIT_MB="${MYPY_MEMORY_LIMIT_MB}" \
    MYPY_TIMEOUT_SECONDS="${MYPY_TIMEOUT_SECONDS}" || return 1
  log "validate: pytest"
  make -C "${root}" test || return 1
  return 0
}

run_cycle() {
  mkdir -p "${REPORT_DIR}"
  local stamp worktree base
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  base="$(git rev-parse HEAD)"
  # Keep our worktree OUT of .worktrees/ — flext-infra fixers own that path for
  # their internal transactions and prune it, which would delete ours mid-cycle.
  worktree="${WORKSPACE_ROOT}/.flext-law-worktrees/flext-law-${stamp}"

  # Self-heal: a cycle killed mid-run (SIGKILL bypasses the trap) can leave
  # stale loop worktrees behind — and a fixer may have nested its own
  # transaction worktree inside ours. Force-remove any registered worktree
  # living under our base dir, delete the trees, then prune the git metadata.
  local stale
  git worktree list --porcelain 2>/dev/null | awk '/^worktree /{print $2}' \
    | grep -F "${WORKSPACE_ROOT}/.flext-law-worktrees/" \
    | sort -r \
    | while IFS= read -r stale; do
        git worktree remove --force "${stale}" 2>/dev/null || true
      done
  rm -rf "${WORKSPACE_ROOT}/.flext-law-worktrees" 2>/dev/null || true
  git worktree prune 2>/dev/null || true


  log "cycle start: base=${base} worktree=${worktree} apply=${APPLY}"
  mkdir -p "${WORKSPACE_ROOT}/.flext-law-worktrees"
  git worktree add --quiet --detach "${worktree}" "${base}"

  # Always clean up the throwaway worktree, green or red (handles nested fixer
  # transaction worktrees by removing the directory after detaching it).
  trap 'git worktree remove --force "${worktree}" 2>/dev/null || true; rm -rf "${worktree}" 2>/dev/null || true; git worktree prune 2>/dev/null || true' RETURN

  local spec group cmd changed=0
  local -a command_parts flags
  for spec in "${FIXER_SPECS[@]}"; do
    read -r -a command_parts <<<"${spec}"
    group="${command_parts[0]}"
    cmd="${command_parts[1]}"
    flags=("${command_parts[@]:2}")
    log "fixer: ${group} ${cmd} ${flags[*]:-}"
    # Apply inside the worktree; flext-infra validates each micro-transaction.
    if "${FLEXT_INFRA[@]}" "${group}" "${cmd}" --workspace "${worktree}" --apply "${flags[@]}" \
        >"${REPORT_DIR}/${stamp}-${group}-${cmd}.log" 2>&1; then
      :
    else
      # rc!=0 from a fixer means "violations found/applied" or a real error;
      # the worktree validation below is the authoritative gate.
      log "fixer ${group} ${cmd} returned non-zero (see ${REPORT_DIR}/${stamp}-${group}-${cmd}.log)"
    fi
  done

  local worktree_status
  worktree_status="$(git -C "${worktree}" status --porcelain)"
  if [ -z "${worktree_status}" ]; then
    log "cycle: no strict-rule changes produced — nothing to validate/apply"
    return 0
  fi
  changed=1

  if ! validate_worktree "${worktree}"; then
    log "cycle RED: worktree validation failed — workspace left untouched"
    git -C "${worktree}" --no-pager diff --stat | tee "${REPORT_DIR}/${stamp}-REJECTED.diffstat" || true
    return 1
  fi

  log "cycle GREEN: worktree validated"
  if [ "${APPLY}" -ne 1 ]; then
    git -C "${worktree}" --no-pager diff --stat | tee "${REPORT_DIR}/${stamp}-VETTED.diffstat" || true
    log "dry-run mode: not syncing to workspace (re-run with --apply to land)"
    return 0
  fi

  # Materialise the vetted diff onto the real workspace and validate once more
  # in place before committing (defence in depth).
  log "apply: syncing vetted diff to workspace"
  git -C "${worktree}" add --intent-to-add -- .
  git -C "${worktree}" diff --binary "${base}" -- . >"${REPORT_DIR}/${stamp}-APPLIED.patch"
  if [ -s "${REPORT_DIR}/${stamp}-APPLIED.patch" ]; then
    local workspace_status
    workspace_status="$(git status --porcelain)"
    if [ -n "${workspace_status}" ]; then
      log "apply refused: workspace has existing changes"
      return 1
    fi
    git apply --index --3way "${REPORT_DIR}/${stamp}-APPLIED.patch"
    if ! validate_worktree "${WORKSPACE_ROOT}"; then
      log "apply RED: reverting the vetted patch"
      if git apply --index --reverse --3way "${REPORT_DIR}/${stamp}-APPLIED.patch"; then
        log "apply RED: revert succeeded; workspace left unchanged"
      else
        log "apply RED: revert FAILED; workspace still holds ${REPORT_DIR}/${stamp}-APPLIED.patch — manual cleanup required"
      fi
      return 1
    fi
    git commit -m "fix: apply validated flext-law sweep"
  fi
  [ "${changed}" -eq 1 ] || return 0
  return 0
}

log "flext-law loop starting (interval=${INTERVAL}s apply=${APPLY} once=${ONCE} mypy_cap=${MYPY_MEMORY_LIMIT_MB}MB/${MYPY_TIMEOUT_SECONDS}s)"
while true; do
  if run_cycle; then
    log "cycle ok"
  else
    log "cycle failed (workspace untouched); will retry next interval"
  fi
  [ "${ONCE}" -eq 1 ] && break
  log "sleeping ${INTERVAL}s until next cycle"
  sleep "${INTERVAL}"
done
