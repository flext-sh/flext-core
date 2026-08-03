#!/usr/bin/env bash
# Owner-Skill: .agents/skills/scripts-validation/SKILL.md
# install-git-hooks.sh — Install the Beads git hooks at the workspace root and
# apply the FLEXT agent-trailer guard.
#
# Canonical owner of git-hook provisioning for this workspace. Reproducible and
# idempotent: safe to run repeatedly and after every `bd hooks install`.
#
# Why the guard:
#   FLEXT law (R5 / ai-hub agent-law §12) forbids agent attribution trailers by
#   default. The Beads `prepare-commit-msg` shim's sole job is adding those
#   trailers, so it must be gated behind an explicit opt-in:
#       BD_ALLOW_AGENT_COMMIT_TRAILERS=1
#   `.github/scripts/check-beads-policy.sh` enforces the guard text is present
#   in the installed hook; `make check WHAT=coordination` fails without it.
#
# Mechanism:
#   `bd hooks install --chain` writes bd-managed sections between markers and
#   preserves any content OUTSIDE those markers across installs/upgrades. This
#   script re-applies bd's install, then injects the guard block above the bd
#   `--- BEGIN BEADS INTEGRATION ---` marker so it survives future bd installs.
#
# Usage:
#   make hooks
#   .github/scripts/install-git-hooks.sh [--verbose]

set -euo pipefail

VERBOSE="${1:-}"
WORKSPACE_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${WORKSPACE_ROOT}"

_log() {
	if [[ "${VERBOSE}" == "--verbose" ]]; then
		echo "[INFO] $*"
	fi
}

fail() {
	printf 'install-git-hooks: %s\n' "$*" >&2
	exit 1
}

command -v pre-commit >/dev/null 2>&1 || fail "pre-commit is not installed; install it before provisioning hooks"
command -v bd >/dev/null 2>&1 || fail "bd is not installed; install Beads before provisioning hooks"

# Why: install both staged workflow entry points before Beads chains its guard.
_log "Installing pre-commit and pre-push hooks at ${WORKSPACE_ROOT}"
pre-commit install -t pre-commit -t pre-push >/dev/null \
	|| fail "pre-commit hook installation failed"
_log "Installing Beads git hooks (chained) at ${WORKSPACE_ROOT}"
bd hooks install --chain >/dev/null || fail "bd hooks install --chain failed"

hook_path="$(git rev-parse --git-path hooks/prepare-commit-msg)"
[ -f "${hook_path}" ] || fail "prepare-commit-msg hook missing after bd hooks install"

_log "Applying FLEXT agent-trailer guard to ${hook_path}"
GUARD_TOKEN="BD_ALLOW_AGENT_COMMIT_TRAILERS" python3 - "${hook_path}" <<'PY'
import os
import pathlib
import sys

token = os.environ["GUARD_TOKEN"]
path = pathlib.Path(sys.argv[1])
text = path.read_text()

if token in text:
    # Guard already present (idempotent): nothing to do.
    sys.exit(0)

guard = (
    "# --- BEGIN FLEXT AGENT-TRAILER GUARD ---\n"
    "# Managed by .github/scripts/install-git-hooks.sh — do not hand-edit.\n"
    "# FLEXT law (R5): prepare-commit-msg must NOT add agent attribution\n"
    "# trailers unless the user opts in with BD_ALLOW_AGENT_COMMIT_TRAILERS=1.\n"
    "# The Beads shim below only adds trailers, so gate it here.\n"
    'if [ "${BD_ALLOW_AGENT_COMMIT_TRAILERS:-0}" != "1" ]; then\n'
    "  exit 0\n"
    "fi\n"
    "# --- END FLEXT AGENT-TRAILER GUARD ---\n"
)

lines = text.splitlines(keepends=True)
marker = "# --- BEGIN BEADS INTEGRATION"
insert_at = next(
    (i for i, line in enumerate(lines) if line.startswith(marker)),
    None,
)
if insert_at is None:
    raise SystemExit(
        "beads integration marker not found; cannot place guard deterministically"
    )

# Insert the guard immediately before the bd-managed section (outside markers,
# so `bd hooks install` preserves it on future upgrades).
lines[insert_at:insert_at] = [guard]
path.write_text("".join(lines))
PY

grep -q 'BD_ALLOW_AGENT_COMMIT_TRAILERS' "${hook_path}" \
	|| fail "guard token missing after injection"
grep -q 'bd hooks run prepare-commit-msg' "${hook_path}" \
	|| fail "bd delegation missing; refusing to leave hook without beads integration"
[ -f "$(git rev-parse --git-path hooks/pre-commit)" ] \
	|| fail "pre-commit hook missing after provisioning"
[ -f "$(git rev-parse --git-path hooks/pre-push)" ] \
	|| fail "pre-push hook missing after provisioning"

echo "install-git-hooks: prepare-commit-msg guarded (BD_ALLOW_AGENT_COMMIT_TRAILERS opt-in)"
