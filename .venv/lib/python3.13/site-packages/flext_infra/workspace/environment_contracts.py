"""Static contracts enforced on every managed direnv environment file.

The lint is pure (no subprocess): gates run it before the runtime smoke so
contract defects fail with precise messages, and the workspace sync runs it
after every generated write so a regression can never land silently.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

_UNGUARDED_DIRENV_DIR: Final[re.Pattern[str]] = re.compile(
    r"\$\{?DIRENV_DIR(?![\s]*[:\-])"
)
_QUOTED_ENV_TARGET: Final[re.Pattern[str]] = re.compile(
    r'^(?:source_env|watch_file)\s+"([^"]+)"\s*$'
)
_HOME_PREFIX: Final[re.Pattern[str]] = re.compile(r"^\$\{?HOME\}?(.*)$")


def _resolve_env_target(raw: str, root: Path, *, resolve_home: bool) -> Path | None:
    """Resolve one quoted target to a concrete path, or None when dynamic."""
    home_match = _HOME_PREFIX.match(raw)
    candidate = home_match.group(1) if home_match is not None else raw
    if candidate.startswith("~"):
        candidate = f"$HOME{candidate[1:]}"
        home_match = _HOME_PREFIX.match(candidate)
        candidate = home_match.group(1) if home_match is not None else candidate
    if "$" in candidate:
        return None
    if home_match is not None and not resolve_home:
        return None
    resolved = Path(candidate)
    if not resolved.is_absolute():
        resolved = root / resolved
    return resolved


def envrc_contract_violations(
    content: str, *, root: Path, resolve_home: bool = True
) -> tuple[str, ...]:
    """Return one message per direnv contract violation in ``content``.

    Contracts enforced:
    - ``DIRENV_DIR`` is never read unguarded: ``strict_env`` does not export
      it, so any ``${DIRENV_DIR}`` / ``${DIRENV_DIR#...}`` / ``$DIRENV_DIR``
      read breaks activation with an unbound-variable error. Guarded reads
      (``${DIRENV_DIR:-...}`` / ``${DIRENV_DIR-}``) stay legal.
    - Every literal ``source_env`` / ``watch_file`` target must exist. Targets
      derived from runtime variables (any remaining ``$``) are skipped. With
      ``resolve_home=False`` (generation-time lint) ``$HOME`` targets are also
      skipped because they describe machine state, not repository state.
    """
    violations: list[str] = []
    for match in _UNGUARDED_DIRENV_DIR.finditer(content):
        line = content.count("\n", 0, match.start()) + 1
        violations.append(
            f"line {line}: DIRENV_DIR read without a `:-` guard "
            f"(strict_env does not export it): {match.group(0)!r}"
        )
    for line_number, line in enumerate(content.splitlines(), start=1):
        target_match = _QUOTED_ENV_TARGET.match(line.strip())
        if target_match is None:
            continue
        resolved = _resolve_env_target(
            target_match.group(1), root, resolve_home=resolve_home
        )
        if resolved is None:
            continue
        if not resolved.exists():
            violations.append(
                f"line {line_number}: environment target does not exist: {resolved}"
            )
    return tuple(violations)


__all__: tuple[str, ...] = ("envrc_contract_violations",)
