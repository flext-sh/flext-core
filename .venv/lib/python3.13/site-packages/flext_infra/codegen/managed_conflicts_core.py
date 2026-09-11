"""Import-light managed TOML conflict recovery.

This module deliberately depends only on the standard library so the process
entrypoint can repair owner-declared metadata before FLEXT facades import it.
"""

from __future__ import annotations

import re
from collections.abc import Sequence


class ManagedConflictError(ValueError):
    """Reject malformed or owner-undeclared merge conflict blocks."""


_TOML_SECTION_RE = re.compile(r"^\s*\[([^\[\]]+)\]\s*(?:#.*)?$")


def recover_managed_toml(content: str, *, conflict_sections: Sequence[str]) -> str:
    """Choose the current projection inside explicitly declared TOML sections."""
    if "<<<<<<< " not in content:
        return content
    allowed = frozenset(conflict_sections)
    lines = content.splitlines(keepends=True)
    recovered: list[str] = []
    section = ""
    index = 0
    while index < len(lines):
        line = lines[index]
        section_match = _TOML_SECTION_RE.match(line.rstrip("\r\n"))
        if section_match is not None:
            section = section_match.group(1)
        if not line.startswith("<<<<<<< "):
            recovered.append(line)
            index += 1
            continue
        if section not in allowed:
            msg = (
                "merge conflict is outside owner-declared TOML sections: "
                f"{section or '<document-root>'}"
            )
            raise ManagedConflictError(msg)
        index += 1
        current: list[str] = []
        while index < len(lines) and not lines[index].startswith("======="):
            if lines[index].startswith(("<<<<<<< ", ">>>>>>> ")):
                msg = "nested or malformed TOML merge conflict"
                raise ManagedConflictError(msg)
            current.append(lines[index])
            index += 1
        if index >= len(lines):
            msg = "TOML merge conflict has no separator"
            raise ManagedConflictError(msg)
        index += 1
        while index < len(lines) and not lines[index].startswith(">>>>>>> "):
            if lines[index].startswith(("<<<<<<< ", "=======")):
                msg = "nested or malformed TOML merge conflict"
                raise ManagedConflictError(msg)
            index += 1
        if index >= len(lines):
            msg = "TOML merge conflict has no closing marker"
            raise ManagedConflictError(msg)
        recovered.extend(current)
        index += 1
    return "".join(recovered)


__all__: tuple[str, ...] = ("ManagedConflictError", "recover_managed_toml")
