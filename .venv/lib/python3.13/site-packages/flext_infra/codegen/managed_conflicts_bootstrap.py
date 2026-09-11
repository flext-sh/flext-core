"""Import-light bootstrap for owner-declared managed metadata recovery."""

from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Sequence
from pathlib import Path
from stat import S_IMODE

from flext_infra.codegen.managed_conflicts_core import (
    ManagedConflictError,
    recover_managed_toml,
)


class ManagedConflictBootstrapError(RuntimeError):
    """Fail before facade imports when bootstrap ownership is ambiguous."""


_MANAGED_FILES_HEADER = "    managed_files:"
_MANAGED_ITEM_PREFIX = "      - path: "
_CONFLICT_SECTIONS_RE = re.compile(
    r"^        conflict_sections: "
    r"\[([A-Za-z0-9_.-]+(?:, [A-Za-z0-9_.-]+)*)\]$"
)


def _option_value(arguments: Sequence[str], option: str) -> str | None:
    """Read one non-duplicated CLI option from its split or equals form."""
    values: list[str] = []
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == option:
            index += 1
            if index >= len(arguments) or arguments[index].startswith("--"):
                msg = f"{option} requires one value"
                raise ManagedConflictBootstrapError(msg)
            values.append(arguments[index])
        elif argument.startswith(f"{option}="):
            values.append(argument.partition("=")[2])
        index += 1
    if len(values) > 1:
        msg = f"{option} cannot be repeated"
        raise ManagedConflictBootstrapError(msg)
    return values[0] if values else None


def _codegen_configuration_path() -> Path:
    """Resolve the same codegen SSOT in editable-source and wheel layouts."""
    module = Path(__file__).resolve()
    candidates = (
        module.parents[1] / "config" / "codegen.yaml",
        module.parents[3] / "config" / "codegen.yaml",
    )
    existing = tuple(path for path in candidates if path.is_file())
    if not existing:
        msg = "managed conflict bootstrap cannot locate config/codegen.yaml"
        raise ManagedConflictBootstrapError(msg)
    if len(existing) > 1:
        payloads = {path.read_bytes() for path in existing}
        if len(payloads) != 1:
            msg = "source and packaged codegen configuration differ"
            raise ManagedConflictBootstrapError(msg)
    return existing[0]


def _declared_pyproject_sections(configuration: Path) -> tuple[str, ...]:
    """Extract the canonical inline declaration without importing YAML facades."""
    lines = configuration.read_text(encoding="utf-8").splitlines()
    headers = tuple(
        index for index, line in enumerate(lines) if line == _MANAGED_FILES_HEADER
    )
    if len(headers) != 1:
        msg = "codegen configuration must declare managed_files exactly once"
        raise ManagedConflictBootstrapError(msg)
    pyproject_items = 0
    declarations: list[tuple[str, ...]] = []
    current_path = ""
    for line in lines[headers[0] + 1 :]:
        if line.startswith("    ") and not line.startswith("      ") and line.strip():
            break
        if line.startswith(_MANAGED_ITEM_PREFIX):
            current_path = line.removeprefix(_MANAGED_ITEM_PREFIX).strip()
            if current_path == "pyproject.toml":
                pyproject_items += 1
            continue
        if current_path != "pyproject.toml" or "conflict_sections:" not in line:
            continue
        matched = _CONFLICT_SECTIONS_RE.fullmatch(line)
        if matched is None:
            msg = "pyproject conflict_sections must use the canonical inline list"
            raise ManagedConflictBootstrapError(msg)
        declarations.append(tuple(matched.group(1).split(", ")))
    if pyproject_items != 1 or len(declarations) != 1 or not declarations[0]:
        msg = "pyproject managed owner must declare conflict_sections exactly once"
        raise ManagedConflictBootstrapError(msg)
    return declarations[0]


def _atomic_write(path: Path, content: str) -> None:
    """Replace one metadata file atomically while preserving its mode."""
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.close(descriptor)
        temporary.write_text(content, encoding="utf-8")
        temporary.chmod(S_IMODE(path.stat().st_mode))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_managed_conflicts(arguments: Sequence[str]) -> Path | None:
    """Repair an apply target before facade imports can parse its metadata."""
    if tuple(arguments[:2]) != ("codegen", "conform"):
        return None
    if _option_value(arguments, "--mode") != "apply":
        return None
    raw_root = _option_value(arguments, "--root")
    if raw_root is None:
        msg = "codegen conform apply requires --root before bootstrap"
        raise ManagedConflictBootstrapError(msg)
    pyproject = Path(raw_root).expanduser().resolve() / "pyproject.toml"
    if not pyproject.is_file():
        return None
    source = pyproject.read_text(encoding="utf-8")
    if "<<<<<<< " not in source:
        return None
    sections = _declared_pyproject_sections(_codegen_configuration_path())
    try:
        recovered = recover_managed_toml(source, conflict_sections=sections)
    except ManagedConflictError as exc:
        raise ManagedConflictBootstrapError(str(exc)) from exc
    _atomic_write(pyproject, recovered)
    return pyproject


__all__: tuple[str, ...] = (
    "ManagedConflictBootstrapError",
    "prepare_managed_conflicts",
)
