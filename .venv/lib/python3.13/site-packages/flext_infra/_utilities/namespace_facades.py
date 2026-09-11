"""Facade and dependency-chain helpers for namespace refactors."""

from __future__ import annotations

import operator
from collections import defaultdict
from collections.abc import MutableMapping
from pathlib import Path
from types import MappingProxyType
from typing import ClassVar

from flext_cli import u
from flext_infra._utilities.dependencies import FlextInfraUtilitiesDependencies
from flext_infra._utilities.namespace import FlextInfraUtilitiesCodegenNamespace
from flext_infra._utilities.namespace_common import (
    FlextInfraUtilitiesRefactorNamespaceCommon,
)
from flext_infra._utilities.rope_module_patch import FlextInfraUtilitiesRopeModulePatch
from flext_infra.constants import c
from flext_infra.models import m
from flext_infra.typings import t

# flext-j47u (codex): annotation-only stdlib types are safe runtime imports;
# TYPE_CHECKING is reserved for real reverse-dependency cycle boundaries.


class FlextInfraUtilitiesRefactorNamespaceFacades:
    """Facade generation and dependency-chain helpers."""

    _base_chains_cache: ClassVar[MutableMapping[Path, t.StrSequenceMapping]] = {}

    @staticmethod
    def build_expected_base_chains(*, project_root: Path) -> t.StrSequenceMapping:
        """Build expected base chains."""
        resolved = project_root.resolve()
        cached = FlextInfraUtilitiesRefactorNamespaceFacades._base_chains_cache.get(
            resolved
        )
        if cached is not None:
            return cached
        result = FlextInfraUtilitiesRefactorNamespaceFacades._compute_base_chains(
            project_root=resolved
        )
        FlextInfraUtilitiesRefactorNamespaceFacades._base_chains_cache[resolved] = (
            result
        )
        return result

    @staticmethod
    def _compute_base_chains(*, project_root: Path) -> t.StrSequenceMapping:
        """Compute base chains."""
        pyproject_path = project_root / c.Infra.PYPROJECT_FILENAME
        if not pyproject_path.exists():
            return MappingProxyType(dict[str, tuple[str, ...]]())
        try:
            raw = pyproject_path.read_text(encoding=c.Cli.ENCODING_DEFAULT)
        except OSError:
            return MappingProxyType(dict[str, tuple[str, ...]]())
        payload = u.Cli.toml_mapping_from_text(raw)
        if payload is None:
            return MappingProxyType(dict[str, tuple[str, ...]]())
        dep_names = (
            FlextInfraUtilitiesDependencies.declared_dependency_names_from_payload(
                t.Infra.INFRA_MAPPING_ADAPTER.validate_python(payload)
            )
        )
        chains: t.MutableStrSequenceMapping = defaultdict(list)
        for dep_name in dep_names:
            if dep_name == c.Infra.PKG_CORE or not dep_name.startswith(
                c.Infra.PKG_PREFIX_HYPHEN
            ):
                continue
            stem = u.derive_class_stem(dep_name)
            for family, suffix in c.Infra.FAMILY_SUFFIXES.items():
                chains[family].append(f"{stem}{suffix}")
        return chains

    @staticmethod
    def _base_import_for_family(
        *, family: str, base_chains: t.StrSequenceMapping | None = None
    ) -> str:
        """Return the base import for family."""
        if base_chains:
            chain = base_chains.get(family, [])
            if chain:
                return "\n".join(
                    f"from {u.class_name_to_module(base)} import {base}"
                    for base in chain
                )
        suffix = c.Infra.FAMILY_SUFFIXES.get(family, "Utilities")
        return f"from flext_core import Flext{suffix}"

    @staticmethod
    def _base_class_for_family(
        *, family: str, base_chains: t.StrSequenceMapping | None = None
    ) -> str:
        """Return the base class for family."""
        if base_chains:
            chain = base_chains.get(family, [])
            if chain:
                return ", ".join(chain)
        suffix = c.Infra.FAMILY_SUFFIXES.get(family, "Utilities")
        return f"Flext{suffix}"

    @staticmethod
    def _write_missing_facade_file(
        *,
        file_path: Path,
        package_root: Path,
        family: str,
        class_name: str,
        base_chains: t.StrSequenceMapping | None = None,
    ) -> None:
        """Write missing facade file."""
        file_path = file_path.resolve()
        if not file_path.is_relative_to(package_root.resolve()):
            msg = f"refusing facade write outside package: {file_path}"
            raise ValueError(msg)
        content = (
            '"""Auto-generated facade to enforce FLEXT namespace contracts."""\n\n'
            "from __future__ import annotations\n\n"
            f"{FlextInfraUtilitiesRefactorNamespaceFacades._base_import_for_family(family=family, base_chains=base_chains)}\n\n"
            f"class {class_name}({FlextInfraUtilitiesRefactorNamespaceFacades._base_class_for_family(family=family, base_chains=base_chains)}):\n"
            "    pass\n\n"
            f"{family} = {class_name}\n\n"
            f'__all__: list[str] = ["{class_name}", "{family}"]\n'
        )
        file_path.parent.mkdir(parents=True, exist_ok=True)
        _ = file_path.write_text(content, encoding=c.Cli.ENCODING_DEFAULT)

    @staticmethod
    def ensure_missing_facades(
        *,
        project_root: Path,
        project_name: str,
        facade_statuses: t.SequenceOf[m.Infra.FacadeStatus],
        repository_root: Path | None = None,
    ) -> None:
        """Ensure missing facades."""
        del project_name
        src_dir = project_root / c.Infra.DEFAULT_SRC_DIR
        if not src_dir.is_dir():
            return
        package_dirs = [
            entry
            for entry in sorted(src_dir.iterdir(), key=operator.attrgetter("name"))
            if entry.is_dir() and (entry / c.Infra.INIT_PY).is_file()
        ]
        layout = FlextInfraUtilitiesCodegenNamespace.layout(project_root)
        if layout is None:
            return
        package_dir = (
            layout.package_dir if layout.package_dir.is_dir() else package_dirs[0]
        )
        stem = layout.class_stem
        base_chains = (
            FlextInfraUtilitiesRefactorNamespaceFacades.build_expected_base_chains(
                project_root=project_root
            )
            if repository_root is not None
            else None
        )
        for status in facade_statuses:
            if status.exists:
                continue
            suffix = c.Infra.FAMILY_SUFFIXES[status.family]
            class_name = f"{stem}{suffix}"
            file_name = c.Infra.FAMILY_FILES.get(
                status.family, c.Infra.UTILITIES_PY
            ).lstrip("*")
            target_path = package_dir / file_name
            if target_path.exists():
                FlextInfraUtilitiesRefactorNamespaceFacades._patch_existing_facade_file(
                    target_path=target_path,
                    package_root=package_dir,
                    family=status.family,
                    class_name=class_name,
                    base_chains=base_chains,
                )
                continue
            FlextInfraUtilitiesRefactorNamespaceFacades._write_missing_facade_file(
                file_path=target_path,
                package_root=package_dir,
                family=status.family,
                class_name=class_name,
                base_chains=base_chains,
            )

    @staticmethod
    def _all_block_end_index(*, lines: list[str], start_index: int) -> int:
        r"""Return the last line index of an ``__all__`` declaration block.

        Handles both single-line (``__all__: list[str] = ["A", "B"]``) and
        multi-line (``__all__: list[str] = [\n    "A",\n    "B",\n]``)
        forms. Counts opening and closing brackets to skip ``list[str]``
        annotation brackets and only stop on the closing bracket of the
        list literal itself.
        """
        depth = 0
        for offset, line in enumerate(lines[start_index:]):
            segment = line.partition("=")[2] if offset == 0 else line
            if not segment:
                continue
            depth += segment.count("[")
            depth -= segment.count("]")
            if depth == 0 and "]" in segment:
                return start_index + offset
        return start_index

    @staticmethod
    def _patch_existing_facade_file(
        *,
        target_path: Path,
        package_root: Path,
        family: str,
        class_name: str,
        base_chains: t.StrSequenceMapping | None = None,
    ) -> None:
        """Patch existing facade file."""
        target_path = target_path.resolve()
        if not target_path.is_relative_to(package_root.resolve()):
            msg = f"refusing facade patch outside package: {target_path}"
            raise ValueError(msg)
        source = target_path.read_text(encoding=c.Cli.ENCODING_DEFAULT)
        lines = source.splitlines()
        base_class = FlextInfraUtilitiesRefactorNamespaceFacades._base_class_for_family(
            family=family, base_chains=base_chains
        )
        base_import = (
            FlextInfraUtilitiesRefactorNamespaceFacades._base_import_for_family(
                family=family, base_chains=base_chains
            )
        )
        canonical_header = f"class {class_name}({base_class}):"
        if base_import not in lines:
            lines = list(
                FlextInfraUtilitiesRefactorNamespaceCommon.insert_import_lines(
                    lines=lines, imports=[base_import, ""]
                )
            )
        class_line_indices = [
            idx
            for idx, line in enumerate(lines)
            if c.Infra.CLASS_HEADER_ANY_RE.match(line)
        ]
        class_header_re = c.Infra.compile_class_header_match(class_name)
        existing_class_index = next(
            (idx for idx in class_line_indices if class_header_re.match(lines[idx])), -1
        )
        if existing_class_index >= 0:
            if lines[existing_class_index] != canonical_header:
                lines[existing_class_index] = canonical_header
        elif len(class_line_indices) == 1:
            if lines[class_line_indices[0]] != canonical_header:
                lines[class_line_indices[0]] = canonical_header
        else:
            lines.extend(["", canonical_header, "    pass"])
        updated_source = "\n".join(lines).rstrip() + "\n"
        updated_source = FlextInfraUtilitiesRopeModulePatch.ensure_runtime_alias(
            updated_source, alias=family, target_name=class_name
        )
        all_line = f'__all__: list[str] = ["{class_name}", "{family}"]'
        if all_line not in updated_source:
            updated_lines = updated_source.splitlines()
            all_index = next(
                (
                    idx
                    for idx, line in enumerate(updated_lines)
                    if c.Infra.DUNDER_ALL_DECL_RE.match(line)
                ),
                -1,
            )
            if all_index < 0:
                updated_lines.extend(["", all_line])
            else:
                end_index = (
                    FlextInfraUtilitiesRefactorNamespaceFacades._all_block_end_index(
                        lines=updated_lines, start_index=all_index
                    )
                )
                updated_lines[all_index : end_index + 1] = [all_line]
            updated_source = "\n".join(updated_lines).rstrip() + "\n"
        if updated_source != source:
            _ = target_path.write_text(updated_source, encoding=c.Cli.ENCODING_DEFAULT)


__all__: list[str] = ["FlextInfraUtilitiesRefactorNamespaceFacades"]
