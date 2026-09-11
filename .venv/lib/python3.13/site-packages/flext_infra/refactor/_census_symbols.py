"""Lightweight Rope symbol indexing + detector context — extracted census concern."""

from __future__ import annotations

from operator import itemgetter
from typing import TYPE_CHECKING

from flext_infra import m, u

if TYPE_CHECKING:
    from pathlib import Path

    from flext_infra import p, t


class FlextInfraRefactorCensusSymbolsMixin:
    """Top-level symbol discovery + detector-context construction for census.

    Composed into FlextInfraRefactorCensus via inheritance; self-contained
    (``cls``-internal recursion + rope/facade calls only, no census state).
    """

    @classmethod
    def _lightweight_symbol_index(
        cls, rope: p.Infra.RopeWorkspaceDsl, file_path: Path
    ) -> dict[str, tuple[str, int]]:
        """Top-level symbol index for detector-only rule sets."""
        resource = rope.resource(file_path)
        if resource is None:
            return {}
        try:
            attributes = u.Infra.get_pymodule(
                rope.rope_project, resource
            ).get_attributes()
        except (
            *u.Infra.rope_runtime_errors(),
            RecursionError,
            SyntaxError,
            ValueError,
        ) as exc:
            msg = (
                "census lightweight symbol discovery failed for "
                f"{file_path}: {type(exc).__name__}: {exc!s}"
            )
            raise RuntimeError(msg) from exc
        symbols: dict[str, tuple[str, int]] = {}
        object_kinds: dict[int, str] = {}
        candidates: list[tuple[int, str, t.Infra.RopePyName]] = []
        for name, pyname in attributes.items():
            if u.Infra.is_imported_name(pyname):
                continue
            line = cls._lightweight_symbol_line(pyname, resource)
            if line is None:
                continue
            candidates.append((line, name, pyname))
        for line, name, pyname in sorted(candidates, key=itemgetter(0)):
            obj = pyname.get_object()
            kind = cls._lightweight_symbol_kind(
                name=name, obj=obj, object_kinds=object_kinds
            )
            symbols.setdefault(name, (kind, line))
            if kind in {"class", "function"}:
                object_kinds[id(obj)] = kind
        return symbols

    @staticmethod
    def _lightweight_symbol_line(
        pyname: t.Infra.RopePyName, resource: t.Infra.RopeResource
    ) -> int | None:
        """Return the local definition line for one top-level Rope symbol."""
        location = pyname.get_definition_location()
        module, line = location
        origin = module.get_resource() if module is not None else None
        if not isinstance(line, int) or line < 1 or origin is None:
            return None
        return line if origin.path == resource.path else None

    @staticmethod
    def _lightweight_symbol_kind(
        *, name: str, obj: t.Infra.RopePyObject | None, object_kinds: dict[int, str]
    ) -> str:
        """Infer a detector-only symbol kind from Rope metadata."""
        inherited_kind = object_kinds.get(id(obj))
        if inherited_kind in {"class", "function"}:
            return inherited_kind
        if u.Infra.is_abstract_class(obj):
            return "class"
        if u.Infra.is_py_function(obj):
            return "function"
        return "constant" if name.isupper() else "assignment"

    @staticmethod
    def _detector_context(
        rope: p.Infra.RopeWorkspaceDsl,
        file_path: Path,
        *,
        convention: m.Infra.RopeModuleConvention | None = None,
        parse_failures: t.MutableSequenceOf[m.Infra.ParseFailureViolation]
        | None = None,
    ) -> m.Infra.DetectorContext:
        """Detector context."""
        resolved_convention = convention or rope.convention(file_path)
        layout = resolved_convention.project_layout
        module_entry = rope.module(file_path)
        project_root = (
            layout.project_root
            if layout is not None
            else module_entry.project_root
            if module_entry is not None
            else None
        )
        project_name = (
            layout.project_name
            if layout is not None
            else project_root.name
            if project_root is not None
            else ""
        )
        return m.Infra.DetectorContext(
            file_path=file_path,
            rope_project=rope.rope_project,
            parse_failures=parse_failures,
            project_name=project_name,
            project_root=project_root,
        )


__all__: list[str] = ["FlextInfraRefactorCensusSymbolsMixin"]
