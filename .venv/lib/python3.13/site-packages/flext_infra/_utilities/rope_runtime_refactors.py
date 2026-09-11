"""Rope refactor and occurrence boundary methods."""

from __future__ import annotations

from typing import ClassVar

from flext_infra._utilities.rope_runtime_base import FlextInfraUtilitiesRopeRuntimeBase
from flext_infra.protocols import p
from flext_infra.typings import t


class FlextInfraUtilitiesRopeRuntimeRefactors(FlextInfraUtilitiesRopeRuntimeBase):
    """Load Rope refactor helpers behind protocols."""

    _WORD_RANGE_SIZE: ClassVar[int] = 2

    @classmethod
    def create_move(
        cls,
        rope_project: t.Infra.RopeProject,
        resource: t.Infra.RopeResource,
        offset: int,
    ) -> t.Infra.RopeMoveGlobal:
        create = cls._runtime_callable("rope.refactor.move", "create_move")
        mover = create(rope_project, resource, offset)
        if not isinstance(mover, p.Infra.RopeMoveGlobal):
            msg = "rope create_move did not return MoveGlobal-compatible object"
            raise TypeError(msg)
        return mover

    @classmethod
    def rename_changes(
        cls,
        rope_project: t.Infra.RopeProject,
        resource: t.Infra.RopeResource,
        offset: int,
        new_name: str,
        *,
        resources: t.SequenceOf[t.Infra.RopeResource],
    ) -> t.Infra.RopeChangeSet:
        renamer_factory = cls._runtime_callable("rope.refactor.rename", "Rename")
        renamer = renamer_factory(rope_project, resource, offset)
        get_changes = getattr(renamer, "get_changes", None)
        if not callable(get_changes):
            msg = "rope Rename does not expose callable get_changes"
            raise TypeError(msg)
        changes = get_changes(new_name, resources=list(resources))
        if not isinstance(changes, p.Infra.RopeChangeSet):
            msg = "rope Rename returned invalid ChangeSet"
            raise TypeError(msg)
        return changes

    @classmethod
    def create_occurrence_finder(
        cls,
        rope_project: t.Infra.RopeProject,
        name: str,
        pyname: t.Infra.RopePyName,
        *,
        imports: bool,
        in_hierarchy: bool,
    ) -> t.Infra.RopeOccurrenceFinder:
        create_finder = cls._runtime_callable(
            "rope.refactor.occurrences", "create_finder"
        )
        finder = create_finder(
            rope_project, name, pyname, imports=imports, in_hierarchy=in_hierarchy
        )
        if not isinstance(finder, p.Infra.RopeOccurrenceFinder):
            msg = "rope occurrence finder does not satisfy p.Infra.RopeOccurrenceFinder"
            raise TypeError(msg)
        return finder

    @classmethod
    def word_primary_range(cls, source: str, offset: int) -> tuple[int, int]:
        word_finder = cls._word_finder(source)
        primary_range = getattr(word_finder, "get_primary_range", None)
        if not callable(primary_range):
            msg = "rope Worder does not expose callable get_primary_range"
            raise TypeError(msg)
        value = primary_range(offset)
        if (
            not isinstance(value, tuple)
            or len(value) != cls._WORD_RANGE_SIZE
            or not all(isinstance(item, int) for item in value)
        ):
            msg = "rope Worder returned invalid primary range"
            raise TypeError(msg)
        return value

    @classmethod
    def word_primary_at(cls, source: str, offset: int) -> str:
        word_finder = cls._word_finder(source)
        primary_at = getattr(word_finder, "get_primary_at", None)
        if not callable(primary_at):
            msg = "rope Worder does not expose callable get_primary_at"
            raise TypeError(msg)
        return str(primary_at(offset))

    @classmethod
    def word_is_function_call(cls, source: str, offset: int) -> bool:
        """Return Rope's syntactic call fact for the primary at ``offset``."""
        word_finder = cls._word_finder(source)
        is_called = getattr(word_finder, "is_a_function_being_called", None)
        if not callable(is_called):
            msg = "rope Worder does not expose callable is_a_function_being_called"
            raise TypeError(msg)
        value = is_called(offset)
        if not isinstance(value, bool):
            msg = "rope Worder returned a non-boolean function-call fact"
            raise TypeError(msg)
        return value

    @classmethod
    def _word_finder(cls, source: str) -> p.AttributeProbe:
        worder_module = cls._module("rope.base.worder")
        worder_factory = getattr(worder_module, "Worder", None)
        if not callable(worder_factory):
            msg = "rope Worder factory is not callable"
            raise TypeError(msg)
        finder: p.AttributeProbe = worder_factory(source, True)
        return finder


__all__: list[str] = ["FlextInfraUtilitiesRopeRuntimeRefactors"]
