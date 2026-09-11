"""Class reconstructor transformer for method ordering — rope-based."""

from __future__ import annotations

import operator
from typing import override

from flext_infra import c, m, t, u
from flext_infra._utilities.rope_analysis import FlextInfraUtilitiesRopeAnalysis
from flext_infra.transformers.base import FlextInfraRopeTransformer


class FlextInfraRefactorClassReconstructor(FlextInfraRopeTransformer):
    """Reorder class methods based on declarative ordering configuration.

    Uses rope's ``PyClass.get_attributes()`` and ``Scope.get_start()`` /
    ``Scope.get_end()`` to locate every method's source range without any
    direct ``ast`` parsing. Source slicing is line-based; column offsets
    are normalized via ``str.splitlines(keepends=True)``.
    """

    _description = "class reconstructor"

    def __init__(
        self,
        order_config: t.SequenceOf[t.JsonMapping],
        on_change: t.Infra.ChangeCallback = None,
    ) -> None:
        """Initialize with rule order settings and optional change callback."""
        super().__init__(on_change=on_change)
        try:
            typed_items = t.Infra.CONTAINER_DICT_SEQ_ADAPTER.validate_python(
                order_config
            )
            self._order_config: t.SequenceOf[m.Infra.MethodOrderRule] = [
                m.Infra.MethodOrderRule.model_validate(item) for item in typed_items
            ]
        except c.ValidationError:
            self._order_config = list[m.Infra.MethodOrderRule]()

    @override
    def apply_to_source(self, source: str) -> t.Infra.TransformResult:
        """Apply method reordering to in-memory source via rope."""
        self.changes.clear()
        if not self._order_config:
            return source, list[str]()
        try:
            pymodule = FlextInfraUtilitiesRopeAnalysis.parse_string_module(source)
        except c.EXC_OS_SYNTAX:
            return source, list[str]()
        if pymodule is None:
            return source, list[str]()
        lines = source.splitlines(keepends=True)
        edits: list[tuple[int, int, str]] = []
        for class_name, class_pyname in pymodule.get_attributes().items():
            class_obj = class_pyname.get_object()
            if not FlextInfraUtilitiesRopeAnalysis.is_pyclass(class_obj):
                continue
            block_edits = self._class_block_edits(
                class_name=class_name, class_obj=class_obj, lines=lines
            )
            edits.extend(block_edits)
        if not edits:
            return source, list(self.changes)
        updated = source
        for start, end, replacement in sorted(
            edits, key=operator.itemgetter(0), reverse=True
        ):
            updated = f"{updated[:start]}{replacement}{updated[end:]}"
        return updated, list(self.changes)

    def _class_block_edits(
        self,
        *,
        class_name: str,
        class_obj: t.Infra.RopePyObject,
        lines: t.SequenceOf[str],
    ) -> list[tuple[int, int, str]]:
        """Return reorder edits for each contiguous method block in one class."""
        method_chunks = self._collect_method_chunks(class_obj=class_obj, lines=lines)
        if len(method_chunks) < c.Infra.MIN_METHODS_FOR_REORDER:
            return []

        def method_sort_key(
            item: tuple[m.Infra.MethodInfo, int, int, str],
        ) -> tuple[int, int, str]:
            return u.Infra.build_method_sort_key(item[0], self._order_config)

        source = "".join(lines)
        edits: list[tuple[int, int, str]] = []
        for block in self._contiguous_method_blocks(
            method_chunks=method_chunks, source=source
        ):
            if len(block) < c.Infra.MIN_METHODS_FOR_REORDER:
                continue
            sorted_chunks = sorted(block, key=method_sort_key)
            if [item[0].name for item in block] == [
                item[0].name for item in sorted_chunks
            ]:
                continue
            separators = [
                source[block[index][2] : block[index + 1][1]]
                for index in range(len(block) - 1)
            ]
            replacement_parts: list[str] = []
            for index, item in enumerate(sorted_chunks):
                replacement_parts.append(item[3])
                if index < len(separators):
                    replacement_parts.append(separators[index])
            edits.append((block[0][1], block[-1][2], "".join(replacement_parts)))
            self._record_change(f"Reordered {len(block)} methods in class {class_name}")
        return edits

    def _collect_method_chunks(
        self, *, class_obj: t.Infra.RopePyObject, lines: t.SequenceOf[str]
    ) -> list[tuple[m.Infra.MethodInfo, int, int, str]]:
        """Collect ``(MethodInfo, start_offset, end_offset, source_chunk)`` ordered by line."""
        line_offsets = self._line_offsets(lines)
        source = "".join(lines)
        raw: list[tuple[int, int, m.Infra.MethodInfo]] = []
        for method_name, method_pyname in class_obj.get_attributes().items():
            method_obj = method_pyname.get_object()
            if not FlextInfraUtilitiesRopeAnalysis.is_pyfunction(method_obj):
                continue
            # flext-j47u (codex): Rope returns a tuple; only its line is optional.
            location = method_pyname.get_definition_location()
            if location[1] is None:
                continue
            raw_line = lines[location[1] - 1]
            definition_line = raw_line.lstrip()
            if not definition_line.startswith(("def ", "async def ", "@")):
                continue
            decorators = FlextInfraUtilitiesRopeAnalysis.decorator_names(method_obj)
            start_line = FlextInfraUtilitiesRopeAnalysis.first_decorator_line(
                method_obj, default_line=location[1]
            )
            method_scope = method_obj.get_scope()
            end_line = (
                method_scope.get_end() if method_scope is not None else location[1]
            )
            method_info = m.Infra.MethodInfo(
                name=method_name,
                category=u.Infra.categorize_method(method_name, decorators),
                node=None,
                decorators=decorators,
            )
            raw.append((start_line, end_line, method_info))
        raw.sort(key=operator.itemgetter(0))
        chunks: list[tuple[m.Infra.MethodInfo, int, int, str]] = []
        for start_line, end_line, method_info in raw:
            block_start = line_offsets[start_line - 1]
            block_end = (
                line_offsets[end_line] if end_line < len(line_offsets) else len(source)
            )
            chunks.append((
                method_info,
                block_start,
                block_end,
                source[block_start:block_end],
            ))
        return chunks

    @staticmethod
    def _contiguous_method_blocks(
        *,
        method_chunks: t.SequenceOf[tuple[m.Infra.MethodInfo, int, int, str]],
        source: str,
    ) -> list[list[tuple[m.Infra.MethodInfo, int, int, str]]]:
        """Split method chunks into reorderable blocks separated only by spacing/comments."""
        if not method_chunks:
            return []
        blocks: list[list[tuple[m.Infra.MethodInfo, int, int, str]]] = [
            [method_chunks[0]]
        ]
        for chunk in method_chunks[1:]:
            previous = blocks[-1][-1]
            gap = source[previous[2] : chunk[1]]
            if FlextInfraRefactorClassReconstructor._is_reorderable_gap(gap):
                blocks[-1].append(chunk)
                continue
            blocks.append([chunk])
        return blocks

    @staticmethod
    def _is_reorderable_gap(gap: str) -> bool:
        """Return whether the gap between methods contains only blank lines or comments."""
        return all(
            not stripped or stripped.startswith("#")
            for stripped in (line.strip() for line in gap.splitlines())
        )

    @staticmethod
    def _line_offsets(lines: t.SequenceOf[str]) -> t.SequenceOf[int]:
        """Return cumulative byte offsets at each line start (plus EOF)."""
        offsets = [0]
        for line in lines:
            offsets.append(offsets[-1] + len(line))
        return offsets


__all__: list[str] = ["FlextInfraRefactorClassReconstructor"]
