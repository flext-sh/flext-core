"""Rope PyModule / identifier helpers — extracted concern of FlextInfraUtilitiesRopeCore."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, ClassVar

from flext_infra._utilities.rope_runtime import FlextInfraUtilitiesRopeRuntime

if TYPE_CHECKING:
    from flext_infra.typings import t


class FlextInfraUtilitiesRopeCorePyModuleMixin:
    """PyModule resolution + import-table + identifier-offset helpers.

    Composed into FlextInfraUtilitiesRopeCore via inheritance so the helpers
    stay reachable as ``FlextInfraUtilitiesRopeCore.<method>`` for all callers.
    """

    _IDENTIFIER_PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"\b[A-Za-z_]\w*\b")

    @staticmethod
    def find_identifier_offset_in_lines(
        lines: t.SequenceOf[str], *, line: int, symbol: str
    ) -> int | None:
        """Return the absolute offset of one exact identifier token on a line.

        Rope returns definition line numbers but not the column for many ``PyName``
        variants. Using ``str.find(symbol)`` is incorrect because it can match a
        substring inside another token or keyword, e.g. ``except ... as e``.
        This helper resolves the first exact identifier token equal to ``symbol``
        on the reported line.
        """
        if line < 1 or line > len(lines):
            return None
        line_start = sum(len(item) for item in lines[: line - 1])
        source_line = lines[line - 1]
        for (
            match
        ) in FlextInfraUtilitiesRopeCorePyModuleMixin._IDENTIFIER_PATTERN.finditer(
            source_line
        ):
            if match.group(0) == symbol:
                offset: int = line_start + match.start()
                return offset
        return None

    @staticmethod
    def get_pymodule(
        rope_project: t.Infra.RopeProject, resource: t.Infra.RopeResource
    ) -> t.Infra.RopePyModule:
        """Resolve one concrete rope PyModule through the validated API boundary."""
        pymodule = rope_project.get_pymodule(resource)
        if not FlextInfraUtilitiesRopeRuntime.is_pymodule(pymodule):
            msg = "rope project returned non-PyModule"
            raise TypeError(msg)
        result: t.Infra.RopePyModule = pymodule
        return result

    @staticmethod
    def get_module_imports(
        rope_project: t.Infra.RopeProject, resource: t.Infra.RopeResource
    ) -> t.Infra.RopeModuleImports | None:
        """Get module imports."""
        try:
            module_imports = FlextInfraUtilitiesRopeRuntime.module_imports_for_pymodule(
                rope_project,
                FlextInfraUtilitiesRopeCorePyModuleMixin.get_pymodule(
                    rope_project, resource
                ),
            )
        except (*FlextInfraUtilitiesRopeRuntime.rope_runtime_errors(), TypeError):
            return None
        result: t.Infra.RopeModuleImports | None = module_imports
        return result


__all__: list[str] = ["FlextInfraUtilitiesRopeCorePyModuleMixin"]
