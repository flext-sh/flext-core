"""Public API facade for flext-cli.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from typing import TYPE_CHECKING, Final, override

from flext_cli import m, p, r, t, u
from flext_cli.services.auth import FlextCliAuth
from flext_cli.services.cli import FlextCliCli
from flext_cli.services.cli_params import FlextCliCommonParams
from flext_cli.services.cmd import FlextCliCmd
from flext_cli.services.file_tools import FlextCliFileTools
from flext_cli.services.formatters import FlextCliFormatters
from flext_cli.services.output import FlextCliOutput
from flext_cli.services.pipeline import FlextCliPipeline
from flext_cli.services.prompts import FlextCliPrompts
from flext_cli.services.rules import FlextCliRules
from flext_cli.services.runtime import FlextCliRuntime
from flext_cli.services.tables import FlextCliTables
from flext_cli.services.yaml_model import FlextCliYamlModel

if TYPE_CHECKING:
    from flext_cli.services.docx import FlextCliDocx
    from flext_cli.services.pptx import FlextCliPptx
    from flext_cli.services.xlsx import FlextCliXlsx


# Why: openpyxl, python-docx, and python-pptx are HARD dependencies, but only
# the document operations need them. Keeping their services in the eager MRO
# made every `flext_cli` consumer in the fleet pay seconds of module
# construction on every invocation for functionality most never call. The
# services now load on first use of a document operation; a broken install
# surfaces its real missing-dependency error from that operation.
_DOCUMENT_SERVICE_IMPORTS: Final[Mapping[str, tuple[str, str]]] = {
    "FlextCliXlsx": ("flext_cli.services.xlsx", "FlextCliXlsx"),
    "FlextCliDocx": ("flext_cli.services.docx", "FlextCliDocx"),
    "FlextCliPptx": ("flext_cli.services.pptx", "FlextCliPptx"),
}

_DOCUMENT_OPERATION_SERVICES: Final[Mapping[str, str]] = {
    "xlsx_render": "FlextCliXlsx",
    "xlsx_snapshot": "FlextCliXlsx",
    "xlsx_inspect": "FlextCliXlsx",
    "xlsx_recalc": "FlextCliXlsx",
    "xlsx_recalc_parity": "FlextCliXlsx",
    "xlsx_defined_name_values": "FlextCliXlsx",
    "xlsx_style_catalog": "FlextCliXlsx",
    "xlsx_style_template": "FlextCliXlsx",
    "xlsx_parse_range": "FlextCliXlsx",
    "xlsx_format_reference": "FlextCliXlsx",
    "docx_read": "FlextCliDocx",
    "docx_render": "FlextCliDocx",
    "pptx_read": "FlextCliPptx",
    "pptx_render": "FlextCliPptx",
    "pptx_open": "FlextCliPptx",
    "pptx_save": "FlextCliPptx",
    # Re-exported python-pptx types: part of the documented surface, so they
    # resolve through the same owner instead of being lost with the eager MRO.
    "Presentation": "FlextCliPptx",
    "PresentationDocument": "FlextCliPptx",
    "RGBColor": "FlextCliPptx",
    "MSO_SHAPE": "FlextCliPptx",
    "MSO_ANCHOR": "FlextCliPptx",
    "MSO_AUTO_SIZE": "FlextCliPptx",
    "PP_ALIGN": "FlextCliPptx",
    "qn": "FlextCliPptx",
    "BaseOxmlElement": "FlextCliPptx",
    "Shape": "FlextCliPptx",
    "Picture": "FlextCliPptx",
    "Slide": "FlextCliPptx",
    "SlideLayout": "FlextCliPptx",
    "TextFrame": "FlextCliPptx",
    "Emu": "FlextCliPptx",
    "Inches": "FlextCliPptx",
    "Length": "FlextCliPptx",
    "Pt": "FlextCliPptx",
}


class _LazyDocumentOperation:
    """Resolve one document operation against its service on first access."""

    __slots__ = ("_operation",)

    def __init__(self, operation: str) -> None:
        self._operation = operation

    def __get__(
        self, instance: object | None, owner: type[object] | None = None
    ) -> object:
        module_name, attribute = _DOCUMENT_SERVICE_IMPORTS[
            _DOCUMENT_OPERATION_SERVICES[self._operation]
        ]
        service_cls = getattr(import_module(module_name), attribute)
        target = service_cls if instance is None else service_cls()
        return getattr(target, self._operation)


if TYPE_CHECKING:

    class FlextCli(  # type-checker view: the full documented surface
        FlextCliAuth,
        FlextCliCli,
        FlextCliCmd,
        FlextCliCommonParams,
        FlextCliDocx,
        FlextCliFileTools,
        FlextCliFormatters,
        FlextCliOutput,
        FlextCliPipeline,
        FlextCliPrompts,
        FlextCliPptx,
        FlextCliRules,
        FlextCliRuntime,
        FlextCliTables,
        FlextCliXlsx,
        FlextCliYamlModel,
    ):
        """Coordinate CLI operations and expose domain services."""

else:

    class FlextCli(
        FlextCliAuth,
        FlextCliCli,
        FlextCliCmd,
        FlextCliCommonParams,
        FlextCliFileTools,
        FlextCliFormatters,
        FlextCliOutput,
        FlextCliPipeline,
        FlextCliPrompts,
        FlextCliRules,
        FlextCliRuntime,
        FlextCliTables,
        FlextCliYamlModel,
    ):
        """Coordinate CLI operations and expose domain services.

        MRO facade over the CLI services (auth, cli, cmd, params, file_tools,
        formatters, output, pipeline, prompts, rules, runtime, tables,
        yaml-model). The document operations (xlsx, docx, pptx) resolve
        against their services on first use instead of loading them at
        import.
        All operations return r[T].
        """

        @override
        def execute(self) -> p.Result[m.Cli.RuntimeStatus]:
            """Report the public CLI runtime surface state."""
            return r[m.Cli.RuntimeStatus].ok(u.Cli.cmd_status())


for _operation in _DOCUMENT_OPERATION_SERVICES:
    setattr(FlextCli, _operation, _LazyDocumentOperation(_operation))


cli: FlextCli = FlextCli.fetch_global()
"""Process-wide ``FlextCli`` facade singleton exposing every CLI service via MRO."""


__all__: t.MutableSequenceOf[str] = ["FlextCli", "cli"]
