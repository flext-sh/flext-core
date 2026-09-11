"""Heavy ``u.Cli`` utility namespace materialized on demand."""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from typing import TYPE_CHECKING, Final

from flext_cli._utilities._options_parts.flextcliutilitiesoptions_part_02 import (
    FlextCliUtilitiesOptions,
)
from flext_cli._utilities.auth import FlextCliUtilitiesAuth
from flext_cli._utilities.cmd import FlextCliUtilitiesCmd
from flext_cli._utilities.commands import FlextCliUtilitiesCommands
from flext_cli._utilities.config import FlextCliUtilitiesConfig
from flext_cli._utilities.conversion import FlextCliUtilitiesConversion
from flext_cli._utilities.env import FlextCliUtilitiesEnv
from flext_cli._utilities.file_test_helpers import FlextCliUtilitiesFileTestHelpersMixin
from flext_cli._utilities.files import FlextCliUtilitiesFiles
from flext_cli._utilities.formatters import FlextCliUtilitiesFormatters
from flext_cli._utilities.framework import FlextCliUtilitiesFramework
from flext_cli._utilities.json import FlextCliUtilitiesJson
from flext_cli._utilities.matching import FlextCliUtilitiesMatching
from flext_cli._utilities.model_commands import FlextCliUtilitiesModelCommands
from flext_cli._utilities.output import FlextCliUtilitiesOutput
from flext_cli._utilities.params import FlextCliUtilitiesParams
from flext_cli._utilities.pipeline import FlextCliUtilitiesPipeline
from flext_cli._utilities.processes import FlextCliUtilitiesProcesses
from flext_cli._utilities.prompts import FlextCliUtilitiesPrompts
from flext_cli._utilities.rules import FlextCliUtilitiesRules
from flext_cli._utilities.runtime import FlextCliUtilitiesRuntime
from flext_cli._utilities.settings import FlextCliUtilitiesSettings
from flext_cli._utilities.tables import FlextCliUtilitiesTables
from flext_cli._utilities.template import FlextCliUtilitiesTemplate
from flext_cli._utilities.toml import FlextCliUtilitiesToml
from flext_cli._utilities.validation import FlextCliUtilitiesValidation
from flext_cli._utilities.yaml import FlextCliUtilitiesYaml
from flext_cli._utilities.yaml_model import FlextCliUtilitiesYamlModel

if TYPE_CHECKING:
    from flext_cli._utilities.docx import FlextCliUtilitiesDocx
    from flext_cli._utilities.xlsx import FlextCliUtilitiesXlsx


# Why: openpyxl and python-docx are HARD dependencies (pyproject.toml), but
# they are only needed by the document operations. Keeping their owners in the
# eager MRO made every `flext_cli.utilities` consumer -- every CLI in the
# fleet, on every invocation including `--help` -- pay seconds of module
# construction for functionality most of them never call. The owners now load
# on first use of a document operation, and a broken install surfaces the real
# missing-dependency error from that operation instead of swallowing it.
_DOCUMENT_OWNER_IMPORTS: Final[Mapping[str, tuple[str, str]]] = {
    "FlextCliUtilitiesXlsx": ("flext_cli._utilities.xlsx", "FlextCliUtilitiesXlsx"),
    "FlextCliUtilitiesDocx": ("flext_cli._utilities.docx", "FlextCliUtilitiesDocx"),
}

_DOCUMENT_OPERATION_OWNERS: Final[Mapping[str, str]] = {
    "xlsx_render": "FlextCliUtilitiesXlsx",
    "xlsx_snapshot": "FlextCliUtilitiesXlsx",
    "xlsx_inspect": "FlextCliUtilitiesXlsx",
    "xlsx_recalc": "FlextCliUtilitiesXlsx",
    "xlsx_recalc_parity": "FlextCliUtilitiesXlsx",
    "xlsx_defined_name_values": "FlextCliUtilitiesXlsx",
    "xlsx_style_catalog": "FlextCliUtilitiesXlsx",
    "xlsx_style_template": "FlextCliUtilitiesXlsx",
    "xlsx_parse_range": "FlextCliUtilitiesXlsx",
    "xlsx_format_reference": "FlextCliUtilitiesXlsx",
    "docx_read": "FlextCliUtilitiesDocx",
    "docx_render": "FlextCliUtilitiesDocx",
}


class _LazyDocumentOperation:
    """Resolve one document operation against its owner on first access."""

    __slots__ = ("_operation",)

    def __init__(self, operation: str) -> None:
        self._operation = operation

    def __get__(
        self, instance: object | None, owner: type[object] | None = None
    ) -> object:
        module_name, attribute = _DOCUMENT_OWNER_IMPORTS[
            _DOCUMENT_OPERATION_OWNERS[self._operation]
        ]
        owner_cls = getattr(import_module(module_name), attribute)
        return getattr(owner_cls, self._operation)


if TYPE_CHECKING:

    class FlextCliUtilitiesCli(  # type-checker view: full documented surface
        FlextCliUtilitiesAuth,
        FlextCliUtilitiesCmd,
        FlextCliUtilitiesCommands,
        FlextCliUtilitiesConfig,
        FlextCliUtilitiesConversion,
        FlextCliUtilitiesEnv,
        FlextCliUtilitiesTemplate,
        FlextCliUtilitiesFileTestHelpersMixin,
        FlextCliUtilitiesFiles,
        FlextCliUtilitiesFramework,
        FlextCliUtilitiesFormatters,
        FlextCliUtilitiesJson,
        FlextCliUtilitiesMatching,
        FlextCliUtilitiesModelCommands,
        FlextCliUtilitiesOptions,
        FlextCliUtilitiesOutput,
        FlextCliUtilitiesParams,
        FlextCliUtilitiesPipeline,
        FlextCliUtilitiesPrompts,
        FlextCliUtilitiesProcesses,
        FlextCliUtilitiesRules,
        FlextCliUtilitiesRuntime,
        FlextCliUtilitiesSettings,
        FlextCliUtilitiesTables,
        FlextCliUtilitiesToml,
        FlextCliUtilitiesValidation,
        FlextCliUtilitiesXlsx,
        FlextCliUtilitiesDocx,
        FlextCliUtilitiesYaml,
        FlextCliUtilitiesYamlModel,
    ):
        """Command line interface specific utilities composed via MRO."""

else:

    class FlextCliUtilitiesCli(
        FlextCliUtilitiesAuth,
        FlextCliUtilitiesCmd,
        FlextCliUtilitiesCommands,
        FlextCliUtilitiesConfig,
        FlextCliUtilitiesConversion,
        FlextCliUtilitiesEnv,
        FlextCliUtilitiesTemplate,
        FlextCliUtilitiesFileTestHelpersMixin,
        FlextCliUtilitiesFiles,
        FlextCliUtilitiesFramework,
        FlextCliUtilitiesFormatters,
        FlextCliUtilitiesJson,
        FlextCliUtilitiesMatching,
        FlextCliUtilitiesModelCommands,
        FlextCliUtilitiesOptions,
        FlextCliUtilitiesOutput,
        FlextCliUtilitiesParams,
        FlextCliUtilitiesPipeline,
        FlextCliUtilitiesPrompts,
        FlextCliUtilitiesProcesses,
        FlextCliUtilitiesRules,
        FlextCliUtilitiesRuntime,
        FlextCliUtilitiesSettings,
        FlextCliUtilitiesTables,
        FlextCliUtilitiesToml,
        FlextCliUtilitiesValidation,
        FlextCliUtilitiesYaml,
        FlextCliUtilitiesYamlModel,
    ):
        """CLI utilities; document owners resolve on first use."""


for _operation in _DOCUMENT_OPERATION_OWNERS:
    setattr(FlextCliUtilitiesCli, _operation, _LazyDocumentOperation(_operation))


__all__: tuple[str, ...] = ("FlextCliUtilitiesCli",)
