"""Flext CLI constants — flat MRO facade."""

from __future__ import annotations

from flext_cli._constants.base import FlextCliConstantsBase
from flext_cli._constants.config import FlextCliConstantsConfig
from flext_cli._constants.docx import FlextCliConstantsDocx
from flext_cli._constants.enums import FlextCliConstantsEnums
from flext_cli._constants.errors import FlextCliConstantsErrors
from flext_cli._constants.exceptions import FlextCliConstantsExceptions
from flext_cli._constants.files import FlextCliConstantsFiles
from flext_cli._constants.output import FlextCliConstantsOutput
from flext_cli._constants.pptx import FlextCliConstantsPptx
from flext_cli._constants.settings import FlextCliConstantsSettings
from flext_cli._constants.xlsx import FlextCliConstantsXlsx
from flext_cli._constants.xlsx_future_functions import (
    FlextCliConstantsXlsxFutureFunctions,
)
from flext_core import c, t


class FlextCliConstants(c):
    """Constants for Flext CLI."""

    class Cli(
        FlextCliConstantsBase,
        FlextCliConstantsConfig,
        FlextCliConstantsEnums,
        FlextCliConstantsErrors,
        FlextCliConstantsDocx,
        FlextCliConstantsPptx,
        FlextCliConstantsExceptions,
        FlextCliConstantsFiles,
        FlextCliConstantsOutput,
        FlextCliConstantsSettings,
        FlextCliConstantsXlsx,
        FlextCliConstantsXlsxFutureFunctions,
    ):
        """CLI related constants."""


c = FlextCliConstants

__all__: t.StrSequence = ("FlextCliConstants", "c")
