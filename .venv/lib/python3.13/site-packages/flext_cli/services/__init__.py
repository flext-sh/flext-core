# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Cli.services package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from types import MappingProxyType

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from . import _cli_parts as _cli_parts
    from .auth import FlextCliAuth
    from .cli import FlextCliCli
    from .cli_params import FlextCliCommonParams
    from .cmd import FlextCliCmd
    from .docx import FlextCliDocx
    from .file_tools import FlextCliFileTools
    from .formatters import FlextCliFormatters
    from .output import FlextCliOutput
    from .pipeline import FlextCliPipeline
    from .pptx import FlextCliPptx
    from .prompts import FlextCliPrompts
    from .rules import FlextCliRules
    from .runtime import FlextCliRuntime
    from .tables import FlextCliTables
    from .xlsx import FlextCliXlsx
    from .yaml_model import FlextCliYamlModel
__all__: tuple[str, ...] = (
    "FlextCliAuth",
    "FlextCliCli",
    "FlextCliCmd",
    "FlextCliCommonParams",
    "FlextCliDocx",
    "FlextCliFileTools",
    "FlextCliFormatters",
    "FlextCliOutput",
    "FlextCliPipeline",
    "FlextCliPptx",
    "FlextCliPrompts",
    "FlextCliRules",
    "FlextCliRuntime",
    "FlextCliTables",
    "FlextCliXlsx",
    "FlextCliYamlModel",
    "_cli_parts",
)

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({
            "._cli_parts": ("_cli_parts",),
            ".auth": ("FlextCliAuth",),
            ".cli": ("FlextCliCli",),
            ".cli_params": ("FlextCliCommonParams",),
            ".cmd": ("FlextCliCmd",),
            ".docx": ("FlextCliDocx",),
            ".file_tools": ("FlextCliFileTools",),
            ".formatters": ("FlextCliFormatters",),
            ".output": ("FlextCliOutput",),
            ".pipeline": ("FlextCliPipeline",),
            ".pptx": ("FlextCliPptx",),
            ".prompts": ("FlextCliPrompts",),
            ".rules": ("FlextCliRules",),
            ".runtime": ("FlextCliRuntime",),
            ".tables": ("FlextCliTables",),
            ".xlsx": ("FlextCliXlsx",),
            ".yaml_model": ("FlextCliYamlModel",),
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
