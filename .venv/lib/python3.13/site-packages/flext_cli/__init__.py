# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Cli package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from types import MappingProxyType

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

from .__version__ import __author__ as __author__
from .__version__ import __author_email__ as __author_email__
from .__version__ import __description__ as __description__
from .__version__ import __license__ as __license__
from .__version__ import __title__ as __title__
from .__version__ import __url__ as __url__
from .__version__ import __version__ as __version__
from .__version__ import __version_info__ as __version_info__

if TYPE_CHECKING:
    from . import services as services
    from flext_core import d, e, h, r, x

    from ._config import FlextCliConfig, config
    from ._settings import FlextCliSettings, settings
    from .api import FlextCli, cli
    from .base import FlextCliServiceBase, FlextCliServiceBase as s
    from .constants import FlextCliConstants, FlextCliConstants as c
    from .models import FlextCliModels, FlextCliModels as m
    from .protocols import FlextCliProtocols, FlextCliProtocols as p
    from .services.auth import FlextCliAuth
    from .services.cli import FlextCliCli
    from .services.cli_params import FlextCliCommonParams
    from .services.cmd import FlextCliCmd
    from .services.docx import FlextCliDocx
    from .services.file_tools import FlextCliFileTools
    from .services.formatters import FlextCliFormatters
    from .services.output import FlextCliOutput
    from .services.pipeline import FlextCliPipeline
    from .services.pptx import FlextCliPptx
    from .services.prompts import FlextCliPrompts
    from .services.rules import FlextCliRules
    from .services.runtime import FlextCliRuntime
    from .services.tables import FlextCliTables
    from .services.xlsx import FlextCliXlsx
    from .services.yaml_model import FlextCliYamlModel
    from .typings import FlextCliTypes, FlextCliTypes as t
    from .utilities import FlextCliUtilities, FlextCliUtilities as u
__all__: tuple[str, ...] = (
    "FlextCli",
    "FlextCliAuth",
    "FlextCliCli",
    "FlextCliCmd",
    "FlextCliCommonParams",
    "FlextCliConfig",
    "FlextCliConstants",
    "FlextCliDocx",
    "FlextCliFileTools",
    "FlextCliFormatters",
    "FlextCliModels",
    "FlextCliOutput",
    "FlextCliPipeline",
    "FlextCliPptx",
    "FlextCliPrompts",
    "FlextCliProtocols",
    "FlextCliRules",
    "FlextCliRuntime",
    "FlextCliServiceBase",
    "FlextCliSettings",
    "FlextCliTables",
    "FlextCliTypes",
    "FlextCliUtilities",
    "FlextCliXlsx",
    "FlextCliYamlModel",
    "__author__",
    "__author_email__",
    "__description__",
    "__license__",
    "__title__",
    "__url__",
    "__version__",
    "__version_info__",
    "c",
    "cli",
    "config",
    "d",
    "e",
    "h",
    "m",
    "p",
    "r",
    "s",
    "services",
    "settings",
    "t",
    "u",
    "x",
)

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({
            "._config": ("FlextCliConfig", "config"),
            "._settings": ("FlextCliSettings", "settings"),
            ".api": ("FlextCli", "cli"),
            ".base": ("FlextCliServiceBase", "s"),
            ".constants": ("FlextCliConstants", "c"),
            ".models": ("FlextCliModels", "m"),
            ".protocols": ("FlextCliProtocols", "p"),
            ".services": ("services",),
            ".services.auth": ("FlextCliAuth",),
            ".services.cli": ("FlextCliCli",),
            ".services.cli_params": ("FlextCliCommonParams",),
            ".services.cmd": ("FlextCliCmd",),
            ".services.docx": ("FlextCliDocx",),
            ".services.file_tools": ("FlextCliFileTools",),
            ".services.formatters": ("FlextCliFormatters",),
            ".services.output": ("FlextCliOutput",),
            ".services.pipeline": ("FlextCliPipeline",),
            ".services.pptx": ("FlextCliPptx",),
            ".services.prompts": ("FlextCliPrompts",),
            ".services.rules": ("FlextCliRules",),
            ".services.runtime": ("FlextCliRuntime",),
            ".services.tables": ("FlextCliTables",),
            ".services.xlsx": ("FlextCliXlsx",),
            ".services.yaml_model": ("FlextCliYamlModel",),
            ".typings": ("FlextCliTypes", "t"),
            ".utilities": ("FlextCliUtilities", "u"),
            "flext_core": ("d", "e", "h", "r", "x"),
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
