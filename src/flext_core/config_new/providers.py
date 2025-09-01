"""Configuration providers (new) that supply raw dicts for merging.

Single-class module: defines `FlextConfigProviders` only.

Providers here are minimal and deterministic (no network). Where possible,
we delegate to centralized utilities (FlextUtilities) and constants.
"""

from __future__ import annotations

import contextlib
import json
import os
from json import JSONDecodeError
from pathlib import Path
from typing import Final

from flext_core.constants import FlextConstants
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes


class FlextConfigProviders:
    """Raw config providers producing dicts for the loader."""

    ENV_PREFIX: Final[str] = "FLEXT_"

    @staticmethod
    def from_constants() -> FlextTypes.Config.ConfigDict:
        return {
            "app_name": "flext-app",
            "app_version": "0.1.0",
            "environment": FlextConstants.Config.DEFAULT_ENVIRONMENT,
            "debug": False,
            "log_level": FlextConstants.Config.LogLevel.INFO.value,
            "timeout_seconds": FlextConstants.Network.DEFAULT_TIMEOUT,
            "max_retries": FlextConstants.Defaults.MAX_RETRIES,
            "enable_caching": True,
            "enable_metrics": True,
            "enable_tracing": False,
            "config_source": FlextConstants.Config.ConfigProvider.CONSTANTS_PROVIDER.value,
        }

    @staticmethod
    def from_env(prefix: str = ENV_PREFIX) -> FlextTypes.Config.ConfigDict:
        cfg: FlextTypes.Config.ConfigDict = {}

        def get(name: str) -> str | None:
            return os.environ.get(f"{prefix}{name}")

        if env := get("ENVIRONMENT"):
            cfg["environment"] = env
        if name := get("APP_NAME"):
            cfg["app_name"] = name
        if version := get("APP_VERSION"):
            cfg["app_version"] = version
        if level := get("LOG_LEVEL"):
            cfg["log_level"] = level
        if dbg := get("DEBUG"):
            cfg["debug"] = dbg.lower() in {"1", "true", "yes", "on"}
        if t := get("TIMEOUT_SECONDS"):
            with contextlib.suppress(Exception):
                cfg["timeout_seconds"] = int(t)
        if r := get("MAX_RETRIES"):
            with contextlib.suppress(Exception):
                cfg["max_retries"] = int(r)
        if v := get("ENABLE_CACHING"):
            cfg["enable_caching"] = v.lower() in {"1", "true", "yes", "on"}
        if v := get("ENABLE_METRICS"):
            cfg["enable_metrics"] = v.lower() in {"1", "true", "yes", "on"}
        if v := get("ENABLE_TRACING"):
            cfg["enable_tracing"] = v.lower() in {"1", "true", "yes", "on"}

        if cfg:
            cfg["config_source"] = FlextConstants.Config.ConfigSource.ENVIRONMENT.value
        return cfg

    @staticmethod
    def from_file(path: str | Path) -> FlextResult[FlextTypes.Config.ConfigDict]:
        p = Path(path)
        if not p.exists():
            return FlextResult[FlextTypes.Config.ConfigDict].fail(
                f"Config file not found: {p}"
            )
        # Quick YAML detection and friendly message (no external dependency)
        if p.suffix.lower() in {".yaml", ".yml"}:
            return FlextResult[FlextTypes.Config.ConfigDict].fail(
                "YAML not supported without optional dependency"
            )
        try:
            # JSON only for now (no external deps), matches constraints
            data = json.loads(p.read_text(encoding="utf-8"))
        except OSError as e:
            return FlextResult[FlextTypes.Config.ConfigDict].fail(
                f"Config read error: {e}"
            )
        except JSONDecodeError as e:
            return FlextResult[FlextTypes.Config.ConfigDict].fail(
                f"Invalid JSON config: {e}"
            )
        if not isinstance(data, dict):
            return FlextResult[FlextTypes.Config.ConfigDict].fail(
                "Config root must be a JSON object"
            )
        cfg: FlextTypes.Config.ConfigDict = {k: data[k] for k in data}
        cfg["config_source"] = FlextConstants.Config.ConfigSource.FILE.value
        return FlextResult[FlextTypes.Config.ConfigDict].ok(cfg)
