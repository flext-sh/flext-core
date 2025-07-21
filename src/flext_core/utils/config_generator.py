"""Centralized configuration generator for FLEXT projects.

This module provides a unified configuration generation system for all FLEXT projects,
eliminating duplicate generate_config.py scripts and providing consistent configuration
patterns across Oracle OIC, Oracle WMS, and other FLEXT components.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from abc import ABC
from abc import abstractmethod
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar

from dotenv import load_dotenv
from pydantic import ValidationError

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


class ProjectType(StrEnum):
    """FLEXT project types for configuration generation."""

    TAP_ORACLE_OIC = "tap-oracle-oic"
    TARGET_ORACLE_OIC = "target-oracle-oic"
    TAP_ORACLE_WMS = "tap-oracle-wms"
    TARGET_ORACLE_WMS = "target-oracle-wms"
    ORACLE_OIC_EXT = "oracle-oic-ext"


class ConfigSection(StrEnum):
    """Configuration section types."""

    OAUTH = "oauth"
    API = "api"
    EXTRACTION = "extraction"
    TARGET = "target"
    PERFORMANCE = "performance"
    DEBUG = "debug"
    WMS = "wms"


class BaseConfigGenerator(ABC):
    """Base configuration generator for FLEXT projects."""

    def __init__(self, project_type: ProjectType) -> None:
        """Initialize configuration generator."""
        self.project_type = project_type
        self.config_sections: dict[ConfigSection, dict[str, Any]] = {}
        load_dotenv()

    @abstractmethod
    def generate_config(self) -> dict[str, Any]:
        """Generate project-specific configuration."""

    def add_oauth_config(self, env_prefix: str = "OIC") -> None:
        """Add OAuth2 configuration section."""
        oauth_config = {
            "base_url": os.getenv(f"{env_prefix}_IDCS_CLIENT_AUD", "").rstrip("/"),
            "oauth_client_id": os.getenv(f"{env_prefix}_IDCS_CLIENT_ID"),
            "oauth_client_secret": os.getenv(f"{env_prefix}_IDCS_CLIENT_SECRET"),
            "oauth_token_url": f"{os.getenv(f'{env_prefix}_IDCS_URL')}/oauth2/v1/token",
            "oauth_scope": os.getenv(f"{env_prefix}_IDCS_CLIENT_AUD"),
        }
        self.config_sections[ConfigSection.OAUTH] = oauth_config

    def add_api_config(self, env_prefix: str = "OIC") -> None:
        """Add API configuration section."""
        api_config = {
            "api_version": os.getenv(f"{env_prefix}_API_VERSION", "v1"),
            "page_size": int(os.getenv(f"{env_prefix}_PAGE_SIZE", "100")),
            "request_timeout": int(os.getenv(f"{env_prefix}_TIMEOUT", "60")),
            "max_retries": int(os.getenv("HTTP_MAX_RETRIES", "3")),
        }
        self.config_sections[ConfigSection.API] = api_config

    def add_performance_config(self, env_prefix: str = "OIC") -> None:
        """Add performance configuration section."""
        performance_config = {
            "batch_size": int(os.getenv(f"{env_prefix}_BATCH_SIZE", "100")),
            "max_concurrent_requests": int(
                os.getenv(f"{env_prefix}_MAX_CONCURRENT_REQUESTS", "5"),
            ),
            "max_concurrent_uploads": int(
                os.getenv(f"{env_prefix}_MAX_CONCURRENT_UPLOADS", "3"),
            ),
            "request_timeout": int(os.getenv(f"{env_prefix}_TIMEOUT", "60")),
        }
        self.config_sections[ConfigSection.PERFORMANCE] = performance_config

    def add_debug_config(self) -> None:
        """Add debug configuration section."""
        debug_config = {
            "debug": os.getenv("DEBUG", "false").lower() == "true",
            "log_level": os.getenv("LOG_LEVEL", "INFO"),
        }
        self.config_sections[ConfigSection.DEBUG] = debug_config

    def add_wms_config(self, env_prefix: str = "WMS") -> None:
        """Add WMS configuration section."""
        wms_config = {
            "base_url": os.getenv(
                f"{env_prefix}_BASE_URL",
                "https://test.oracle.com/wms/api/v1",
            ),
            "username": os.getenv(f"{env_prefix}_USERNAME", "test_user"),
            "password": os.getenv(f"{env_prefix}_PASSWORD", "test_password"),
            "timeout": int(os.getenv(f"{env_prefix}_TIMEOUT", "300")),
            "enable_kpi_calculation": os.getenv(
                f"{env_prefix}_ENABLE_KPI",
                "true",
            ).lower()
            == "true",
            "enable_alerts": os.getenv(f"{env_prefix}_ENABLE_ALERTS", "true").lower()
            == "true",
            "expiry_alert_days": int(
                os.getenv(f"{env_prefix}_EXPIRY_ALERT_DAYS", "30"),
            ),
            "output_path": os.getenv(f"{env_prefix}_OUTPUT_PATH", "./output"),
            "output_format": os.getenv(f"{env_prefix}_OUTPUT_FORMAT", "json"),
        }

        # Add optional database URL
        db_url = os.getenv(f"{env_prefix}_DATABASE_URL")
        if db_url:
            wms_config["database_url"] = db_url

        # Add optional webhook URL
        webhook_url = os.getenv(f"{env_prefix}_ALERT_WEBHOOK_URL")
        if webhook_url:
            wms_config["alert_webhook_url"] = webhook_url

        # Add test mode flag
        if os.getenv(f"{env_prefix}_TEST_MODE", "false").lower() == "true":
            wms_config["test_mode"] = True

        self.config_sections[ConfigSection.WMS] = wms_config

    def merge_sections(self) -> dict[str, Any]:
        """Merge all configuration sections into single config."""
        merged_config = {}
        for section_config in self.config_sections.values():
            merged_config.update(section_config)

        # Remove None values
        return {k: v for k, v in merged_config.items() if v is not None}

    def save_config(
        self,
        config_path: str | Path = "config.json",
        *,
        overwrite: bool = False,
    ) -> None:
        """Save configuration to JSON file."""
        config_path = Path(config_path)

        # Check if config file exists
        if config_path.exists() and not overwrite:
            logger.warning(
                "Configuration file %s already exists. Use overwrite=True to replace.",
                config_path,
            )
            return

        # Backup existing config if it exists
        if config_path.exists():
            backup_path = config_path.with_suffix(f"{config_path.suffix}.bak")
            config_path.rename(backup_path)
            logger.info("Backed up existing config to %s", backup_path)

        # Generate and save new config
        config = self.generate_config()

        try:
            with config_path.open("w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)
            logger.info("Generated configuration saved to %s", config_path)
        except Exception:
            logger.exception("Failed to save configuration")
            raise


class TapOracleOICConfigGenerator(BaseConfigGenerator):
    """Configuration generator for tap-oracle-oic."""

    def __init__(self) -> None:
        """Initialize tap-oracle-oic configuration generator."""
        super().__init__(ProjectType.TAP_ORACLE_OIC)

    def generate_config(self) -> dict[str, Any]:
        """Generate tap-oracle-oic configuration."""
        # Add standard sections
        self.add_oauth_config("OIC")
        self.add_api_config("OIC")
        self.add_debug_config()

        # Add extraction-specific configuration
        entities = os.getenv(
            "OIC_ENTITIES",
            "connections,integrations,packages,lookups",
        )
        extraction_config = {
            "start_date": os.getenv("OIC_START_DATE", "2024-01-01T00:00:00Z"),
            "entities": entities.split(",") if entities else [],
            "enable_incremental": os.getenv("OIC_ENABLE_INCREMENTAL", "true").lower()
            == "true",
            "max_concurrent_requests": int(
                os.getenv("OIC_MAX_CONCURRENT_REQUESTS", "5"),
            ),
        }
        self.config_sections[ConfigSection.EXTRACTION] = extraction_config

        return self.merge_sections()


class TargetOracleOICConfigGenerator(BaseConfigGenerator):
    """Configuration generator for target-oracle-oic."""

    def __init__(self) -> None:
        """Initialize target-oracle-oic configuration generator."""
        super().__init__(ProjectType.TARGET_ORACLE_OIC)

    def generate_config(self) -> dict[str, Any]:
        """Generate target-oracle-oic configuration."""
        # Add standard sections
        self.add_oauth_config("OIC")
        self.add_performance_config("OIC")
        self.add_debug_config()

        # Add target-specific configuration
        target_config = {
            "import_mode": os.getenv("OIC_IMPORT_MODE", "create_or_update"),
            "activate_integrations": os.getenv(
                "OIC_ACTIVATE_INTEGRATIONS",
                "false",
            ).lower()
            == "true",
            "overwrite_existing": os.getenv("OIC_OVERWRITE_EXISTING", "false").lower()
            == "true",
            "validate_before_import": os.getenv(
                "OIC_VALIDATE_BEFORE_IMPORT",
                "true",
            ).lower()
            == "true",
        }
        self.config_sections[ConfigSection.TARGET] = target_config

        return self.merge_sections()


class TargetOracleWMSConfigGenerator(BaseConfigGenerator):
    """Configuration generator for target-oracle-wms."""

    def __init__(self) -> None:
        """Initialize target-oracle-wms configuration generator."""
        super().__init__(ProjectType.TARGET_ORACLE_WMS)

    def generate_config(self) -> dict[str, Any]:
        """Generate target-oracle-wms configuration."""
        # Add WMS-specific configuration
        self.add_wms_config("WMS")
        self.add_debug_config()

        return self.merge_sections()


class OracleOICExtConfigGenerator(BaseConfigGenerator):
    """Configuration generator for oracle-oic-ext."""

    def __init__(self) -> None:
        """Initialize oracle-oic-ext configuration generator."""
        super().__init__(ProjectType.ORACLE_OIC_EXT)

    def generate_config(self) -> dict[str, Any]:
        """Generate oracle-oic-ext configuration."""
        # Add standard sections
        self.add_oauth_config("OIC")
        self.add_api_config("OIC")
        self.add_debug_config()

        return self.merge_sections()


class ConfigGeneratorFactory:
    """Factory for creating project-specific configuration generators."""

    _generators: ClassVar[dict[ProjectType, Callable[[], BaseConfigGenerator]]] = {
        ProjectType.TAP_ORACLE_OIC: TapOracleOICConfigGenerator,
        ProjectType.TARGET_ORACLE_OIC: TargetOracleOICConfigGenerator,
        ProjectType.TARGET_ORACLE_WMS: TargetOracleWMSConfigGenerator,
        ProjectType.ORACLE_OIC_EXT: OracleOICExtConfigGenerator,
    }

    @classmethod
    def create_generator(cls, project_type: ProjectType) -> BaseConfigGenerator:
        """Create configuration generator for project type."""
        generator_class = cls._generators.get(project_type)
        if generator_class is None:
            msg = (
                f"No configuration generator available for project type: {project_type}"
            )
            raise ValueError(
                msg,
            )

        return generator_class()

    @classmethod
    def get_supported_project_types(cls) -> list[ProjectType]:
        """Get list of supported project types."""
        return list(cls._generators.keys())


def generate_project_config(
    project_type: ProjectType,
    config_path: str | Path = "config.json",
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Generate configuration for specified project type.

    Args:
        project_type: The type of FLEXT project to generate config for
        config_path: Path to save the configuration file
        overwrite: Whether to overwrite existing configuration file

    Returns:
        Generated configuration dictionary

    Raises:
        ValueError: If project type is not supported
        ValidationError: If environment variables are invalid

    """
    try:
        # Create generator
        generator = ConfigGeneratorFactory.create_generator(project_type)

        # Generate and save configuration
        generator.save_config(config_path, overwrite=overwrite)

        # Return generated config
        return generator.generate_config()

    except ValidationError:
        logger.exception("Configuration validation failed")
        raise
    except Exception:
        logger.exception("Failed to generate configuration")
        raise


def main() -> None:
    """CLI entry point for configuration generation."""
    parser = argparse.ArgumentParser(description="Generate FLEXT project configuration")
    parser.add_argument(
        "project_type",
        type=str,
        choices=[pt.value for pt in ProjectType],
        help="Project type to generate configuration for",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="config.json",
        help="Path to save configuration file (default: config.json)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing configuration file",
    )

    args = parser.parse_args()

    try:
        project_type = ProjectType(args.project_type)
        generate_project_config(
            project_type=project_type,
            config_path=args.config_path,
            overwrite=args.overwrite,
        )

    except (ValueError, FileNotFoundError, PermissionError, OSError, KeyError):
        sys.exit(1)


if __name__ == "__main__":
    main()
