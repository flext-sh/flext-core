"""Base Tap class for FLEXT data extraction components.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

This module provides a unified base class for all tap implementations
to eliminate duplication and standardize the interface.
"""

from __future__ import annotations

import logging
import time
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

from flext_core.domain.types import ServiceResult

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass
class TapMetrics:
    """Metrics collection for tap operations."""

    records_extracted: int = 0
    records_transformed: int = 0
    bytes_processed: int = 0
    execution_time: float = 0.0
    errors_count: int = 0
    start_time: float = field(default_factory=time.time)

    def add_record(self, record_size: int = 0) -> None:
        """Add a processed record to metrics."""
        self.records_extracted += 1
        self.bytes_processed += record_size

    def add_error(self) -> None:
        """Add an error to metrics."""
        self.errors_count += 1

    def finalize(self) -> None:
        """Finalize metrics collection."""
        self.execution_time = time.time() - self.start_time


@dataclass
class ValidationResult:
    """Result of configuration validation."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class BaseTap(ABC):
    """Base class for all tap implementations.

    Implements the Template Method pattern to provide common functionality
    while allowing subclasses to override specific behavior.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize tap with configuration.

        Args:
            config: Tap configuration dictionary

        """
        self.config = config
        self.metrics = TapMetrics()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._validate_config()

    def discover(self) -> ServiceResult[dict[str, Any]]:
        """Discover available streams and their schemas.

        Returns:
            ServiceResult containing stream discovery information

        """
        try:
            schema = self._get_schema()
            return ServiceResult.ok(schema)
        except (ValueError, KeyError, TypeError) as e:
            return ServiceResult.fail(f"Discovery failed: {e!s}")
        except OSError as e:
            return ServiceResult.fail(f"Discovery I/O error: {e!s}")
        except Exception as e:
            # Re-raise unexpected exceptions for proper debugging
            msg = f"Unexpected error during discovery: {e!s}"
            raise RuntimeError(msg) from e

    def sync(self, config: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
        """Sync data from the source.

        Args:
            config: Optional sync configuration

        Yields:
            Records from the source

        """
        sync_config = config or self.config
        self._validate_sync_config(sync_config)

        for record in self._extract_data(sync_config):
            transformed_record = self._transform_record(record)
            self.metrics.add_record(len(str(transformed_record)))
            yield transformed_record

    @abstractmethod
    def _get_schema(self) -> dict[str, dict[str, Any]]:
        """Get schema for available streams.

        Returns:
            Schema dictionary

        """
        raise NotImplementedError

    @abstractmethod
    def _extract_data(self, config: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """Extract data from the source.

        Args:
            config: Sync configuration

        Yields:
            Raw records from source

        """
        raise NotImplementedError

    def _validate_config(self) -> None:
        """Validate tap configuration.

        Raises:
            ValueError: If configuration is invalid

        """
        if not isinstance(self.config, dict):
            msg = "Config must be a dictionary"  # type: ignore[unreachable]
            raise TypeError(msg)

    def _validate_sync_config(self, config: dict[str, Any]) -> None:
        """Validate sync configuration.

        Args:
            config: Sync configuration to validate

        Raises:
            ValueError: If configuration is invalid

        """
        if not isinstance(config, dict):
            msg = "Sync config must be a dictionary"  # type: ignore[unreachable]
            raise TypeError(msg)

    def _transform_record(self, record: dict[str, Any]) -> dict[str, Any]:
        """Transform a record before yielding.

        Args:
            record: Raw record from source

        Returns:
            Transformed record

        """
        # Default implementation returns record as-is
        # Subclasses can override for custom transformations
        return record

    def get_stream_names(self) -> list[str]:
        """Get list of available stream names.

        Returns:
            List of stream names

        """
        schema = self._get_schema()
        return list(schema.get("streams", {}).keys())

    def get_stream_schema(self, stream_name: str) -> dict[str, Any] | None:
        """Get schema for a specific stream.

        Args:
            stream_name: Name of the stream

        Returns:
            Stream schema or None if not found

        """
        schema = self._get_schema()
        streams = schema.get("streams", {})
        stream_schema = streams.get(stream_name)
        return cast("dict[str, Any] | None", stream_schema)

    def collect_metrics(self) -> dict[str, Any]:
        """Collect performance and execution metrics.

        Returns:
            Dictionary containing all tap execution metrics

        """
        self.metrics.finalize()
        return {
            "records_extracted": self.metrics.records_extracted,
            "records_transformed": self.metrics.records_transformed,
            "bytes_processed": self.metrics.bytes_processed,
            "execution_time": self.metrics.execution_time,
            "errors_count": self.metrics.errors_count,
            "records_per_second": (
                self.metrics.records_extracted / self.metrics.execution_time
                if self.metrics.execution_time > 0
                else 0
            ),
            "bytes_per_second": (
                self.metrics.bytes_processed / self.metrics.execution_time
                if self.metrics.execution_time > 0
                else 0
            ),
        }

    def validate_config(self) -> ValidationResult:
        """Validate tap configuration comprehensively.

        Returns:
            ValidationResult with validation status and details

        """
        result = ValidationResult(is_valid=True)
        try:
            self._validate_config()

            # Additional enterprise validations
            if not self.config.get("name"):
                result.warnings.append("Tap name not specified")

            if not self.config.get("version"):
                result.warnings.append("Tap version not specified")

            # Check for required connection parameters
            required_fields = getattr(self, "_required_config_fields", [])
            for field in required_fields:
                if field not in self.config:
                    result.errors.append(f"Required field missing: {field}")
                    result.is_valid = False

        except (TypeError, ValueError, AttributeError, RuntimeError) as e:
            result.errors.append(f"Configuration validation failed: {e}")
            result.is_valid = False

        return result

    def health_check(self) -> ServiceResult[dict[str, Any]]:
        """Perform health check on the tap data source.

        Returns:
            ServiceResult with health status information

        """
        try:
            # Basic connection test
            health_info = {
                "status": "healthy",
                "timestamp": time.time(),
                "tap_name": self.__class__.__name__,
                "config_valid": True,
            }

            # Validate configuration
            validation = self.validate_config()
            health_info["config_valid"] = validation.is_valid
            if not validation.is_valid:
                health_info["config_errors"] = validation.errors
                health_info["status"] = "unhealthy"

            # Test schema discovery
            try:
                schema_result = self.discover()
                if schema_result.is_success:
                    health_info["schema_available"] = True
                    stream_data = schema_result.data or {}
                    health_info["stream_count"] = len(stream_data.get("streams", {}))
                else:
                    health_info["schema_available"] = False
                    health_info["schema_error"] = schema_result.error
                    health_info["status"] = "degraded"
            except (TypeError, ValueError, AttributeError) as e:
                health_info["schema_available"] = False
                health_info["schema_error"] = str(e)
                health_info["status"] = "unhealthy"

            return ServiceResult.ok(health_info)

        except (TypeError, ValueError, AttributeError) as e:
            return ServiceResult.fail(f"Health check failed: {e}")
        except Exception as e:
            # Re-raise unexpected exceptions for proper debugging
            msg = f"Unexpected error during health check: {e!s}"
            raise RuntimeError(msg) from e

    def get_tap_info(self) -> dict[str, Any]:
        """Get comprehensive information about the tap.

        Returns:
            Tap information dictionary with metadata and capabilities

        """
        return {
            "type": self.__class__.__name__,
            "config_keys": list(self.config.keys()),
            "available_streams": self.get_stream_names(),
            "metrics": self.collect_metrics(),
            "capabilities": {
                "discovery": True,
                "sync": True,
                "metrics": True,
                "health_check": True,
                "validation": True,
            },
        }
