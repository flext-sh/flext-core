"""Base configuration class for FLEXT components.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

This module provides a unified base class for all configuration
implementations to eliminate duplication.
"""

from __future__ import annotations

from typing import Any

from flext_core.config.base import BaseConfig


class BaseComponentConfig(BaseConfig):
    """Base configuration class for FLEXT components.

    Provides common configuration functionality for taps, targets,
    and other components.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize configuration with keyword arguments.

        Args:
            **kwargs: Configuration parameters

        """
        super().__init__(**kwargs)

    def get_connection_config(self) -> dict[str, Any]:
        """Get connection configuration.

        Returns:
            Connection configuration dictionary

        """
        return self.get_subsection("connection")

    def get_auth_config(self) -> dict[str, Any]:
        """Get authentication configuration.

        Returns:
            Authentication configuration dictionary

        """
        return self.get_subsection("auth")

    def get_stream_config(self, stream_name: str) -> dict[str, Any]:
        """Get configuration for a specific stream.

        Args:
            stream_name: Name of the stream

        Returns:
            Stream configuration dictionary

        """
        streams_config = self.get_subsection("streams") or {}
        stream_config = streams_config.get(stream_name, {})
        # Ensure we return a dict even if the value is not a dict
        if not isinstance(stream_config, dict):
            return {}
        return stream_config

    def validate_required_fields(self, required_fields: list[str]) -> None:
        """Validate that required fields are present.

        Args:
            required_fields: List of required field names

        Raises:
            ValueError: If any required field is missing

        """
        missing_fields = [
            field
            for field in required_fields
            if not hasattr(self, field) or getattr(self, field) is None
        ]

        if missing_fields:
            msg = f"Missing required configuration fields: {', '.join(missing_fields)}"
            raise ValueError(msg)

    def get_logging_config(self) -> dict[str, Any]:
        """Get logging configuration.

        Returns:
            Logging configuration dictionary

        """
        return self.get_subsection("logging")

    def get_performance_config(self) -> dict[str, Any]:
        """Get performance configuration.

        Returns:
            Performance configuration dictionary

        """
        return self.get_subsection("performance")
