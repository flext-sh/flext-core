"""Base Target class for FLEXT data loading components.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

This module provides a unified base class for all target implementations
to eliminate duplication and standardize the interface.
"""

from __future__ import annotations

import logging
from abc import ABC
from abc import abstractmethod
from typing import Any

from flext_core.domain.types import ServiceResult


class BaseTarget(ABC):
    """Base class for all target implementations.

    Implements the Template Method pattern to provide common functionality
    while allowing subclasses to override specific behavior.
    """

    def __init__(self, config: Any) -> None:
        """Initialize target with configuration.

        Args:
            config: Target configuration dictionary

        """
        self.config = config
        self._validate_config()

    def write_batch(self, records: list[dict[str, Any]]) -> ServiceResult[None]:
        """Write a batch of records to the target.

        Args:
            records: List of records to write

        Returns:
            ServiceResult indicating success or failure

        """
        try:
            self._validate_records(records)
            self._write_records(records)
            return ServiceResult[None].ok(None)
        except (ValueError, KeyError, TypeError) as e:
            return ServiceResult[None].fail(f"Write validation failed: {e!s}")
        except OSError as e:
            return ServiceResult[None].fail(f"Write I/O error: {e!s}")
        except Exception as e:
            # Log the exception for debugging
            logging.getLogger(__name__).exception("Unexpected error during write")
            return ServiceResult.fail(f"Unexpected error during write: {e!s}")

    def write_record(self, record: dict[str, Any]) -> ServiceResult[None]:
        """Write a single record to the target.

        Args:
            record: Record to write

        Returns:
            ServiceResult indicating success or failure

        """
        return self.write_batch([record])

    def flush(self) -> ServiceResult[None]:
        """Flush any buffered data to the target.

        Returns:
            ServiceResult indicating success or failure

        """
        try:
            self._flush_buffers()
            return ServiceResult[None].ok(None)
        except (ValueError, OSError) as e:
            return ServiceResult[None].fail(f"Flush failed: {e!s}")
        except Exception as e:
            # Log the exception for debugging
            logging.getLogger(__name__).exception("Unexpected error during flush")
            return ServiceResult.fail(f"Unexpected error during flush: {e!s}")

    @abstractmethod
    def _write_records(self, records: list[dict[str, Any]]) -> None:
        """Write records to the target.

        Args:
            records: Records to write

        """
        raise NotImplementedError

    def _validate_config(self) -> None:
        """Validate target configuration.

        Raises:
            ValueError: If configuration is invalid

        """
        if not isinstance(self.config, dict):
            msg = "Config must be a dictionary"
            raise TypeError(msg)

    def _validate_records(self, records: Any) -> None:
        """Validate records before writing.

        Args:
            records: Records to validate

        Raises:
            ValueError: If records are invalid

        """
        if not isinstance(records, list):
            msg = "Records must be a list"
            raise TypeError(msg)

        for record in records:
            if not isinstance(record, dict):
                msg = "Each record must be a dictionary"
                raise TypeError(msg)

    @abstractmethod
    def _flush_buffers(self) -> None:
        """Flush any buffered data.

        Subclasses must implement custom buffering behavior.
        """
        raise NotImplementedError

    def get_target_info(self) -> dict[str, Any]:
        """Get information about the target.

        Returns:
            Target information dictionary

        """
        return {
            "type": self.__class__.__name__,
            "config_keys": list(self.config.keys()),
        }
