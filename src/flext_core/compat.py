"""Backward compatibility layer for FLEXT Core models.

This module provides mixin classes that add legacy methods to the modern
Pydantic models, allowing them to work with older test suites or consumers
without modifying the core model logic.
"""

from __future__ import annotations

from typing import cast


class ValueObjectCompatibilityMixin:
    """Provides legacy methods for FlextValueObject."""

    def _should_include_attribute(self, attr_name: str) -> bool:
        """Determines if an attribute should be included in serialization."""
        if attr_name.startswith("_") or attr_name == "metadata":
            return False
        attr = getattr(self, attr_name, None)
        return not callable(attr)

    def _safely_get_attribute(self, attr_name: str) -> object | None:
        """Safely gets an attribute, returning None if it doesn't exist."""
        return getattr(self, attr_name, None)

    def _get_fallback_info(self) -> dict[str, object]:
        """Provides fallback information for serialization errors."""
        return cast(
            "dict[str, object]",
            {
                "error": "Could not serialize value object",
                "class_name": self.__class__.__name__,
                "module": self.__class__.__module__,
            },
        )

    def format_dict(self, data: dict[str, object]) -> str:
        """Formats a dictionary into a key='value' string representation."""
        return ", ".join([f"{k}={v!r}" for k, v in data.items()])

    def _try_manual_extraction(self) -> dict[str, object]:
        """Placeholder for legacy manual extraction."""
        # In the new implementation, this just calls the main extraction method.
        return self._extract_serializable_attributes()

    def _process_serializable_values(
        self, data: dict[str, object]
    ) -> dict[str, object]:
        """Processes values for serialization, converting complex types to string."""
        processed: dict[str, object] = {}
        for key, value in data.items():
            if isinstance(value, (dict, list, tuple)):
                processed[key] = str(
                    cast(
                        "dict[object, object] | list[object] | tuple[object, ...]",
                        value,
                    )
                )
            else:
                processed[key] = value
        return processed

    def _extract_serializable_attributes(self) -> dict[str, object]:
        """Extracts serializable attributes from the value object."""
        try:
            processed: dict[str, object] = {}
            model_fields = cast(
                "dict[str, object]", getattr(self.__class__, "model_fields", {})
            )
            for attr_name in model_fields:
                attr_name_str = attr_name
                if self._should_include_attribute(attr_name_str):
                    processed[attr_name_str] = self._safely_get_attribute(attr_name_str)
            return processed
        except Exception:
            return self._get_fallback_info()
