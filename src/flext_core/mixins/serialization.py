"""FLEXT Serialization - JSON and dictionary conversion functionality.

Provides comprehensive serialization capabilities through hierarchical organization
of conversion utilities and mixin classes. Built for JSON serialization, dictionary
conversion, and data loading with enterprise-grade patterns.

Module Role in Architecture:
    FlextSerialization serves as the serialization foundation providing data
    conversion patterns for object-oriented applications. Integrates with
    FlextResult for type-safe loading and FlextProtocols for serialization contracts.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import cast

from flext_core.protocols import FlextProtocols
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes
from flext_core.utilities import FlextUtilities

# =============================================================================
# TIER 1 MODULE PATTERN - SINGLE MAIN EXPORT
# =============================================================================


class FlextSerialization:
    """Unified serialization system implementing single class pattern.

    This class serves as the single main export consolidating ALL serialization
    functionality with enterprise-grade patterns. Provides comprehensive
    JSON and dictionary conversion capabilities while maintaining clean API.

    Tier 1 Module Pattern: serialization.py -> FlextSerialization
    All serialization functionality is accessible through this single interface.
    """

    # =============================================================================
    # INTERNAL HELPERS
    # =============================================================================

    @staticmethod
    def _serialize_value(value: object) -> object:
        """Serialize a value for JSON compatibility.

        Args:
            value: Value to serialize

        Returns:
            JSON-compatible value

        """
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        if isinstance(value, (list, tuple)):
            typed_value: Iterable[object] = cast("Iterable[object]", value)
            return [FlextSerialization._serialize_value(item) for item in typed_value]
        if isinstance(value, dict):
            typed_dict: dict[object, object] = cast("dict[object, object]", value)
            return {
                str(k): FlextSerialization._serialize_value(v)
                for k, v in typed_dict.items()
            }
        # For complex objects, use safe string representation
        return FlextUtilities.TextProcessor.safe_string(value)

    # =============================================================================
    # CORE SERIALIZATION OPERATIONS
    # =============================================================================

    @staticmethod
    def to_dict_basic(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
    ) -> FlextTypes.Core.Dict:
        """Convert object to basic dictionary representation.

        Args:
            obj: Object to convert

        Returns:
            Dictionary representation of object

        """
        from flext_core.mixins.identification import FlextIdentification
        from flext_core.mixins.timestamps import FlextTimestamps

        result = {}

        # Get object attributes
        try:
            obj_dict = object.__getattribute__(obj, "__dict__")
            for key, value in obj_dict.items():
                if not key.startswith("_"):
                    result[key] = FlextSerialization._serialize_value(value)
        except Exception as e:
            msg = f"Failed to get object attributes: {e}"
            raise ValueError(msg) from e

        # Add timestamp info if available
        if hasattr(obj, "_timestamp_initialized"):
            result["created_at"] = FlextTimestamps.get_created_at(obj)
            result["updated_at"] = FlextTimestamps.get_updated_at(obj)

        # Add ID if available
        if FlextIdentification.has_id(obj):
            result["id"] = FlextIdentification.ensure_id(obj)

        return cast("FlextTypes.Core.Dict", result)

    @staticmethod
    def to_dict(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
    ) -> FlextTypes.Core.Dict:
        """Convert object to dictionary with advanced serialization.

        Args:
            obj: Object to convert

        Returns:
            Dictionary representation with nested object serialization

        """
        from flext_core.protocols import FlextProtocols

        result: FlextTypes.Core.Dict = {}

        try:
            obj_dict = object.__getattribute__(obj, "__dict__")
            for key, value in obj_dict.items():
                if key.startswith("_"):
                    continue

                # Try to_dict_basic first
                if isinstance(value, FlextProtocols.Foundation.HasToDictBasic):
                    try:
                        result[key] = value.to_dict_basic()
                        continue
                    except Exception as e:
                        msg = f"Failed to serialize {key}: {e}"
                        raise ValueError(msg) from e

                # Try to_dict
                if isinstance(value, FlextProtocols.Foundation.HasToDict):
                    try:
                        result[key] = value.to_dict()
                        continue
                    except Exception as e:
                        msg = f"Failed to serialize {key}: {e}"
                        raise ValueError(msg) from e

                # Handle lists
                if isinstance(value, list):
                    serialized_list: FlextTypes.Core.List = []
                    item_list: list[object] = cast("list[object]", value)
                    for item in item_list:
                        if isinstance(item, FlextProtocols.Foundation.HasToDictBasic):
                            try:
                                item_dict = item.to_dict_basic()
                                serialized_list.append(item_dict)
                                continue
                            except Exception as e:
                                msg = f"Failed to serialize list item: {e}"
                                raise ValueError(msg) from e
                        serialized_list.append(item)
                    result[key] = serialized_list
                    continue

                # Skip None values
                if value is None:
                    continue

                result[key] = value
        except Exception as e:
            msg = f"Failed to get object attributes: {e}"
            raise ValueError(msg) from e

        return result

    @staticmethod
    def to_json(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
        indent: int | None = None,
    ) -> str:
        """Convert object to JSON string.

        Args:
            obj: Object to convert
            indent: JSON indentation level

        Returns:
            JSON string representation

        """
        data = FlextSerialization.to_dict_basic(obj)
        return json.dumps(data, indent=indent, default=str)

    @staticmethod
    def load_from_dict(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
        data: FlextTypes.Core.Dict,
    ) -> None:
        """Load object attributes from dictionary.

        Args:
            obj: Object to load data into
            data: Dictionary of attributes to load

        """
        for key, value in data.items():
            try:
                setattr(obj, key, value)
            except Exception:  # noqa: S112  # nosec B112
                # Best-effort: skip attributes that cannot be set
                continue

    @staticmethod
    def load_from_json(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
        json_str: str,
    ) -> FlextResult[None]:
        """Load object attributes from JSON string.

        Args:
            obj: Object to load data into
            json_str: JSON string to parse

        Returns:
            FlextResult indicating success or failure

        """
        from flext_core.result import FlextResult

        try:
            data = json.loads(json_str)
            if not isinstance(data, dict):
                return FlextResult[None].fail("JSON data must be a dictionary")
            FlextSerialization.load_from_dict(obj, cast("FlextTypes.Core.Dict", data))
            return FlextResult[None].ok(None)
        except Exception as e:
            return FlextResult[None].fail(f"Failed to load from JSON: {e}")

    # =============================================================================
    # MIXIN CLASS
    # =============================================================================

    class Serializable:
        """Mixin class providing serialization functionality.

        This mixin adds JSON and dictionary serialization capabilities to any class,
        including conversion and loading methods.
        """

        def to_dict_basic(self) -> FlextTypes.Core.Dict:
            """Convert object to basic dictionary representation."""
            return FlextSerialization.to_dict_basic(self)

        def to_dict(self) -> FlextTypes.Core.Dict:
            """Convert object to dictionary with advanced serialization."""
            return FlextSerialization.to_dict(self)

        def to_json(self, indent: int | None = None) -> str:
            """Convert object to JSON string."""
            return FlextSerialization.to_json(self, indent)

        def load_from_dict(self, data: FlextTypes.Core.Dict) -> None:
            """Load object attributes from dictionary."""
            FlextSerialization.load_from_dict(self, data)

        def load_from_json(self, json_str: str) -> None:
            """Load object attributes from JSON string."""
            result = FlextSerialization.load_from_json(self, json_str)
            if result.is_failure:
                raise ValueError(result.error)


__all__ = [
    "FlextSerialization",
]
