"""Pydantic and data model type guard implementations for Flext core."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeIs

from pydantic import BaseModel

from flext_core import FlextUtilitiesGuardsTypeCore, t


class FlextUtilitiesGuardsTypeModel:
    """Pydantic and data model type guards.

    Provides type guard functions for validating Pydantic models, configuration
    structures, and normalized value collections (lists, tuples, dicts).
    """

    @staticmethod
    def is_object_list(
        value: t.NormalizedValue,
    ) -> TypeIs[t.ContainerList]:
        """Check if value is a list of normalized values.

        Args:
            value: Value to check.

        Returns:
            True if value is a list, False otherwise.

        """
        return isinstance(value, list)

    @staticmethod
    def is_object_tuple(
        value: t.GuardInput,
    ) -> TypeIs[tuple[t.NormalizedValue, ...]]:
        """Check if value is a tuple of normalized values.

        Args:
            value: Value to check.

        Returns:
            True if value is a tuple, False otherwise.

        """
        return isinstance(value, tuple)

    @staticmethod
    def is_config_value(value: t.NormalizedValue) -> TypeIs[t.NormalizedValue]:
        """Check if value is a valid configuration value.

        Configuration values are None, scalars, or containers (list, tuple, dict)
        with scalar/None contents.

        Args:
            value: Value to check.

        Returns:
            True if value is a valid configuration value, False otherwise.

        """
        if value is None or FlextUtilitiesGuardsTypeCore.is_scalar(value):
            return True
        if isinstance(value, (list, tuple)):
            for item in value:
                if not (item is None or FlextUtilitiesGuardsTypeCore.is_scalar(item)):
                    return False
            return True
        if isinstance(value, Mapping):
            for item in value.values():
                if not (item is None or FlextUtilitiesGuardsTypeCore.is_scalar(item)):
                    return False
            return True
        return False

    @staticmethod
    def is_configuration_dict(
        value: t.ValueOrModel,
    ) -> TypeIs[t.Dict]:
        """Check if value is a valid configuration dictionary.

        Configuration dicts are Dict model instances or mappings with container values.

        Args:
            value: Value to check.

        Returns:
            True if value is a valid configuration dict, False otherwise.

        """
        if isinstance(value, t.Dict):
            for item_value in value.root.values():
                if isinstance(
                    item_value,
                    BaseModel,
                ) or not FlextUtilitiesGuardsTypeCore.is_container(item_value):
                    return False
            return True
        return isinstance(
            value,
            Mapping,
        ) and FlextUtilitiesGuardsTypeCore.all_container_mapping_values(value)

    @staticmethod
    def is_configuration_mapping(
        value: t.ContainerMapping | t.ConfigMap | t.Dict,
    ) -> TypeIs[t.ConfigMap]:
        """Check if value is a valid configuration mapping.

        Accepts ConfigMap/Dict model instances or mappings with container values.

        Args:
            value: Value to check.

        Returns:
            True if value is a valid configuration mapping, False otherwise.

        """
        candidate: Mapping[str, t.ValueOrModel] = value.root if isinstance(value, (t.ConfigMap, t.Dict)) else value
        for item_value in candidate.values():
            if isinstance(
                item_value,
                BaseModel,
            ) or not FlextUtilitiesGuardsTypeCore.is_container(item_value):
                return False
        return True

    @staticmethod
    def is_pydantic_model(value: t.ValueOrModel) -> TypeIs[BaseModel]:
        """Check if value is a Pydantic BaseModel with model_dump method.

        Args:
            value: Value to check.

        Returns:
            True if value is a Pydantic model with callable model_dump, False otherwise.

        """
        return (
            isinstance(value, BaseModel)
            and hasattr(value, "model_dump")
            and callable(value.model_dump)
        )


__all__ = ["FlextUtilitiesGuardsTypeModel"]
