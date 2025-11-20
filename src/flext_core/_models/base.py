"""Base utility patterns extracted from FlextModels.

This module contains the FlextModelsBase class with all base utility patterns
as nested classes. It should NOT be imported directly - use FlextModels.Base instead.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, ClassVar, Self

from beartype.door import is_bearable
from pydantic import Field, HttpUrl, computed_field, model_validator

from flext_core._models.collections import FlextModelsCollections
from flext_core._models.entity import FlextModelsEntity
from flext_core._models.metadata import Metadata
from flext_core.constants import FlextConstants
from flext_core.utilities import FlextUtilities

if TYPE_CHECKING:
    from collections.abc import Callable

    from flext_core._models.service import OperationCallable


class FlextModelsBase:
    """Base utility pattern container class.

    This class acts as a namespace container for base utility patterns.
    All nested classes are accessed via FlextModels.Base.* in the main models.py.
    """

    # Use zero-dependency Metadata from _models/metadata.py
    Metadata = Metadata

    class Payload[T](
        FlextModelsEntity.ArbitraryTypesModel,
        FlextModelsEntity.IdentifiableMixin,
        FlextModelsEntity.TimestampableMixin,
    ):
        """Enhanced payload model with runtime type validation.

        Uses IdentifiableMixin for id and TimestampableMixin for created_at.
        Runtime Type Validation:
            Payload[User](data=user)    # Validates user is User type
            Payload[User](data=product) # TypeError automatically
        """

        _expected_data_type: ClassVar[type | None] = None

        data: T = Field(...)
        metadata: dict[str, str | int | float] = Field(default_factory=dict)
        expires_at: datetime | None = None
        correlation_id: str | None = None
        source_service: str | None = None
        message_type: str | None = None

        def __class_getitem__(cls, item: type | tuple[type, ...]) -> type[Self]:
            """Intercept Payload[T] to create typed subclass for runtime validation.

            Returns:
                A typed subclass with _expected_data_type set to the provided type.

            """
            actual_type = item[0] if isinstance(item, tuple) else item
            cls_name = getattr(cls, "__name__", "Payload")
            cls_qualname = getattr(cls, "__qualname__", "Payload")
            type_name = getattr(actual_type, "__name__", str(actual_type))

            return FlextUtilities.Generators.create_dynamic_type_subclass(
                f"{cls_name}[{type_name}]",
                cls,
                {
                    "_expected_data_type": actual_type,
                    "__module__": cls.__module__,
                    "__qualname__": f"{cls_qualname}[{type_name}]",
                },
            )

        @model_validator(mode="after")
        def _validate_data_type(self) -> Self:
            """Validate data field matches expected type.

            Returns:
                Self: The validated instance with data matching expected type.

            Raises:
                TypeError: If data field doesn't match expected type.

            """
            if self._expected_data_type is not None and self.data is not None:
                try:
                    if isinstance(self._expected_data_type, type):
                        type_mismatch = not isinstance(
                            self.data,
                            self._expected_data_type,
                        )
                    else:
                        type_mismatch = not is_bearable(
                            self.data,
                            self._expected_data_type,
                        )
                except (TypeError, AttributeError):
                    type_mismatch = False

                if type_mismatch:
                    expected_name = getattr(
                        self._expected_data_type,
                        "__name__",
                        str(self._expected_data_type),
                    )
                    actual_name = type(self.data).__name__
                    msg = (
                        f"Payload[{expected_name}] received data of type {actual_name} "
                        f"instead of {expected_name}. Data: {self.data!r}"
                    )
                    raise TypeError(msg)
            return self

        @computed_field
        def is_expired(self) -> bool:
            """Check if payload is expired."""
            if self.expires_at is None:
                return False
            return FlextUtilities.Generators.generate_datetime_utc() > self.expires_at

    class Url(FlextModelsEntity.Value):
        """Enhanced URL value object using Pydantic v2 HttpUrl validation."""

        url: HttpUrl = Field(description="HTTP/HTTPS URL validated by Pydantic v2")

    class LogOperation(FlextModelsEntity.ArbitraryTypesModel):
        """Enhanced log operation model."""

        level: str = Field(default_factory=lambda: "INFO")
        message: str
        context: dict[str, object] = Field(default_factory=dict)
        timestamp: datetime = Field(
            default_factory=FlextUtilities.Generators.generate_datetime_utc
        )
        source: str | None = None
        operation: str | None = None
        obj: object | None = None

    class TimestampConfig(FlextModelsCollections.Config):
        """Enhanced timestamp configuration."""

        obj: object
        use_utc: bool = Field(default_factory=lambda: True)
        auto_update: bool = Field(default_factory=lambda: True)
        format: str = "%Y-%m-%dT%H:%M:%S.%fZ"
        timezone: str | None = None
        created_at_field: str = Field("created_at", pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$")
        updated_at_field: str = Field("updated_at", pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$")
        field_names: dict[str, str] = Field(
            default_factory=lambda: {
                "created_at": "created_at",
                "updated_at": "updated_at",
            },
        )

    class SerializationRequest(FlextModelsEntity.ArbitraryTypesModel):
        """Enhanced serialization request."""

        data: object
        format: str = Field(
            default_factory=lambda: FlextConstants.Cqrs.SerializationFormat.JSON,
        )
        encoding: str = Field(default_factory=lambda: "utf-8")
        compression: str | None = None
        pretty_print: bool = False
        use_model_dump: bool = True
        indent: int | None = None
        sort_keys: bool = False
        ensure_ascii: bool = False

    class ConditionalExecutionRequest(FlextModelsEntity.ArbitraryTypesModel):
        """Conditional execution request."""

        condition: Callable[[object], bool]
        true_action: OperationCallable | None = None
        false_action: OperationCallable | None = None
        context: dict[str, object] = Field(default_factory=dict)

        @classmethod
        def validate_condition(
            cls,
            v: OperationCallable | None,
        ) -> OperationCallable | None:
            """Validate callables are properly defined (Pydantic v2 mode='after')."""
            return v

    class StateInitializationRequest(FlextModelsEntity.ArbitraryTypesModel):
        """State initialization request."""

        data: object
        state_key: str
        initial_value: object
        ttl_seconds: int | None = None
        persistence_level: str = Field(
            default_factory=lambda: FlextConstants.Cqrs.PersistenceLevel.MEMORY,
        )
        field_name: str = "state"
        state: object


__all__ = ["FlextModelsBase"]
