"""Container models for the FLEXT type system.

Pydantic RootModel-based container classes for dictionaries, lists, and validators.
These are the concrete implementations exposed through FlextModels (``m.*``).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import typing
from collections.abc import Callable, ItemsView, Iterator, KeysView, Mapping, ValuesView
from datetime import datetime
from pathlib import Path
from typing import Annotated, ClassVar, cast, override

from pydantic import BaseModel, ConfigDict, Field, RootModel

from flext_core.typings import t


class FlextModelsContainers:
    """Pydantic container models for the FLEXT type system.

    Provides RootModel-based dict and list containers for type-safe
    configuration, service maps, and validator collections.
    Access via ``m.ConfigMap``, ``m.Dict``, etc.
    """

    class ErrorCodeMap(BaseModel):
        """Model for nested error code mappings."""

        codes: Annotated[
            dict[str, int],
            Field(
                default_factory=dict,
                title="Error Codes",
                description="Mapping from error keys to numeric error codes.",
                examples=[{"timeout": 504, "invalid_payload": 400}],
            ),
        ]

    class ObjectList(RootModel[list[str | int | float | bool | datetime | Path]]):
        """Sequence of container values for batch operations."""

        root: Annotated[
            list[str | int | float | bool | datetime | Path],
            Field(
                default_factory=list,
                title="Object List",
                description=(
                    "Ordered container values for batch operations "
                    "(scalar, BaseModel, or Path)."
                ),
                examples=[["item-1", 2, True]],
            ),
        ]

    class _RootDict[RootValueT](typing.Protocol):
        root: dict[str, RootValueT]

    class _RootDictModel[DictValueT](RootModel[dict[str, DictValueT]]):
        def __getitem__(self, key: str) -> DictValueT:
            return self.root[key]

        def __setitem__(self, key: str, value: DictValueT) -> None:
            self.root[key] = value

        def __delitem__(self, key: str) -> None:
            del self.root[key]

        def __len__(self) -> int:
            return len(self.root)

        def __contains__(self, key: str) -> bool:
            return key in self.root

        def clear(self) -> None:
            self.root.clear()

        def get(self, key: str, default: DictValueT | None = None) -> DictValueT | None:
            return self.root.get(key, default)

        def items(self) -> ItemsView[str, DictValueT]:
            return self.root.items()

        def keys(self) -> KeysView[str]:
            return self.root.keys()

        def pop(self, key: str, default: DictValueT | None = None) -> DictValueT | None:
            return self.root.pop(key, default)

        def popitem(self) -> tuple[str, DictValueT]:
            return self.root.popitem()

        def setdefault(self, key: str, default: DictValueT) -> DictValueT:
            return self.root.setdefault(key, default)

        def update(self, other: Mapping[str, DictValueT]) -> None:
            self.root.update(other)

        @override
        def __iter__(self) -> Iterator[str]:
            return iter(self.root)

        def values(self) -> ValuesView[DictValueT]:
            return self.root.values()

    cast("type", Mapping).register(_RootDictModel)

    class Dict(_RootDictModel[object]):
        """Generic dictionary container. Use ``m.Dict``."""

        # Used by: flext-core CQRS/message payloads (`_models/base.py`, `_models/cqrs.py`),
        # context and handlers (`_models/context.py`, `_models/handler.py`), plus
        # consumer runtime payload shims in flext-observability and flext-db-oracle.
        # Migration note: command/query/event payloads should move to domain-specific
        # Pydantic models instead of generic key-value dictionaries.

        root: Annotated[
            dict[str, object],
            Field(
                default_factory=dict,
                title="Dictionary Payload",
                description=(
                    "Dictionary payload storing strict container values "
                    "(scalar, BaseModel, or Path)."
                ),
                examples=[{"request_id": "req-123", "retry_count": 3, "dry_run": True}],
            ),
        ]

    class ConfigMap(_RootDictModel[object]):
        """Configuration map container. Use ``m.ConfigMap``."""

        # Used by: flext-core container/context/runtime/logging/exceptions, flext-tests
        # fixtures, and consumer projects (notably flext-db-oracle, flext-ldif,
        # flext-target-ldif). Most call sites represent typed config contracts and can
        # be replaced by explicit domain settings models over time.

        root: Annotated[
            dict[str, object],
            Field(
                default_factory=dict,
                title="Configuration Map",
                description="Configuration entries keyed by normalized setting names.",
                examples=[
                    {"timeout_seconds": 30, "environment": "dev", "debug": False}
                ],
            ),
        ]

    class ServiceMap(_RootDictModel[type[BaseModel] | Callable[..., BaseModel]]):
        """Service registry map container. Use ``m.ServiceMap``."""

        # Used by: exported alias surface in `flext_core/models.py`; no direct runtime
        # constructor usage currently found by ast-grep. Intended payload represents
        # service classes or service factory callables, not generic container values.

        root: Annotated[
            dict[str, type[BaseModel] | Callable[..., BaseModel]],
            Field(
                default_factory=dict,
                title="Service Registry Map",
                description="Service registry entries keyed by service identifiers.",
                examples=[
                    {"user_service": "UserService", "session_factory": "build_session"}
                ],
            ),
        ]

    class ErrorMap(_RootDictModel[int | str | BaseModel]):
        """Error type mapping container.

        Replaces: ErrorTypeMapping
        """

        root: Annotated[
            dict[str, int | str | BaseModel],
            Field(
                default_factory=dict,
                title="Error Map",
                description="Error catalog mapping keys to codes, messages, or nested code maps.",
                examples=[{"user_missing": 404, "bad_input": "invalid data"}],
            ),
        ]

    class FactoryMap(_RootDictModel[t.FactoryCallable]):
        """Map of factory registration callables.

        Replaces: Mapping[str, FactoryRegistrationCallable]
        """

        root: Annotated[
            dict[str, t.FactoryCallable],
            Field(
                default_factory=dict,
                title="Factory Map",
                description="Factory callables keyed by registration name.",
                examples=[{"db_client": "create_db_client"}],
            ),
        ]

    class ResourceMap(_RootDictModel[t.ResourceCallable]):
        """Map of resource callables.

        Replaces: Mapping[str, ResourceCallable]
        """

        root: Annotated[
            dict[str, t.ResourceCallable],
            Field(
                default_factory=dict,
                title="Resource Map",
                description="Lifecycle resource factories keyed by resource name.",
                examples=[{"connection": "open_connection"}],
            ),
        ]

    class ValidatorCallable(
        RootModel[
            Callable[
                [t.Scalar | BaseModel | None],
                t.Scalar | BaseModel | None,
            ]
        ]
    ):
        """Callable validator container. Fixed types: ScalarValue | BaseModel."""

        root: Callable[
            [t.Scalar | BaseModel | None],
            t.Scalar | BaseModel | None,
        ] = Field(
            title="Validator Callable",
            description="Callable that validates or transforms one scalar/model input value.",
            examples=["identity_validator"],
        )

        def __call__(
            self, value: t.Scalar | BaseModel | None
        ) -> t.Scalar | BaseModel | None:
            """Execute validator."""
            return self.root(value)

    class _RootValidatorMapModel(
        RootModel[
            dict[
                str,
                Callable[
                    [t.Scalar | BaseModel | None],
                    t.Scalar | BaseModel | None,
                ],
            ]
        ]
    ):
        """Shared API for validator map containers."""

        def items(
            self,
        ) -> ItemsView[
            str,
            Callable[
                [t.Scalar | BaseModel | None],
                t.Scalar | BaseModel | None,
            ],
        ]:
            """Get validator items."""
            validated: dict[
                str,
                Callable[
                    [t.Scalar | BaseModel | None],
                    t.Scalar | BaseModel | None,
                ],
            ] = {key: value for key, value in self.root.items() if callable(value)}
            return validated.items()

        def values(
            self,
        ) -> ValuesView[
            Callable[
                [t.Scalar | BaseModel | None],
                t.Scalar | BaseModel | None,
            ],
        ]:
            """Get validator values."""
            validated: dict[
                str,
                Callable[
                    [t.Scalar | BaseModel | None],
                    t.Scalar | BaseModel | None,
                ],
            ] = {key: value for key, value in self.root.items() if callable(value)}
            return validated.values()

    class FieldValidatorMap(_RootValidatorMapModel):
        """Map of field validators."""

        root: Annotated[
            dict[
                str,
                Callable[
                    [t.Scalar | BaseModel | None],
                    t.Scalar | BaseModel | None,
                ],
            ],
            Field(
                default_factory=dict,
                title="Field Validator Map",
                description="Field-level validators keyed by field name.",
                examples=[{"email": "validate_email"}],
            ),
        ]

    class ConsistencyRuleMap(_RootValidatorMapModel):
        """Map of consistency rules."""

        root: Annotated[
            dict[
                str,
                Callable[
                    [t.Scalar | BaseModel | None],
                    t.Scalar | BaseModel | None,
                ],
            ],
            Field(
                default_factory=dict,
                title="Consistency Rule Map",
                description="Consistency rule callables keyed by rule identifier.",
                examples=[{"order_total": "validate_order_total"}],
            ),
        ]

    class EventValidatorMap(_RootValidatorMapModel):
        """Map of event validators."""

        root: Annotated[
            dict[
                str,
                Callable[
                    [t.Scalar | BaseModel | None],
                    t.Scalar | BaseModel | None,
                ],
            ],
            Field(
                default_factory=dict,
                title="Event Validator Map",
                description="Event validator callables keyed by event type or alias.",
                examples=[{"user.created": "validate_user_created"}],
            ),
        ]

    class BatchResultDict(BaseModel):
        """Result payload model for batch operation outputs."""

        model_config: ClassVar[ConfigDict] = ConfigDict(
            validate_assignment=True, extra="forbid"
        )
        results: Annotated[
            list[t.Scalar | None],
            Field(
                default_factory=list,
                title="Batch Results",
                description="Batch result values in processing order.",
                examples=[["ok", 1, None]],
            ),
        ]

        errors: Annotated[
            list[tuple[int, str]],
            Field(
                default_factory=list,
                title="Batch Errors",
                description="Batch error tuples as (index, message).",
                examples=[[(0, "invalid payload")]],
            ),
        ]
        total: Annotated[
            int,
            Field(
                default=0,
                title="Total Items",
                description="Total number of batch items processed.",
                examples=[10],
            ),
        ] = 0
        success_count: Annotated[
            int,
            Field(
                default=0,
                title="Success Count",
                description="Number of batch items processed successfully.",
                examples=[8],
            ),
        ] = 0
        error_count: Annotated[
            int,
            Field(
                default=0,
                title="Error Count",
                description="Number of batch items that failed with errors.",
                examples=[2],
            ),
        ] = 0


__all__ = ["FlextModelsContainers"]
