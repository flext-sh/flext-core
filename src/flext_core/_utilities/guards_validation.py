from __future__ import annotations

import re
from collections.abc import Mapping, Sized

from pydantic import BaseModel, ValidationError

from flext_core import r, t


class FlextUtilitiesGuardsValidation:
    @staticmethod
    def validate_hostname(hostname: str, field_name: str = "hostname") -> r[str]:
        hostname_pattern = (
            "^(?!-)[a-zA-Z0-9-]{1,63}(?<!-)(\\.[a-zA-Z0-9-]{1,63}(?<!-))*$"
        )
        return FlextUtilitiesGuardsValidation.validate_pattern(
            hostname, hostname_pattern, field_name
        )

    @staticmethod
    def validate_length[T: Sized](
        value: T,
        *,
        min_length: int | None = None,
        max_length: int | None = None,
        field_name: str = "value",
    ) -> r[T]:
        try:
            length = len(value)
        except (TypeError, ValueError):
            return r[T].fail(f"{field_name} length is invalid")
        if min_length is not None and length < min_length:
            return r[T].fail(
                f"{field_name} must have at least {min_length} characters/items"
            )
        if max_length is not None and length > max_length:
            return r[T].fail(
                f"{field_name} must have at most {max_length} characters/items"
            )
        return r[T].ok(value)

    @staticmethod
    def validate_pattern(value: str, pattern: str, field_name: str = "value") -> r[str]:
        if re.search(pattern, value) is None:
            return r[str].fail(f"{field_name} has invalid format")
        return r[str].ok(value)

    @staticmethod
    def validate_port_number(port: int, field_name: str = "port") -> r[int]:
        if isinstance(port, bool):
            return r[int].fail(f"{field_name} must be an integer")
        max_port = 65535
        if port < 1 or port > max_port:
            return r[int].fail(f"{field_name} must be between 1 and 65535")
        return r[int].ok(port)

    @staticmethod
    def validate_positive(value: float, field_name: str = "value") -> r[int | float]:
        if isinstance(value, bool) or value <= 0:
            return r[int | float].fail(f"{field_name} must be positive")
        return r[int | float].ok(value)

    @staticmethod
    def validate_uri(uri: str, field_name: str = "uri") -> r[str]:
        uri_pattern = "^[a-zA-Z][a-zA-Z0-9+.-]*://[^\\s]+$"
        return FlextUtilitiesGuardsValidation.validate_pattern(
            uri, uri_pattern, field_name
        )

    @staticmethod
    def validate_pydantic_model[T: BaseModel](
        model_class: type[T], data: Mapping[str, t.NormalizedValue]
    ) -> r[T]:
        try:
            validated = model_class.model_validate(data)
            return r[T].ok(validated)
        except (ValidationError, TypeError, ValueError) as exc:
            return r[T].fail(f"Validation failed: {exc}")


__all__ = ["FlextUtilitiesGuardsValidation"]
