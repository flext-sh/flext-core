"""CLI validation helpers shared through ``u.Cli``."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import ClassVar

from flext_cli import c, m, p, r, t
from flext_core import u

PydanticValidationError = m.ValidationError


class FlextCliUtilitiesValidation:
    """Validation methods exposed directly on ``u.Cli``."""

    _module_logger: ClassVar[p.Logger] = u.fetch_logger(__name__)

    @staticmethod
    def process_mapping[T, U](
        items: t.MappingKV[str, T],
        processor: t.Cli.MappingProcessor[T, U],
        on_error: str = "fail",
    ) -> p.Result[t.MappingKV[str, U]]:
        """Process a mapping of items with canonical error handling."""
        errors: t.MutableSequenceOf[str] = []
        values: MutableMapping[str, U] = {}
        for key, value in items.items():
            try:
                values[key] = processor(key, value)
            except c.Cli.CLI_SAFE_EXCEPTIONS as exc:
                if on_error == "fail":
                    return r[t.MappingKV[str, U]].fail(f"Error processing {key}: {exc}")
                if on_error == "collect":
                    errors.append(f"{key}: {exc}")
                else:
                    FlextCliUtilitiesValidation._module_logger.debug(
                        f"process_mapping skip key {key}: {exc}", exc_info=False
                    )
        return (
            r[t.MappingKV[str, U]].fail("; ".join(errors))
            if errors
            else r[t.MappingKV[str, U]].ok(values)
        )

    @staticmethod
    def validate_not_empty(
        val: t.Cli.CliValue | None, *, name: str = "field"
    ) -> p.Result[bool]:
        """Validate that a value is not empty."""
        if val is None:
            return r[bool].fail(
                c.Cli.VALIDATION_MSG_FIELD_CANNOT_BE_EMPTY.format(field_name=name)
            )
        if isinstance(val, str):
            stripped = val.strip()
            if not stripped:
                return r[bool].fail(
                    c.Cli.VALIDATION_MSG_FIELD_CANNOT_BE_EMPTY.format(field_name=name)
                )
        return r[bool].ok(True)

    @staticmethod
    def validate_format(format_type: str) -> p.Result[str]:
        """Validate one CLI output format."""
        fmt = format_type.lower()
        valid = FlextCliUtilitiesValidation.validate_not_empty(fmt, name="format")
        if valid.failure or fmt not in set(c.Cli.OUTPUT_FORMATS):
            return r[str].fail(
                c.Cli.ERR_INVALID_OUTPUT_FORMAT.format(format=format_type)
            )
        return r[str].ok(fmt)

    @staticmethod
    def format_validation_errors(
        exc: PydanticValidationError, *, command: str, model: str
    ) -> str:
        """Render pydantic errors as located command/model/field lines."""
        lines: list[str] = []
        for error in exc.errors():
            loc = ".".join(str(part) for part in error.get("loc", ()) if part != "body")
            field = loc or "<input>"
            message = error.get("msg", "invalid value")
            lines.append(
                c.Cli.ERR_CLI_DEFINITION_FIELD.format(
                    command=command, model=model, field=field, reason=message
                )
            )
        if lines:
            return "\n".join(lines)
        return c.Cli.ERR_CLI_DEFINITION_INVALID_MODEL.format(
            command=command, model=model, reason="validation failed"
        )

    @staticmethod
    def first_error_field(exc: PydanticValidationError) -> str | None:
        """Return the first pydantic error location as a dotted field name."""
        errors = exc.errors()
        if not errors:
            return None
        loc = ".".join(str(part) for part in errors[0].get("loc", ()) if part != "body")
        return loc or None

    @staticmethod
    def assert_model_definition(model_cls: object, *, command: str) -> None:
        """Raise a located definition error if ``model_cls`` is not pydantic."""
        model_name = getattr(model_cls, "__name__", repr(model_cls))
        model_fields = getattr(model_cls, "model_fields", None)
        if not isinstance(model_fields, dict) or not model_fields:
            raise c.Cli.CliDefinitionError(
                c.Cli.ERR_CLI_DEFINITION_INVALID_MODEL.format(
                    command=command,
                    model=model_name,
                    reason="missing pydantic model_fields",
                ),
                command=command,
                model=model_name,
            )


__all__: t.MutableSequenceOf[str] = ["FlextCliUtilitiesValidation"]
