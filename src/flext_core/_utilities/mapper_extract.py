"""Path-based extract pipeline on top of ``FlextUtilitiesMapperAccess``."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from flext_core import (
    FlextModelsContainers,
    FlextModelsExceptionParams,
    FlextModelsPydantic,
    FlextRuntime,
    FlextUtilitiesGuards,
    FlextUtilitiesMapperAccess,
    c,
    e,
    m,
    p,
    r,
    t,
)


class FlextUtilitiesMapperExtract(FlextUtilitiesMapperAccess):
    """Path-based ``extract`` method and its helpers."""

    @staticmethod
    def _extract_fail_or_default(
        msg: str,
        *,
        default: t.JsonPayload | None,
        required: bool,
    ) -> p.Result[t.JsonPayload]:
        """Return fail (required) or ok(default) / fail (no default) for extract paths."""
        if required:
            return r[t.JsonPayload].fail_op("extract required path", msg)
        if default is None:
            return r[t.JsonPayload].fail_op(
                "extract path default",
                e.render_template(
                    c.ERR_TEMPLATE_MESSAGE_AND_DEFAULT_IS_NONE,
                    message=msg,
                ),
            )
        return r[t.JsonPayload].ok(default)

    @staticmethod
    def _extract_resolve_path_part(
        current: t.JsonPayload
        | t.JsonMapping
        | FlextModelsContainers.ConfigMap
        | Sequence[t.JsonPayload]
        | None,
        part: str,
        *,
        path_context: str,
        default: t.JsonPayload | None,
        required: bool,
    ) -> tuple[t.JsonPayload | None, p.Result[t.JsonPayload] | None]:
        """Resolve one path segment; returns (next_current, None) or (None, early_result)."""
        if "[" in part and part.endswith("]"):
            bracket_pos = part.index("[")
            array_match = part[bracket_pos + 1 : -1]
            key_part = part[:bracket_pos]
        else:
            key_part, array_match = part, ""

        get_result = FlextUtilitiesMapperExtract._extract_get_value(current, key_part)
        next_val: t.JsonPayload | None
        if get_result.failure:
            if "found_none:" in (get_result.error or ""):
                next_val = None
            else:
                return None, FlextUtilitiesMapperExtract._extract_fail_or_default(
                    e.render_template(
                        c.ERR_TEMPLATE_KEY_NOT_FOUND_AT_PATH,
                        key=key_part,
                        path=path_context,
                    ),
                    default=default,
                    required=required,
                )
        else:
            next_val = get_result.unwrap_or(None)

        if array_match and next_val is not None:
            narrowed_for_index = (
                next_val
                if isinstance(next_val, Sequence)
                and not isinstance(next_val, (str, bytes))
                else FlextRuntime.normalize_to_container(next_val)
            )
            index_result = FlextUtilitiesMapperExtract._extract_handle_array_index(
                narrowed_for_index,
                array_match,
            )
            if index_result.failure:
                if "found_none:" in (index_result.error or ""):
                    next_val = None
                else:
                    return None, FlextUtilitiesMapperExtract._extract_fail_or_default(
                        e.render_template(
                            c.ERR_TEMPLATE_ARRAY_ERROR_AT_KEY,
                            key=key_part,
                            error=index_result.error,
                        ),
                        default=default,
                        required=required,
                    )
            else:
                next_val = index_result.unwrap_or(None)

        return next_val, None

    @staticmethod
    def _extract_seed_current(
        data: p.AccessibleData,
    ) -> t.JsonPayload | t.JsonMapping | FlextModelsContainers.ConfigMap | None:
        """Build the initial ``current`` cursor for path traversal."""
        if isinstance(data, FlextModelsPydantic.BaseModel):
            return data
        if isinstance(data, Mapping):
            return m.ConfigMap(
                root={
                    str(k): FlextUtilitiesMapperExtract._normalize_accessible_value(v)
                    for k, v in data.items()
                },
            )
        model_dump_attr = getattr(data, "model_dump", None)
        if callable(model_dump_attr):
            return m.ConfigMap.model_validate(model_dump_attr())
        if isinstance(data, p.ValidatorSpec):
            return str(data)
        if data is None or isinstance(data, (*t.SCALAR_TYPES, Path, list, tuple)):
            return data
        return None

    @staticmethod
    def extract(
        data: p.AccessibleData,
        path: str,
        *,
        default: t.JsonPayload | None = None,
        required: bool = False,
        separator: str = ".",
    ) -> p.Result[t.JsonPayload]:
        """Extract nested value via dot-notation path with array index support."""
        try:
            parts = path.split(separator)
            current = FlextUtilitiesMapperExtract._extract_seed_current(data)
            for i, part in enumerate(parts):
                if current is None:
                    return FlextUtilitiesMapperExtract._extract_fail_or_default(
                        e.render_template(
                            c.ERR_TEMPLATE_PATH_IS_NONE,
                            path=separator.join(parts[:i]),
                        ),
                        default=default,
                        required=required,
                    )
                current, early_return = (
                    FlextUtilitiesMapperExtract._extract_resolve_path_part(
                        current,
                        part,
                        path_context=separator.join(parts[:i]),
                        default=default,
                        required=required,
                    )
                )
                if early_return is not None:
                    return early_return

            if current is None:
                return FlextUtilitiesMapperExtract._extract_fail_or_default(
                    c.ERR_TEMPLATE_EXTRACTED_VALUE_IS_NONE,
                    default=default,
                    required=required,
                )
            return r[t.JsonPayload].ok(
                current if FlextUtilitiesGuards.container(current) else str(current),
            )
        except (AttributeError, TypeError, ValueError, KeyError, IndexError) as exc:
            return r[t.JsonPayload].fail_op(
                "extract path",
                e.render_error_template(
                    c.ERR_TEMPLATE_EXTRACT_FAILED,
                    operation="extract",
                    error=exc,
                    params=FlextModelsExceptionParams.OperationErrorParams(
                        operation="extract",
                        reason=str(exc),
                    ),
                ),
            )


__all__: list[str] = ["FlextUtilitiesMapperExtract"]
