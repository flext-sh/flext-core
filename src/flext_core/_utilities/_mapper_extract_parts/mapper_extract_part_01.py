"""Path-based extract pipeline on top of ``FlextUtilitiesMapperAccess``."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, TYPE_CHECKING

from flext_core import t, p, m, r, FlextRuntime
from flext_core._models.pydantic import FlextModelsPydantic
from flext_core._utilities.mapper_access import FlextUtilitiesMapperAccess

if TYPE_CHECKING:
    from flext_core._models.containers import FlextModelsContainers


class FlextUtilitiesMapperExtract(FlextUtilitiesMapperAccess):
    """Path-based ``extract`` method and its helpers."""

    class ExtractResolvePathPartContext(m.Value):
        """Validated context envelope for one path-part extraction step."""

        path_context: Annotated[
            str, m.Field(default="", description="Resolved parent path context")
        ]
        default: Annotated[
            t.JsonPayload | None,
            m.Field(default=None, description="Default fallback payload"),
        ]
        required: Annotated[
            bool, m.Field(default=False, description="Whether missing values are fatal")
        ]

    @staticmethod
    def _extract_fail_or_default(
        msg: str, *, default: t.JsonPayload | None, required: bool
    ) -> p.Result[t.JsonPayload]:
        """Return required failure, configured default, or missing-default failure."""
        if not required and default is not None:
            return r[t.JsonPayload].ok(default)
        return r[t.JsonPayload].fail_op("extract path", msg)

    @staticmethod
    def _extract_resolve_result(
        result: p.Result[t.JsonPayload],
        *,
        default: t.JsonPayload | None,
        required: bool,
    ) -> tuple[t.JsonPayload | None, p.Result[t.JsonPayload] | None]:
        """Resolve extractor step result into next value or early fallback result."""
        if result.failure:
            if not required and default is not None:
                return None, r[t.JsonPayload].ok(default)
            return None, r[t.JsonPayload].from_failure(result)
        return result.unwrap(), None

    @staticmethod
    def _extract_resolve_path_part(
        current: t.JsonPayload
        | t.JsonMapping
        | FlextModelsContainers.ConfigMap
        | t.SequenceOf[t.JsonPayload]
        | None,
        part: str,
        *,
        context: ExtractResolvePathPartContext,
    ) -> tuple[t.JsonPayload | None, p.Result[t.JsonPayload] | None]:
        """Resolve one path segment and return its cursor or an early result."""
        if "[" in part and part.endswith("]"):
            bracket_pos = part.index("[")
            array_match = part[bracket_pos + 1 : -1]
            key_part = part[:bracket_pos]
        else:
            key_part, array_match = part, ""

        get_result = FlextUtilitiesMapperExtract._extract_get_value(current, key_part)
        next_val, early_result = FlextUtilitiesMapperExtract._extract_resolve_result(
            get_result, default=context.default, required=context.required
        )
        if early_result is not None:
            return None, early_result

        if array_match and next_val is not None:
            narrowed_for_index = (
                next_val
                if isinstance(next_val, Sequence)
                and not isinstance(next_val, t.STR_BYTES_TYPES)
                else FlextRuntime.normalize_to_container(next_val)
            )
            index_result = FlextUtilitiesMapperExtract._extract_handle_array_index(
                narrowed_for_index, array_match
            )
            next_val, early_result = (
                FlextUtilitiesMapperExtract._extract_resolve_result(
                    index_result, default=context.default, required=context.required
                )
            )
            if early_result is not None:
                return None, early_result

        return next_val, None

    @staticmethod
    def _extract_seed_current(
        data: p.AccessibleData,
    ) -> t.JsonPayload | t.JsonMapping | FlextModelsContainers.ConfigMap | None:
        """Build the initial ``current`` cursor for path traversal."""
        seed_current: (
            t.JsonPayload | t.JsonMapping | FlextModelsContainers.ConfigMap | None
        ) = None
        if isinstance(data, FlextModelsPydantic.BaseModel):
            seed_current = data
        elif isinstance(data, Mapping):
            seed_current = m.ConfigMap(
                root={
                    k: FlextUtilitiesMapperExtract._normalize_accessible_value(v)
                    for k, v in data.items()
                }
            )
        else:
            model_dump_attr = getattr(data, "model_dump", None)
            if callable(model_dump_attr):
                seed_current = m.ConfigMap.model_validate(model_dump_attr())
            elif isinstance(data, p.ValidatorSpec):
                seed_current = str(data)
            elif data is None or isinstance(data, (*t.SCALAR_TYPES, Path, list, tuple)):
                seed_current = data
        return seed_current


__all__: list[str] = ["FlextUtilitiesMapperExtract"]
