from __future__ import annotations

from collections.abc import Mapping, Sequence

from flext_core._utilities._guards_type_protocol_types import ProtocolGuardInput


class FlextUtilitiesGuardsTypeProtocolStringMixin:
    @staticmethod
    def _run_string_type_check(type_name: str, value: ProtocolGuardInput) -> bool:
        match type_name:
            case "str":
                return isinstance(value, str)
            case "dict":
                return isinstance(value, dict)
            case "list":
                return isinstance(value, list)
            case "tuple":
                return isinstance(value, tuple)
            case "sequence":
                return isinstance(value, (list, tuple, range))
            case "mapping":
                return isinstance(value, Mapping)
            case "list_or_tuple":
                return isinstance(value, (list, tuple))
            case "sequence_not_str":
                return isinstance(value, (list, tuple, range)) and not isinstance(
                    value,
                    str,
                )
            case "sequence_not_str_bytes":
                return isinstance(value, (list, tuple, range)) and not isinstance(
                    value,
                    (str, bytes),
                )
            case "sized":
                return hasattr(value, "__len__")
            case "callable":
                return callable(value)
            case "bytes":
                return isinstance(value, bytes)
            case "int":
                return isinstance(value, int)
            case "float":
                return isinstance(value, float)
            case "bool":
                return isinstance(value, bool)
            case "none":
                return value is None
            case "string_non_empty":
                return isinstance(value, str) and bool(value.strip())
            case "dict_non_empty":
                return isinstance(value, Mapping) and len(value) > 0
            case "list_non_empty":
                return (
                    isinstance(value, Sequence)
                    and not isinstance(value, (str, bytes, bytearray))
                    and len(value) > 0
                )
            case _:
                return False


__all__: list[str] = ["FlextUtilitiesGuardsTypeProtocolStringMixin"]
