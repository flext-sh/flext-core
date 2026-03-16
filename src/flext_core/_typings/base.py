from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from pathlib import Path


class FlextTypingBase:
    type Primitives = str | int | float | bool
    type Scalar = str | int | float | bool | datetime
    type Container = Scalar | Path

    PRIMITIVES_TYPES: tuple[type[str], type[int], type[float], type[bool]] = (
        str,
        int,
        float,
        bool,
    )
    SCALAR_TYPES: tuple[
        type[str], type[int], type[float], type[bool], type[datetime]
    ] = (str, int, float, bool, datetime)
    CONTAINER_TYPES: tuple[
        type[str], type[int], type[float], type[bool], type[datetime], type[Path]
    ] = (str, int, float, bool, datetime, Path)

    type NormalizedValue = (
        Container
        | list[FlextTypingBase.NormalizedValue]
        | Mapping[str, FlextTypingBase.NormalizedValue]
        | tuple[FlextTypingBase.NormalizedValue, ...]
        | None
    )
    type ContainerMapping = Mapping[str, FlextTypingBase.NormalizedValue]
    type ContainerList = list[FlextTypingBase.NormalizedValue]
