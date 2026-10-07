"""FlextTypesTypeAdapterScalars part.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from enum import StrEnum
from functools import cache

from pydantic import ConfigDict, TypeAdapter

from flext_core._typings.annotateds import FlextTypesAnnotateds
from flext_core._typings.base import FlextTypingBase
from flext_core._typings.core import FlextTypesCore
from flext_core._typings.pydantic import FlextTypesPydantic


class FlextTypesTypeAdapterScalars:
    """Scalar and binary cached TypeAdapter factories."""

    @classmethod
    @cache
    def bool_adapter(cls) -> FlextTypesPydantic.TypeAdapter[bool]:
        return TypeAdapter(bool)

    @classmethod
    @cache
    def int_adapter(
        cls,
    ) -> FlextTypesPydantic.TypeAdapter[FlextTypesPydantic.StrictInt]:
        return TypeAdapter(FlextTypesPydantic.StrictInt)

    @classmethod
    @cache
    def scalar_adapter(cls) -> FlextTypesPydantic.TypeAdapter[FlextTypingBase.Scalar]:
        return TypeAdapter(FlextTypingBase.Scalar)

    @classmethod
    @cache
    def scalar_mapping_adapter(
        cls,
    ) -> FlextTypesPydantic.TypeAdapter[FlextTypingBase.ScalarMapping]:
        return TypeAdapter(FlextTypingBase.ScalarMapping)

    @classmethod
    @cache
    def float_adapter(
        cls,
    ) -> FlextTypesPydantic.TypeAdapter[FlextTypesPydantic.StrictFloat]:
        return TypeAdapter(FlextTypesPydantic.StrictFloat)

    @classmethod
    @cache
    def str_adapter(
        cls,
    ) -> FlextTypesPydantic.TypeAdapter[FlextTypesPydantic.StrictStr]:
        return TypeAdapter(FlextTypesPydantic.StrictStr)

    @classmethod
    @cache
    def binary_content_adapter(
        cls,
    ) -> FlextTypesPydantic.TypeAdapter[FlextTypesPydantic.StrictBytes]:
        return TypeAdapter(FlextTypesPydantic.StrictBytes)

    @classmethod
    @cache
    def str_mapping_adapter(
        cls,
    ) -> FlextTypesPydantic.TypeAdapter[FlextTypingBase.StrMapping]:
        return TypeAdapter(FlextTypingBase.StrMapping)

    @classmethod
    @cache
    def header_mapping_adapter(
        cls,
    ) -> FlextTypesPydantic.TypeAdapter[FlextTypingBase.HeaderMapping]:
        return TypeAdapter(FlextTypingBase.HeaderMapping)

    @classmethod
    @cache
    def str_dict_adapter(
        cls,
    ) -> FlextTypesPydantic.TypeAdapter[FlextTypingBase.StrDict]:
        return TypeAdapter(FlextTypingBase.StrDict)

    @classmethod
    @cache
    def int_dict_adapter(
        cls,
    ) -> FlextTypesPydantic.TypeAdapter[FlextTypingBase.IntDict]:
        return TypeAdapter(FlextTypingBase.IntDict)

    @classmethod
    @cache
    def hostname_str_adapter(
        cls,
    ) -> FlextTypesPydantic.TypeAdapter[FlextTypesAnnotateds.HostnameStr]:
        return TypeAdapter(FlextTypesAnnotateds.HostnameStr)

    @classmethod
    @cache
    def port_number_adapter(
        cls,
    ) -> FlextTypesPydantic.TypeAdapter[FlextTypesAnnotateds.PortNumber]:
        return TypeAdapter(FlextTypesAnnotateds.PortNumber)

    @classmethod
    @cache
    def str_sequence_adapter(
        cls,
    ) -> FlextTypesPydantic.TypeAdapter[FlextTypingBase.StrSequence]:
        return TypeAdapter(FlextTypingBase.StrSequence)

    @classmethod
    @cache
    def strict_str_sequence_adapter(
        cls,
    ) -> FlextTypesPydantic.TypeAdapter[FlextTypingBase.StrSequence]:
        return TypeAdapter(FlextTypingBase.StrSequence, config=ConfigDict(strict=True))

    @classmethod
    @cache
    def str_or_bytes_adapter(
        cls,
    ) -> FlextTypesPydantic.TypeAdapter[FlextTypesCore.TextOrBinaryContent]:
        return TypeAdapter(FlextTypesCore.TextOrBinaryContent)

    @classmethod
    @cache
    def enum_type_adapter(cls) -> FlextTypesPydantic.TypeAdapter[type[StrEnum]]:
        return TypeAdapter(type[StrEnum])

    @classmethod
    @cache
    def primitive_metadata_mapping_adapter(
        cls,
    ) -> FlextTypesPydantic.TypeAdapter[Mapping[str, FlextTypingBase.Primitives]]:
        return TypeAdapter(Mapping[str, FlextTypingBase.Primitives])

    @classmethod
    @cache
    def structlog_processor_adapter(
        cls,
    ) -> FlextTypesPydantic.TypeAdapter[Callable[..., FlextTypesPydantic.JsonValue]]:
        return TypeAdapter(Callable[..., FlextTypesPydantic.JsonValue])
