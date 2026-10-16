"""Scalar and binary cached TypeAdapter factories for the FlextUtilities facade.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from functools import cache

from pydantic import ConfigDict
from typing_extensions import TypeForm

from flext_core._typings import (
    FlextTypesAnnotateds,
    FlextTypesCore,
    FlextTypesPydantic,
    FlextTypingBase,
)
from flext_core._utilities import FlextUtilitiesBase


class FlextUtilitiesTypingAdapterScalars(FlextUtilitiesBase):
    """Scalar and binary cached TypeAdapter factories."""

    @classmethod
    @cache
    def bool_adapter(cls) -> FlextTypesPydantic.TypeAdapter[bool]:
        """Return the cached adapter for ``bool``."""
        return cls.type_adapter(TypeForm(bool))

    @classmethod
    @cache
    def int_adapter(
        cls,
    ) -> FlextTypesPydantic.TypeAdapter[FlextTypesPydantic.StrictInt]:
        """Return the cached adapter for strict ``int``."""
        return cls.type_adapter(TypeForm(FlextTypesPydantic.StrictInt))

    @classmethod
    @cache
    def scalar_adapter(cls) -> FlextTypesPydantic.TypeAdapter[FlextTypingBase.Scalar]:
        """Return the cached adapter for ``Scalar``."""
        return cls.type_adapter(TypeForm(FlextTypingBase.Scalar))

    @classmethod
    @cache
    def scalar_mapping_adapter(
        cls,
    ) -> FlextTypesPydantic.TypeAdapter[FlextTypingBase.ScalarMapping]:
        """Return the cached adapter for ``ScalarMapping``."""
        return cls.type_adapter(TypeForm(FlextTypingBase.ScalarMapping))

    @classmethod
    @cache
    def float_adapter(
        cls,
    ) -> FlextTypesPydantic.TypeAdapter[FlextTypesPydantic.StrictFloat]:
        """Return the cached adapter for strict ``float``."""
        return cls.type_adapter(TypeForm(FlextTypesPydantic.StrictFloat))

    @classmethod
    @cache
    def str_adapter(
        cls,
    ) -> FlextTypesPydantic.TypeAdapter[FlextTypesPydantic.StrictStr]:
        """Return the cached adapter for strict ``str``."""
        return cls.type_adapter(TypeForm(FlextTypesPydantic.StrictStr))

    @classmethod
    @cache
    def binary_content_adapter(
        cls,
    ) -> FlextTypesPydantic.TypeAdapter[FlextTypesPydantic.StrictBytes]:
        """Return the cached adapter for strict ``bytes``."""
        return cls.type_adapter(TypeForm(FlextTypesPydantic.StrictBytes))

    @classmethod
    @cache
    def str_mapping_adapter(
        cls,
    ) -> FlextTypesPydantic.TypeAdapter[FlextTypingBase.StrMapping]:
        """Return the cached adapter for ``StrMapping``."""
        return cls.type_adapter(TypeForm(FlextTypingBase.StrMapping))

    @classmethod
    @cache
    def header_mapping_adapter(
        cls,
    ) -> FlextTypesPydantic.TypeAdapter[FlextTypingBase.HeaderMapping]:
        """Return the cached adapter for ``HeaderMapping``."""
        return cls.type_adapter(TypeForm(FlextTypingBase.HeaderMapping))

    @classmethod
    @cache
    def str_dict_adapter(
        cls,
    ) -> FlextTypesPydantic.TypeAdapter[FlextTypingBase.StrDict]:
        """Return the cached adapter for ``StrDict``."""
        return cls.type_adapter(TypeForm(FlextTypingBase.StrDict))

    @classmethod
    @cache
    def int_dict_adapter(
        cls,
    ) -> FlextTypesPydantic.TypeAdapter[FlextTypingBase.IntDict]:
        """Return the cached adapter for ``IntDict``."""
        return cls.type_adapter(TypeForm(FlextTypingBase.IntDict))

    @classmethod
    @cache
    def hostname_str_adapter(
        cls,
    ) -> FlextTypesPydantic.TypeAdapter[FlextTypesAnnotateds.HostnameStr]:
        """Return the cached adapter for ``HostnameStr``."""
        return cls.type_adapter(TypeForm(FlextTypesAnnotateds.HostnameStr))

    @classmethod
    @cache
    def port_number_adapter(
        cls,
    ) -> FlextTypesPydantic.TypeAdapter[FlextTypesAnnotateds.PortNumber]:
        """Return the cached adapter for ``PortNumber``."""
        return cls.type_adapter(TypeForm(FlextTypesAnnotateds.PortNumber))

    @classmethod
    @cache
    def str_sequence_adapter(
        cls,
    ) -> FlextTypesPydantic.TypeAdapter[FlextTypingBase.StrSequence]:
        """Return the cached adapter for ``StrSequence``."""
        return cls.type_adapter(TypeForm(FlextTypingBase.StrSequence))

    @classmethod
    @cache
    def strict_str_sequence_adapter(
        cls,
    ) -> FlextTypesPydantic.TypeAdapter[FlextTypingBase.StrSequence]:
        """Return the cached strict-mode adapter for ``StrSequence``."""
        return cls.type_adapter(
            TypeForm(FlextTypingBase.StrSequence),
            config=ConfigDict(strict=True),
        )

    @classmethod
    @cache
    def str_or_bytes_adapter(
        cls,
    ) -> FlextTypesPydantic.TypeAdapter[FlextTypesCore.TextOrBinaryContent]:
        """Return the cached adapter for text-or-binary content."""
        return cls.type_adapter(TypeForm(FlextTypesCore.TextOrBinaryContent))

    @classmethod
    @cache
    def enum_type_adapter(cls) -> FlextTypesPydantic.TypeAdapter[type[StrEnum]]:
        """Return the cached adapter for ``StrEnum`` types."""
        return cls.type_adapter(type[StrEnum])

    @classmethod
    @cache
    def primitive_metadata_mapping_adapter(
        cls,
    ) -> FlextTypesPydantic.TypeAdapter[Mapping[str, FlextTypingBase.Primitives]]:
        """Return the cached adapter for primitive-valued mappings."""
        return cls.type_adapter(Mapping[str, FlextTypingBase.Primitives])


__all__: list[str] = ["FlextUtilitiesTypingAdapterScalars"]
