"""Scalar and binary cached TypeAdapter factories for the FlextUtilities facade.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
)

from collections.abc import Mapping, Sequence
from enum import StrEnum
from functools import cache

from pydantic import ConfigDict
from typing_extensions import TypeForm

from flext_core import t
from flext_core._utilities import FlextUtilitiesBase


class FlextUtilitiesTypingAdapterScalars(FlextUtilitiesBase):
    """Scalar and binary cached TypeAdapter factories."""

    @classmethod
    @cache
    def bool_adapter(cls) -> t.TypeAdapter[bool]:
        """Return the cached adapter for ``bool``."""
        return cls.type_adapter(TypeForm(bool))

    @classmethod
    @cache
    def int_adapter(
        cls,
    ) -> t.TypeAdapter[t.StrictInt]:
        """Return the cached adapter for strict ``int``."""
        return cls.type_adapter(TypeForm(t.StrictInt))

    @classmethod
    @cache
    def scalar_adapter(cls) -> t.TypeAdapter[t.Scalar]:
        """Return the cached adapter for ``Scalar``."""
        return cls.type_adapter(TypeForm(t.Scalar))

    @classmethod
    @cache
    def scalar_mapping_adapter(
        cls,
    ) -> t.TypeAdapter[Mapping[str, t.Scalar]]:
        """Return the cached adapter for ``Mapping[str, Scalar]``."""
        return cls.type_adapter(TypeForm(Mapping[str, t.Scalar]))

    @classmethod
    @cache
    def float_adapter(
        cls,
    ) -> t.TypeAdapter[t.StrictFloat]:
        """Return the cached adapter for strict ``float``."""
        return cls.type_adapter(TypeForm(t.StrictFloat))

    @classmethod
    @cache
    def str_adapter(
        cls,
    ) -> t.TypeAdapter[t.StrictStr]:
        """Return the cached adapter for strict ``str``."""
        return cls.type_adapter(TypeForm(t.StrictStr))

    @classmethod
    @cache
    def binary_content_adapter(
        cls,
    ) -> t.TypeAdapter[t.StrictBytes]:
        """Return the cached adapter for strict ``bytes``."""
        return cls.type_adapter(TypeForm(t.StrictBytes))

    @classmethod
    @cache
    def str_mapping_adapter(
        cls,
    ) -> t.TypeAdapter[Mapping[str, str]]:
        """Return the cached adapter for ``Mapping[str, str]``."""
        return cls.type_adapter(TypeForm(Mapping[str, str]))

    @classmethod
    @cache
    def header_mapping_adapter(
        cls,
    ) -> t.TypeAdapter[t.HeaderMapping]:
        """Return the cached adapter for ``HeaderMapping``."""
        return cls.type_adapter(TypeForm(t.HeaderMapping))

    @classmethod
    @cache
    def str_dict_adapter(
        cls,
    ) -> t.TypeAdapter[dict[str, str]]:
        """Return the cached adapter for ``dict[str, str]``."""
        return cls.type_adapter(TypeForm(dict[str, str]))

    @classmethod
    @cache
    def int_dict_adapter(
        cls,
    ) -> t.TypeAdapter[t.IntDict]:
        """Return the cached adapter for ``IntDict``."""
        return cls.type_adapter(TypeForm(t.IntDict))

    @classmethod
    @cache
    def hostname_str_adapter(
        cls,
    ) -> t.TypeAdapter[t.HostnameStr]:
        """Return the cached adapter for ``HostnameStr``."""
        return cls.type_adapter(TypeForm(t.HostnameStr))

    @classmethod
    @cache
    def port_number_adapter(
        cls,
    ) -> t.TypeAdapter[t.PortNumber]:
        """Return the cached adapter for ``PortNumber``."""
        return cls.type_adapter(TypeForm(t.PortNumber))

    @classmethod
    @cache
    def str_sequence_adapter(
        cls,
    ) -> t.TypeAdapter[Sequence[str]]:
        """Return the cached adapter for ``Sequence[str]``."""
        return cls.type_adapter(TypeForm(Sequence[str]))

    @classmethod
    @cache
    def strict_str_sequence_adapter(
        cls,
    ) -> t.TypeAdapter[Sequence[str]]:
        """Return the cached strict-mode adapter for ``Sequence[str]``."""
        return cls.type_adapter(
            TypeForm(Sequence[str]),
            config=ConfigDict(strict=True),
        )

    @classmethod
    @cache
    def strict_str_tuple_adapter(
        cls,
    ) -> t.TypeAdapter[t.StrTuple]:
        """Return the cached strict-mode adapter for ``StrTuple``."""
        return cls.type_adapter(
            TypeForm(t.StrTuple),
            config=ConfigDict(strict=True),
        )

    @classmethod
    @cache
    def str_or_bytes_adapter(
        cls,
    ) -> t.TypeAdapter[t.TextOrBinaryContent]:
        """Return the cached adapter for text-or-binary content."""
        return cls.type_adapter(TypeForm(t.TextOrBinaryContent))

    @classmethod
    @cache
    def enum_type_adapter(cls) -> t.TypeAdapter[type[StrEnum]]:
        """Return the cached adapter for ``StrEnum`` types."""
        return cls.type_adapter(type[StrEnum])

    @classmethod
    @cache
    def primitive_metadata_mapping_adapter(
        cls,
    ) -> t.TypeAdapter[Mapping[str, t.Primitives]]:
        """Return the cached adapter for primitive-valued mappings."""
        return cls.type_adapter(Mapping[str, t.Primitives])


__all__: list[str] = ["FlextUtilitiesTypingAdapterScalars"]
