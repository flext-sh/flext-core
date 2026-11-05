"""Scalar and binary cached TypeAdapter factories for the FlextUtilities facade.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from functools import cache

from pydantic import ConfigDict

import flext_core
from flext_core._utilities import FlextUtilitiesBase


class FlextUtilitiesTypingAdapterScalars(FlextUtilitiesBase):
    """Scalar and binary cached TypeAdapter factories."""

    @classmethod
    @cache
    def bool_adapter(cls) -> flext_core.t.TypeAdapter[bool]:
        """Return the cached adapter for ``bool``."""
        return cls.type_adapter(flext_core.t.TypeForm(bool))

    @classmethod
    @cache
    def int_adapter(
        cls,
    ) -> flext_core.t.TypeAdapter[flext_core.t.StrictInt]:
        """Return the cached adapter for strict ``int``."""
        return cls.type_adapter(flext_core.t.TypeForm(flext_core.t.StrictInt))

    @classmethod
    @cache
    def scalar_adapter(cls) -> flext_core.t.TypeAdapter[flext_core.t.Scalar]:
        """Return the cached adapter for ``Scalar``."""
        return cls.type_adapter(flext_core.t.TypeForm(flext_core.t.Scalar))

    @classmethod
    @cache
    def scalar_mapping_adapter(
        cls,
    ) -> flext_core.t.TypeAdapter[Mapping[str, flext_core.t.Scalar]]:
        """Return the cached adapter for ``Mapping[str, Scalar]``."""
        return cls.type_adapter(
            flext_core.t.TypeForm(Mapping[str, flext_core.t.Scalar])
        )

    @classmethod
    @cache
    def float_adapter(
        cls,
    ) -> flext_core.t.TypeAdapter[flext_core.t.StrictFloat]:
        """Return the cached adapter for strict ``float``."""
        return cls.type_adapter(flext_core.t.TypeForm(flext_core.t.StrictFloat))

    @classmethod
    @cache
    def str_adapter(
        cls,
    ) -> flext_core.t.TypeAdapter[flext_core.t.StrictStr]:
        """Return the cached adapter for strict ``str``."""
        return cls.type_adapter(flext_core.t.TypeForm(flext_core.t.StrictStr))

    @classmethod
    @cache
    def binary_content_adapter(
        cls,
    ) -> flext_core.t.TypeAdapter[flext_core.t.StrictBytes]:
        """Return the cached adapter for strict ``bytes``."""
        return cls.type_adapter(flext_core.t.TypeForm(flext_core.t.StrictBytes))

    @classmethod
    @cache
    def str_mapping_adapter(
        cls,
    ) -> flext_core.t.TypeAdapter[Mapping[str, str]]:
        """Return the cached adapter for ``Mapping[str, str]``."""
        return cls.type_adapter(flext_core.t.TypeForm(Mapping[str, str]))

    @classmethod
    @cache
    def header_mapping_adapter(
        cls,
    ) -> flext_core.t.TypeAdapter[flext_core.t.HeaderMapping]:
        """Return the cached adapter for ``HeaderMapping``."""
        return cls.type_adapter(flext_core.t.TypeForm(flext_core.t.HeaderMapping))

    @classmethod
    @cache
    def str_dict_adapter(
        cls,
    ) -> flext_core.t.TypeAdapter[dict[str, str]]:
        """Return the cached adapter for ``dict[str, str]``."""
        return cls.type_adapter(flext_core.t.TypeForm(dict[str, str]))

    @classmethod
    @cache
    def int_dict_adapter(
        cls,
    ) -> flext_core.t.TypeAdapter[dict[str, int]]:
        """Return the cached adapter for ``IntDict``."""
        return cls.type_adapter(flext_core.t.TypeForm(dict[str, int]))

    @classmethod
    @cache
    def hostname_str_adapter(
        cls,
    ) -> flext_core.t.TypeAdapter[flext_core.t.HostnameStr]:
        """Return the cached adapter for ``HostnameStr``."""
        return cls.type_adapter(flext_core.t.TypeForm(flext_core.t.HostnameStr))

    @classmethod
    @cache
    def port_number_adapter(
        cls,
    ) -> flext_core.t.TypeAdapter[flext_core.t.PortNumber]:
        """Return the cached adapter for ``PortNumber``."""
        return cls.type_adapter(flext_core.t.TypeForm(flext_core.t.PortNumber))

    @classmethod
    @cache
    def str_sequence_adapter(
        cls,
    ) -> flext_core.t.TypeAdapter[Sequence[str]]:
        """Return the cached adapter for ``Sequence[str]``."""
        return cls.type_adapter(flext_core.t.TypeForm(Sequence[str]))

    @classmethod
    @cache
    def strict_str_sequence_adapter(
        cls,
    ) -> flext_core.t.TypeAdapter[Sequence[str]]:
        """Return the cached strict-mode adapter for ``Sequence[str]``."""
        return cls.type_adapter(
            flext_core.t.TypeForm(Sequence[str]),
            config=ConfigDict(strict=True),
        )

    @classmethod
    @cache
    def strict_str_tuple_adapter(
        cls,
    ) -> flext_core.t.TypeAdapter[flext_core.t.StrTuple]:
        """Return the cached strict-mode adapter for ``StrTuple``."""
        return cls.type_adapter(
            flext_core.t.TypeForm(flext_core.t.StrTuple),
            config=ConfigDict(strict=True),
        )

    @classmethod
    @cache
    def str_or_bytes_adapter(
        cls,
    ) -> flext_core.t.TypeAdapter[flext_core.t.TextOrBinaryContent]:
        """Return the cached adapter for text-or-binary content."""
        return cls.type_adapter(flext_core.t.TypeForm(flext_core.t.TextOrBinaryContent))

    @classmethod
    @cache
    def enum_type_adapter(cls) -> flext_core.t.TypeAdapter[type[StrEnum]]:
        """Return the cached adapter for ``StrEnum`` types."""
        return cls.type_adapter(type[StrEnum])

    @classmethod
    @cache
    def primitive_metadata_mapping_adapter(
        cls,
    ) -> flext_core.t.TypeAdapter[Mapping[str, flext_core.t.Primitives]]:
        """Return the cached adapter for primitive-valued mappings."""
        return cls.type_adapter(Mapping[str, flext_core.t.Primitives])


__all__: list[str] = ["FlextUtilitiesTypingAdapterScalars"]
