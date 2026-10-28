"""JSON-shape cached TypeAdapter factories for the FlextUtilities facade.

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
from functools import cache

from pydantic import ConfigDict
from typing_extensions import TypeForm

from flext_core import t
from flext_core._utilities import FlextUtilitiesBase


class FlextUtilitiesTypingAdapterJson(FlextUtilitiesBase):
    """JSON-shape cached TypeAdapter factories.

    Shared through the ``FlextUtilities`` facade.
    """

    @classmethod
    @cache
    def metadata_map_adapter(
        cls,
    ) -> t.TypeAdapter[Mapping[str, t.JsonValue]]:
        """Return the cached adapter for ``Mapping[str, JsonValue]``."""
        return cls.type_adapter(Mapping[str, t.JsonValue])

    @classmethod
    @cache
    def json_value_adapter(
        cls,
    ) -> t.TypeAdapter[t.JsonValue]:
        """Return the cached adapter for ``JsonValue``."""
        return cls.type_adapter(TypeForm(t.JsonValue))

    @classmethod
    @cache
    def json_mapping_adapter(
        cls,
    ) -> t.TypeAdapter[t.JsonMapping]:
        """Return the cached adapter for ``JsonMapping``."""
        return cls.type_adapter(TypeForm(t.JsonMapping))

    @classmethod
    @cache
    def strict_json_mapping_adapter(
        cls,
    ) -> t.TypeAdapter[t.JsonMapping]:
        """Return the cached strict-mode adapter for ``JsonMapping``."""
        return cls.type_adapter(
            TypeForm(t.JsonMapping),
            config=ConfigDict(strict=True),
        )

    @classmethod
    @cache
    def json_dict_adapter(
        cls,
    ) -> t.TypeAdapter[t.JsonDict]:
        """Return the cached adapter for ``JsonDict``."""
        return cls.type_adapter(TypeForm(t.JsonDict))

    @classmethod
    @cache
    def json_dict_sequence_adapter(
        cls,
    ) -> t.TypeAdapter[Sequence[t.JsonDict]]:
        """Return the cached adapter for ``JsonDict`` sequences."""
        return cls.type_adapter(TypeForm(Sequence[t.JsonDict]))

    @classmethod
    @cache
    def json_mapping_sequence_adapter(
        cls,
    ) -> t.TypeAdapter[Sequence[t.JsonMapping]]:
        """Return the cached adapter for ``JsonMapping`` sequences."""
        return cls.type_adapter(TypeForm(Sequence[t.JsonMapping]))

    @classmethod
    @cache
    def json_mapping_by_str_adapter(
        cls,
    ) -> t.TypeAdapter[Mapping[str, t.JsonMapping]]:
        """Return the cached adapter for str-keyed ``JsonMapping`` mappings."""
        return cls.type_adapter(TypeForm(Mapping[str, t.JsonMapping]))

    @classmethod
    @cache
    def json_list_adapter(
        cls,
    ) -> t.TypeAdapter[t.JsonList]:
        """Return the cached adapter for ``JsonList``."""
        return cls.type_adapter(TypeForm(t.JsonList))

    @classmethod
    @cache
    def strict_json_list_adapter(
        cls,
    ) -> t.TypeAdapter[t.JsonList]:
        """Return the cached strict-mode adapter for ``JsonList``."""
        return cls.type_adapter(TypeForm(t.JsonList), config=ConfigDict(strict=True))

    @classmethod
    @cache
    def primitives_adapter(
        cls,
    ) -> t.TypeAdapter[t.Primitives]:
        """Return the cached adapter for ``Primitives``."""
        return cls.type_adapter(TypeForm(t.Primitives))

    @classmethod
    @cache
    def container_set_adapter(
        cls,
    ) -> t.TypeAdapter[set[t.JsonValue]]:
        """Return the cached adapter for ``JsonValue`` sets."""
        return cls.type_adapter(set[t.JsonValue])

    @classmethod
    @cache
    def string_set_adapter(cls) -> t.TypeAdapter[set[str]]:
        """Return the cached adapter for ``str`` sets."""
        return cls.type_adapter(set[str])

    @classmethod
    @cache
    def scalar_set_adapter(
        cls,
    ) -> t.TypeAdapter[set[t.Scalar]]:
        """Return the cached adapter for ``Scalar`` sets."""
        return cls.type_adapter(TypeForm(set[t.Scalar]))

    @classmethod
    @cache
    def sortable_dict_adapter(
        cls,
    ) -> t.TypeAdapter[
        Mapping[
            t.SortableObjectType,
            t.JsonValue | None,
        ]
    ]:
        """Return the cached adapter for sortable-object mappings."""
        return cls.type_adapter(
            TypeForm(
                Mapping[
                    t.SortableObjectType,
                    t.JsonValue | None,
                ],
            )
        )


__all__: list[str] = ["FlextUtilitiesTypingAdapterJson"]
