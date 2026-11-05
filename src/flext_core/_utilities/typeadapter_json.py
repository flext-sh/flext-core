"""JSON-shape cached TypeAdapter factories for the FlextUtilities facade.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import cache

from pydantic import ConfigDict

import flext_core
from flext_core._utilities import FlextUtilitiesBase


class FlextUtilitiesTypingAdapterJson(FlextUtilitiesBase):
    """JSON-shape cached TypeAdapter factories.

    Shared through the ``FlextUtilities`` facade.
    """

    @classmethod
    @cache
    def metadata_map_adapter(
        cls,
    ) -> flext_core.t.TypeAdapter[Mapping[str, flext_core.t.JsonValue]]:
        """Return the cached adapter for ``Mapping[str, JsonValue]``."""
        return cls.type_adapter(Mapping[str, flext_core.t.JsonValue])

    @classmethod
    @cache
    def json_value_adapter(
        cls,
    ) -> flext_core.t.TypeAdapter[flext_core.t.JsonValue]:
        """Return the cached adapter for ``JsonValue``."""
        return cls.type_adapter(flext_core.t.TypeForm(flext_core.t.JsonValue))

    @classmethod
    @cache
    def json_mapping_adapter(
        cls,
    ) -> flext_core.t.TypeAdapter[flext_core.t.JsonMapping]:
        """Return the cached adapter for ``JsonMapping``."""
        return cls.type_adapter(flext_core.t.TypeForm(flext_core.t.JsonMapping))

    @classmethod
    @cache
    def strict_json_mapping_adapter(
        cls,
    ) -> flext_core.t.TypeAdapter[flext_core.t.JsonMapping]:
        """Return the cached strict-mode adapter for ``JsonMapping``."""
        return cls.type_adapter(
            flext_core.t.TypeForm(flext_core.t.JsonMapping),
            config=ConfigDict(strict=True),
        )

    @classmethod
    @cache
    def json_dict_adapter(
        cls,
    ) -> flext_core.t.TypeAdapter[flext_core.t.JsonDict]:
        """Return the cached adapter for ``JsonDict``."""
        return cls.type_adapter(flext_core.t.TypeForm(flext_core.t.JsonDict))

    @classmethod
    @cache
    def json_dict_sequence_adapter(
        cls,
    ) -> flext_core.t.TypeAdapter[Sequence[flext_core.t.JsonDict]]:
        """Return the cached adapter for ``JsonDict`` sequences."""
        return cls.type_adapter(flext_core.t.TypeForm(Sequence[flext_core.t.JsonDict]))

    @classmethod
    @cache
    def json_mapping_sequence_adapter(
        cls,
    ) -> flext_core.t.TypeAdapter[Sequence[flext_core.t.JsonMapping]]:
        """Return the cached adapter for ``JsonMapping`` sequences."""
        return cls.type_adapter(
            flext_core.t.TypeForm(Sequence[flext_core.t.JsonMapping])
        )

    @classmethod
    @cache
    def json_mapping_by_str_adapter(
        cls,
    ) -> flext_core.t.TypeAdapter[Mapping[str, flext_core.t.JsonMapping]]:
        """Return the cached adapter for str-keyed ``JsonMapping`` mappings."""
        return cls.type_adapter(
            flext_core.t.TypeForm(Mapping[str, flext_core.t.JsonMapping])
        )

    @classmethod
    @cache
    def json_list_adapter(
        cls,
    ) -> flext_core.t.TypeAdapter[flext_core.t.JsonList]:
        """Return the cached adapter for ``JsonList``."""
        return cls.type_adapter(flext_core.t.TypeForm(flext_core.t.JsonList))

    @classmethod
    @cache
    def strict_json_list_adapter(
        cls,
    ) -> flext_core.t.TypeAdapter[flext_core.t.JsonList]:
        """Return the cached strict-mode adapter for ``JsonList``."""
        return cls.type_adapter(
            flext_core.t.TypeForm(flext_core.t.JsonList), config=ConfigDict(strict=True)
        )

    @classmethod
    @cache
    def primitives_adapter(
        cls,
    ) -> flext_core.t.TypeAdapter[flext_core.t.Primitives]:
        """Return the cached adapter for ``Primitives``."""
        return cls.type_adapter(flext_core.t.TypeForm(flext_core.t.Primitives))

    @classmethod
    @cache
    def container_set_adapter(
        cls,
    ) -> flext_core.t.TypeAdapter[set[flext_core.t.JsonValue]]:
        """Return the cached adapter for ``JsonValue`` sets."""
        return cls.type_adapter(set[flext_core.t.JsonValue])

    @classmethod
    @cache
    def string_set_adapter(cls) -> flext_core.t.TypeAdapter[set[str]]:
        """Return the cached adapter for ``str`` sets."""
        return cls.type_adapter(set[str])

    @classmethod
    @cache
    def scalar_set_adapter(
        cls,
    ) -> flext_core.t.TypeAdapter[set[flext_core.t.Scalar]]:
        """Return the cached adapter for ``Scalar`` sets."""
        return cls.type_adapter(flext_core.t.TypeForm(set[flext_core.t.Scalar]))

    @classmethod
    @cache
    def sortable_dict_adapter(
        cls,
    ) -> flext_core.t.TypeAdapter[
        Mapping[
            flext_core.t.SortableObjectType,
            flext_core.t.JsonValue | None,
        ]
    ]:
        """Return the cached adapter for sortable-object mappings."""
        return cls.type_adapter(
            flext_core.t.TypeForm(
                Mapping[
                    flext_core.t.SortableObjectType,
                    flext_core.t.JsonValue | None,
                ],
            )
        )


__all__: list[str] = ["FlextUtilitiesTypingAdapterJson"]
