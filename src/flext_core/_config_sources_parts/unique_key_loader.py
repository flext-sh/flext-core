"""Unique-key safe YAML loader for the Flext config/settings layer.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import cast

from pydantic import JsonValue
from yaml import MappingNode, SafeLoader
from yaml.constructor import ConstructorError
from yaml.resolver import BaseResolver


class _UniqueKeySafeLoader(SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys at every depth."""


def _construct_unique_mapping(
    loader: SafeLoader,
    node: MappingNode,
    *,
    deep: bool = False,
) -> dict[str, JsonValue]:
    """Construct one JSON mapping and fail before a duplicate can overwrite.

    Args:
        loader: The YAML loader constructing the node.
        node: The mapping node being constructed.
        deep: Whether nested nodes are constructed deeply.

    Returns:
        The resulting ``dict[str, JsonValue]``.

    Raises:
        ConstructorError: If a key is not a string or declares a duplicate.
    """
    values: dict[str, JsonValue] = {}
    for key_node, value_node in node.value:
        key = cast("JsonValue", loader.construct_object(key_node, deep=deep))
        if not isinstance(key, str):
            context = "while constructing a config mapping"
            problem = "config mapping keys must be strings"
            raise ConstructorError(
                context,
                node.start_mark,
                problem,
                key_node.start_mark,
            )
        if key in values:
            context = "while constructing a config mapping"
            problem = f"duplicate config key: {key}"
            raise ConstructorError(
                context,
                node.start_mark,
                problem,
                key_node.start_mark,
            )
        values[key] = cast("JsonValue", loader.construct_object(value_node, deep=deep))
    return values


_UniqueKeySafeLoader.add_constructor(
    BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)

__all__: list[str] = ["_UniqueKeySafeLoader"]
