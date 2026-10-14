"""Strict YAML config sources for the Flext config/settings layer.

Public home of ``FlextStrictYamlConfigSource`` (ADR-018: root-facade exports of
private root modules must follow the module suffix contract; a source class
has no ``Config``/``Settings`` suffix, so it lives in a public owner module).
The strict loader is an implementation detail of this foundational source,
so configuration ingress does not depend on generated private-package exports.
Its native JSON annotations keep this layer-zero bridge independent of its own
facades, whose construction consumes the configuration it loads.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Sequence, Sequence as _Sequence
from importlib.resources.abc import Traversable
from pathlib import Path, Path as _Path
from typing import IO, Protocol, cast, override

from pydantic import JsonValue, TypeAdapter
from pydantic_settings import BaseSettings, YamlConfigSettingsSource
from pydantic_settings.sources import ConfigFileSourceType, PathType
from pydantic_settings.sources.utils import InitState
from yaml import MappingNode, SafeLoader
from yaml.constructor import ConstructorError
from yaml.nodes import Node
from yaml.resolver import BaseResolver


class FlextStrictYamlConfigSource(YamlConfigSettingsSource):
    """Pydantic settings source backed by the unique-key safe loader.

    Accepts an optional ``transform`` callable applied to the fully merged
    YAML data before pydantic validates it. Subclasses of ``FlextConfig``
    use this to apply env-expansion, filtering, or section reshaping without
    reinstantiating the source or overriding ``settings_customise_sources``.
    """

    class _Loader(Protocol):
        """Typed view of the YAML loader's object-construction boundary."""

        def construct_object(
            self,
            node: Node,
            *,
            deep: bool = False,
        ) -> JsonValue: ...

    class UniqueKeyLoader(SafeLoader):
        """Safe YAML loader rejecting duplicate and non-string mapping keys."""

        @staticmethod
        def construct_unique_mapping(
            loader: SafeLoader,
            node: MappingNode,
            *,
            deep: bool = False,
        ) -> dict[str, JsonValue]:
            """Construct one mapping without silently replacing a declared key.

            Merge keys flatten first (SafeLoader semantics): anchored pairs
            enter as defaults, an explicit pair may override one, and a
            duplicated explicit key stays a loud parse error.

            Returns:
                The parsed JSON-compatible mapping.

            Raises:
                ConstructorError: If a key is not a string or is duplicated.

            """
            typed_loader = cast("FlextStrictYamlConfigSource._Loader", loader)
            declared: set[str] = set()
            for key_node, _value_node in node.value:
                if key_node.tag == "tag:yaml.org,2002:merge":
                    continue
                key = typed_loader.construct_object(key_node, deep=deep)
                if not isinstance(key, str):
                    msg = "while constructing a config mapping"
                    raise ConstructorError(
                        msg,
                        node.start_mark,
                        "config mapping keys must be strings",
                        key_node.start_mark,
                    )
                if key in declared:
                    msg = "while constructing a config mapping"
                    raise ConstructorError(
                        msg,
                        node.start_mark,
                        f"duplicate config key: {key}",
                        key_node.start_mark,
                    )
                declared.add(key)
            loader.flatten_mapping(node)
            values: dict[str, JsonValue] = {}
            for key_node, value_node in node.value:
                key = typed_loader.construct_object(key_node, deep=deep)
                if not isinstance(key, str):
                    msg = "while constructing a config mapping"
                    raise ConstructorError(
                        msg,
                        node.start_mark,
                        "config mapping keys must be strings",
                        key_node.start_mark,
                    )
                values[key] = typed_loader.construct_object(value_node, deep=deep)
            return values

    UniqueKeyLoader.add_constructor(
        BaseResolver.DEFAULT_MAPPING_TAG,
        UniqueKeyLoader.construct_unique_mapping,
    )

    @override
    def __init__(
        self,
        settings_cls: type[BaseSettings],
        yaml_file: ConfigFileSourceType | None = None,
        yaml_file_encoding: str | None = None,
        yaml_config_section: str | None = None,
        deep_merge: bool = False,
        _init_state: InitState | None = None,
        *,
        transform: Callable[[dict[str, JsonValue]], dict[str, JsonValue]] | None = None,
    ) -> None:
        """Initialize the YAML source with an optional deep merge and transform hook.

        Args:
            settings_cls: The pydantic settings class this source serves.
            yaml_file: Optional YAML file selector forwarded to the base source.
            yaml_file_encoding: Optional file encoding forwarded to the base
                source.
            yaml_config_section: Optional settings section selector forwarded to
                the base source.
            deep_merge: Whether multiple files deep-merge instead of last-wins.
            _init_state: Optional pydantic-settings initialization state.
            transform: Optional callable applied to the fully merged YAML data
                before pydantic validates it.

        """
        self._transform = transform
        super().__init__(
            settings_cls,
            yaml_file=yaml_file,
            yaml_file_encoding=yaml_file_encoding,
            yaml_config_section=yaml_config_section,
            deep_merge=deep_merge,
            _init_state=_init_state,
        )

    @override
    def __call__(self) -> dict[str, JsonValue]:
        """Return merged YAML data, applying the transform hook if set."""
        data = super().__call__()
        if self._transform is not None:
            data = self._transform(data)
        return data

    @staticmethod
    def unique_key_load(stream: str | IO[str]) -> JsonValue:
        """Parse safe YAML with unique keys and validate the JSON value boundary.

        Args:
            stream: The YAML text or text stream to parse.

        Returns:
            The parsed JSON-compatible value.

        """
        loader = FlextStrictYamlConfigSource.UniqueKeyLoader(stream)
        try:
            # Validate after YAML finishes constructing aliases and nested values.
            return TypeAdapter(JsonValue).validate_python(
                loader.get_single_data(),
                strict=True,
            )
        finally:
            loader.dispose()

    @override
    def _read_file(self, file_path: Path | Traversable) -> dict[str, JsonValue]:
        """Parse one YAML config file exactly once with strict mapping keys.

        Returns:
            The parsed native JSON mapping.

        Raises:
            TypeError: If the config YAML root is not a mapping.

        """
        with file_path.open(encoding=self.yaml_file_encoding) as yaml_file:
            loaded = self.unique_key_load(yaml_file)
        if loaded is None:
            return {}
        if not isinstance(loaded, dict):
            msg = f"config YAML root must be a mapping: {file_path}"
            raise TypeError(msg)
        return loaded

    @override
    def _read_files(
        self,
        files: PathType | Traversable | Sequence[PathType | Traversable] | None,
        deep_merge: bool = False,
    ) -> dict[str, JsonValue]:
        """Read multiple YAML files with list-aware deep merge.

        The upstream ``deep_update`` only recurses into dicts; when two files
        declare the same list key (e.g. ``command_rules``), the second file
        *replaces* the first list entirely. This override concatenates lists
        instead, enabling domain-split config files to each contribute rules
        to the same list.

        Returns:
            The merged native JSON mapping.

        """
        if files is None:
            return {}
        if isinstance(files, str) or not isinstance(files, _Sequence):
            files = [files]
        merged: dict[str, JsonValue] = {}
        for file in files:
            raw_path = _Path(file) if isinstance(file, str) else file
            if not isinstance(raw_path, _Path):
                continue
            file_path = raw_path.expanduser()
            if not file_path.is_file():
                continue
            updating: dict[str, JsonValue] = dict(self._read_file(file_path))
            if deep_merge:
                merged = self.merge_values(merged, updating)
            else:
                merged.update(updating)
        return merged

    @staticmethod
    def merge_values(
        base: dict[str, JsonValue],
        updating: dict[str, JsonValue],
    ) -> dict[str, JsonValue]:
        """Deep-merge two config dicts, concatenating list values.

        Args:
            base: The accumulated mapping so far.
            updating: The newly read mapping merged over ``base``.

        Returns:
            The resulting ``FlextTypingBase.JsonDict``.

        """
        result = dict(base)
        for key, value in updating.items():
            existing = result.get(key)
            if isinstance(existing, dict) and isinstance(value, dict):
                result[key] = FlextStrictYamlConfigSource.merge_values(
                    existing,
                    value,
                )
            elif isinstance(existing, list) and isinstance(value, list):
                result[key] = [*existing, *value]
            else:
                result[key] = value
        return result


__all__ = ("FlextStrictYamlConfigSource",)
