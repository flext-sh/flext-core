"""Round-trip YAML engine and load/dump surface behind ``u.Cli.yaml_*``.

Builds fresh comment-preserving ``ruamel.yaml`` engines per load/dump call
and owns the ``r[T]`` load/dump operations. Composed into
``FlextCliUtilitiesYaml`` via MRO in ``yaml.py``.

NOTE (multi-agent): mro-i6nq.13 — extracted from the removed
``_yaml_roundtrip_parts/..._part_01`` (engine + load/dump half). The
conversion helpers live in ``_yaml/_convert.py``; do not re-create a second
ruamel engine in a leaf module.

NOTE (multi-agent): ai-hub-gwbu.1 — ruamel ``YAML`` instances keep mutable
reader/scanner state during a parse and are not thread-safe. A module-level
shared instance corrupted concurrent loads under ``crg-maintain --jobs 3``
(``IndexError: string index out of range`` in ``ruamel/yaml/reader.py``).
Engines are therefore built per call; never reintroduce a shared instance.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, TextIO

import ruamel.yaml
from ruamel.yaml.comments import CommentedMap

from flext_cli import c, p, r, t

from ._convert import FlextCliUtilitiesYamlConvertMixin

if TYPE_CHECKING:
    from pathlib import Path


def _roundtrip_yaml() -> ruamel.yaml.YAML:
    """Build a fresh comment/quote-preserving round-trip engine.

    A new instance per call keeps load/dump operations thread-safe: ruamel
    stores mutable parser state on the instance while parsing.
    """
    yaml = ruamel.yaml.YAML()
    yaml.preserve_quotes = True
    yaml.width = 4096
    yaml.indent(mapping=2, sequence=4, offset=2)
    return yaml


class FlextCliUtilitiesYamlEngineMixin(FlextCliUtilitiesYamlConvertMixin):
    """Round-trip (comment/quote-preserving) YAML load/dump surface.

    Loading and dumping return ``r[T]`` so parse and validation failures
    propagate as typed failures, never as silent defaults.
    """

    @staticmethod
    def yaml_roundtrip_load(path: Path) -> p.Result[t.Cli.YamlNode]:
        """Load a YAML file preserving comments/quoting -> ``r[YamlNode]``."""
        if not path.is_file():
            return r[t.Cli.YamlNode].fail(f"YAML file not found: {path}")
        try:
            with path.open("r", encoding=c.Cli.ENCODING_DEFAULT) as fh:
                loaded = _roundtrip_yaml().load(fh)
            node = FlextCliUtilitiesYamlEngineMixin._yaml_coerce_node(loaded)
        except OSError as exc:
            return r[t.Cli.YamlNode].fail(f"YAML read error: {exc}")
        except c.Cli.YamlRoundtripError as exc:
            return r[t.Cli.YamlNode].fail(f"YAML parse error: {exc}")
        except TypeError as exc:
            return r[t.Cli.YamlNode].fail(f"YAML content error: {exc}")
        if node is None:
            return r[t.Cli.YamlNode].fail("YAML document is empty (no content)")
        return r[t.Cli.YamlNode].ok(node)

    @staticmethod
    def yaml_roundtrip_load_text(text: str) -> p.Result[t.Cli.YamlNode]:
        """Parse YAML text preserving comments/quoting -> ``r[YamlNode]``."""
        try:
            loaded = _roundtrip_yaml().load(text)
            node = FlextCliUtilitiesYamlEngineMixin._yaml_coerce_node(loaded)
        except c.Cli.YamlRoundtripError as exc:
            return r[t.Cli.YamlNode].fail(f"YAML parse error: {exc}")
        except TypeError as exc:
            return r[t.Cli.YamlNode].fail(f"YAML content error: {exc}")
        if node is None:
            return r[t.Cli.YamlNode].fail("YAML document is empty (no content)")
        return r[t.Cli.YamlNode].ok(node)

    @staticmethod
    def yaml_roundtrip_load_map(path: Path) -> p.Result[CommentedMap]:
        """Load a YAML file and require a mapping root -> ``r[CommentedMap]``."""
        loaded = FlextCliUtilitiesYamlEngineMixin.yaml_roundtrip_load(path)
        if not loaded.success:
            message = (
                loaded.error if loaded.error is not None else f"YAML load error: {path}"
            )
            return r[CommentedMap].fail(message)
        node = loaded.unwrap()
        if not isinstance(node, CommentedMap):
            return r[CommentedMap].fail(f"{path}: YAML document must be a mapping")
        return r[CommentedMap].ok(node)

    @staticmethod
    def yaml_roundtrip_load_map_text(text: str) -> p.Result[CommentedMap]:
        """Parse YAML text and require a mapping root -> ``r[CommentedMap]``."""
        loaded = FlextCliUtilitiesYamlEngineMixin.yaml_roundtrip_load_text(text)
        if not loaded.success:
            message = (
                loaded.error if loaded.error is not None else "YAML text parse error"
            )
            return r[CommentedMap].fail(message)
        node = loaded.unwrap()
        if not isinstance(node, CommentedMap):
            return r[CommentedMap].fail("YAML text: YAML document must be a mapping")
        return r[CommentedMap].ok(node)

    @staticmethod
    def yaml_roundtrip_dump(data: t.Cli.YamlNode, stream: TextIO) -> p.Result[bool]:
        """Serialize a YAML tree to *stream* -> ``r[bool]``."""
        try:
            _roundtrip_yaml().dump(data, stream)
        except (OSError, c.Cli.YamlRoundtripError, TypeError, ValueError) as exc:
            return r[bool].fail(f"YAML dump error: {exc}")
        return r[bool].ok(True)

    @staticmethod
    def yaml_roundtrip_dump_text(data: t.Cli.YamlNode) -> p.Result[str]:
        """Serialize a YAML tree to text -> ``r[str]``."""
        buffer = io.StringIO()
        dumped = FlextCliUtilitiesYamlEngineMixin.yaml_roundtrip_dump(data, buffer)
        if not dumped.success:
            message = dumped.error if dumped.error is not None else "YAML dump error"
            return r[str].fail(message)
        return r[str].ok(buffer.getvalue())


__all__: list[str] = ["FlextCliUtilitiesYamlEngineMixin"]
