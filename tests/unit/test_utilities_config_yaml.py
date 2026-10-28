"""The YAML transport facade: safe parsing and duplicate-key rejection.

The facade is the only door consumers may use for YAML (transport ownership
stays with flext-core). These tests prove the two public parse contracts:
plain safe parsing, and the unique-key loader that turns a duplicated config
key into a parse error instead of a silent overwrite.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from typing_extensions import TypeForm

from flext_core import FlextConfig, FlextStrictYamlConfigSource
from tests import m, t, tm, u


class TestsFlextCoreUtilitiesYaml:
    """Prove the public YAML transport contract on the facade."""

    @staticmethod
    def test_public_merge_matches_ordered_yaml_source(tmp_path: Path) -> None:
        """In-memory consumers use the same mapping/list semantics as file sources."""
        documents = (
            "payload:\n  mode: first\n  entries: [first]\n  nested:\n    first: 1\n",
            "payload:\n  mode: second\n  entries: [second]\n  nested:\n    second: 2\n",
        )
        paths = tuple(
            tmp_path / f"layer-{index}.yaml" for index in range(len(documents))
        )
        for path, document in zip(paths, documents, strict=True):
            path.write_text(document, encoding="utf-8")
        baseline: t.JsonDict = u.type_adapter(TypeForm(t.JsonDict)).validate_python(
            u.Yaml.unique_key_load(documents[0]),
        )
        override: t.JsonDict = u.type_adapter(TypeForm(t.JsonDict)).validate_python(
            u.Yaml.unique_key_load(documents[1]),
        )
        merged = FlextStrictYamlConfigSource.merge_values(baseline, override)
        loaded = FlextStrictYamlConfigSource(
            FlextConfig,
            yaml_file=paths,
            deep_merge=True,
        )()
        tm.that(merged, eq=loaded)
        tm.that(baseline, eq=u.Yaml.unique_key_load(documents[0]))
        tm.that(override, eq=u.Yaml.unique_key_load(documents[1]))

    @staticmethod
    def test_unique_key_load_parses_nested_documents() -> None:
        """Test unique key load parses nested documents."""
        parsed = u.Yaml.unique_key_load("top:\n  inner: 1\n  other: two\n")
        tm.that(parsed, eq={"top": {"inner": 1, "other": "two"}})

    @staticmethod
    def test_unique_key_load_preserves_completed_alias_values() -> None:
        """Validation sees fully constructed YAML aliases, not placeholder lists."""
        parsed = u.Yaml.unique_key_load("first: &values [one, two]\nsecond: *values\n")
        tm.that(parsed, eq={"first": ["one", "two"], "second": ["one", "two"]})

    @staticmethod
    @pytest.mark.parametrize(
        "document",
        [
            "value: !!timestamp 2026-01-01\n",
            "value: !!binary aGVsbG8=\n",
            "value: !!set {one: null}\n",
        ],
        ids=["timestamp", "binary", "set"],
    )
    def test_unique_key_load_rejects_non_json_yaml_values(document: str) -> None:
        """Safe YAML's non-JSON constructors cannot escape the typed boundary."""
        with pytest.raises(m.ValidationError):
            u.Yaml.unique_key_load(document)

    @staticmethod
    def test_unique_key_load_rejects_duplicate_keys_at_depth() -> None:
        """Test unique key load rejects duplicate keys at depth."""
        with pytest.raises(u.Yaml.YAMLError, match="duplicate config key"):
            u.Yaml.unique_key_load("top:\n  dup: 1\n  dup: 2\n")

    @staticmethod
    def test_unique_key_load_flattens_merge_keys_with_explicit_override(
        fixture_tree: Callable[[str], Path],
    ) -> None:
        """Merge keys flatten to defaults; one explicit pair overrides a merged one."""
        document = fixture_tree("yaml_merge_keys") / "override.yaml"
        parsed = u.Yaml.unique_key_load(document.read_text(encoding="utf-8"))
        tm.that(
            parsed,
            eq={
                "base": {"host": "relay", "port": 8080},
                "served": {"host": "relay", "port": 9090},
            },
        )

    @staticmethod
    def test_unique_key_load_rejects_duplicate_explicit_key_beside_merge(
        fixture_tree: Callable[[str], Path],
    ) -> None:
        """Flattening a merge never forgives a duplicated explicit key."""
        document = fixture_tree("yaml_merge_keys") / "duplicate_beside_merge.yaml"
        with pytest.raises(u.Yaml.YAMLError, match="duplicate config key"):
            u.Yaml.unique_key_load(document.read_text(encoding="utf-8"))

    @staticmethod
    def test_unique_key_load_rejects_non_string_keys() -> None:
        """Configuration ingress requires string keys before model validation."""
        with pytest.raises(
            u.Yaml.YAMLError,
            match="config mapping keys must be strings",
        ):
            u.Yaml.unique_key_load("top:\n  1: value\n")

    @staticmethod
    def test_unique_key_load_rejects_recursive_aliases() -> None:
        """Retain rejection of YAML graphs outside the JSON tree contract."""
        with pytest.raises(u.Yaml.YAMLError):
            u.Yaml.unique_key_load("top: &entry\n  reference: *entry\n")

    @staticmethod
    def test_unique_key_load_rejects_malformed_input() -> None:
        """Test unique key load rejects malformed input."""
        with pytest.raises(u.Yaml.YAMLError):
            u.Yaml.unique_key_load("top: [unclosed\n")

    @staticmethod
    def test_unique_key_load_rejects_python_object_tags() -> None:
        """Reject constructors outside the safe YAML protocol."""
        with pytest.raises(u.Yaml.YAMLError):
            u.Yaml.unique_key_load("!!python/object/apply:builtins.str [value]\n")
