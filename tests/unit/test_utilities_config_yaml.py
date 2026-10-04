"""The YAML transport facade: safe parsing and duplicate-key rejection.

The facade is the only door consumers may use for YAML (transport ownership
stays with flext-core). These tests prove the two public parse contracts:
plain safe parsing, and the unique-key loader that turns a duplicated config
key into a parse error instead of a silent overwrite.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import pytest
from flext_tests import tm

from flext_core import u


class TestsFlextCoreUtilitiesYaml:
    """Prove the public YAML transport contract on the facade."""

    @staticmethod
    def test_unique_key_load_parses_nested_documents() -> None:
        """Test unique key load parses nested documents."""
        parsed = u.Yaml.unique_key_load("top:\n  inner: 1\n  other: two\n")
        tm.that(parsed, eq={"top": {"inner": 1, "other": "two"}})

    @staticmethod
    def test_unique_key_load_rejects_duplicate_keys_at_depth() -> None:
        """Test unique key load rejects duplicate keys at depth."""
        with pytest.raises(u.Yaml.YAMLError, match="duplicate config key"):
            u.Yaml.unique_key_load("top:\n  dup: 1\n  dup: 2\n")

    @staticmethod
    def test_unique_key_load_rejects_malformed_input() -> None:
        """Test unique key load rejects malformed input."""
        with pytest.raises(u.Yaml.YAMLError):
            u.Yaml.unique_key_load("top: [unclosed\n")
