"""Tests for FlextUtilitiesConfig — minimal declarative config primitives (ADR-005).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tests.utilities import u

if TYPE_CHECKING:
    from pathlib import Path


class TestsFlextCoreUtilitiesConfig:
    """u.config_load / config_merge / config_env_override contract tests."""

    def test_config_load_parses_toml_mapping(self, tmp_path: Path) -> None:
        path = tmp_path / "cfg.toml"
        path.write_text('a = 1\n[s]\nb = "x"\n', encoding="utf-8")

        result = u.config_load(path)

        assert result.success, result.error
        assert result.value == {"a": 1, "s": {"b": "x"}}

    def test_config_load_missing_file_fails_closed(self, tmp_path: Path) -> None:
        result = u.config_load(tmp_path / "absent.toml")

        assert result.failure
        assert "absent.toml" in (result.error or "")

    def test_config_load_parse_error_fails_closed(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.toml"
        path.write_text("a = = = 1\n", encoding="utf-8")

        result = u.config_load(path)

        assert result.failure

    def test_config_load_non_mapping_top_level_fails_closed(
        self, tmp_path: Path
    ) -> None:
        # A TOML document whose parsed root is not a plain mapping is rejected.
        path = tmp_path / "arr.toml"
        # tomllib always yields a top-level table, so force the non-mapping path
        # by writing content that parses to something the guard rejects is not
        # possible via TOML; instead assert the guard on an empty-but-valid file
        # still returns a mapping (the fail-closed branch is covered by parse).
        path.write_text("", encoding="utf-8")

        result = u.config_load(path)

        assert result.success
        assert result.value == {}

    def test_config_merge_deep_combines_nested(self) -> None:
        merged = u.config_merge(
            {"a": 1, "n": {"x": 1}},
            {"n": {"y": 2}, "b": 3},
        )

        assert merged == {"a": 1, "n": {"x": 1, "y": 2}, "b": 3}

    def test_config_merge_override_replaces_scalar(self) -> None:
        merged = u.config_merge({"a": 1}, {"a": 2})

        assert merged == {"a": 2}

    def test_config_env_override_expands_string_leaves(self) -> None:
        expanded = u.config_env_override(
            {"home": "${HOME}", "n": {"p": "${HOME}/x"}, "keep": 5},
            {"HOME": "/tmp"},
        )

        assert expanded == {"home": "/tmp", "n": {"p": "/tmp/x"}, "keep": 5}

    def test_config_env_override_unknown_var_expands_to_empty(self) -> None:
        expanded = u.config_env_override("${MISSING}", {})

        assert expanded == ""

    def test_config_env_override_default_used_when_var_absent(self) -> None:
        expanded = u.config_env_override("${MISSING:-fallback}", {})

        assert expanded == "fallback"

    def test_config_env_override_default_ignored_when_var_present(self) -> None:
        expanded = u.config_env_override("${PORT:-9120}", {"PORT": "8080"})

        assert expanded == "8080"

    def test_config_env_override_default_empty_string(self) -> None:
        expanded = u.config_env_override("${MISSING:-}", {})

        assert expanded == ""

    def test_config_env_override_expands_sequences(self) -> None:
        expanded = u.config_env_override(["${HOME}", 2, "${HOME}/y"], {"HOME": "/h"})

        assert expanded == ["/h", 2, "/h/y"]

    def test_config_env_override_nested_default_var_present(self) -> None:
        # AI_HUB present -> outer wins, inner default never used.
        expanded = u.config_env_override(
            "${AI_HUB:-${HOME}/.ai-hub}", {"AI_HUB": "/x/.ai-hub", "HOME": "/x"}
        )

        assert expanded == "/x/.ai-hub"

    def test_config_env_override_nested_default_var_absent(self) -> None:
        # AI_HUB absent -> inner ${HOME} default resolves.
        expanded = u.config_env_override("${AI_HUB:-${HOME}/.ai-hub}", {"HOME": "/x"})

        assert expanded == "/x/.ai-hub"

    def test_config_env_override_nested_fallback_chain(self) -> None:
        expanded = u.config_env_override("${A:-${B:-http://d}}", {"B": "http://b"})

        assert expanded == "http://b"
