"""Tests for flext_infra.deps.__main__ subcommand dispatch."""

from __future__ import annotations

import importlib
import sys

import pytest

from flext_infra.deps.__main__ import _SUBCOMMANDS, main
from flext_tests import tm


class TestSubcommandMapping:
    """Test subcommand mapping completeness."""

    EXPECTED_SUBCOMMANDS: dict[str, str] = {
        "detect": "flext_infra.deps.detector",
        "extra-paths": "flext_infra.deps.extra_paths",
        "internal-sync": "flext_infra.deps.internal_sync",
        "modernize": "flext_infra.deps.modernizer",
        "path-sync": "flext_infra.deps.path_sync",
    }

    def test_subcommands_count(self) -> None:
        """Test correct number of subcommands."""
        tm.that(len(_SUBCOMMANDS), eq=5)

    @pytest.mark.parametrize(
        ("name", "module"),
        list(EXPECTED_SUBCOMMANDS.items()),
        ids=list(EXPECTED_SUBCOMMANDS.keys()),
    )
    def test_subcommand_mapping(self, name: str, module: str) -> None:
        """Test each subcommand maps to correct module."""
        tm.that(name in _SUBCOMMANDS, eq=True, msg=f"Missing subcommand: {name}")
        tm.that(_SUBCOMMANDS[name], eq=module)

    @pytest.mark.parametrize("name", list(EXPECTED_SUBCOMMANDS.keys()))
    def test_subcommand_module_importable(self, name: str) -> None:
        """Test each subcommand module can be imported."""
        module = importlib.import_module(_SUBCOMMANDS[name])
        tm.that(hasattr(module, "main"), eq=True, msg=f"{name} module has no main()")


class TestMainHelpAndErrors:
    """Test main function help and error handling."""

    def test_main_with_help_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test main with -h flag returns 0."""
        monkeypatch.setattr(sys, "argv", ["prog", "-h"])
        result = main()
        tm.that(result, eq=0)

    def test_main_with_no_arguments(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test main with no arguments returns 1."""
        monkeypatch.setattr(sys, "argv", ["prog"])
        result = main()
        tm.that(result, eq=1)

    def test_main_with_unknown_subcommand(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test main with unknown subcommand returns 1."""
        monkeypatch.setattr(sys, "argv", ["prog", "unknown"])
        result = main()
        tm.that(result, eq=1)


class TestMainReturnValues:
    """Test main function return value normalization."""

    @pytest.mark.parametrize(
        ("return_val", "expected"),
        [
            (0, 0),
            (None, 0),
            (False, 0),
            (42, 42),
            (True, 1),
        ],
        ids=["zero", "none", "false", "nonzero", "true"],
    )
    def test_return_value_normalization(
        self,
        monkeypatch: pytest.MonkeyPatch,
        return_val: object,
        expected: int,
    ) -> None:
        """Test main normalizes subcommand return values."""
        monkeypatch.setattr(sys, "argv", ["prog", "detect"])

        class FakeModule:
            @staticmethod
            def main() -> object:
                return return_val

        monkeypatch.setattr(
            "flext_infra.deps.__main__.importlib.import_module",
            lambda _: FakeModule(),
        )
        result = main()
        tm.that(result, eq=expected)
