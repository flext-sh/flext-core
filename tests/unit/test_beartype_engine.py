"""Tests for beartype.door-powered annotation inspection engine.

Verifies FlextUtilitiesBeartypeEngine check functions detect
forbidden patterns in type annotations using beartype.door.TypeHint.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import sys
import textwrap
import typing
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
from beartype import BeartypeConf, BeartypeStrategy

from flext_core import FlextUtilitiesBeartypeConf, FlextUtilitiesBeartypeEngine as be
from tests import m, u

# ------------------------------------------------------------------ #
# contains_any                                                        #
# ------------------------------------------------------------------ #


class TestContainsAny:
    """Verify recursive Any detection via beartype.door.TypeHint."""

    def test_direct_any(self) -> None:
        """typing.Any is detected."""
        assert be.contains_any(typing.Any) is True

    def test_str_clean(self) -> None:
        """Plain str has no Any."""
        assert be.contains_any(str) is False

    def test_int_clean(self) -> None:
        """Plain int has no Any."""
        assert be.contains_any(int) is False

    def test_mapping_with_any_value(self) -> None:
        """Mapping[str, Any] detected at 1 level."""
        assert be.contains_any(Mapping[str, typing.Any]) is True

    def test_deep_nested_any(self) -> None:
        """Mapping[str, Sequence[Any]] detected at 2 levels."""
        assert be.contains_any(Mapping[str, Sequence[typing.Any]]) is True

    def test_clean_mapping(self) -> None:
        """Mapping[str, int] has no Any."""
        assert be.contains_any(Mapping[str, int]) is False

    def test_clean_sequence(self) -> None:
        """Sequence[str] has no Any."""
        assert be.contains_any(Sequence[str]) is False

    def test_union_with_any(self) -> None:
        """Str | Any detected."""
        assert be.contains_any(str | typing.Any) is True

    def test_optional_any(self) -> None:
        """Any | None detected."""
        assert be.contains_any(typing.Any | None) is True

    def test_clean_optional(self) -> None:
        """Str | None has no Any."""
        assert be.contains_any(str | None) is False

    def test_complex_clean_type(self) -> None:
        """Mapping[str, Sequence[int]] has no Any."""
        assert be.contains_any(Mapping[str, Sequence[int]]) is False


# ------------------------------------------------------------------ #
# has_forbidden_collection_origin                                     #
# ------------------------------------------------------------------ #


class TestForbiddenCollectionOrigin:
    """Verify bare dict/list/set detection."""

    FORBIDDEN: frozenset[str] = frozenset({"dict", "list", "set"})

    def test_bare_dict(self) -> None:
        """dict[str, int] is forbidden."""
        result = be.has_forbidden_collection_origin(dict[str, int], self.FORBIDDEN)
        assert result == (True, "dict")

    def test_bare_list(self) -> None:
        """list[str] is forbidden."""
        result = be.has_forbidden_collection_origin(list[str], self.FORBIDDEN)
        assert result == (True, "list")

    def test_bare_set(self) -> None:
        """set[int] is forbidden."""
        result = be.has_forbidden_collection_origin(set[int], self.FORBIDDEN)
        assert result == (True, "set")

    def test_mapping_ok(self) -> None:
        """Mapping[str, int] is not forbidden."""
        result = be.has_forbidden_collection_origin(Mapping[str, int], self.FORBIDDEN)
        assert result == (False, "")

    def test_sequence_ok(self) -> None:
        """Sequence[str] is not forbidden."""
        result = be.has_forbidden_collection_origin(Sequence[str], self.FORBIDDEN)
        assert result == (False, "")

    def test_plain_str_ok(self) -> None:
        """Plain str has no origin."""
        result = be.has_forbidden_collection_origin(str, self.FORBIDDEN)
        assert result == (False, "")


# ------------------------------------------------------------------ #
# count_union_members                                                 #
# ------------------------------------------------------------------ #


class TestCountUnionMembers:
    """Verify non-None union member counting."""

    def test_simple_union(self) -> None:
        """Str | int has 2 members."""
        assert be.count_union_members(str | int) == 2

    def test_optional(self) -> None:
        """Str | None has 1 non-None member."""
        assert be.count_union_members(str | None) == 1

    def test_complex_union(self) -> None:
        """Str | int | float | None has 3 non-None members."""
        assert be.count_union_members(str | int | float | None) == 3

    def test_non_union(self) -> None:
        """Plain str returns 0."""
        assert be.count_union_members(str) == 0

    def test_triple_union(self) -> None:
        """Str | int | float has 3 members."""
        assert be.count_union_members(str | int | float) == 3


# ------------------------------------------------------------------ #
# is_str_none_union                                                   #
# ------------------------------------------------------------------ #


class TestIsStrNoneUnion:
    """Verify str | None pattern detection."""

    def test_str_none(self) -> None:
        """Str | None detected."""
        assert be.is_str_none_union(str | None) is True

    def test_plain_str(self) -> None:
        """Plain str is not str | None."""
        assert be.is_str_none_union(str) is False

    def test_int_none(self) -> None:
        """Int | None is NOT str | None."""
        assert be.is_str_none_union(int | None) is False

    def test_str_int(self) -> None:
        """Str | int is NOT str | None."""
        assert be.is_str_none_union(str | int) is False

    def test_str_int_none(self) -> None:
        """Str | int | None IS str | None (str and None both present)."""
        assert be.is_str_none_union(str | int | None) is True


# ------------------------------------------------------------------ #
# alias_contains_any                                                  #
# ------------------------------------------------------------------ #


type _AnyAlias = str | typing.Any
type _CleanAlias = str | int
type _NestedAnyAlias = Mapping[str, typing.Any]


class TestAliasContainsAny:
    """Verify PEP 695 type alias Any detection."""

    def test_alias_with_any(self) -> None:
        """Type alias containing Any is detected."""
        assert be.alias_contains_any(_AnyAlias.__value__) is True

    def test_clean_alias(self) -> None:
        """Type alias without Any passes."""
        assert be.alias_contains_any(_CleanAlias.__value__) is False

    def test_nested_any_in_alias(self) -> None:
        """Nested Any in alias is detected."""
        assert be.alias_contains_any(_NestedAnyAlias.__value__) is True


# ------------------------------------------------------------------ #
# BeartypeConf factory                                                #
# ------------------------------------------------------------------ #


class TestBeartypeConf:
    """Verify centralized BeartypeConf factory."""

    def test_warn_mode_conf(self) -> None:
        """Default mode (warn) produces UserWarning violation type."""
        conf = FlextUtilitiesBeartypeConf.get_beartype_conf()
        assert conf.violation_type is UserWarning

    def test_warn_mode_strategy(self) -> None:
        """Default mode uses O1 strategy."""
        conf = FlextUtilitiesBeartypeConf.get_beartype_conf()
        assert conf.strategy is BeartypeStrategy.O1

    def test_conf_is_beartype_conf(self) -> None:
        """Factory returns a proper BeartypeConf instance."""
        conf = FlextUtilitiesBeartypeConf.get_beartype_conf()
        assert isinstance(conf, BeartypeConf)


# ------------------------------------------------------------------ #
# Facade accessibility                                                #
# ------------------------------------------------------------------ #


class TestFacadeAccessibility:
    """Verify beartype engine accessible via u.* facade."""

    def test_contains_any_on_facade(self) -> None:
        """u.contains_any is accessible."""
        assert hasattr(u, "contains_any")

    def test_get_beartype_conf_on_facade(self) -> None:
        """u.get_beartype_conf is accessible."""
        assert hasattr(u, "get_beartype_conf")

    def test_has_forbidden_collection_origin_on_facade(self) -> None:
        """u.has_forbidden_collection_origin is accessible."""
        assert hasattr(u, "has_forbidden_collection_origin")

    @pytest.mark.parametrize(
        "method",
        [
            "contains_any",
            "has_forbidden_collection_origin",
            "count_union_members",
            "is_str_none_union",
            "alias_contains_any",
            "get_beartype_conf",
        ],
    )
    def test_all_methods_on_facade(self, method: str) -> None:
        """All beartype engine + conf methods on u.*."""
        assert hasattr(u, method), f"u.{method} not found on facade"


# ------------------------------------------------------------------ #
# beartype.claw compatibility                                         #
# ------------------------------------------------------------------ #


class TestBeartypeClawCompatibility:
    """Verify current beartype.claw compatibility boundaries."""

    @staticmethod
    def _run_python(script: str, cwd: Path) -> m.Cli.CommandOutput:
        """Run a Python snippet in a subprocess and capture text output."""
        result = u.Cli.run_raw(
            [sys.executable, "-c", script],
            cwd=cwd,
        )
        if result.is_success:
            return result.value
        return m.Cli.CommandOutput(
            stdout="",
            stderr=result.error or "python snippet execution failed",
            exit_code=1,
        )

    def test_claw_supports_pydantic_and_runtime_protocols(
        self,
        tmp_path: Path,
    ) -> None:
        """Synthetic packages with Pydantic and runtime-checkable Protocols work."""
        package_dir = tmp_path / "pkgprobe"
        package_dir.mkdir()
        (package_dir / "__init__.py").write_text(
            textwrap.dedent(
                """
                from beartype.claw import beartype_this_package
                from flext_core import FlextUtilitiesBeartypeConf

                beartype_this_package(conf=FlextUtilitiesBeartypeConf.get_beartype_conf())
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )
        (package_dir / "models.py").write_text(
            textwrap.dedent(
                """
                from pydantic import BaseModel


                class ProbeModel(BaseModel):
                    value: int


                ProbeModel(value=1)
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )
        (package_dir / "protocols.py").write_text(
            textwrap.dedent(
                """
                from typing import Protocol, runtime_checkable


                @runtime_checkable
                class ProbeProtocol(Protocol):
                    def run(self) -> int: ...


                class ProbeImpl:
                    def run(self) -> int:
                        return 1


                assert isinstance(ProbeImpl(), ProbeProtocol)
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )

        result = self._run_python(
            textwrap.dedent(
                f"""
                import sys

                sys.path.insert(0, {str(tmp_path)!r})
                import pkgprobe.models
                import pkgprobe.protocols
                print("pkgprobe_ok")
                """
            ),
            cwd=Path(__file__).resolve().parents[2],
        )

        assert result.exit_code == 0, result.stderr
        assert "pkgprobe_ok" in result.stdout

    def test_claw_supports_recursive_aliases_in_synthetic_package(
        self,
        tmp_path: Path,
    ) -> None:
        """Synthetic recursive PEP 695 aliases import cleanly under claw."""
        package_dir = tmp_path / "aliasprobe"
        package_dir.mkdir()
        (package_dir / "__init__.py").write_text(
            textwrap.dedent(
                """
                from beartype.claw import beartype_this_package
                from flext_core import FlextUtilitiesBeartypeConf

                beartype_this_package(conf=FlextUtilitiesBeartypeConf.get_beartype_conf())
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )
        (package_dir / "aliases.py").write_text(
            textwrap.dedent(
                """
                type JsonLike = dict[str, JsonLike] | list[JsonLike] | str | int | float | bool | None

                VALUE: JsonLike = {"ok": [1, "x", None]}
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )

        result = self._run_python(
            textwrap.dedent(
                f"""
                import sys

                sys.path.insert(0, {str(tmp_path)!r})
                import aliasprobe.aliases as aliases
                print("aliasprobe_ok", aliases.VALUE["ok"][1])
                """
            ),
            cwd=Path(__file__).resolve().parents[2],
        )

        assert result.exit_code == 0, result.stderr
        assert "aliasprobe_ok x" in result.stdout

    def test_claw_current_config_imports_flext_core(self) -> None:
        """Current flext_core beartype config imports successfully."""
        result = self._run_python(
            textwrap.dedent(
                """
                from beartype.claw import beartype_package
                from flext_core import FlextUtilitiesBeartypeConf

                beartype_package(
                    "flext_core",
                    conf=FlextUtilitiesBeartypeConf.get_beartype_conf(),
                )
                import flext_core
                print("unexpected_success", hasattr(flext_core, "u"))
                """
            ),
            cwd=Path(__file__).resolve().parents[2],
        )

        combined_output = result.stdout + result.stderr
        assert result.exit_code == 0, combined_output
        assert "unexpected_success True" in combined_output
        assert "GuardInput" not in combined_output
        assert "unquoted relative forward reference 'p'" not in combined_output

    def test_claw_without_skip_hits_recursive_container_schema(self) -> None:
        """Removing skip settings still fails on recursive container schema generation."""
        result = self._run_python(
            textwrap.dedent(
                """
                from beartype import BeartypeConf, BeartypeStrategy
                from beartype.claw import beartype_package

                beartype_package(
                    "flext_core",
                    conf=BeartypeConf(
                        violation_type=UserWarning,
                        strategy=BeartypeStrategy.O1,
                    ),
                )
                import flext_core
                print("unexpected_success", hasattr(flext_core, "u"))
                """
            ),
            cwd=Path(__file__).resolve().parents[2],
        )

        combined_output = result.stdout + result.stderr
        assert result.exit_code != 0
        assert "PydanticSchemaGenerationError" in combined_output
        assert "RecursiveContainerMapping" in combined_output
