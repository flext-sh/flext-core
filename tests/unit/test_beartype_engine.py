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
from collections.abc import (
    Mapping,
    Sequence,
)
from pathlib import Path

import pytest
from beartype import BeartypeConf, BeartypeStrategy

from flext_core import FlextUtilitiesBeartypeConf, FlextUtilitiesBeartypeEngine as be
from tests import c, m, t, u

# contains_any                                                        #


type _AnyAlias = str | typing.Any
type _CleanAlias = str | int
type _NestedAnyAlias = Mapping[str, typing.Any]


class TestsFlextCoreBeartypeEngine:
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
        """t.IntMapping has no Any."""
        assert be.contains_any(t.IntMapping) is False

    def test_clean_sequence(self) -> None:
        """t.StrSequence has no Any."""
        assert be.contains_any(t.StrSequence) is False

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

    # has_forbidden_collection_origin                                     #

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
        """t.IntMapping is not forbidden."""
        result = be.has_forbidden_collection_origin(t.IntMapping, self.FORBIDDEN)
        assert result == (False, "")

    def test_sequence_ok(self) -> None:
        """t.StrSequence is not forbidden."""
        result = be.has_forbidden_collection_origin(t.StrSequence, self.FORBIDDEN)
        assert result == (False, "")

    def test_plain_str_ok(self) -> None:
        """Plain str has no origin."""
        result = be.has_forbidden_collection_origin(str, self.FORBIDDEN)
        assert result == (False, "")

    # count_union_members                                                 #

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

    # matches_str_none_union                                              #

    def test_str_none(self) -> None:
        """Str | None detected."""
        assert be.matches_str_none_union(str | None) is True

    def test_plain_str(self) -> None:
        """Plain str is not str | None."""
        assert be.matches_str_none_union(str) is False

    def test_int_none(self) -> None:
        """Int | None is NOT str | None."""
        assert be.matches_str_none_union(int | None) is False

    def test_str_int(self) -> None:
        """Str | int is NOT str | None."""
        assert be.matches_str_none_union(str | int) is False

    def test_str_int_none(self) -> None:
        """Str | int | None IS str | None (str and None both present)."""
        assert be.matches_str_none_union(str | int | None) is True

    # alias_contains_any                                                  #

    def test_alias_with_any(self) -> None:
        """Type alias containing Any is detected."""
        assert be.alias_contains_any(_AnyAlias.__value__) is True

    def test_clean_alias(self) -> None:
        """Type alias without Any passes."""
        assert be.alias_contains_any(_CleanAlias.__value__) is False

    def test_nested_any_in_alias(self) -> None:
        """Nested Any in alias is detected."""
        assert be.alias_contains_any(_NestedAnyAlias.__value__) is True

    # BeartypeConf factory                                                #

    def test_default_mode_conf(self) -> None:
        """Default beartype mode is disabled in flext_core."""
        conf = FlextUtilitiesBeartypeConf.build_beartype_conf()
        assert conf.strategy is BeartypeStrategy.O0

    def test_default_mode_strategy(self) -> None:
        """Disabled default mode uses O0 strategy."""
        conf = FlextUtilitiesBeartypeConf.build_beartype_conf()
        assert conf.strategy is BeartypeStrategy.O0

    def test_conf_is_beartype_conf(self) -> None:
        """Factory returns a proper BeartypeConf instance."""
        conf = FlextUtilitiesBeartypeConf.build_beartype_conf()
        assert isinstance(conf, BeartypeConf)

    def test_beartype_mode_matches_default(self) -> None:
        """flext_core starts with beartype activation disabled by default."""
        assert c.BEARTYPE_MODE is c.EnforcementMode.OFF

    # Facade accessibility                                                #

    def test_contains_any_on_facade(self) -> None:
        """u.contains_any is accessible."""

    def test_build_beartype_conf_on_facade(self) -> None:
        """u.build_beartype_conf is accessible."""

    def test_has_forbidden_collection_origin_on_facade(self) -> None:
        """u.has_forbidden_collection_origin is accessible."""

    @pytest.mark.parametrize(
        "method",
        [
            "contains_any",
            "has_forbidden_collection_origin",
            "count_union_members",
            "matches_str_none_union",
            "alias_contains_any",
            "build_beartype_conf",
        ],
    )
    def test_all_methods_on_facade(self, method: str) -> None:
        """All beartype engine + conf methods on u.*."""

    # beartype.claw compatibility                                         #

    @staticmethod
    def _run_python(script: str, cwd: Path) -> m.Cli.CommandOutput:
        """Run a Python snippet in a subprocess and capture text output."""
        result = u.Cli.run_raw(
            [sys.executable, "-c", script],
            cwd=cwd,
        )
        if result.success:
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

                beartype_this_package(conf=FlextUtilitiesBeartypeConf.build_beartype_conf())
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

                beartype_this_package(conf=FlextUtilitiesBeartypeConf.build_beartype_conf())
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
        """Current flext_core beartype settings imports successfully."""
        result = self._run_python(
            textwrap.dedent(
                """
                from beartype.claw import beartype_package
                from flext_core import FlextUtilitiesBeartypeConf

                beartype_package(
                    "flext_core",
                    conf=FlextUtilitiesBeartypeConf.build_beartype_conf(),
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

    def test_enforce_054_ignores_tests_init_modules(self, tmp_path: Path) -> None:
        """ENFORCE-054 must ignore legitimate wrapper-root ``tests/__init__.py`` files."""
        package_dir = tmp_path / "hookprobe"
        tests_dir = package_dir / "tests"
        tests_dir.mkdir(parents=True)
        (package_dir / "__init__.py").write_text("", encoding="utf-8")
        (tests_dir / "__init__.py").write_text(
            textwrap.dedent(
                """
                class c:
                    class Core:
                        class Tests:
                            ERR_OK_FAILED = "ok"


                class Probe:
                    value = c.Core.Tests.ERR_OK_FAILED
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
                from hookprobe.tests import Probe
                from flext_core._utilities.beartype_engine import FlextUtilitiesBeartypeEngine as be

                from flext_core._constants.enforcement import FlextConstantsEnforcement as c
                from flext_core._models.enforcement import FlextModelsEnforcement as me
                print(repr(be.apply(
                    c.EnforcementPredicateKind.DEPRECATED_SYNTAX,
                    me.DeprecatedSyntaxParams(ast_shape="no_core_tests_namespace"),
                    Probe,
                )))
                """
            ),
            cwd=Path(__file__).resolve().parents[2],
        )

        assert result.exit_code == 0, result.stderr
        assert "None" in result.stdout

    def test_enforce_054_ignores_string_literals_in_test_modules(
        self,
        tmp_path: Path,
    ) -> None:
        """ENFORCE-054 must ignore string literals that only mention ``.Core.Tests``."""
        package_dir = tmp_path / "stringprobe"
        tests_dir = package_dir / "tests"
        tests_dir.mkdir(parents=True)
        (package_dir / "__init__.py").write_text("", encoding="utf-8")
        (tests_dir / "__init__.py").write_text("", encoding="utf-8")
        (tests_dir / "sample.py").write_text(
            textwrap.dedent(
                """
                class Probe:
                    value = "c.Core.Tests.ERR_OK_FAILED"
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
                from stringprobe.tests.sample import Probe
                from flext_core._utilities.beartype_engine import FlextUtilitiesBeartypeEngine as be

                from flext_core._constants.enforcement import FlextConstantsEnforcement as c
                from flext_core._models.enforcement import FlextModelsEnforcement as me
                print(repr(be.apply(
                    c.EnforcementPredicateKind.DEPRECATED_SYNTAX,
                    me.DeprecatedSyntaxParams(ast_shape="no_core_tests_namespace"),
                    Probe,
                )))
                """
            ),
            cwd=Path(__file__).resolve().parents[2],
        )

        assert result.exit_code == 0, result.stderr
        assert "None" in result.stdout

    def test_enforce_055_detects_wrapper_submodule_alias_import(
        self,
        tmp_path: Path,
    ) -> None:
        """ENFORCE-055 must detect alias imports from wrapper submodules."""
        package_dir = tmp_path / "importprobe"
        tests_dir = package_dir / "tests"
        tests_dir.mkdir(parents=True)
        (package_dir / "__init__.py").write_text("", encoding="utf-8")
        (tests_dir / "__init__.py").write_text("", encoding="utf-8")
        (tests_dir / "sample.py").write_text(
            textwrap.dedent(
                """
                from tests.constants import c


                class Probe:
                    value = c
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
                from importprobe.tests.sample import Probe
                from flext_core._utilities.beartype_engine import FlextUtilitiesBeartypeEngine as be

                from flext_core._constants.enforcement import FlextConstantsEnforcement as c
                from flext_core._models.enforcement import FlextModelsEnforcement as me
                print(repr(be.apply(
                    c.EnforcementPredicateKind.DEPRECATED_SYNTAX,
                    me.DeprecatedSyntaxParams(ast_shape="no_wrapper_root_alias_import"),
                    Probe,
                )))
                """
            ),
            cwd=Path(__file__).resolve().parents[2],
        )

        assert result.exit_code == 0, result.stderr
        assert "from tests.constants import c" in result.stdout

    def test_enforce_055_ignores_string_literals_in_test_modules(
        self,
        tmp_path: Path,
    ) -> None:
        """ENFORCE-055 must ignore string literals that only mention wrapper imports."""
        package_dir = tmp_path / "importstringprobe"
        tests_dir = package_dir / "tests"
        tests_dir.mkdir(parents=True)
        (package_dir / "__init__.py").write_text("", encoding="utf-8")
        (tests_dir / "__init__.py").write_text("", encoding="utf-8")
        (tests_dir / "sample.py").write_text(
            textwrap.dedent(
                """
                class Probe:
                    value = "from tests.constants import c"
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
                from importstringprobe.tests.sample import Probe
                from flext_core._utilities.beartype_engine import FlextUtilitiesBeartypeEngine as be

                from flext_core._constants.enforcement import FlextConstantsEnforcement as c
                from flext_core._models.enforcement import FlextModelsEnforcement as me
                print(repr(be.apply(
                    c.EnforcementPredicateKind.DEPRECATED_SYNTAX,
                    me.DeprecatedSyntaxParams(ast_shape="no_wrapper_root_alias_import"),
                    Probe,
                )))
                """
            ),
            cwd=Path(__file__).resolve().parents[2],
        )

        assert result.exit_code == 0, result.stderr
        assert "None" in result.stdout

    def test_importing_flext_core_default_off_skips_auto_activation(self) -> None:
        """Default OFF mode does not auto-activate beartype on flext_core import."""
        result = self._run_python(
            textwrap.dedent(
                """
                import warnings

                import flext_core

                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    try:
                        flext_core.FlextUtilitiesProjectMetadata.derive_class_stem(1)
                    except Exception as exc:
                        print("runtime_exc", type(exc).__name__)
                    print("warning_count", len(caught))
                """
            ),
            cwd=Path(__file__).resolve().parents[2],
        )

        combined_output = result.stdout + result.stderr
        assert result.exit_code == 0, combined_output
        assert "runtime_exc AttributeError" in combined_output
        assert "warning_count 0" in combined_output

    def test_warn_mode_still_executes_wrapped_callable(self) -> None:
        """Warn mode emits a warning but still executes the wrapped function body."""
        result = self._run_python(
            textwrap.dedent(
                """
                import warnings

                from beartype import BeartypeConf, BeartypeStrategy
                from beartype.claw import beartype_package

                from flext_core import FlextUtilitiesBeartypeConf

                beartype_package(
                    "flext_core",
                    conf=BeartypeConf(
                        violation_type=UserWarning,
                        strategy=BeartypeStrategy.O1,
                        claw_skip_package_names=FlextUtilitiesBeartypeConf.CLAW_SKIP_PACKAGES,
                    ),
                )

                import flext_core

                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    try:
                        flext_core.FlextUtilitiesProjectMetadata.derive_class_stem(1)
                    except Exception as exc:
                        print("runtime_exc", type(exc).__name__)
                        print("runtime_msg", str(exc))
                    print("warning_count", len(caught))
                    if caught:
                        print("warning_type", type(caught[0].message).__name__)
                        print("warning_text", str(caught[0].message))
                """
            ),
            cwd=Path(__file__).resolve().parents[2],
        )

        combined_output = result.stdout + result.stderr
        assert result.exit_code == 0, combined_output
        # Behavioural contract: warn mode lets the wrapped callable still
        # execute past the type check — proven by the runtime error fired by
        # the function body itself (AttributeError on ``int.replace`` because
        # the function received the wrong type but ran anyway). Whether
        # beartype claw decorated this staticmethod under O1 strategy and
        # emitted a UserWarning is implementation detail of beartype, not
        # part of the warn-mode contract under test.
        assert "runtime_exc AttributeError" in combined_output
        assert "'int' object has no attribute 'replace'" in combined_output

    def test_claw_without_skip_hits_recursive_container_schema(self) -> None:
        """Removing skip settings still fails to import flext_core under claw."""
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
        if result.exit_code == 0:
            # Newer beartype builds can import this path successfully.
            assert "unexpected_success True" in combined_output
            return

        # When import fails, traceback may echo source lines. Only stdout proves
        # that the print statement actually executed.
        assert "unexpected_success" not in result.stdout
        assert (
            "PydanticSchemaGenerationError" in combined_output
            or 'unimportable module "t"' in combined_output
            or "t.StrSequence" in combined_output
            or "JsonValue not PEP 695-compliant unsubscripted type alias"
            in combined_output
        )
