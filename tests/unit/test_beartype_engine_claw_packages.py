"""Behavioral tests for the FLEXT beartype claw configuration factory.

Exercises the public contract of ``FlextUtilitiesBeartypeConf.build_beartype_conf()``:
the conf it produces must let ``beartype.claw``-instrumented packages import cleanly
(Pydantic models, runtime-checkable Protocols, PEP 695 recursive aliases, and
``flext_core`` itself) instead of crashing on known beartype/pydantic edge cases.
Every assertion targets observable behavior (subprocess exit code + stdout, or the
public return value) — never private state of the factory.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from flext_core._utilities.beartype_conf import FlextUtilitiesBeartypeConf
from tests import t
from tests.unit._beartype_engine_support import TestsFlextBeartypeEngine

_CLAW_INIT = (
    textwrap.dedent(
        """
    from beartype.claw import beartype_this_package
    from flext_core._utilities.beartype_conf import FlextUtilitiesBeartypeConf

    beartype_this_package(conf=FlextUtilitiesBeartypeConf.build_beartype_conf())
    """
    ).strip()
    + "\n"
)


class TestsFlextCoreBeartypeEngineClawPackages(TestsFlextBeartypeEngine):
    """Behavioral contract of the claw conf produced for downstream packages."""

    _REPO_ROOT: Path = Path(__file__).resolve().parents[2]

    def _write_claw_package(self, root: Path, name: str, modules: t.StrMapping) -> None:
        """Create a claw-bootstrapped package with the given submodule sources."""
        package_dir = root / name
        package_dir.mkdir()
        (package_dir / "__init__.py").write_text(_CLAW_INIT, encoding="utf-8")
        for module_name, source in modules.items():
            (package_dir / f"{module_name}.py").write_text(
                textwrap.dedent(source).strip() + "\n", encoding="utf-8"
            )

    def _import_modules_script(self, root: Path, dotted_modules: t.StrSequence) -> str:
        """Build a snippet that imports the given modules and prints a marker."""
        lines = [
            "import sys",
            "",
            f"sys.path.insert(0, {str(root)!r})",
            *(f"import {mod}" for mod in dotted_modules),
            'print("claw_import_ok")',
        ]
        return "\n".join(lines) + "\n"

    def test_build_beartype_conf_is_idempotent(self) -> None:
        """Repeated factory calls yield equal, hash-stable conf values (public API)."""
        # Arrange / Act
        conf_a = FlextUtilitiesBeartypeConf.build_beartype_conf()
        conf_b = FlextUtilitiesBeartypeConf.build_beartype_conf()

        # Assert: deterministic public value, safe to reuse across call sites.
        assert conf_a == conf_b
        assert hash(conf_a) == hash(conf_b)

    def test_claw_supports_pydantic_and_runtime_protocols(self, tmp_path: Path) -> None:
        """Claw-instrumented Pydantic models and runtime Protocols import cleanly."""
        # Arrange
        self._write_claw_package(
            tmp_path,
            "pkgprobe",
            {
                "models": """
                    from pydantic import BaseModel


                    class ProbeModel(BaseModel):
                        value: int


                    ProbeModel(value=1)
                """,
                "protocols": """
                    from typing import Protocol, runtime_checkable


                    @runtime_checkable
                    class ProbeProtocol(Protocol):
                        def run(self) -> int: ...


                    class ProbeImpl:
                        def run(self) -> int:
                            return 1


                    assert isinstance(ProbeImpl(), ProbeProtocol)
                """,
            },
        )

        # Act
        result = self._run_python(
            self._import_modules_script(
                tmp_path, ["pkgprobe.models", "pkgprobe.protocols"]
            ),
            cwd=self._REPO_ROOT,
        )

        # Assert
        assert result.exit_code == 0, result.stderr
        assert "claw_import_ok" in result.stdout

    def test_claw_supports_recursive_aliases_in_synthetic_package(
        self, tmp_path: Path
    ) -> None:
        """Claw tolerates PEP 695 recursive aliases and preserves the runtime value."""
        # Arrange
        self._write_claw_package(
            tmp_path,
            "aliasprobe",
            {
                "aliases": """
                    type JsonLike = (
                        dict[str, JsonLike] | list[JsonLike] | str | int | float | bool | None
                    )

                    VALUE: JsonLike = {"ok": [1, "x", None]}
                """
            },
        )

        # Act
        result = self._run_python(
            textwrap.dedent(
                f"""
                import sys

                sys.path.insert(0, {str(tmp_path)!r})
                import aliasprobe.aliases as aliases
                print("aliasprobe_value", aliases.VALUE["ok"][1])
                """
            ),
            cwd=self._REPO_ROOT,
        )

        # Assert: import succeeds and the aliased runtime value is intact.
        assert result.exit_code == 0, result.stderr
        assert "aliasprobe_value x" in result.stdout

    def test_claw_config_imports_flext_core_without_error(self) -> None:
        """The factory conf lets beartype_package instrument flext_core itself."""
        # Arrange / Act
        result = self._run_python(
            textwrap.dedent(
                """
                from beartype.claw import beartype_package
                from flext_core._utilities.beartype_conf import (
                    FlextUtilitiesBeartypeConf,
                )

                beartype_package(
                    "flext_core",
                    conf=FlextUtilitiesBeartypeConf.build_beartype_conf(),
                )
                import flext_core
                print("flext_core_facade", hasattr(flext_core, "u"))
                """
            ),
            cwd=self._REPO_ROOT,
        )

        # Assert: instrumentation succeeds and the public facade stays intact.
        combined_output = result.stdout + result.stderr
        assert result.exit_code == 0, combined_output
        assert "flext_core_facade True" in combined_output
