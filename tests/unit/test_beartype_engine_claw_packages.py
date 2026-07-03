"""Beartype claw package import tests."""

from __future__ import annotations

import textwrap
from pathlib import Path

from tests.unit._beartype_engine_support import (
    TestsFlextBeartypeEngine,
)


class TestsFlextBeartypeEngineClawPackages(TestsFlextBeartypeEngine):
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
