"""Behavioral tests for the beartype enforcement import-hook predicate.

Exercises the public contract of ``FlextUtilitiesBeartypeEngine.apply`` for the
``DEPRECATED_SYNTAX`` / ``no_wrapper_root_alias_import`` shape (ENFORCE-055):
given a class defined in a test/example/script wrapper module, the engine
returns a violation mapping when that module contains a forbidden facade alias
import, and ``None`` otherwise. Assertions target the returned
``StrMapping | None`` contract, never the engine's internals.
"""

from __future__ import annotations

import importlib
import sys
import textwrap
from pathlib import Path

import pytest

from flext_core._constants.enforcement import FlextConstantsEnforcement as c
from flext_core._models.enforcement import FlextModelsEnforcement as me
from flext_core._utilities.beartype_engine import FlextUtilitiesBeartypeEngine as be
from tests.typings import p, t

_FORBIDDEN_IMPORT = "from tests.constants import c"


class TestsFlextCoreBeartypeEngineImportHooks:
    """Public-contract tests for the wrapper-alias-import enforcement predicate."""

    @staticmethod
    def _probe_class(
        tmp_path: Path,
        *,
        package: str,
        under_tests: bool,
        body: str,
    ) -> type:
        """Materialize ``<package>[/tests]/sample.py`` and import its ``Probe`` class.

        Returns the freshly imported ``Probe`` type bound to a real on-disk
        module, so the engine can introspect it exactly as it would in a live
        enforcement pass.
        """
        root = tmp_path / package
        module_dir = root / "tests" if under_tests else root
        module_dir.mkdir(parents=True)
        (root / "__init__.py").write_text("", encoding="utf-8")
        if under_tests:
            (module_dir / "__init__.py").write_text("", encoding="utf-8")
        (module_dir / "sample.py").write_text(body, encoding="utf-8")

        dotted = f"{package}.tests.sample" if under_tests else f"{package}.sample"
        sys.path.insert(0, str(tmp_path))
        try:
            module = importlib.import_module(dotted)
        finally:
            sys.path.remove(str(tmp_path))
        probe: type = module.Probe
        return probe

    def _apply(self, target: type) -> t.StrMapping | None:
        """Invoke the public engine dispatch for the wrapper-alias-import shape."""
        return be.apply(
            c.EnforcementPredicateKind.DEPRECATED_SYNTAX,
            me.DeprecatedSyntaxParams(ast_shape="no_wrapper_root_alias_import"),
            target,
        )

    def test_detects_forbidden_facade_alias_import_in_wrapper_module(
        self,
        tmp_path: Path,
    ) -> None:
        """A forbidden facade alias import in a tests module yields a violation."""
        probe = self._probe_class(
            tmp_path,
            package="importprobe",
            under_tests=True,
            body=textwrap.dedent(
                """
                from tests.constants import c


                class Probe:
                    value = c
                """,
            ).strip()
            + "\n",
        )

        violation = self._apply(probe)

        assert violation is not None
        assert violation["statement"] == _FORBIDDEN_IMPORT
        assert violation["file"] == "sample.py"
        assert violation["line"] == "1"

    def test_ignores_string_literal_that_merely_mentions_a_forbidden_import(
        self,
        tmp_path: Path,
    ) -> None:
        """A string literal spelling the import is not an import; no violation."""
        probe = self._probe_class(
            tmp_path,
            package="importstringprobe",
            under_tests=True,
            body=textwrap.dedent(
                f"""
                class Probe:
                    value = "{_FORBIDDEN_IMPORT}"
                """,
            ).strip()
            + "\n",
        )

        assert self._apply(probe) is None

    def test_ignores_forbidden_import_outside_wrapper_module(
        self,
        tmp_path: Path,
    ) -> None:
        """The predicate only scans test/example/script wrapper modules."""
        probe = self._probe_class(
            tmp_path,
            package="plainprobe",
            under_tests=False,
            body=textwrap.dedent(
                """
                from tests.constants import c


                class Probe:
                    value = c
                """,
            ).strip()
            + "\n",
        )

        assert self._apply(probe) is None

    def test_unrelated_shape_reports_no_violation_for_clean_class(
        self,
        tmp_path: Path,
    ) -> None:
        """An unrelated ast_shape on a clean wrapper class returns no violation."""
        probe = self._probe_class(
            tmp_path,
            package="cleanshapeprobe",
            under_tests=True,
            body=textwrap.dedent(
                """
                class Probe:
                    value = 1
                """,
            ).strip()
            + "\n",
        )

        result = be.apply(
            c.EnforcementPredicateKind.DEPRECATED_SYNTAX,
            me.DeprecatedSyntaxParams(ast_shape="model_rebuild_call"),
            probe,
        )

        assert result is None

    def test_detection_is_idempotent_across_repeated_applications(
        self,
        tmp_path: Path,
    ) -> None:
        """Re-applying the predicate yields an equal result (stable contract)."""
        probe = self._probe_class(
            tmp_path,
            package="idempotentprobe",
            under_tests=True,
            body=textwrap.dedent(
                """
                from tests.constants import c


                class Probe:
                    value = c
                """,
            ).strip()
            + "\n",
        )

        first = self._apply(probe)
        second = self._apply(probe)

        assert first == second
        assert first is not None

    @pytest.mark.parametrize("attempts", [2, 3])
    def test_string_literal_case_stays_clean_under_repeat(
        self,
        tmp_path: Path,
        attempts: int,
    ) -> None:
        """The no-false-positive guarantee holds across repeated applications."""
        probe = self._probe_class(
            tmp_path,
            package=f"repeatstringprobe{attempts}",
            under_tests=True,
            body=textwrap.dedent(
                f"""
                class Probe:
                    value = "{_FORBIDDEN_IMPORT}"
                """,
            ).strip()
            + "\n",
        )

        assert all(self._apply(probe) is None for _ in range(attempts))
