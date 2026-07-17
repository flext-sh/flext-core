"""Behavioral tests for the beartype engine DEPRECATED_SYNTAX public contract.

Every test asserts the observable return value of the public dispatch entry
point ``FlextUtilitiesBeartypeEngine.apply`` for the ``DEPRECATED_SYNTAX``
predicate kind: ``None`` means *no violation*, a mapping means *violation
detected* (with its public payload). The engine inspects whole importable
modules, so each case materializes a real package on disk and imports it inside
an isolated subprocess before invoking ``apply``.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from flext_tests import tm

from tests.protocols import p
from tests.typings import t
from tests.unit._beartype_engine_support import TestsFlextBeartypeEngine


class TestsFlextBeartypeEngineNamespaceHooks(TestsFlextBeartypeEngine):
    """DEPRECATED_SYNTAX predicate behavior via the public ``apply`` contract."""

    _REPO_ROOT: Path = Path(__file__).resolve().parents[2]

    def _apply_deprecated_syntax(
        self,
        tmp_path: Path,
        files: t.MappingKV[str, str],
        import_target: str,
        ast_shape: str,
    ) -> p.Cli.CommandOutput:
        """Materialize ``files``, import ``Probe`` from ``import_target``, run ``apply``.

        Returns the captured subprocess output; ``stdout`` holds ``repr`` of the
        value returned by ``apply`` for the DEPRECATED_SYNTAX predicate.
        """
        for relative_path, content in files.items():
            target_file = tmp_path / relative_path
            target_file.parent.mkdir(parents=True, exist_ok=True)
            target_file.write_text(content, encoding="utf-8")

        script = textwrap.dedent(
            f"""
            import sys

            sys.path.insert(0, {str(tmp_path)!r})
            from {import_target} import Probe
            from flext_core._utilities.beartype_engine import (
                FlextUtilitiesBeartypeEngine as be,
            )
            from flext_core._constants.enforcement import (
                FlextConstantsEnforcement as c,
            )
            from flext_core._models.enforcement import (
                FlextModelsEnforcement as me,
            )

            print(repr(be.apply(
                c.EnforcementPredicateKind.DEPRECATED_SYNTAX,
                me.DeprecatedSyntaxParams(ast_shape={ast_shape!r}),
                Probe,
            )))
            """
        )
        return self._run_python(script, cwd=self._REPO_ROOT)

    @pytest.mark.parametrize(
        ("case_id", "files", "import_target"),
        [
            pytest.param(
                "wrapper_root_tests_init",
                {
                    "hookprobe/__init__.py": "",
                    "hookprobe/tests/__init__.py": textwrap.dedent(
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
                },
                "hookprobe.tests",
                id="wrapper_root_tests_init_is_exempt",
            ),
            pytest.param(
                "string_literal_mention",
                {
                    "stringprobe/__init__.py": "",
                    "stringprobe/tests/__init__.py": "",
                    "stringprobe/tests/sample.py": textwrap.dedent(
                        """
                        class Probe:
                            value = "c.Core.Tests.ERR_OK_FAILED"
                        """
                    ).strip()
                    + "\n",
                },
                "stringprobe.tests.sample",
                id="string_literal_mention_is_ignored",
            ),
        ],
    )
    def test_no_core_tests_namespace_reports_no_violation_for_legitimate_code(
        self,
        tmp_path: Path,
        case_id: str,
        files: t.MappingKV[str, str],
        import_target: str,
    ) -> None:
        """``no_core_tests_namespace`` returns None for exempt / non-alias code."""
        result = self._apply_deprecated_syntax(
            tmp_path, files, import_target, ast_shape="no_core_tests_namespace"
        )

        tm.that(result.exit_code, eq=0, msg=result.stderr)
        tm.that(result.stdout.strip(), eq="None", msg=f"{case_id}: {result.stdout}")

    def test_private_attr_probe_detects_getattr_on_private_attribute(
        self, tmp_path: Path
    ) -> None:
        """``private_attr_probe`` returns a violation payload for a private probe."""
        result = self._apply_deprecated_syntax(
            tmp_path,
            {
                "probepkg/__init__.py": "",
                "probepkg/mod.py": textwrap.dedent(
                    """
                    def peek(target):
                        return getattr(target, "_secret")


                    class Probe:
                        pass
                    """
                ).strip()
                + "\n",
            },
            import_target="probepkg.mod",
            ast_shape="private_attr_probe",
        )

        tm.that(result.exit_code, eq=0, msg=result.stderr)
        payload = result.stdout.strip()
        tm.that(payload, ne="None", msg=payload)
        tm.that(payload, contains="'probe': 'getattr'", msg=payload)
        tm.that(payload, contains="'name': '_secret'", msg=payload)
        tm.that(payload, contains="'file': 'mod.py'", msg=payload)

    def test_private_attr_probe_reports_no_violation_for_clean_module(
        self, tmp_path: Path
    ) -> None:
        """A module with no private-attribute probe yields None (no false positive)."""
        result = self._apply_deprecated_syntax(
            tmp_path,
            {
                "cleanpkg/__init__.py": "",
                "cleanpkg/mod.py": textwrap.dedent(
                    """
                    def compute(target):
                        return target.public_value + 1


                    class Probe:
                        pass
                    """
                ).strip()
                + "\n",
            },
            import_target="cleanpkg.mod",
            ast_shape="private_attr_probe",
        )

        tm.that(result.exit_code, eq=0, msg=result.stderr)
        tm.that(result.stdout.strip(), eq="None", msg=result.stdout)

    def test_apply_returns_none_for_unrecognized_ast_shape(
        self, tmp_path: Path
    ) -> None:
        """An unknown ``ast_shape`` is a no-op: apply returns None, never raises."""
        result = self._apply_deprecated_syntax(
            tmp_path,
            {
                "shapepkg/__init__.py": "",
                "shapepkg/mod.py": textwrap.dedent(
                    """
                    def peek(target):
                        return getattr(target, "_secret")


                    class Probe:
                        pass
                    """
                ).strip()
                + "\n",
            },
            import_target="shapepkg.mod",
            ast_shape="totally_unrecognized_shape",
        )

        tm.that(result.exit_code, eq=0, msg=result.stderr)
        tm.that(result.stdout.strip(), eq="None", msg=result.stdout)
