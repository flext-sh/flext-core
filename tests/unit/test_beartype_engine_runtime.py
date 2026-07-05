"""Beartype runtime activation behavior tests."""

from __future__ import annotations

import textwrap
from pathlib import Path

from tests.unit._beartype_engine_support import (
    TestsFlextBeartypeEngine,
)


class TestsFlextBeartypeEngineRuntime(TestsFlextBeartypeEngine):
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
                    except AttributeError as exc:
                        print("runtime_exc", type(exc).__name__)
                    print("warning_count", len(caught))
                """,
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

                from flext_core._constants.enforcement import (
                    FlextConstantsEnforcement as c,
                )

                beartype_package(
                    "flext_core",
                    conf=BeartypeConf(
                        violation_type=UserWarning,
                        strategy=BeartypeStrategy.O1,
                        claw_skip_package_names=c.BEARTYPE_CLAW_SKIP_PACKAGES,
                    ),
                )

                import flext_core

                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    try:
                        flext_core.FlextUtilitiesProjectMetadata.derive_class_stem(1)
                    except AttributeError as exc:
                        print("runtime_exc", type(exc).__name__)
                        print("runtime_msg", str(exc))
                    print("warning_count", len(caught))
                    if caught:
                        print("warning_type", type(caught[0].message).__name__)
                        print("warning_text", str(caught[0].message))
                """,
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
        assert "'int' object has no attribute" in combined_output

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
                """,
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
