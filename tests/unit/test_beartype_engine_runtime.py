"""Behavioral tests for the flext_core beartype runtime activation contract.

These tests assert only the observable public contract of how ``flext_core``
behaves with respect to beartype's runtime ``claw`` type-checking:

- the public :func:`flext_core.u.build_beartype_conf` factory return value,
- that a default ``import flext_core`` does not silently enable runtime type
  enforcement, and
- that activating the beartype claw over ``flext_core`` without its shipped
  configuration does not yield a silently-working package.

No test inspects private modules, beartype/pydantic internal error strings, or
whether a specific internal collaborator was invoked.

The warn-mode "wrapped callable still executes" behavior is intentionally not
covered: ``BEARTYPE_MODE`` ships as ``off`` (a compile-time constant with no
public runtime override) and warn-mode claw activation is a deliberately
disabled path (blocked upstream in beartype, per the src constant docstring).
It is only reachable by importing a private module first to trigger a load-order
side effect, which is exactly the implementation coupling these tests remove.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from beartype import BeartypeConf, BeartypeStrategy

import flext_core
from tests.unit._beartype_engine_support import TestsFlextBeartypeEngine

_FLEXT_CORE_ROOT: Path = Path(__file__).resolve().parents[2]


class TestsFlextCoreBeartypeEngineRuntime(TestsFlextBeartypeEngine):
    """Observable contract of flext_core's beartype.claw runtime activation."""

    def test_build_beartype_conf_returns_non_checking_conf_for_shipped_mode(
        self,
    ) -> None:
        """The public factory reflects the shipped OFF beartype mode.

        The shipped ``BEARTYPE_MODE`` is ``off``; the documented contract of the
        factory is that ``off`` yields a non-checking ``O0`` configuration with
        no violation type. This is the public return value a downstream project
        receives when it calls ``u.build_beartype_conf()``.
        """
        # Act
        conf = flext_core.u.build_beartype_conf()

        # Assert
        assert isinstance(conf, BeartypeConf)
        assert conf.strategy is BeartypeStrategy.O0
        assert conf.violation_type is None

    @pytest.mark.parametrize(
        ("arg_literal", "expected_exc"),
        [
            ("1", "AttributeError"),
            ("3.5", "AttributeError"),
            ("['a']", "AttributeError"),
            ("None", "AttributeError"),
        ],
    )
    def test_default_import_does_not_intercept_wrongly_typed_public_call(
        self, arg_literal: str, expected_exc: str
    ) -> None:
        """A default ``import flext_core`` adds no runtime type enforcement.

        With the shipped OFF mode, a public callable invoked with a wrongly
        typed argument is *not* intercepted at the boundary: the call reaches
        the function body, which raises the body's own error. No beartype
        ``UserWarning``/``TypeError`` is emitted by the type checker.
        """
        # Arrange
        result = self._run_python(
            textwrap.dedent(
                f"""
                import warnings

                import flext_core

                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    try:
                        flext_core.FlextUtilitiesProjectMetadata.derive_class_stem(
                            {arg_literal}
                        )
                    except (AttributeError, ValueError) as exc:
                        print("runtime_exc", type(exc).__name__)
                    print("warning_count", len(caught))
                """
            ),
            cwd=_FLEXT_CORE_ROOT,
        )

        # Assert
        combined_output = result.stdout + result.stderr
        assert result.exit_code == 0, combined_output
        assert f"runtime_exc {expected_exc}" in result.stdout
        assert "warning_count 0" in result.stdout

    def test_claw_without_flext_skip_configuration_breaks_flext_core_import(
        self,
    ) -> None:
        """Without the FLEXT skip configuration the claw import path is unusable.

        Activating the claw over ``flext_core`` while omitting the FLEXT skip
        package names does not yield a working package on the supported beartype
        build: the import fails and the success marker is never printed. This is
        the observable justification for the shipped skip configuration; the
        specific internal failure (a pydantic/beartype schema error) is an
        implementation detail and is intentionally not asserted.
        """
        # Arrange
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
            cwd=_FLEXT_CORE_ROOT,
        )

        # Assert: import must not silently succeed; stdout proves the success
        # print never executed.
        assert result.exit_code != 0
        assert "unexpected_success" not in result.stdout
