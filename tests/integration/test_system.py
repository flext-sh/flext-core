"""Behavioral integration tests for the flext-core public surface.

Every assertion here exercises OBSERVABLE PUBLIC CONTRACT only: the ``r[T]``
railway outcome (``success`` / ``failure`` / ``value`` / ``error`` / combinators),
the ``FlextExceptions`` family (class + public fields + message), the constants
facade, public utilities, and the ``FlextContainer`` dependency-injection API.
No private attributes, no internal-collaborator spying, no patching of the unit
under test.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import pytest
from flext_tests import e, r, tm

from flext_core import FlextContainer
from tests import c
from tests import u

if TYPE_CHECKING:
    from tests import p

_UUID_TEXT_LENGTH = 36


class TestsFlextCoreSystem:
    """Behavioral contract of the composed flext-core public API."""

    # ------------------------------------------------------------------ #
    # Railway-oriented programming: r[T] outcome contract                 #
    # ------------------------------------------------------------------ #

    def test_ok_result_exposes_success_contract(self) -> None:
        """A successful result is success, not failure, carries the value, no error."""
        result = r[str].ok("payload")

        tm.that(result.success, eq=True)
        tm.that(result.failure, eq=False)
        tm.that(result.value, eq="payload")
        tm.that(result.unwrap(), eq="payload")
        tm.that(result.error, none=True)

    def test_fail_result_exposes_failure_contract(self) -> None:
        """A failed result is failure, exposes the error, and carries the error code."""
        result: p.Result[str] = r[str].fail(
            "processing_failed",
            error_code=c.ErrorCode.VALIDATION_ERROR,
        )

        tm.that(result.success, eq=False)
        tm.that(result.failure, eq=True)
        tm.that(result.error, eq="processing_failed")
        tm.that(result.error_code, eq=c.ErrorCode.VALIDATION_ERROR)

    def test_unwrap_or_returns_default_only_on_failure(self) -> None:
        """unwrap_or yields the value on success and the default on failure."""
        tm.that(r[str].ok("real").unwrap_or("default"), eq="real")
        tm.that(r[str].fail("boom").unwrap_or("default"), eq="default")

    @pytest.mark.parametrize(
        ("start", "expected"),
        [
            ("dados_iniciais", "processado-DADOS-INICIAIS"),
            ("abc", "processado-ABC"),
        ],
    )
    def test_map_chain_transforms_success_value(
        self, start: str, expected: str
    ) -> None:
        """Chained map applies each transform in order to a successful value."""
        result = (
            r[str]
            .ok(start)
            .map(lambda x: x.upper())
            .map(lambda x: f"processado_{x}")
            .map(lambda x: x.replace("_", "-"))
        )

        tm.that(result.success, eq=True)
        tm.that(result.value, eq=expected)

    def test_map_is_skipped_on_failure(self) -> None:
        """Map does not run its function once the result is a failure."""
        result = r[str].fail("boom").map(lambda x: x.upper())

        tm.that(result.failure, eq=True)
        tm.that(result.error, eq="boom")

    @pytest.mark.parametrize(
        ("data", "expect_success", "expected"),
        [
            ("dados_iniciais", True, "validado_dados_iniciais"),
            ("dados_invalido", False, "dados_invalidos"),
        ],
    )
    def test_flat_map_chains_or_short_circuits(
        self, data: str, expected: str, *, expect_success: bool
    ) -> None:
        """flat_map threads a fallible op and short-circuits on its failure."""

        def maybe_fail(value: str) -> p.Result[str]:
            if "invalido" in value:
                return r[str].fail("dados_invalidos")
            return r[str].ok(f"validado_{value}")

        result = r[str].ok(data).flat_map(maybe_fail)

        tm.that(result.success, eq=expect_success)
        tm.that(result.value if expect_success else result.error, eq=expected)

    def test_lash_replaces_failure_with_success(self) -> None:
        """Lash turns a failure into a successful fallback result."""
        recovered = (
            r[str]
            .fail("erro_original")
            .lash(
                lambda _error: r[str].ok("valor_recuperado"),
            )
        )

        tm.that(recovered.success, eq=True)
        tm.that(recovered.value, eq="valor_recuperado")

    def test_flat_map_pipeline_propagates_first_failure(self) -> None:
        """A multi-stage flat_map pipeline stops at the first failing stage."""

        def stage_1(data: str) -> p.Result[str]:
            return r[str].ok(f"etapa1_{data}")

        def stage_2(data: str) -> p.Result[str]:
            if "erro" in data:
                return r[str].fail("erro_na_etapa2")
            return r[str].ok(f"etapa2_{data}")

        def stage_3(data: str) -> p.Result[str]:
            return r[str].ok(f"final_{data}")

        ok_pipe = (
            r[str].ok("dados").flat_map(stage_1).flat_map(stage_2).flat_map(stage_3)
        )
        fail_pipe = (
            r[str]
            .ok("dados_erro")
            .flat_map(stage_1)
            .flat_map(stage_2)
            .flat_map(stage_3)
        )

        tm.that(ok_pipe.value, eq="final_etapa2_etapa1_dados")
        tm.that(fail_pipe.failure, eq=True)
        tm.that(fail_pipe.error, eq="erro_na_etapa2")

    # ------------------------------------------------------------------ #
    # Constants facade contract                                           #
    # ------------------------------------------------------------------ #

    def test_default_timeout_is_a_positive_number(self) -> None:
        """The default timeout constant is a positive numeric value."""
        tm.that(c.DEFAULT_TIMEOUT_SECONDS, is_=(int, float), gt=0)

    @pytest.mark.parametrize(
        ("code", "expected"),
        [
            (c.ErrorCode.VALIDATION_ERROR, "VALIDATION_ERROR"),
            (c.ErrorCode.CONFIG_ERROR, "CONFIG_ERROR"),
        ],
    )
    def test_error_codes_render_as_their_string_value(
        self, code: str, expected: str
    ) -> None:
        """Error-code constants are string-valued and stable."""
        tm.that(code, is_=str)
        tm.that(code, eq=expected)

    # ------------------------------------------------------------------ #
    # FlextExceptions family contract                                     #
    # ------------------------------------------------------------------ #

    @pytest.mark.parametrize(
        ("factory", "message", "code"),
        [
            (e.ValidationError, "campo_invalido", "VALIDATION_ERROR"),
            (e.OperationError, "operacao_falhada", "OPERATION_ERROR"),
        ],
    )
    def test_exception_carries_code_and_message_publicly(
        self, factory: type[e.BaseError], message: str, code: str
    ) -> None:
        """Each family exception is a BaseError exposing its code and message."""
        exc = factory(message)

        tm.that(exc, is_=e.BaseError)
        tm.that(exc, is_=Exception)
        tm.that(str(exc), has=message)
        tm.that(str(exc), has=f"[{code}]")

    def test_family_exceptions_are_raisable_and_catchable_as_base(self) -> None:
        """A specific family error is caught through its BaseError supertype."""
        message = "campo_invalido"
        with pytest.raises(e.BaseError) as caught:
            raise e.ValidationError(message)

        tm.that(caught.value, is_=e.ValidationError)
        tm.that(caught.value.error_code, eq="VALIDATION_ERROR")

    def test_family_hierarchy_subclasses_base_error(self) -> None:
        """Concrete family classes derive from BaseError."""
        tm.that(issubclass(e.ValidationError, e.BaseError), eq=True)
        tm.that(issubclass(e.OperationError, e.BaseError), eq=True)

    # ------------------------------------------------------------------ #
    # Public utilities contract                                           #
    # ------------------------------------------------------------------ #

    def test_generate_returns_a_parseable_uuid_string(self) -> None:
        """Generate returns a 36-char string that round-trips as a UUID."""
        generated = u.generate()

        tm.that(generated, is_=str)
        tm.that(len(generated), eq=_UUID_TEXT_LENGTH)
        tm.that(str(uuid.UUID(generated)), eq=generated)

    def test_generate_produces_unique_values(self) -> None:
        """Successive generate calls produce distinct identifiers."""
        tm.that(u.generate(), ne=u.generate())

    def test_iso_timestamp_is_a_non_empty_string(self) -> None:
        """generate_iso_timestamp returns a non-empty string."""
        timestamp = u.generate_iso_timestamp()

        tm.that(timestamp, is_=str, empty=False)

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("payload", True),
            ("", False),
            ("   ", False),
        ],
    )
    def test_string_non_empty_reports_meaningful_content(
        self, value: str, *, expected: bool
    ) -> None:
        """string_non_empty is True only for strings with non-whitespace content."""
        tm.that(u.string_non_empty(value), eq=expected)

    # ------------------------------------------------------------------ #
    # FlextContainer dependency-injection contract                        #
    # ------------------------------------------------------------------ #

    def test_bind_is_fluent_and_resolve_returns_bound_value(self) -> None:
        """Bind returns the container for chaining; resolve yields the value."""
        container = FlextContainer()

        tm.that(container.bind("service", "value") is container, eq=True)

        resolved = container.resolve("service")
        tm.that(resolved.success, eq=True)
        tm.that(resolved.value, eq="value")

    def test_resolve_unknown_key_fails_with_error(self) -> None:
        """Resolving an unregistered key yields a failure carrying an error."""
        resolved = FlextContainer().resolve("missing_service")

        tm.that(resolved.success, eq=False)
        tm.that(resolved.error, none=False)

    # ------------------------------------------------------------------ #
    # End-to-end domain workflow over the public r[T] surface             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _process_user_data(data: dict[str, str]) -> p.Result[dict[str, str]]:
        if not data:
            return r[dict[str, str]].fail(
                "Dados não fornecidos",
                error_code=c.ErrorCode.VALIDATION_ERROR,
            )
        processed: dict[str, str] = {}
        for key, value in data.items():
            if not u.string_non_empty(value):
                return r[dict[str, str]].fail(
                    f"Campo '{key}' não pode estar vazio",
                    error_code=c.ErrorCode.VALIDATION_ERROR,
                )
            processed[key] = f"processado_{value}"
        return r[dict[str, str]].ok(processed)

    def test_workflow_transforms_every_field_on_success(self) -> None:
        """A valid record yields a success whose fields are all transformed."""
        result = self._process_user_data(
            {"nome": "João", "email": "joao@exemplo.com"},
        )

        tm.that(result.success, eq=True)
        tm.that(result.value["nome"], eq="processado_João")
        tm.that(result.value["email"], eq="processado_joao@exemplo.com")

    def test_workflow_fails_with_field_error_on_empty_value(self) -> None:
        """An empty field aborts the workflow with a descriptive validation error."""
        result = self._process_user_data(
            {"nome": "", "email": "joao@exemplo.com"},
        )

        tm.that(result.success, eq=False)
        tm.that(result.error, none=False)
        tm.that(tm.not_none(result.error), has="não pode estar vazio")
        tm.that(result.error_code, eq=c.ErrorCode.VALIDATION_ERROR)
