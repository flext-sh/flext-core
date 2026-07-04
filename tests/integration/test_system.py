"""Teste de integração completo para o sistema FLEXT com refatoração adequada.

Este teste único e abrangente valida que todo o sistema flext-core funciona
corretamente após wildcard imports, com foco em railway-oriented programming,
hierarquia de constantes, sistema de exceções e utilitários.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from flext_tests import e, r

from flext_core import FlextContainer
from tests.constants import c
from tests.typings import t
from tests.utilities import u

from .system_integration_cases import TestsFlextFlextSystemWorkflowCases

if TYPE_CHECKING:
    from tests.protocols import p


class TestsFlextSystemIntegration(TestsFlextFlextSystemWorkflowCases):
    """Teste de integração completo do sistema FLEXT.

    Este teste único valida todo o ecosistema flext-core através de cenários
    realistas que demonstram a integração correta entre componentes.
    """

    def test_complete_system_integration_workflow(self) -> None:
        """Teste completo de integração do sistema FLEXT.

        Este teste abrange:
        1. Wildcard imports funcionais
        2. Railway-oriented programming com r
        3. Sistema hierárquico de constantes
        4. Hierarquia de exceções estruturada
        5. Utilitários e funções auxiliares
        7. Validação e configuração
        8. Cenários de erro e recuperação

        """
        self._validate_imports()
        self._test_railway_programming()
        self._test_constants_system()
        self._test_exceptions_system()
        self._test_utilities()
        self._test_container_system()
        self._test_complex_integration()
        self._test_error_recovery()
        self._validate_final_system()

    def _validate_imports(self) -> None:
        """Validate that all main components are available."""
        assert r is not None, "r não está disponível"
        assert c is not None, "c não está disponível"
        assert e is not None, "e não está disponível"
        assert u is not None, "u não está disponível"
        assert t is not None, "t não está disponível"

    def _test_railway_programming(self) -> None:
        """Test railway-oriented programming with r."""
        success_result = r[str].ok("dados_iniciais")
        assert success_result.success
        assert success_result.success
        assert success_result.failure is False
        assert success_result.value == "dados_iniciais"
        assert success_result.error is None
        pipeline_result = (
            success_result
            .map(lambda x: x.upper())
            .map(lambda x: f"processado_{x}")
            .map(lambda x: x.replace("_", "-"))
        )
        assert pipeline_result.success is True
        assert pipeline_result.value == "processado-DADOS-INICIAIS"
        failure_result: p.Result[str] = r[str].fail(
            "erro_de_processamento",
        )
        assert failure_result.success is False
        assert failure_result.failure is True
        assert failure_result.error == "erro_de_processamento"
        assert failure_result.unwrap_or("") == ""

        def operacao_que_pode_falhar(data: str) -> p.Result[str]:
            if "invalido" in data:
                return r[str].fail("dados_invalidos")
            return r[str].ok(f"validado_{data}")

        flat_map_success = success_result.flat_map(operacao_que_pode_falhar)
        assert flat_map_success.success is True
        assert flat_map_success.value == "validado_dados_iniciais"
        invalid_data = r[str].ok("dados_invalido")
        flat_map_failure = invalid_data.flat_map(operacao_que_pode_falhar)
        assert flat_map_failure.success is False
        assert flat_map_failure.error == "dados_invalidos"

    def _test_constants_system(self) -> None:
        """Test hierarchical constants system."""
        timeout_default = c.DEFAULT_TIMEOUT_SECONDS
        assert isinstance(timeout_default, (int, float))
        assert timeout_default > 0
        validation_error_code = c.ErrorCode.VALIDATION_ERROR
        assert isinstance(validation_error_code, str)
        assert validation_error_code == "VALIDATION_ERROR"
        config_error_code = c.ErrorCode.CONFIG_ERROR
        assert isinstance(config_error_code, str)
        assert config_error_code == "CONFIG_ERROR"

    def _test_exceptions_system(self) -> None:
        """Test structured exceptions system."""
        validation_exception = e.ValidationError("campo_invalido")
        assert isinstance(validation_exception, Exception)
        assert isinstance(validation_exception, e.BaseError)
        error_message = str(validation_exception)
        assert "[VALIDATION_ERROR]" in error_message
        assert "campo_invalido" in error_message
        operation_exception = e.OperationError("operacao_falhada")
        assert isinstance(operation_exception, e.BaseError)
        assert "operacao_falhada" in str(operation_exception)
        assert issubclass(e.ValidationError, e.BaseError)
        assert issubclass(e.OperationError, e.BaseError)

    def _test_utilities(self) -> None:
        """Test utilities and helper functions."""
        generated_id = u.generate()
        assert isinstance(generated_id, str)
        assert len(generated_id) == 36
        uuid_obj = uuid.UUID(generated_id)
        assert str(uuid_obj) == generated_id
        timestamp = u.generate_iso_timestamp()
        assert isinstance(timestamp, str)
        assert timestamp
        try:
            safe_int_success = int("42")
            assert safe_int_success == 42
        except ValueError:
            safe_int_success = -1
        try:
            safe_int_failure = int("not_a_number")
        except ValueError:
            safe_int_failure = -1
        assert safe_int_failure == -1

    def _test_container_system(self) -> None:
        """Test container system (Dependency Injection)."""
        container = FlextContainer()
        register_result = container.bind("test_service", "test_value")
        assert register_result is container
        retrieved_service_result = container.resolve("test_service")
        assert retrieved_service_result.success is True
        retrieved_service = retrieved_service_result.value
        assert retrieved_service == "test_value"
        not_found_result = container.resolve("servico_inexistente")
        assert not_found_result.success is False
        assert not_found_result.error is not None

    def _validate_final_system(self) -> None:
        """Validate final system state."""
        assert r is not None
        assert c is not None
        assert e is not None
        assert u is not None
        assert t is not None
        container_final = FlextContainer()
        assert container_final is not None
        resultado_final = r[str].ok("sistema_funcionando")
        assert resultado_final.success is True
        assert resultado_final.value == "sistema_funcionando"
