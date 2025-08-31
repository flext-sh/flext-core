"""Teste de integração completo para o sistema FLEXT Core.

Este teste único valida que todo o sistema flext-core funciona corretamente
através de wildcard imports, focando em railway-oriented programming,
hierarquia de constantes, sistema de exceções e utilitários.
"""

from __future__ import annotations

import uuid

import pytest

# Wildcard import para testar todo o sistema (intencional para validar __all__)
from flext_core import *  # noqa: F403, F401
from flext_core import FlextConstants, FlextExceptions, FlextResult, FlextUtilities


class TestFlextCoreIntegration:
    """Teste de integração completo do sistema FLEXT Core.

    Este teste único valida o ecosistema flext-core através de cenários
    realistas que demonstram a integração correta entre componentes.
    """

    def test_wildcard_imports_available(self) -> None:
        """Valida que wildcard imports funcionam e módulos essenciais estão disponíveis."""
        # Verificar que os módulos essenciais estão realmente disponíveis
        current_globals = globals()

        # Módulos que devem estar sempre disponíveis (baseado no __init__.py real)
        essential_modules = [
            "FlextResult",
            "FlextConstants",
            "FlextExceptions",
            "FlextUtilities",
        ]

        missing_modules = []
        for module_name in essential_modules:
            if module_name not in current_globals:
                missing_modules.append(module_name)
            else:
                module = current_globals[module_name]
                assert module is not None, f"Módulo '{module_name}' é None"

        assert not missing_modules, f"Módulos essenciais ausentes: {missing_modules}"

    def test_flext_result_railway_programming(self) -> None:
        """Testa o padrão railway-oriented programming com FlextResult."""
        # Cenário de sucesso
        success_result = FlextResult[str].ok("dados_iniciais")
        assert success_result.success is True
        assert success_result.is_success is True
        assert success_result.is_failure is False
        assert success_result.value == "dados_iniciais"
        assert success_result.error is None

        # Teste de encadeamento (pipeline)
        pipeline_result = success_result.map(lambda x: x.upper()).map(
            lambda x: f"processado_{x}"
        )
        assert pipeline_result.success is True
        assert pipeline_result.value == "processado_DADOS_INICIAIS"

        # Cenário de falha
        failure_result = FlextResult[str].fail("erro_processamento")
        assert failure_result.success is False
        assert failure_result.is_failure is True
        assert failure_result.error == "erro_processamento"

        # Verificar proteção de tipo (não pode acessar value em failure)
        with pytest.raises(
            TypeError, match="Attempted to access value on failed result"
        ):
            _ = failure_result.value

        # Teste flat_map para operações que podem falhar
        def operacao_validacao(data: str) -> FlextResult[str]:
            if "invalido" in data:
                return FlextResult[str].fail("dados_invalidos")
            return FlextResult[str].ok(f"validado_{data}")

        flat_map_success = success_result.flat_map(operacao_validacao)
        assert flat_map_success.success is True
        assert flat_map_success.value == "validado_dados_iniciais"

    def test_flext_constants_hierarchy(self) -> None:
        """Testa o sistema hierárquico de constantes."""
        # Verificar acesso às constantes hierárquicas
        timeout = FlextConstants.Defaults.TIMEOUT
        assert isinstance(timeout, int)
        assert timeout > 0

        # Constantes de erro com formato FLEXT_xxxx
        validation_error = FlextConstants.Errors.VALIDATION_ERROR
        assert isinstance(validation_error, str)
        assert validation_error.startswith("FLEXT_")
        assert "3001" in validation_error

        # Constantes de mensagem
        success_msg = FlextConstants.Messages.SUCCESS
        assert isinstance(success_msg, str)
        assert len(success_msg) > 0

        # Constantes de padrões
        email_pattern = FlextConstants.Patterns.EMAIL_PATTERN
        assert isinstance(email_pattern, str)
        assert "@" in email_pattern

    def test_flext_exceptions_hierarchy(self) -> None:
        """Testa o sistema de exceções estruturado."""
        # Teste ValidationError
        validation_error = FlextExceptions.ValidationError("campo_invalido")
        assert isinstance(validation_error, Exception)
        assert isinstance(validation_error, ValueError)

        # Verificar formato da mensagem
        error_str = str(validation_error)
        assert "[FLEXT_3001]" in error_str
        assert "campo_invalido" in error_str

        # Teste OperationError
        operation_error = FlextExceptions.OperationError("operacao_falhada")
        assert isinstance(operation_error, Exception)
        assert "operacao_falhada" in str(operation_error)

        # Verificar que as exceções são do tipo correto
        assert "_ValidationError" in str(type(validation_error))
        assert "_OperationError" in str(type(operation_error))

    def test_flext_utilities_functionality(self) -> None:
        """Testa as funcionalidades dos utilitários."""
        # Geração de UUID
        generated_uuid = FlextUtilities.generate_uuid()
        assert isinstance(generated_uuid, str)
        assert len(generated_uuid) == 36

        # Verificar que é UUID válido
        uuid_obj = uuid.UUID(generated_uuid)
        assert str(uuid_obj) == generated_uuid

        # Geração de timestamp
        timestamp = FlextUtilities.generate_timestamp()
        assert isinstance(timestamp, str)
        assert len(timestamp) > 0

        # Conversão segura de tipos
        safe_int_result = FlextUtilities.safe_int("42")
        assert safe_int_result == 42

        safe_int_with_default = FlextUtilities.safe_int("not_a_number", default=-1)
        assert safe_int_with_default == -1

    def test_end_to_end_workflow(self) -> None:
        """Teste de fluxo completo end-to-end."""

        def processar_dados_completo(
            dados: dict[str, str],
        ) -> FlextResult[dict[str, str]]:
            """Simulação de processamento usando todo o sistema."""
            # Validação inicial
            if not dados:
                return FlextResult[dict[str, str]].fail(
                    "Dados não fornecidos",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

            # Validação de username
            username = dados.get("username", "")
            if len(username) < 3:
                return FlextResult[dict[str, str]].fail(
                    "Username deve ter pelo menos 3 caracteres"
                )

            # Processamento dos dados
            dados_processados = {
                "username": username,
                "id": FlextUtilities.generate_uuid(),
                "timestamp": str(FlextUtilities.generate_timestamp()),
                "status": "processado",
            }

            return FlextResult[dict[str, str]].ok(dados_processados)

        # Teste de sucesso
        dados_validos = {"username": "usuario_teste"}
        resultado_sucesso = processar_dados_completo(dados_validos)

        assert resultado_sucesso.success is True
        dados_finais = resultado_sucesso.value
        assert dados_finais["username"] == "usuario_teste"
        assert "id" in dados_finais
        assert "timestamp" in dados_finais
        assert dados_finais["status"] == "processado"

        # Teste de falha
        dados_invalidos = {"username": "ab"}  # Muito curto
        resultado_falha = processar_dados_completo(dados_invalidos)

        assert resultado_falha.success is False
        assert resultado_falha.error is not None
        assert "pelo menos 3 caracteres" in resultado_falha.error

    def test_system_stability(self) -> None:
        """Teste de estabilidade do sistema."""
        # Verificar que múltiplas operações funcionam consistentemente
        for i in range(10):
            # Múltiplas gerações de UUID devem ser únicas
            uuid1 = FlextUtilities.generate_uuid()
            uuid2 = FlextUtilities.generate_uuid()
            assert uuid1 != uuid2

            # Múltiplas criações de FlextResult devem funcionar
            result = FlextResult[int].ok(i)
            assert result.value == i

            # Múltiplos acessos a constantes devem ser consistentes
            timeout = FlextConstants.Defaults.TIMEOUT
            assert timeout == FlextConstants.Defaults.TIMEOUT
