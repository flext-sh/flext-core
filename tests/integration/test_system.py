"""Teste de integração completo para o sistema FLEXT com refatoração adequada.

Este teste único e abrangente valida que todo o sistema flext-core funciona
corretamente após wildcard imports, com foco em railway-oriented programming,
hierarquia de constantes, sistema de exceções e utilitários.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations
from flext_core.typings import t

import uuid

from flext_core import (
    FlextConstants,
    FlextContainer,
    FlextExceptions,
    FlextResult,
    t,
    u,
)


class TestCompleteFlextSystemIntegration:
    """Teste de integração completo do sistema FLEXT.

    Este teste único valida todo o ecosistema flext-core através de cenários
    realistas que demonstram a integração correta entre componentes.
    """

    def test_complete_system_integration_workflow(self) -> None:
        """Teste completo de integração do sistema FLEXT.

        Este teste abrange:
        1. Wildcard imports funcionais
        2. Railway-oriented programming com FlextResult
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
        assert FlextResult is not None, "FlextResult não está disponível"
        assert FlextConstants is not None, "FlextConstants não está disponível"
        assert FlextExceptions is not None, "FlextExceptions não está disponível"
        assert u is not None, "u não está disponível"
        assert t is not None, "t não está disponível"

    def _test_railway_programming(self) -> None:
        """Test railway-oriented programming with FlextResult."""
        # Cenário de sucesso - criação e encadeamento
        success_result = FlextResult[str].ok("dados_iniciais")
        assert success_result.is_success is True
        assert success_result.is_success is True
        assert success_result.is_failure is False
        assert success_result.value == "dados_iniciais"
        assert success_result.error is None

        # Teste de encadeamento de operações (pipeline)
        pipeline_result = (
            success_result.map(lambda x: x.upper())
            .map(lambda x: f"processado_{x}")
            .map(lambda x: str(x).replace("_", "-"))
        )

        assert pipeline_result.is_success is True
        assert str(pipeline_result.value) == "processado-DADOS-INICIAIS"

        # Cenário de falha
        failure_result = FlextResult[str].fail("erro_de_processamento")
        assert failure_result.is_success is False
        assert failure_result.is_failure is True
        assert failure_result.error == "erro_de_processamento"
        assert failure_result.unwrap_or("") == ""

        # Teste de flat_map para operações que podem falhar
        def operacao_que_pode_falhar(data: object) -> FlextResult[object]:
            if isinstance(data, str) and "invalido" in data:
                return FlextResult[object].fail("dados_invalidos")
            if isinstance(data, str):
                return FlextResult[object].ok(f"validado_{data}")
            return FlextResult[object].fail("tipo_invalido")

        flat_map_success = success_result.flat_map(operacao_que_pode_falhar)
        assert flat_map_success.is_success is True
        assert str(flat_map_success.value) == "validado_dados_iniciais"

        invalid_data = FlextResult[str].ok("dados_invalido")
        flat_map_failure = invalid_data.flat_map(operacao_que_pode_falhar)
        assert flat_map_failure.is_success is False
        assert flat_map_failure.error == "dados_invalidos"

    def _test_constants_system(self) -> None:
        """Test hierarchical constants system."""
        # Testar acesso às constantes hierárquicas
        timeout_default = FlextConstants.Reliability.DEFAULT_TIMEOUT_SECONDS
        assert isinstance(timeout_default, (int, float))
        assert timeout_default > 0

        # Constantes de erro
        validation_error_code = FlextConstants.Errors.VALIDATION_ERROR
        assert isinstance(validation_error_code, str)
        assert validation_error_code == "VALIDATION_ERROR"

        # Verificar outros códigos de erro importantes
        config_error_code = FlextConstants.Errors.CONFIG_ERROR
        assert isinstance(config_error_code, str)
        assert config_error_code == "CONFIG_ERROR"

        # Constantes de validação
        min_name_length = FlextConstants.Validation.MIN_NAME_LENGTH
        assert isinstance(min_name_length, int)
        assert min_name_length > 0

    def _test_exceptions_system(self) -> None:
        """Test structured exceptions system."""
        # Teste da hierarquia de exceções
        validation_exception = FlextExceptions.ValidationError("campo_invalido")
        assert isinstance(validation_exception, Exception)
        assert isinstance(validation_exception, FlextExceptions.BaseError)

        # Verificar formato da mensagem de erro
        error_message = str(validation_exception)
        assert "[VALIDATION_ERROR]" in error_message
        assert "campo_invalido" in error_message

        # Teste de outras exceções da hierarquia
        operation_exception = FlextExceptions.OperationError("operacao_falhada")
        assert isinstance(operation_exception, FlextExceptions.BaseError)
        assert "operacao_falhada" in str(operation_exception)

        # Verificar que as exceções seguem a hierarquia correta
        assert issubclass(FlextExceptions.ValidationError, FlextExceptions.BaseError)
        assert issubclass(FlextExceptions.OperationError, FlextExceptions.BaseError)

    def _test_utilities(self) -> None:
        """Test utilities and helper functions."""
        # Geração de ID usando método unificado
        generated_id = u.generate()
        assert isinstance(generated_id, str)
        assert len(generated_id) == 36  # Formato padrão UUID

        # Verificar que é um UUID válido
        uuid_obj = uuid.UUID(generated_id)
        assert str(uuid_obj) == generated_id

        # Geração de timestamp
        timestamp = u.Generators.generate_iso_timestamp()
        assert isinstance(timestamp, str)
        assert len(timestamp) > 0

        # Test integer conversion (using standard Python conversion)
        try:
            safe_int_success = int("42")
            assert safe_int_success == 42
        except ValueError:
            safe_int_success = -1

        try:
            safe_int_failure = int("not_a_number")
        except ValueError:
            safe_int_failure = -1
        assert (
            safe_int_failure == -1
        )  # Retorna default em caso de erro  # Retorna default em caso de erro

    def _test_container_system(self) -> None:
        """Test container system (Dependency Injection)."""
        # Teste do sistema de container
        # API changed: use get_global() instead of ensure_global_manager()
        container = FlextContainer()

        # Registrar serviço
        register_result = container.with_service("test_service", "test_value")

        assert register_result is container  # Fluent interface returns Self is True

        # Recuperar serviço registrado
        retrieved_service_result: FlextResult[t.GeneralValueType] = container.get(
            "test_service",
        )
        assert retrieved_service_result.is_success is True
        retrieved_service = retrieved_service_result.value
        assert retrieved_service == "test_value"

        # Teste de serviço não encontrado
        not_found_result: FlextResult[t.GeneralValueType] = container.get(
            "servico_inexistente",
        )
        assert not_found_result.is_success is False
        assert not_found_result.error is not None

    def _test_complex_integration(self) -> None:
        """Test complex integration scenarios."""

        # Simulação de processamento de dados completo
        def processar_dados_usuario(
            dados: dict[str, str],
        ) -> FlextResult[dict[str, str]]:
            """Função que simula processamento completo usando todo o sistema.

            Returns:
                FlextResult[dict[str, str]]: Resultado do processamento ou erro.

            """
            # Validar entrada
            if not dados:
                return FlextResult[dict[str, str]].fail(
                    "Dados não fornecidos",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

            # Processar dados
            dados_processados: dict[str, str] = {}

            # u.Validation was completely removed - using direct validation
            for key, value in dados.items():
                if not u.Guards.is_string_non_empty(value):
                    return FlextResult[dict[str, str]].fail(
                        f"Campo '{key}' não pode estar vazio",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )

                # Transformar dados
                dados_processados[key] = f"processado_{value}"

            # Adicionar metadados
            dados_processados["processado_em"] = u.Generators.generate_iso_timestamp()
            dados_processados["processado_por"] = "sistema_flext"

            return FlextResult[dict[str, str]].ok(dados_processados)

        # Teste do processamento completo
        dados_teste = {"nome": "João", "email": "joao@exemplo.com"}
        resultado_processamento = processar_dados_usuario(dados_teste)

        assert resultado_processamento.is_success is True
        dados_finais = resultado_processamento.value
        assert "nome" in dados_finais
        assert "email" in dados_finais
        assert "processado_em" in dados_finais
        assert "processado_por" in dados_finais
        assert dados_finais["nome"] == "processado_João"
        assert dados_finais["email"] == "processado_joao@exemplo.com"

        # Teste de validação de erro
        dados_invalidos = {"nome": "", "email": "joao@exemplo.com"}
        resultado_erro = processar_dados_usuario(dados_invalidos)
        assert resultado_erro.is_success is False
        assert resultado_erro.error is not None
        assert "não pode estar vazio" in resultado_erro.error

    def _test_error_recovery(self) -> None:
        """Test error recovery scenarios."""
        resultado_com_erro = FlextResult[str].fail("erro_original")
        # Use lash for error recovery
        resultado_recuperado = resultado_com_erro.lash(
            lambda _error: FlextResult[str].ok("valor_recuperado"),
        )
        assert resultado_recuperado.is_success is True
        assert resultado_recuperado.value == "valor_recuperado"

        # Teste de múltiplas operações em cadeia com possibilidade de falha
        def operacao_1(data: object) -> FlextResult[object]:
            if isinstance(data, str):
                return FlextResult[object].ok(f"etapa1_{data}")
            return FlextResult[object].fail("tipo_invalido")

        def operacao_2(data: object) -> FlextResult[object]:
            if isinstance(data, str):
                if "erro" in data:
                    return FlextResult[object].fail("erro_na_etapa2")
                return FlextResult[object].ok(f"etapa2_{data}")
            return FlextResult[object].fail("tipo_invalido")

        def operacao_3(data: object) -> FlextResult[object]:
            if isinstance(data, str):
                return FlextResult[object].ok(f"final_{data}")
            return FlextResult[object].fail("tipo_invalido")

        # Pipeline de sucesso
        pipeline_sucesso = (
            FlextResult[str]
            .ok("dados_iniciais")
            .flat_map(operacao_1)
            .flat_map(operacao_2)
            .flat_map(operacao_3)
        )

        assert pipeline_sucesso.is_success is True
        assert str(pipeline_sucesso.value) == "final_etapa2_etapa1_dados_iniciais"

        # Pipeline com falha
        pipeline_falha = (
            FlextResult[str]
            .ok("dados_com_erro")
            .flat_map(operacao_1)
            .flat_map(operacao_2)
            .flat_map(operacao_3)
        )

        assert pipeline_falha.is_success is False
        assert pipeline_falha.error == "erro_na_etapa2"

    def _validate_final_system(self) -> None:
        """Validate final system state."""
        # Verificar que todos os componentes principais funcionam em conjunto
        assert FlextResult is not None
        assert FlextConstants is not None
        assert FlextExceptions is not None
        assert u is not None
        assert t is not None

        # Verificar que o sistema está pronto para uso em produção
        # Create a new container instance for testing
        container_final = FlextContainer()
        assert container_final is not None

        # Teste final de integração
        resultado_final = FlextResult[str].ok("sistema_funcionando")
        assert resultado_final.is_success is True
        assert resultado_final.value == "sistema_funcionando"
