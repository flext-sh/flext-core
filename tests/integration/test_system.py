"""Teste de integração completo para o sistema FLEXT com refatoração adequada.

Este teste único e abrangente valida que todo o sistema flext-core funciona
corretamente após wildcard imports, com foco em railway-oriented programming,
hierarquia de constantes, sistema de exceções e utilitários.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import uuid

from flext_core import (
    FlextConstants,
    FlextContainer,
    FlextExceptions,
    FlextResult,
    FlextTypes,
    FlextUtilities,
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

        Returns:
            None: Este teste não retorna valor; apenas valida cenários.

        """
        # =========================================================================
        # FASE 1: Validação de imports e disponibilidade de módulos
        # =========================================================================

        # Verificar que todos os componentes principais estão disponíveis
        # Usar verificação direta em vez de globals() para evitar problemas de contexto
        assert FlextResult is not None, "FlextResult não está disponível"
        assert FlextConstants is not None, "FlextConstants não está disponível"
        assert FlextExceptions is not None, "FlextExceptions não está disponível"
        assert FlextUtilities is not None, "FlextUtilities não está disponível"
        assert FlextTypes is not None, "FlextTypes não está disponível"

        # =========================================================================
        # FASE 2: Railway-Oriented Programming com FlextResult
        # =========================================================================

        # Cenário de sucesso - criação e encadeamento
        success_result = FlextResult[str].ok("dados_iniciais")
        assert success_result.success is True
        assert success_result.is_success is True
        assert success_result.is_failure is False
        assert success_result.value == "dados_iniciais"
        assert success_result.error is None

        # Teste de encadeamento de operações (pipeline)
        pipeline_result = (
            success_result.map(lambda x: x.upper())
            .map(lambda x: f"processado_{x}")
            .map(lambda x: x.replace("_", "-"))
        )

        assert pipeline_result.success is True
        assert pipeline_result.value == "processado-DADOS-INICIAIS"

        # Cenário de falha
        failure_result = FlextResult[str].fail("erro_de_processamento")
        assert failure_result.success is False
        assert failure_result.is_failure is True
        assert failure_result.error == "erro_de_processamento"

        # Em falha, `.value_or_none` retorna None para inspeção segura
        assert failure_result.value_or_none is None

        # Teste de flat_map para operações que podem falhar
        def operacao_que_pode_falhar(data: str) -> FlextResult[str]:
            if "invalido" in data:
                return FlextResult[str].fail("dados_invalidos")
            return FlextResult[str].ok(f"validado_{data}")

        flat_map_success = success_result.flat_map(operacao_que_pode_falhar)
        assert flat_map_success.success is True
        assert flat_map_success.value == "validado_dados_iniciais"

        invalid_data = FlextResult[str].ok("dados_invalido")
        flat_map_failure = invalid_data.flat_map(operacao_que_pode_falhar)
        assert flat_map_failure.success is False
        assert flat_map_failure.error == "dados_invalidos"

        # =========================================================================
        # FASE 3: Sistema hierárquico de constantes
        # =========================================================================

        # Testar acesso às constantes hierárquicas
        timeout_default = FlextConstants.Defaults.TIMEOUT
        assert isinstance(timeout_default, int)
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

        # =========================================================================
        # FASE 4: Sistema de exceções estruturado
        # =========================================================================

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

        # =========================================================================
        # FASE 5: Utilitários e funções auxiliares
        # =========================================================================

        # Geração de UUID
        generated_uuid = FlextUtilities.Generators.generate_uuid()
        assert isinstance(generated_uuid, str)
        assert len(generated_uuid) == 36  # Formato padrão UUID

        # Verificar que é um UUID válido
        uuid_obj = uuid.UUID(generated_uuid)
        assert str(uuid_obj) == generated_uuid

        # Geração de timestamp
        timestamp = FlextUtilities.Generators.generate_iso_timestamp()
        assert isinstance(timestamp, str)
        assert len(timestamp) > 0

        # Conversão segura de tipos
        safe_int_success = FlextUtilities.Conversions.safe_int("42")
        assert safe_int_success == 42

        safe_int_failure = FlextUtilities.Conversions.safe_int(
            "not_a_number", default=-1,
        )
        assert (
            safe_int_failure == -1
        )  # Retorna default em caso de erro        # =========================================================================
        # FASE 6: Sistema de validação (FlextValidations was removed)
        # =========================================================================

        # =========================================================================
        # FASE 7: Sistema de container (Dependency Injection)
        # =========================================================================

        # Teste do sistema de container
        container = FlextContainer.get_global()

        # Registrar serviço
        register_result = container.register("test_service", "test_value")
        assert register_result.success is True

        # Recuperar serviço registrado
        retrieved_service_result = container.get("test_service")
        assert retrieved_service_result.success is True
        retrieved_service = retrieved_service_result.value
        assert retrieved_service == "test_value"

        # Teste de serviço não encontrado
        not_found_result = container.get("servico_inexistente")
        assert not_found_result.success is False
        assert not_found_result.error is not None

        # =========================================================================
        # FASE 9: Cenários de integração complexa
        # =========================================================================

        # Simulação de processamento de dados completo
        def processar_dados_usuario(
            dados: FlextTypes.Core.Headers,
        ) -> FlextResult[FlextTypes.Core.Headers]:
            """Função que simula processamento completo usando todo o sistema."""
            # Validar entrada
            if not dados:
                return FlextResult[FlextTypes.Core.Headers].fail(
                    "Dados não fornecidos",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

            # Processar dados
            dados_processados = {}

            # FlextValidations was completely removed - using direct validation
            if "email" in dados:
                email = dados["email"]
                # Simple email validation since FlextValidations was removed
                if "@" not in email or "." not in email:
                    return FlextResult[FlextTypes.Core.Headers].fail(
                        f"Email inválido: {email}",
                    )
                dados_processados["email"] = email

            # Gerar ID único
            dados_processados["id"] = FlextUtilities.Generators.generate_uuid()

            # Adicionar timestamp
            dados_processados["created_at"] = (
                FlextUtilities.Generators.generate_iso_timestamp()
            )

            return FlextResult[FlextTypes.Core.Headers].ok(dados_processados)

        # Teste do fluxo completo - sucesso
        dados_entrada = {"email": "usuario@teste.com"}
        resultado_processamento = processar_dados_usuario(dados_entrada)

        assert resultado_processamento.success is True
        dados_finais = resultado_processamento.value
        assert "email" in dados_finais
        assert "id" in dados_finais
        assert "created_at" in dados_finais
        assert dados_finais["email"] == "usuario@teste.com"

        # Teste do fluxo completo - falha
        dados_invalidos = {"email": "email_invalido"}
        resultado_falha = processar_dados_usuario(dados_invalidos)

        assert resultado_falha.success is False
        assert resultado_falha.error is not None
        assert "email inválido" in resultado_falha.error.lower()

        # =========================================================================
        # FASE 10: Verificação de tipos e protocolos
        # =========================================================================

        # Verificar que os tipos estão disponíveis
        assert hasattr(FlextTypes, "Core")
        assert hasattr(FlextTypes, "Config")
        assert hasattr(FlextTypes, "Result")
