"""Teste de integração completo para o sistema FLEXT com refatoração adequada.

Este teste único e abrangente valida que todo o sistema flext-core funciona
corretamente após wildcard imports, com foco em railway-oriented programming,
hierarquia de constantes, sistema de exceções e utilitários.
"""

from __future__ import annotations

import uuid

import pytest

from flext_core import (
    FlextConstants,
    FlextExceptions,
    FlextFields,
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
        6. Sistema de campos (FlextFields)
        7. Validação e configuração
        8. Cenários de erro e recuperação
        """
        # =========================================================================
        # FASE 1: Validação de imports e disponibilidade de módulos
        # =========================================================================

        # Verificar que todos os componentes principais estão disponíveis
        expected_modules = [
            "FlextResult",
            "FlextConstants",
            "FlextExceptions",
            "FlextUtilities",
            "FlextTypes",
            "FlextConfig",
            "FlextContainer",
            "FlextFields",
            "FlextValidation",
            "FlextCommands",
        ]

        current_globals = globals()
        for module_name in expected_modules:
            assert module_name in current_globals, (
                f"Módulo '{module_name}' não encontrado nos exports do wildcard"
            )
            module = current_globals[module_name]
            assert module is not None, f"Módulo '{module_name}' é None"

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

        # Verificar que não podemos acessar value em failure (proteção de tipo)
        with pytest.raises(
            TypeError, match="Attempted to access value on failed result"
        ):
            _ = failure_result.value

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

        # Constantes de erro com formato FLEXT_xxxx
        validation_error_code = FlextConstants.Errors.VALIDATION_ERROR
        assert isinstance(validation_error_code, str)
        assert validation_error_code.startswith("FLEXT_")
        assert "3001" in validation_error_code  # Código específico para validação

        # Constantes de mensagem
        success_message = FlextConstants.Messages.SUCCESS
        assert isinstance(success_message, str)
        assert len(success_message) > 0

        # Constantes de padrões (regex)
        email_pattern = FlextConstants.Patterns.EMAIL_PATTERN
        assert isinstance(email_pattern, str)
        assert "@" in email_pattern

        # =========================================================================
        # FASE 4: Sistema de exceções estruturado
        # =========================================================================

        # Teste da hierarquia de exceções
        validation_exception = FlextExceptions.ValidationError("campo_invalido")
        assert isinstance(validation_exception, Exception)
        assert isinstance(validation_exception, FlextExceptions.Base)

        # Verificar formato da mensagem de erro
        error_message = str(validation_exception)
        assert "[FLEXT_3001]" in error_message
        assert "campo_invalido" in error_message

        # Teste de outras exceções da hierarquia
        operation_exception = FlextExceptions.OperationError("operacao_falhada")
        assert isinstance(operation_exception, FlextExceptions.Base)
        assert "operacao_falhada" in str(operation_exception)

        # Verificar que as exceções seguem a hierarquia correta
        assert issubclass(FlextExceptions.ValidationError, FlextExceptions.Base)
        assert issubclass(FlextExceptions.OperationError, FlextExceptions.Base)

        # =========================================================================
        # FASE 5: Utilitários e funções auxiliares
        # =========================================================================

        # Geração de UUID
        generated_uuid = FlextUtilities.generate_uuid()
        assert isinstance(generated_uuid, str)
        assert len(generated_uuid) == 36  # Formato padrão UUID

        # Verificar que é um UUID válido
        uuid_obj = uuid.UUID(generated_uuid)
        assert str(uuid_obj) == generated_uuid

        # Geração de timestamp
        timestamp = FlextUtilities.generate_timestamp()
        assert isinstance(timestamp, str)
        assert len(timestamp) > 0

        # Conversão segura de tipos
        safe_int_success = FlextUtilities.safe_int("42")
        assert safe_int_success == 42

        safe_int_failure = FlextUtilities.safe_int("not_a_number", default=-1)
        assert (
            safe_int_failure == -1
        )  # Retorna default em caso de erro        # =========================================================================
        # FASE 6: Sistema de campos (FlextFields)
        # =========================================================================

        # Criação de campo string básico
        string_field = FlextFields.Core.StringField(
            "username", min_length=3, max_length=20, required=True
        )

        # Validação de campo - caso de sucesso
        valid_username = string_field.validate("joao123")
        assert valid_username.success is True
        assert valid_username.value == "joao123"

        # Validação de campo - caso de falha (muito curto)
        invalid_username = string_field.validate("jo")
        assert invalid_username.success is False
        assert "length" in invalid_username.error.lower()

        # Criação usando factory pattern
        email_field_result = FlextFields.Factory.create_field(
            "email", "user_email", required=True, description="Email do usuário"
        )
        assert email_field_result.success is True
        email_field = email_field_result.value

        # Validação de email
        valid_email = email_field.validate("user@example.com")
        assert valid_email.success is True

        invalid_email = email_field.validate("invalid-email")
        assert invalid_email.success is False

        # =========================================================================
        # FASE 7: Registro e gerenciamento de campos
        # =========================================================================

        # Teste do sistema de registro
        registry = FlextFields.Registry.FieldRegistry()

        # Registrar campo
        register_result = registry.register_field("user_name", string_field)
        assert register_result.success is True

        # Recuperar campo registrado
        retrieved_field_result = registry.get_field("user_name")
        assert retrieved_field_result.success is True
        retrieved_field = retrieved_field_result.value
        assert retrieved_field.name == "username"

        # Teste de campo não encontrado
        not_found_result = registry.get_field("campo_inexistente")
        assert not_found_result.success is False
        assert "not found" in not_found_result.error.lower()

        # =========================================================================
        # FASE 8: Validação múltipla e processamento de schema
        # =========================================================================

        # Criação de múltiplos campos
        integer_field = FlextFields.Core.IntegerField(
            "age", min_value=0, max_value=120, required=True
        )

        # Validação múltipla usando o sistema
        fields = [string_field, integer_field]
        values = {"username": "maria123", "age": 25}

        validation_result = FlextFields.Validation.validate_multiple_fields(
            fields, values
        )
        assert validation_result.success is True
        validated_data = validation_result.value
        assert validated_data["username"] == "maria123"
        assert validated_data["age"] == 25

        # Teste com dados inválidos
        invalid_values = {
            "username": "jo",  # Muito curto
            "age": 150,  # Muito alto
        }

        invalid_validation = FlextFields.Validation.validate_multiple_fields(
            fields, invalid_values
        )
        assert invalid_validation.success is False
        assert (
            "length" in invalid_validation.error.lower()
            or "value" in invalid_validation.error.lower()
        )

        # =========================================================================
        # FASE 9: Cenários de integração complexa
        # =========================================================================

        # Simulação de processamento de dados completo
        def processar_dados_usuario(
            dados: dict[str, str],
        ) -> FlextResult[dict[str, str]]:
            """Função que simula processamento completo usando todo o sistema."""
            # Validar entrada
            if not dados:
                return FlextResult[dict[str, str]].fail(
                    "Dados não fornecidos",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

            # Processar dados
            dados_processados = {}

            # Validar username
            if "username" in dados:
                username_result = string_field.validate(dados["username"])
                if not username_result.success:
                    return FlextResult[dict[str, str]].fail(
                        f"Username inválido: {username_result.error}"
                    )
                dados_processados["username"] = username_result.value

            # Gerar ID único
            dados_processados["id"] = FlextUtilities.generate_uuid()

            # Adicionar timestamp
            dados_processados["created_at"] = FlextUtilities.generate_timestamp()

            return FlextResult[dict[str, str]].ok(dados_processados)

        # Teste do fluxo completo - sucesso
        dados_entrada = {"username": "usuario_teste"}
        resultado_processamento = processar_dados_usuario(dados_entrada)

        assert resultado_processamento.success is True
        dados_finais = resultado_processamento.value
        assert "username" in dados_finais
        assert "id" in dados_finais
        assert "created_at" in dados_finais
        assert dados_finais["username"] == "usuario_teste"

        # Teste do fluxo completo - falha
        dados_invalidos = {"username": "u"}  # Muito curto
        resultado_falha = processar_dados_usuario(dados_invalidos)

        assert resultado_falha.success is False
        assert "username inválido" in resultado_falha.error.lower()

        # =========================================================================
        # FASE 10: Verificação de tipos e protocolos
        # =========================================================================

        # Verificar que os tipos estão disponíveis
        assert hasattr(FlextTypes, "ResultProtocol")
        assert hasattr(FlextTypes, "ConfigProtocol")
        assert hasattr(FlextTypes, "ContainerProtocol")

        # =========================================================================
        # CONCLUSÃO: Sistema totalmente integrado e funcional
        # =========================================================================

        print("✅ Teste de integração completo passou com sucesso!")
        print("✅ Wildcard imports funcionais")
        print("✅ Railway-oriented programming operacional")
        print("✅ Sistema hierárquico de constantes acessível")
        print("✅ Hierarquia de exceções estruturada")
        print("✅ Utilitários e funções auxiliares funcionais")
        print("✅ Sistema de campos FlextFields operacional")
        print("✅ Validação e processamento de dados funcionais")
        print("✅ Cenários de integração complexa validados")
        print("🎉 FLEXT Core está completamente funcional!")
