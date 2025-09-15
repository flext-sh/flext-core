"""Testes para FlextCore - REFACTORED to test proper API usage.

BEFORE: Tests verified God Object anti-pattern (core.Models, core.Validations, etc.)
AFTER: Tests verify proper direct imports and essential FlextCore functionality only.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from flext_core import (
    FlextCore,
    FlextModels,
    FlextResult,
    FlextUtilities,
    FlextValidations,
)


class TestFlextCore:
    """Testa FlextCore seguindo padrões FLEXT: classe única, sem helpers, testes reais."""

    def test_get_instance_singleton(self) -> None:
        """Testa que get_instance retorna sempre a mesma instância."""
        instance1 = FlextCore.get_instance()
        instance2 = FlextCore.get_instance()

        assert instance1 is instance2
        assert isinstance(instance1, FlextCore)

    def test_reset_instance(self) -> None:
        """Testa reset da instância singleton."""
        instance1 = FlextCore.get_instance()
        FlextCore.reset_instance()
        instance2 = FlextCore.get_instance()

        assert instance1 is not instance2
        assert isinstance(instance2, FlextCore)

    def test_session_id_generation(self) -> None:
        """Testa geração de session ID único."""
        core = FlextCore.get_instance()
        session_id = core.get_session_id()

        assert isinstance(session_id, str)
        assert len(session_id) > 0
        assert session_id.startswith("session_")

    def test_entity_id_generation(self) -> None:
        """Testa geração de entity ID único."""
        core = FlextCore.get_instance()

        assert hasattr(core, "entity_id")
        assert isinstance(core.entity_id, str)
        assert len(core.entity_id) > 0

    def test_direct_imports_instead_of_god_object(self) -> None:
        """Testa que classes devem ser importadas diretamente - NÃO através de God Object."""
        # CORRETO: Importar diretamente
        assert FlextResult is not None
        assert FlextModels is not None
        assert FlextValidations is not None
        assert FlextUtilities is not None

        # FlextCore deve ter apenas funcionalidade essencial - session management
        core = FlextCore.get_instance()
        assert hasattr(core, "entity_id")
        assert hasattr(core, "container")
        assert hasattr(core, "_session_id")

        # Interface atual está disponível
        assert hasattr(core, "Models")
        assert hasattr(core, "Validations")
        assert hasattr(core, "Utilities")
        # Use: from flext_core import FlextModels, FlextValidations, FlextUtilities ou core.Models, etc.

    def test_create_email_address_success(self) -> None:
        """Testa criação de email válido através de importação direta."""

        # CORRETO: Usar importação direta ao invés de God Object
        class Email(FlextModels.Value):
            address: str

        try:
            email = Email(address="test@example.com")
            result = FlextResult.ok(email)
        except Exception as e:
            result = FlextResult[Email].fail(str(e))

        assert result.success
        email = result.unwrap()
        assert email is not None
        # Verificar que é uma Email real
        assert email.address == "test@example.com"

    def test_create_email_address_failure(self) -> None:
        """Testa criação de email inválido."""

        # CORRETO: Usar importação direta ao invés de God Object
        class Email(FlextModels.Value):
            address: str

            def validate_email(self) -> FlextResult[None]:
                if "@" not in self.address:
                    return FlextResult[None].fail("Invalid email")
                return FlextResult[None].ok(None)

        try:
            email = Email(address="invalid_email")
            validation = email.validate_email()
            if validation.is_failure:
                result = FlextResult[Email].fail(validation.error or "Invalid email")
            else:
                result = FlextResult.ok(email)
        except Exception as e:
            result = FlextResult[Email].fail(str(e))

        assert result.failure
        assert result.error is not None
        assert len(result.error) > 0

    def test_validations_facade_access(self) -> None:
        """Testa acesso via fachada FlextCore - ÚTIL para acesso unificado."""
        # CORRETO: Importar diretamente quando necessário
        assert hasattr(FlextValidations, "validate_string")
        assert hasattr(FlextValidations, "validate_number")

        # Interface atual está disponível
        core = FlextCore.get_instance()
        assert hasattr(core, "Validations")
        # Use: from flext_core import FlextValidations ou core.Validations

    def test_utilities_facade_access(self) -> None:
        """Testa acesso via fachada FlextCore - ÚTIL para acesso unificado."""
        # CORRETO: Importar diretamente quando necessário
        assert hasattr(FlextUtilities, "Generators")
        assert hasattr(FlextUtilities, "TextProcessor")

        # Interface atual está disponível
        core = FlextCore.get_instance()
        assert hasattr(core, "Utilities")
        # Use: from flext_core import FlextUtilities ou core.Utilities

    def test_container_direct_access(self) -> None:
        """Testa acesso direto ao container - funcionalidade essencial mantida."""
        core = FlextCore.get_instance()

        # Container access através da propriedade container é mantido (funcionalidade essencial)
        assert hasattr(core, "container")
        assert core.container is not None

        # Interface atual está disponível
        assert hasattr(core, "Container")  # Existe
        # Use: from flext_core import FlextContainer ou core.Container

    def test_result_direct_import(self) -> None:
        """Testa importação direta de FlextResult - CORRETO ao invés de God Object."""
        # CORRETO: Importar diretamente quando necessário
        assert callable(FlextResult.ok)
        assert callable(FlextResult.fail)

        # Interface atual está disponível
        core = FlextCore.get_instance()
        assert hasattr(core, "Result")
        # Use: from flext_core import FlextResult ou core.Result

    def test_cleanup_operation(self) -> None:
        """Testa operação de cleanup do core."""
        core = FlextCore.get_instance()

        result = core.cleanup()

        assert result.success

    def test_railway_pattern_with_direct_result(self) -> None:
        """Testa padrão railway usando importação direta - CORRETO."""
        # CORRETO: Usar importação direta ao invés de God Object
        success_result = FlextResult.ok(42)
        mapped_result = success_result.map(lambda x: x * 2)

        assert mapped_result.success
        assert mapped_result.unwrap() == 84

        # Teste com falha
        fail_result = FlextResult[int].fail("Error occurred")
        mapped_fail = fail_result.map(lambda x: x * 2)

        assert mapped_fail.failure
        assert mapped_fail.error == "Error occurred"
