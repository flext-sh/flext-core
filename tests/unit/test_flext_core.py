"""Testes para FlextCore.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from flext_core import FlextCore, FlextResult


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

    def test_direct_access_to_classes(self) -> None:
        """Testa acesso direto às classes aninhadas - padrão FLEXT."""
        core = FlextCore.get_instance()

        # Verificar que todas as classes estão disponíveis
        assert hasattr(core, "Models")
        assert hasattr(core, "Validations")
        assert hasattr(core, "Utilities")
        assert hasattr(core, "Container")
        assert hasattr(core, "Result")
        assert hasattr(core, "Config")
        assert hasattr(core, "Logger")
        assert hasattr(core, "Constants")

    def test_create_email_address_success(self) -> None:
        """Testa criação de email válido através das classes aninhadas."""
        core = FlextCore.get_instance()

        # Use a classe aninhada Models para criar EmailAddress
        try:
            email = core.Models.EmailAddress("test@example.com")
            result = FlextResult.ok(email)
        except Exception as e:
            result = FlextResult[core.Models.EmailAddress].fail(str(e))

        assert result.success
        email = result.unwrap()
        assert email is not None
        # Verificar que é uma EmailAddress real
        assert "test@example.com" in str(email)

    def test_create_email_address_failure(self) -> None:
        """Testa criação de email inválido."""
        core = FlextCore.get_instance()

        # Use a classe aninhada Models para tentar criar EmailAddress inválido
        try:
            email = core.Models.EmailAddress("invalid_email")
            result = FlextResult.ok(email)
        except Exception as e:
            result = FlextResult[core.Models.EmailAddress].fail(str(e))

        assert result.failure
        assert result.error is not None
        assert len(result.error) > 0

    def test_validations_access(self) -> None:
        """Testa acesso à classe Validations através do core."""
        core = FlextCore.get_instance()

        # Validations deve estar disponível
        assert hasattr(core, "Validations")
        assert hasattr(core.Validations, "validate_string")
        assert hasattr(core.Validations, "validate_number")

    def test_utilities_access(self) -> None:
        """Testa acesso à classe Utilities através do core."""
        core = FlextCore.get_instance()

        # Utilities deve estar disponível
        assert hasattr(core, "Utilities")

    def test_container_access(self) -> None:
        """Testa acesso ao Container através do core."""
        core = FlextCore.get_instance()

        # Container deve estar disponível
        assert hasattr(core, "Container")
        assert hasattr(core.Container, "get_global")

    def test_result_access(self) -> None:
        """Testa acesso à classe Result através do core."""
        core = FlextCore.get_instance()

        # Result deve estar disponível
        assert hasattr(core, "Result")
        assert callable(core.Result.ok)
        assert callable(core.Result.fail)

    def test_cleanup_operation(self) -> None:
        """Testa operação de cleanup do core."""
        core = FlextCore.get_instance()

        result = core.cleanup()

        assert result.success

    def test_railway_pattern_with_result_class(self) -> None:
        """Testa padrão railway usando a classe Result acessível."""
        core = FlextCore.get_instance()

        # Usar FlextResult para railway pattern
        success_result = core.Result.ok(42)
        mapped_result = success_result.map(lambda x: x * 2)

        assert mapped_result.success
        assert mapped_result.unwrap() == 84

        # Teste com falha
        fail_result = core.Result[int].fail("Error occurred")
        mapped_fail = fail_result.map(lambda x: x * 2)

        assert mapped_fail.failure
        assert mapped_fail.error == "Error occurred"
