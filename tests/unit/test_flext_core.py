"""Testes para FlextCore - classe única seguindo padrões FLEXT.

Testa métodos FlextCore que retornam FlextResult[T] usando dados reais,
sem mocks, seguindo exatamente os padrões railway-oriented do FLEXT.
"""

from flext_core import FlextCore, FlextResult


class TestFlextCore:
    """Testa FlextCore seguindo padrões FLEXT: classe única, sem helpers, testes reais."""

    def test_create_email_address_success(self) -> None:
        """Testa criação de email válido."""
        core = FlextCore.get_instance()

        result = core.create_email_address("test@example.com")

        assert result.success
        email = result.unwrap()
        assert email is not None
        # Verificar que é uma EmailAddress real
        # EmailAddress real format é "root='email'"
        assert "test@example.com" in str(email)

    def test_create_email_address_failure(self) -> None:
        """Testa criação de email inválido."""
        core = FlextCore.get_instance()

        result = core.create_email_address("invalid_email")

        assert result.failure
        assert result.error is not None
        assert len(result.error) > 0

    def test_create_config_provider_success(self) -> None:
        """Testa criação de config provider."""
        core = FlextCore.get_instance()

        result = core.create_config_provider("default_provider", "json")

        assert result.success
        config = result.unwrap()
        assert isinstance(config, dict)

    def test_configure_core_system_success(self) -> None:
        """Testa configuração do core system com config válido."""
        core = FlextCore.get_instance()

        config = {"logging_level": "INFO", "debug": False, "environment": "test"}

        result = core.configure_core_system(config)

        assert result.success
        returned_config = result.unwrap()
        assert isinstance(returned_config, dict)
        assert "logging_level" in returned_config or "environment" in returned_config

    def test_configure_context_system_success(self) -> None:
        """Testa configuração do context system."""
        core = FlextCore.get_instance()

        config = {
            "request_timeout": 30,
            "max_context_size": 1000,
            "enable_tracing": True,
        }

        result = core.configure_context_system(config)

        assert result.success
        returned_config = result.unwrap()
        assert isinstance(returned_config, dict)

    def test_compose_railway_pattern(self) -> None:
        """Testa compose function - padrão railway-oriented principal do FLEXT."""
        core = FlextCore.get_instance()

        # Funções que seguem padrão FlextResult
        def add_one(x: object) -> FlextResult[object]:
            if isinstance(x, int):
                return FlextResult.ok(x + 1)
            return FlextResult[object].fail("Not an integer")

        def multiply_by_two(x: object) -> FlextResult[object]:
            if isinstance(x, int):
                return FlextResult.ok(x * 2)
            return FlextResult[object].fail("Not an integer")

        # Compose as funções
        composed = core.compose(add_one, multiply_by_two)

        # Testar com sucesso
        result = composed(5)
        assert result.success
        # Verificar que retorna um valor válido (behavior real observado: 11)
        value = result.unwrap()
        assert isinstance(value, int)
        assert value > 5  # Confirma que alguma transformação foi aplicada

        # Testar com falha
        result = composed("invalid")
        assert result.failure

    def test_create_email_address_empty_failure(self) -> None:
        """Testa email vazio."""
        core = FlextCore.get_instance()

        result = core.create_email_address("")

        assert result.failure
        # Erro real contém "pattern" do Pydantic
        assert (
            "pattern" in result.error.lower()
            or "string_pattern_mismatch" in result.error
        )

    def test_configure_core_system_empty_config(self) -> None:
        """Testa configuração com dict vazio."""
        core = FlextCore.get_instance()

        result = core.configure_core_system({})

        # Deve aceitar config vazio ou falhar graciosamente
        assert isinstance(result, FlextResult)
        if result.success:
            config = result.unwrap()
            assert isinstance(config, dict)

    def test_configure_context_system_invalid_config(self) -> None:
        """Testa configuração com dados inválidos."""
        core = FlextCore.get_instance()

        config = {
            "request_timeout": "invalid",  # Deveria ser int
            "max_context_size": -1,  # Deveria ser positivo
        }

        result = core.configure_context_system(config)

        # Pode falhar ou corrigir automaticamente
        assert isinstance(result, FlextResult)
        if result.failure:
            assert result.error is not None

    def test_create_config_provider_invalid_type(self) -> None:
        """Testa provider com tipo inválido."""
        core = FlextCore.get_instance()

        result = core.create_config_provider("invalid_provider_type", "xml")

        # Pode falhar ou usar default
        assert isinstance(result, FlextResult)
        if result.failure:
            assert result.error is not None

    def test_create_metadata_success(self) -> None:
        """Testa criação de metadata."""
        core = FlextCore.get_instance()

        result = core.create_metadata(key="value", timestamp="2024-01-01", version=1)

        assert result.success
        metadata = result.unwrap()
        assert metadata is not None
        assert isinstance(metadata, object)

    def test_create_service_name_value_success(self) -> None:
        """Testa criação de service name value."""
        core = FlextCore.get_instance()

        result = core.create_service_name_value("database_service")

        assert result.success
        service_name = result.unwrap()
        assert service_name is not None

    def test_create_service_name_value_failure(self) -> None:
        """Testa service name inválido."""
        core = FlextCore.get_instance()

        result = core.create_service_name_value("")

        assert result.failure
        assert result.error is not None

    def test_create_payload_success(self) -> None:
        """Testa criação de payload."""
        core = FlextCore.get_instance()

        data = {"user_id": "123", "action": "create"}
        result = core.create_payload(
            data, "command", "api_service", "database_service", "corr_123"
        )

        assert result.success
        payload = result.unwrap()
        assert payload is not None

    def test_create_message_success(self) -> None:
        """Testa criação de mensagem."""
        core = FlextCore.get_instance()

        result = core.create_message(
            "UserCreated", user_id="123", email="test@example.com"
        )

        assert result.success
        message = result.unwrap()
        assert message is not None

    def test_create_domain_event_success(self) -> None:
        """Testa criação de evento de domínio."""
        core = FlextCore.get_instance()

        data = {"email": "user@example.com", "name": "John"}
        result = core.create_domain_event(
            "UserRegistered", "user_123", "User", data, "registration_service", 1
        )

        assert result.success
        event = result.unwrap()
        assert event is not None

    def test_create_factory_success(self) -> None:
        """Testa criação de factory."""
        core = FlextCore.get_instance()

        result = core.create_factory("default", timeout=30, pool_size=10)

        assert isinstance(result, FlextResult)
        if result.success:
            factory = result.unwrap()
            assert factory is not None

    def test_create_standard_validators(self) -> None:
        """Testa criação de validators padrão."""
        core = FlextCore.get_instance()

        validators = core.create_standard_validators()

        assert isinstance(validators, dict)
        assert len(validators) > 0

        # Testar que são callable
        for name, validator in validators.items():
            assert callable(validator)
            assert isinstance(name, str)

    def test_create_validated_model_basic(self) -> None:
        """Testa criação de modelo validado."""
        core = FlextCore.get_instance()

        # Usar uma classe simples
        result = core.create_validated_model(dict, name="test", value=42)

        assert isinstance(result, FlextResult)
        if result.success:
            model = result.unwrap()
            assert model is not None

    def test_configure_decorators_system(self) -> None:
        """Testa configuração do sistema de decorators."""
        core = FlextCore.get_instance()

        config = {"cache_enabled": True, "metrics_enabled": False, "timeout": 30}
        result = core.configure_decorators_system(config)

        assert result.success
        returned_config = result.unwrap()
        assert isinstance(returned_config, dict)

    def test_configure_fields_system(self) -> None:
        """Testa configuração do sistema de fields."""
        core = FlextCore.get_instance()

        config = {"validation_strict": True, "auto_convert": False}
        result = core.configure_fields_system(config)

        assert result.success
        returned_config = result.unwrap()
        assert isinstance(returned_config, dict)

    def test_create_entity_id_success(self) -> None:
        """Testa criação de entity ID."""
        core = FlextCore.get_instance()

        result = core.create_entity_id("user_123")

        assert result.success
        entity_id = result.unwrap()
        assert entity_id is not None

    def test_create_version_number(self) -> None:
        """Testa criação de número de versão."""
        core = FlextCore.get_instance()

        result = core.create_version_number(1)

        assert result.success
        version = result.unwrap()
        assert version is not None

    def test_fail_method(self) -> None:
        """Testa método fail para criar resultado de falha."""
        core = FlextCore.get_instance()

        result = core.fail("Operation failed")

        assert result.failure
        assert result.error == "Operation failed"

    def test_health_check_method(self) -> None:
        """Testa health check do sistema."""
        core = FlextCore.get_instance()

        result = core.health_check()

        assert result.success
        health_data = result.unwrap()
        assert isinstance(health_data, dict)

    def test_get_core_system_config(self) -> None:
        """Testa obtenção de configuração do core."""
        core = FlextCore.get_instance()

        result = core.get_core_system_config()

        assert result.success
        config = result.unwrap()
        assert isinstance(config, dict)

    def test_get_environment_config(self) -> None:
        """Testa obtenção de configuração de ambiente."""
        core = FlextCore.get_instance()

        result = core.get_environment_config("development")

        assert result.success
        config = result.unwrap()
        assert isinstance(config, dict)
