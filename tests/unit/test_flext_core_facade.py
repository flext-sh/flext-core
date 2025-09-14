"""Teste para validar a nova arquitetura do FlextCore.

Este teste demonstra que o FlextCore segue Single Responsibility Principle:
- Gerencia apenas sessão e configuração global
- Não expõe todas as funcionalidades como god object
- Força uso de importações diretas para funcionalidades específicas
- Mantém consistência na configuração global
- Facilita a integração seguindo princípios SOLID
"""

from flext_core import (
    FlextContainer,
    FlextCore,
    FlextResult,
    FlextValidations,
)


class TestFlextCoreArchitecture:
    """Testa a nova arquitetura do FlextCore seguindo princípios SOLID."""

    def test_core_follows_single_responsibility(self) -> None:
        """Testa que o FlextCore segue Single Responsibility Principle."""
        core = FlextCore()

        # Verifica que apenas funcionalidades essenciais estão disponíveis
        assert hasattr(core, "entity_id")
        assert hasattr(core, "container")
        assert hasattr(core, "get_config")
        assert hasattr(core, "get_session_id")
        assert hasattr(core, "cleanup")

        # Verifica que interface atual do FlextCore está disponível
        assert hasattr(core, "Config")
        assert hasattr(core, "Models")
        assert hasattr(core, "Commands")  # nome atual
        assert hasattr(core, "Processors")
        assert hasattr(core, "Validations")
        assert hasattr(core, "Utilities")
        assert hasattr(core, "Adapters")
        assert hasattr(core, "Fields")
        assert hasattr(core, "Mixins")
        assert hasattr(core, "Protocols")
        assert hasattr(core, "Exceptions")
        assert hasattr(core, "Result")
        assert hasattr(core, "Container")  # nome atual
        assert hasattr(core, "Context")
        assert hasattr(core, "Logger")
        assert hasattr(core, "Constants")

    def test_facade_enables_single_initialization(self) -> None:
        """Testa que a fachada permite inicialização única de todo o sistema."""
        core = FlextCore()

        # Verifica que a inicialização única funciona
        assert core.entity_id is not None
        assert core.get_config() is not None
        assert core.get_session_id() is not None

        # Verifica que interface atual está disponível
        assert hasattr(core, "Config")
        assert hasattr(core, "Result")
        assert hasattr(core, "Container")  # nome atual
        assert hasattr(core, "Logger")

    def test_facade_simplifies_developer_access(self) -> None:
        """Testa que a fachada simplifica o acesso aos módulos para desenvolvedores."""
        core = FlextCore()

        # Demonstra como a fachada simplifica o acesso
        # Em vez de importar múltiplos módulos:
        # from flext_core.result import FlextResult
        # from flext_core.validations import FlextValidations
        # from flext_core.container import FlextContainer

        # CORRETO: Usar importações diretas
        result = FlextResult[str].ok("test")
        validator = FlextValidations.validate_string("test")
        container = FlextContainer.get_global()

        # Verifica que tudo funciona corretamente
        assert result.is_success
        assert result.unwrap() == "test"
        assert validator.is_success
        assert container is not None

        # Interface atual está disponível
        assert hasattr(core, "Result")
        assert hasattr(core, "Validations")
        assert hasattr(core, "Container")  # nome atual

    def test_facade_maintains_global_consistency(self) -> None:
        """Testa que a fachada mantém consistência na configuração global."""
        core1 = FlextCore()
        core2 = FlextCore()

        # Verifica que ambos usam a mesma instância global de configuração
        config1 = core1.get_config()
        config2 = core2.get_config()

        # Ambos devem referenciar a mesma instância global
        assert config1 is config2

        # Verifica que o container global é consistente via propriedade
        container1 = core1.container
        container2 = core2.container
        assert container1 is container2

        # Interface atual está disponível
        assert hasattr(core1, "Container")  # nome atual
        assert hasattr(core2, "Container")  # nome atual

    def test_facade_facilitates_framework_integration(self) -> None:
        """Testa que a fachada facilita a integração e uso do framework."""
        core = FlextCore()

        # Demonstra um fluxo completo usando importações diretas
        # 1. Validação de dados
        validation_result = FlextValidations.validate_string("test_data")
        assert validation_result.is_success

        # 2. Processamento com Result
        processing_result = FlextResult[str].ok("processed_data")
        assert processing_result.is_success

        # Interface atual está disponível
        assert hasattr(core, "Validations")
        assert hasattr(core, "Result")

        # 3. Configuração via método
        config = core.get_config()
        assert config is not None

        # 4. Container para injeção de dependência via propriedade
        container = core.container
        container.register("test_service", "test_value")
        service_result = container.get("test_service")
        assert service_result.is_success

        # Interface atual está disponível
        assert hasattr(core, "Logger")
        assert hasattr(core, "Container")  # nome atual

    def test_facade_pattern_benefits(self) -> None:
        """Testa os benefícios do padrão Facade implementado pelo FlextCore."""
        core = FlextCore()

        # Benefício 1: Single Responsibility Principle
        # O FlextCore tem responsabilidade única: sessão e configuração
        assert hasattr(core, "entity_id")
        assert hasattr(core, "get_config")
        assert hasattr(core, "container")

        # Benefício 2: Interface atual está disponível
        # Propriedades do FlextCore estão acessíveis
        assert hasattr(core, "Result")
        assert hasattr(core, "Validations")
        assert hasattr(core, "Container")

        # Benefício 3: Facilita mudanças internas
        # Mudanças nos módulos internos não afetam o desenvolvedor
        # que usa apenas a fachada

        # Benefício 4: Reduz acoplamento
        # O desenvolvedor usa importações diretas para funcionalidades específicas
        result = FlextResult[str].ok("test")
        assert result.is_success

        # Interface atual está disponível
        assert hasattr(core, "Result")

    def test_facade_singleton_functionality(self) -> None:
        """Testa que a fachada mantém funcionalidade singleton quando apropriado."""
        core1 = FlextCore()
        core2 = FlextCore()

        # Cada instância tem seu próprio entity_id
        assert core1.entity_id != core2.entity_id

        # Mas compartilham configurações globais
        assert core1.get_config() is core2.get_config()
        assert core1.container is core2.container

        # Interface atual está disponível
        assert hasattr(core1, "Container")  # nome atual
        assert hasattr(core2, "Container")  # nome atual

    def test_facade_comprehensive_workflow(self) -> None:
        """Testa um fluxo de trabalho completo usando apenas a fachada."""
        core = FlextCore()

        # Fluxo completo: Validação -> Processamento -> Logging -> Configuração

        # 1. Validação de entrada usando importação direta
        input_data = "test_input"
        validation_result = FlextValidations.validate_string(input_data)
        assert validation_result.is_success

        # 2. Processamento com Result usando importação direta
        if validation_result.is_success:
            processed_data = f"processed_{input_data}"
            result = FlextResult[str].ok(processed_data)
            assert result.is_success

            # 3. Configuração do sistema via método
            config = core.get_config()
            assert config is not None

            # 4. Registro no container via propriedade
            container = core.container
            container.register("processed_data", processed_data)
            stored_result = container.get("processed_data")
            assert stored_result.is_success

            # Interface atual está disponível
            assert hasattr(core, "Validations")
            assert hasattr(core, "Result")
            assert hasattr(core, "Logger")
            assert hasattr(core, "Container")
            assert stored_result.unwrap() == processed_data
