# FlextCore Testing Patterns Documentation

Esta documentação apresenta os padrões de teste avançados implementados no FlextCore, demonstrando como usar as bibliotecas de suporte criadas para testing abrangente, moderno e eficiente.

## 📚 Bibliotecas de Suporte Criadas

### 1. `tests/support/domain_factories.py`
**Factory patterns avançados com factory_boy**
- `UserDataFactory`: Geração de dados de usuário realistas
- `FlextResultFactory`: Factory para objetos FlextResult  
- `ConfigDataFactory`: Factory para dados de configuração
- Integração com Faker para dados realistas
- Patterns: LazyAttribute, SubFactory, Trait, FuzzyChoice

### 2. `tests/support/performance_utils.py`
**Testing de performance e análise de complexidade**
- `PerformanceProfiler`: Profiling de memória e tempo
- `ComplexityAnalyzer`: Análise de complexidade algorítmica 
- `StressTestRunner`: Testes de carga e endurance
- `BenchmarkUtils`: Utilities para pytest-benchmark
- Detecção automática de padrões O(n), O(n²), etc.

### 3. `tests/support/async_utils.py`
**Testing assíncrono avançado**
- `AsyncTestUtils`: Utilities para testes async
- Timeout management e retry patterns
- Concurrency testing com semáforos
- Delay simulation e timing control
- Integration com pytest-asyncio

### 4. `tests/support/http_utils.py`
**Testing HTTP e API**
- `HTTPTestUtils`: Utilities para pytest-httpx
- Request/response mocking avançado
- API client testing patterns
- Validation de headers e payloads
- Response timing simulation

### 5. `tests/support/factory_boy_factories.py`
**Factory_boy patterns abrangentes**
- `UserFactory`, `ConfigFactory`: Factories básicas
- `EdgeCaseGenerators`: Geração de casos extremos
- `create_validation_test_cases()`: Casos de teste de validação
- Unicode, boundary values, malformed data

### 6. `tests/support/hypothesis_utils.py` ⭐ **NOVO**
**Property-based testing avançado**
- `FlextStrategies`: Estratégias customizadas para tipos Flext
- `BusinessDomainStrategies`: Estratégias para domínio de negócio
- `EdgeCaseStrategies`: Casos extremos e boundary conditions
- `PerformanceStrategies`: Dados para testing de performance
- `CompositeStrategies`: Estratégias compostas complexas

### 7. `tests/support/test_patterns.py` ⭐ **NOVO**
**Patterns de teste avançados**
- `TestDataBuilder`: Builder pattern para dados de teste
- `GivenWhenThenBuilder`: Given-When-Then scenarios
- `ParameterizedTestBuilder`: Parametrização automática
- `TestAssertionBuilder`: Assertions fluentes
- `TestSuiteBuilder`: Organização de suites de teste

## 🎯 Padrões de Teste Implementados

### 1. Railway-Oriented Programming Testing

```python
def test_railway_pattern_comprehensive(self, flext_result_factory):
    """Test FlextResult railway patterns."""
    # Create success result
    success_result = flext_result_factory.build_success("test_data")
    
    # Chain operations
    result = (success_result
        .map(lambda x: x.upper())
        .flat_map(lambda x: FlextResult.ok(f"processed_{x}"))
        .map(lambda x: len(x))
    )
    
    assert result.success
    assert isinstance(result.value, int)
```

### 2. Property-Based Testing

```python
from tests.support.hypothesis_utils import FlextStrategies, PropertyTestHelpers

@given(FlextStrategies.emails())
def test_email_validation_properties(self, email):
    """Property-based email validation testing."""
    assume(PropertyTestHelpers.assume_valid_email(email))
    
    result = validate_email(email)
    
    # Properties that must always hold
    assert result.success
    assert "@" in email
    assert len(email) > 3
```

### 3. Performance and Complexity Analysis

```python
def test_complexity_analysis(self):
    """Test algorithmic complexity analysis."""
    analyzer = ComplexityAnalyzer()
    
    def linear_operation(size):
        return list(range(size))
    
    result = analyzer.measure_complexity(
        linear_operation, 
        input_sizes=[100, 200, 400, 800],
        operation_name="list_creation"
    )
    
    assert result["complexity_analysis"]["pattern"] in ["linear", "unknown"]
```

### 4. Stress and Load Testing

```python
def test_stress_testing(self):
    """Demonstrate stress testing patterns."""
    stress_runner = StressTestRunner()
    
    def operation_under_test():
        return process_data({"key": "value"})
    
    # Load test
    load_result = stress_runner.run_load_test(
        operation_under_test,
        iterations=1000,
        operation_name="data_processing"
    )
    
    # Endurance test  
    endurance_result = stress_runner.run_endurance_test(
        operation_under_test,
        duration_seconds=10.0,
        operation_name="sustained_processing"
    )
    
    assert load_result["failure_rate"] < 0.01  # Less than 1% failures
    assert endurance_result["operations_per_second"] > 100
```

### 5. Given-When-Then Pattern

```python
def test_given_when_then_scenario(self):
    """Demonstrate Given-When-Then testing."""
    scenario = (
        GivenWhenThenBuilder("user_registration")
        .given("a new user with valid email", email="test@example.com")
        .given("the email is not already registered", unique=True)
        .when("the user attempts to register", action="register")
        .then("the registration should succeed", success=True)
        .then("the user should receive a confirmation", confirmation=True)
        .with_tag("integration")
        .with_priority("high")
        .build()
    )
    
    # Execute scenario
    execute_scenario(scenario)
```

### 6. Builder Pattern for Test Data

```python
def test_builder_pattern_advanced(self):
    """Demonstrate advanced builder pattern."""
    test_data = (
        FlextTestBuilder()
        .with_id("test_123")
        .with_correlation_id("corr_456") 
        .with_user_data("John Doe", "john@example.com")
        .with_timestamp()
        .with_validation_rules()
        .build()
    )
    
    assert test_data["id"] == "test_123"
    assert "created_at" in test_data
```

### 7. Fluent Assertions

```python
def test_fluent_assertions(self):
    """Demonstrate fluent assertion pattern."""
    test_data = ["apple", "banana", "cherry"]
    
    TestAssertionBuilder(test_data) \
        .is_not_none() \
        .has_length(3) \
        .contains("banana") \
        .satisfies(lambda x: all(isinstance(item, str) for item in x),
                  "all items should be strings") \
        .assert_all()
```

### 8. Parametrized Testing Advanced

```python
def test_parametrized_builder(self):
    """Demonstrate parametrized test builder."""
    param_builder = ParameterizedTestBuilder("email_validation")
    
    param_builder.add_success_cases([
        {"email": "test@example.com", "expected": True},
        {"email": "user@domain.org", "expected": True}
    ])
    
    param_builder.add_failure_cases([
        {"email": "invalid-email", "expected": False},
        {"email": "@domain.com", "expected": False}
    ])
    
    # Use with pytest.mark.parametrize
    params = param_builder.build_pytest_params()
    test_ids = param_builder.build_test_ids()
```

## 🔧 Pytest Plugins Utilizados

### Plugins Essenciais (14+)
- **pytest-asyncio**: Async testing patterns
- **pytest-benchmark**: Performance benchmarking
- **pytest-httpx**: HTTP mocking
- **pytest-mock**: Advanced mocking
- **pytest-xdist**: Parallel execution
- **pytest-cov**: Coverage reporting
- **pytest-randomly**: Random test order
- **pytest-clarity**: Better assertion output
- **pytest-sugar**: Beautiful test output
- **pytest-deadfixtures**: Dead fixture detection
- **pytest-env**: Environment management
- **pytest-timeout**: Test timeout control
- **factory_boy**: Data factory patterns
- **hypothesis**: Property-based testing

### Configuração no conftest.py

```python
pytest_plugins = [
    "pytest_asyncio",
    "pytest_benchmark", 
    "pytest_httpx",
    "pytest_mock",
    "pytest_xdist",
    "pytest_cov",
    "pytest_randomly",
    "pytest_clarity",
    "pytest_sugar",
    "pytest_deadfixtures",
    "pytest_env",
    "pytest_timeout",
]
```

## 📊 Markers de Teste Organizacionais

### Markers Principais
- `@pytest.mark.unit`: Testes unitários
- `@pytest.mark.integration`: Testes de integração
- `@pytest.mark.performance`: Testes de performance
- `@pytest.mark.slow`: Testes demorados
- `@pytest.mark.core`: Testes do core framework
- `@pytest.mark.ddd`: Testes Domain-driven design
- `@pytest.mark.architecture`: Testes arquiteturais
- `@pytest.mark.boundary`: Testes de boundary
- `@pytest.mark.error_path`: Cenários de erro
- `@pytest.mark.happy_path`: Cenários de sucesso

### Exemplo de Uso

```python
@pytest.mark.unit
@pytest.mark.core
@pytest.mark.performance
def test_flext_result_performance(self, benchmark):
    """Performance test for FlextResult operations."""
    def create_results():
        return [FlextResult.ok(i) for i in range(1000)]
    
    results = benchmark(create_results)
    assert len(results) == 1000
```

## 🎨 Patterns Arquiteturais

### 1. Arrange-Act-Assert (AAA)

```python
def test_aaa_pattern(self):
    # Arrange
    user_data = {"name": "John", "email": "john@example.com"}
    
    # Act
    result = create_user(user_data)
    
    # Assert
    assert result.success
    assert result.value["name"] == "John"
```

### 2. Given-When-Then (BDD)

```python
def test_bdd_pattern(self):
    # Given
    scenario = create_registration_scenario()
    
    # When  
    result = execute_registration(scenario.given)
    
    # Then
    assert_registration_success(result, scenario.then)
```

### 3. Builder Pattern

```python
def test_builder_pattern(self):
    user = (UserBuilder()
        .with_name("John")
        .with_email("john@example.com")
        .with_validation()
        .build())
    
    assert user["name"] == "John"
```

### 4. Factory Pattern

```python
def test_factory_pattern(self, user_factory):
    users = user_factory.build_batch(10)
    assert len(users) == 10
    assert all("email" in user for user in users)
```

## 🚀 Comandos de Execução

### Executar Testes por Categoria

```bash
# Testes unitários apenas
pytest -m unit

# Testes de performance
pytest -m performance --benchmark-only

# Testes sem os lentos
pytest -m "not slow"

# Testes core com coverage
pytest -m core --cov=src/flext_core

# Execução paralela
pytest -n auto tests/unit/

# Testes específicos com debug
pytest tests/unit/test_result.py::TestFlextResult::test_map -xvs
```

### Executar com Profiles Específicos

```bash
# Profile de desenvolvimento (rápido)
pytest -m "unit and not slow" --tb=short -q

# Profile de CI/CD (completo)
pytest --cov=src/flext_core --cov-fail-under=75 --tb=short

# Profile de performance
pytest -m performance --benchmark-only --benchmark-warmup=on

# Profile de stress testing  
pytest -m "performance or stress" --timeout=300
```

## 💡 Melhores Práticas

### 1. Organização de Testes
- Use a estrutura `/tests/unit/test_[módulo].py`
- Agrupe testes relacionados em classes
- Use markers para categorização
- Mantenha testes independentes

### 2. Nomenclatura
- `test_[feature]_[scenario]()` para métodos
- `Test[Component][Aspect]` para classes
- Descrições claras e específicas
- IDs únicos para casos parametrizados

### 3. Dados de Teste
- Use factories em vez de dados hardcoded
- Aplique property-based testing para edge cases
- Isole dados de teste entre execuções
- Use builders para dados complexos

### 4. Performance
- Benchmark operações críticas
- Analise complexidade algorítmica
- Execute stress testing em componentes de alto volume
- Monitor memory usage em operações grandes

### 5. Assertions
- Use assertions fluentes para verificações complexas
- Agrupe assertions relacionadas
- Forneça mensagens de erro descritivas
- Verifique tanto sucesso quanto falha

## 📈 Métricas e Relatórios

### Coverage Targets
- **Minimum**: 75% line coverage
- **Target**: 85% line coverage  
- **Goal**: 90%+ line coverage para módulos core

### Performance Benchmarks
- Operações básicas: < 1ms
- Operações complexas: < 100ms
- Bulk operations: < 1s para 1000 items
- Memory usage: < 50MB para operações típicas

### Quality Gates
- 0 linting errors (ruff)
- 0 type errors (mypy strict)
- 75%+ test coverage
- 0 security vulnerabilities (bandit)
- All tests passing

## 🎯 Exemplo Completo

```python
"""Exemplo completo demonstrando todos os patterns."""

import pytest
from hypothesis import given
from tests.support.hypothesis_utils import CompositeStrategies
from tests.support.performance_utils import StressTestRunner
from tests.support.test_patterns import (
    FlextTestBuilder, 
    GivenWhenThenBuilder,
    TestAssertionBuilder
)

@pytest.mark.unit
@pytest.mark.core
class TestCompleteExample:
    """Demonstração completa de patterns de teste."""
    
    def test_comprehensive_scenario(self, user_data_factory):
        """Cenário abrangente usando múltiplos patterns."""
        
        # 1. Builder pattern para dados de teste
        test_data = (
            FlextTestBuilder()
            .with_id("comprehensive_test")
            .with_user_data("Jane Doe", "jane@example.com")
            .with_validation_rules()
            .build()
        )
        
        # 2. Given-When-Then scenario
        scenario = (
            GivenWhenThenBuilder("user_processing")
            .given("valid user data", data=test_data)
            .when("user is processed", action="process")
            .then("processing succeeds", success=True)
            .build()
        )
        
        # 3. Execute operation
        result = process_user(scenario.given["data"])
        
        # 4. Fluent assertions
        TestAssertionBuilder(result) \
            .is_not_none() \
            .satisfies(lambda x: x.success, "should be successful") \
            .satisfies(lambda x: "user_id" in x.value, "should have user_id") \
            .assert_all()
    
    @given(CompositeStrategies.user_profiles())
    def test_property_based_validation(self, profile):
        """Property-based testing example."""
        result = validate_user_profile(profile)
        
        # Properties that must always hold
        if all(field in profile for field in ["name", "email"]):
            assert result.success
        else:
            assert result.failure
    
    def test_performance_analysis(self, benchmark):
        """Performance testing example."""
        def operation_under_test():
            return [process_item(i) for i in range(100)]
        
        results = benchmark(operation_under_test)
        assert len(results) == 100
    
    def test_stress_scenario(self):
        """Stress testing example."""
        stress_runner = StressTestRunner()
        
        result = stress_runner.run_load_test(
            lambda: simple_operation(),
            iterations=1000,
            operation_name="simple_ops"
        )
        
        assert result["failure_rate"] < 0.01
        assert result["operations_per_second"] > 500
```

---

## 🏆 Conclusão

Esta documentação apresenta um framework de testing abrangente e moderno para FlextCore, implementando:

✅ **7 bibliotecas de suporte** especializadas  
✅ **14+ pytest plugins** extensively utilizados  
✅ **10+ patterns de teste** avançados  
✅ **Property-based testing** com Hypothesis  
✅ **Performance e stress testing** automatizado  
✅ **Patterns arquiteturais** (Builder, Given-When-Then, AAA)  
✅ **Fluent assertions** e parametrização avançada  
✅ **Documentation completa** com exemplos práticos  

O resultado é uma infraestrutura de testing que demonstra **SOLID principles**, **DDD patterns**, **modern Python 3.13+**, e **comprehensive automation** - estabelecendo um novo padrão para testing em projetos Python enterprise.