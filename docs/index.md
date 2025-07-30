# FLEXT Core Library

**Biblioteca fundamental do ecossistema FLEXT - Fundação empresarial pronta para produção**

![Python Version](https://img.shields.io/badge/python-3.13+-blue.svg)
![Type Checking](https://img.shields.io/badge/mypy-strict-green.svg)
![Code Quality](https://img.shields.io/badge/lint-PEP8%20strict-green.svg)
![Test Coverage](https://img.shields.io/badge/coverage-90%25+-green.svg)

## 🎯 Visão Geral

FLEXT Core é a biblioteca fundamental que serve como base arquitetural para todo o ecossistema FLEXT - um framework empresarial moderno construído sobre Clean Architecture, Domain-Driven Design (DDD) e Python 3.13. Esta é uma biblioteca pura (sem CLI) que fornece os componentes fundamentais para 25+ projetos FLEXT.

### Características Principais

- **Python 3.13 only** com type hints modernos
- **Zero dependências externas** na runtime (biblioteca pura)
- **Clean Architecture + DDD** patterns
- **Qualidade empresarial** com padrões rigorosos
- **90%+ cobertura de testes** obrigatória
- **PEP8 strict compliance** (79 caracteres)
- **MyPy strict mode** com zero tolerância

## 🏗️ Arquitetura

```
FLEXT Core Architecture
├── Domain Layer (Entidades, Value Objects, Aggregates)
├── Application Layer (Services, Commands, Handlers)
├── Infrastructure Layer (Container DI, Configuration)
└── Patterns Layer (Validation, Logging, Fields)
```

### Componentes Principais

| Componente         | Descrição                                | Status     |
| ------------------ | ---------------------------------------- | ---------- |
| `FlextResult[T]`   | Sistema de tratamento de erros type-safe | ✅ Moderno |
| `FlextContainer`   | Injeção de dependência empresarial       | ✅ Moderno |
| `FlextCommand`     | Padrão Command com validação             | ✅ Moderno |
| `FlextHandler`     | Sistema de processamento de mensagens    | ✅ Moderno |
| `FlextValidator`   | Validação robusta com regras             | ✅ Moderno |
| `FlextEntity`      | Base para entidades DDD                  | ✅ Moderno |
| `FlextValueObject` | Value objects imutáveis                  | ✅ Moderno |

## 🚀 Início Rápido

### Instalação

```bash
# Via Poetry (recomendado)
poetry add flext-core

# Via pip
pip install flext-core
```

### Exemplo Básico

```python
from flext_core import FlextResult, FlextContainer

# 1. Tratamento de erros type-safe
def fetch_user(user_id: str) -> FlextResult[dict]:
    if not user_id:
        return FlextResult.fail("User ID é obrigatório")

    user_data = {"id": user_id, "name": "João"}
    return FlextResult.ok(user_data)

# 2. Injeção de dependência
container = FlextContainer()
result = container.register("user_service", UserService())

if result.is_success:
    service = container.get("user_service").data
    print(f"Serviço registrado: {service}")

# 3. Uso do resultado
user_result = fetch_user("123")
if user_result.is_success:
    print(f"Usuário: {user_result.data}")
else:
    print(f"Erro: {user_result.error}")
```

## 📚 Documentação

### Guias Essenciais

- **[Arquitetura](architecture/overview.md)** - Design e princípios fundamentais
- **[Instalação](getting-started/installation.md)** - Setup e configuração
- **[Guia Rápido](getting-started/quickstart.md)** - Primeiros passos
- **[Padrões](patterns/overview.md)** - Padrões de design implementados

### APIs e Referências

- **[API Core](api/core.md)** - FlextResult, FlextContainer, configuração
- **[API Patterns](api/patterns.md)** - Commands, Handlers, Validation
- **[API Domain](api/domain.md)** - Entities, Value Objects, Aggregates
- **[Migração](migration/guide.md)** - Migração de código legado

### Desenvolvimento

- **[Boas Práticas](development/best-practices.md)** - Padrões recomendados
- **[Contribuição](development/contributing.md)** - Como contribuir
- **[Testing](development/testing.md)** - Estratégias de teste
- **[Examples](examples/overview.md)** - Exemplos práticos

## 🎨 Padrões Fundamentais

### 1. FlextResult Pattern

```python
# Type-safe error handling sem exceções
def process_payment(amount: float) -> FlextResult[str]:
    if amount <= 0:
        return FlextResult.fail("Valor deve ser positivo")

    # Processar pagamento...
    return FlextResult.ok("payment_id_123")
```

### 2. Dependency Injection

```python
# Container moderno com type safety
container = FlextContainer()
container.register("database", DatabaseService())
container.register("email", EmailService())

# Resolução automática de dependências
app_service = container.get("app_service").data
```

### 3. Command Pattern

```python
# Commands com validação integrada
class CreateUserCommand(FlextCommand):
    def __init__(self, name: str, email: str):
        super().__init__()
        self.name = name
        self.email = email

    def validate(self) -> FlextResult[None]:
        if not self.name:
            return FlextResult.fail("Nome é obrigatório")
        return FlextResult.ok(None)
```

### 4. Domain-Driven Design

```python
# Entidades e Value Objects
class User(FlextEntity[UserId]):
    def __init__(self, user_id: UserId, name: str, email: Email):
        super().__init__(user_id)
        self.name = name
        self.email = email  # Value Object

    def change_email(self, new_email: Email) -> FlextResult[None]:
        # Lógica de domínio aqui
        self.email = new_email
        return FlextResult.ok(None)
```

## 🔧 Comandos de Desenvolvimento

### Quality Gates (OBRIGATÓRIO)

```bash
# Validação completa - TODOS devem passar
make validate              # pep8 + type-check + security + test

# Verificações essenciais
make check                 # lint + type-check + test
make lint                  # Ruff linting PEP8 strict
make type-check            # MyPy strict mode
make test                  # 90% coverage mínimo
```

### Setup e Desenvolvimento

```bash
make setup                 # Setup completo do ambiente
make install               # Instalar dependências
make dev-install           # Ambiente de desenvolvimento
make pre-commit            # Setup pre-commit hooks
```

## 📊 Qualidade e Padrões

### Padrões de Qualidade

- **Linha**: 79 caracteres máximo (PEP8 strict)
- **Docstrings**: Google style, 72 caracteres máximo
- **Type Hints**: Obrigatório em todas as funções
- **Cobertura**: 90% mínimo obrigatório
- **MyPy**: Strict mode com zero tolerância

### Compatibilidade

```python
# ✅ Moderno (use isto)
from flext_core import FlextContainer, FlextResult

# ⚠️ Legado (compatibilidade mantida)
from flext_core import DIContainer, FlextResult
```

## 🌟 Casos de Uso

### Para Desenvolvedores

- **APIs Robustas**: Tratamento de erro type-safe
- **Microserviços**: Arquitetura limpa e testável
- **Sistemas Complexos**: DDD patterns para modelagem

### Para Equipes

- **Padrões Consistentes**: Arquitetura unificada
- **Qualidade Garantida**: Quality gates obrigatórios
- **Colaboração**: Type safety para múltiplos desenvolvedores

### Para Empresas

- **Produção Ready**: Zero dependências, máxima estabilidade
- **Escalabilidade**: Clean Architecture patterns
- **Manutenibilidade**: Código autodocumentado e testado

## 🤝 Contribuição

1. **Fork** o repositório
2. **Clone** para desenvolvimento local
3. **Setup** environment: `make setup`
4. **Desenvolva** seguindo os padrões
5. **Teste** completamente: `make validate`
6. **Submit** Pull Request

### Padrões de Contribuição

- Código deve passar em `make validate`
- Cobertura de 90%+ obrigatória
- Documentação atualizada
- Type hints completos
- PEP8 strict compliance

## 📄 Licença

MIT License - veja [LICENSE](../LICENSE) para detalhes.

## 🔗 Links

- **Repositório**: [GitHub](https://github.com/flext/flext-core)
- **Documentação**: [Docs](https://docs.flext.dev)
- **Issues**: [GitHub Issues](https://github.com/flext/flext-core/issues)
- **PyPI**: [flext-core](https://pypi.org/project/flext-core)

---

**FLEXT Core** - A fundação sólida para sistemas empresariais modernos em Python 3.13+
