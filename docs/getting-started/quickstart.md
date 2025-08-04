# Quickstart - FLEXT Core

**Guia prático para começar rapidamente com FLEXT Core**

## 🚀 Primeiro Uso - 5 Minutos

### Instalação Rápida

```bash
# Instalar FLEXT Core
pip install flext-core

# Ou com Poetry
poetry add flext-core
```

### Hello World com FLEXT

```python
# hello_flext.py
from flext_core import FlextResult

def hello_world() -> FlextResult[str]:
    """Primeiro exemplo com FLEXT Core."""
    return FlextResult.ok("Hello, FLEXT World! 🚀")

# Executar
result = hello_world()
if result.success:
    print(result.data)  # Output: Hello, FLEXT World! 🚀
else:
    print(f"Erro: {result.error}")
```

## 📋 Conceitos Fundamentais

### 1. FlextResult - Error Handling Type-Safe

**O coração do FLEXT Core - substitui exceções por resultados explícitos.**

```python
from flext_core import FlextResult

def divide_numbers(a: float, b: float) -> FlextResult[float]:
    """Divisão segura com tratamento de erro."""
    if b == 0:
        return FlextResult.fail("Divisão por zero não permitida")

    result = a / b
    return FlextResult.ok(result)

# Uso seguro
result = divide_numbers(10, 2)
if result.success:
    print(f"Resultado: {result.data}")  # 5.0
else:
    print(f"Erro: {result.error}")

# Caso de erro
error_result = divide_numbers(10, 0)
print(error_result.is_failure)  # True
print(error_result.error)       # "Divisão por zero não permitida"
```

### 2. FlextContainer - Dependency Injection

**Container IoC type-safe para gerenciamento de dependências.**

```python
from flext_core import FlextContainer

# Criar container
container = FlextContainer()

# Registrar serviços
database_url = "postgresql://localhost/mydb"
container.register("database_url", database_url)

# Registrar classe de serviço
class EmailService:
    def send_email(self, to: str, subject: str) -> str:
        return f"Email enviado para {to}: {subject}"

email_service = EmailService()
container.register("email_service", email_service)

# Resolver dependências
db_result = container.get("database_url")
if db_result.success:
    print(f"Database: {db_result.data}")

email_result = container.get("email_service")
if email_result.success:
    service = email_result.data
    message = service.send_email("user@test.com", "Bem-vindo!")
    print(message)
```

### 3. Domain Entities - DDD Building Blocks

**Entidades de domínio com identidade única.**

```python
from flext_core import FlextEntity, FlextResult
from typing import NewType

# Value objects
UserId = NewType("UserId", str)
Email = NewType("Email", str)

class User(FlextEntity[UserId]):
    """Entidade User com regras de negócio."""

    def __init__(self, user_id: UserId, name: str, email: Email):
        super().__init__(user_id)
        self._name = name
        self._email = email
        self._is_active = True

    @property
    def name(self) -> str:
        return self._name

    @property
    def email(self) -> Email:
        return self._email

    @property
    def is_active(self) -> bool:
        return self._is_active

    def change_email(self, new_email: Email) -> FlextResult[None]:
        """Mudar email com validação de negócio."""
        if "@" not in new_email:
            return FlextResult.fail("Email deve conter @")

        if new_email == self._email:
            return FlextResult.fail("Novo email deve ser diferente do atual")

        self._email = new_email
        return FlextResult.ok(None)

    def deactivate(self) -> FlextResult[None]:
        """Desativar usuário."""
        if not self._is_active:
            return FlextResult.fail("Usuário já está inativo")

        self._is_active = False
        return FlextResult.ok(None)

# Criar e usar entidade
user_id = UserId("user_123")
user = User(user_id, "João Silva", Email("joao@test.com"))

print(f"Usuário: {user.name}")
print(f"ID: {user.id}")
print(f"Email: {user.email}")

# Mudar email
email_result = user.change_email(Email("joao.silva@newcompany.com"))
if email_result.success:
    print(f"Email atualizado: {user.email}")
```

## 🎯 Padrões Essenciais

### 1. Command Pattern - CQRS

**Commands representam intenções de mudança no sistema.**

```python
from flext_core.patterns import FlextCommand, FlextCommandHandler
from flext_core import FlextResult

# Command - Intenção de criar usuário
class CreateUserCommand(FlextCommand):
    def __init__(self, name: str, email: str):
        super().__init__()
        self.name = name
        self.email = email

    def validate(self) -> FlextResult[None]:
        """Validação de entrada do command."""
        if not self.name.strip():
            return FlextResult.fail("Nome é obrigatório")

        if "@" not in self.email:
            return FlextResult.fail("Email deve ser válido")

        if len(self.name) < 2:
            return FlextResult.fail("Nome deve ter pelo menos 2 caracteres")

        return FlextResult.ok(None)

# Handler - Processa o command
class CreateUserHandler(FlextCommandHandler[CreateUserCommand, User]):
    def __init__(self, container: FlextContainer):
        super().__init__()
        self._container = container

    def can_handle(self, command) -> bool:
        return isinstance(command, CreateUserCommand)

    def handle(self, command: CreateUserCommand) -> FlextResult[User]:
        """Processar criação de usuário."""
        # Gerar ID único
        user_id = UserId(f"user_{hash(command.email)}")

        # Criar entidade
        user = User(user_id, command.name, Email(command.email))

        # Simular persistência
        save_result = self._save_user(user)
        if save_result.is_failure:
            return FlextResult.fail(f"Erro ao salvar usuário: {save_result.error}")

        return FlextResult.ok(user)

    def _save_user(self, user: User) -> FlextResult[None]:
        """Simular salvamento no banco."""
        # Em aplicação real, usaria repository
        return FlextResult.ok(None)

# Uso do Command Pattern
container = FlextContainer()
handler = CreateUserHandler(container)

# Criar command
command = CreateUserCommand("Ana Paula", "ana@empresa.com")

# Processar command
result = handler.process_command(command)
if result.success:
    user = result.data
    print(f"✅ Usuário criado: {user.name} ({user.id})")
else:
    print(f"❌ Erro: {result.error}")

# Teste com dados inválidos
invalid_command = CreateUserCommand("", "email-invalid")
invalid_result = handler.process_command(invalid_command)
print(f"Validação: {invalid_result.error}")  # "Nome é obrigatório"
```

### 2. Validation Pattern - Business Rules

**Sistema de validação extensível para regras de negócio.**

```python
from flext_core.patterns import FlextValidator, FlextValidationResult
from flext_core.patterns.validation import ValidationRule

# Dados a serem validados
class UserRegistrationData:
    def __init__(self, name: str, email: str, age: int, password: str):
        self.name = name
        self.email = email
        self.age = age
        self.password = password

# Regras de validação customizadas
class MinimumAgeRule(ValidationRule[UserRegistrationData]):
    def __init__(self, min_age: int = 18):
        self.min_age = min_age

    def validate(self, data: UserRegistrationData) -> FlextValidationResult:
        if data.age < self.min_age:
            return FlextValidationResult.with_errors([
                f"Idade mínima é {self.min_age} anos"
            ])
        return FlextValidationResult.success()

class StrongPasswordRule(ValidationRule[UserRegistrationData]):
    def validate(self, data: UserRegistrationData) -> FlextValidationResult:
        result = FlextValidationResult.success()

        if len(data.password) < 8:
            result.add_error("Senha deve ter pelo menos 8 caracteres")

        if not any(c.isupper() for c in data.password):
            result.add_error("Senha deve ter pelo menos 1 letra maiúscula")

        if not any(c.isdigit() for c in data.password):
            result.add_error("Senha deve ter pelo menos 1 número")

        return result

# Validator principal
class UserRegistrationValidator(FlextValidator[UserRegistrationData]):
    def __init__(self):
        super().__init__()
        # Adicionar regras
        self.add_rule(MinimumAgeRule(18))
        self.add_rule(StrongPasswordRule())

    def validate_business_rules(self, data: UserRegistrationData) -> FlextValidationResult:
        """Validações específicas de negócio."""
        result = FlextValidationResult.success()

        # Regra: email deve ser corporativo
        if not data.email.endswith(".com"):
            result.add_error("Email deve ser de domínio corporativo (.com)")

        # Regra: nome deve ter sobrenome
        if len(data.name.split()) < 2:
            result.add_error("Nome completo deve incluir sobrenome")

        return result

# Teste do sistema de validação
validator = UserRegistrationValidator()

# Caso válido
valid_data = UserRegistrationData(
    name="Carlos Alberto Santos",
    email="carlos@empresa.com",
    age=25,
    password="MinhaSenh@123"
)

valid_result = validator.validate(valid_data)
print(f"Dados válidos: {valid_result.is_valid}")

# Caso inválido
invalid_data = UserRegistrationData(
    name="João",                    # Sem sobrenome
    email="joao@test.net",          # Não é .com
    age=16,                         # Menor de idade
    password="123"                  # Senha fraca
)

invalid_result = validator.validate(invalid_data)
print(f"Dados inválidos: {invalid_result.is_valid}")
print("Erros encontrados:")
for error in invalid_result.errors:
    print(f"  - {error}")

# Output:
# Dados válidos: True
# Dados inválidos: False
# Erros encontrados:
#   - Idade mínima é 18 anos
#   - Senha deve ter pelo menos 8 caracteres
#   - Senha deve ter pelo menos 1 letra maiúscula
#   - Email deve ser de domínio corporativo (.com)
#   - Nome completo deve incluir sobrenome
```

## 🏗️ Exemplo Completo - Sistema de Pedidos

**Sistema completo usando todos os conceitos fundamentais.**

```python
from flext_core import FlextEntity, FlextResult, FlextContainer
from flext_core.patterns import FlextCommand, FlextCommandHandler, FlextValidator
from typing import NewType, List
from datetime import datetime
from enum import Enum

# ========================
# DOMAIN LAYER
# ========================

# IDs e Value Objects
OrderId = NewType("OrderId", str)
ProductId = NewType("ProductId", str)
CustomerId = NewType("CustomerId", str)

class OrderStatus(str, Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"

class Product(FlextEntity[ProductId]):
    def __init__(self, product_id: ProductId, name: str, price: float, stock: int):
        super().__init__(product_id)
        self._name = name
        self._price = price
        self._stock = stock

    @property
    def name(self) -> str:
        return self._name

    @property
    def price(self) -> float:
        return self._price

    @property
    def stock(self) -> int:
        return self._stock

    def reserve_stock(self, quantity: int) -> FlextResult[None]:
        """Reservar estoque."""
        if quantity <= 0:
            return FlextResult.fail("Quantidade deve ser positiva")

        if self._stock < quantity:
            return FlextResult.fail(f"Estoque insuficiente. Disponível: {self._stock}")

        self._stock -= quantity
        return FlextResult.ok(None)

class OrderItem:
    def __init__(self, product: Product, quantity: int):
        self.product = product
        self.quantity = quantity

    def total_price(self) -> float:
        return self.product.price * self.quantity

class Order(FlextEntity[OrderId]):
    def __init__(self, order_id: OrderId, customer_id: CustomerId):
        super().__init__(order_id)
        self._customer_id = customer_id
        self._items: List[OrderItem] = []
        self._status = OrderStatus.PENDING
        self._created_at = datetime.now()
        self._total = 0.0

    @property
    def customer_id(self) -> CustomerId:
        return self._customer_id

    @property
    def status(self) -> OrderStatus:
        return self._status

    @property
    def total(self) -> float:
        return sum(item.total_price() for item in self._items)

    @property
    def items(self) -> List[OrderItem]:
        return self._items.copy()

    def add_item(self, product: Product, quantity: int) -> FlextResult[None]:
        """Adicionar item ao pedido."""
        if self._status != OrderStatus.PENDING:
            return FlextResult.fail("Não é possível modificar pedido já processado")

        # Reservar estoque
        reserve_result = product.reserve_stock(quantity)
        if reserve_result.is_failure:
            return reserve_result

        # Adicionar item
        item = OrderItem(product, quantity)
        self._items.append(item)

        return FlextResult.ok(None)

    def confirm(self) -> FlextResult[None]:
        """Confirmar pedido."""
        if self._status != OrderStatus.PENDING:
            return FlextResult.fail("Pedido deve estar pendente para ser confirmado")

        if not self._items:
            return FlextResult.fail("Pedido deve ter pelo menos um item")

        self._status = OrderStatus.CONFIRMED
        return FlextResult.ok(None)

# ========================
# APPLICATION LAYER
# ========================

# Command para criar pedido
class CreateOrderCommand(FlextCommand):
    def __init__(self, customer_id: str, items: List[dict]):
        super().__init__()
        self.customer_id = customer_id
        self.items = items  # [{"product_id": "p1", "quantity": 2}]

    def validate(self) -> FlextResult[None]:
        if not self.customer_id:
            return FlextResult.fail("Customer ID é obrigatório")

        if not self.items:
            return FlextResult.fail("Pedido deve ter pelo menos um item")

        for item in self.items:
            if "product_id" not in item or "quantity" not in item:
                return FlextResult.fail("Item deve ter product_id e quantity")

            if item["quantity"] <= 0:
                return FlextResult.fail("Quantidade deve ser positiva")

        return FlextResult.ok(None)

# Handler para processar criação de pedidos
class CreateOrderHandler(FlextCommandHandler[CreateOrderCommand, Order]):
    def __init__(self, container: FlextContainer):
        super().__init__()
        self._container = container

    def can_handle(self, command) -> bool:
        return isinstance(command, CreateOrderCommand)

    def handle(self, command: CreateOrderCommand) -> FlextResult[Order]:
        """Processar criação de pedido."""
        # Gerar ID do pedido
        order_id = OrderId(f"order_{hash(command.customer_id)}")
        customer_id = CustomerId(command.customer_id)

        # Criar pedido
        order = Order(order_id, customer_id)

        # Adicionar itens
        for item_data in command.items:
            product_result = self._get_product(item_data["product_id"])
            if product_result.is_failure:
                return FlextResult.fail(f"Produto não encontrado: {item_data['product_id']}")

            product = product_result.data
            add_result = order.add_item(product, item_data["quantity"])
            if add_result.is_failure:
                return FlextResult.fail(f"Erro ao adicionar item: {add_result.error}")

        # Confirmar pedido
        confirm_result = order.confirm()
        if confirm_result.is_failure:
            return confirm_result

        return FlextResult.ok(order)

    def _get_product(self, product_id: str) -> FlextResult[Product]:
        """Obter produto (simulado)."""
        # Em aplicação real, usaria repository
        products = {
            "p1": Product(ProductId("p1"), "Notebook", 2500.00, 10),
            "p2": Product(ProductId("p2"), "Mouse", 50.00, 100),
            "p3": Product(ProductId("p3"), "Teclado", 150.00, 50)
        }

        if product_id not in products:
            return FlextResult.fail(f"Produto {product_id} não encontrado")

        return FlextResult.ok(products[product_id])

# ========================
# USAGE EXEMPLO
# ========================

def run_order_system_example():
    """Exemplo completo do sistema de pedidos."""
    print("🛒 Sistema de Pedidos FLEXT Core\n")

    # Setup
    container = FlextContainer()
    handler = CreateOrderHandler(container)

    # Criar pedido válido
    print("📝 Criando pedido...")
    command = CreateOrderCommand(
        customer_id="customer_123",
        items=[
            {"product_id": "p1", "quantity": 1},  # Notebook
            {"product_id": "p2", "quantity": 2},  # 2x Mouse
        ]
    )

    result = handler.process_command(command)
    if result.success:
        order = result.data
        print(f"✅ Pedido criado: {order.id}")
        print(f"   Cliente: {order.customer_id}")
        print(f"   Status: {order.status}")
        print(f"   Total: R$ {order.total:.2f}")
        print(f"   Itens: {len(order.items)}")

        for item in order.items:
            print(f"     - {item.product.name}: {item.quantity}x R$ {item.product.price:.2f}")
    else:
        print(f"❌ Erro: {result.error}")

    print("\n" + "="*50 + "\n")

    # Teste com erro - produto inexistente
    print("🚫 Testando erro - produto inexistente...")
    invalid_command = CreateOrderCommand(
        customer_id="customer_456",
        items=[{"product_id": "invalid", "quantity": 1}]
    )

    invalid_result = handler.process_command(invalid_command)
    if invalid_result.is_failure:
        print(f"❌ Erro esperado: {invalid_result.error}")

    print("\n" + "="*50 + "\n")

    # Teste com validação - dados inválidos
    print("🚫 Testando validação - quantidade inválida...")
    validation_command = CreateOrderCommand(
        customer_id="",  # Customer ID vazio
        items=[{"product_id": "p1", "quantity": -1}]  # Quantidade negativa
    )

    validation_result = handler.process_command(validation_command)
    if validation_result.is_failure:
        print(f"❌ Erro de validação: {validation_result.error}")

# Executar exemplo
if __name__ == "__main__":
    run_order_system_example()
```

## 🎯 Próximos Passos

### 1. Aprofundar Conhecimento

- **[Arquitetura](../architecture/overview.md)** - Entender design patterns
- **[API Core](../api/core.md)** - Referência completa das APIs
- **[Patterns](../api/patterns.md)** - Padrões avançados

### 2. Explorar Exemplos

- **[Examples](../examples/overview.md)** - Casos de uso reais
- **[Best Practices](../development/best-practices.md)** - Melhores práticas

### 3. Desenvolvimento

```bash
# Configurar ambiente de desenvolvimento
git clone https://github.com/flext/flext-core.git
cd flext-core
make setup

# Executar testes
make test

# Verificar qualidade
make validate
```

## 🆘 Suporte

**Precisa de ajuda?**

- **Issues**: [GitHub Issues](https://github.com/flext/flext-core/issues)
- **Documentação**: [Docs Completa](https://docs.flext.dev)
- **Discussões**: [GitHub Discussions](https://github.com/flext/flext-core/discussions)

---

**Parabéns!** 🎉 Você já tem o conhecimento fundamental para usar FLEXT Core em seus projetos empresariais!
