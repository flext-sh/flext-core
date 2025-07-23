# API Patterns - FLEXT Core

**Padrões empresariais para Commands, Handlers, Validation e mais**

## 🎯 Visão Geral

A API Patterns do FLEXT Core implementa padrões de design empresariais que facilitam a construção de aplicações robustas, escaláveis e testáveis. Todos os padrões são type-safe e seguem princípios de Clean Architecture.

## 📦 Imports

```python
# Command Pattern
from flext_core.patterns import (
    FlextCommand,
    FlextCommandHandler,
    FlextCommandBus,
    FlextCommandResult
)

# Handler Pattern  
from flext_core.patterns import (
    FlextHandler,
    FlextMessageHandler,
    FlextEventHandler,
    FlextRequestHandler,
    FlextHandlerRegistry
)

# Validation Pattern
from flext_core.patterns import (
    FlextValidator,
    FlextValidationResult,
    FlextValidationRule,
    FlextFieldValidator,
    NotEmptyRule,
    MinLengthRule,
    MaxLengthRule,
    RangeRule
)

# Type Definitions
from flext_core.patterns import (
    FlextCommandId,
    FlextCommandType,
    FlextHandlerId,
    FlextHandlerName,
    FlextValidatorId
)
```

## 🎭 Command Pattern - CQRS Implementation

**Implementação completa do padrão Command para operações de escrita.**

### FlextCommand - Base Command

```python
from flext_core.patterns import FlextCommand
from flext_core import FlextResult

class CreateUserCommand(FlextCommand):
    """Command para criar usuário."""
    
    def __init__(self, name: str, email: str, age: int):
        super().__init__(
            command_type=FlextCommandType("create_user")
        )
        self.name = name
        self.email = email
        self.age = age
    
    def validate(self) -> FlextResult[None]:
        """Validação específica do command."""
        if not self.name.strip():
            return FlextResult.fail("Nome é obrigatório")
        
        if "@" not in self.email:
            return FlextResult.fail("Email inválido")
        
        if self.age < 0 or self.age > 150:
            return FlextResult.fail("Idade deve estar entre 0 e 150")
        
        return FlextResult.ok(None)
    
    def get_payload(self) -> dict[str, Any]:
        """Payload para logging/auditoria."""
        return {
            "name": self.name,
            "email": self.email,
            "age": self.age
        }

# Usage
command = CreateUserCommand("João Silva", "joao@example.com", 30)

# Validação automática
validation_result = command.validate()
if validation_result.is_failure:
    print(f"Command inválido: {validation_result.error}")

# Metadados
metadata = command.get_command_metadata()
print(f"Command ID: {metadata['command_id']}")
print(f"Command Type: {metadata['command_type']}")
```

### FlextCommandHandler - Command Processing

```python
from flext_core.patterns import FlextCommandHandler

class CreateUserHandler(FlextCommandHandler[CreateUserCommand, User]):
    """Handler para CreateUserCommand."""
    
    def __init__(self, user_repository: UserRepository):
        super().__init__(handler_id="create_user_handler")
        self._repository = user_repository
    
    def can_handle(self, command: FlextCommand) -> bool:
        """Verifica se pode processar o command."""
        return isinstance(command, CreateUserCommand)
    
    def handle(self, command: CreateUserCommand) -> FlextResult[User]:
        """Processa o command."""
        # Business logic
        user = User.create(
            name=command.name,
            email=command.email,
            age=command.age
        )
        
        # Persistência
        save_result = self._repository.save(user)
        if save_result.is_failure:
            return FlextResult.fail(f"Erro ao salvar: {save_result.error}")
        
        return FlextResult.ok(user)

# Usage
handler = CreateUserHandler(user_repository)
command = CreateUserCommand("João", "joao@test.com", 30)

# Processamento completo com validação
result = handler.process_command(command)
if result.is_success:
    print(f"Usuário criado: {result.data}")
else:
    print(f"Erro: {result.error}")
```

### FlextCommandBus - Central Command Processing

```python
from flext_core.patterns import FlextCommandBus

# Setup do bus
bus = FlextCommandBus()

# Registro de handlers
create_handler = CreateUserHandler(user_repository)
update_handler = UpdateUserHandler(user_repository)

bus.register_handler(create_handler)
bus.register_handler(update_handler)

# Execução de commands
create_command = CreateUserCommand("Maria", "maria@test.com", 25)
result = bus.execute(create_command)

if result.is_success:
    user = result.data.result  # Resultado do handler
    print(f"Usuário criado: {user.name}")
    
    # Metadados da execução
    metadata = result.get_result_metadata()
    print(f"Command ID: {metadata['command_id']}")
    print(f"Executado com sucesso: {metadata['is_success']}")
else:
    print(f"Falha na execução: {result.error}")

# Gerenciamento de handlers
all_handlers = bus.get_all_handlers()
print(f"Handlers registrados: {len(all_handlers)}")

# Busca por handler específico
handler = bus.find_handler(create_command)
if handler:
    print(f"Handler encontrado: {handler.__class__.__name__}")
```

### Command com Resultados Complexos

```python
class ProcessPaymentCommand(FlextCommand):
    def __init__(self, order_id: str, amount: float, payment_method: str):
        super().__init__(command_type=FlextCommandType("process_payment"))
        self.order_id = order_id
        self.amount = amount
        self.payment_method = payment_method
    
    def validate(self) -> FlextResult[None]:
        if self.amount <= 0:
            return FlextResult.fail("Valor deve ser positivo")
        
        valid_methods = ["credit_card", "debit_card", "pix"]
        if self.payment_method not in valid_methods:
            return FlextResult.fail(f"Método inválido: {self.payment_method}")
        
        return FlextResult.ok(None)

class PaymentResult:
    def __init__(self, transaction_id: str, status: str, amount: float):
        self.transaction_id = transaction_id
        self.status = status
        self.amount = amount

class ProcessPaymentHandler(FlextCommandHandler[ProcessPaymentCommand, PaymentResult]):
    def handle(self, command: ProcessPaymentCommand) -> FlextResult[PaymentResult]:
        # Simular processamento de pagamento
        if command.amount > 10000:
            return FlextResult.fail("Valor acima do limite permitido")
        
        # Processar pagamento
        transaction_id = f"txn_{uuid.uuid4().hex[:8]}"
        payment_result = PaymentResult(
            transaction_id=transaction_id,
            status="approved",
            amount=command.amount
        )
        
        return FlextResult.ok(payment_result)

# Usage
payment_command = ProcessPaymentCommand("order_123", 250.0, "credit_card")
payment_handler = ProcessPaymentHandler()

result = payment_handler.process_command(payment_command)
if result.is_success:
    payment = result.data
    print(f"Pagamento aprovado: {payment.transaction_id}")
```

## 🎪 Handler Pattern - Message Processing

**Sistema unificado de processamento de mensagens, eventos e requests.**

### FlextMessageHandler - General Messages

```python
from flext_core.patterns import FlextMessageHandler

class EmailMessage:
    def __init__(self, to: str, subject: str, body: str):
        self.to = to
        self.subject = subject
        self.body = body

class EmailHandler(FlextMessageHandler[EmailMessage, str]):
    """Handler para processar emails."""
    
    def __init__(self, email_service: EmailService):
        super().__init__()
        self._email_service = email_service
    
    def can_handle(self, message: Any) -> bool:
        """Verifica se pode processar a mensagem."""
        return isinstance(message, EmailMessage)
    
    def handle_message(self, message: EmailMessage) -> FlextResult[str]:
        """Processa o email."""
        try:
            message_id = self._email_service.send(
                to=message.to,
                subject=message.subject,
                body=message.body
            )
            return FlextResult.ok(f"Email enviado: {message_id}")
        except Exception as e:
            return FlextResult.fail(f"Falha no envio: {str(e)}")

# Usage
email_handler = EmailHandler(email_service)
email_message = EmailMessage(
    to="user@example.com",
    subject="Bem-vindo!",
    body="Obrigado por se cadastrar."
)

# Processamento completo
result = email_handler.process(email_message)
if result.is_success:
    print(f"Resultado: {result.data}")
```

### FlextEventHandler - Domain Events

```python
from flext_core.patterns import FlextEventHandler, FlextMessageType

class UserCreatedEvent:
    def __init__(self, user_id: str, name: str, email: str):
        self.event_type = "user.created"
        self.user_id = user_id
        self.name = name
        self.email = email
        self.timestamp = datetime.now()

class UserCreatedHandler(FlextEventHandler[UserCreatedEvent]):
    """Handler para evento de usuário criado."""
    
    def __init__(self, notification_service: NotificationService):
        super().__init__()
        self._notification_service = notification_service
    
    def get_event_type(self) -> FlextMessageType:
        """Tipo de evento que este handler processa."""
        return FlextMessageType("user.created")
    
    def handle_event(self, event: UserCreatedEvent) -> FlextResult[None]:
        """Processa o evento."""
        try:
            # Enviar notificação de boas-vindas
            self._notification_service.send_welcome_notification(
                user_id=event.user_id,
                name=event.name,
                email=event.email
            )
            
            # Log do evento
            logger.info(f"Welcome notification sent to {event.email}")
            
            return FlextResult.ok(None)
        except Exception as e:
            return FlextResult.fail(f"Notification failed: {str(e)}")

# Usage
handler = UserCreatedHandler(notification_service)
event = UserCreatedEvent("user_123", "João", "joao@test.com")

# Processamento de evento
result = handler.process_event(event)
if result.is_success:
    print("Evento processado com sucesso")
```

### FlextRequestHandler - Request/Response

```python
from flext_core.patterns import FlextRequestHandler

class GetUserRequest:
    def __init__(self, user_id: str):
        self.user_id = user_id

class UserResponse:
    def __init__(self, user_id: str, name: str, email: str):
        self.user_id = user_id
        self.name = name
        self.email = email

class GetUserHandler(FlextRequestHandler[GetUserRequest, UserResponse]):
    """Handler para buscar usuário."""
    
    def __init__(self, user_repository: UserRepository):
        super().__init__()
        self._repository = user_repository
    
    def get_request_type(self) -> type[GetUserRequest]:
        """Tipo de request que este handler processa."""
        return GetUserRequest
    
    def handle_request(self, request: GetUserRequest) -> FlextResult[UserResponse]:
        """Processa a request.""" 
        user_result = self._repository.find_by_id(request.user_id)
        
        if user_result.is_failure:
            return FlextResult.fail(f"Usuário não encontrado: {request.user_id}")
        
        user = user_result.data
        response = UserResponse(
            user_id=user.id,
            name=user.name,
            email=user.email
        )
        
        return FlextResult.ok(response)

# Usage
handler = GetUserHandler(user_repository)
request = GetUserRequest("user_123")

result = handler.process_request(request)
if result.is_success:
    user_response = result.data
    print(f"Usuário: {user_response.name} ({user_response.email})")
```

### FlextHandlerRegistry - Handler Management

```python
from flext_core.patterns import FlextHandlerRegistry

# Criar registry
registry = FlextHandlerRegistry()

# Registrar handlers
email_handler = EmailHandler(email_service)
user_created_handler = UserCreatedHandler(notification_service)
get_user_handler = GetUserHandler(user_repository)

registry.register(email_handler)
registry.register(user_created_handler)
registry.register(get_user_handler)

# Buscar handlers para mensagens específicas
email_message = EmailMessage("test@example.com", "Subject", "Body")
handlers = registry.find_handlers(email_message)
print(f"Handlers encontrados: {len(handlers)}")

# Processar com handlers encontrados
for handler in handlers:
    if isinstance(handler, FlextMessageHandler):
        result = handler.process(email_message)
        if result.is_success:
            print(f"Processado: {result.data}")

# Buscar handler por ID
handler_id = email_handler.handler_id
found_handler = registry.get_handler_by_id(handler_id)  
if found_handler:
    print(f"Handler encontrado: {found_handler.__class__.__name__}")

# Informações dos handlers
handler_info = registry.get_handler_info()
for info in handler_info:
    print(f"Handler: {info['handler_class']} (ID: {info['handler_id']})")
```

## ✅ Validation Pattern - Robust Validation

**Sistema de validação robusto com regras reutilizáveis.**

### FlextValidationResult - Validation Results

```python
from flext_core.patterns import FlextValidationResult

# Resultado de sucesso
result = FlextValidationResult.success()
print(f"Válido: {result.is_valid}")  # True

# Resultado com warnings
result = FlextValidationResult.success(warnings=["Campo obsoleto"])
print(f"Warnings: {result.warnings}")

# Resultado de falha
result = FlextValidationResult.failure(
    errors=["Erro geral"],
    field_errors={"email": ["Formato inválido"]}
)
print(f"Válido: {result.is_valid}")  # False
print(f"Erros: {result.get_all_errors()}")

# Adicionando erros dinamicamente
result = FlextValidationResult.success()
result.add_error("Erro geral")
result.add_field_error("name", "Nome muito curto")
result.add_warning("Campo será removido")

# Verificações
print(f"Tem erros: {result.has_errors()}")
print(f"Tem warnings: {result.has_warnings()}")

# Merge de resultados
result1 = FlextValidationResult.success()
result1.add_error("Erro 1")

result2 = FlextValidationResult.failure(errors=["Erro 2"])

result1.merge(result2)  # Combina ambos os resultados
print(f"Erros combinados: {result1.errors}")
```

### FlextValidationRule - Built-in Rules

```python
from flext_core.patterns import (
    NotEmptyRule,
    MinLengthRule,
    MaxLengthRule,
    RangeRule
)

# Regra de não vazio
not_empty = NotEmptyRule()
print(not_empty.validate("texto"))      # True
print(not_empty.validate(""))           # False
print(not_empty.validate("   "))        # False

# Regra de tamanho mínimo
min_length = MinLengthRule(3)
print(min_length.validate("abc"))       # True
print(min_length.validate("ab"))        # False

# Mensagem de erro
error_msg = min_length.get_error_message("ab")
print(error_msg)  # "Value must be at least 3 characters"

# Regra de tamanho máximo
max_length = MaxLengthRule(10)
print(max_length.validate("texto"))     # True
print(max_length.validate("texto muito longo"))  # False

# Regra de range numérico
age_range = RangeRule(0, 120)
print(age_range.validate(25))           # True
print(age_range.validate(-5))           # False
print(age_range.validate(150))          # False
```

### Custom Validation Rules

```python
from flext_core.patterns import FlextValidationRule, FlextRuleName

class EmailRule(FlextValidationRule[str]):
    """Regra customizada para validação de email."""
    
    def __init__(self):
        super().__init__(FlextRuleName("email_format"))
    
    def validate(self, value: str) -> bool:
        """Valida formato de email."""
        if not value:
            return False
        
        return "@" in value and "." in value.split("@")[1]
    
    def _get_default_error_message(self) -> str:
        return "Email deve ter formato válido (exemplo@dominio.com)"

class CPFRule(FlextValidationRule[str]):
    """Regra para validação de CPF."""
    
    def __init__(self):
        super().__init__(FlextRuleName("cpf_format"))
    
    def validate(self, value: str) -> bool:
        """Valida CPF brasileiro."""
        if not value:
            return False
        
        # Remove formatação
        cpf = ''.join(filter(str.isdigit, value))
        
        # Verifica tamanho
        if len(cpf) != 11:
            return False
        
        # Verifica se não são todos iguais
        if cpf == cpf[0] * 11:
            return False
        
        # Algoritmo de validação do CPF
        return self._validate_cpf_digits(cpf)
    
    def _validate_cpf_digits(self, cpf: str) -> bool:
        """Valida dígitos verificadores do CPF."""
        # Implementação do algoritmo de validação
        for i in range(9, 11):
            value = sum((int(cpf[num]) * ((i+1) - num) for num in range(0, i)))
            digit = ((value * 10) % 11) % 10
            if digit != int(cpf[i]):
                return False
        return True
    
    def _get_default_error_message(self) -> str:
        return "CPF deve ter formato válido (XXX.XXX.XXX-XX)"

# Usage
email_rule = EmailRule()
print(email_rule.validate("user@example.com"))  # True
print(email_rule.validate("invalid"))           # False

cpf_rule = CPFRule()
print(cpf_rule.validate("123.456.789-09"))      # Depends on algorithm
```

### FlextFieldValidator - Field-Level Validation

```python
from flext_core.patterns import FlextFieldValidator, FlextFieldPath

# Validador para campo de nome
name_validator = FlextFieldValidator(
    field_path=FlextFieldPath("name"),
    rules=[
        NotEmptyRule(),
        MinLengthRule(2),
        MaxLengthRule(50)
    ],
    required=True
)

# Validação de campo
result = name_validator.validate("João")
print(f"Nome válido: {result.is_valid}")

result = name_validator.validate("")
print(f"Nome vazio válido: {result.is_valid}")
print(f"Erros: {result.field_errors}")

# Validador para email
email_validator = FlextFieldValidator(
    field_path=FlextFieldPath("email"), 
    rules=[
        NotEmptyRule(),
        EmailRule()  # Custom rule
    ],
    required=True
)

# Validador para idade  
age_validator = FlextFieldValidator(
    field_path=FlextFieldPath("age"),
    rules=[RangeRule(0, 120)],
    required=False  # Campo opcional
)

# Campo opcional com None
result = age_validator.validate(None)
print(f"Idade None (opcional): {result.is_valid}")  # True

result = age_validator.validate(25)
print(f"Idade 25: {result.is_valid}")  # True

result = age_validator.validate(-5)
print(f"Idade -5: {result.is_valid}")  # False
```

### FlextValidator - Complete Entity Validation

```python
from flext_core.patterns import FlextValidator, FlextValidatorId

class UserValidator(FlextValidator[dict[str, Any]]):
    """Validador completo para dados de usuário."""
    
    def __init__(self):
        super().__init__(FlextValidatorId("user_validator"))
        
        # Configurar validadores de campo
        self.add_field_validator("name", FlextFieldValidator(
            FlextFieldPath("name"),
            rules=[NotEmptyRule(), MinLengthRule(2), MaxLengthRule(50)],
            required=True
        ))
        
        self.add_field_validator("email", FlextFieldValidator(
            FlextFieldPath("email"),
            rules=[NotEmptyRule(), EmailRule()],
            required=True
        ))
        
        self.add_field_validator("age", FlextFieldValidator(
            FlextFieldPath("age"),
            rules=[RangeRule(0, 120)],
            required=False
        ))
        
        self.add_field_validator("cpf", FlextFieldValidator(
            FlextFieldPath("cpf"),
            rules=[CPFRule()],
            required=False
        ))
    
    def validate_business_rules(self, data: dict[str, Any]) -> FlextValidationResult:
        """Regras de negócio específicas."""
        result = FlextValidationResult.success()
        
        # Regra: usuários menores de 18 anos precisam de responsável
        age = data.get("age")
        guardian = data.get("guardian")
        
        if age and age < 18 and not guardian:
            result.add_error("Usuários menores de 18 anos precisam de responsável")
        
        # Regra: email corporativo para funcionários
        role = data.get("role")
        email = data.get("email", "")
        
        if role == "employee" and not email.endswith("@company.com"):
            result.add_field_error("email", "Funcionários devem usar email corporativo")
        
        # Warning para campos opcionais em branco
        if not data.get("phone"):
            result.add_warning("Telefone não informado - recomendado para contato")
        
        return result

# Usage
validator = UserValidator()

# Dados válidos
valid_data = {
    "name": "João Silva",
    "email": "joao@example.com", 
    "age": 30,
    "phone": "11999999999"
}

result = validator.validate(valid_data)
print(f"Dados válidos: {result.is_valid}")

# Dados inválidos
invalid_data = {
    "name": "",  # Vazio
    "email": "invalid",  # Formato inválido
    "age": -5,  # Fora do range
    "role": "employee"  # Sem email corporativo
}

result = validator.validate(invalid_data)
print(f"Dados inválidos: {result.is_valid}")
print(f"Erros: {result.get_all_errors()}")
print(f"Warnings: {result.warnings}")

# Dados de menor de idade sem responsável
minor_data = {
    "name": "Maria Silva",
    "email": "maria@example.com",
    "age": 16
    # guardian não informado
}

result = validator.validate(minor_data)
print(f"Menor sem responsável: {result.is_valid}")
print(f"Erros de negócio: {result.errors}")
```

### Complex Validation Scenarios

```python
class OrderValidator(FlextValidator[dict[str, Any]]):
    """Validador para pedidos complexos."""
    
    def __init__(self):
        super().__init__(FlextValidatorId("order_validator"))
        
        # Campos do pedido
        self.add_field_validator("customer_id", FlextFieldValidator(
            FlextFieldPath("customer_id"),
            rules=[NotEmptyRule()],
            required=True
        ))
        
        self.add_field_validator("total_amount", FlextFieldValidator(
            FlextFieldPath("total_amount"),
            rules=[RangeRule(0.01, 100000.0)],
            required=True
        ))
    
    def validate_business_rules(self, data: dict[str, Any]) -> FlextValidationResult:
        result = FlextValidationResult.success()
        
        # Validar itens do pedido
        items = data.get("items", [])
        if not items:
            result.add_error("Pedido deve ter pelo menos um item")
            return result
        
        # Validar cada item
        for i, item in enumerate(items):
            item_result = self._validate_order_item(item, i)
            result.merge(item_result)
        
        # Validar total calculado vs informado
        calculated_total = sum(
            item.get("quantity", 0) * item.get("price", 0) 
            for item in items
        )
        informed_total = data.get("total_amount", 0)
        
        if abs(calculated_total - informed_total) > 0.01:
            result.add_error(
                f"Total informado ({informed_total}) difere do calculado ({calculated_total})"
            )
        
        # Regras de desconto
        discount = data.get("discount", 0)
        if discount > calculated_total * 0.5:  # Max 50% desconto
            result.add_error("Desconto não pode ser maior que 50% do total")
        
        return result
    
    def _validate_order_item(self, item: dict, index: int) -> FlextValidationResult:
        """Valida item individual do pedido."""
        result = FlextValidationResult.success()
        
        # Campos obrigatórios do item
        if not item.get("product_id"):
            result.add_field_error(f"items[{index}].product_id", "Product ID é obrigatório")
        
        quantity = item.get("quantity", 0)
        if quantity <= 0:
            result.add_field_error(f"items[{index}].quantity", "Quantidade deve ser positiva")
        
        price = item.get("price", 0)
        if price <= 0:
            result.add_field_error(f"items[{index}].price", "Preço deve ser positivo")
        
        return result

# Usage
order_validator = OrderValidator()

# Pedido válido
valid_order = {
    "customer_id": "customer_123",
    "total_amount": 150.0,
    "discount": 10.0,
    "items": [
        {"product_id": "prod_1", "quantity": 2, "price": 50.0},
        {"product_id": "prod_2", "quantity": 1, "price": 50.0}
    ]
}

result = order_validator.validate(valid_order)
print(f"Pedido válido: {result.is_valid}")

# Pedido inválido
invalid_order = {
    "customer_id": "",  # Vazio
    "total_amount": 200.0,  # Total incorreto
    "discount": 100.0,  # Desconto acima de 50%
    "items": [
        {"product_id": "prod_1", "quantity": 0, "price": 50.0},  # Quantidade zero
        {"product_id": "", "quantity": 1, "price": -10.0}  # Preço negativo
    ]
}

result = order_validator.validate(invalid_order)
print(f"Pedido inválido: {result.is_valid}")
print("\nErros encontrados:")
for error in result.get_all_errors():
    print(f"- {error}")
```

## 🎨 Integration Patterns

### Command + Handler + Validation Integration

```python
# Command com validação integrada
class CreateOrderCommand(FlextCommand):
    def __init__(self, customer_id: str, items: list[dict], discount: float = 0.0):
        super().__init__(command_type=FlextCommandType("create_order"))
        self.customer_id = customer_id
        self.items = items
        self.discount = discount
        self.total_amount = sum(item["quantity"] * item["price"] for item in items)
    
    def validate(self) -> FlextResult[None]:
        # Usar validador reutilizável
        validator = OrderValidator()
        validation_result = validator.validate({
            "customer_id": self.customer_id,
            "items": self.items,
            "total_amount": self.total_amount,
            "discount": self.discount
        })
        
        if validation_result.is_valid:
            return FlextResult.ok(None)
        else:
            errors = "; ".join(validation_result.get_all_errors())
            return FlextResult.fail(f"Command validation failed: {errors}")

# Handler que processa o command
class CreateOrderHandler(FlextCommandHandler[CreateOrderCommand, Order]):
    def __init__(self, order_repository: OrderRepository, inventory_service: InventoryService):
        super().__init__(handler_id="create_order_handler")
        self._order_repo = order_repository
        self._inventory = inventory_service
    
    def can_handle(self, command: FlextCommand) -> bool:
        return isinstance(command, CreateOrderCommand)
    
    def handle(self, command: CreateOrderCommand) -> FlextResult[Order]:
        # Verificar estoque
        for item in command.items:
            stock_result = self._inventory.check_availability(
                item["product_id"], 
                item["quantity"]
            )
            if stock_result.is_failure:
                return FlextResult.fail(f"Insufficient stock: {stock_result.error}")
        
        # Criar pedido
        order = Order.create(
            customer_id=command.customer_id,
            items=command.items,
            discount=command.discount
        )
        
        # Salvar
        save_result = self._order_repo.save(order)
        if save_result.is_failure:
            return FlextResult.fail(f"Failed to save order: {save_result.error}")
        
        # Reservar estoque
        for item in command.items:
            self._inventory.reserve_stock(item["product_id"], item["quantity"])
        
        return FlextResult.ok(order)

# Event handler para side effects
class OrderCreatedHandler(FlextEventHandler[OrderCreatedEvent]):
    def __init__(self, email_service: EmailService, audit_service: AuditService):
        super().__init__()
        self._email = email_service
        self._audit = audit_service
    
    def get_event_type(self) -> FlextMessageType:
        return FlextMessageType("order.created")
    
    def handle_event(self, event: OrderCreatedEvent) -> FlextResult[None]:
        # Enviar email de confirmação
        email_result = self._email.send_order_confirmation(
            event.customer_email,
            event.order_id
        )
        
        if email_result.is_failure:
            # Log error but don't fail the event processing
            logger.error(f"Failed to send confirmation email: {email_result.error}")
        
        # Auditoria
        self._audit.log_order_creation(event.order_id, event.customer_id)
        
        return FlextResult.ok(None)

# Integration usage
def setup_order_processing():
    # Setup dependencies
    order_repo = OrderRepository()
    inventory_service = InventoryService()
    email_service = EmailService()
    audit_service = AuditService()
    
    # Setup command bus
    command_bus = FlextCommandBus()
    create_order_handler = CreateOrderHandler(order_repo, inventory_service)
    command_bus.register_handler(create_order_handler)
    
    # Setup event handlers
    event_registry = FlextHandlerRegistry()
    order_created_handler = OrderCreatedHandler(email_service, audit_service)
    event_registry.register(order_created_handler)
    
    return command_bus, event_registry

# Usage
command_bus, event_registry = setup_order_processing()

# Criar pedido
create_command = CreateOrderCommand(
    customer_id="customer_123",
    items=[
        {"product_id": "prod_1", "quantity": 2, "price": 50.0},
        {"product_id": "prod_2", "quantity": 1, "price": 30.0}
    ],
    discount=10.0
)

# Executar command
result = command_bus.execute(create_command)
if result.is_success:
    order = result.data.result
    print(f"Pedido criado: {order.id}")
    
    # Publicar evento
    event = OrderCreatedEvent(
        order_id=order.id,
        customer_id=order.customer_id,
        customer_email="customer@example.com"
    )
    
    # Processar evento
    event_handlers = event_registry.find_handlers(event)
    for handler in event_handlers:
        if isinstance(handler, FlextEventHandler):
            event_result = handler.process_event(event)
            if event_result.is_failure:
                logger.error(f"Event processing failed: {event_result.error}")
else:
    print(f"Falha ao criar pedido: {result.error}")
```

## 📝 Best Practices

### 1. Command Design

```python
# ✅ Commands são imutáveis após criação
class CreateUserCommand(FlextCommand):
    def __init__(self, name: str, email: str):
        super().__init__()
        self._name = name  # Private, immutable
        self._email = email
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def email(self) -> str:
        return self._email

# ✅ Validação abrangente
def validate(self) -> FlextResult[None]:
    errors = []
    
    if not self.name.strip():
        errors.append("Nome é obrigatório")
    
    if not self.email or "@" not in self.email:
        errors.append("Email inválido")
    
    if errors:
        return FlextResult.fail("; ".join(errors))
    
    return FlextResult.ok(None)
```

### 2. Handler Implementation

```python
# ✅ Handlers focados em uma responsabilidade
class CreateUserHandler(FlextCommandHandler[CreateUserCommand, User]):
    def __init__(self, user_repo: UserRepository, email: EmailService):
        super().__init__()
        self._user_repo = user_repo
        self._email = email
    
    def handle(self, command: CreateUserCommand) -> FlextResult[User]:
        # Single responsibility: create user
        user = User.create(command.name, command.email)
        
        save_result = self._user_repo.save(user)
        if save_result.is_failure:
            return save_result
        
        # Side effects via events, not directly here
        # self._publish_event(UserCreatedEvent(user))
        
        return FlextResult.ok(user)
```

### 3. Validation Rules

```python
# ✅ Regras reutilizáveis e compostas
class EmailValidator(FlextValidator[str]):
    def __init__(self):
        super().__init__()
        self.add_field_validator("email", FlextFieldValidator(
            FlextFieldPath(""),  # Root field
            rules=[
                NotEmptyRule(),
                EmailRule(),
                MaxLengthRule(254)  # RFC limit
            ]
        ))
    
    def validate_business_rules(self, email: str) -> FlextValidationResult:
        result = FlextValidationResult.success()
        
        # Business rule: no disposable emails
        disposable_domains = ["tempmail.com", "10minutemail.com"]
        domain = email.split("@")[1] if "@" in email else ""
        
        if domain in disposable_domains:
            result.add_error("Emails temporários não são permitidos")
        
        return result
```

### 4. Error Handling

```python
# ✅ Erros específicos e informativos
def handle(self, command: CreateUserCommand) -> FlextResult[User]:
    try:
        # Business validation
        if self._user_exists(command.email):
            return FlextResult.fail(f"Usuário já existe com email: {command.email}")
        
        # Create user
        user = User.create(command.name, command.email)
        
        # Persist
        save_result = self._user_repo.save(user)
        if save_result.is_failure:
            return FlextResult.fail(f"Falha ao salvar usuário: {save_result.error}")
        
        return FlextResult.ok(user)
        
    except DatabaseConnectionError as e:
        return FlextResult.fail(f"Erro de conexão com banco: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in CreateUserHandler: {str(e)}")
        return FlextResult.fail("Erro interno do servidor")
```

---

Esta API Patterns fornece todos os padrões necessários para construir aplicações empresariais robustas e escaláveis com FLEXT Core.
