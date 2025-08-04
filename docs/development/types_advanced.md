# FLEXT Core Advanced Types

Sistema de tipos avançados para redução de boilerplate em aplicações.

## Tipos Funcionais

### Either[T, R]

Representa sucesso ou erro sem exceções:

```python
from flext_core import Either

# Criar valores
success = Either.right("dados")
error = Either.left("erro")

# Usar com chaining
result = (
    Either.right("hello")
    .map(str.upper)
    .map(lambda x: f"{x}!")
)
```

### Pipe[T, R]

Pipeline type-safe para transformações:

```python
from flext_core import Pipe, FlextResult

def double(x: int) -> FlextResult[int]:
    return FlextResult.ok(x * 2)

def add_one(x: int) -> FlextResult[int]:
    return FlextResult.ok(x + 1)

pipe = Pipe(double).then(Pipe(add_one))
result = pipe(5)  # 11
```

## Protocols Estruturais

### Identifiable

Para objetos com ID:

```python
from flext_core import Identifiable, is_identifiable

class User:
    def __init__(self, user_id: str):
        self.id = user_id

user = User("123")
assert is_identifiable(user)  # True
```

### Serializable

Para objetos serializáveis:

```python
from flext_core import Serializable, is_serializable

class Config:
    def to_dict(self) -> dict:
        return {"key": "value"}

    @classmethod
    def from_dict(cls, data: dict):
        return cls()

config = Config()
assert is_serializable(config)  # True
```

## Utilitários de Conversão

### ensure_result

Garante que valor seja FlextResult:

```python
from flext_core import ensure_result

# Com valor regular
result = ensure_result("data")
assert result.success

# Com FlextResult existente (não modifica)
existing = FlextResult.ok("data")
result = ensure_result(existing)
assert result is existing
```

### ensure_list

Garante que valor seja lista:

```python
from flext_core import ensure_list

assert ensure_list("single") == ["single"]
assert ensure_list(["already", "list"]) == ["already", "list"]
assert ensure_list(("a", "b")) == ["a", "b"]
```

### ensure_dict

Garante que valor seja dicionário:

```python
from flext_core import ensure_dict

# Com objeto tendo to_dict
class Obj:
    def to_dict(self):
        return {"key": "value"}

assert ensure_dict(Obj()) == {"key": "value"}

# Fallback
assert ensure_dict("simple") == {"value": "simple"}
```

## Type Aliases Semânticos

Reduza boilerplate usando types semânticos:

```python
from flext_core import EntityId, UserId, FlextDict, MetadataDict

def process_user(
    user_id: UserId,
    data: FlextDict,
    metadata: MetadataDict
) -> FlextResult[dict]:
    # Código mais claro e type-safe
    pass
```

## Tipos Genéricos

### Repository[T]

Protocol para repositórios:

```python
from flext_core import Repository, FlextResult

class UserRepository(Repository[User]):
    def find_by_id(self, entity_id: str) -> FlextResult[User]:
        # implementação
        pass

    def save(self, entity: User) -> FlextResult[None]:
        # implementação
        pass
```

### Factory[T]

Protocol para factories:

```python
from flext_core import Factory, FlextResult

class UserFactory(Factory[User]):
    def create(self, **kwargs) -> FlextResult[User]:
        return FlextResult.ok(User(**kwargs))
```

## Importação

Todos os tipos estão disponíveis na raiz:

```python
from flext_core import (
    Either, Pipe, Repository, Factory,
    Identifiable, Serializable, Timestamped,
    ensure_result, ensure_list, ensure_dict,
    EntityId, UserId, FlextDict, MetadataDict
)
```
