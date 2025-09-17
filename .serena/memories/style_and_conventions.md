# FLEXT-DB-ORACLE Style and Conventions

## FLEXT Architectural Patterns (MANDATORY)

### 1. Unified Class Pattern (ZERO TOLERANCE)
- **Single class per module** with proper FlextDb prefix
- **Nested helper classes** instead of loose functions
- **No aliases, wrappers, or legacy access patterns**

```python
class FlextDbOracleService(FlextDomainService):
    """Single responsibility class with nested helpers."""
    
    def __init__(self, **data) -> None:
        super().__init__(**data)
        self._container = FlextContainer.get_global()
        self._logger = FlextLogger(__name__)
    
    class _ValidationHelper:
        """Nested helper class - no loose functions."""
        @staticmethod
        def validate_rules(data: dict) -> FlextResult[dict]:
            pass
```

### 2. FlextResult Pattern (MANDATORY)
- **ALL operations return FlextResult[T]** for type-safe error handling
- **Use .unwrap() for safe extraction**, .is_failure for checks
- **NO try/except fallbacks** - explicit error checking only

```python
def process_data(self, input_data: dict) -> FlextResult[ProcessedData]:
    if not input_data:
        return FlextResult[ProcessedData].fail("Input cannot be empty")
        
    validation_result = self._validate(input_data)
    if validation_result.is_failure:
        return FlextResult[ProcessedData].fail(f"Validation failed: {validation_result.error}")
        
    return FlextResult[ProcessedData].ok(processed_data)
```

### 3. Import Strategy (ROOT-LEVEL ONLY)
```python
# ✅ CORRECT
from flext_core import FlextResult, FlextLogger, FlextContainer
from flext_cli import FlextCliApi, FlextCliMain

# ❌ FORBIDDEN  
from flext_core.result import FlextResult  # Internal imports prohibited
import click  # Direct CLI imports prohibited
```

### 4. CLI Implementation (ZERO TOLERANCE)
```python
from flext_cli import FlextCliApi, FlextCliMain, FlextCliConfig

class FlextDbOracleCliService:
    def __init__(self) -> None:
        self._cli_api = FlextCliApi()
        
    def create_cli_interface(self) -> FlextResult[FlextCliMain]:
        main_cli = FlextCliMain(name="oracle-cli")
        # Use flext-cli for ALL output - NO Rich directly
        return FlextResult[FlextCliMain].ok(main_cli)
```

## Code Quality Standards

### Type Hints (STRICT)
- **Python 3.13** with strict typing everywhere
- **MyPy strict mode** with zero errors in src/
- **Pydantic v2** for data validation
- **Generic types** where applicable

### Naming Conventions
- **Classes**: `FlextDbOracle{Purpose}` prefix (e.g., FlextDbOracleApi, FlextDbOracleModels)
- **Methods**: snake_case with descriptive names
- **Variables**: snake_case, avoid abbreviations
- **Constants**: UPPER_CASE in FlextDbOracleConstants class
- **Private**: Leading underscore for internal methods/attributes

### Documentation
- **Docstrings**: Google style for all public APIs
- **Type annotations**: Comprehensive type hints
- **Examples**: Working code examples in docstrings
- **Security considerations**: Document Oracle security aspects

### File Organization
- **Single responsibility**: One main class per module
- **Logical grouping**: Related functionality together
- **Import order**: Standard library, third-party, flext, local
- **Module exports**: Explicit __all__ definitions

## ABSOLUTE PROHIBITIONS

### ❌ CLI Project Violations
- **FORBIDDEN**: Direct `import click`, `import rich`, or any Rich components
- **MANDATORY**: ALL CLI projects MUST use `flext-cli` exclusively

### ❌ Code Quality Violations  
- **FORBIDDEN**: Multiple classes per module
- **FORBIDDEN**: Helper functions outside classes
- **FORBIDDEN**: `try/except` fallback mechanisms
- **FORBIDDEN**: `# type: ignore` without specific error codes
- **FORBIDDEN**: `Any` types instead of proper annotations

### ❌ Oracle Integration Violations
- **FORBIDDEN**: Custom SQLAlchemy/oracledb implementations outside flext-db-oracle
- **FORBIDDEN**: Direct database connections bypassing flext-db-oracle
- **FORBIDDEN**: Incomplete Oracle abstraction layers