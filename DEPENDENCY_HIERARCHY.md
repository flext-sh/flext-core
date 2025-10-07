# FLEXT-CORE DEPENDENCY HIERARCHY

**Version**: 1.0.0  
**Date**: 2025-10-07  
**Status**: ✅ Circular Import Free

## Overview

This document defines the strict dependency hierarchy for flext-core modules, ensuring zero circular imports between layers.

## Layer Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 7: utilities.py                                       │
│ - FlextUtilities (validation, helpers)                      │
│ - Uses TYPE_CHECKING for config.py imports                  │
│ - Late import pattern in functions when needed              │
└─────────────────────────────────────────────────────────────┘
                          ↑ imports from
┌─────────────────────────────────────────────────────────────┐
│ Layer 6: models.py                                          │
│ - FlextModels (Entity, Value, AggregateRoot)                │
│ - Domain-driven design patterns                             │
│ - Imports: constants, typings, runtime, exceptions, result  │
└─────────────────────────────────────────────────────────────┘
                          ↑ imports from
┌─────────────────────────────────────────────────────────────┐
│ Layer 5: config.py                                          │
│ - FlextConfig (Pydantic Settings integration)               │
│ - NO imports from higher layers (models, utilities)         │
│ - Imports: constants, typings, runtime, exceptions, result  │
└─────────────────────────────────────────────────────────────┘
                          ↑ imports from
┌─────────────────────────────────────────────────────────────┐
│ Layer 4: loggings.py                                        │
│ - FlextLogger (structured logging)                          │
│ - NO imports from higher layers (config, models, utilities) │
│ - Imports: constants, typings, runtime, result              │
└─────────────────────────────────────────────────────────────┘
                          ↑ imports from
┌─────────────────────────────────────────────────────────────┐
│ Layer 3: protocols.py                                       │
│ - FlextProtocols (interface definitions)                    │
│ - Uses TYPE_CHECKING for models and result                  │
│ - Imports: typings (runtime)                                │
└─────────────────────────────────────────────────────────────┘
                          ↑ imports from
┌─────────────────────────────────────────────────────────────┐
│ Layer 2: result.py                                          │
│ - FlextResult[T] (Railway pattern)                          │
│ - Imports: constants, typings, exceptions                   │
└─────────────────────────────────────────────────────────────┘
                          ↑ imports from
┌─────────────────────────────────────────────────────────────┐
│ Layer 1: exceptions.py                                      │
│ - FlextExceptions (error hierarchy)                         │
│ - Imports: constants, typings, runtime                      │
└─────────────────────────────────────────────────────────────┘
                          ↑ imports from
┌─────────────────────────────────────────────────────────────┐
│ Layer 0.5: runtime.py                                       │
│ - FlextRuntime (type guards, utilities)                     │
│ - External library connectors (structlog, DI)               │
│ - Imports: constants ONLY                                   │
└─────────────────────────────────────────────────────────────┘
                          ↑ imports from
┌─────────────────────────────────────────────────────────────┐
│ Layer 0: constants.py, typings.py                           │
│ - FlextConstants (enums, error codes, patterns)             │
│ - FlextTypes (TypeVar definitions)                          │
│ - NO flext_core imports (pure Python foundation)            │
└─────────────────────────────────────────────────────────────┘
```

## Import Rules

### ✅ ALLOWED Import Patterns

1. **Downward Imports**: object layer can import from lower layers

   ```python
   # utilities.py (Layer 7) can import from:
   from flext_core.constants import FlextConstants  # Layer 0
   from flext_core.result import FlextResult        # Layer 2
   from flext_core.exceptions import FlextExceptions # Layer 1
   ```

2. **TYPE_CHECKING Imports**: Higher-layer types can be used in annotations

   ```python
   from typing import TYPE_CHECKING

   if TYPE_CHECKING:
       from flext_core.config import FlextConfig  # Type hint only

   def process_config(config: "FlextConfig") -> None:
       # Use string annotation for type hint
       pass
   ```

3. **Late Import Pattern**: Import inside functions when runtime access needed

   ```python
   def create_config() -> FlextResult[FlextConfig]:
       # Import at function level to break circular dependency
       from flext_core.config import FlextConfig
       return FlextResult[FlextConfig].ok(FlextConfig())
   ```

### ❌ FORBIDDEN Import Patterns

1. **Upward Imports**: NEVER import from higher layers at module level

   ```python
   # ❌ FORBIDDEN in config.py (Layer 5):
   from flext_core.utilities import FlextUtilities  # Layer 7
   from flext_core.models import FlextModels        # Layer 6
   ```

2. **Circular Module Imports**: NEVER create circular dependencies

   ```python
   # ❌ FORBIDDEN: utilities.py imports config at module level
   from flext_core.config import FlextConfig  # Circular with config → utilities
   ```

## Verification Commands

### Check for Circular Imports

```bash
# Test all layers import successfully
python3 -c "
import sys
sys.path.insert(0, 'src')

# Import each layer in order
from flext_core.constants import FlextConstants  # Layer 0
from flext_core.typings import FlextTypes        # Layer 0
from flext_core.runtime import FlextRuntime      # Layer 0.5
from flext_core.exceptions import FlextExceptions # Layer 1
from flext_core.result import FlextResult        # Layer 2
from flext_core.protocols import FlextProtocols  # Layer 3
from flext_core.loggings import FlextLogger      # Layer 4
from flext_core.config import FlextConfig        # Layer 5
from flext_core.models import FlextModels        # Layer 6
from flext_core.utilities import FlextUtilities  # Layer 7

print('✅ ALL LAYERS IMPORTED SUCCESSFULLY')
"
```

### Check for Upward Dependencies

```bash
# Verify config.py has no upward imports
grep -E "^from flext_core\.(models|utilities|loggings|protocols)" src/flext_core/config.py
# Should return nothing

# Verify models.py doesn't import utilities
grep -E "^from flext_core\.utilities" src/flext_core/models.py
# Should return nothing

# Verify loggings.py doesn't import from higher layers
grep -E "^from flext_core\.(config|models|utilities)" src/flext_core/loggings.py
# Should return nothing
```

## Key Architectural Decisions

### 1. utilities.py Circular Import Fix (2025-10-07)

**Problem**: utilities.py (Layer 7) had module-level import of config.py (Layer 5), creating circular dependency.

**Solution**:

- Removed module-level late import: `from flext_core.config import FlextConfig as _FlextConfig`
- Kept FlextConfig in TYPE_CHECKING block for type hints
- Moved actual import inside `create_flext_core_config()` function

**Files Changed**:

- `src/flext_core/utilities.py` lines 64-68 (removed module-level import)
- `src/flext_core/utilities.py` lines 3623-3659 (added function-level import)

### 2. Layer Isolation Verification (2025-10-07)

**Verified**:

- ✅ config.py (Layer 5): No imports from models (Layer 6) or utilities (Layer 7)
- ✅ models.py (Layer 6): No imports from utilities (Layer 7)
- ✅ loggings.py (Layer 4): No imports from config, models, or utilities

### 3. TYPE_CHECKING Pattern Usage

**Modules using TYPE_CHECKING correctly**:

- `protocols.py`: Uses TYPE_CHECKING for models and result imports
- `utilities.py`: Uses TYPE_CHECKING for config type hints

## Testing

### Test Results (2025-10-07)

```bash
# All 254 FlextResult tests passing
pytest tests/unit/test_result.py -v
# Result: 254 passed in 12.32s

# Import validation
python3 -c "import sys; sys.path.insert(0, 'src'); from flext_core import *"
# Result: ✅ Success - no circular import errors
```

## Maintenance Guidelines

### Adding New Modules

1. **Identify Layer**: Determine which existing layer the new module depends on
2. **Place Correctly**: New module goes in layer ABOVE its highest dependency
3. **Verify Imports**: Ensure no upward imports at module level
4. **Test Isolation**: Run import verification after adding module

### Modifying Existing Modules

1. **Check Current Layer**: Understand module's position in hierarchy
2. **Verify Dependencies**: Ensure new imports don't violate layer rules
3. **Use TYPE_CHECKING**: For higher-layer type hints, use TYPE_CHECKING pattern
4. **Late Import**: For runtime access to higher layers, use function-level imports
5. **Test Thoroughly**: Run full import validation and test suite

## References

- **CLAUDE.md**: Project-specific development standards
- **runtime.py docstring**: Layer 0.5 architecture documentation
- **result.py**: Railway pattern foundation (Layer 2)
- **config.py lines 35-36**: Example of avoiding circular imports with FlextLogger

## Validation Status

✅ **Layer 0**: constants.py, typings.py - Pure Python, no flext_core imports  
✅ **Layer 0.5**: runtime.py - Imports Layer 0 only  
✅ **Layer 1**: exceptions.py - Imports Layer 0, 0.5 only  
✅ **Layer 2**: result.py - Imports Layer 0, 1 only  
✅ **Layer 3**: protocols.py - Uses TYPE_CHECKING for higher layers  
✅ **Layer 4**: loggings.py - No upward dependencies verified  
✅ **Layer 5**: config.py - No upward dependencies verified  
✅ **Layer 6**: models.py - No imports from Layer 7 verified  
✅ **Layer 7**: utilities.py - TYPE_CHECKING + late import pattern verified

**Last Validated**: 2025-10-07  
**Status**: ✅ Zero circular imports across all layers
