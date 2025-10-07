"""Public export surface for FLEXT-Core 1.0.0 - Foundation of the FLEXT Ecosystem.

FLEXT-CORE OPTIMIZATION PATTERNS SHOWCASE
==========================================

This module demonstrates the complete unified module optimization patterns
implemented across the FLEXT ecosystem. All exports follow strict optimization
principles including namespace class patterns, flext-core integration, and
domain library architecture.

KEY OPTIMIZATION PATTERNS DEMONSTRATED:

🚀 NAMESPACE CLASS PATTERN
All major exports use the single-class-with-nested-namespaces pattern:
- FlextConstants: Centralized constants with nested groups
- FlextModels: DDD base classes (Entity, Value, AggregateRoot)
- FlextTypes: Type system with 40+ TypeVars and modern Python 3.13+ patterns
- FlextExceptions: Exception hierarchy with error codes
- FlextProtocols: Interface definitions for domain contracts

🔧 FLEXT-CORE INTEGRATION
All components integrate with the complete flext-core ecosystem:
- FlextConfig: Pydantic 2.11+ BaseSettings with environment integration
- FlextResult: Railway pattern for monadic error handling
- FlextLogger: Structured logging with context and correlation
- FlextService: Service base class with dependency injection
- FlextContainer: Global DI container with type safety

📚 DOMAIN LIBRARY ARCHITECTURE
This module serves as the foundation for 32+ dependent projects:
- Domain libraries (flext-*) extend these patterns
- Enterprise tools consume domain libraries (no direct imports)
- Zero tolerance for anti-patterns and architectural violations

USAGE EXAMPLES:

```python
# ✅ CORRECT - Complete flext-core integration
from flext_core import (
    FlextResult,  # Railway pattern foundation
    FlextConfig,  # Configuration with environment integration
    FlextConstants,  # Centralized constants
    FlextModels,  # DDD base classes
    FlextTypes,  # Type system with 40+ TypeVars
    FlextExceptions,  # Exception hierarchy
    FlextProtocols,  # Interface definitions
    FlextLogger,  # Structured logging
    FlextService,  # Service base class
    FlextContainer,  # Dependency injection


OPTIMIZATION PRINCIPLES DEMONSTRATED:

✅ Single Source of Truth: All types, constants, and utilities centralized
✅ Namespace Class Pattern: Single class with nested namespaces for organization
✅ flext-core Integration: All components work together seamlessly
✅ Railway Pattern: Monadic error handling eliminates exceptions in business logic
✅ Type Safety: Complete type annotations with 40+ TypeVars
✅ Configuration Integration: Environment-based configuration with validation
✅ Dependency Injection: Global container with type-safe service registration
✅ Domain Library Foundation: Base for all 32+ ecosystem projects

ARCHITECTURAL GUARANTEES:

🔒 ZERO BREAKING CHANGES: API compatibility maintained across versions
🔒 QUALITY ASSURANCE: Zero linting errors, complete type safety
🔒 ECOSYSTEM STABILITY: Foundation for 32+ dependent projects
🔒 DOMAIN LIBRARY COMPLIANCE: Strict adherence to domain library principles

See individual module documentation for detailed usage patterns and examples.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core.__version__ import __version__, __version_info__
from flext_core.api import FlextCore
from flext_core.bus import FlextBus
from flext_core.config import FlextConfig
from flext_core.constants import FlextConstants
from flext_core.container import FlextContainer
from flext_core.context import FlextContext
from flext_core.decorators import (
    combined,
    inject,
    log_operation,
    railway,
    track_performance,
)
from flext_core.dispatcher import FlextDispatcher
from flext_core.exceptions import FlextExceptions
from flext_core.handlers import FlextHandlers
from flext_core.loggings import FlextLogger
from flext_core.mixins import FlextMixins
from flext_core.models import FlextModels
from flext_core.processors import FlextProcessors
from flext_core.protocols import FlextProtocols
from flext_core.registry import FlextRegistry
from flext_core.result import FlextResult
from flext_core.runtime import FlextRuntime
from flext_core.service import FlextService
from flext_core.typings import FlextTypes
from flext_core.utilities import FlextUtilities

__all__ = [
    "FlextBus",
    "FlextConfig",
    "FlextConstants",
    "FlextContainer",
    "FlextContext",
    "FlextCore",
    "FlextDispatcher",
    "FlextExceptions",
    "FlextHandlers",
    "FlextLogger",
    "FlextMixins",
    "FlextModels",
    "FlextProcessors",
    "FlextProtocols",
    "FlextRegistry",
    "FlextResult",
    "FlextRuntime",
    "FlextService",
    "FlextTypes",
    "FlextUtilities",
    "__version__",
    "__version_info__",
    "combined",
    "inject",
    "log_operation",
    "railway",
    "track_performance",
]
