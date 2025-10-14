# !/usr/bin/env python3
"""09 - FlextCore.Context: Request Context and Correlation Management.

This example demonstrates the COMPLETE FlextCore.Context API for managing
request contexts, correlation IDs, performance tracking, and context variables
across distributed operations in the FLEXT ecosystem.

Key Concepts Demonstrated:
- Context Variables: Thread-safe context storage and retrieval
- Correlation Tracking: Request correlation across services
- Service Context: Service metadata and information
- Request Context: HTTP request context management
- Performance Tracking: Request timing and metrics
- Serialization Context: Context serialization for distributed ops
- Utilities: Context manipulation helpers

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import json
import time
import warnings
from uuid import uuid4

from flext_core import FlextCore


class ContextManagementService(FlextCore.Service[FlextCore.Types.Dict]):
    """Service demonstrating ALL FlextCore.Context patterns with FlextCore.Mixins infrastructure.

    This service inherits from FlextCore.Service to demonstrate:
    - Inherited container property (FlextCore.Container singleton)
    - Inherited logger property (FlextCore.Logger with service context - CONTEXT FOCUS!)
    - Inherited context property (FlextCore.Context for request/correlation tracking)
    - Inherited config property (FlextCore.Config with application settings)
    - Inherited metrics property (FlextMetrics for observability)

    The focus is on demonstrating FlextCore.Context patterns: correlation tracking,
    context variables, service context, request context, and performance tracking,
    while leveraging complete FlextCore.Mixins infrastructure.
    """

    def __init__(self) -> None:
        """Initialize with inherited FlextCore.Mixins infrastructure.

        Note: No manual logger initialization needed!
        All infrastructure is inherited from FlextCore.Service base class:
        - self.logger: FlextCore.Logger with service context (ALREADY CONFIGURED!)
        - self.container: FlextCore.Container global singleton
        - self.context: FlextCore.Context for request/correlation tracking
        - self.config: FlextCore.Config with application settings
        - self.metrics: FlextMetrics for observability
        """
        super().__init__()

        # Demonstrate inherited logger (no manual instantiation needed!)
        self.logger.info(
            "ContextManagementService initialized with inherited infrastructure",
            extra={
                "service_type": "Context Management demonstration",
                "context_features": [
                    "correlation_tracking",
                    "context_variables",
                    "service_context",
                    "request_context",
                    "performance_tracking",
                ],
            },
        )

    def execute(self) -> FlextCore.Result[FlextCore.Types.Dict]:
        """Execute all context management demonstrations.

        Demonstrates inherited infrastructure alongside context patterns:
        - Inherited logger for structured logging
        - Inherited context for correlation tracking
        - Complete FlextCore.Context API patterns
        - Context variable management
        - Performance tracking

        Returns:
            FlextCore.Result[Dict] with demonstration summary including infrastructure details

        """
        self.logger.info("Starting comprehensive FlextCore.Context demonstration")

        try:
            # Context demonstrations
            self.demonstrate_context_variables()
            self.demonstrate_correlation_tracking()
            self.demonstrate_service_context()
            self.demonstrate_request_context()
            self.demonstrate_performance_tracking()
            self.demonstrate_serialization_context()
            self.demonstrate_context_utilities()

            # NEW: FlextCore.Result v0.9.9+ methods for context are demonstrated later

            # Deprecation warnings
            self.demonstrate_deprecated_patterns()

            summary: FlextCore.Types.Dict = {
                "demonstrations_completed": "12",
                "status": "completed",
                "context_managed": "true",
                "infrastructure": {
                    "logger": type(self.logger).__name__,
                    "container": type(self.container).__name__,
                    "context": type(self.context).__name__,
                    "config": type(self.config).__name__,
                },
                "context_features": {
                    "correlation_tracking": "true",
                    "context_variables": "true",
                    "service_context": "true",
                    "request_context": "true",
                    "performance_tracking": "true",
                },
            }

            self.logger.info(
                "FlextCore.Context demonstration completed successfully", extra=summary
            )

            return FlextCore.Result[FlextCore.Types.Dict].ok(summary)

        except Exception as e:
            error_msg = f"FlextCore.Context demonstration failed: {e}"
            self.logger.exception(error_msg)
            return FlextCore.Result[FlextCore.Types.Dict].fail(error_msg)

    # ========== CONTEXT VARIABLES ==========

    def demonstrate_context_variables(self) -> None:
        """Show context variable management."""
        print("\n=== Context Variables ===")

        # Set context variables using the actual context variables
        FlextCore.Context.Variables.Request.USER_ID.set("USER-123")
        FlextCore.Context.Variables.Request.REQUEST_ID.set(str(uuid4()))
        FlextCore.Context.Variables.Service.SERVICE_NAME.set("example-service")

        print("✅ Context variables set:")
        print(f"  User ID: {FlextCore.Context.Variables.Request.USER_ID.get()}")
        print(f"  Request ID: {FlextCore.Context.Variables.Request.REQUEST_ID.get()}")
        print(
            f"  Service Name: {FlextCore.Context.Variables.Service.SERVICE_NAME.get()}"
        )

        # Get with default
        correlation_id = FlextCore.Context.Variables.Correlation.CORRELATION_ID.get()
        print(f"  Correlation ID (with default): {correlation_id}")

        # Get all variables (manually collect them)
        all_vars: dict[str, str | None] = {
            "user_id": FlextCore.Context.Variables.Request.USER_ID.get(),
            "request_id": FlextCore.Context.Variables.Request.REQUEST_ID.get(),
            "service_name": FlextCore.Context.Variables.Service.SERVICE_NAME.get(),
            "correlation_id": FlextCore.Context.Variables.Correlation.CORRELATION_ID.get(),
        }
        print(f"\nAll context variables: {len(all_vars)} items")
        for key, value in list(all_vars.items())[:3]:
            print(f"  {key}: {value}")

        # Update variable
        FlextCore.Context.Variables.Request.USER_ID.set("USER-789")
        print(f"\nUpdated user ID: {FlextCore.Context.Variables.Request.USER_ID.get()}")

        # Clear variable (set to None)
        FlextCore.Context.Variables.Request.USER_ID.set(None)
        print(
            f"User ID after clear: {FlextCore.Context.Variables.Request.USER_ID.get()}"
        )

        # Check existence
        has_user = FlextCore.Context.Variables.Request.USER_ID.get() is not None
        has_theme = FlextCore.Context.Variables.Service.SERVICE_NAME.get() is not None
        print(f"\nHas user_id: {has_user}")
        print(f"Has service_name: {has_theme}")

        # Clear all variables (set to None)
        FlextCore.Context.Variables.Request.USER_ID.set(None)
        FlextCore.Context.Variables.Request.REQUEST_ID.set(None)
        FlextCore.Context.Variables.Service.SERVICE_NAME.set(None)
        FlextCore.Context.Variables.Correlation.CORRELATION_ID.set(None)
        print("\nAfter clear: All variables cleared")

    # ========== CORRELATION TRACKING ==========

    def demonstrate_correlation_tracking(self) -> None:
        """Show correlation ID tracking across operations."""
        print("\n=== Correlation Tracking ===")

        # Create new correlation ID
        correlation_id = FlextCore.Context.Correlation.generate_correlation_id()
        print(f"✅ Created correlation ID: {correlation_id}")

        # Set current correlation ID
        FlextCore.Context.Correlation.set_correlation_id(correlation_id)
        print(
            f"Current correlation: {FlextCore.Context.Correlation.get_correlation_id()}"
        )

        # Check if correlation exists
        has_correlation = FlextCore.Context.Correlation.get_correlation_id() is not None
        print(f"Has correlation: {has_correlation}")

        # Create child correlation (for sub-operations)
        with FlextCore.Context.Correlation.new_correlation() as child_id:
            print(f"Child correlation: {child_id}")

        # Get correlation chain (manually collect)
        chain = [
            FlextCore.Context.Correlation.get_correlation_id(),
            FlextCore.Context.Correlation.get_parent_correlation_id(),
        ]
        chain = [cid for cid in chain if cid is not None]
        print(f"Correlation chain: {len(chain)} items")
        for i, cid in enumerate(chain):
            print(f"  Level {i}: {cid}")

        # Validate correlation ID format
        is_valid = len(correlation_id) > 0
        print(f"\nIs valid correlation: {is_valid}")

        # Invalid correlation
        invalid_id = "invalid-id"
        is_invalid = len(invalid_id) > 0  # invalid_id is always a string, not None
        print(f"Is 'invalid-id' valid: {is_invalid}")

        # Clear correlation
        FlextCore.Context.Variables.Correlation.CORRELATION_ID.set(None)
        print(
            f"After clear: {FlextCore.Context.Correlation.get_correlation_id() or 'None'}"
        )

    # ========== SERVICE CONTEXT ==========

    def demonstrate_service_context(self) -> None:
        """Show service metadata and context."""
        print("\n=== Service Context ===")

        # Set service metadata
        FlextCore.Context.Service.set_service_name("order-service")
        FlextCore.Context.Service.set_service_version("1.0.0")

        # Get service info
        service_name = FlextCore.Context.Service.get_service_name()
        service_version = FlextCore.Context.Service.get_service_version()
        print("✅ Service information:")
        print(f"  name: {service_name}")
        print(f"  version: {service_version}")

        # Get specific metadata
        service_name = FlextCore.Context.Service.get_service_name()
        service_version = FlextCore.Context.Service.get_service_version()
        print(f"\nService: {service_name} v{service_version}")

        # Update metadata
        FlextCore.Context.Service.set_service_name("order-service-updated")
        FlextCore.Context.Service.set_service_version("1.1.0")

        # Get all metadata
        service_name = FlextCore.Context.Service.get_service_name()
        service_version = FlextCore.Context.Service.get_service_version()
        print("\nAll metadata (2 items):")
        print(f"  name: {service_name}")
        print(f"  version: {service_version}")

        # Clear service context
        FlextCore.Context.Variables.Service.SERVICE_NAME.set(None)
        FlextCore.Context.Variables.Service.SERVICE_VERSION.set(None)
        print("After clear: service context cleared")

    # ========== REQUEST CONTEXT ==========

    def demonstrate_request_context(self) -> None:
        """Show HTTP request context management."""
        print("\n=== Request Context ===")

        # Set request context
        request_data: FlextCore.Types.Dict = {
            "method": "POST",
            "path": "/api/orders",
            "user_id": "USER-456",
            "request_id": str(uuid4()),
            "headers": {
                "Content-Type": "application/json",
                "Authorization": "Bearer token123",
                "X-Request-ID": str(uuid4()),
            },
            "query": {"page": "1", "limit": "10"},
            "body": {"product_id": "PROD-123", "quantity": "2"},
            "ip": "192.168.1.100",
            "user_agent": "Mozilla/5.0",
        }

        # Extract string values for context setting
        user_id = str(request_data["user_id"])
        request_id = str(request_data["request_id"])
        FlextCore.Context.Request.set_user_id(user_id)
        FlextCore.Context.Request.set_request_id(request_id)
        print("✅ Request context set")

        # Get full request
        user_id = FlextCore.Context.Request.get_user_id() or "Not set"
        request_id = FlextCore.Context.Request.get_request_id() or "Not set"
        print(f"User: {user_id}")
        print(f"Request ID: {request_id}")

        # Get specific parts
        user_id = FlextCore.Context.Request.get_user_id() or "Not set"
        request_id = FlextCore.Context.Request.get_request_id() or "Not set"
        print("\nRequest data (2 items):")
        print(f"  user_id: {user_id}")
        print(f"  request_id: {request_id}")

        # Context data already demonstrated above

        # Clear request context
        FlextCore.Context.Variables.Request.USER_ID.set(None)
        FlextCore.Context.Variables.Request.REQUEST_ID.set(None)
        print("\nAfter clear: request context cleared")

    # ========== PERFORMANCE TRACKING ==========

    def demonstrate_performance_tracking(self) -> None:
        """Show performance metrics and timing."""
        print("\n=== Performance Tracking ===")

        # Start timing an operation
        FlextCore.Context.Request.set_operation_name("database_query")
        FlextCore.Context.Performance.set_operation_start_time()
        print("✅ Timer started: database_query")

        # Simulate work
        time.sleep(0.1)

        # End timing
        start_time = FlextCore.Context.Performance.get_operation_start_time()
        if start_time:
            duration = time.time() - start_time.timestamp()
            print(f"Duration: {duration:.3f} seconds")

        # Record metric
        FlextCore.Context.Performance.set_operation_metadata({
            "items_processed": "150",
            "cache_hits": "45",
            "cache_misses": "5",
        })

        # Get metrics
        metadata = FlextCore.Context.Performance.get_operation_metadata()
        print(f"\nMetrics ({len(metadata or {})} items):")
        if metadata:
            for name, value in metadata.items():
                print(f"  {name}: {value}")

        # Get specific metric
        items = metadata.get("items_processed") if metadata else None
        print(f"\nItems processed: {items}")

        # Record timing directly
        FlextCore.Context.Performance.add_operation_metadata("api_call", "0.250")

        # Get all timings
        metadata = FlextCore.Context.Performance.get_operation_metadata()
        print(f"\nTimings ({len(metadata or {})} items):")
        if metadata:
            for name, duration_obj in metadata.items():
                duration_str: str = str(duration_obj)
                if duration_str.replace(".", "").isdigit():
                    print(f"  {name}: {duration_str}s")

        # Get summary
        metadata = FlextCore.Context.Performance.get_operation_metadata()
        print("\nPerformance summary:")
        print(f"  Total operations: {len(metadata or {})}")
        print("  Average time: 0.175s")

        # Clear performance data
        FlextCore.Context.Variables.Performance.OPERATION_NAME.set(None)
        FlextCore.Context.Variables.Performance.OPERATION_START_TIME.set(None)
        FlextCore.Context.Variables.Performance.OPERATION_METADATA.set(None)
        print("After clear: performance data cleared")

    # ========== SERIALIZATION CONTEXT ==========

    def demonstrate_serialization_context(self) -> None:
        """Show context serialization for distributed operations."""
        print("\n=== Serialization Context ===")

        # Set up context data
        FlextCore.Context.Variables.Request.USER_ID.set("USER-123")
        FlextCore.Context.Correlation.set_correlation_id(str(uuid4()))
        FlextCore.Context.Request.set_user_id("USER-123")
        FlextCore.Context.Request.set_request_id(str(uuid4()))

        # Serialize context
        serialized = FlextCore.Context.Serialization.get_full_context()
        print("✅ Context serialized")
        print(f"Serialized data keys: {list(serialized.keys())}")

        # Convert to JSON
        json_context = json.dumps(serialized)
        print(f"\nJSON length: {len(json_context)} characters")
        print(f"First 100 chars: {json_context[:100]}...")

        # Clear context
        FlextCore.Context.Variables.Request.USER_ID.set(None)
        FlextCore.Context.Variables.Request.REQUEST_ID.set(None)
        FlextCore.Context.Variables.Correlation.CORRELATION_ID.set(None)
        print("\n✅ Context cleared")

        # Deserialize context
        FlextCore.Context.Serialization.set_from_context(serialized)
        print("✅ Context deserialized")

        # Verify restoration
        print(f"Restored user_id: {FlextCore.Context.Request.get_user_id()}")
        print(
            f"Restored correlation: {FlextCore.Context.Correlation.get_correlation_id()}"
        )
        print(f"Restored request_id: {FlextCore.Context.Request.get_request_id()}")

        # From JSON
        FlextCore.Context.Variables.Request.USER_ID.set(None)
        FlextCore.Context.Variables.Request.REQUEST_ID.set(None)
        FlextCore.Context.Variables.Correlation.CORRELATION_ID.set(None)
        parsed_context = json.loads(json_context)
        FlextCore.Context.Serialization.set_from_context(parsed_context)
        print(f"\nRestored from JSON: {FlextCore.Context.Request.get_user_id()}")

    # ========== CONTEXT UTILITIES ==========

    def demonstrate_context_utilities(self) -> None:
        """Show context utility functions."""
        print("\n=== Context Utilities ===")

        # Merge contexts
        context1: FlextCore.Types.StringDict = {"user": "alice", "role": "admin"}
        context2: FlextCore.Types.StringDict = {"tenant": "acme", "role": "superadmin"}

        merged: FlextCore.Types.StringDict = {**context1, **context2}
        print("✅ Contexts merged:")
        print(f"  Result: {merged}")

        # Copy context
        original: FlextCore.Types.Dict = {
            "data": {"nested": "value"},
            "id": 123,
        }
        copied = original.copy()
        print("\n✅ Context copied")
        print(f"  Same object: {original is copied}")
        print(f"  Equal values: {original == copied}")

        # Clear all contexts
        FlextCore.Context.Utilities.clear_context()
        print("\n✅ All contexts cleared")

        # Get context snapshot
        FlextCore.Context.Variables.Request.USER_ID.set("example")
        FlextCore.Context.Correlation.set_correlation_id(str(uuid4()))

        snapshot = FlextCore.Context.Serialization.get_full_context()
        print("\nContext snapshot:")
        print(f"  Keys: {list(snapshot.keys())}")

        # Restore from snapshot
        FlextCore.Context.Utilities.clear_context()
        FlextCore.Context.Serialization.set_from_context(snapshot)
        print("\n✅ Restored from snapshot")
        print(f"  User ID: {FlextCore.Context.Request.get_user_id()}")

    # ========== CONTEXT ==========

    def demonstrate_context(self) -> None:
        """Show context preservation across operations."""
        print("\n=== Context Management ===")

        # Set context in main thread
        FlextCore.Context.Variables.Request.USER_ID.set("-user-123")
        FlextCore.Context.Correlation.set_correlation_id(str(uuid4()))

        correlation = FlextCore.Context.Correlation.get_correlation_id()
        print(f"Main thread correlation: {correlation}")

        # Define tasks that use context
        def sync_task(task_id: int) -> str:
            """Synchronous task that accesses context."""
            # Context should be preserved
            user = FlextCore.Context.Request.get_user_id()
            corr = FlextCore.Context.Correlation.get_correlation_id()

            # Simulate work
            time.sleep(0.1)

            return f"Task {task_id}: user={user}, correlation={corr[:8] if corr else 'None'}..."

        # Run multiple tasks
        results = [sync_task(i) for i in range(3)]

        print("\nSync task results:")
        for result in results:
            print(f"  {result}")

        # Context still available after operations
        print(f"\nContext after sync tasks: {FlextCore.Context.Request.get_user_id()}")

    # ========== CONTEXT PATTERNS ==========

    def demonstrate_context_patterns(self) -> None:
        """Show common context usage patterns."""
        print("\n=== Context Patterns ===")

        # Pattern 1: Request processing with correlation
        print("\n1. Request Processing Pattern:")

        # Incoming request
        request_id = str(uuid4())
        FlextCore.Context.Correlation.set_correlation_id(request_id)
        FlextCore.Context.Request.set_user_id("USER-123")
        FlextCore.Context.Request.set_request_id(request_id)
        FlextCore.Context.Request.set_operation_name("request_processing")
        FlextCore.Context.Performance.set_operation_start_time()

        # Process request
        print(f"  Processing request {request_id[:8]}...")
        time.sleep(0.05)  # Simulate work

        # Complete processing
        start_time = FlextCore.Context.Performance.get_operation_start_time()
        duration = (time.time() - start_time.timestamp()) if start_time else 0.0
        print(f"  Completed in {duration:.3f}s")

        # Pattern 2: Distributed tracing
        print("\n2. Distributed Tracing Pattern:")

        # Parent service
        parent_correlation = FlextCore.Context.Correlation.generate_correlation_id()
        FlextCore.Context.Correlation.set_correlation_id(parent_correlation)
        print(f"  Parent service: {parent_correlation[:8]}...")

        # Child service call
        with FlextCore.Context.Correlation.new_correlation() as child_correlation:
            print(f"  Child service: {child_correlation[:8]}...")

        # Get full trace
        chain = [
            FlextCore.Context.Correlation.get_correlation_id(),
            FlextCore.Context.Correlation.get_parent_correlation_id(),
        ]
        chain = [cid for cid in chain if cid is not None]
        print(f"  Trace chain: {len(chain)} hops")

        # Pattern 3: Multi-tenant context
        print("\n3. Multi-Tenant Pattern:")

        # Set tenant context
        FlextCore.Context.Variables.Request.USER_ID.set("tenant-123")
        FlextCore.Context.Variables.Service.SERVICE_NAME.set("Acme Corp")
        FlextCore.Context.Variables.Service.SERVICE_VERSION.set("premium")

        # Access tenant info throughout request
        tenant = FlextCore.Context.Request.get_user_id()
        tier = FlextCore.Context.Service.get_service_version()
        print(f"  Processing for {tenant} (tier: {tier})")

    # ========== DEPRECATED PATTERNS ==========

    # ========== NEW FlextCore.Result METHODS (v0.9.9+) ==========

    def demonstrate_new_flextresult_methods(self) -> None:
        """Demonstrate the 5 new FlextCore.Result methods in context management.

        Shows how the new v0.9.9+ methods integrate with context operations:
        - from_callable: Safe context operations
        - flow_through: Context pipeline composition
        - lash: Context fallback recovery
        - alt: Context provider alternatives
        - value_or_call: Lazy context loading
        """
        print("\n=== NEW FlextCore.Result METHODS (v0.9.9+) ===")

        # 1. from_callable - Safe Context Operations
        print("\n1. from_callable: Safe Context Operations")

        def risky_context_operation() -> FlextCore.Types.Dict:
            """Context operation that might raise exceptions."""
            user_id = FlextCore.Context.Variables.Request.USER_ID.get()
            if not user_id:
                msg = "User ID not found in context"
                raise FlextCore.Exceptions.ValidationError(
                    msg,
                    field="user_id",
                    value=None,
                )
            return {
                "user_id": user_id,
                "request_id": FlextCore.Context.Variables.Request.REQUEST_ID.get()
                or "unknown",
                "service": FlextCore.Context.Variables.Service.SERVICE_NAME.get()
                or "unknown",
            }

        # Set up context
        FlextCore.Context.Variables.Request.USER_ID.set("USER-SAFE-123")
        FlextCore.Context.Variables.Request.REQUEST_ID.set(str(uuid4()))
        FlextCore.Context.Variables.Service.SERVICE_NAME.set("context-service")

        # Safe context extraction without try/except
        result = FlextCore.Result[FlextCore.Types.Dict].from_callable(
            risky_context_operation
        )
        if result.is_success:
            context_data = result.unwrap()
            print(f"✅ Context extracted safely: user={context_data['user_id']}")
        else:
            print(f"❌ Context extraction failed: {result.error}")

        # 2. flow_through - Context Pipeline Composition
        print("\n2. flow_through: Context Pipeline Composition")

        def validate_user_context(
            data: FlextCore.Types.Dict,
        ) -> FlextCore.Result[FlextCore.Types.Dict]:
            """Validate user context exists."""
            if not data.get("user_id"):
                return FlextCore.Result[FlextCore.Types.Dict].fail(
                    "User context required"
                )
            return FlextCore.Result[FlextCore.Types.Dict].ok(data)

        def enrich_with_correlation(
            data: FlextCore.Types.Dict,
        ) -> FlextCore.Result[FlextCore.Types.Dict]:
            """Add correlation ID from context."""
            correlation_id = FlextCore.Context.Correlation.get_correlation_id()
            enriched: FlextCore.Types.Dict = {**data, "correlation_id": correlation_id}
            return FlextCore.Result[FlextCore.Types.Dict].ok(enriched)

        def add_service_metadata(
            data: FlextCore.Types.Dict,
        ) -> FlextCore.Result[FlextCore.Types.Dict]:
            """Add service metadata from context."""
            enriched: FlextCore.Types.Dict = {
                **data,
                "service_name": FlextCore.Context.Service.get_service_name(),
                "service_version": FlextCore.Context.Service.get_service_version(),
            }
            return FlextCore.Result[FlextCore.Types.Dict].ok(enriched)

        def validate_complete_context(
            data: FlextCore.Types.Dict,
        ) -> FlextCore.Result[FlextCore.Types.Dict]:
            """Ensure all required context fields present."""
            required = ["user_id", "correlation_id", "service_name"]
            missing = [f for f in required if not data.get(f)]
            if missing:
                return FlextCore.Result[FlextCore.Types.Dict].fail(
                    f"Missing context fields: {missing}"
                )
            return FlextCore.Result[FlextCore.Types.Dict].ok(data)

        # Flow through context enrichment pipeline
        pipeline_context_data: FlextCore.Types.Dict = {
            "user_id": FlextCore.Context.Variables.Request.USER_ID.get(),
            "timestamp": time.time(),
        }
        pipeline_result = (
            FlextCore.Result[FlextCore.Types.Dict]
            .ok(pipeline_context_data)
            .flow_through(
                validate_user_context,
                enrich_with_correlation,
                add_service_metadata,
                validate_complete_context,
            )
        )

        if pipeline_result.is_success:
            enriched_context = pipeline_result.unwrap()
            print(f"✅ Context pipeline complete: {len(enriched_context)} fields")
            print(f"   User: {enriched_context.get('user_id')}")
            print(f"   Correlation: {enriched_context.get('correlation_id')}")
            print(f"   Service: {enriched_context.get('service_name')}")
        else:
            print(f"❌ Pipeline failed: {pipeline_result.error}")

        # 3. lash - Context Fallback Recovery
        print("\n3. lash: Context Fallback Recovery")

        def try_get_user_from_context() -> FlextCore.Result[FlextCore.Types.Dict]:
            """Try to get user from primary context."""
            user_id = FlextCore.Context.Variables.Request.USER_ID.get()
            if not user_id:
                return FlextCore.Result[FlextCore.Types.Dict].fail(
                    "User not in request context"
                )
            return FlextCore.Result[FlextCore.Types.Dict].ok({
                "user_id": user_id,
                "source": "context",
            })

        def fallback_to_default_user(
            error: str,
        ) -> FlextCore.Result[FlextCore.Types.Dict]:
            """Fallback to default anonymous user."""
            print(f"   ⚠️  Context lookup failed: {error}, using fallback...")
            return FlextCore.Result[FlextCore.Types.Dict].ok({
                "user_id": "anonymous",
                "source": "fallback",
            })

        # Clear user context to trigger fallback
        FlextCore.Context.Variables.Request.USER_ID.set(None)

        # Try context, fallback to default on failure
        user_result = try_get_user_from_context().lash(fallback_to_default_user)

        if user_result.is_success:
            user_data = user_result.unwrap()
            print(
                f"✅ User resolved: {user_data['user_id']} (source: {user_data['source']})"
            )
        else:
            print(f"❌ All user resolution failed: {user_result.error}")

        # 4. alt - Context Provider Alternatives
        print("\n4. alt: Context Provider Alternatives")

        def get_cached_service_config() -> FlextCore.Result[FlextCore.Types.Dict]:
            """Try to get service config from cache context."""
            # Simulate cache miss
            return FlextCore.Result[FlextCore.Types.Dict].fail(
                "Config not in cache context"
            )

        def get_default_service_config() -> FlextCore.Result[FlextCore.Types.Dict]:
            """Get default service configuration."""
            return FlextCore.Result[FlextCore.Types.Dict].ok({
                "timeout": 30,
                "max_retries": 3,
                "log_level": "INFO",
                "environment": FlextCore.Context.Service.get_service_name()
                or "unknown",
            })

        # Try cached config, fall back to default
        config = get_cached_service_config().alt(get_default_service_config())

        if config.is_success:
            config_data = config.unwrap()
            print(f"✅ Service config loaded: timeout={config_data.get('timeout')}s")
            print(f"   Max retries: {config_data.get('max_retries')}")
            print(f"   Environment: {config_data.get('environment')}")
        else:
            print(f"❌ No config sources available: {config.error}")

        # 5. value_or_call - Lazy Context Loading
        print("\n5. value_or_call: Lazy Context Loading")

        def load_expensive_user_profile() -> dict[str, str]:
            """Expensive user profile loading from database."""
            print("   ⚙️  Loading user profile from database...")
            time.sleep(0.1)  # Simulate expensive operation
            return {
                "user_id": "USER-LOADED-456",
                "name": "John Doe",
                "email": "john@example.com",
                "role": "admin",
                "preferences": json.dumps({"theme": "dark", "lang": "en"}),
            }

        # User not in context, lazy-load profile
        user_context: FlextCore.Result[dict[str, str]] = FlextCore.Result[
            dict[str, str]
        ].fail(
            "User not in context",
        )

        # Only loads if result is failure
        user_profile = user_context.value_or_call(load_expensive_user_profile)
        print(f"✅ User profile loaded: {user_profile.get('name')}")
        print(f"   Email: {user_profile.get('email')}")
        print(f"   Role: {user_profile.get('role')}")

        # When user is in context, doesn't call the expensive function
        cached_user: FlextCore.Result[dict[str, str]] = FlextCore.Result[
            dict[str, str]
        ].ok({
            "user_id": "USER-CACHED-789",
            "name": "Jane Cached",
        })
        cached_profile = cached_user.value_or_call(
            load_expensive_user_profile
        )  # Won't execute
        print(f"✅ Cached profile used: {cached_profile}")

        print("\n✅ NEW FlextCore.Result METHODS DEMONSTRATED!")
        print("All 5 methods integrated with context management operations")

    def demonstrate_deprecated_patterns(self) -> None:
        """Show deprecated context patterns."""
        print("\n=== ⚠️ DEPRECATED PATTERNS ===")

        # OLD: Global variables for context (DEPRECATED)
        warnings.warn(
            "Global variables for context are DEPRECATED! Use FlextCore.Context.",
            DeprecationWarning,
            stacklevel=2,
        )
        print("❌ OLD WAY (global variables):")
        print("CURRENT_USER = None")
        print("REQUEST_ID = None")
        print("def process():")
        print("    global CURRENT_USER")
        print("    CURRENT_USER = 'user123'")

        print("\n✅ CORRECT WAY (FlextCore.Context):")
        print("FlextCore.Context.Variables.set('user_id', 'user123')")
        print("FlextCore.Context.Correlation.set(request_id)")

        # OLD: Thread locals (DEPRECATED)
        warnings.warn(
            "Thread locals are DEPRECATED! Use FlextCore.Context for thread safety.",
            DeprecationWarning,
            stacklevel=2,
        )
        print("\n❌ OLD WAY (thread locals):")
        print("import threading")
        print("local = threading.local()")
        print("local.user = 'user123'")

        print("\n✅ CORRECT WAY (FlextCore.Context):")
        print("FlextCore.Context.Variables.set('user', 'user123')")
        print("# Thread-safe by design")

        # OLD: Manual correlation tracking (DEPRECATED)
        warnings.warn(
            "Manual correlation is DEPRECATED! Use FlextCore.Context.Correlation.",
            DeprecationWarning,
            stacklevel=2,
        )
        print("\n❌ OLD WAY (manual):")
        print("correlation_id = str(uuid4())")
        print("logger.info('message', extra={'correlation_id': correlation_id})")

        print("\n✅ CORRECT WAY (FlextCore.Context):")
        print("correlation = FlextCore.Context.Correlation.create()")
        print("FlextCore.Context.Correlation.set(correlation)")
        print("# Automatically available to all components")


def main() -> None:
    """Main entry point demonstrating all FlextCore.Context capabilities."""
    service = ContextManagementService()

    print("=" * 60)
    print("FLEXTCONTEXT COMPLETE API DEMONSTRATION")
    print("Request Context and Correlation Management")
    print("=" * 60)

    # Core patterns
    service.demonstrate_context_variables()
    service.demonstrate_correlation_tracking()

    # Service patterns
    service.demonstrate_service_context()
    service.demonstrate_request_context()

    # Advanced features
    service.demonstrate_performance_tracking()
    service.demonstrate_serialization_context()

    # Utilities and patterns
    service.demonstrate_context_utilities()
    service.demonstrate_context_patterns()

    # Context demo (synchronous execution)
    print("\n=== Running Context Demo ===")
    service.demonstrate_context()

    # New FlextCore.Result methods (v0.9.9+)
    service.demonstrate_new_flextresult_methods()

    # Deprecation warnings
    service.demonstrate_deprecated_patterns()

    print("\n" + "=" * 60)
    print("✅ ALL FlextCore.Context methods demonstrated!")
    print("🎯 Next: See 10_cqrs_patterns.py for CQRS patterns")
    print("=" * 60)


if __name__ == "__main__":
    main()
