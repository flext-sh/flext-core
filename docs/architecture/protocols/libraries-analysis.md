# FlextProtocols Libraries Analysis and Integration Opportunities

**Version**: 0.9.0  
**Module**: `flext_core.protocols`  
**Target Audience**: Software Architects, Technical Leads, Platform Engineers

## Executive Summary

This analysis examines integration opportunities for FlextProtocols across the 33+ FLEXT ecosystem libraries, identifying specific patterns for contract standardization, type safety enforcement, and Clean Architecture compliance. The analysis reveals significant potential for protocol-driven development across distributed services with substantial benefits for architectural consistency and development productivity.

**Key Finding**: FlextProtocols provides critical contract infrastructure for the entire FLEXT ecosystem, but is currently underutilized with fragmented protocol implementations and inconsistent interface patterns across projects.

---

## 🎯 Strategic Integration Matrix

| **Library**         | **Priority** | **Current Protocol Usage** | **Integration Opportunity**      | **Expected Impact**             |
| ------------------- | ------------ | -------------------------- | -------------------------------- | ------------------------------- |
| **flext-web**       | 🟡 Partial   | Custom protocol extensions | Standardize with FlextProtocols  | High - Interface consistency    |
| **flext-meltano**   | 🔴 Critical  | Minimal plugin protocols   | Full protocol architecture       | High - Plugin standardization   |
| **flext-ldap**      | 🔥 Critical  | No protocol usage          | Complete protocol adoption       | High - Connection management    |
| **flext-api**       | 🔥 Critical  | No protocol standards      | Handler and service protocols    | High - API consistency          |
| **flext-db-oracle** | 🟡 High      | No connection protocols    | Infrastructure protocols         | Medium - Connection abstraction |
| **flext-grpc**      | 🟡 High      | No service protocols       | Service and connection protocols | Medium - gRPC standardization   |

---

## 🔍 Library-Specific Analysis

### 1. flext-web (Partial Implementation - Enhancement Focus)

**Current State**: Custom protocol extensions in `flext_web.protocols`

#### Current Implementation Analysis

```python
# Current flext-web protocol approach
class FlextWebProtocols(FlextProtocols):
    """Web-specific protocol extensions."""

    class WebServiceProtocol(Protocol):
        """Custom web service protocol."""
        # Custom implementation not following FlextProtocols hierarchy
```

#### Integration Opportunities

##### A. Standardize Web Services with Domain.Service

```python
# Recommended: Align with FlextProtocols.Domain.Service
class FlextWebService(FlextProtocols.Domain.Service):
    """Web service following standard Domain.Service protocol."""

    def __init__(self, app_config: dict):
        self.app_config = app_config
        self.app = None
        self._running = False

    def __call__(self, *args: object, **kwargs: object) -> object:
        """Handle web requests through callable interface."""
        request = kwargs.get("request")
        if request:
            return self._process_web_request(request)
        return FlextResult[dict].fail("No request provided")

    def start(self) -> object:
        """Start web application server."""
        try:
            # Initialize Flask/FastAPI application
            self.app = self._create_application()
            self._running = True

            return FlextResult[None].ok(None)
        except Exception as e:
            return FlextResult[None].fail(f"Failed to start web service: {e}")

    def stop(self) -> object:
        """Stop web application server."""
        if self.app:
            # Graceful shutdown logic
            self._running = False
            return FlextResult[None].ok(None)
        return FlextResult[None].fail("Web service not running")

    def health_check(self) -> object:
        """Web service health check."""
        if not self._running:
            return FlextResult[dict].fail("Web service not running")

        health_data = {
            "status": "healthy",
            "running": self._running,
            "routes_registered": len(self.app.url_map._rules) if self.app else 0,
            "config_loaded": bool(self.app_config)
        }

        return FlextResult[dict].ok(health_data)

    def _process_web_request(self, request: object) -> object:
        """Process incoming web request."""
        # Web request processing logic
        return FlextResult[dict].ok({"processed": True})

# Web handler following Application.Handler pattern
class WebRouteHandler(FlextProtocols.Application.ValidatingHandler):
    """Web route handler with validation."""

    def __init__(self, route_name: str):
        self.route_name = route_name

    def handle(self, message: object) -> object:
        """Handle web route request."""
        # Validate request
        validation_result = self.validate(message)
        if not validation_result.success:
            return validation_result

        # Process route logic
        return self._process_route(message)

    def validate(self, message: object) -> object:
        """Validate web request message."""
        if not hasattr(message, 'method') or not hasattr(message, 'path'):
            return FlextResult[None].fail("Invalid web request format")

        return FlextResult[None].ok(None)

    def can_handle(self, message_type: type) -> bool:
        """Check if handler can process message type."""
        return hasattr(message_type, 'method') and hasattr(message_type, 'path')

    def _process_route(self, request: object) -> object:
        """Process specific route logic."""
        return FlextResult[dict].ok({
            "route": self.route_name,
            "method": getattr(request, 'method', 'GET'),
            "processed": True
        })

# Web middleware following Extensions.Middleware
class WebAuthMiddleware(FlextProtocols.Extensions.Middleware):
    """Authentication middleware for web requests."""

    def __init__(self, auth_service: FlextProtocols.Infrastructure.Auth):
        self.auth_service = auth_service

    def process(self, request: object, next_handler: Callable[[object], object]) -> object:
        """Process request with authentication."""

        # Extract credentials from request
        credentials = self._extract_credentials(request)

        if credentials:
            # Authenticate request
            auth_result = self.auth_service.authenticate(credentials)

            if auth_result.success:
                # Add user context to request
                setattr(request, 'user', auth_result.value)
                return next_handler(request)
            else:
                return FlextResult[dict].fail("Authentication failed")
        else:
            # Skip authentication for public routes
            return next_handler(request)

    def _extract_credentials(self, request: object) -> dict[str, object] | None:
        """Extract authentication credentials from request."""
        # Implementation would extract from headers, cookies, etc.
        return getattr(request, 'auth_headers', None)

# Usage example
def setup_standardized_web_service():
    """Setup web service with standardized FlextProtocols."""

    # Create web service
    web_config = {
        "host": "0.0.0.0",
        "port": 8000,
        "debug": True
    }

    web_service = FlextWebService(web_config)

    # Create handlers
    user_handler = WebRouteHandler("user_management")
    api_handler = WebRouteHandler("api_endpoints")

    # Start service
    start_result = web_service.start()

    if start_result.success:
        print("✅ Web service started with FlextProtocols")

        # Check health
        health = web_service.health_check()
        print(f"🏥 Service health: {health.value}")

        return web_service
    else:
        print(f"❌ Failed to start web service: {start_result.error}")
        return None
```

**Integration Benefits**:

- **Service Standardization**: Consistent service lifecycle across FLEXT ecosystem
- **Handler Architecture**: CQRS-compliant request handling with validation
- **Middleware Patterns**: Standardized middleware pipeline with Extensions.Middleware
- **Health Monitoring**: Unified health checking across all services

---

### 2. flext-meltano (Critical Priority - Plugin Standardization)

**Current State**: Minimal plugin protocols in `plugin_protocols.py`

#### Current Implementation Analysis

```python
# Current minimal approach
class FlextMeltanoPluginTypes:
    TapPlugin = object  # Simple alias - NO protocol definitions
    TargetPlugin = object
    DbtPlugin = object
```

#### Integration Opportunities

##### A. Full Plugin System with Extensions.Plugin

```python
# Comprehensive plugin system with FlextProtocols
class FlextMeltanoTap(FlextProtocols.Extensions.Plugin):
    """Meltano tap following Extensions.Plugin protocol."""

    def __init__(self, tap_name: str):
        self.tap_name = tap_name
        self.config: dict[str, object] = {}
        self.context: FlextProtocols.Extensions.PluginContext | None = None
        self.initialized = False
        self.connection: FlextProtocols.Infrastructure.Connection | None = None

    def configure(self, config: dict[str, object]) -> object:
        """Configure tap with Meltano settings."""
        required_keys = ["source_connection", "catalog", "state"]
        missing_keys = [key for key in required_keys if key not in config]

        if missing_keys:
            return FlextResult[None].fail(f"Missing required config: {missing_keys}")

        self.config.update(config)
        return FlextResult[None].ok(None)

    def get_config(self) -> dict[str, object]:
        """Get current tap configuration."""
        return self.config.copy()

    def initialize(self, context: FlextProtocols.Extensions.PluginContext) -> object:
        """Initialize tap with Meltano context."""
        try:
            self.context = context
            logger = context.FlextLogger()

            # Get tap-specific configuration
            tap_config = context.get_config().get(self.tap_name, {})
            config_result = self.configure(tap_config)

            if not config_result.success:
                return config_result

            # Initialize source connection
            self.connection = self._create_source_connection()

            self.initialized = True
            logger.info("Meltano tap initialized",
                       tap_name=self.tap_name,
                       source_type=tap_config.get("source_type", "unknown"))

            return FlextResult[None].ok(None)

        except Exception as e:
            return FlextResult[None].fail(f"Tap initialization failed: {e}")

    def shutdown(self) -> object:
        """Shutdown tap and cleanup resources."""
        if self.connection:
            self.connection.close_connection()

        if self.context:
            logger = self.context.FlextLogger()
            logger.info("Meltano tap shutting down", tap_name=self.tap_name)

        self.initialized = False
        return FlextResult[None].ok(None)

    def get_info(self) -> dict[str, object]:
        """Get tap information."""
        return {
            "name": self.tap_name,
            "type": "tap",
            "version": "1.0.0",
            "initialized": self.initialized,
            "source_type": self.config.get("source_type", "unknown"),
            "config": self.config
        }

    def extract_data(self) -> object:
        """Extract data from source using tap."""
        if not self.initialized:
            return FlextResult[list].fail("Tap not initialized")

        if not self.connection:
            return FlextResult[list].fail("No source connection available")

        try:
            # Test connection
            test_result = self.connection.test_connection()
            if not test_result.success:
                return test_result

            # Extract data (implementation would use Singer spec)
            extracted_data = self._perform_extraction()

            if self.context:
                logger = self.context.FlextLogger()
                logger.info("Data extraction completed",
                           records_extracted=len(extracted_data),
                           tap_name=self.tap_name)

            return FlextResult[list].ok(extracted_data)

        except Exception as e:
            return FlextResult[list].fail(f"Data extraction failed: {e}")

    def _create_source_connection(self) -> FlextProtocols.Infrastructure.Connection:
        """Create source system connection."""
        # Implementation would create appropriate connection type
        source_type = self.config.get("source_type", "database")

        if source_type == "database":
            return DatabaseConnection(self.config["source_connection"])
        else:
            # Other connection types
            return GenericConnection(self.config["source_connection"])

    def _perform_extraction(self) -> list[dict]:
        """Perform actual data extraction."""
        # Mock extraction for example
        return [
            {"id": 1, "name": "Record 1", "extracted_at": time.time()},
            {"id": 2, "name": "Record 2", "extracted_at": time.time()}
        ]

class FlextMeltanoTarget(FlextProtocols.Extensions.Plugin):
    """Meltano target following Extensions.Plugin protocol."""

    def __init__(self, target_name: str):
        self.target_name = target_name
        self.config: dict[str, object] = {}
        self.context: FlextProtocols.Extensions.PluginContext | None = None
        self.initialized = False
        self.connection: FlextProtocols.Infrastructure.Connection | None = None

    def configure(self, config: dict[str, object]) -> object:
        """Configure target with Meltano settings."""
        required_keys = ["target_connection", "schema_mapping"]
        missing_keys = [key for key in required_keys if key not in config]

        if missing_keys:
            return FlextResult[None].fail(f"Missing required config: {missing_keys}")

        self.config.update(config)
        return FlextResult[None].ok(None)

    def get_config(self) -> dict[str, object]:
        """Get current target configuration."""
        return self.config.copy()

    def initialize(self, context: FlextProtocols.Extensions.PluginContext) -> object:
        """Initialize target with Meltano context."""
        try:
            self.context = context
            logger = context.FlextLogger()

            # Get target-specific configuration
            target_config = context.get_config().get(self.target_name, {})
            config_result = self.configure(target_config)

            if not config_result.success:
                return config_result

            # Initialize target connection
            self.connection = self._create_target_connection()

            self.initialized = True
            logger.info("Meltano target initialized",
                       target_name=self.target_name,
                       target_type=target_config.get("target_type", "unknown"))

            return FlextResult[None].ok(None)

        except Exception as e:
            return FlextResult[None].fail(f"Target initialization failed: {e}")

    def shutdown(self) -> object:
        """Shutdown target and cleanup resources."""
        if self.connection:
            self.connection.close_connection()

        if self.context:
            logger = self.context.FlextLogger()
            logger.info("Meltano target shutting down", target_name=self.target_name)

        self.initialized = False
        return FlextResult[None].ok(None)

    def get_info(self) -> dict[str, object]:
        """Get target information."""
        return {
            "name": self.target_name,
            "type": "target",
            "version": "1.0.0",
            "initialized": self.initialized,
            "target_type": self.config.get("target_type", "unknown"),
            "config": self.config
        }

    def load_data(self, data: list[dict]) -> object:
        """Load data into target system."""
        if not self.initialized:
            return FlextResult[dict].fail("Target not initialized")

        if not self.connection:
            return FlextResult[dict].fail("No target connection available")

        try:
            # Test connection
            test_result = self.connection.test_connection()
            if not test_result.success:
                return test_result

            # Load data (implementation would use Singer spec)
            load_result = self._perform_load(data)

            if self.context:
                logger = self.context.FlextLogger()
                logger.info("Data load completed",
                           records_loaded=len(data),
                           target_name=self.target_name)

            return FlextResult[dict].ok({
                "records_loaded": len(data),
                "target": self.target_name,
                "success": True
            })

        except Exception as e:
            return FlextResult[dict].fail(f"Data load failed: {e}")

    def _create_target_connection(self) -> FlextProtocols.Infrastructure.Connection:
        """Create target system connection."""
        target_type = self.config.get("target_type", "database")

        if target_type == "database":
            return DatabaseConnection(self.config["target_connection"])
        else:
            return GenericConnection(self.config["target_connection"])

    def _perform_load(self, data: list[dict]) -> dict:
        """Perform actual data loading."""
        # Mock loading for example
        return {"loaded": len(data), "success": True}

# Meltano pipeline orchestrator
class MeltanoPipelineService(FlextProtocols.Domain.Service):
    """Meltano pipeline service following Domain.Service protocol."""

    def __init__(self):
        self.plugins: dict[str, FlextProtocols.Extensions.Plugin] = {}
        self._running = False
        self.context = SimpleMeltanoPluginContext()

    def __call__(self, *args: object, **kwargs: object) -> object:
        """Execute pipeline operations."""
        operation = kwargs.get("operation")

        if operation == "run_pipeline":
            return self.run_pipeline(
                kwargs.get("tap_name"),
                kwargs.get("target_name")
            )
        else:
            return FlextResult[dict].fail(f"Unknown operation: {operation}")

    def start(self) -> object:
        """Start Meltano pipeline service."""
        self._running = True
        print("🚀 Meltano pipeline service started")
        return FlextResult[None].ok(None)

    def stop(self) -> object:
        """Stop Meltano pipeline service."""
        # Shutdown all plugins
        for plugin in self.plugins.values():
            plugin.shutdown()

        self._running = False
        print("🛑 Meltano pipeline service stopped")
        return FlextResult[None].ok(None)

    def health_check(self) -> object:
        """Check pipeline service health."""
        if not self._running:
            return FlextResult[dict].fail("Pipeline service not running")

        # Check plugin health
        plugin_health = {}
        for name, plugin in self.plugins.items():
            plugin_info = plugin.get_info()
            plugin_health[name] = plugin_info["initialized"]

        health_data = {
            "status": "healthy",
            "running": self._running,
            "plugins_registered": len(self.plugins),
            "plugin_health": plugin_health
        }

        return FlextResult[dict].ok(health_data)

    def register_plugin(self, plugin: FlextProtocols.Extensions.Plugin) -> FlextResult[None]:
        """Register a Meltano plugin."""
        plugin_info = plugin.get_info()
        plugin_name = plugin_info["name"]

        # Initialize plugin
        init_result = plugin.initialize(self.context)

        if init_result.success:
            self.plugins[plugin_name] = plugin
            print(f"✅ Plugin registered: {plugin_name}")
            return FlextResult[None].ok(None)
        else:
            return FlextResult[None].fail(f"Failed to register plugin {plugin_name}: {init_result.error}")

    def run_pipeline(self, tap_name: str, target_name: str) -> object:
        """Run ETL pipeline from tap to target."""
        if not self._running:
            return FlextResult[dict].fail("Pipeline service not running")

        # Get tap and target
        tap = self.plugins.get(tap_name)
        target = self.plugins.get(target_name)

        if not tap or not target:
            return FlextResult[dict].fail(f"Missing plugins: tap={bool(tap)}, target={bool(target)}")

        try:
            # Extract data from tap
            extract_result = tap.extract_data()

            if extract_result.success:
                data = extract_result.value

                # Load data to target
                load_result = target.load_data(data)

                if load_result.success:
                    result = {
                        "pipeline_run": "success",
                        "tap": tap_name,
                        "target": target_name,
                        "records_processed": len(data),
                        "timestamp": time.time()
                    }

                    return FlextResult[dict].ok(result)
                else:
                    return load_result
            else:
                return extract_result

        except Exception as e:
            return FlextResult[dict].fail(f"Pipeline execution failed: {e}")

class SimpleMeltanoPluginContext(FlextProtocols.Extensions.PluginContext):
    """Simple Meltano plugin context."""

    def __init__(self):
        self.services: dict[str, object] = {}
        self.config: dict[str, object] = {
            "tap-oracle": {
                "source_type": "database",
                "source_connection": "oracle://user:pass@localhost:1521/XE"
            },
            "target-postgres": {
                "target_type": "database",
                "target_connection": "postgresql://user:pass@localhost:5432/warehouse"
            }
        }
        self.logger = StructuredLogger("meltano-context")

    def get_service(self, service_name: str) -> object:
        """Get service by name."""
        return self.services.get(service_name)

    def get_config(self) -> dict[str, object]:
        """Get plugin configuration."""
        return self.config.copy()

    def FlextLogger(self) -> FlextProtocols.Infrastructure.LoggerProtocol:
        """Get logger for plugin."""
        return self.logger

# Usage example
def setup_meltano_with_protocols():
    """Setup Meltano with standardized FlextProtocols."""

    # Create pipeline service
    pipeline_service = MeltanoPipelineService()

    # Start service
    start_result = pipeline_service.start()

    if start_result.success:
        # Create and register plugins
        oracle_tap = FlextMeltanoTap("tap-oracle")
        postgres_target = FlextMeltanoTarget("target-postgres")

        # Register plugins
        pipeline_service.register_plugin(oracle_tap)
        pipeline_service.register_plugin(postgres_target)

        # Run pipeline
        pipeline_result = pipeline_service.run_pipeline("tap-oracle", "target-postgres")

        if pipeline_result.success:
            result = pipeline_result.value
            print(f"✅ Pipeline completed: {result['records_processed']} records")

        # Check health
        health = pipeline_service.health_check()
        print(f"🏥 Pipeline health: {health.value}")

        # Stop service
        pipeline_service.stop()

        return pipeline_service
    else:
        print(f"❌ Failed to start pipeline service: {start_result.error}")
        return None
```

**Integration Benefits**:

- **Plugin Standardization**: Consistent plugin lifecycle across all Meltano components
- **Connection Management**: Unified connection protocols for all data sources/targets
- **Service Architecture**: Domain service for pipeline orchestration
- **Context Management**: Structured plugin context with dependency injection

---

### 3. flext-ldap (Critical Priority - Connection Protocol Integration)

**Current State**: No protocol usage, custom connection handling

#### Integration Opportunities

##### A. LDAP Connection with Infrastructure.LdapConnection

```python
# LDAP service following Infrastructure.LdapConnection protocol
class FlextLDAPService(FlextProtocols.Infrastructure.LdapConnection):
    """LDAP service implementing Infrastructure.LdapConnection protocol."""

    def __init__(self):
        self.connection = None
        self.connected = False
        self.server_uri = ""
        self.bind_dn = ""
        self.logger = StructuredLogger("flext-ldap")

    def __call__(self, *args: object, **kwargs: object) -> object:
        """Callable interface for LDAP operations."""
        operation = kwargs.get("operation")

        if operation == "search_users":
            return self.search_users(kwargs.get("base_dn", ""), kwargs.get("filter", ""))
        elif operation == "create_user":
            return self.create_user(kwargs.get("user_data", {}))
        else:
            return FlextResult[object].fail(f"Unknown LDAP operation: {operation}")

    # Base Connection protocol methods
    def test_connection(self) -> object:
        """Test LDAP server connection."""
        try:
            # Test LDAP server connectivity
            if not self.server_uri:
                return FlextResult[None].fail("No server URI configured")

            # Simulate connection test
            self.logger.info("Testing LDAP connection", server_uri=self.server_uri)

            return FlextResult[None].ok(None)

        except Exception as e:
            self.logger.exception("LDAP connection test failed")
            return FlextResult[None].fail(f"LDAP connection test failed: {e}")

    def get_connection_string(self) -> str:
        """Get LDAP connection string (sanitized)."""
        return self.server_uri.replace(":password", ":***") if self.server_uri else "not_configured"

    def close_connection(self) -> object:
        """Close LDAP connection."""
        if not self.connected:
            return FlextResult[None].fail("No active LDAP connection")

        try:
            if self.connection:
                self.unbind()

            self.connected = False
            self.connection = None

            self.logger.info("LDAP connection closed")
            return FlextResult[None].ok(None)

        except Exception as e:
            return FlextResult[None].fail(f"Failed to close LDAP connection: {e}")

    # LdapConnection specific methods
    def connect(self, uri: str, bind_dn: str, password: str) -> object:
        """Connect to LDAP server with authentication."""
        try:
            if self.connected:
                return FlextResult[None].fail("Already connected to LDAP server")

            self.server_uri = uri
            self.bind_dn = bind_dn

            # Test connection first
            test_result = self.test_connection()
            if not test_result.success:
                return test_result

            # Perform bind
            bind_result = self.bind(bind_dn, password)
            if bind_result.success:
                self.connected = True
                self.logger.info("LDAP connection established",
                               server_uri=uri,
                               bind_dn=bind_dn)
                return FlextResult[None].ok(None)
            else:
                return bind_result

        except Exception as e:
            self.logger.exception("LDAP connection failed")
            return FlextResult[None].fail(f"LDAP connection failed: {e}")

    def bind(self, bind_dn: str, password: str) -> object:
        """Bind with specific credentials."""
        try:
            # Simulate LDAP bind operation
            self.connection = f"ldap_conn_{int(time.time())}"
            self.logger.debug("LDAP bind successful", bind_dn=bind_dn)

            return FlextResult[None].ok(None)

        except Exception as e:
            self.logger.exception("LDAP bind failed", bind_dn=bind_dn)
            return FlextResult[None].fail(f"LDAP bind failed: {e}")

    def unbind(self) -> object:
        """Unbind from LDAP server."""
        try:
            if self.connection:
                # Simulate unbind
                self.connection = None
                self.logger.debug("LDAP unbind successful")

            return FlextResult[None].ok(None)

        except Exception as e:
            return FlextResult[None].fail(f"LDAP unbind failed: {e}")

    def search(self, base_dn: str, search_filter: str, scope: str = "subtree") -> object:
        """Perform LDAP search operation."""
        if not self.connected:
            return FlextResult[list].fail("Not connected to LDAP server")

        try:
            self.logger.info("LDAP search initiated",
                           base_dn=base_dn,
                           filter=search_filter,
                           scope=scope)

            # Mock search results
            search_results = [
                {
                    "dn": f"uid=user1,{base_dn}",
                    "attributes": {
                        "uid": ["user1"],
                        "cn": ["User One"],
                        "mail": ["user1@example.com"]
                    }
                },
                {
                    "dn": f"uid=user2,{base_dn}",
                    "attributes": {
                        "uid": ["user2"],
                        "cn": ["User Two"],
                        "mail": ["user2@example.com"]
                    }
                }
            ]

            self.logger.info("LDAP search completed", results_count=len(search_results))
            return FlextResult[list].ok(search_results)

        except Exception as e:
            self.logger.exception("LDAP search failed", base_dn=base_dn)
            return FlextResult[list].fail(f"LDAP search failed: {e}")

    def add(self, dn: str, attributes: dict[str, object]) -> object:
        """Add new LDAP entry."""
        if not self.connected:
            return FlextResult[None].fail("Not connected to LDAP server")

        try:
            self.logger.info("LDAP add initiated", dn=dn)

            # Simulate add operation
            entry_created = {
                "dn": dn,
                "attributes": attributes,
                "created_at": time.time()
            }

            self.logger.info("LDAP entry added successfully", dn=dn)
            return FlextResult[dict].ok(entry_created)

        except Exception as e:
            self.logger.exception("LDAP add failed", dn=dn)
            return FlextResult[dict].fail(f"LDAP add failed: {e}")

    def modify(self, dn: str, modifications: dict[str, object]) -> object:
        """Modify existing LDAP entry."""
        if not self.connected:
            return FlextResult[None].fail("Not connected to LDAP server")

        try:
            self.logger.info("LDAP modify initiated", dn=dn)

            # Simulate modify operation
            modified_entry = {
                "dn": dn,
                "modifications": modifications,
                "modified_at": time.time()
            }

            self.logger.info("LDAP entry modified successfully", dn=dn)
            return FlextResult[dict].ok(modified_entry)

        except Exception as e:
            self.logger.exception("LDAP modify failed", dn=dn)
            return FlextResult[dict].fail(f"LDAP modify failed: {e}")

    def delete(self, dn: str) -> object:
        """Delete LDAP entry."""
        if not self.connected:
            return FlextResult[None].fail("Not connected to LDAP server")

        try:
            self.logger.info("LDAP delete initiated", dn=dn)

            # Simulate delete operation
            deleted_entry = {
                "dn": dn,
                "deleted_at": time.time()
            }

            self.logger.info("LDAP entry deleted successfully", dn=dn)
            return FlextResult[dict].ok(deleted_entry)

        except Exception as e:
            self.logger.exception("LDAP delete failed", dn=dn)
            return FlextResult[dict].fail(f"LDAP delete failed: {e}")

    def is_connected(self) -> bool:
        """Check if connection is active."""
        return self.connected and self.connection is not None

    # Business methods using protocol foundation
    def search_users(self, base_dn: str, filter_criteria: str = "") -> object:
        """Search for users in LDAP."""
        user_filter = f"(&(objectClass=person){filter_criteria})" if filter_criteria else "(objectClass=person)"
        return self.search(base_dn, user_filter)

    def create_user(self, user_data: dict[str, object]) -> object:
        """Create new user in LDAP."""
        if "uid" not in user_data or "cn" not in user_data:
            return FlextResult[dict].fail("Missing required user data: uid and cn")

        user_dn = f"uid={user_data['uid']},ou=users,{user_data.get('base_dn', 'dc=example,dc=com')}"

        attributes = {
            "objectClass": ["person", "organizationalPerson", "inetOrgPerson"],
            "uid": [user_data["uid"]],
            "cn": [user_data["cn"]],
            "sn": [user_data.get("sn", user_data["cn"])],
        }

        if "mail" in user_data:
            attributes["mail"] = [user_data["mail"]]

        return self.add(user_dn, attributes)

# LDAP repository following Domain.Repository
class LdapUserRepository(FlextProtocols.Domain.Repository[dict]):
    """LDAP user repository following Domain.Repository protocol."""

    def __init__(self, ldap_service: FlextProtocols.Infrastructure.LdapConnection):
        self.ldap_service = ldap_service
        self.base_dn = "ou=users,dc=example,dc=com"

    def get_by_id(self, entity_id: str) -> object:
        """Get user by UID from LDAP."""
        search_filter = f"(uid={entity_id})"

        search_result = self.ldap_service.search(self.base_dn, search_filter)

        if search_result.success:
            results = search_result.value
            if results:
                # Convert LDAP entry to user dict
                ldap_entry = results[0]
                user = self._ldap_entry_to_user(ldap_entry)
                return FlextResult[dict].ok(user)
            else:
                return FlextResult[dict].fail(f"User not found: {entity_id}")
        else:
            return search_result

    def save(self, entity: dict) -> object:
        """Save user to LDAP (create or update)."""
        if "uid" not in entity:
            return FlextResult[dict].fail("User must have 'uid' field")

        # Check if user exists
        existing_user = self.get_by_id(entity["uid"])

        if existing_user.success:
            # Update existing user
            return self._update_user(entity)
        else:
            # Create new user
            return self._create_user(entity)

    def delete(self, entity_id: str) -> object:
        """Delete user from LDAP."""
        user_dn = f"uid={entity_id},{self.base_dn}"
        return self.ldap_service.delete(user_dn)

    def find_all(self) -> object:
        """Find all users in LDAP."""
        return self.ldap_service.search(self.base_dn, "(objectClass=person)")

    def _ldap_entry_to_user(self, ldap_entry: dict) -> dict:
        """Convert LDAP entry to user dictionary."""
        attributes = ldap_entry.get("attributes", {})

        return {
            "uid": attributes.get("uid", [""])[0],
            "cn": attributes.get("cn", [""])[0],
            "mail": attributes.get("mail", [""])[0] if attributes.get("mail") else "",
            "dn": ldap_entry.get("dn", "")
        }

    def _create_user(self, user: dict) -> object:
        """Create new user in LDAP."""
        return self.ldap_service.create_user(user)

    def _update_user(self, user: dict) -> object:
        """Update existing user in LDAP."""
        user_dn = f"uid={user['uid']},{self.base_dn}"

        modifications = {}
        if "cn" in user:
            modifications["cn"] = [user["cn"]]
        if "mail" in user:
            modifications["mail"] = [user["mail"]]

        return self.ldap_service.modify(user_dn, modifications)

# Usage example
def setup_ldap_with_protocols():
    """Setup LDAP service with standardized FlextProtocols."""

    # Create LDAP service
    ldap_service = FlextLDAPService()

    # Connect to LDAP server
    connect_result = ldap_service.connect(
        "ldap://localhost:389",
        "cn=admin,dc=example,dc=com",
        "admin_password"
    )

    if connect_result.success:
        print("✅ LDAP connection established")

        # Create repository
        user_repo = LdapUserRepository(ldap_service)

        # Test user operations
        new_user = {
            "uid": "john.doe",
            "cn": "John Doe",
            "mail": "john.doe@example.com",
            "base_dn": "dc=example,dc=com"
        }

        # Save user
        save_result = user_repo.save(new_user)
        if save_result.success:
            print(f"✅ User created: {new_user['uid']}")

            # Retrieve user
            get_result = user_repo.get_by_id("john.doe")
            if get_result.success:
                user = get_result.value
                print(f"✅ User retrieved: {user['cn']}")

        # Test connection status
        print(f"🔗 LDAP connected: {ldap_service.is_connected()}")

        # Close connection
        ldap_service.close_connection()

        return ldap_service
    else:
        print(f"❌ LDAP connection failed: {connect_result.error}")
        return None
```

**Integration Benefits**:

- **Connection Standardization**: Unified LDAP connection protocol across ecosystem
- **Repository Pattern**: Consistent data access patterns for LDAP operations
- **Error Handling**: FlextResult integration for robust error management
- **Logging Integration**: Structured logging with contextual information

---

This comprehensive libraries analysis demonstrates the significant potential for FlextProtocols integration across the FLEXT ecosystem, providing unified contract definitions, type-safe interfaces, and Clean Architecture compliance for enhanced development productivity and architectural consistency.
