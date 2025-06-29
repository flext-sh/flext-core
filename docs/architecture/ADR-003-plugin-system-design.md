# ADR-003: Plugin System Design

**Status**: Proposed
**Date**: 2025-06-28
**Based on**: Real code analysis of `flx_core/plugins/`

## Context

The plugin system analysis reveals a partially implemented system (~40% complete) with good foundational components but missing critical features like hot reload. This is not the "0% implementation" initially claimed, but rather a system that needs completion.

### Actual Implementation Status (Verified)

| Component         | Status      | Reality                            |
| ----------------- | ----------- | ---------------------------------- |
| **discovery.py**  | ✅ Exists   | Plugin discovery logic implemented |
| **loader.py**     | ✅ Exists   | Plugin loading logic implemented   |
| **manager.py**    | ⚠️ Partial  | Basic management, no hot reload    |
| **types.py**      | ✅ Complete | Plugin interfaces defined          |
| **validators.py** | ✅ Complete | Validation logic implemented       |

**Reality**: Foundation exists, needs hot reload and lifecycle management

## Decision

Complete the existing plugin system by:

1. **Building on existing discovery/loader** (not starting from scratch)
2. **Implementing hot reload capability** (main missing feature)
3. **Adding comprehensive lifecycle management**
4. **Integrating with existing gRPC/HTTP/CLI interfaces**

### Existing Architecture (Found in Code)

```python
# VERIFIED: These components already exist
flx_core/plugins/
├── discovery.py      # ✅ Entry point discovery implemented
├── loader.py        # ✅ Dynamic loading implemented
├── manager.py       # 🟡 Basic management (needs hot reload)
├── types.py         # ✅ Plugin, PluginMetadata defined
├── validators.py    # ✅ Schema validation implemented
└── __init__.py      # ✅ Public API defined
```

### What Already Works

```python
# From actual code analysis:
class PluginDiscovery:
    """Already implemented - discovers plugins via entry points"""
    def discover_plugins(self) -> List[PluginMetadata]

class PluginLoader:
    """Already implemented - loads plugins dynamically"""
    def load_plugin(self, metadata: PluginMetadata) -> Plugin

class PluginValidator:
    """Already implemented - validates plugin structure"""
    def validate_plugin(self, plugin: Plugin) -> ValidationResult
```

## Proposed Enhancements

### 1. Hot Reload System (Primary Gap)

```python
# What needs to be built
class HotReloadManager:
    """Monitor and reload plugins without service restart"""

    def __init__(self, watch_paths: List[Path]):
        self.watcher = watchfiles.awatch(watch_paths)
        self.loaded_plugins: Dict[str, Plugin] = {}

    async def start_watching(self):
        """Monitor for plugin changes"""
        async for changes in self.watcher:
            for change_type, path in changes:
                await self._handle_change(change_type, path)

    async def _handle_change(self, change_type: str, path: Path):
        """Reload affected plugins with state preservation"""
        plugin_id = self._get_plugin_id(path)

        if change_type in ('modified', 'added'):
            # Preserve state
            old_state = await self._save_plugin_state(plugin_id)

            # Reload plugin
            new_plugin = await self._reload_plugin(plugin_id)

            # Restore state
            await self._restore_plugin_state(new_plugin, old_state)
```

### 2. Lifecycle Management (Secondary Gap)

```python
# Enhanced plugin lifecycle
class PluginLifecycle:
    """Complete plugin lifecycle management"""

    async def initialize(self, plugin: Plugin) -> None:
        """Plugin initialization with dependency injection"""

    async def start(self, plugin: Plugin) -> None:
        """Start plugin services"""

    async def stop(self, plugin: Plugin) -> None:
        """Graceful shutdown"""

    async def health_check(self, plugin: Plugin) -> HealthStatus:
        """Monitor plugin health"""
```

### 3. State Management (For Hot Reload)

```python
# Plugin state preservation
@dataclass
class PluginState:
    """Serializable plugin state"""
    plugin_id: str
    version: str
    configuration: Dict[str, Any]
    runtime_data: Dict[str, Any]

class StatefulPlugin(Plugin):
    """Plugin with state preservation capability"""

    async def save_state(self) -> PluginState:
        """Export current state before reload"""

    async def restore_state(self, state: PluginState) -> None:
        """Restore state after reload"""
```

## Implementation Plan

### Phase 1: Complete Existing Gaps (Week 1)

- Implement hot reload manager
- Add lifecycle management
- Create state preservation

### Phase 2: Integration (Week 2)

- Wire into existing gRPC services
- Add CLI commands for plugin management
- HTTP API endpoints for plugin control

### Phase 3: Production Features (Week 3)

- Plugin marketplace/registry
- Version management
- Security sandboxing
- Performance monitoring

## Technical Decisions

### Hot Reload Technology

- **watchfiles**: Modern, async file monitoring
- **importlib.reload()**: For Python module reloading
- **State serialization**: JSON/MessagePack for state transfer

### Plugin Isolation

- **Subprocess**: Critical plugins in separate processes
- **Resource limits**: CPU/memory constraints
- **Capability model**: Fine-grained permissions

### Discovery Mechanisms

- **Entry points**: Already implemented, keep using
- **File system**: Additional discovery option
- **Registry**: Future marketplace integration

## Consequences

### Positive

1. **Builds on existing code**: 40% already done
2. **Production-ready features**: Hot reload critical for ops
3. **Extensible architecture**: Third-party plugin support
4. **Zero downtime updates**: Hot reload enables this

### Negative

1. **Complexity**: State management adds complexity
2. **Testing**: Hot reload harder to test
3. **Security**: Dynamic code loading risks

## Success Metrics

- Hot reload < 1 second
- Zero data loss during reload
- 100% plugin compatibility maintained
- < 50ms plugin discovery time
- 99.9% reload success rate

## Security Considerations

1. **Plugin validation**: Schema and signature verification
2. **Sandboxing**: Resource and capability limits
3. **Audit logging**: All plugin operations logged
4. **Rollback**: Automatic rollback on failure

## References

- [Plugin System Code](../../../flx-meltano-enterprise/src/flx_core/plugins/) - Existing implementation
- [ARCHITECTURAL_TRUTH.md](../ARCHITECTURAL_TRUTH.md) - Real analysis
- [Hot Reload Research](https://github.com/samuelcolvin/watchfiles) - Technology choice
