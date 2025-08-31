# FlextCommands Implementation Guide

**Version**: 0.9.0 
**Target**: FLEXT Library Developers  
**Complexity**: Intermediate  
**Estimated Time**: 2-4 hours per library

## 📋 Overview

This guide provides step-by-step instructions for implementing FlextCommands CQRS patterns in FLEXT ecosystem libraries. It covers architecture design, code implementation, testing strategies, and integration patterns.

## 🎯 Implementation Phases

### Phase 1: Analysis & Planning (30 minutes)
### Phase 2: Command/Query Design (1 hour)  
### Phase 3: Handler Implementation (1-2 hours)
### Phase 4: Integration & Testing (1 hour)

---

## 🔍 Phase 1: Analysis & Planning

### 1.1 Identify CQRS Candidates

**Commands (Write Operations)**:
- User actions requiring validation
- Data modifications with business rules
- System operations with side effects
- Audit-required operations

**Queries (Read Operations)**:
- Data retrieval with filtering
- Report generation
- Status checks
- Search operations

### 1.2 Current Code Analysis Template

```python
# Analyze your current implementation
class CurrentImplementation:
    """Analyze what you have now"""
    
    # ❌ Identify mixed operations
    def process_request(self, data: dict):
        # Validation + business logic + persistence mixed
        pass
    
    # ❌ Identify error handling inconsistencies  
    def handle_error(self, error):
        # Inconsistent error responses
        pass
    
    # ❌ Identify type safety gaps
    def operation(self, params):
        # Untyped parameters and returns
        pass
```

### 1.3 Planning Checklist

- [ ] **Commands identified**: List all write operations
- [ ] **Queries identified**: List all read operations  
- [ ] **Validation rules**: Document business rules per operation
- [ ] **Dependencies**: Identify external services/repositories needed
- [ ] **Error scenarios**: Map potential failure cases
- [ ] **Testing strategy**: Plan test coverage approach

---

## 🏗️ Phase 2: Command/Query Design

### 2.1 Command Design Pattern

```python
from flext_core import FlextCommands, FlextResult

class YourActionCommand(FlextCommands.Models.Command):
    """Template for command design"""
    
    # Required fields
    primary_field: str
    
    # Optional fields with defaults
    optional_field: str = "default_value"
    flag_field: bool = False
    
    # Complex types
    config_data: dict[str, object] = Field(default_factory=dict)
    
    def validate_command(self) -> FlextResult[None]:
        """Business rule validation"""
        return (
            self.require_field("primary_field", self.primary_field)
            .flat_map(lambda _: self._validate_business_rules())
        )
    
    def _validate_business_rules(self) -> FlextResult[None]:
        """Custom business validation"""
        if len(self.primary_field) < 3:
            return FlextResult[None].fail("Primary field too short")
        return FlextResult[None].ok(None)
```

### 2.2 Query Design Pattern

```python
class YourDataQuery(FlextCommands.Models.Query):
    """Template for query design"""
    
    # Filter fields
    status_filter: str | None = None
    date_from: datetime | None = None
    date_to: datetime | None = None
    
    # Search fields
    search_term: str | None = None
    
    def validate_query(self) -> FlextResult[None]:
        """Query parameter validation"""
        base_validation = super().validate_query()
        if base_validation.is_failure:
            return base_validation
            
        # Custom query validation
        if self.date_from and self.date_to and self.date_from > self.date_to:
            return FlextResult[None].fail("date_from must be before date_to")
            
        return FlextResult[None].ok(None)
```

### 2.3 Design Validation Checklist

- [ ] **Immutable design**: Commands/queries are frozen
- [ ] **Type safety**: All fields properly typed
- [ ] **Validation logic**: Business rules implemented  
- [ ] **Default values**: Sensible defaults provided
- [ ] **Documentation**: Fields and validation documented

---

## ⚙️ Phase 3: Handler Implementation

### 3.1 Command Handler Pattern

```python
class YourActionHandler(
    FlextCommands.Handlers.CommandHandler[YourActionCommand, YourResultType]
):
    """Template for command handler implementation"""
    
    def __init__(self, dependencies: YourDependencies):
        """Initialize with required dependencies"""
        super().__init__(handler_name="YourActionHandler")
        self.service = dependencies.service
        self.repository = dependencies.repository
        self.logger = FlextLogger(__name__)
    
    def handle(self, command: YourActionCommand) -> FlextResult[YourResultType]:
        """Process command with full error handling"""
        try:
            self.log_info("Processing command", command_id=command.command_id)
            
            # Business logic execution
            result = self.service.execute_business_logic(command)
            
            # Persistence if needed
            if result.success:
                persistence_result = self.repository.save(result.value)
                if persistence_result.is_failure:
                    return FlextResult[YourResultType].fail(
                        f"Save failed: {persistence_result.error}"
                    )
            
            self.log_info("Command processed successfully", 
                         command_id=command.command_id)
            return result
            
        except Exception as e:
            self.log_error("Command processing failed", 
                          command_id=command.command_id, error=str(e))
            return FlextResult[YourResultType].fail(f"Processing failed: {e}")
```

### 3.2 Query Handler Pattern

```python
class YourDataHandler(
    FlextCommands.Handlers.QueryHandler[YourDataQuery, list[YourDataType]]
):
    """Template for query handler implementation"""
    
    def __init__(self, repository: YourRepository):
        """Initialize with data access dependencies"""
        super().__init__(handler_name="YourDataHandler")
        self.repository = repository
    
    def handle(self, query: YourDataQuery) -> FlextResult[list[YourDataType]]:
        """Execute query with pagination and filtering"""
        try:
            # Build query parameters
            params = self._build_query_params(query)
            
            # Execute data retrieval
            data = self.repository.find_with_params(params)
            
            # Apply additional filtering if needed
            filtered_data = self._apply_business_filters(data, query)
            
            # Apply pagination
            paginated_data = self._apply_pagination(filtered_data, query)
            
            return FlextResult[list[YourDataType]].ok(paginated_data)
            
        except Exception as e:
            return FlextResult[list[YourDataType]].fail(f"Query failed: {e}")
    
    def _build_query_params(self, query: YourDataQuery) -> dict[str, object]:
        """Convert query to repository parameters"""
        params = {}
        if query.status_filter:
            params["status"] = query.status_filter
        if query.search_term:
            params["search"] = query.search_term
        return params
    
    def _apply_pagination(self, data: list[YourDataType], 
                         query: YourDataQuery) -> list[YourDataType]:
        """Apply pagination to results"""
        start = (query.page_number - 1) * query.page_size
        end = start + query.page_size
        return data[start:end]
```

### 3.3 Handler Registration Pattern

```python
class YourLibraryCommandBus:
    """Centralized command bus for your library"""
    
    def __init__(self, dependencies: YourDependencies):
        self.bus = FlextCommands.Factories.create_command_bus()
        self._register_handlers(dependencies)
    
    def _register_handlers(self, deps: YourDependencies):
        """Register all command and query handlers"""
        
        # Command handlers
        self.bus.register_handler(YourActionHandler(deps))
        self.bus.register_handler(AnotherActionHandler(deps))
        
        # Query handlers  
        self.bus.register_handler(YourDataHandler(deps.repository))
        self.bus.register_handler(AnotherQueryHandler(deps.repository))
    
    def execute_command(self, command: object) -> FlextResult[object]:
        """Execute command with validation and error handling"""
        return self.bus.execute(command)
```

---

## 🔗 Phase 4: Integration & Testing

### 4.1 Library Integration Pattern

```python
# your_library/__init__.py
from flext_core import FlextCommands

# Export command/query models for consumers
from .commands import YourActionCommand, YourDataQuery
from .handlers import YourActionHandler, YourDataHandler

# Export configured bus for easy usage
from .bus import YourLibraryCommandBus

__all__ = [
    "YourActionCommand",
    "YourDataQuery", 
    "YourActionHandler",
    "YourDataHandler",
    "YourLibraryCommandBus",
]
```

### 4.2 API Integration Example

```python
# For REST APIs (Flask/FastAPI)
from your_library import YourActionCommand, YourLibraryCommandBus

@app.route('/api/v1/action', methods=['POST'])
def action_endpoint():
    # Parse request
    request_data = request.get_json()
    
    # Create command
    try:
        command = YourActionCommand(**request_data)
    except ValidationError as e:
        return jsonify({"error": str(e)}), 400
    
    # Execute via command bus
    result = command_bus.execute_command(command)
    
    # Return structured response
    if result.success:
        return jsonify({"success": True, "data": result.value})
    else:
        return jsonify({"success": False, "error": result.error}), 400
```

### 4.3 CLI Integration Example

```python
# For CLI interfaces (Click)
import click
from your_library import YourActionCommand

@click.command()
@click.option('--primary-field', required=True)
@click.option('--optional-field', default="default")
def action_command(primary_field: str, optional_field: str):
    """Execute action via CQRS command"""
    
    # Create command
    command = YourActionCommand(
        primary_field=primary_field,
        optional_field=optional_field
    )
    
    # Execute
    result = command_bus.execute_command(command) 
    
    # Handle result
    if result.success:
        click.echo(f"✅ Success: {result.value}")
    else:
        click.echo(f"❌ Error: {result.error}")
        sys.exit(1)
```

### 4.4 Testing Strategy

```python
# test_your_commands.py
import pytest
from flext_core import FlextResult
from your_library import YourActionCommand, YourActionHandler

class TestYourActionCommand:
    """Test command validation and behavior"""
    
    def test_valid_command_creation(self):
        """Test creating valid command"""
        command = YourActionCommand(primary_field="valid_value")
        assert command.primary_field == "valid_value"
        assert command.validate_command().success
    
    def test_invalid_command_validation(self):
        """Test command validation failures"""
        command = YourActionCommand(primary_field="")  # Invalid
        validation_result = command.validate_command()
        assert validation_result.is_failure
        assert "Primary field" in validation_result.error

class TestYourActionHandler:
    """Test handler processing logic"""
    
    @pytest.fixture
    def handler(self, mock_dependencies):
        return YourActionHandler(mock_dependencies)
    
    @pytest.fixture  
    def valid_command(self):
        return YourActionCommand(primary_field="test_value")
    
    def test_successful_handling(self, handler, valid_command):
        """Test successful command processing"""
        result = handler.handle(valid_command)
        assert result.success
        assert result.value is not None
    
    def test_failure_handling(self, handler, valid_command, mock_dependencies):
        """Test error handling in command processing"""
        mock_dependencies.service.execute_business_logic.side_effect = Exception("Test error")
        
        result = handler.handle(valid_command)
        assert result.is_failure
        assert "Test error" in result.error

class TestIntegration:
    """Test full integration with command bus"""
    
    def test_end_to_end_command_execution(self, command_bus):
        """Test complete command execution flow"""
        command = YourActionCommand(primary_field="integration_test")
        result = command_bus.execute_command(command)
        
        assert result.success
        # Add specific assertions based on your business logic
```

---

## ✅ Implementation Checklist

### Pre-Implementation
- [ ] **Analysis complete**: Commands/queries identified
- [ ] **Architecture designed**: Clear separation of concerns
- [ ] **Dependencies mapped**: External services identified

### Implementation
- [ ] **Commands implemented**: All write operations
- [ ] **Queries implemented**: All read operations
- [ ] **Handlers implemented**: Processing logic complete
- [ ] **Validation added**: Business rules enforced
- [ ] **Bus configured**: Handler registration complete

### Integration  
- [ ] **API integration**: REST endpoints updated
- [ ] **CLI integration**: Command-line interface updated
- [ ] **Error handling**: Consistent error responses
- [ ] **Logging integrated**: Structured logging added

### Testing
- [ ] **Unit tests**: Commands, handlers, validation tested
- [ ] **Integration tests**: End-to-end scenarios tested
- [ ] **Error scenarios**: Failure cases tested
- [ ] **Performance tests**: Load/stress testing complete

### Documentation
- [ ] **Usage examples**: Code examples provided
- [ ] **API documentation**: Commands/queries documented
- [ ] **Migration guide**: Upgrade path documented

---

## 🚨 Common Pitfalls & Solutions

### 1. **Overly Complex Commands**
```python
# ❌ Don't create god commands
class MegaCommand(FlextCommands.Models.Command):
    # 20+ fields doing everything
    pass

# ✅ Create focused commands
class CreateUserCommand(FlextCommands.Models.Command):
    email: str
    name: str
    
class UpdateUserProfileCommand(FlextCommands.Models.Command):
    user_id: str
    profile_data: dict[str, object]
```

### 2. **Business Logic in Commands**
```python
# ❌ Don't put business logic in commands
class BadCommand(FlextCommands.Models.Command):
    def execute_business_logic(self):  # Wrong!
        pass

# ✅ Keep commands as data containers
class GoodCommand(FlextCommands.Models.Command):
    field: str
    
    def validate_command(self) -> FlextResult[None]:  # Only validation!
        return self.require_field("field", self.field)
```

### 3. **Handler Dependencies**
```python
# ❌ Don't create dependencies in handlers
class BadHandler(FlextCommands.Handlers.CommandHandler):
    def handle(self, command):
        service = SomeService()  # Wrong!

# ✅ Inject dependencies through constructor  
class GoodHandler(FlextCommands.Handlers.CommandHandler):
    def __init__(self, service: SomeService):
        self.service = service
```

### 4. **Error Handling**
```python
# ❌ Don't swallow errors
def bad_handle(self, command):
    try:
        return self.service.process(command)
    except:
        return "error"  # Lost error context!

# ✅ Use FlextResult for proper error handling
def good_handle(self, command):
    try:
        result = self.service.process(command)
        return FlextResult[ResultType].ok(result)
    except Exception as e:
        return FlextResult[ResultType].fail(f"Processing failed: {e}")
```

---

## 📈 Success Metrics

Track these metrics to measure implementation success:

### Code Quality
- **Type Coverage**: 100% type annotations
- **Test Coverage**: >90% line coverage  
- **Documentation**: All public APIs documented

### Architecture Quality
- **Separation**: Clear command/query separation
- **Validation**: Comprehensive business rule coverage
- **Error Handling**: Consistent FlextResult usage

### Performance  
- **Response Time**: <100ms for simple operations
- **Throughput**: Handle expected concurrent load
- **Memory Usage**: No memory leaks in long-running processes

### Developer Experience
- **API Consistency**: Uniform patterns across operations
- **Error Messages**: Clear, actionable error messages
- **Documentation**: Examples and guides available

---

## 🔗 Next Steps

1. **Start Small**: Implement 1-2 commands/queries first
2. **Test Thoroughly**: Validate approach with real scenarios
3. **Iterate**: Refine patterns based on experience
4. **Document**: Share learnings with team
5. **Scale**: Apply patterns to remaining operations

This implementation guide provides the foundation for successful FlextCommands adoption. Adapt the patterns to your specific library needs while maintaining consistency with FLEXT architectural principles.
