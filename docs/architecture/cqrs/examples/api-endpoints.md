# API Endpoints CQRS Examples

**Version**: 1.0  
**Target**: Backend Developers  
**Framework**: FastAPI, Flask  
**Complexity**: Intermediate

## 📋 Overview

This document provides practical examples of implementing FlextCommands CQRS patterns in REST API endpoints. It covers command processing, query handling, error management, and integration with popular Python web frameworks.

## 🎯 Key Benefits

- ✅ **Clear Separation**: Commands for writes, queries for reads
- ✅ **Consistent Validation**: Structured business rule validation
- ✅ **Type Safety**: Full type checking from request to response
- ✅ **Better Testing**: Focused, testable components
- ✅ **Audit Trails**: Automatic command logging and tracking

---

## 🚀 FastAPI Integration Examples

### User Management API

#### Command Example: Create User

```python
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from flext_core import FlextCommands, FlextResult

# Request/Response Models
class CreateUserRequest(BaseModel):
    email: str
    name: str
    role: str = "user"
    department: str | None = None

class CreateUserResponse(BaseModel):
    user_id: str
    email: str
    name: str
    role: str
    created_at: str

# CQRS Command
class CreateUserCommand(FlextCommands.Models.Command):
    email: str
    name: str
    role: str = "user"
    department: str | None = None
    
    def validate_command(self) -> FlextResult[None]:
        return (
            self.require_email(self.email)
            .flat_map(lambda _: self.require_min_length(self.name, 2, "name"))
            .flat_map(lambda _: self._validate_role())
        )
    
    def _validate_role(self) -> FlextResult[None]:
        valid_roles = {"user", "admin", "manager", "viewer"}
        if self.role not in valid_roles:
            return FlextResult[None].fail(f"Invalid role. Valid options: {valid_roles}")
        return FlextResult[None].ok(None)

# Command Handler
class CreateUserHandler(FlextCommands.Handlers.CommandHandler[CreateUserCommand, dict]):
    def __init__(self, user_service: UserService, email_service: EmailService):
        super().__init__(handler_name="CreateUserHandler")
        self.user_service = user_service
        self.email_service = email_service
    
    def handle(self, command: CreateUserCommand) -> FlextResult[dict]:
        try:
            # Check if user already exists
            existing_user = self.user_service.find_by_email(command.email)
            if existing_user:
                return FlextResult[dict].fail(
                    "User with this email already exists",
                    error_code="USER_ALREADY_EXISTS"
                )
            
            # Create user
            user = self.user_service.create_user(
                email=command.email,
                name=command.name,
                role=command.role,
                department=command.department
            )
            
            # Send welcome email (async)
            self.email_service.send_welcome_email_async(user.email, user.name)
            
            self.log_info("User created successfully", 
                         user_id=user.id, email=command.email)
            
            return FlextResult[dict].ok({
                "user_id": user.id,
                "email": user.email,
                "name": user.name,
                "role": user.role,
                "created_at": user.created_at.isoformat()
            })
            
        except Exception as e:
            self.log_error("User creation failed", 
                          email=command.email, error=str(e))
            return FlextResult[dict].fail(f"User creation failed: {e}")

# FastAPI Endpoint
router = APIRouter(prefix="/api/v1/users", tags=["users"])

@router.post("/", response_model=CreateUserResponse)
async def create_user(request: CreateUserRequest):
    """Create a new user with validation and business rules."""
    
    # Convert request to command
    command = CreateUserCommand(
        email=request.email,
        name=request.name,
        role=request.role,
        department=request.department
    )
    
    # Execute via command bus
    result = command_bus.execute(command)
    
    # Handle result
    if result.success:
        return CreateUserResponse(**result.value)
    else:
        # Convert FlextResult error to HTTP exception
        error_code = getattr(result, 'error_code', 'VALIDATION_ERROR')
        status_code = {
            'VALIDATION_ERROR': 400,
            'USER_ALREADY_EXISTS': 409,
            'PERMISSION_DENIED': 403
        }.get(error_code, 500)
        
        raise HTTPException(
            status_code=status_code,
            detail={
                "error": result.error,
                "error_code": error_code,
                "command_id": command.command_id
            }
        )
```

#### Query Example: Get User

```python
# Query Models
class GetUserQuery(FlextCommands.Models.Query):
    user_id: str
    include_permissions: bool = False
    include_activity: bool = False
    
    def validate_query(self) -> FlextResult[None]:
        if not self.user_id or len(self.user_id) < 3:
            return FlextResult[None].fail("user_id must be at least 3 characters")
        return FlextResult[None].ok(None)

class GetUserHandler(FlextCommands.Handlers.QueryHandler[GetUserQuery, dict]):
    def __init__(self, user_service: UserService, permission_service: PermissionService):
        super().__init__(handler_name="GetUserHandler")
        self.user_service = user_service
        self.permission_service = permission_service
    
    def handle(self, query: GetUserQuery) -> FlextResult[dict]:
        try:
            # Get user
            user = self.user_service.find_by_id(query.user_id)
            if not user:
                return FlextResult[dict].fail(
                    "User not found",
                    error_code="USER_NOT_FOUND"
                )
            
            # Build response
            user_data = {
                "user_id": user.id,
                "email": user.email,
                "name": user.name,
                "role": user.role,
                "department": user.department,
                "created_at": user.created_at.isoformat(),
                "last_login": user.last_login.isoformat() if user.last_login else None
            }
            
            # Include permissions if requested
            if query.include_permissions:
                permissions = self.permission_service.get_user_permissions(user.id)
                user_data["permissions"] = [p.name for p in permissions]
            
            # Include activity if requested  
            if query.include_activity:
                activity = self.user_service.get_recent_activity(user.id, limit=10)
                user_data["recent_activity"] = [
                    {"action": a.action, "timestamp": a.timestamp.isoformat()}
                    for a in activity
                ]
            
            return FlextResult[dict].ok(user_data)
            
        except Exception as e:
            return FlextResult[dict].fail(f"Failed to retrieve user: {e}")

# FastAPI Endpoint
@router.get("/{user_id}", response_model=dict)
async def get_user(
    user_id: str,
    include_permissions: bool = False,
    include_activity: bool = False
):
    """Get user details with optional extended information."""
    
    # Create query
    query = GetUserQuery(
        user_id=user_id,
        include_permissions=include_permissions,
        include_activity=include_activity
    )
    
    # Execute query
    result = query_handler.handle(query)
    
    # Handle result
    if result.success:
        return result.value
    else:
        error_code = getattr(result, 'error_code', 'QUERY_ERROR')
        status_code = {
            'USER_NOT_FOUND': 404,
            'PERMISSION_DENIED': 403,
            'VALIDATION_ERROR': 400
        }.get(error_code, 500)
        
        raise HTTPException(status_code=status_code, detail=result.error)
```

#### Query Example: List Users with Pagination

```python
class ListUsersQuery(FlextCommands.Models.Query):
    role_filter: str | None = None
    department_filter: str | None = None
    active_only: bool = True
    search_term: str | None = None
    
    def validate_query(self) -> FlextResult[None]:
        # Call parent validation for pagination
        base_result = super().validate_query()
        if base_result.is_failure:
            return base_result
        
        # Custom validation
        if self.role_filter:
            valid_roles = {"user", "admin", "manager", "viewer"}
            if self.role_filter not in valid_roles:
                return FlextResult[None].fail(f"Invalid role filter: {self.role_filter}")
        
        return FlextResult[None].ok(None)

class ListUsersHandler(FlextCommands.Handlers.QueryHandler[ListUsersQuery, dict]):
    def __init__(self, user_service: UserService):
        super().__init__(handler_name="ListUsersHandler")
        self.user_service = user_service
    
    def handle(self, query: ListUsersQuery) -> FlextResult[dict]:
        try:
            # Build filters
            filters = {}
            if query.role_filter:
                filters['role'] = query.role_filter
            if query.department_filter:
                filters['department'] = query.department_filter
            if query.active_only:
                filters['active'] = True
            if query.search_term:
                filters['search'] = query.search_term
            
            # Get total count
            total_count = self.user_service.count_users(filters)
            
            # Get users with pagination
            offset = (query.page_number - 1) * query.page_size
            users = self.user_service.find_users(
                filters=filters,
                offset=offset,
                limit=query.page_size,
                sort_by=query.sort_by or 'created_at',
                sort_order=query.sort_order
            )
            
            # Build response
            return FlextResult[dict].ok({
                "users": [
                    {
                        "user_id": u.id,
                        "email": u.email,
                        "name": u.name,
                        "role": u.role,
                        "department": u.department,
                        "active": u.active,
                        "created_at": u.created_at.isoformat()
                    }
                    for u in users
                ],
                "pagination": {
                    "page_number": query.page_number,
                    "page_size": query.page_size,
                    "total_count": total_count,
                    "total_pages": (total_count + query.page_size - 1) // query.page_size,
                    "has_next": query.page_number * query.page_size < total_count,
                    "has_previous": query.page_number > 1
                },
                "filters": {
                    "role": query.role_filter,
                    "department": query.department_filter,
                    "active_only": query.active_only,
                    "search_term": query.search_term
                }
            })
            
        except Exception as e:
            return FlextResult[dict].fail(f"Failed to list users: {e}")

@router.get("/", response_model=dict)
async def list_users(
    role: str | None = None,
    department: str | None = None,
    active_only: bool = True,
    search: str | None = None,
    page: int = 1,
    page_size: int = 20,
    sort_by: str = "created_at",
    sort_order: str = "desc"
):
    """List users with filtering, pagination, and sorting."""
    
    query = ListUsersQuery(
        role_filter=role,
        department_filter=department,
        active_only=active_only,
        search_term=search,
        page_number=page,
        page_size=min(page_size, 100),  # Limit max page size
        sort_by=sort_by,
        sort_order=sort_order
    )
    
    result = list_users_handler.handle(query)
    
    if result.success:
        return result.value
    else:
        raise HTTPException(status_code=400, detail=result.error)
```

---

## 🔧 Flask Integration Examples

### Configuration Management API

```python
from flask import Blueprint, request, jsonify
from flask import current_app as app
from flext_core import FlextCommands, FlextResult

config_bp = Blueprint('config', __name__, url_prefix='/api/v1/config')

# Command: Update Configuration
class UpdateConfigCommand(FlextCommands.Models.Command):
    config_key: str
    config_value: object
    environment: str = "development"
    validate_only: bool = False
    
    def validate_command(self) -> FlextResult[None]:
        # Validate key format
        if not self.config_key or '.' not in self.config_key:
            return FlextResult[None].fail("config_key must be in format 'section.key'")
        
        # Validate environment
        valid_envs = {"development", "staging", "production"}
        if self.environment not in valid_envs:
            return FlextResult[None].fail(f"Invalid environment: {self.environment}")
        
        return FlextResult[None].ok(None)

class UpdateConfigHandler(FlextCommands.Handlers.CommandHandler[UpdateConfigCommand, dict]):
    def __init__(self, config_service: ConfigService, validator_service: ValidatorService):
        super().__init__(handler_name="UpdateConfigHandler")
        self.config_service = config_service
        self.validator_service = validator_service
    
    def handle(self, command: UpdateConfigCommand) -> FlextResult[dict]:
        try:
            # Validate configuration value
            validation_result = self.validator_service.validate_config(
                key=command.config_key,
                value=command.config_value,
                environment=command.environment
            )
            
            if not validation_result.is_valid:
                return FlextResult[dict].fail(
                    f"Configuration validation failed: {validation_result.error}",
                    error_code="CONFIG_VALIDATION_ERROR"
                )
            
            # If validate_only, don't actually update
            if command.validate_only:
                return FlextResult[dict].ok({
                    "config_key": command.config_key,
                    "valid": True,
                    "message": "Configuration is valid"
                })
            
            # Get current value for audit
            current_value = self.config_service.get_config(
                command.config_key, command.environment
            )
            
            # Update configuration
            updated_config = self.config_service.update_config(
                key=command.config_key,
                value=command.config_value,
                environment=command.environment,
                user_id=command.user_id  # From command metadata
            )
            
            # Log the change
            self.log_info("Configuration updated",
                         config_key=command.config_key,
                         environment=command.environment,
                         old_value=current_value,
                         new_value=command.config_value)
            
            return FlextResult[dict].ok({
                "config_key": command.config_key,
                "config_value": command.config_value,
                "environment": command.environment,
                "updated_at": updated_config.updated_at.isoformat(),
                "version": updated_config.version
            })
            
        except Exception as e:
            self.log_error("Configuration update failed",
                          config_key=command.config_key,
                          error=str(e))
            return FlextResult[dict].fail(f"Configuration update failed: {e}")

@config_bp.route('/', methods=['PUT'])
def update_config():
    """Update configuration with validation."""
    try:
        data = request.get_json()
        
        command = UpdateConfigCommand(
            config_key=data['config_key'],
            config_value=data['config_value'],
            environment=data.get('environment', 'development'),
            validate_only=data.get('validate_only', False),
            user_id=get_current_user_id()  # From auth context
        )
        
        result = command_bus.execute(command)
        
        if result.success:
            return jsonify({
                "success": True,
                "data": result.value
            }), 200
        else:
            error_code = getattr(result, 'error_code', 'CONFIG_ERROR')
            status_code = {
                'CONFIG_VALIDATION_ERROR': 400,
                'PERMISSION_DENIED': 403,
                'CONFIG_NOT_FOUND': 404
            }.get(error_code, 500)
            
            return jsonify({
                "success": False,
                "error": result.error,
                "error_code": error_code
            }), status_code
            
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Request processing failed: {e}"
        }), 500

# Query: Get Configuration
class GetConfigQuery(FlextCommands.Models.Query):
    config_key: str | None = None  # If None, get all configs
    environment: str = "development"
    include_metadata: bool = False
    
    def validate_query(self) -> FlextResult[None]:
        valid_envs = {"development", "staging", "production"}
        if self.environment not in valid_envs:
            return FlextResult[None].fail(f"Invalid environment: {self.environment}")
        return FlextResult[None].ok(None)

@config_bp.route('/', methods=['GET'])
@config_bp.route('/<config_key>', methods=['GET'])
def get_config(config_key=None):
    """Get configuration values."""
    try:
        query = GetConfigQuery(
            config_key=config_key,
            environment=request.args.get('environment', 'development'),
            include_metadata=request.args.get('include_metadata', False)
        )
        
        result = get_config_handler.handle(query)
        
        if result.success:
            return jsonify({
                "success": True,
                "data": result.value
            })
        else:
            return jsonify({
                "success": False,
                "error": result.error
            }), 404
            
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Request processing failed: {e}"
        }), 500
```

---

## 🔐 Advanced Patterns

### Authentication Middleware Integration

```python
from functools import wraps
from flext_core import FlextResult

def require_auth(permission: str = None):
    """Decorator to require authentication and optional permission."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Get auth token
            auth_header = request.headers.get('Authorization')
            if not auth_header or not auth_header.startswith('Bearer '):
                raise HTTPException(status_code=401, detail="Authentication required")
            
            token = auth_header[7:]  # Remove 'Bearer '
            
            # Validate token using CQRS
            validate_query = ValidateTokenQuery(token=token)
            auth_result = auth_query_handler.handle(validate_query)
            
            if auth_result.is_failure:
                raise HTTPException(status_code=401, detail="Invalid token")
            
            user_info = auth_result.value
            
            # Check permission if required
            if permission:
                check_query = CheckPermissionQuery(
                    user_id=user_info['user_id'],
                    permission=permission
                )
                perm_result = permission_query_handler.handle(check_query)
                
                if perm_result.is_failure:
                    raise HTTPException(status_code=403, detail="Permission denied")
            
            # Add user info to request context
            request.current_user = user_info
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# Usage in endpoints
@router.post("/admin/users", response_model=CreateUserResponse)
@require_auth(permission="users.create")
async def create_user_admin(request: CreateUserRequest):
    """Create user (admin only)."""
    # User info available via request.current_user
    command = CreateUserCommand(**request.dict())
    command.user_id = request.current_user['user_id']  # Add audit info
    
    result = command_bus.execute(command)
    # ... handle result
```

### Bulk Operations Example

```python
class BulkUpdateUsersCommand(FlextCommands.Models.Command):
    user_updates: list[dict[str, object]]
    rollback_on_error: bool = True
    
    def validate_command(self) -> FlextResult[None]:
        if not self.user_updates:
            return FlextResult[None].fail("user_updates cannot be empty")
        
        if len(self.user_updates) > 100:
            return FlextResult[None].fail("Maximum 100 users can be updated at once")
        
        # Validate each update
        for i, update in enumerate(self.user_updates):
            if 'user_id' not in update:
                return FlextResult[None].fail(f"user_id missing in update {i}")
        
        return FlextResult[None].ok(None)

class BulkUpdateUsersHandler(FlextCommands.Handlers.CommandHandler[BulkUpdateUsersCommand, dict]):
    def handle(self, command: BulkUpdateUsersCommand) -> FlextResult[dict]:
        results = []
        failed_updates = []
        
        # Process each update
        for i, update_data in enumerate(command.user_updates):
            try:
                # Create individual update command
                update_command = UpdateUserCommand(**update_data)
                update_result = update_user_handler.handle(update_command)
                
                if update_result.success:
                    results.append({
                        "index": i,
                        "user_id": update_data['user_id'],
                        "status": "success",
                        "data": update_result.value
                    })
                else:
                    failed_updates.append({
                        "index": i,
                        "user_id": update_data['user_id'],
                        "status": "failed",
                        "error": update_result.error
                    })
                    
                    if command.rollback_on_error:
                        # Implement rollback logic
                        self._rollback_updates(results)
                        return FlextResult[dict].fail(
                            f"Bulk update failed at index {i}, all changes rolled back"
                        )
                
            except Exception as e:
                failed_updates.append({
                    "index": i,
                    "user_id": update_data.get('user_id', 'unknown'),
                    "status": "error",
                    "error": str(e)
                })
        
        return FlextResult[dict].ok({
            "total_updates": len(command.user_updates),
            "successful_updates": len(results),
            "failed_updates": len(failed_updates),
            "results": results,
            "failures": failed_updates
        })

@router.put("/bulk", response_model=dict)
@require_auth(permission="users.bulk_update")
async def bulk_update_users(request: dict):
    """Bulk update multiple users."""
    command = BulkUpdateUsersCommand(**request)
    result = bulk_update_handler.handle(command)
    
    if result.success:
        return result.value
    else:
        raise HTTPException(status_code=400, detail=result.error)
```

---

## 📊 Error Handling Patterns

### Standardized Error Response

```python
from enum import Enum
from typing import Dict, Any

class ErrorCode(Enum):
    VALIDATION_ERROR = "VALIDATION_ERROR"
    NOT_FOUND = "NOT_FOUND"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    DUPLICATE_RESOURCE = "DUPLICATE_RESOURCE"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    INTERNAL_ERROR = "INTERNAL_ERROR"

class APIErrorResponse(BaseModel):
    success: bool = False
    error_code: str
    message: str
    details: Dict[str, Any] | None = None
    timestamp: str
    request_id: str

def handle_command_result(result: FlextResult, command_id: str = None) -> tuple[dict, int]:
    """Convert FlextResult to standardized API response."""
    
    if result.success:
        return {
            "success": True,
            "data": result.value,
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": command_id or str(uuid4())
        }, 200
    
    # Map error codes to HTTP status codes
    error_code = getattr(result, 'error_code', ErrorCode.INTERNAL_ERROR.value)
    status_code_map = {
        ErrorCode.VALIDATION_ERROR.value: 400,
        ErrorCode.NOT_FOUND.value: 404,
        ErrorCode.PERMISSION_DENIED.value: 403,
        ErrorCode.DUPLICATE_RESOURCE.value: 409,
        ErrorCode.RATE_LIMIT_EXCEEDED.value: 429,
        ErrorCode.INTERNAL_ERROR.value: 500
    }
    
    status_code = status_code_map.get(error_code, 500)
    
    error_response = {
        "success": False,
        "error_code": error_code,
        "message": result.error,
        "details": getattr(result, 'error_data', None),
        "timestamp": datetime.utcnow().isoformat(),
        "request_id": command_id or str(uuid4())
    }
    
    return error_response, status_code

# Usage in endpoints
@router.post("/users", response_model=CreateUserResponse)
async def create_user(request: CreateUserRequest):
    command = CreateUserCommand(**request.dict())
    result = command_bus.execute(command)
    
    response_data, status_code = handle_command_result(result, command.command_id)
    
    if status_code == 200:
        return CreateUserResponse(**response_data['data'])
    else:
        raise HTTPException(status_code=status_code, detail=response_data)
```

---

## ⚡ Performance Optimization

### Async Command Processing

```python
import asyncio
from flext_core import FlextResult

class AsyncCreateUserHandler(FlextCommands.Handlers.CommandHandler[CreateUserCommand, dict]):
    async def handle_async(self, command: CreateUserCommand) -> FlextResult[dict]:
        try:
            # Parallel async operations
            user_creation_task = self.user_service.create_user_async(command)
            email_validation_task = self.email_service.validate_email_async(command.email)
            permission_setup_task = self.permission_service.setup_default_permissions_async(command.role)
            
            # Wait for all operations
            user, email_valid, permissions = await asyncio.gather(
                user_creation_task,
                email_validation_task,
                permission_setup_task,
                return_exceptions=True
            )
            
            # Check for errors
            if isinstance(user, Exception):
                return FlextResult[dict].fail(f"User creation failed: {user}")
            if isinstance(email_valid, Exception) or not email_valid:
                return FlextResult[dict].fail("Email validation failed")
            
            return FlextResult[dict].ok({
                "user_id": user.id,
                "email": user.email,
                "permissions_count": len(permissions) if not isinstance(permissions, Exception) else 0
            })
            
        except Exception as e:
            return FlextResult[dict].fail(f"Async processing failed: {e}")

# FastAPI async endpoint
@router.post("/users/async", response_model=CreateUserResponse)
async def create_user_async(request: CreateUserRequest):
    """Create user with async processing for better performance."""
    command = CreateUserCommand(**request.dict())
    
    # Use async handler
    result = await async_create_user_handler.handle_async(command)
    
    response_data, status_code = handle_command_result(result, command.command_id)
    
    if status_code == 200:
        return CreateUserResponse(**response_data['data'])
    else:
        raise HTTPException(status_code=status_code, detail=response_data)
```

These examples demonstrate comprehensive CQRS integration patterns for REST APIs, providing a solid foundation for implementing FlextCommands in web applications with proper separation of concerns, validation, error handling, and performance optimization.
