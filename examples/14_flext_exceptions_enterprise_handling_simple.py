#!/usr/bin/env python3
"""Enterprise exception handling with FlextExceptions - Simplified Version.

Demonstrates correct usage of the FlextExceptions system with proper parameter handling
and enterprise patterns for robust error handling.

Key Components:
    - FlextError: Base exception with context management
    - FlextValidationError: Field validation failures 
    - FlextConnectionError: Connection-related errors
    - FlextOperationError: Operation failures
    - FlextAuthenticationError: Authentication errors
    - Domain-specific exception handling patterns

This example shows real-world enterprise exception handling scenarios
with correct parameter usage for all exception types.
"""

from typing import Mapping

from flext_core import (
    FlextAuthenticationError,
    FlextConnectionError,
    FlextOperationError, 
    FlextResult,
    FlextValidationError,
)


# =============================================================================
# DOMAIN MODELS
# =============================================================================

class User:
    """User domain model."""

    def __init__(self, user_id: str, name: str, email: str, age: int | None = None) -> None:
        self.user_id = user_id
        self.name = name
        self.email = email
        self.age = age

    def __repr__(self) -> str:
        return f"User(id={self.user_id}, name='{self.name}', email='{self.email}')"


class DatabaseConnection:
    """Database connection example."""

    def __init__(self, host: str, port: int, database: str) -> None:
        self.host = host
        self.port = port
        self.database = database
        self.connected = False

    def connect(self) -> FlextResult[None]:
        """Connect to database."""
        if self.host == "unreachable-host":
            raise FlextConnectionError(
                "Database connection failed",
                service=f"database://{self.host}:{self.port}/{self.database}",
                context={
                    "host": self.host,
                    "port": self.port,
                    "database": self.database,
                    "timeout_seconds": 30,
                },
            )

        self.connected = True
        return FlextResult.ok(None)

    def authenticate(self, username: str, password: str) -> FlextResult[None]:
        """Authenticate with database."""
        if not self.connected:
            raise FlextOperationError(
                "Cannot authenticate without connection",
                operation="database_authentication",
                stage="pre_authentication_check",
            )

        if username != "REDACTED_LDAP_BIND_PASSWORD" or password != "secret":
            raise FlextAuthenticationError(
                "Invalid database credentials",
                service=f"database://{self.database}",
                context={"username": username},
            )

        return FlextResult.ok(None)


# =============================================================================
# VALIDATION FUNCTIONS  
# =============================================================================

def validate_user_data(data: Mapping[str, object]) -> FlextResult[User]:
    """Validate user data with proper exception handling."""
    
    # Check required fields
    if "name" not in data:
        raise FlextValidationError(
            "Missing required field: name",
            field="name",
            context={"provided_fields": list(data.keys())},
        )
    
    if "email" not in data:
        raise FlextValidationError(
            "Missing required field: email", 
            field="email",
            context={"provided_fields": list(data.keys())},
        )

    # Validate data types
    name = data["name"]
    if not isinstance(name, str):
        raise FlextValidationError(
            f"Field 'name' must be string, got {type(name).__name__}",
            field="name",
            context={"expected_type": "str", "actual_type": type(name).__name__},
        )

    email = data["email"]
    if not isinstance(email, str):
        raise FlextValidationError(
            f"Field 'email' must be string, got {type(email).__name__}",
            field="email", 
            context={"expected_type": "str", "actual_type": type(email).__name__},
        )

    # Validate email format
    if "@" not in email:
        raise FlextValidationError(
            "Invalid email format",
            field="email",
            context={"email_value": email},
        )

    # Validate age if provided
    age = None
    if "age" in data:
        age_value = data["age"]
        if not isinstance(age_value, int):
            raise FlextValidationError(
                f"Field 'age' must be integer, got {type(age_value).__name__}",
                field="age",
                context={"expected_type": "int", "actual_type": type(age_value).__name__},
            )
        
        if age_value < 18 or age_value > 120:
            raise FlextValidationError(
                "Age must be between 18 and 120",
                field="age",
                context={"age_value": age_value, "min_age": 18, "max_age": 120},
            )
        age = age_value

    user = User(
        user_id=f"user_{len(str(name))}_{len(str(email))}",
        name=str(name),
        email=str(email),
        age=age,
    )
    
    return FlextResult.ok(user)


def create_user(user_data: Mapping[str, object]) -> FlextResult[User]:
    """Create user with validation and error handling."""
    try:
        return validate_user_data(user_data)
    except FlextValidationError:
        # Re-raise validation errors as-is
        raise
    except Exception as e:
        # Wrap unexpected errors in operation error
        raise FlextOperationError(
            f"User creation failed: {e}",
            operation="create_user",
            stage="data_processing",
            context={"original_error": str(e)},
        ) from e


# =============================================================================
# DEMONSTRATION FUNCTIONS
# =============================================================================

def demonstrate_connection_errors() -> None:
    """Demonstrate connection error handling."""
    print("=== Connection Error Handling ===")
    
    # Success case
    try:
        db = DatabaseConnection("localhost", 5432, "testdb")
        result = db.connect()
        print(f"✅ Connection successful: {result.success}")
    except FlextConnectionError as e:
        print(f"❌ Connection failed: {e}")
        print(f"   Service: {e.service}")
        print(f"   Context: {e.context}")

    # Failure case
    try:
        db = DatabaseConnection("unreachable-host", 5432, "testdb")
        result = db.connect()
        print(f"✅ Connection successful: {result.success}")
    except FlextConnectionError as e:
        print(f"❌ Connection failed: {e}")
        print(f"   Service: {e.service}")
        print(f"   Context: {e.context}")


def demonstrate_authentication_errors() -> None:
    """Demonstrate authentication error handling."""
    print("\n=== Authentication Error Handling ===")
    
    db = DatabaseConnection("localhost", 5432, "testdb")
    db.connect()

    # Success case
    try:
        result = db.authenticate("REDACTED_LDAP_BIND_PASSWORD", "secret")
        print(f"✅ Authentication successful: {result.success}")
    except FlextAuthenticationError as e:
        print(f"❌ Authentication failed: {e}")

    # Failure case
    try:
        result = db.authenticate("user", "wrong")
        print(f"✅ Authentication successful: {result.success}")
    except FlextAuthenticationError as e:
        print(f"❌ Authentication failed: {e}")
        print(f"   Service: {e.service}")
        print(f"   Context: {e.context}")


def demonstrate_validation_errors() -> None:
    """Demonstrate validation error handling."""
    print("\n=== Validation Error Handling ===")
    
    # Success case
    try:
        valid_data = {"name": "John Doe", "email": "john@example.com", "age": 30}
        result = create_user(valid_data)
        if result.success:
            print(f"✅ User created: {result.data}")
    except FlextValidationError as e:
        print(f"❌ Validation failed: {e}")

    # Field missing error
    try:
        invalid_data = {"email": "john@example.com"}
        result = create_user(invalid_data)
        if result.success:
            print(f"✅ User created: {result.data}")
    except FlextValidationError as e:
        print(f"❌ Validation failed: {e}")
        print(f"   Field: {e.field}")
        print(f"   Context: {e.context}")

    # Type error
    try:
        invalid_data = {"name": 123, "email": "john@example.com"}
        result = create_user(invalid_data)
        if result.success:
            print(f"✅ User created: {result.data}")
    except FlextValidationError as e:
        print(f"❌ Validation failed: {e}")
        print(f"   Field: {e.field}")
        print(f"   Context: {e.context}")

    # Age validation error
    try:
        invalid_data = {"name": "John Doe", "email": "john@example.com", "age": 150}
        result = create_user(invalid_data)
        if result.success:
            print(f"✅ User created: {result.data}")
    except FlextValidationError as e:
        print(f"❌ Validation failed: {e}")
        print(f"   Field: {e.field}")
        print(f"   Context: {e.context}")


def demonstrate_operation_errors() -> None:
    """Demonstrate operation error handling."""
    print("\n=== Operation Error Handling ===")
    
    db = DatabaseConnection("localhost", 5432, "testdb")
    
    # Try authentication without connection
    try:
        result = db.authenticate("REDACTED_LDAP_BIND_PASSWORD", "secret")
        print(f"✅ Authentication successful: {result.success}")
    except FlextOperationError as e:
        print(f"❌ Operation failed: {e}")
        print(f"   Operation: {e.operation}")
        print(f"   Stage: {e.stage}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main() -> None:
    """Main execution function."""
    print("🚀 FLEXT Enterprise Exception Handling Demo")
    print("=" * 50)
    
    demonstrate_connection_errors()
    demonstrate_authentication_errors()
    demonstrate_validation_errors()
    demonstrate_operation_errors()
    
    print("\n✅ Exception handling demonstration completed!")


if __name__ == "__main__":
    main()