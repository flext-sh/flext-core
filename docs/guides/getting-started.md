<!-- Generated from docs/guides/getting-started.md for flext-core. -->
<!-- Source of truth: workspace docs/guides/. -->

# flext-core - Getting Started with FLEXT

> Project profile: `flext-core`







## What is FLEXT

FLEXT is an enterprise-grade data integration platform built with Python 3.13+ and modern architectural patterns. It provides:

- **Unified API**: Single facade pattern across all libraries
- **Type Safety**: Full Pydantic v2 integration
- **Enterprise Patterns**: CQRS, Railway-oriented programming, Dependency Injection
- **Extensible**: Plugin architecture with flext-core patterns
- **Production Ready**: Comprehensive testing, monitoring, and error handling

## Prerequisites

- **Python 3.13+**: FLEXT requires Python 3.13 or higher
- **pip**: For package installation
- **virtualenv** (recommended): For isolated environments

## Installation

### Basic Installation

Install FLEXT core and commonly used libraries:

```bash
# Install core framework
pip install flext-core

# Install LDIF processing (most common use case)
pip install flext-ldif

# Install additional libraries as needed
pip install flext-api flext-auth flext-ldap
```

### Development Installation

For development and testing:

```bash
# Clone the repository
git clone https://github.com/flext-sh/flext.git
cd flext

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate

# Install in development mode
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

### Docker Installation

For containerized deployments:

```bash
# Build FLEXT image
docker build -t flext:latest -f docker/Dockerfile .

# Run FLEXT container
docker run -v $(pwd)/data:/app/data flext:latest
```

## Your First FLEXT Application

### 1. Basic Setup

```python
from flext_core import FlextBus
from flext_core import FlextSettings
from flext_core import FlextConstants
from flext_core import FlextContainer
from flext_core import FlextContext
from flext_core import FlextDecorators
from flext_core import FlextDispatcher
from flext_core import FlextExceptions
from flext_core import h
from flext_core import FlextLogger
from flext_core import x
from flext_core import FlextModels
from flext_core import FlextProcessors
from flext_core import p
from flext_core import FlextRegistry
from flext_core import FlextResult
from flext_core import FlextRuntime
from flext_core import FlextService
from flext_core import t
from flext_core import u

# Create dependency injection container
container = FlextContainer()

# Register services (example)
# container.register(IService, ServiceImplementation())

print("FLEXT application initialized!")
```

### 2. Using flext-ldif for LDIF Processing

```python
from flext_ldif import FlextLdif

# Initialize LDIF API
ldif = FlextLdif()

# Parse LDIF content
ldif_content = """dn: cn=test,dc=example,dc=com
cn: test
sn: user
objectClass: inetOrgPerson"""

result = ldif.parse(ldif_content)
if result.is_success:
    entries = result.unwrap()
    print(f"Successfully parsed {len(entries)} LDIF entries")
else:
    print(f"Failed to parse LDIF: {result.failure()}")
```

### 3. Railway-Oriented Error Handling

```python
from flext_core import FlextBus
from flext_core import FlextSettings
from flext_core import FlextConstants
from flext_core import FlextContainer
from flext_core import FlextContext
from flext_core import FlextDecorators
from flext_core import FlextDispatcher
from flext_core import FlextExceptions
from flext_core import h
from flext_core import FlextLogger
from flext_core import x
from flext_core import FlextModels
from flext_core import FlextProcessors
from flext_core import p
from flext_core import FlextRegistry
from flext_core import FlextResult
from flext_core import FlextRuntime
from flext_core import FlextService
from flext_core import t
from flext_core import u

def process_ldif_data(content: str) -> FlextResult[str, Exception]:
    # Parse LDIF
    parse_result = ldif.parse(content)
    if parse_result.is_failure:
        return FlextResult.failure(parse_result.failure())

    entries = parse_result.unwrap()

    # Process entries
    try:
        processed_data = process_entries(entries)
        return FlextResult.success(processed_data)
    except Exception as e:
        return FlextResult.failure(e)

def process_entries(entries: list) -> str:
    # Your processing logic here
    return f"Processed {len(entries)} entries"

# Usage
result = process_ldif_data(ldif_content)
if result.is_success:
    print(f"Success: {result.unwrap()}")
else:
    print(f"Error: {result.failure()}")
```

### 4. CQRS Pattern with Commands and Queries

```python
from flext_core import FlextBus
from flext_core import FlextSettings
from flext_core import FlextConstants
from flext_core import FlextContainer
from flext_core import FlextContext
from flext_core import FlextDecorators
from flext_core import FlextDispatcher
from flext_core import FlextExceptions
from flext_core import h
from flext_core import FlextLogger
from flext_core import x
from flext_core import FlextModels
from flext_core import FlextProcessors
from flext_core import p
from flext_core import FlextRegistry
from flext_core import FlextResult
from flext_core import FlextRuntime
from flext_core import FlextService
from flext_core import t
from flext_core import u
from dataclasses import dataclass

@dataclass
class CreateUserCommand:
    username: str
    email: str

@dataclass
class GetUserQuery:
    user_id: str

class UserService:
    def create_user(self, cmd: CreateUserCommand) -> FlextResult[str, Exception]:
        # Create user logic
        return FlextResult.success(f"User {cmd.username} created")

    def get_user(self, query: GetUserQuery) -> FlextResult[str, Exception]:
        # Get user logic
        return FlextResult.success(f"User {query.user_id} data")

# Setup dispatcher
dispatcher = FlextDispatcher()
user_service = UserService()

dispatcher.register_handler(CreateUserCommand, user_service.create_user)
dispatcher.register_handler(GetUserQuery, user_service.get_user)

# Use the dispatcher
create_result = dispatcher.dispatch(CreateUserCommand("john", "john@example.com"))
get_result = dispatcher.dispatch(GetUserQuery("user123"))
```

## Configuration

### Basic Configuration

FLEXT uses environment variables for configuration:

```bash
# Set configuration
export FLEXT_LOG_LEVEL=INFO
export FLEXT_LDIF_DEFAULT_ENCODING=utf-8
export FLEXT_LDIF_STRICT_VALIDATION=true
```

### Programmatic Configuration

```python
from flext_ldif import FlextLdifSettings

# Create custom configuration
config = FlextLdifSettings(
    default_encoding="utf-8",
    strict_validation=True,
    servers_enabled=True,
    batch_size=1000
)

# Use configuration
ldif = FlextLdif(config=config)
```

## Next Steps

### Explore the Ecosystem

1. **flext-core**: Master the core patterns and abstractions
2. **flext-ldif**: Learn LDIF processing and migration
3. **flext-api**: Build REST APIs with FLEXT
4. **flext-auth**: Implement authentication and authorization
5. **flext-ldap**: Integrate with LDAP servers

### Learn Key Patterns

- **Railway-Oriented Programming**: Functional error handling
- **CQRS**: Command Query Responsibility Segregation
- **Dependency Injection**: Managing component dependencies
- **Domain Events**: Event-driven architecture

### Build Real Applications

- **Data Migration**: Migrate LDIF data between LDAP servers
- **API Development**: Create REST APIs with automatic documentation
- **Data Processing**: Build data pipelines with FLEXT patterns
- **Enterprise Integration**: Connect with existing enterprise systems

## Getting Help

- 📖 **Documentation**: Browse the complete documentation
- 🐛 **Issues**: Report bugs and request features
- 💬 **Discussions**: Ask questions and share knowledge
- 📧 **Support**: Contact the development team

## What's Next

Now that you have FLEXT installed and running, explore these areas:

1. **Architecture Guide**: Understand FLEXT's design principles
2. **API Reference**: Complete API documentation
3. **Project Guides**: Deep dive into specific libraries
4. **Examples**: Real-world usage examples

Happy coding with FLEXT! 🚀
