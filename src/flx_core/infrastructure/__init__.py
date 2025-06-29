"""Infrastructure layer - Technical implementations and external system integrations.

This package implements the infrastructure layer following Domain-Driven Design principles,
providing concrete implementations for the abstractions defined in the domain layer. It
handles all technical concerns and external system integrations while keeping the domain
layer pure and focused on business logic.

Key responsibilities:
- Database persistence with SQLAlchemy ORM mappings
- External API integrations (Meltano, authentication providers)
- Message queue implementations (Redis, RabbitMQ)
- File system operations and storage abstractions
- Network communication (HTTP, gRPC clients)
- Caching mechanisms (Redis, in-memory)
- Monitoring and metrics collection
- Configuration loading from external sources

The infrastructure layer implements:
- Repository pattern implementations for domain entities
- Unit of Work pattern for transaction management
- External service adapters following the ports and adapters pattern
- Infrastructure-specific error handling and retry logic
- Connection pooling and resource management
- Data serialization and deserialization

All infrastructure components are designed to be replaceable through dependency
injection, allowing the domain and application layers to remain decoupled from
specific technical implementations.
"""
