"""Persistence layer infrastructure for data storage and retrieval.

This package provides concrete implementations of the repository pattern using
SQLAlchemy as the ORM (Object-Relational Mapping) framework. It handles all
database interactions while maintaining a clean separation between the domain
model and persistence concerns.

Key components:
- SQLAlchemy model definitions mapping to database tables
- Repository implementations for each aggregate root
- Unit of Work pattern for transaction management
- Database session management and connection pooling
- Query builders and specifications for complex queries
- Data mapping between domain entities and database models

The persistence layer ensures:
- ACID compliance through proper transaction boundaries
- Optimistic locking for concurrent updates
- Lazy loading strategies to prevent N+1 queries
- Database vendor independence through SQLAlchemy abstraction
- Migration support for schema evolution
- Query optimization through proper indexing

All persistence operations are abstracted behind repository interfaces,
allowing the domain and application layers to remain database-agnostic.
"""
