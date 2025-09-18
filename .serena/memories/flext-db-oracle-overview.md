# FLEXT-DB-ORACLE Project Overview

## Purpose

Oracle Database Integration for the FLEXT Ecosystem providing Oracle connectivity using SQLAlchemy 2.0 and Python-oracledb with FLEXT patterns.

## Key Responsibilities

1. **Oracle Connectivity** - Database connections and resource management
2. **Schema Operations** - Metadata extraction and schema introspection
3. **Query Execution** - SQL operations with FlextResult error handling
4. **Foundation Library** - Base for Oracle-related FLEXT projects

## Current Status (v0.9.9)

- **Working Features**: SQLAlchemy 2.0 Oracle integration, FlextResult error handling, Connection pooling, Schema introspection, CLI interface structure
- **Critical Gaps**: CLI formatters use SimpleNamespace placeholders, No async support, No DataFrame integration, No Oracle 23ai features
- **Source Code**: 12 Python modules, 4,517 lines of code, 511 functions, 0 async functions

## Tech Stack

- **Python**: 3.13+ with strict typing
- **Database**: Oracle XE 21c with SQLAlchemy 2.0 + oracledb
- **FLEXT Dependencies**: flext-core (foundation), flext-cli (CLI patterns), flext-observability
- **Build**: Poetry package management
- **Quality**: Ruff linting, MyPy strict mode, Bandit security, pytest testing

## Architecture

- **Clean Architecture** with Domain-Driven Design
- **FLEXT Integration**: Uses FlextResult, FlextContainer, FlextLogger, FlextDomainService
- **Zero Tolerance**: Single class per module, no aliases/wrappers, explicit error handling

## Source Structure

```
src/flext_db_oracle/
├── __init__.py          # Public API exports
├── api.py              # FlextDbOracleApi main orchestrator (18KB)
├── client.py           # FlextDbOracleClient CLI integration (16KB)
├── cli.py              # CLI interface (18KB)
├── constants.py        # Oracle-specific constants (5KB)
├── exceptions.py       # Oracle error hierarchy (9KB)
├── mixins.py          # Validation patterns (4KB)
├── models.py          # Domain models (31KB)
├── plugins.py         # Extension system (6KB)
├── services.py        # SQL query building (61KB)
├── utilities.py       # Helper functions (12KB)
└── py.typed           # Type declarations
```

## Integration Points

- **flext-tap-oracle** → Oracle data extraction
- **flext-target-oracle** → Oracle data loading
- **FLEXT ecosystem** → Oracle operations for 32+ projects
