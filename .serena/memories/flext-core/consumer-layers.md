# flext-core: Consumer & Layer Dependencies

## Consumer Projects (Direct)

| Category        | Projects                                            |
| --------------- | --------------------------------------------------- |
| **Platform**    | cli, meltano, api, web, auth, grpc, plugin, quality |
| **Infra**       | infra, tests, observability                         |
| **Integration** | tap-*, target-*, dbt-* (inherit via platform)       |

## Layer Mapping

- **L3 (Orchestration)**: `cli`, `meltano`, `dbt-*`
- **L2 (Domain)**: `ldif`, `ldap`, `oracle-*`, `db-oracle`
- **L1 (Bridge)**: `infra` (DI bridge), `observability` (logger bridge)
- **L0 (Contracts)**: `core` (pydantic v2, protocols, base models)

## No Direct Imports Rule

Abstracted libs (pydantic, dependency_injector, structlog, returns):
- ✓ Access via `c.*`, `m.*`, `p.*`, `t.*`, `u.*`, `r[T]`
- ✗ NEVER `from pydantic import`, `from dependency_injector import`, etc.

**Last Updated**: 2026-04-14