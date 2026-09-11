"""Canonical per-group lazy resolution for every flext-infra CLI route."""

from __future__ import annotations

import functools
import importlib
from typing import TYPE_CHECKING, Final

from flext_infra import c

if TYPE_CHECKING:
    from flext_infra import m

# Why (ai-hub-xkux, fleet-wide fix): the previous CliRouteService composed
# CodegenRoutes + ValidationRoutes + WorkspaceRoutes via multi-inheritance and
# built every group's ClassVar route table (docs/refactor/release/check/...)
# at CLASS-DEFINITION time, so importing this module -- which cli_dispatch.py
# does unconditionally on every CLI invocation -- eagerly imported all three
# owning modules and their entire transitive dependency graph (~5.9s measured
# via python -X importtime), even though exactly one command group is ever
# dispatched per invocation. Only the owning module for the RESOLVED group is
# imported now, cutting startup to that one module's cost.
_GROUP_OWNERS: Final[dict[str, tuple[str, str, str]]] = {
    c.Infra.CLI_GROUP_CHECK: (
        "flext_infra.services.cli_routes_codegen",
        "CodegenRoutes",
        "codegen_routes",
    ),
    c.Infra.CLI_GROUP_CODEGEN: (
        "flext_infra.services.cli_routes_codegen",
        "CodegenRoutes",
        "codegen_routes",
    ),
    c.Infra.CLI_GROUP_DEPS: (
        "flext_infra.services.cli_routes_codegen",
        "CodegenRoutes",
        "codegen_routes",
    ),
    c.Infra.CLI_GROUP_DOCS: (
        "flext_infra.services.cli_routes_validate",
        "ValidationRoutes",
        "validation_routes",
    ),
    c.Infra.CLI_GROUP_GITHUB: (
        "flext_infra.services.cli_routes_validate",
        "ValidationRoutes",
        "validation_routes",
    ),
    c.Infra.CLI_GROUP_MAINTENANCE: (
        "flext_infra.services.cli_routes_validate",
        "ValidationRoutes",
        "validation_routes",
    ),
    c.Infra.CLI_GROUP_VALIDATE: (
        "flext_infra.services.cli_routes_validate",
        "ValidationRoutes",
        "validation_routes",
    ),
    c.Infra.CLI_GROUP_REFACTOR: (
        "flext_infra.services.cli_routes_workspace",
        "WorkspaceRoutes",
        "workspace_routes",
    ),
    c.Infra.CLI_GROUP_RELEASE: (
        "flext_infra.services.cli_routes_workspace",
        "WorkspaceRoutes",
        "workspace_routes",
    ),
    c.Infra.CLI_GROUP_WORKSPACE: (
        "flext_infra.services.cli_routes_workspace",
        "WorkspaceRoutes",
        "workspace_routes",
    ),
}


class CliRouteService:
    """Resolve one group's CLI route table on demand, never all of them."""

    @classmethod
    @functools.cache
    def route_table_for(cls, group: str) -> tuple[m.Cli.ResultCommandRoute, ...]:
        """Return the routes for one command group, importing only its owner."""
        owner = _GROUP_OWNERS.get(group)
        if owner is None:
            return ()
        module_path, class_name, attr_name = owner
        module = importlib.import_module(module_path)
        owner_class = getattr(module, class_name)
        table: dict[str, tuple[m.Cli.ResultCommandRoute, ...]] = getattr(
            owner_class, attr_name
        )
        return table[group]

    @classmethod
    def known_groups(cls) -> tuple[str, ...]:
        """Return every command group this service can resolve."""
        return tuple(_GROUP_OWNERS)


__all__: list[str] = ["CliRouteService"]
