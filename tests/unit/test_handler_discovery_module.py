"""Handler module discovery tests."""

from __future__ import annotations

import types

from flext_tests import tm

from tests import h, p, r


class TestsFlextHandlerDiscoveryModule:
    def test_scan_module_finds_decorated_functions(self) -> None:
        class CreateCommand:
            def __init__(self, name: str) -> None:
                self.name = name

        class DeleteCommand:
            def __init__(self, user_id: str) -> None:
                self.user_id = user_id

        module = types.ModuleType("decorated_module")

        @h.handler(command=CreateCommand, priority=100)
        def handle_user_create_globally(cmd: CreateCommand) -> p.Result[str]:
            return r[str].ok(f"global_create_{cmd.name}")

        @h.handler(command=DeleteCommand, priority=50)
        def handle_user_delete_globally(cmd: DeleteCommand) -> p.Result[str]:
            return r[str].ok(f"global_delete_{cmd.user_id}")

        setattr(module, "handle_user_create_globally", handle_user_create_globally)
        setattr(module, "handle_user_delete_globally", handle_user_delete_globally)
        setattr(module, "non_callable", 123)

        handlers = h.Discovery.scan_module(module)
        tm.that(len(handlers), eq=2)
        tm.that(handlers[0][0], eq="handle_user_create_globally")
        tm.that(handlers[1][0], eq="handle_user_delete_globally")

    def test_scan_module_ignores_private_functions(self) -> None:
        class CreateCommand:
            pass

        module = types.ModuleType("private_check_module")

        @h.handler(command=CreateCommand)
        def _private_handler(cmd: CreateCommand) -> p.Result[str]:
            _ = cmd
            return r[str].ok("private")

        @h.handler(command=CreateCommand)
        def public_handler(cmd: CreateCommand) -> p.Result[str]:
            _ = cmd
            return r[str].ok("public")

        setattr(module, "_private_handler", _private_handler)
        setattr(module, "public_handler", public_handler)
        handlers = h.Discovery.scan_module(module)
        names = [name for name, _, _ in handlers]
        tm.that("_private_handler" not in names, eq=True)
        tm.that(names, has="public_handler")
