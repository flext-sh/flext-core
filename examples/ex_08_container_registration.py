"""Container registration example section."""

from __future__ import annotations

from typing import TYPE_CHECKING

from examples.constants import c
from examples.protocols import p
from examples.shared import ExamplesFlextShared
from examples.typings import t
from examples.utilities import u
from flext_core import e, r

if TYPE_CHECKING:
    from collections.abc import Callable


class Ex08ContainerRegistration(ExamplesFlextShared):
    """Registration and resolution checks for the container example."""

    _registered_service_name: str = u.PrivateAttr(default_factory=str)
    _registered_service_value: int = u.PrivateAttr(default_factory=int)

    @staticmethod
    def _write_outcome(write: Callable[[], p.Container]) -> p.Result[bool]:
        """Convert one container write into a result that keeps the raised cause."""
        try:
            _ = write()
        except e.ValidationError as exc:
            return r[bool].fail_op("container write", exc)
        return r[bool].ok(True)

    def _exercise_registration_and_resolution(self, container: p.Container) -> None:
        """Exercise register APIs plus get/get_typed/list/has checks."""
        self.section("registration_and_resolution")
        service_name = f"svc.{self.rand_str(6)}"
        service_value = self.rand_int(1, 1000)
        self._registered_service_name = service_name
        self._registered_service_value = service_value
        factory_name = f"svc.{self.rand_str(6)}"
        resource_name = f"svc.{self.rand_str(6)}"
        missing_name = f"svc.{self.rand_str(6)}"
        bad_factory_name = f"svc.{self.rand_str(6)}"
        register_ok = container.bind(service_name, service_value)
        self.audit_check("register.service.returns_self", register_ok is container)
        self.audit_check(
            "register.service.success", container.resolve(service_name).success
        )
        self.audit_check(
            "register.service.stored_value_matches",
            container.resolve(service_name, type_cls=int).unwrap() == service_value,
        )
        duplicate = self._write_outcome(
            lambda: container.bind(service_name, self.rand_int(1, 1000))
        )
        self.audit_check(
            "register.service.duplicate_rejected",
            duplicate.failure and service_name in (duplicate.error or ""),
        )
        self.audit_check(
            "register.service.duplicate_keeps_first",
            container.resolve(service_name, type_cls=int).unwrap() == service_value,
        )
        empty = self._write_outcome(lambda: container.bind("", self.rand_int(1, 1000)))
        self.audit_check(
            "register.service.empty_name_rejected",
            c.ERR_CONTAINER_NAME_EMPTY in (empty.error or ""),
        )
        reserved = self._write_outcome(
            lambda: container.bind(c.ServiceName.LOGGER, self.rand_int(1, 1000))
        )
        self.audit_check(
            "register.service.reserved_name_rejected",
            reserved.failure and c.ServiceName.LOGGER in (reserved.error or ""),
        )
        factory_calls = {"count": 0}

        def _factory_counter() -> int:
            factory_calls["count"] += 1
            return factory_calls["count"]

        register_factory_ok = container.factory(factory_name, _factory_counter)
        factory_duplicate = self._write_outcome(
            lambda: container.factory(factory_name, _factory_counter)
        )
        self.audit_check(
            "register.factory.duplicate_rejected",
            factory_duplicate.failure
            and factory_name in (factory_duplicate.error or ""),
        )

        def _factory_raises() -> int:
            error_message = self.rand_str(10)
            raise RuntimeError(error_message)

        register_factory_bad = container.factory(bad_factory_name, _factory_raises)
        self.audit_check(
            "register.factory.returns_self", register_factory_ok is container
        )
        self.audit_check(
            "register.factory.raising_factory_registers",
            register_factory_bad is container and container.has(bad_factory_name),
        )
        resource_calls = {"count": 0}

        def _resource_data() -> t.IntMapping:
            resource_calls["count"] += 1
            return {self.rand_str(4): resource_calls["count"]}

        register_resource_ok = container.resource(resource_name, _resource_data)
        cross_kind = self._write_outcome(
            lambda: container.resource(service_name, _resource_data)
        )
        self.audit_check(
            "register.resource.cross_kind_duplicate_rejected",
            cross_kind.failure and service_name in (cross_kind.error or ""),
        )
        self.audit_check(
            "register.resource.returns_self", register_resource_ok is container
        )
        get_service = container.resolve(service_name)
        get_factory = container.resolve(factory_name)
        get_resource = container.resolve(resource_name)
        get_missing = container.resolve(missing_name)
        get_bad_factory = container.resolve(bad_factory_name)
        self.audit_check("get.service.success", get_service.success)
        self.audit_check(
            "get.service.value_matches", get_service.unwrap() == service_value
        )
        self.audit_check("get.factory.success", get_factory.success)
        self.audit_check("get.factory.invoked_per_resolve", factory_calls["count"])
        self.audit_check("get.resource.success", get_resource.success)
        self.audit_check("get.resource.invoked_per_resolve", resource_calls["count"])
        self.audit_check("get.missing.failure", get_missing.failure)
        self.audit_check("get.bad_factory.failure", get_bad_factory.failure)
        get_typed_service = container.resolve(service_name, type_cls=int)
        get_typed_service_bad = container.resolve(service_name, type_cls=str)
        get_typed_factory = container.resolve(factory_name, type_cls=int)
        get_typed_missing = container.resolve(missing_name, type_cls=int)
        self.audit_check("get_typed.service.success", get_typed_service.success)
        self.audit_check(
            "get_typed.service.value_matches",
            (get_typed_service.value if get_typed_service.success else -1)
            == service_value,
        )
        self.audit_check(
            "get_typed.service.type_mismatch_failure", get_typed_service_bad.failure
        )
        self.audit_check("get_typed.factory.success", get_typed_factory.success)
        self.audit_check(
            "get_typed.resource_via_get.success",
            container.resolve(resource_name).success,
        )
        self.audit_check("get_typed.missing.failure", get_typed_missing.failure)
        self.audit_check("has_service.service.true", container.has(service_name))
        self.audit_check("has_service.factory.true", container.has(factory_name))
        self.audit_check("has_service.resource.true", container.has(resource_name))
        self.audit_check("has_service.missing.false", not container.has(missing_name))
        service_list = list(container.names())
        self.audit_check("list_services.contains.service", service_name in service_list)
        self.audit_check("list_services.contains.factory", factory_name in service_list)
        self.audit_check(
            "list_services.contains.resource", resource_name in service_list
        )
