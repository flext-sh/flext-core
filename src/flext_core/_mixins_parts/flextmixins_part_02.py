"""Reusable mixins for service infrastructure.

Provide shared behaviors for services and handlers including structured
logging, DI-backed context handling, and operation tracking.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from flext_core import (
    c,
    e,
    m,
    p,
    r,
    u,
)

from .flextmixins_part_01 import (
    FlextMixins as FlextMixinsPart01,
)


class FlextMixins(FlextMixinsPart01):
    def _init_service(self, service_name: str | None = None) -> None:
        """Initialize service with automatic container registration."""
        effective_service_name: str = (
            service_name
            if service_name is not None and u.matches_type(service_name, str)
            else self.__class__.__name__
        )
        register_result = self._register_in_container(effective_service_name)
        if register_result.failure:
            error_msg = register_result.error
            if error_msg is None:
                error_msg = c.ERR_SERVICE_REGISTRATION_FAILED
            if "already registered" not in error_msg.lower():
                self.logger.warning(
                    c.LOG_SERVICE_REGISTRATION_FAILED,
                    service_name=effective_service_name,
                    error=register_result.error or c.ERR_SERVICE_REGISTRATION_FAILED,
                )

    def _register_in_container(self, service_name: str) -> p.Result[bool]:
        """Register self in global container for service discovery."""
        container = self.container
        was_registered = container.has(service_name)
        _ = container.bind(service_name, self)
        if was_registered or container.has(service_name):
            return r[bool].ok(True)
        operation = "register service in container"
        return r[bool].fail(
            e.render_template(
                c.ERR_TEMPLATE_FAILED_WITH_ERROR,
                operation=operation,
                error=c.ERR_SERVICE_REGISTRATION_FAILED,
                params=m.OperationErrorParams(
                    operation=operation,
                    reason=c.ERR_SERVICE_REGISTRATION_FAILED,
                ),
            ),
        )


__all__: list[str] = ["FlextMixins"]
