"""Handler execution helper extracted from FlextDispatcher.

Encapsulates the success/failure pipeline that runs a resolved
``t.RoutedHandlerCallable`` against a routable message and adapts the
return value (``r[T]`` or raw payload) to the dispatcher's canonical
``r[t.JsonPayload]`` contract. Owned exclusively by ``FlextDispatcher``;
not part of any public surface.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import (
    FlextUtilitiesGuardsTypeCore,
    FlextUtilitiesGuardsTypeModel,
    FlextUtilitiesGuardsTypeProtocol,
    c,
    p,
    r,
    t,
)


def execute_dispatcher_handler(
    *,
    resolved_handler: t.RoutedHandlerCallable,
    message: p.Routable,
    route_name: str,
    logger: p.Logger,
) -> p.Result[t.JsonPayload]:
    """Execute ``resolved_handler(message)`` and adapt the outcome to ``r[JsonPayload]``.

    The handler may return either an ``r[T]`` instance (Result-like
    canonical) or a raw payload (container or Pydantic model). All other
    shapes are rejected with the canonical fail-op messages from the
    enforcement constants.
    """
    dispatch_result = r[t.JsonPayload]
    try:
        raw_output = resolved_handler(message)
        result: p.Result[t.JsonPayload]
        if FlextUtilitiesGuardsTypeProtocol.result_like(raw_output):
            if raw_output.failure:
                error_data_value = raw_output.error_data
                result = dispatch_result.fail(
                    raw_output.error or c.ERR_HANDLER_FAILED,
                    error_code=raw_output.error_code,
                    error_data=(
                        error_data_value
                        if FlextUtilitiesGuardsTypeModel.pydantic_model(
                            error_data_value,
                        )
                        else None
                    ),
                )
            else:
                output_value = raw_output.value
                if FlextUtilitiesGuardsTypeCore.container(
                    output_value
                ) or FlextUtilitiesGuardsTypeModel.pydantic_model(output_value):
                    result = dispatch_result.ok(output_value)
                else:
                    result = dispatch_result.fail_op(
                        "validate handler success payload",
                        c.ERR_HANDLER_RETURNED_NON_CONTAINER_SUCCESS_RESULT,
                    )
        elif FlextUtilitiesGuardsTypeCore.container(
            raw_output
        ) or FlextUtilitiesGuardsTypeModel.pydantic_model(raw_output):
            result = dispatch_result.ok(raw_output)
        else:
            result = dispatch_result.fail_op(
                "validate handler return payload",
                c.ERR_HANDLER_RETURNED_NON_CONTAINER_VALUE,
            )
        return result
    except (
        TypeError,
        ValueError,
        RuntimeError,
        KeyError,
        AttributeError,
        OSError,
        LookupError,
        ArithmeticError,
    ) as exc:
        logger.exception(c.LOG_HANDLER_EXECUTION_FAILED, route=route_name)
        return dispatch_result.fail_op("execute resolved handler", exc)


__all__ = ["execute_dispatcher_handler"]
