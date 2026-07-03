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

from typing import cast

from flext_core import c, p, r, t
from flext_core._utilities.guards_type_core import FlextUtilitiesGuardsTypeCore
from flext_core._utilities.guards_type_model import FlextUtilitiesGuardsTypeModel


def _adapt_dispatcher_output(
    raw_output: t.JsonPayload | p.Result[t.JsonPayload] | None,
    dispatch_result: type[r[t.JsonPayload]],
) -> p.Result[t.JsonPayload]:
    result: p.Result[t.JsonPayload]
    if raw_output is None:
        result = dispatch_result.fail_op(
            "validate handler return payload",
            c.ERR_HANDLER_RETURNED_NONE,
        )
    elif isinstance(raw_output, p.ResultLike):
        if raw_output.failure:
            error_data_value = raw_output.error_data
            result = dispatch_result.fail(
                raw_output.error or c.ERR_HANDLER_FAILED,
                error_code=raw_output.error_code,
                error_data=(
                    error_data_value
                    if FlextUtilitiesGuardsTypeModel.pydantic_model(error_data_value)
                    else None
                ),
            )
        else:
            output_value = raw_output.value
            if FlextUtilitiesGuardsTypeCore.container(output_value):
                payload: t.JsonPayload = output_value
                result = dispatch_result.ok(payload)
            elif FlextUtilitiesGuardsTypeModel.pydantic_model(output_value):
                model_payload: t.JsonPayload = output_value
                result = dispatch_result.ok(model_payload)
            else:
                result = dispatch_result.fail_op(
                    "validate handler success payload",
                    c.ERR_HANDLER_RETURNED_NON_CONTAINER_SUCCESS_RESULT,
                )
    elif FlextUtilitiesGuardsTypeCore.container(
        raw_output,
    ) or FlextUtilitiesGuardsTypeModel.pydantic_model(raw_output):
        result = dispatch_result.ok(raw_output)
    else:
        result = dispatch_result.fail_op(
            "validate handler return payload",
            c.ERR_HANDLER_RETURNED_NON_CONTAINER_VALUE,
        )
    return result


def _normalize_dispatcher_output(
    raw_candidate: t.JsonPayload | p.ResultLike[t.JsonPayload] | None,
    dispatch_result: type[r[t.JsonPayload]],
) -> t.JsonPayload | p.Result[t.JsonPayload] | None:
    if isinstance(raw_candidate, p.Result):
        return cast("p.Result[t.JsonPayload]", raw_candidate)
    if raw_candidate is None:
        return None
    if FlextUtilitiesGuardsTypeCore.container(
        raw_candidate,
    ) or FlextUtilitiesGuardsTypeModel.pydantic_model(raw_candidate):
        return raw_candidate
    return dispatch_result.fail_op(
        "validate handler return payload",
        c.ERR_HANDLER_RETURNED_NON_CONTAINER_VALUE,
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
        raw_candidate = resolved_handler(message)
        raw_output = _normalize_dispatcher_output(raw_candidate, dispatch_result)
        return _adapt_dispatcher_output(raw_output, dispatch_result)
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
