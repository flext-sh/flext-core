# Logging Guide

<!-- TOC START -->

- [Calls and settings](#calls-and-settings)
- [Native capture and lexical context](#native-capture-and-lexical-context)
- [Pipeline and safety](#pipeline-and-safety)
- [Delivery and integration](#delivery-and-integration)

<!-- TOC END -->

## Calls and settings

Use the public Core `u` facade over native `structlog` and logging handlers:

| Call | Meaning |
| --- | --- |
| `u.log(message, *args, level=None, **context)` | Ordinary INFO; `level` selects another standard severity. |
| `u.debug(...)` / `u.warning(...)` | DEBUG / WARNING through the same pipeline. |
| `u.trace(...)` | DEBUG detail, admitted only when `trace` is enabled and the threshold permits DEBUG. |
| `u.failure(..., exception=cause)` | ERROR with the original `BaseException`, including its saved traceback. |

An explicit `log_level` wins over `debug`; debug selects DEBUG only without an
explicit level. Trace changes detail, never severity or threshold: ERROR plus
trace still suppresses DEBUG/trace events but adds detailed ERROR stacks.

Core settings own `log_level`, `debug`, `trace`, `logging_profile`, and
`async_logging`, sourced through `FLEXT_*` environment fields. Child settings keep
their declared prefix; use their inherited fields, not a guessed `cli_log_level`.
The composition root supplies the actual provider as `runtime_settings` to
`u.configure_structlog`. `console` and `json` profiles select presentation, not
different events. Read active values with `u.logging_settings()` and composition
options with `u.logging_options()`; do not duplicate their defaults.

## Native capture and lexical context

Select synchronous handlers and real streams for deterministic capture; no
stdout redirection, polling, or sleeps are needed. This example deliberately
selects DEBUG and trace, captures actual observer/output receipts, and restores
the prior settings and options rather than assuming their values.

```python
import json
from io import StringIO

from structlog.contextvars import bound_contextvars

from flext_core import c, m, p, t, u

prior_settings: p.LoggingSettings = u.logging_settings()
prior_options: p.StructlogOptions = u.logging_options()
receipts: MutableSequence[p.LoggingRecord] = []
json_stream = StringIO()
console_stream = StringIO()


def observe(event: p.LoggingRecord) -> None:
    receipts.append(event)


selected = u.resolve_logging_settings(
    runtime_settings=prior_settings,
    log_level=c.LogLevel.DEBUG,
    debug=False,
    trace=True,
)
options = m.StructlogOptions(
    async_logging=False,
    handlers=(
        u.create_logging_handler(stream=json_stream),
        u.create_logging_handler(stream=console_stream, console_renderer=True),
    ),
    observers=(observe,),
)
cause = ValueError("invalid input")
try:
    u.configure_structlog(settings=options, runtime_settings=selected)
    with bound_contextvars(request_id="req-123"):
        _ = u.log("processed %s rows", 2)
        _ = u.debug("batch_ready")
        _ = u.warning("partial_batch")
        _ = u.trace("batch_detail", row_count=2)
        _ = u.failure("batch_failed", exception=cause)
    output = json_stream.getvalue().splitlines()
    assert len(receipts) == len(output)
    assert json.loads(output[0])[c.HandlerType.EVENT] == receipts[0].message
    assert receipts[0].message in console_stream.getvalue()
    assert receipts[-1].exception is cause
finally:
    u.configure_structlog(settings=prior_options, runtime_settings=prior_settings)
```

Native `bound_contextvars` restores lexical bindings, including nested scopes and
exception exits. Existing `u.bind_global_context`, `u.bind_context`, unbind, and
clear helpers remain valid; clearing removes keys, not a saved outer value. Do
not add a context-copy engine or manual restoration stack.

## Pipeline and safety

Threshold filtering precedes native printf interpolation and enrichment. Ordered
`processing_stages` implement `p.LoggingStage` over `t.LoggingEvent`; they run
before exception rendering and sanitization, not as a second logging engine.
Typed `p.LoggingObserver` callbacks receive one sanitized `p.LoggingRecord`
before native handler fanout. JSON and console render the same enriched event;
capture once at that boundary, never once per sink or by parsing rendered text.

Built-in sensitive-key redaction plus `redacted_keys`/`redacted_values` protects
rendered context; detailed stacks omit locals. The observer's original exception
is local-only and excluded from serialization: treat that object as sensitive.
If formatting, a stage, observer, or sink fails, its native error escapes. When
a business exception is present, Core preserves that same cause and traceback,
adding a note for the logging failure rather than replacing the business error.

## Delivery and integration

Explicit configuration reaches existing public/cached loggers through their
native lazy binding; no fresh-logger workaround is needed. An optionless
`u.configure_structlog()` initializes only once. Selected handlers own their
delivery lifecycle; use `u.create_logging_handler` for native formatting,
locking, and fail-loud stream errors, not a swallowing stdlib handler.

Without selected handlers/factory, Core diagnostics use stderr. Its managed
asynchronous writer acknowledges emission flushes, drains admitted work on
shutdown, and preserves the first stream failure without retry. Shutdown timeout
and admission errors propagate; failed teardown retains ownership. Closing does
not close the caller's stream. Native metadata/missing attributes retain native
errors; print's text, newline, and flush calls are not one atomic shutdown unit.

Core is the foundation and never imports outer FLEXT packages. The application
composition root injects settings, handlers, and observer ports. See
[Observability](https://github.com/flext-sh/flext-observability/blob/0.12.0-dev/docs/index.md) for capture/export
integration and [CLI](https://github.com/flext-sh/flext-cli/blob/0.12.0-dev/docs/index.md) for transport composition.
Retirement of duplicate Observability logging remains pending; this guide does
not certify that cutover or its gates.
