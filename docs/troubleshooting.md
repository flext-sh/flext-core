# Troubleshooting

Reference for common issues encountered while adopting the FLEXT Core 1.0.0 modernization pillars.

---

## Dispatcher Registration Fails

**Symptom**: `FlextDispatcher.register_command` returns a failure result or `FlextDispatcherRegistry` summary lists errors.

**Fix**:

- Ensure handlers subclass `FlextHandlers` or expose a `handle` method that returns `FlextResult`.
- Check that message types are hashable and consistent (e.g., use the same class for registration and dispatch).
- Inspect the error payload via `summary.errors` or the `FlextResult.error` string.

```python
result = dispatcher.register_command(MyCommand, handler)
if result.is_failure:
    raise RuntimeError(result.error)
```

---

## Missing Context Metadata in Logs

**Symptom**: Logs lack correlation IDs or request metadata after migration.

**Fix**:

- Wrap entry points with `FlextContext.Operation.scope`.
- Verify logging uses `FlextLogger`; other loggers skip context processors.
- Check that services call `FlextContext.Request.set_user_id` or related helpers before emitting logs.

---

## Configuration Not Loaded

**Symptom**: `FlextContainer.get("MyConfig")` returns a failure or config values are `None`.

**Fix**:

- Instantiate the `FlextConfig` subclass before registering it with the container.
- Confirm the `.env` file (if used) is in the working directory when the config is created.
- Use `TargetConfig.model_dump()` for debugging to ensure fields are populated.

---

## Dispatcher Dispatch Returns Failure

**Symptom**: `dispatcher.dispatch(message)` returns a failure result.

**Fix**:

- Inspect `result.error` – underlying handlers return detailed messages.
- Confirm the handler registration used the correct message type.
- If using `register_function`, ensure the callable returns either a bare value or a `FlextResult`.

---

## Container Already Registered Error

**Symptom**: Registering a service raises/returns “already registered”.

**Fix**:

- Either reuse the existing instance (`container.get(...)`) or explicitly call `container.unregister(...)` before registering a replacement.
- In tests, call `FlextContainer.reset_global()` (available in `container.py`) to clear state between runs.

---

## Version Drift

**Symptom**: Documentation references a different version than the installed package.

**Fix**:

- Check `flext_core.__version__` and `FlextVersionManager` values.
- Regenerate documentation snippets after bumping `pyproject.toml` and `version.py`.

---

Reach out to the maintainers if issues persist; provide dispatcher summaries, context dumps, and configuration payloads where possible to speed up diagnosis.
