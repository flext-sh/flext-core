# Logging Migration Report for flext-core

## Summary

Total files with logging imports: 0

## Files to Migrate

## Migration Steps

1. Replace logging imports:

    ```python
    # Old
    import logging
    logger = logging.getLogger(__name__)

    # New
    from flext_observability.logging import get_logger
    logger = get_logger(__name__)
    ```

2. Add setup_logging to your main entry point:

    ```python
    from flext_observability import setup_logging

    setup_logging(
        service_name="flext-core",
        log_level="INFO",
        json_logs=True
    )
    ```

3. Update logging calls to use structured format:

    ```python
    # Old
    logger.info("Processing %s items", count)

    # New
    logger.info("Processing items", count=count)
    ```

See `examples/logging_migration.py` for a complete example.
