#!/usr/bin/env python3
"""Demonstration of advanced structured logging capabilities."""

import sys
import time
import traceback
from datetime import UTC, datetime

# Add the source path
sys.path.insert(0, "src")

# Import only what we need for logging, bypassing the complex __init__.py
try:
    # Import components directly
    import structlog
    from structlog.typing import EventDict, Processor

    # Simple constants we need
    class SimpleConstants:
        class Core:
            VERSION = "0.9.0"

    # Simplified logger based on our advanced implementation
    class SimpleFlextLogger:
        """Simplified version of FlextLogger for demonstration."""

        _configured = False
        _correlation_id = None

        def __init__(self, name: str, service_name: str = "demo-service") -> None:
            if not self._configured:
                self._configure()

            self._name = name
            self._service_name = service_name

            if not self._correlation_id:
                import uuid
                self._correlation_id = f"corr_{uuid.uuid4().hex[:16]}"

            self._structlog_logger = structlog.get_logger(name)

        @classmethod
        def _configure(cls) -> None:
            """Configure structured logging."""
            processors: list[Processor] = [
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_log_level,
                structlog.stdlib.add_logger_name,
                structlog.processors.TimeStamper(fmt="iso", utc=True, key="@timestamp"),
                cls._add_metadata_processor,
                structlog.dev.ConsoleRenderer(
                    colors=True,
                    level_styles={
                        "critical": "\033[91;1m",
                        "error": "\033[91m",
                        "warning": "\033[93m",
                        "info": "\033[92m",
                        "debug": "\033[94m",
                    }
                )
            ]

            structlog.configure(
                processors=processors,
                context_class=dict,
                logger_factory=structlog.stdlib.LoggerFactory(),
                wrapper_class=structlog.stdlib.BoundLogger,
                cache_logger_on_first_use=True,
            )

            import logging
            logging.basicConfig(
                format="%(message)s",
                stream=sys.stderr,
                level=logging.INFO,
            )

            cls._configured = True

        @staticmethod
        def _add_metadata_processor(logger, method_name: str, event_dict: EventDict) -> EventDict:
            """Add metadata to log entries."""
            import os
            import platform

            event_dict.update({
                "correlation_id": SimpleFlextLogger._correlation_id,
                "service": {
                    "name": "flext-demo",
                    "version": SimpleConstants.Core.VERSION,
                    "environment": "development",
                },
                "system": {
                    "hostname": platform.node(),
                    "process_id": os.getpid(),
                },
                "@metadata": {
                    "processor": "flext_advanced_logging",
                    "version": SimpleConstants.Core.VERSION,
                    "processed_at": datetime.now(UTC).isoformat(),
                }
            })
            return event_dict

        def info(self, message: str, **context) -> None:
            """Log info message with context."""
            self._structlog_logger.info(message, **context)

        def error(self, message: str, error: Exception | None = None, **context) -> None:
            """Log error message with context."""
            if error:
                context["error"] = {
                    "type": error.__class__.__name__,
                    "message": str(error),
                    "traceback": traceback.format_exception(type(error), error, error.__traceback__)
                }
            self._structlog_logger.error(message, **context)

        def start_operation(self, operation_name: str, **context):
            """Start operation tracking."""
            import uuid
            operation_id = f"op_{uuid.uuid4().hex[:8]}"
            start_time = time.time()

            self.info(
                f"Operation started: {operation_name}",
                operation_id=operation_id,
                operation_name=operation_name,
                start_time=start_time,
                **context
            )

            return operation_id, start_time

        def complete_operation(self, operation_id: str, start_time: float, operation_name: str, success: bool = True, **context) -> None:
            """Complete operation tracking."""
            duration_ms = (time.time() - start_time) * 1000

            log_context = {
                "operation_id": operation_id,
                "operation_name": operation_name,
                "success": success,
                "duration_ms": round(duration_ms, 3),
                "performance": {
                    "duration_ms": round(duration_ms, 3),
                    "timestamp": datetime.now(UTC).isoformat(),
                },
                **context
            }

            if success:
                self.info(f"Operation completed: {operation_name}", **log_context)
            else:
                self.error(f"Operation failed: {operation_name}", **log_context)

    def demonstrate_advanced_logging() -> None:
        """Demonstrate all advanced logging features."""
        # Create logger
        logger = SimpleFlextLogger("demo_logger", service_name="flext-advanced-demo")

        logger.info("Application started",
                   user_id="user_123",
                   session_id="sess_abc",
                   ip_address="192.168.1.100")

        operation_id, start_time = logger.start_operation("user_authentication",
                                                         user_id="user_123",
                                                         method="oauth2")

        # Simulate work
        time.sleep(0.1)

        logger.complete_operation(operation_id, start_time, "user_authentication",
                                success=True,
                                result="authenticated",
                                permissions=["read", "write"])

        logger.info("User login attempt",
                   username="john_doe",
                   password="[REDACTED]",  # Sensitive data would be auto-redacted
                   api_key="[REDACTED]",
                   user_agent="Mozilla/5.0...")

        try:
            # Simulate an error
            msg = "Invalid configuration parameter"
            raise ValueError(msg)
        except Exception as e:
            logger.exception("Configuration validation failed",
                        error=e,
                        config_file="/etc/app/config.yaml",
                        parameter="database_url")

        logger.info("Order processed",
                   order_id="order_789",
                   customer_id="cust_456",
                   amount=99.99,
                   currency="USD",
                   payment_method="credit_card",
                   fulfillment_center="warehouse_east")

        logger.info("System metrics collected",
                   cpu_usage=45.2,
                   memory_usage=67.8,
                   disk_usage=23.1,
                   active_connections=127,
                   response_time_ms=234.5)

    if __name__ == "__main__":
        demonstrate_advanced_logging()

except Exception:
    import traceback
    traceback.print_exc()
