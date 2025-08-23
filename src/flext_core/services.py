"""FlextCore Services - Service base classes and processors.

Service-related classes separated from utilities to avoid circular imports.
Provides base classes for service processors, simple services, and common patterns.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

from flext_core.mixins import FlextServiceMixin
from flext_core.result import FlextResult, FlextResultUtils
from flext_core.utilities import FlextPerformance, FlextProcessingUtils, FlextUtilities

# =============================================================================
# FLEXT SERVICE PROCESSOR - Template base for processors to eliminate boilerplate
# =============================================================================


class FlextServiceProcessor[ServiceRequestT, ServiceDomainT, ServiceResultT](
    FlextServiceMixin, ABC
):
    """Base class for processors providing JSON and batch helpers.

    Subclass in concrete processors to reuse common patterns without repeating
    boilerplate for JSON parsing, model validation, batch handling and logging.
    """

    # --- Common defaults to remove boilerplate ---
    def __init__(self) -> None:
        super().__init__()

    def get_service_name(self) -> str:  # pragma: no cover - simple default
        return getattr(self, "service_name", self.__class__.__name__)

    def initialize_service(
        self,
    ) -> FlextResult[None]:  # pragma: no cover - simple default
        return FlextResult[None].ok(None)

    # --- Abstract Template Methods ---
    @abstractmethod
    def process(self, request: ServiceRequestT) -> FlextResult[ServiceDomainT]:
        """Process the request and return a domain object result."""

    @abstractmethod
    def build(self, domain: ServiceDomainT, *, correlation_id: str) -> ServiceResultT:
        """Build the final output from the domain object (pure function)."""

    # --- Template Orchestration with Auto Performance ---
    def run_with_metrics(
        self,
        category: str,
        request: ServiceRequestT,
    ) -> FlextResult[ServiceResultT]:
        """Run process→build with automatic performance tracking for the category."""
        # Import here to avoid circular imports

        @FlextPerformance.track_performance(category)
        def _inner(req: ServiceRequestT) -> FlextResult[ServiceResultT]:
            result = self.process(req)
            if result.is_failure:
                return FlextResult[ServiceResultT].fail(
                    result.error or "Processing failed"
                )
            corr = FlextUtilities.generate_correlation_id()
            built = self.build(result.value, correlation_id=corr)
            return FlextResult[ServiceResultT].ok(built)

        return _inner(request)

    def process_json[TReq, TRes](
        self,
        json_text: str,
        model_cls: type[TReq],
        handler: Callable[[TReq], FlextResult[TRes]],
        *,
        correlation_label: str = "correlation_id",
    ) -> FlextResult[TRes]:
        """Parse JSON into model and dispatch to handler with structured logging."""
        # Import here to avoid circular imports

        correlation_id = FlextUtilities.generate_correlation_id()
        self.log_info("Processing JSON", **{correlation_label: correlation_id})

        model_result = FlextProcessingUtils.parse_json_to_model(json_text, model_cls)
        if model_result.is_failure:
            error_msg = model_result.error or "Invalid JSON"
            self.log_error(f"JSON parsing/validation failed: {error_msg}")
            return FlextResult[TRes].fail(error_msg)

        result = handler(model_result.value)
        if result.is_success:
            self.log_info("Operation successful", **{correlation_label: correlation_id})
        else:
            self.log_error(
                "Operation failed",
                error=result.error,
                **{correlation_label: correlation_id},
            )
        return result

    def run_batch[TReq, TRes](
        self,
        items: list[TReq],
        handler: Callable[[TReq], FlextResult[TRes]],
    ) -> tuple[list[TRes], list[str]]:
        """Run a batch with standard collection of successes and errors."""
        return FlextResultUtils.batch_process(items, handler)


__all__ = [
    "FlextServiceProcessor",
]
