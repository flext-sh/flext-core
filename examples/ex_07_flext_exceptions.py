"""FlextExceptions (e) golden-file example for public API coverage."""

from __future__ import annotations

from shared import Examples

from flext_core import FlextExceptions, c, e, r


class Ex07FlextExceptions(Examples):
    """Exercise FlextExceptions public API."""

    def _exercise_base_error(self) -> None:
        """Exercise BaseError construction and serialization helpers."""
        self.section("base_error")

        base_message = self.rand_str(10)
        base_error_code = f"E_{self.rand_str(6).upper()}"
        base_scope = self.rand_str(6)
        base_channel = self.rand_str(6)
        base_corr = f"corr-{self.rand_str(8)}"
        base_operation = self.rand_str(8)

        base = e.BaseError(
            base_message,
            error_code=base_error_code,
            context={"scope": base_scope},
            metadata={"channel": base_channel},
            correlation_id=base_corr,
            auto_correlation=False,
            auto_log=False,
            operation=base_operation,
        )
        self.check("base.class", type(base).__name__)
        self.check("base.message_matches", base.message == base_message)
        self.check("base.error_code_matches", base.error_code == base_error_code)
        self.check("base.correlation_id_matches", base.correlation_id == base_corr)
        self.check("base.auto_log", base.auto_log)
        self.check(
            "base.meta.scope_matches",
            base.metadata.attributes.get("scope") == base_scope,
        )
        self.check(
            "base.meta.channel_matches",
            base.metadata.attributes.get("channel") == base_channel,
        )
        self.check(
            "base.meta.operation_matches",
            base.metadata.attributes.get("operation") == base_operation,
        )

        payload = base.to_dict()
        self.check(
            "base.to_dict.type_matches", payload.get("error_type") == "BaseError"
        )
        self.check(
            "base.to_dict.message_matches", payload.get("message") == base_message
        )
        self.check(
            "base.to_dict.error_code_matches",
            payload.get("error_code") == base_error_code,
        )
        self.check("base.to_dict.has_timestamp", "timestamp" in payload)

        auto_message = self.rand_str(10)
        auto_error_code = f"E_{self.rand_str(6).upper()}"

        auto_corr = e.BaseError(
            auto_message,
            error_code=auto_error_code,
            auto_correlation=True,
            auto_log=False,
        )
        self.check(
            "base.auto_corr.prefix",
            auto_corr.correlation_id is not None
            and auto_corr.correlation_id.startswith("exc_"),
        )

    def _exercise_specific_exceptions(self) -> None:
        """Exercise specialized exception subclasses."""
        self.section("subclasses")

        msg = self.rand_str(10)
        field = self.rand_str(7)
        field_value = self.rand_str(7)
        try:
            raise e.ValidationError(msg, field=field, value=field_value)
        except e.ValidationError as exc:
            self.check("ValidationError.message_matches", str(exc) == msg)
            self.check("ValidationError.field_matches", exc.field == field)
            self.check("ValidationError.value_matches", exc.value == field_value)

        msg = self.rand_str(10)
        config_key = self.rand_str(10)
        config_source = self.rand_str(8)
        try:
            raise e.ConfigurationError(
                msg,
                config_key=config_key,
                config_source=config_source,
            )
        except e.ConfigurationError as exc:
            self.check("ConfigurationError.message_matches", str(exc) == msg)
            self.check(
                "ConfigurationError.config_key_matches", exc.config_key == config_key
            )
            self.check(
                "ConfigurationError.config_source_matches",
                exc.config_source == config_source,
            )

        msg = self.rand_str(10)
        host = self.rand_str(8)
        port = self.rand_int(1, 65000)
        timeout = self.rand_float(0.1, 10.0)
        try:
            raise e.ConnectionError(msg, host=host, port=port, timeout=timeout)
        except e.ConnectionError as exc:
            self.check("ConnectionError.message_matches", str(exc) == msg)
            self.check("ConnectionError.host_matches", exc.host == host)
            self.check("ConnectionError.port_matches", exc.port == port)
            self.check("ConnectionError.timeout_matches", exc.timeout == timeout)

        msg = self.rand_str(10)
        timeout_seconds = self.rand_float(0.1, 10.0)
        operation = self.rand_str(9)
        try:
            raise e.TimeoutError(
                msg, timeout_seconds=timeout_seconds, operation=operation
            )
        except e.TimeoutError as exc:
            self.check("TimeoutError.message_matches", str(exc) == msg)
            self.check(
                "TimeoutError.timeout_seconds_matches",
                exc.timeout_seconds == timeout_seconds,
            )
            self.check("TimeoutError.operation_matches", exc.operation == operation)

        msg = self.rand_str(10)
        auth_method = self.rand_str(8)
        user_id = self.rand_str(8)
        try:
            raise e.AuthenticationError(msg, auth_method=auth_method, user_id=user_id)
        except e.AuthenticationError as exc:
            self.check("AuthenticationError.message_matches", str(exc) == msg)
            self.check(
                "AuthenticationError.auth_method_matches",
                exc.auth_method == auth_method,
            )
            self.check("AuthenticationError.user_id_matches", exc.user_id == user_id)

        msg = self.rand_str(10)
        user_id = self.rand_str(8)
        resource = self.rand_str(10)
        permission = self.rand_str(8)
        try:
            raise e.AuthorizationError(
                msg,
                user_id=user_id,
                resource=resource,
                permission=permission,
            )
        except e.AuthorizationError as exc:
            self.check("AuthorizationError.message_matches", str(exc) == msg)
            self.check("AuthorizationError.user_id_matches", exc.user_id == user_id)
            self.check("AuthorizationError.resource_matches", exc.resource == resource)
            self.check(
                "AuthorizationError.permission_matches",
                exc.permission == permission,
            )

        msg = self.rand_str(10)
        resource_type = self.rand_str(7)
        resource_id = self.rand_str(7)
        try:
            raise e.NotFoundError(
                msg, resource_type=resource_type, resource_id=resource_id
            )
        except e.NotFoundError as exc:
            self.check("NotFoundError.message_matches", str(exc) == msg)
            self.check(
                "NotFoundError.resource_type_matches",
                exc.resource_type == resource_type,
            )
            self.check(
                "NotFoundError.resource_id_matches", exc.resource_id == resource_id
            )

        msg = self.rand_str(10)
        resource_type = self.rand_str(7)
        resource_id = self.rand_str(7)
        conflict_reason = self.rand_str(9)
        try:
            raise e.ConflictError(
                msg,
                resource_type=resource_type,
                resource_id=resource_id,
                conflict_reason=conflict_reason,
            )
        except e.ConflictError as exc:
            self.check("ConflictError.message_matches", str(exc) == msg)
            self.check(
                "ConflictError.resource_type_matches",
                exc.resource_type == resource_type,
            )
            self.check(
                "ConflictError.resource_id_matches", exc.resource_id == resource_id
            )
            self.check(
                "ConflictError.conflict_reason_matches",
                exc.conflict_reason == conflict_reason,
            )

        msg = self.rand_str(10)
        limit = self.rand_int(1, 1000)
        window_seconds = self.rand_int(1, 1000)
        retry_after = self.rand_float(0.1, 10.0)
        try:
            raise e.RateLimitError(
                msg,
                limit=limit,
                window_seconds=window_seconds,
                retry_after=retry_after,
            )
        except e.RateLimitError as exc:
            self.check("RateLimitError.message_matches", str(exc) == msg)
            self.check("RateLimitError.limit_matches", exc.limit == limit)
            self.check(
                "RateLimitError.window_seconds_matches",
                exc.window_seconds == window_seconds,
            )
            self.check(
                "RateLimitError.retry_after_matches", exc.retry_after == retry_after
            )

        msg = self.rand_str(10)
        service_name = self.rand_str(8)
        failure_count = self.rand_int(1, 100)
        reset_timeout = self.rand_float(0.1, 60.0)
        try:
            raise e.CircuitBreakerError(
                msg,
                service_name=service_name,
                failure_count=failure_count,
                reset_timeout=reset_timeout,
            )
        except e.CircuitBreakerError as exc:
            self.check("CircuitBreakerError.message_matches", str(exc) == msg)
            self.check(
                "CircuitBreakerError.service_name_matches",
                exc.service_name == service_name,
            )
            self.check(
                "CircuitBreakerError.failure_count_matches",
                exc.failure_count == failure_count,
            )
            self.check(
                "CircuitBreakerError.reset_timeout_matches",
                exc.reset_timeout == reset_timeout,
            )

        msg = self.rand_str(10)
        expected_type = str
        actual_type = int
        try:
            raise e.TypeError(msg, expected_type=expected_type, actual_type=actual_type)
        except e.TypeError as exc:
            self.check("TypeError.message_matches", str(exc) == msg)
            self.check(
                "TypeError.expected_type_matches",
                exc.expected_type is expected_type,
            )
            self.check(
                "TypeError.actual_type_matches",
                exc.actual_type is actual_type,
            )

        msg = self.rand_str(10)
        operation = self.rand_str(8)
        reason = self.rand_str(8)
        try:
            raise e.OperationError(msg, operation=operation, reason=reason)
        except e.OperationError as exc:
            self.check("OperationError.message_matches", str(exc) == msg)
            self.check("OperationError.operation_matches", exc.operation == operation)
            self.check("OperationError.reason_matches", exc.reason == reason)

        msg = self.rand_str(10)
        attribute_name = self.rand_str(8)
        attribute_context = self.rand_str(10)
        try:
            raise e.AttributeAccessError(
                msg,
                attribute_name=attribute_name,
                attribute_context=attribute_context,
            )
        except e.AttributeAccessError as exc:
            self.check("AttributeAccessError.message_matches", str(exc) == msg)
            self.check(
                "AttributeAccessError.attribute_name_matches",
                exc.attribute_name == attribute_name,
            )
            self.check(
                "AttributeAccessError.attribute_context_matches",
                exc.attribute_context == attribute_context,
            )

    def _exercise_factories_and_helpers(self) -> None:
        """Exercise factory helpers and kwargs extraction APIs."""
        self.section("factories_helpers")

        validation_msg = self.rand_str(10)
        created_validation = e.create_error("ValidationError", validation_msg)
        self.check(
            "create_error.ValidationError.type_matches",
            type(created_validation).__name__ == "ValidationError",
        )
        self.check(
            "create_error.ValidationError.message_matches",
            str(created_validation) == validation_msg,
        )

        attribute_msg = self.rand_str(10)
        created_attribute = e.create_error("AttributeError", attribute_msg)
        self.check(
            "create_error.AttributeError.type_matches",
            type(created_attribute).__name__ == "AttributeError",
        )
        self.check(
            "create_error.AttributeError.message_matches",
            str(created_attribute) == attribute_msg,
        )

        dynamic_message = self.rand_str(10)
        dynamic_error_code = f"E_{self.rand_str(6).upper()}"
        dynamic_field = self.rand_str(7)
        dynamic_value = self.rand_str(7)
        dynamic_caller = self.rand_str(8)
        dynamic_corr = f"corr-{self.rand_str(8)}"

        created_dynamic = e.create(
            dynamic_message,
            error_code=dynamic_error_code,
            field=dynamic_field,
            value=dynamic_value,
            metadata={"caller": dynamic_caller},
            correlation_id=dynamic_corr,
        )
        self.check(
            "create.type_matches", type(created_dynamic).__name__ == "ValidationError"
        )
        self.check(
            "create.error_code_matches",
            created_dynamic.error_code == dynamic_error_code,
        )
        self.check(
            "create.metadata.caller_matches",
            created_dynamic.metadata.attributes.get("caller") == dynamic_caller,
        )
        self.check(
            "create.correlation_id_matches",
            created_dynamic.correlation_id == dynamic_corr,
        )

        prep_corr_input = f"corr-{self.rand_str(8)}"
        prep_meta_key = self.rand_str(6)
        prep_meta_value = self.rand_str(6)
        prep_config = self.rand_str(7)
        prep_field_existing = self.rand_str(7)
        prep_custom = self.rand_str(7)
        prep_field_forced = self.rand_str(7)

        prepared = e.prepare_exception_kwargs(
            {
                "correlation_id": prep_corr_input,
                "metadata": {prep_meta_key: prep_meta_value},
                "auto_log": True,
                "auto_correlation": True,
                "config": prep_config,
                "field": prep_field_existing,
                "custom": prep_custom,
            },
            {"field": prep_field_forced},
        )
        (
            prep_corr,
            prep_metadata,
            prep_auto_log,
            prep_auto_corr,
            prep_config_out,
            prep_extra,
        ) = prepared
        self.check("prepare.correlation_id_matches", prep_corr == prep_corr_input)
        self.check("prepare.metadata_type", type(prep_metadata).__name__)
        self.check("prepare.auto_log", prep_auto_log)
        self.check("prepare.auto_correlation", prep_auto_corr)
        self.check("prepare.config_matches", prep_config_out == prep_config)
        self.check(
            "prepare.extra.field_matches", prep_extra.get("field") == prep_field_forced
        )
        self.check(
            "prepare.extra.custom_matches", prep_extra.get("custom") == prep_custom
        )

        extract_corr_in = f"corr-{self.rand_str(8)}"
        extract_meta_key = self.rand_str(6)
        extract_meta_value = self.rand_str(6)
        extracted_corr, extracted_meta = e.extract_common_kwargs({
            "correlation_id": extract_corr_in,
            "metadata": {extract_meta_key: extract_meta_value},
            "field": self.rand_str(5),
        })
        self.check("extract.correlation_id_matches", extracted_corr == extract_corr_in)
        self.check("extract.metadata.kind", type(extracted_meta).__name__)

        instance_factory = FlextExceptions()
        call_message = self.rand_str(10)
        call_error_code = f"E_{self.rand_str(6).upper()}"
        call_field = self.rand_str(7)
        call_value = self.rand_str(7)
        instance_created = instance_factory(
            call_message,
            error_code=call_error_code,
            field=call_field,
            value=call_value,
        )
        self.check(
            "__call__.type_matches",
            type(instance_created).__name__ == "ValidationError",
        )
        self.check(
            "__call__.error_code_matches",
            instance_created.error_code == call_error_code,
        )
        self.check("__call__.message_matches", str(instance_created) == call_message)

    def _exercise_metrics(self) -> None:
        """Exercise exception metrics recording and reset APIs."""
        self.section("metrics")

        e.clear_metrics()
        e.record_exception(e.ValidationError)
        e.record_exception(e.ValidationError)
        e.record_exception(e.ConfigurationError)

        metrics = e.get_metrics()
        metric_map = metrics.root
        self.check(
            "metrics.total_exceptions_matches", metric_map.get("total_exceptions") == 3
        )
        counts_obj = metric_map.get("exception_counts")
        if isinstance(counts_obj, dict):
            counts_map = dict(counts_obj)
            self.check(
                "metrics.validation_count_matches",
                counts_map.get("ValidationError") == 2,
            )
            self.check(
                "metrics.configuration_count_matches",
                counts_map.get("ConfigurationError") == 1,
            )
        else:
            self.check("metrics.validation_count", "missing")
            self.check("metrics.configuration_count", "missing")

        self.check(
            "metrics.summary_nonempty", bool(metric_map.get("exception_counts_summary"))
        )
        self.check(
            "metrics.unique_types_matches",
            metric_map.get("unique_exception_types") == 2,
        )

        e.clear_metrics()
        after_clear = e.get_metrics().root
        self.check(
            "metrics.cleared_total_matches", after_clear.get("total_exceptions") == 0
        )

    def exercise(self) -> None:
        """Run all sections for golden-file verification."""
        self.section("imports")
        self.check("import.e_is_FlextExceptions", e is FlextExceptions)
        import_value = self.rand_str(8)
        self.check("import.r_ok", r[str].ok(import_value).value == import_value)
        self.check("import.constant", c.Errors.UNKNOWN_ERROR)

        msg = self.rand_str(10)
        try:
            raise ValueError(msg)
        except ValueError as exc:
            self.check("style.raise_msg_matches", str(exc) == msg)

        self._exercise_base_error()
        self._exercise_specific_exceptions()
        self._exercise_factories_and_helpers()
        self._exercise_metrics()


if __name__ == "__main__":
    Ex07FlextExceptions(__file__).run()
