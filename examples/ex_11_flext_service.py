"""Service example exercising the public ``s`` contract and runtime accessors."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, override

from examples import m
from examples import p, t
from examples.shared import ExamplesFlextShared
from examples import u
from flext_core import r, s


class _EchoService(s[str]):
    payload: Annotated[
        m.Examples.Payload,
        u.Field(description="Payload processed by the example service"),
    ]
    rule_error: Annotated[
        str,
        u.Field(
            description="Optional validation error returned by the example service"
        ),
    ] = ""

    @override
    def execute(self) -> p.Result[str]:
        validation = self.validate_business_rules()
        if validation.failure:
            return r[str].fail(validation.error or "rule_validation_failed")
        return r[str].ok(f"echo:{self.payload.text}")

    def service_info(self) -> t.JsonMapping:
        return {"service": type(self).__name__, "payload": self.payload.text}

    def valid(self) -> bool:
        validation: p.SuccessCheckable = self.validate_business_rules()
        return validation.success

    def validate_business_rules(self) -> p.Result[bool]:
        if self.rule_error:
            return r[bool](error=self.rule_error, success=False)
        return r[bool](value=True, success=True)


class ExampleService:
    """Public service example used by documentation snippets."""

    @classmethod
    def run(cls) -> p.Result[str]:
        """Execute the public example service and return its result."""
        service = _EchoService(payload=m.Examples.Payload(text="ok"))
        return service.execute()


class _ExampleServiceGolden(ExamplesFlextShared):
    """Golden-file harness for the service example."""

    @override
    def exercise(self) -> None:
        """Exercise service execution, validation, and runtime accessors."""
        service = _EchoService(payload=m.Examples.Payload(text="ok"))
        validation_failure = _EchoService(
            payload=m.Examples.Payload(text="ok"),
            rule_error="Failed to validate business_rule: E_RULE",
        )

        self.section("service_core_api")
        execute_result = service.execute()
        failed_validation = validation_failure.validate_business_rules()
        service_runtime = u.build_service_runtime(service)
        self.audit_check("execute.unwrap", execute_result.unwrap_or(""))
        self.audit_check(
            "execute.unwrap.matches", execute_result.unwrap_or("") == "echo:ok"
        )
        self.audit_check("runtime.type", type(service_runtime).__name__)
        self.audit_check("context.type", type(service.context).__name__)
        self.audit_check("settings.type", type(service.settings).__name__)
        self.audit_check("container.type", type(service.container).__name__)
        self.audit_check(
            "validate_business_rules.default.success",
            service.validate_business_rules().success,
        )
        self.audit_check("valid.default", service.valid())
        self.audit_check(
            "validate_business_rules.override.success", failed_validation.success
        )
        self.audit_check(
            "validate_business_rules.override.error", failed_validation.error
        )
        self.audit_check("valid.override", validation_failure.valid())

        self.section("runtime_creation_and_serialization")
        runtime_default = u.build_service_runtime(service)
        runtime_with_override = u.build_service_runtime(service, subproject="examples")
        info = service.service_info()
        self.audit_check(
            "create_runtime.default.context", type(runtime_default.context).__name__
        )
        self.audit_check(
            "create_runtime.full.container",
            type(runtime_with_override.container).__name__,
        )
        self.audit_check(
            "create_runtime.full.settings",
            type(runtime_with_override.settings).__name__,
        )
        self.audit_check("service_info.type", type(info).__name__)
        self.audit_check("service_info.payload", info["payload"])

        self.section("result_and_failures")
        ok_result = service.execute()
        failed_execute = validation_failure.execute()
        self.audit_check("ok.unwrap", ok_result.unwrap_or(""))
        self.audit_check("fail.error", failed_execute.failure)
        self.audit_check("fail.error.message", failed_execute.error)


if __name__ == "__main__":
    _ExampleServiceGolden(caller_file=Path(__file__)).run()
