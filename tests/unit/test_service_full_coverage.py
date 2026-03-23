"""Tests for FlextService to achieve full coverage.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import cast, override

import pytest

from flext_core import FlextContext, FlextModelsService, FlextService, FlextSettings, r
from tests import c, m, p, t, u


class TestServiceFullCoverage:
    class _Svc(FlextService[bool]):
        @override
        def execute(self) -> r[bool]:
            return r[bool].ok(True)

    class _FakeConfig:
        version = "1"

    def test_service_init_type_guards_and_properties(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        assert c.UNKNOWN_ERROR
        assert isinstance(m.Categories(), m.Categories)
        assert r[int].ok(1).is_success
        assert isinstance(t.ConfigMap({"k": 1}), t.ConfigMap)
        assert u.to_str(1) == "1"
        bad_ctx_runtime = m.ServiceRuntime.model_construct(
            config=FlextSettings(),
            context=cast("p.Context", "invalid-context"),
            container=cast("p.Container", "invalid-container"),
        )

        def _bad_ctx_runtime_factory(
            _cls: type[TestServiceFullCoverage._Svc],
        ) -> m.ServiceRuntime:
            return bad_ctx_runtime

        monkeypatch.setattr(
            self._Svc,
            "_create_initial_runtime",
            classmethod(_bad_ctx_runtime_factory),
        )
        service_with_bad_ctx = self._Svc()
        assert service_with_bad_ctx.context == "invalid-context"
        good_ctx = FlextContext.create()
        bad_cfg_runtime = m.ServiceRuntime.model_construct(
            config=cast("p.Settings", self._FakeConfig()),
            context=good_ctx,
            container=cast("p.Container", "invalid-container"),
        )

        def _bad_cfg_runtime_factory(
            _cls: type[TestServiceFullCoverage._Svc],
        ) -> m.ServiceRuntime:
            return bad_cfg_runtime

        monkeypatch.setattr(
            self._Svc,
            "_create_initial_runtime",
            classmethod(_bad_cfg_runtime_factory),
        )
        service_with_bad_cfg = self._Svc()
        assert isinstance(service_with_bad_cfg.config, self._FakeConfig)

    def test_service_create_runtime_container_overrides_branch(self) -> None:
        runtime = self._Svc._create_runtime(container_overrides={"strict": True})
        assert isinstance(runtime, m.ServiceRuntime)

    def test_service_create_initial_runtime_prefers_custom_config_type_and_context_property(
        self,
    ) -> None:
        class _CustomSettings(FlextSettings):
            pass

        svc_type = self._Svc

        class _CustomSvc(svc_type):
            @classmethod
            @override
            def _runtime_bootstrap_options(
                cls,
            ) -> FlextModelsService.RuntimeBootstrapOptions:
                return FlextModelsService.RuntimeBootstrapOptions(
                    config_type=_CustomSettings,
                )

        runtime = _CustomSvc()._create_initial_runtime()
        assert isinstance(runtime.config, _CustomSettings)
        service = self._Svc()
        assert service.context is not None
