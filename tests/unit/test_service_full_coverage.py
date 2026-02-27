"""Tests for FlextService to achieve full coverage.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import cast

import flext_core.service as service_mod
import pytest
from flext_core import c, m, p, r, t, u
from flext_core.service import FlextService, FlextSettings


class _Svc(FlextService[bool]):
    def execute(self) -> r[bool]:
        return r[bool].ok(True)


class _FakeConfig:
    version = "1"


def test_service_init_type_guards_and_properties(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert c.Errors.UNKNOWN_ERROR
    assert isinstance(m.Categories(), m.Categories)
    assert r[int].ok(1).is_success
    assert isinstance(t.ConfigMap.model_validate({"k": 1}), t.ConfigMap)
    assert u.Conversion.to_str(1) == "1"

    bad_ctx_runtime = m.ServiceRuntime.model_construct(
        config=FlextSettings(),
        context=cast("p.Context", "invalid-context"),
        container=cast("p.DI", "invalid-container"),
    )
    monkeypatch.setattr(
        _Svc, "_create_initial_runtime", classmethod(lambda cls: bad_ctx_runtime),
    )
    with pytest.raises(TypeError, match="Expected FlextContext"):
        _Svc()

    good_ctx = __import__(
        "flext_core.context", fromlist=["FlextContext"],
    ).FlextContext.create()
    bad_cfg_runtime = m.ServiceRuntime.model_construct(
        config=cast("p.Config", _FakeConfig()),
        context=good_ctx,
        container=cast("p.DI", "invalid-container"),
    )
    monkeypatch.setattr(
        _Svc, "_create_initial_runtime", classmethod(lambda cls: bad_cfg_runtime),
    )
    with pytest.raises(TypeError, match="Expected FlextSettings"):
        _Svc()


def test_service_create_runtime_container_overrides_branch() -> None:
    runtime = _Svc._create_runtime(container_overrides={"strict": True})
    assert isinstance(runtime, m.ServiceRuntime)


def test_service_create_initial_runtime_prefers_custom_config_type_and_context_property() -> (
    None
):
    class _CustomSettings(FlextSettings):
        pass

    class _CustomSvc(_Svc):
        @classmethod
        def _runtime_bootstrap_options(cls) -> p.RuntimeBootstrapOptions:
            return service_mod.p.RuntimeBootstrapOptions(
                config_type=_CustomSettings,
            )

    runtime = _CustomSvc._create_initial_runtime()
    assert isinstance(runtime.config, _CustomSettings)

    service = _Svc()
    assert service.context is not None
