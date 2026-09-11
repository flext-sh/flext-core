"""Test-scope context manager for matchers.

Exposes ``Tests.Matchers.scope`` for isolated test execution scopes.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Generator
from contextlib import contextmanager, nullcontext
from pathlib import Path

from flext_core import u
from flext_tests import c, m, t
from flext_tests._utilities.settings import FlextTestsConfigHelpersUtilitiesMixin


class FlextTestsMatchersScopeMixin:
    """Isolated test-scope helper exposed under ``Tests.Matchers``."""

    class Tests:
        """Container for test utility storages and aliases."""

        class Matchers:
            """Test matchers with powerful generalist methods."""

            @staticmethod
            @contextmanager
            def scope(
                **kwargs: t.Tests.TestobjectSerializable,
            ) -> Generator[m.Tests.TestScope]:
                """Enhanced isolated test execution scope.

                Provides isolated configuration, container, and context for tests.
                Supports temporary environment variables, working directory changes,
                and automatic cleanup functions.

                Args:
                    **kwargs: Parameters validated via m.ScopeParams model
                        - settings: Initial configuration values
                        - container: Initial container/service mappings
                        - context: Initial context values
                        - cleanup: Sequence of cleanup functions to call on exit
                        - env: Temporary environment variables (restored on exit)
                        - cwd: Temporary working directory (restored on exit)

                Yields:
                    TestScope with settings, container, and context dicts

                Raises:
                    ValueError: If parameter validation fails (via Pydantic model)

                """
                try:
                    params = m.Tests.ScopeParams.model_validate(kwargs)
                except c.EXC_BASIC_TYPE as exc:
                    message = f"Parameter validation failed: {exc}"
                    raise ValueError(message) from exc
                original_cwd: Path | None = None
                env_context = (
                    FlextTestsConfigHelpersUtilitiesMixin.env_vars_context(params.env)
                    if params.env is not None
                    else nullcontext()
                )
                try:
                    with env_context:
                        if params.cwd is not None:
                            original_cwd = Path.cwd()
                            cwd_path = (
                                Path(params.cwd)
                                if u.matches_type(params.cwd, "str")
                                else params.cwd
                            )
                            os.chdir(cwd_path)
                        cfg: t.MappingKV[str, t.Tests.TestobjectSerializable] = {}
                        if params.settings:
                            cfg = dict(params.settings)
                        container_dict = {
                            k: v
                            for k, v in (params.container or {}).items()
                            if t.Tests.general_value(v)
                        }
                        context_map: t.MappingKV[
                            str, t.Tests.TestobjectSerializable
                        ] = {}
                        if params.context:
                            context_map = dict(params.context)
                        yield m.Tests.TestScope.model_validate({
                            "settings": cfg,
                            "container": container_dict,
                            "context": context_map,
                        })
                finally:
                    if original_cwd is not None:
                        os.chdir(original_cwd)
                    if params.cleanup is not None:
                        for cleanup_func in params.cleanup:
                            try:
                                cleanup_func()
                            except (
                                OSError,
                                RuntimeError,
                                TypeError,
                                ValueError,
                                AttributeError,
                            ) as e:
                                warnings.warn(
                                    c.Tests.ERR_SCOPE_CLEANUP_FAILED.format(
                                        error=str(e)
                                    ),
                                    RuntimeWarning,
                                    stacklevel=2,
                                )


__all__: list[str] = ["FlextTestsMatchersScopeMixin"]
