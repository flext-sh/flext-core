"""Generate and validate base.mk files from templates."""

from __future__ import annotations

import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, TextIO, cast, override

from flext_core import FlextService, r, t
from flext_infra.basemk.engine import FlextInfraBaseMkTemplateEngine
from flext_infra.constants import FlextInfraConstants as c
from flext_infra.models import FlextInfraModels as m
from flext_infra.subprocess import FlextInfraCommandRunner


class _TemplateRenderer(Protocol):
    def render_all(self, config: m.Infra.BaseMkConfig | None = None) -> r[str]: ...


class FlextInfraBaseMkGenerator(FlextService[str]):
    """Generate base.mk content and write to file or stream."""

    def __init__(self, template_engine: _TemplateRenderer | None = None) -> None:
        """Initialize the base.mk generator."""
        super().__init__()
        self._template_engine = template_engine or FlextInfraBaseMkTemplateEngine()
        self._runner: FlextInfraCommandRunner | None = None

    @property
    def _get_runner(self) -> FlextInfraCommandRunner:
        """Lazily initialize the command runner."""
        if self._runner is None:
            self._runner = FlextInfraCommandRunner()
        return self._runner

    @override
    def execute(self) -> r[str]:
        return self.generate()

    def generate(
        self,
        config: m.Infra.BaseMkConfig | Mapping[str, t.Scalar] | None = None,
    ) -> r[str]:
        """Generate base.mk content from configuration."""
        config_result = self._normalize_config(config)
        if config_result.is_failure:
            return r[str].fail(config_result.error or "invalid base.mk configuration")

        config_value: m.Infra.BaseMkConfig | None = cast(
            "m.Infra.BaseMkConfig | None",
            config_result.value,
        )
        render_result = self._template_engine.render_all(config_value)
        if render_result.is_failure:
            return r[str].fail(render_result.error or "base.mk render failed")

        return self._validate_generated_output(render_result.value)

    def write(
        self,
        content: str,
        *,
        output: Path | None = None,
        stream: TextIO | None = None,
    ) -> r[bool]:
        """Write generated content to file or stream."""
        if output is None:
            target_stream = stream
            if target_stream is None:
                return r[bool].fail("stdout stream is required for console output")
            try:
                _ = target_stream.write(content)
                return r[bool].ok(True)
            except OSError as exc:
                return r[bool].fail(f"base.mk stdout write failed: {exc}")

        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            _ = output.write_text(content, encoding=c.Infra.Encoding.DEFAULT)
            return r[bool].ok(True)
        except OSError as exc:
            return r[bool].fail(f"base.mk write failed: {exc}")

    def _normalize_config(
        self,
        config: m.Infra.BaseMkConfig | Mapping[str, t.Scalar] | None,
    ) -> r[m.Infra.BaseMkConfig]:
        if config is None:
            return r[m.Infra.BaseMkConfig].ok(
                FlextInfraBaseMkTemplateEngine.default_config()
            )
        if isinstance(config, m.Infra.BaseMkConfig):
            return r[m.Infra.BaseMkConfig].ok(config)
        try:
            normalized = m.Infra.BaseMkConfig.model_validate(dict(config))
            return r[m.Infra.BaseMkConfig].ok(normalized)
        except (TypeError, ValueError) as exc:
            return r[m.Infra.BaseMkConfig].fail(
                f"base.mk configuration validation failed: {exc}",
            )

    def _validate_generated_output(self, content: str) -> r[str]:
        """Validate generated base.mk by running make --dry-run."""
        try:
            with tempfile.TemporaryDirectory(prefix="flext-basemk-") as temp_dir_name:
                temp_dir = Path(temp_dir_name)
                base_mk_path = temp_dir / c.Infra.Files.BASE_MK
                makefile_path = temp_dir / c.Infra.Files.MAKEFILE_FILENAME

                _ = base_mk_path.write_text(content, encoding=c.Infra.Encoding.DEFAULT)
                _ = makefile_path.write_text(
                    "include base.mk\n",
                    encoding=c.Infra.Encoding.DEFAULT,
                )

                process_result = self._get_runner.run(
                    ["make", "-C", str(temp_dir), "--dry-run", "help"],
                )
                if process_result.is_failure:
                    error_text = process_result.error or "make validation failed"
                    return r[str].fail(
                        f"generated base.mk validation failed: {error_text}",
                    )
        except OSError as exc:
            return r[str].fail(f"generated base.mk validation failed: {exc}")

        return r[str].ok(content)


__all__ = ["FlextInfraBaseMkGenerator"]
