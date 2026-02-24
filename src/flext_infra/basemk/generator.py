"""Generate and validate base.mk files from templates."""

from __future__ import annotations

import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, TextIO, override

from flext_core.result import FlextResult as r
from flext_core.service import FlextService
from flext_core.typings import t

from flext_infra.basemk.engine import TemplateEngine
from flext_infra.constants import ic
from flext_infra.models import im
from flext_infra.subprocess import CommandRunner


class _TemplateRenderer(Protocol):
    def render_all(self, config: im.BaseMkConfig | None = None) -> r[str]: ...


class BaseMkGenerator(FlextService[str]):
    """Generate base.mk content and write to file or stream."""

    def __init__(self, template_engine: _TemplateRenderer | None = None) -> None:
        """Initialize the base.mk generator."""
        super().__init__()
        self._template_engine = template_engine or TemplateEngine()
        self._runner = CommandRunner()

    @override
    def execute(self) -> r[str]:
        return self.generate()

    def generate(
        self, config: im.BaseMkConfig | Mapping[str, t.ScalarValue] | None = None
    ) -> r[str]:
        """Generate base.mk content from configuration."""
        config_result = self._normalize_config(config)
        if config_result.is_failure:
            return r[str].fail(config_result.error or "invalid base.mk configuration")

        render_result = self._template_engine.render_all(config_result.value)
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
            _ = output.write_text(content, encoding=ic.Encoding.DEFAULT)
            return r[bool].ok(True)
        except OSError as exc:
            return r[bool].fail(f"base.mk write failed: {exc}")

    def _normalize_config(
        self,
        config: im.BaseMkConfig | Mapping[str, t.ScalarValue] | None,
    ) -> r[im.BaseMkConfig]:
        if config is None:
            return r[im.BaseMkConfig].ok(TemplateEngine.default_config())
        if type(config) is im.BaseMkConfig or (im.BaseMkConfig in type(config).__mro__):
            return r[im.BaseMkConfig].ok(config)
        try:
            normalized = im.BaseMkConfig.model_validate(dict(config))
            return r[im.BaseMkConfig].ok(normalized)
        except (TypeError, ValueError) as exc:
            return r[im.BaseMkConfig].fail(
                f"base.mk configuration validation failed: {exc}",
            )

    def _validate_generated_output(self, content: str) -> r[str]:
        """Validate generated base.mk by running make --dry-run."""
        try:
            with tempfile.TemporaryDirectory(prefix="flext-basemk-") as temp_dir_name:
                temp_dir = Path(temp_dir_name)
                base_mk_path = temp_dir / "base.mk"
                makefile_path = temp_dir / ic.Files.MAKEFILE_FILENAME

                _ = base_mk_path.write_text(content, encoding=ic.Encoding.DEFAULT)
                _ = makefile_path.write_text(
                    "include base.mk\n",
                    encoding=ic.Encoding.DEFAULT,
                )

                process_result = self._runner.run(
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


__all__ = ["BaseMkGenerator"]
