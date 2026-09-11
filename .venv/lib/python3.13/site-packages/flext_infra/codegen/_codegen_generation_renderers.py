"""Template renderers for generated ``__init__`` files."""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import TYPE_CHECKING

from flext_infra import c, t, u
from flext_infra.codegen._codegen_generation_lazy_entries import (
    FlextInfraCodegenGenerationLazyEntriesMixin,
)

if TYPE_CHECKING:
    from flext_infra import p


# NOTE (multi-agent, flext-wkii.17.26 / agent: codex): lazy generation delegates
# exact models to flext-cli and proves every rendered initializer is Ruff-clean.
class FlextInfraCodegenGenerationRenderersMixin(
    FlextInfraCodegenGenerationLazyEntriesMixin
):
    """Render codegen models through the canonical CLI template facade."""

    @staticmethod
    def _render_model(
        template_name: str, context: p.Model, *, target_filename: str
    ) -> str:
        """Render and deterministically format a typed Python artifact.

        flext-perf.3 (agent: codex): the two Ruff invocations (import
        organization + format) run as a single piped shell pipeline
        (``set -o pipefail; ruff check --fix-only | ruff format``) instead
        of two separate ``run_raw`` subprocess round-trips, halving the
        per-file Python/subprocess orchestration overhead.
        """
        template_root = Path(__file__).resolve().parent.parent / "templates"
        rendered = u.Cli.template_render(
            template_root / template_name, context
        ).unwrap()
        compile(rendered, target_filename, "exec")
        filename = shlex.quote(target_filename)
        input_bytes = rendered.encode(c.Cli.ENCODING_DEFAULT)
        pipeline_cmd = (
            f"set -o pipefail;"
            f" {c.Infra.RUFF} {c.Infra.CHECK} --fix-only"
            f" --stdin-filename {filename} -"
            f" | {c.Infra.RUFF} {c.Infra.FORMAT}"
            f" --stdin-filename {filename} -"
        )
        pipe_result = u.Cli.run_raw(
            ["bash", "-c", pipeline_cmd], cwd=template_root, input_data=input_bytes
        )
        if pipe_result.failure:
            raise ValueError(pipe_result.error or "ruff formatting pipeline failed")
        output = pipe_result.unwrap()
        if output.exit_code != 0:
            detail = (output.stderr or output.stdout).strip()
            msg = f"ruff formatting pipeline failed ({output.exit_code}): {detail}"
            raise ValueError(msg)
        rendered_output: str = (
            t.Infra.STR_ADAPTER.validate_python(output.stdout).rstrip() + "\n"
        )
        return rendered_output


__all__: list[str] = ["FlextInfraCodegenGenerationRenderersMixin"]
