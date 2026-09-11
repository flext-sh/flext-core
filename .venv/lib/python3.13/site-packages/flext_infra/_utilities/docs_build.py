"""Build helpers for docs services."""

from __future__ import annotations

import sys
from collections.abc import MutableMapping
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, cast

from flext_cli import u
from flext_infra._utilities.docs import FlextInfraUtilitiesDocs
from flext_infra.constants import c
from flext_infra.models import m

if TYPE_CHECKING:
    from types import ModuleType

    from flext_infra.protocols import p


class FlextInfraUtilitiesDocsBuild:
    """Reusable build helpers exposed through ``u.Infra``."""

    @staticmethod
    def _module_callable(module: ModuleType, name: str) -> p.Infra.MkDocsAnyCallable:
        """Return a named callable from a lazily loaded module."""
        value: p.AttributeProbe = getattr(module, name)
        if callable(value):
            return cast("p.Infra.MkDocsAnyCallable", value)
        msg = f"{module.__name__}.{name} is not callable"
        raise OSError(msg)

    @staticmethod
    def _mkdocs_exception_types(module: ModuleType) -> tuple[type[BaseException], ...]:
        """Return MkDocs exception classes from a lazily loaded module."""
        names = (
            "Abort",
            "BuildError",
            "ConfigurationError",
            "MkDocsException",
            "PluginError",
        )
        errors: list[type[BaseException]] = []
        for name in names:
            value: p.AttributeProbe = getattr(module, name)
            if not isinstance(value, type) or not issubclass(value, BaseException):
                msg = f"{module.__name__}.{name} is not an exception type"
                raise OSError(msg)
            errors.append(value)
        return tuple(errors)

    @staticmethod
    def _load_mkdocs_config(
        load: p.Infra.MkDocsLoadConfig, settings: Path, site_dir: Path
    ) -> MutableMapping[str, p.AttributeProbe]:
        """Load and validate a MkDocs config mapping."""
        config_raw = load(config_file_path=str(settings), site_dir=str(site_dir))
        if not isinstance(config_raw, MutableMapping):
            msg = "mkdocs.config.load_config did not return a mutable mapping"
            raise OSError(msg)
        return config_raw

    @staticmethod
    def docs_mkdocs_config_files(scope: m.Infra.DocScope) -> tuple[Path, ...]:
        """Return primary mkdocs.yml then optional product mkdocs.yaml."""
        configs: list[Path] = []
        primary = scope.path / "mkdocs.yml"
        secondary = scope.path / "mkdocs.yaml"
        if primary.is_file():
            configs.append(primary)
        if secondary.is_file() and secondary.resolve() != primary.resolve():
            configs.append(secondary)
        return tuple(configs)

    @staticmethod
    def docs_run_mkdocs(
        scope: m.Infra.DocScope, *, runner: p.Cli.CommandRunner
    ) -> m.Infra.DocsPhaseReport:
        """Run MkDocs for primary yml and optional product yaml configs."""
        configs = FlextInfraUtilitiesDocsBuild.docs_mkdocs_config_files(scope)
        if not configs:
            return m.Infra.DocsPhaseReport(
                phase="build",
                scope=scope.name,
                result=c.Infra.ResultStatus.FAIL,
                reason="mkdocs.yml not found",
                site_dir="",
                passed=False,
            )
        primary_report: m.Infra.DocsPhaseReport | None = None
        for settings in configs:
            suffix = "" if settings.suffix == ".yml" else "-product"
            report = FlextInfraUtilitiesDocsBuild._docs_run_one_mkdocs(
                scope, settings=settings, runner=runner, site_suffix=suffix
            )
            if primary_report is None:
                primary_report = report
            if not report.passed:
                return report
        if primary_report is None:
            return m.Infra.DocsPhaseReport(
                phase="build",
                scope=scope.name,
                result=c.Infra.ResultStatus.FAIL,
                reason="mkdocs build produced no report",
                site_dir="",
                passed=False,
            )
        if len(configs) > 1:
            return primary_report.model_copy(
                update={
                    "reason": f"{primary_report.reason}; product mkdocs.yaml also built"
                }
            )
        return primary_report

    @staticmethod
    def _docs_run_one_mkdocs(
        scope: m.Infra.DocScope,
        *,
        settings: Path,
        runner: p.Cli.CommandRunner,
        site_suffix: str,
    ) -> m.Infra.DocsPhaseReport:
        """Build one MkDocs config file into a site directory."""
        site_dir = (
            scope.path
            / c.Infra.DEFAULT_DOCS_OUTPUT_DIR
            / f"{c.Infra.DIR_SITE}{site_suffix}"
        ).resolve()
        if not isinstance(runner, type):
            completed = runner.run_raw(
                [
                    sys.executable,
                    "-m",
                    "mkdocs",
                    c.Infra.DIR_BUILD,
                    "--strict",
                    "-f",
                    str(settings),
                    "-d",
                    str(site_dir),
                ],
                cwd=scope.path,
            )
            if completed.failure:
                return m.Infra.DocsPhaseReport(
                    phase="build",
                    scope=scope.name,
                    result=c.Infra.ResultStatus.FAIL,
                    reason=completed.error or f"mkdocs build failed ({settings.name})",
                    site_dir=site_dir.as_posix(),
                    passed=False,
                )
            output = completed.value
            if output.exit_code == 0:
                return m.Infra.DocsPhaseReport(
                    phase="build",
                    scope=scope.name,
                    result=c.Infra.ResultStatus.OK,
                    reason=f"build succeeded ({settings.name})",
                    site_dir=site_dir.as_posix(),
                    passed=True,
                )
            reason_lines = (output.stderr or output.stdout).strip().splitlines()
            return m.Infra.DocsPhaseReport(
                phase="build",
                scope=scope.name,
                result=c.Infra.ResultStatus.FAIL,
                reason=(
                    reason_lines[-1]
                    if reason_lines
                    else f"mkdocs exited {output.exit_code} ({settings.name})"
                ),
                site_dir=site_dir.as_posix(),
                passed=False,
            )
        try:
            FlextInfraUtilitiesDocsBuild._run_mkdocs_api(settings, site_dir)
        except c.EXC_OS_VALUE as exc:
            return m.Infra.DocsPhaseReport(
                phase="build",
                scope=scope.name,
                result=c.Infra.ResultStatus.FAIL,
                reason=str(exc) or f"mkdocs build failed ({settings.name})",
                site_dir=site_dir.as_posix(),
                passed=False,
            )
        return m.Infra.DocsPhaseReport(
            phase="build",
            scope=scope.name,
            result=c.Infra.ResultStatus.OK,
            reason=f"build succeeded ({settings.name})",
            site_dir=site_dir.as_posix(),
            passed=True,
        )

    @staticmethod
    def _run_mkdocs_api(settings: Path, site_dir: Path) -> None:
        """Run MkDocs build via the Python API with lazy imports.

        Converts mkdocs-specific exceptions to ``OSError`` so callers only
        need to catch standard exception types.
        """
        mkdocs_build = import_module("mkdocs.commands.build")
        mkdocs_config = import_module("mkdocs.config")
        mkdocs_exceptions = import_module("mkdocs.exceptions")
        load = cast(
            "p.Infra.MkDocsLoadConfig",
            FlextInfraUtilitiesDocsBuild._module_callable(mkdocs_config, "load_config"),
        )
        build = cast(
            "p.Infra.MkDocsBuild",
            FlextInfraUtilitiesDocsBuild._module_callable(mkdocs_build, "build"),
        )
        mkdocs_error_types = FlextInfraUtilitiesDocsBuild._mkdocs_exception_types(
            mkdocs_exceptions
        )
        site_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            config_obj = FlextInfraUtilitiesDocsBuild._load_mkdocs_config(
                load, settings, site_dir
            )
            config_obj["strict"] = True
            _ = build(config_obj, dirty=False)
        except mkdocs_error_types as exc:
            msg = str(exc) or "mkdocs build failed"
            raise OSError(msg) from exc

    @staticmethod
    def docs_serve_mkdocs(
        scope: m.Infra.DocScope, *, dev_addr: str, livereload: bool, strict: bool
    ) -> m.Infra.DocsPhaseReport:
        """Serve one scope through the MkDocs Python serve API (blocking)."""
        settings = scope.path / "mkdocs.yml"
        if not settings.exists():
            return m.Infra.DocsPhaseReport(
                phase="serve",
                scope=scope.name,
                result=c.Infra.ResultStatus.FAIL,
                reason="mkdocs.yml not found",
                site_dir="",
                passed=False,
            )
        try:
            serve_module = import_module("mkdocs.commands.serve")
            serve_fn = cast(
                "p.Infra.MkDocsServe",
                FlextInfraUtilitiesDocsBuild._module_callable(serve_module, "serve"),
            )
            serve_fn(
                config_file=str(settings),
                livereload=livereload,
                dev_addr=dev_addr,
                strict=strict,
            )
        except c.EXC_OS_VALUE as exc:
            return m.Infra.DocsPhaseReport(
                phase="serve",
                scope=scope.name,
                result=c.Infra.ResultStatus.FAIL,
                reason=str(exc) or "mkdocs serve failed",
                site_dir="",
                passed=False,
            )
        return m.Infra.DocsPhaseReport(
            phase="serve",
            scope=scope.name,
            result=c.Infra.ResultStatus.OK,
            reason="dev server stopped",
            site_dir="",
            passed=True,
        )

    @staticmethod
    def docs_write_build_reports(
        scope: m.Infra.DocScope, report: m.Infra.DocsPhaseReport
    ) -> None:
        """Persist the standard build summary and markdown report."""
        _ = u.Cli.json_write(
            scope.report_dir / "build-summary.json",
            {c.Infra.RK_SUMMARY: report.model_dump()},
        )
        _ = FlextInfraUtilitiesDocs.write_markdown(
            scope.report_dir / "build-report.md",
            [
                "# Docs Build Report",
                "",
                f"Scope: {report.scope}",
                f"Result: {report.result}",
                f"Reason: {report.reason}",
                f"Site dir: {report.site_dir}",
            ],
        )


__all__: list[str] = ["FlextInfraUtilitiesDocsBuild"]
