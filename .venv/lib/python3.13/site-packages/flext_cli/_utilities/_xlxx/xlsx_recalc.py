"""Generic headless recalculation and cache parity for XLSX bytes."""

from __future__ import annotations

import tempfile
from pathlib import Path

from flext_cli import c, m, p, r
from flext_cli._utilities.processes import FlextCliUtilitiesProcesses

from .xlsx_recalc_evidence import FlextCliUtilitiesXlsxRecalcEvidence
from .xlsx_snapshot import FlextCliUtilitiesXlsxSnapshot


class FlextCliUtilitiesXlsxRecalc(
    FlextCliUtilitiesXlsxSnapshot, FlextCliUtilitiesXlsxRecalcEvidence
):
    """Recalculate formula caches and prove parity through typed evidence."""

    # NOTE (multi-agent, mro-j2yt.1): the headless engine process terminates
    # at this private adapter; process spawning is consumed from the generic
    # processes facade without polluting the XLSX composition order.
    @classmethod
    def xlsx_recalc(
        cls, request: m.Cli.XlsxRecalcRequest
    ) -> p.Result[m.Cli.XlsxRecalcResult]:
        """Recalculate every formula cache through the headless office engine."""
        try:
            return cls._xlsx_recalc_unchecked(request)
        except (OSError, ValueError) as exc:
            detail = str(exc).strip() or exc.__class__.__name__
            return r[m.Cli.XlsxRecalcResult].fail(
                f"{c.Cli.XlsxError.RECALC_FAILED}: {detail}"
            )

    @staticmethod
    def _xlsx_recalc_unchecked(
        request: m.Cli.XlsxRecalcRequest,
    ) -> p.Result[m.Cli.XlsxRecalcResult]:
        with tempfile.TemporaryDirectory(
            prefix=c.Cli.XLSX_RECALC_TEMP_PREFIX
        ) as workspace:
            workdir = Path(workspace)
            input_dir = workdir / "input"
            output_dir = workdir / "output"
            profile_dir = workdir / c.Cli.XLSX_RECALC_PROFILE_DIR_NAME
            input_dir.mkdir()
            output_dir.mkdir()
            profile_dir.mkdir()
            source_path = input_dir / c.Cli.XLSX_RECALC_SOURCE_NAME
            source_path.write_bytes(request.source)
            executable, *arguments = c.Cli.XLSX_RECALC_COMMAND
            started = FlextCliUtilitiesProcesses.process_start(
                (
                    executable,
                    f"{c.Cli.XLSX_RECALC_USER_PROFILE_ARGUMENT_PREFIX}{profile_dir.as_uri()}",
                    *arguments,
                    str(output_dir),
                    str(source_path),
                ),
                cwd=workdir,
            )
            if started.failure:
                return r[m.Cli.XlsxRecalcResult].fail(
                    f"{c.Cli.XlsxError.RECALC_FAILED}: {started.error}"
                )
            process = started.value
            completed = process.wait(timeout=c.Cli.XLSX_RECALC_TIMEOUT_SECONDS)
            if completed.failure:
                killed = process.kill()
                detail = completed.error or "process wait failed"
                if killed.failure:
                    detail = f"{detail}; kill failed: {killed.error}"
                return r[m.Cli.XlsxRecalcResult].fail(
                    f"{c.Cli.XlsxError.RECALC_FAILED}: {detail}"
                )
            if completed.value != 0:
                detail = process.stderr.strip() or process.stdout.strip()
                return r[m.Cli.XlsxRecalcResult].fail(
                    f"{c.Cli.XlsxError.RECALC_FAILED}: exit={completed.value}: {detail}"
                )
            content = (output_dir / c.Cli.XLSX_RECALC_SOURCE_NAME).read_bytes()
        return r[m.Cli.XlsxRecalcResult].ok(m.Cli.XlsxRecalcResult(content=content))

    @classmethod
    def xlsx_recalc_parity(
        cls, request: m.Cli.XlsxRecalcParityRequest
    ) -> p.Result[m.Cli.XlsxRecalcParityReport]:
        """Recalculate and compare cached values against source formulas."""
        formula_snapshot = cls.xlsx_snapshot(
            m.Cli.XlsxSnapshotRequest(source=request.source, data_only=False)
        )
        if formula_snapshot.failure:
            return r[m.Cli.XlsxRecalcParityReport].fail(
                f"{c.Cli.XlsxError.PARITY_FAILED}: {formula_snapshot.error}"
            )
        recalculated = cls.xlsx_recalc(m.Cli.XlsxRecalcRequest(source=request.source))
        if recalculated.failure:
            return r[m.Cli.XlsxRecalcParityReport].fail(
                f"{c.Cli.XlsxError.PARITY_FAILED}: {recalculated.error}"
            )
        value_snapshot = cls.xlsx_snapshot(
            m.Cli.XlsxSnapshotRequest(source=recalculated.value.content, data_only=True)
        )
        if value_snapshot.failure:
            return r[m.Cli.XlsxRecalcParityReport].fail(
                f"{c.Cli.XlsxError.PARITY_FAILED}: {value_snapshot.error}"
            )
        cache_evidence = cls._formula_cache_evidence(recalculated.value.content)
        if cache_evidence.failure:
            return r[m.Cli.XlsxRecalcParityReport].fail(
                f"{c.Cli.XlsxError.PARITY_FAILED}: {cache_evidence.error}"
            )
        uncached_cells, empty_result_cells = cache_evidence.value
        error_cells: tuple[str, ...] = ()
        for sheet in value_snapshot.value.sheets:
            for cell in sheet.cells:
                if cell.formula is None:
                    continue
                value = cell.value
                if isinstance(value, m.Cli.XlsxTextValue) and value.value.startswith(
                    c.Cli.XLSX_ERROR_CELL_PREFIX
                ):
                    error_cells = (*error_cells, f"{sheet.name}!{cell.coordinate}")
        formula_count = formula_snapshot.value.formula_count
        count_matches = (
            request.expected_formula_count is None
            or formula_count == request.expected_formula_count
        )
        report = m.Cli.XlsxRecalcParityReport(
            content=recalculated.value.content,
            recalculated=True,
            formula_count=formula_count,
            error_cells=error_cells,
            uncached_cells=uncached_cells,
            empty_result_cells=empty_result_cells,
            ok=not error_cells and not uncached_cells and count_matches,
        )
        return r[m.Cli.XlsxRecalcParityReport].ok(report)


__all__: tuple[str, ...] = ("FlextCliUtilitiesXlsxRecalc",)
