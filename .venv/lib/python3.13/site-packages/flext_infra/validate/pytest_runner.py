"""Typed, profiled pytest execution under one absolute process deadline."""

from __future__ import annotations

import os
import shlex
import sys
from defusedxml import ElementTree as DefusedET
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Self, override

from flext_core import r

from flext_infra import c, config, m, u
from flext_infra.base import s
from flext_infra.validate.cprofile_report import FlextInfraCProfileReport
from flext_infra.validate.pytest_diag import FlextInfraPytestDiagExtractor
from flext_infra.validate.pytest_selector import FlextInfraPytestSelectorValidator
from flext_infra.validate.testmon_db import FlextInfraTestmonDbInspector

if TYPE_CHECKING:
    from flext_infra._models.deps_tool_config import FlextInfraModelsDepsToolSettings
    from flext_infra import p, t

    PytestPolicy = FlextInfraModelsDepsToolSettings.PytestConfig


class FlextInfraPytestRunner(s[int]):
    """Run one exact pytest request and persist its performance evidence."""

    started_at_monotonic: Annotated[
        float,
        m.Field(gt=0, description="Monotonic timestamp captured before FLEXT imports."),
    ]
    file: Annotated[
        str | None, m.Field(default=None, description="Exact pytest file or nodeid.")
    ] = None
    match: Annotated[
        str | None, m.Field(default=None, description="Exact pytest -k expression.")
    ] = None
    what: Annotated[
        str | None, m.Field(default=None, description="Exact pytest execution mode.")
    ] = None
    target: Annotated[
        str, m.Field(min_length=1, description="Default repository-relative test root.")
    ]
    reports: Annotated[
        str, m.Field(min_length=1, description="Repository-relative test report root.")
    ]
    verbose: Annotated[bool, m.Field(description="Expose child output live.")] = False
    diagnostic: Annotated[
        bool, m.Field(description="Use expanded pytest diagnostics.")
    ] = False
    profile: Annotated[
        bool, m.Field(description="Wrap the child pytest process in cProfile.")
    ] = False

    @staticmethod
    def _environment_value(name: str) -> str:
        """Read one runner input through the canonical environment facade."""
        return u.Cli.env_read(name).unwrap().strip()

    @staticmethod
    def _environment_flag(name: str) -> bool:
        """Parse one strict Make boolean without accepting arbitrary text."""
        raw = FlextInfraPytestRunner._environment_value(name)
        if raw in {"", "0", "N"}:
            return False
        if raw in {"1", "Y"}:
            return True
        msg = f"{name} must be empty, 0, 1, N, or Y"
        raise ValueError(msg)

    @classmethod
    def from_environment(cls, *, started_at_monotonic: float) -> Self:
        """Parse the exact Make-owned environment boundary once."""
        raw_args = cls._environment_value(c.Infra.PYTEST_ENV_ARGS)
        raw_files = cls._environment_value(c.Infra.PYTEST_ENV_FILES)
        if raw_args:
            msg = "PYTEST_ARGS is forbidden; use typed FILE, MATCH, and WHAT selectors"
            raise ValueError(msg)
        if raw_files:
            msg = "FILES is forbidden for pytest; use one exact FILE selector"
            raise ValueError(msg)
        return cls(
            repository_root=Path.cwd(),
            started_at_monotonic=started_at_monotonic,
            file=cls._environment_value(c.Infra.PYTEST_ENV_FILE) or None,
            match=cls._environment_value(c.Infra.PYTEST_ENV_MATCH) or None,
            what=cls._environment_value(c.Infra.PYTEST_ENV_WHAT) or None,
            target=cls._environment_value(c.Infra.PYTEST_ENV_TARGET),
            reports=cls._environment_value(c.Infra.PYTEST_ENV_REPORTS),
            verbose=cls._environment_flag(c.Infra.PYTEST_ENV_VERBOSE),
            diagnostic=cls._environment_flag(c.Infra.PYTEST_ENV_DIAG),
            profile=cls._environment_flag(c.Infra.PYTEST_ENV_PROFILE),
        )

    @u.model_validator(mode="after")
    def _validate_paths_and_selectors(self) -> Self:
        """Reject selector and output paths that escape the active project."""
        selector = FlextInfraPytestSelectorValidator(
            repository_root=self.root, file=self.file, match=self.match, what=self.what
        )
        resolved_selector = selector.execute()
        if resolved_selector.failure:
            raise ValueError(resolved_selector.error or "invalid pytest selector")
        for field_name, value in (("target", self.target), ("reports", self.reports)):
            path = Path(value)
            if (
                path.is_absolute()
                or any(part in {"", ".", ".."} for part in path.parts)
                or any(character in value for character in "\0\r\n")
            ):
                msg = f"{field_name} must be a normalized repository-relative path"
                raise ValueError(msg)
        target_result = FlextInfraPytestSelectorValidator.resolve_file(
            self.root, self.target
        )
        if target_result.failure:
            raise ValueError(target_result.error or "invalid pytest target")
        return self

    def _report_directory(self) -> Path:
        """Create one collision-resistant report directory under the project."""
        run_id = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%S.%fZ") + f"-{os.getpid()}"
        # Annotated explicitly: `root` is a computed_field on a generic base,
        # so its Path return is erased to Any through the s[int] alias and the
        # derived path would silently propagate Any into the return.
        report_root: Path = self.root / self.reports
        report_dir: Path = report_root / run_id
        u.Cli.ensure_dir(report_dir).unwrap()
        return report_dir

    _CACHE_WHATS: frozenset[str] = frozenset({
        "cache-status",
        "cache-clear",
        "cache-checkpoint",
    })

    @staticmethod
    def _memory_gb() -> int:
        """Read physical memory from the operating-system owner."""
        page_size = os.sysconf("SC_PAGE_SIZE")
        physical_pages = os.sysconf("SC_PHYS_PAGES")
        if page_size <= 0 or physical_pages <= 0:
            msg = "physical memory capacity is unavailable"
            raise ValueError(msg)
        return max(1, (page_size * physical_pages) // (1024 * 1024 * 1024))

    def parallel_worker_budget(self, policy: PytestPolicy) -> int:
        """Derive one bounded xdist budget from typed CPU and memory owners."""
        cpu_budget = os.cpu_count() or 1
        memory_budget = max(1, self._memory_gb() // policy.parallel_worker_memory_gb)
        return max(1, min(policy.parallel_workers, cpu_budget, memory_budget))

    def _ci_disables_coverage(self) -> bool:
        """True when Make CI token is exact make.ci.value (CI=Y)."""
        raw = self._environment_value(c.Infra.PYTEST_ENV_CI)
        return raw == config.Infra.codegen.make.ci.value

    def _cov_enabled(self) -> bool:
        """True when COV=Y selects the coverage measurement run.

        Coverage is its own explicit request. A full test run is not one: it
        asks for every test to execute, which testmon expresses by keeping its
        database and skipping only the selection step.
        """
        return self._environment_flag(c.Infra.PYTEST_ENV_COV)

    def _testmon_db_path(self) -> Path:
        """Return the repository-local pytest-testmon SQLite path."""
        path: Path = Path(self.root) / ".testmondata"
        return path

    def _is_cache_maintenance(self) -> bool:
        """True when WHAT selects a testmon DB maintenance handler."""
        return self.what in self._CACHE_WHATS

    def _require_junit(self, junit_file: Path, pytest_log: Path) -> p.Result[bool]:
        """Require a non-empty parseable JUnit document after a green run."""
        if not junit_file.is_file() or junit_file.stat().st_size == 0:
            return r[bool].fail(
                self._artifact_failure_detail(
                    f"junit report was not generated or is empty: {junit_file}",
                    pytest_log,
                )
            )
        try:
            DefusedET.parse(junit_file)
        except DefusedET.ParseError as exc:
            return r[bool].fail(
                self._artifact_failure_detail(
                    f"junit report is not parseable: {junit_file}: {exc}", pytest_log
                )
            )
        return r[bool].ok(True)

    def _coverage_argv(self, report_dir: Path, *, focused: bool) -> tuple[str, ...]:
        """Build mutually exclusive testmon vs coverage argv.

        pytest-testmon nests its Coverage object under an outer pytest-cov
        stack when both are active. Flushing that stack calls get_data() on an
        empty outer collector and emits CoverageWarning: No data was collected.
        With filterwarnings=["error"] that kills xdist workers, so the two
        never share an argv.

        Every run that is not an explicit coverage measurement keeps the
        testmon database. ``WHAT=full`` asks for every test to execute, which
        testmon expresses by skipping selection while still recording what it
        observed: dropping the plugin instead would discard that record and
        leave the next incremental run with nothing to select from.
        """
        if self._cov_enabled():
            if focused:
                msg = (
                    "COV=Y forbids FILE=/MATCH= selectors; subset coverage is "
                    "not a valid fail-under measurement"
                )
                raise ValueError(msg)
            return (
                "--cov",
                f"--cov-report=xml:{report_dir / 'coverage.xml'}",
                "--no-cov-on-fail",
            )
        if self.what == "full":
            return ("--testmon", "--testmon-noselect", "--no-cov")
        return ("--testmon", "--no-cov")

    def build_command(self, report_dir: Path) -> tuple[str, ...]:
        """Build the exact child argv from the typed tooling policy."""
        pytest = config.Infra.tooling.tools.pytest
        focused = self.file is not None or self.match is not None
        target = self.file or self.target
        report_args = pytest.diagnostic_args if self.diagnostic else pytest.report_args
        coverage_args = self._coverage_argv(report_dir, focused=focused)
        parallel_args = (
            ("-n", "0")
            if focused
            else (
                "-n",
                str(self.parallel_worker_budget(pytest)),
                "--dist",
                pytest.parallel_distribution,
                # pytest-benchmark disables itself under xdist and warns while
                # configuring. Projects run filterwarnings=["error"], which
                # turns that warning into an INTERNALERROR before collection.
                # Asking for the same outcome up front keeps the run truthful:
                # benchmarks are off because the run is parallel, not silenced.
                "--benchmark-disable",
            )
        )
        # CI=Y deselects docker and remote suites entirely. Fixture-level
        # skips in flext-tests remain as a second fail-closed boundary for
        # unmarked tests that still call FlextTestsDocker.
        ci_marker_args = (
            ("-m", "not docker and not remote") if self._ci_disables_coverage() else ()
        )
        optional_args = (
            *(("-k", self.match) if self.match is not None else ()),
            *(("-x",) if self.fail_fast else ()),
        )
        python_argv = (
            (
                sys.executable,
                "-m",
                "cProfile",
                "-o",
                str(report_dir / "pytest.pstats"),
                "-m",
                "pytest",
            )
            if self.profile
            else (sys.executable, "-m", "pytest")
        )
        return (
            *python_argv,
            target,
            *pytest.progress_args,
            *report_args,
            "-p",
            pytest.enforcement_plugin,
            "-p",
            "no:metadata",
            f"--timeout={pytest.case_timeout_seconds}",
            f"--junitxml={report_dir / 'junit.xml'}",
            *coverage_args,
            *parallel_args,
            *ci_marker_args,
            *optional_args,
        )

    @staticmethod
    def _write_diagnostic_files(
        report_dir: Path, diagnostics: m.Infra.PytestDiagnostics
    ) -> None:
        """Persist the existing typed diagnostic model without another CLI process."""
        outputs: tuple[tuple[str, t.StrSequence, str], ...] = (
            ("failed-tests.txt", diagnostics.failed_cases, "\n\n"),
            ("errors.txt", diagnostics.error_traces, "\n\n"),
            ("warnings.txt", diagnostics.warning_lines, "\n"),
            ("slowest-tests.txt", diagnostics.slow_entries, "\n"),
            ("skipped-tests.txt", diagnostics.skip_cases, "\n"),
        )
        for filename, values, separator in outputs:
            u.Cli.atomic_write_text_file(
                report_dir / filename, separator.join(values) + ("\n" if values else "")
            ).unwrap()

    def _extract_diagnostics(
        self, report_dir: Path
    ) -> p.Result[m.Infra.PytestDiagnostics]:
        """Compose the existing JUnit/log diagnostic owner in-process."""
        extractor = FlextInfraPytestDiagExtractor(
            repository_root=self.root,
            junit=report_dir / "junit.xml",
            log_path=report_dir / "pytest.log",
        )
        return extractor.extract(extractor.junit, extractor.log_path)

    def _execute_cache_maintenance(self) -> p.Result[int]:
        """Run one typed testmon DB maintenance WHAT without invoking pytest."""
        db = self._testmon_db_path()
        if self.what == "cache-clear":
            apply = (
                u.Cli
                .env_read(config.Infra.codegen.make.apply_variable)
                .unwrap()
                .strip()
            )
            if apply != config.Infra.codegen.make.apply_value:
                return r[int].fail(
                    "make test WHAT=cache-clear requires "
                    f"{config.Infra.codegen.make.apply_variable}="
                    f"{config.Infra.codegen.make.apply_value}"
                )
            for path in (db, Path(f"{db}-wal"), Path(f"{db}-shm")):
                if path.exists() or path.is_symlink():
                    path.unlink()
            sys.stderr.write(f"testmon cache cleared under {self.root}\n")
            return r[int].ok(0)
        if self.what == "cache-status":
            digest = FlextInfraTestmonDbInspector.digest_file(db)
            exists = db.is_file() and not db.is_symlink()
            size = db.stat().st_size if exists else 0
            sys.stdout.write(
                f"path={db}\nexists={exists}\nsize={size}\ndigest={digest or 'none'}\n"
            )
            return r[int].ok(0)
        state = FlextInfraTestmonDbInspector(
            repository_root=self.root,
            db_path=db,
            pre_run_digest=FlextInfraTestmonDbInspector.digest_file(db),
            run_succeeded=True,
            mode="test",
        ).execute()
        if state.failure:
            return r[int].fail(state.error or "testmon checkpoint failed")
        value = state.value
        sys.stdout.write(
            f"seed_needed={value.seed_needed}\n"
            f"restored_accepted={value.restored_accepted}\n"
            f"changed={value.changed}\n"
            f"saveable={value.saveable}\n"
            f"reason={value.reason}\n"
        )
        return r[int].ok(0 if value.reason != "testmon db missing or empty" else 1)

    @override
    def execute(self) -> p.Result[int]:
        """Execute pytest, profile it, and preserve reports under one deadline."""
        if self._is_cache_maintenance():
            return self._execute_cache_maintenance()
        pytest = config.Infra.tooling.tools.pytest
        report_dir = self._report_directory()
        pre_run_digest = FlextInfraTestmonDbInspector.digest_file(
            self._testmon_db_path()
        )
        command = self.build_command(report_dir)
        u.Cli.atomic_write_text_file(
            report_dir / "command.txt", f"{shlex.join(command)}\n"
        ).unwrap()
        deadline = m.Cli.ProcessDeadline(
            expires_at_monotonic=(
                self.started_at_monotonic + pytest.run_timeout_seconds
            ),
            termination_grace_seconds=pytest.termination_grace_seconds,
            timeout_exit_code=c.Infra.PROCESS_TIMEOUT_EXIT_CODE,
        )
        # Strip host PYTEST_ADDOPTS/PYTHONPATH, then pin this checkout's src so a
        # borrowed shared editable cannot make pytest execute another tree.
        project_src = str(self.root / c.Infra.DEFAULT_SRC_DIR)
        child_env = u.Cli.process_env(
            remove_keys=c.Infra.PYTEST_INHERITED_ENV_REMOVE_KEYS,
            overrides={c.Infra.ORCHESTRATOR_ENV_PYTHONPATH: project_src},
        )
        run_result = u.Cli.run_to_file(
            command,
            report_dir / "pytest.log",
            cwd=self.root,
            env=child_env,
            live=True,
            deadline=deadline,
        )
        if run_result.failure:
            return r[int].fail(run_result.error or "pytest process execution failed")
        exit_code = run_result.value
        if self.profile:
            profile_result = FlextInfraCProfileReport(
                repository_root=self.root,
                profile=report_dir / "pytest.pstats",
                output=report_dir / "pytest-profile.txt",
                sort=pytest.profile_sort,
                limit=pytest.profile_limit,
            ).execute()
            if profile_result.failure and exit_code == 0:
                return r[int].fail(profile_result.error or "cProfile report failed")
        diagnostics_result = self._extract_diagnostics(report_dir)
        if diagnostics_result.failure:
            return r[int].fail(
                diagnostics_result.error or "pytest diagnostic extraction failed"
            )
        diagnostics = diagnostics_result.value
        self._write_diagnostic_files(report_dir, diagnostics)
        # The child status is not authoritative when its durable diagnostics
        # contain failures. Compute the effective status before persisting the
        # summary so the report and returned result cannot contradict each other.
        if exit_code == 0 and any((
            diagnostics.failed_count,
            diagnostics.error_count,
            diagnostics.warning_count,
        )):
            exit_code = 1
        timed_out = exit_code in {
            c.Infra.PROCESS_TIMEOUT_EXIT_CODE,
            c.Infra.PROCESS_SIGNAL_EXIT_OFFSET + 9,
        }
        timeout_state = "TIMED_OUT" if timed_out else "COMPLETED"
        junit_file = report_dir / "junit.xml"
        coverage_file = report_dir / "coverage.xml"
        junit_value = str(junit_file) if junit_file.is_file() else "not-generated"
        coverage_value = (
            str(coverage_file) if coverage_file.is_file() else "not-generated"
        )
        summary = (
            f"junit={junit_value}\n"
            f"coverage={coverage_value}\n"
            f"failed={diagnostics.failed_count}\n"
            f"errors={diagnostics.error_count}\n"
            f"warnings={diagnostics.warning_count}\n"
            f"skipped={diagnostics.skipped_count}\n"
            f"exit={exit_code}\n"
            f"state={timeout_state}\n"
        )
        u.Cli.atomic_write_text_file(report_dir / "summary.txt", summary).unwrap()
        u.Cli.atomic_write_text_file(
            self.root / self.reports / "latest.txt", f"{report_dir.name}\n"
        ).unwrap()
        pytest_log = report_dir / "pytest.log"
        coverage_enabled = self._cov_enabled()
        if exit_code == 0:
            junit_ok = self._require_junit(junit_file, pytest_log)
            if junit_ok.failure:
                return r[int].fail(junit_ok.error or "junit validation failed")
        if (
            exit_code == 0
            and coverage_enabled
            and self._pytest_log_reports_coverage_failure(pytest_log)
        ):
            # pytest-cov under xdist can print fail-under and still return 0.
            return r[int].fail(
                self._artifact_failure_detail(
                    "coverage fail-under reported while pytest exit was 0", pytest_log
                )
            )
        if (
            exit_code == 0
            and coverage_enabled
            and (not coverage_file.is_file() or coverage_file.stat().st_size == 0)
        ):
            return r[int].fail(
                self._artifact_failure_detail(
                    f"coverage report was not generated or is empty: {coverage_file}",
                    pytest_log,
                )
            )
        if exit_code == 0 and not coverage_enabled:
            timed_out_or_signal = timed_out
            inspector = FlextInfraTestmonDbInspector(
                repository_root=self.root,
                db_path=self._testmon_db_path(),
                pre_run_digest=pre_run_digest,
                run_succeeded=not timed_out_or_signal,
                mode="test",
            )
            state_result = inspector.execute()
            if state_result.failure:
                return r[int].fail(state_result.error or "testmon db inspection failed")
            state = state_result.value
            u.Cli.atomic_write_text_file(
                report_dir / "testmon-cache-state.txt",
                (
                    f"seed_needed={state.seed_needed}\n"
                    f"restored_accepted={state.restored_accepted}\n"
                    f"changed={state.changed}\n"
                    f"saveable={state.saveable}\n"
                    f"reason={state.reason}\n"
                ),
            ).unwrap()
        sys.stderr.write(
            f"Reports: {report_dir} (latest: {self.root / self.reports / 'latest.txt'})\n"
        )
        if timed_out:
            sys.stderr.write(
                "ERROR: pytest invocation reached the configured hard wall "
                f"{pytest.run_timeout_seconds}s (exit={exit_code})\n"
            )
        return r[int].ok(exit_code)

    @staticmethod
    def _pytest_log_reports_coverage_failure(pytest_log: Path) -> bool:
        """Detect pytest-cov fail-under text when the child exit code stayed 0."""
        if not pytest_log.is_file():
            return False
        body = pytest_log.read_text(encoding="utf-8", errors="replace")
        return "Coverage failure:" in body or "not reached" in body

    @staticmethod
    def _artifact_failure_detail(message: str, pytest_log: Path) -> str:
        """Attach a pytest.log tail so CI extract_errors keeps actionable context."""
        if not pytest_log.is_file():
            return message
        lines = pytest_log.read_text(encoding="utf-8", errors="replace").splitlines()
        log_tail = "\n".join(lines[-40:])
        if not log_tail:
            return message
        return f"{message}\n--- pytest.log (tail) ---\n{log_tail}"


__all__: list[str] = ["FlextInfraPytestRunner"]
