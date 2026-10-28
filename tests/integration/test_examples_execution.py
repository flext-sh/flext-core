"""Behavioral tests for the public flext-core example scripts.

Golden-backed scripts are self-verifying harnesses: ``python -m examples.<name>``
it exercises the public API of one flext-core primitive, compares every audited
value against a committed golden file, and reports the outcome on stdout. These
tests assert only that observable contract -- process exit status, the announced
``PASS`` marker, the reported check count, the absence of failure/traceback
markers, and the golden-file artifacts -- never any harness internals. Smaller
non-golden examples are checked through their public module execution contract.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
)

import re
import sys
from pathlib import Path

import pytest

from tests import c, m, p, tm, u


class TestsFlextExamplesExecution:
    """Assert the observable execution contract of every public example script."""

    @pytest.mark.parametrize("module_name", c.Tests.NON_GOLDEN_EXAMPLE_MODULES)
    def test_non_golden_example_module_executes(self, module_name: str) -> None:
        """The module executes its class entry without a free-function shim."""
        output: p.Cli.CommandOutput = tm.ok(
            u.Cli.run_raw(
                [sys.executable, "-m", module_name],
                cwd=self._repo_root(),
                options=m.Cli.ProcessOptions(remove_env_keys=("PYTHONPATH",)),
            ),
        )
        tm.that(
            u.Cli.process_succeeded(output.outcome),
            eq=True,
            msg=output.stdout + output.stderr,
        )
        tm.that(output.stderr, lacks="Traceback")
        tm.that(
            re.search(r"\b\w*Warning:", output.stderr), none=True, msg=output.stderr
        )

    @staticmethod
    def _repo_root() -> Path:
        """Return the flext-core repository root that hosts ``examples/``.

        Returns:
            The flext-core repository root that hosts ``examples/``.

        """
        return Path(__file__).resolve().parents[c.Tests.REPO_ROOT_PARENT_DEPTH]

    @pytest.mark.parametrize(
        ("example_name", "module_name", "script_name"),
        c.Tests.PUBLIC_EXAMPLES,
    )
    def test_public_example_scripts_match_golden_files(
        self,
        example_name: str,
        module_name: str,
        script_name: str,
    ) -> None:
        """A public example runs to completion and matches its golden file.

        Observable public contract asserted here:
        - the process exits successfully (return code ``0``);
        - stdout announces ``PASS: <name>`` (golden comparison succeeded);
        - stdout reports a positive audited-check count (the API was exercised);
        - stdout carries no ``FAIL`` marker (no golden mismatch);
        - stderr carries no ``Traceback`` (no unhandled exception escaped);
        - the committed golden ``.expected`` fixture exists;
        - no ``.actual`` diff artifact is left behind (clean pass).
        """
        repo_root = self._repo_root()
        script_path = repo_root / c.Tests.EXAMPLES_DIR / script_name
        actual_path = script_path.with_suffix(".actual")
        expected_path = script_path.with_suffix(".expected")
        output: p.Cli.CommandOutput = tm.ok(
            u.Cli.run_raw(
                [sys.executable, "-m", module_name],
                cwd=repo_root,
                options=m.Cli.ProcessOptions(remove_env_keys=("PYTHONPATH",)),
            ),
        )
        stdout, stderr = output.stdout, output.stderr

        tm.that(u.Cli.process_succeeded(output.outcome), eq=True, msg=stdout + stderr)
        tm.that(stdout, has=f"PASS: {example_name}")
        tm.that(stdout, lacks="FAIL")
        tm.that(stderr, lacks="Traceback")
        tm.that(stderr, lacks="FlextMroViolation")
        tm.that(re.search(r"\b\w*Warning:", stderr), none=True, msg=stderr)

        check_match = tm.not_none(
            re.search(c.Tests.EXAMPLE_CHECK_COUNT_PATTERN, stdout)
        )
        tm.that(int(check_match.group(1)), gt=0)

        tm.ok(expected_path.exists())
        tm.fail(actual_path.exists())
