"""Scanning utilities for batch file analysis.

All methods are static — exposed via u.Infra.scan_files_batch() through MRO.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from flext_infra.models import FlextInfraModels as m
from flext_infra.protocols import FlextInfraProtocols as p


class FlextInfraUtilitiesScanning:
    """Static scanning utilities for batch file analysis.

    All methods are ``@staticmethod`` — no instantiation required.
    Exposed via ``u.Infra.scan_files_batch()`` through MRO.
    """

    @staticmethod
    def scan_files_batch(
        scanner: p.Infra.Scanner,
        files: Sequence[Path],
    ) -> list[m.Infra.Utilities.ScanResult]:
        """Run a scanner across multiple files and collect results.

        Args:
            scanner object implementing p.Infra.Scanner.
            files: Sequence of file paths to scan.

        Returns:
            List of ScanResult from each file.

        """
        results: list[m.Infra.Utilities.ScanResult] = []
        for file_path in files:
            result = scanner.scan_file(file_path=file_path)
            results.append(result)
        return results


__all__ = ["FlextInfraUtilitiesScanning"]
