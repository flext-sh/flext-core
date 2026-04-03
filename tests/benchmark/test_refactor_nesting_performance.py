"""Performance benchmarks for class nesting refactor engine."""

from __future__ import annotations

import tempfile
import time
import tracemalloc
from pathlib import Path

from flext_infra import (
    FlextInfraClassNestingRefactorRule as ClassNestingRefactorRule,
    FlextInfraRefactorLooseClassScanner,
    u,
)


class TestPerformanceBenchmarks:
    """Benchmark performance of refactor engine."""

    def test_process_1000_files_in_30_seconds(self) -> None:
        """Benchmark: Process 1000 files in < 30 seconds."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            for i in range(1000):
                file_dir = tmp_path / f"pkg{i // 100}" / f"subpkg{i // 10}"
                file_dir.mkdir(parents=True, exist_ok=True)
                test_file = file_dir / f"module_{i}.py"
                test_file.write_text(
                    f'\nclass LooseClass{i}:\n    """Loose class {i}."""\n    pass\n\ndef helper_{i}():\n    return {i}\n',
                )
            scanner = FlextInfraRefactorLooseClassScanner()
            start = time.perf_counter()
            _ = scanner.scan(tmp_path)
            elapsed = time.perf_counter() - start
            assert elapsed < 30.0, f"Scan took {elapsed:.2f}s, expected < 30s"

    def test_peak_memory_under_500mb(self) -> None:
        """Benchmark: Peak memory < 500MB for workspace scan."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            for i in range(500):
                file_dir = tmp_path / f"project{i // 50}" / "src"
                file_dir.mkdir(parents=True, exist_ok=True)
                test_file = file_dir / f"file_{i}.py"
                test_file.write_text(
                    f'\n"""Module {i} with substantial content."""\nfrom __future__ import annotations\n\nfrom typing import Optional, List, Dict, Any\n\nclass ClassA{i}:\n    """Class A variant {i}."""\n    \n    def __init__(self, value: int) -> None:\n        self.value = value\n    \n    def process(self, items: List[str]) -> Dict[str, Any]:\n        return {{"items": items, "value": self.value}}\n\nclass ClassB{i}:\n    """Class B variant {i}."""\n    \n    @staticmethod\n    def helper(x: Optional[int]) -> int:\n        return x or 0\n\ndef standalone_func_{i}(a: int, b: int) -> int:\n    return a + b\n',
                )
            scanner = FlextInfraRefactorLooseClassScanner()
            tracemalloc.start()
            _ = scanner.scan(tmp_path)
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak_mb = peak / 1024 / 1024
            assert peak_mb < 500, f"Peak memory was {peak_mb:.1f}MB, expected < 500MB"

    def test_rule_application_performance(self) -> None:
        """Benchmark rule application on single file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            test_file = tmp_path / "test.py"
            test_file.write_text(
                "\nclass TimeoutEnforcer:\n    def enforce(self, timeout: int) -> bool:\n        return True\n\nclass RateLimiter:\n    def limit(self, rate: int) -> bool:\n        return True\n",
            )
            config_file = tmp_path / "mappings.yml"
            config_file.write_text(
                "\nclass_nesting:\n  - loose_name: TimeoutEnforcer\n    current_file: test.py\n    target_namespace: FlextDispatcher\n    target_name: TimeoutEnforcer\n    confidence: high\n  - loose_name: RateLimiter\n    current_file: test.py\n    target_namespace: FlextDispatcher\n    target_name: RateLimiter\n    confidence: high\n",
            )
            rule = ClassNestingRefactorRule(config_file)
            rope_project = u.Infra.init_rope_project(tmp_path)
            resource = u.Infra.get_resource_from_path(rope_project, test_file)
            if resource is None:
                raise FileNotFoundError(test_file)
            start = time.perf_counter()
            try:
                for _ in range(100):
                    _ = rule.apply(rope_project, resource, dry_run=True)
                elapsed = time.perf_counter() - start
                avg_time = elapsed / 100
                assert avg_time < 0.1, (
                    f"Rule application too slow: {avg_time * 1000:.2f}ms"
                )
            finally:
                rope_project.close()
