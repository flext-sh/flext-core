"""Performance benchmarks for class nesting refactor engine."""

from __future__ import annotations

import tempfile
import time
import tracemalloc
from pathlib import Path

import pytest

from flext_infra.refactor.scanner import FlextInfraRefactorLooseClassScanner
from flext_infra.refactor.rules.class_nesting import ClassNestingRefactorRule


class TestPerformanceBenchmarks:
    """Benchmark performance of refactor engine."""

    def test_process_1000_files_in_30_seconds(self) -> None:
        """Benchmark: Process 1000 files in < 30 seconds."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create 1000 Python files with loose classes
            for i in range(1000):
                file_dir = tmp_path / f"pkg{i // 100}" / f"subpkg{i // 10}"
                file_dir.mkdir(parents=True, exist_ok=True)
                test_file = file_dir / f"module_{i}.py"
                test_file.write_text(f'''
class LooseClass{i}:
    """Loose class {i}."""
    pass

def helper_{i}():
    return {i}
''')

            scanner = FlextInfraRefactorLooseClassScanner()

            # Measure time
            start = time.perf_counter()
            result = scanner.scan(tmp_path)
            elapsed = time.perf_counter() - start

            # Verify performance target
            assert elapsed < 30.0, f"Scan took {elapsed:.2f}s, expected < 30s"
            assert result["files_scanned"] >= 1000
            print(f"Scanned {result['files_scanned']} files in {elapsed:.2f}s")

    def test_peak_memory_under_500mb(self) -> None:
        """Benchmark: Peak memory < 500MB for workspace scan."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create substantial codebase
            for i in range(500):
                file_dir = tmp_path / f"project{i // 50}" / "src"
                file_dir.mkdir(parents=True, exist_ok=True)
                test_file = file_dir / f"file_{i}.py"
                # Create files with substantial content
                test_file.write_text(f'''
"""Module {i} with substantial content."""
from __future__ import annotations

from typing import Optional, List, Dict, Any

class ClassA{i}:
    """Class A variant {i}."""
    
    def __init__(self, value: int) -> None:
        self.value = value
    
    def process(self, items: List[str]) -> Dict[str, Any]:
        return {{"items": items, "value": self.value}}

class ClassB{i}:
    """Class B variant {i}."""
    
    @staticmethod
    def helper(x: Optional[int]) -> int:
        return x or 0

def standalone_func_{i}(a: int, b: int) -> int:
    return a + b
''')

            scanner = FlextInfraRefactorLooseClassScanner()

            # Measure memory
            tracemalloc.start()
            result = scanner.scan(tmp_path)
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            peak_mb = peak / 1024 / 1024

            # Verify memory target
            assert peak_mb < 500, f"Peak memory was {peak_mb:.1f}MB, expected < 500MB"
            print(f"Peak memory: {peak_mb:.1f}MB for {result['files_scanned']} files")

    def test_rule_application_performance(self) -> None:
        """Benchmark rule application on single file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create test file
            test_file = tmp_path / "test.py"
            test_file.write_text("""
class TimeoutEnforcer:
    def enforce(self, timeout: int) -> bool:
        return True

class RateLimiter:
    def limit(self, rate: int) -> bool:
        return True
""")

            # Create config
            config_file = tmp_path / "mappings.yml"
            config_file.write_text("""
class_nesting:
  - loose_name: TimeoutEnforcer
    current_file: test.py
    target_namespace: FlextDispatcher
    target_name: TimeoutEnforcer
    confidence: high
  - loose_name: RateLimiter
    current_file: test.py
    target_namespace: FlextDispatcher
    target_name: RateLimiter
    confidence: high
""")

            rule = ClassNestingRefactorRule(config_file)

            # Measure time
            start = time.perf_counter()
            for _ in range(100):  # Apply 100 times
                result = rule.apply(test_file, dry_run=True)
            elapsed = time.perf_counter() - start

            avg_time = elapsed / 100
            print(f"Average rule application: {avg_time * 1000:.2f}ms")

            # Should be fast per file
            assert avg_time < 0.1, f"Rule application too slow: {avg_time * 1000:.2f}ms"
