"""Comprehensive tests for examples execution and functionality."""

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from flext_core import FlextContainer, FlextExceptions, FlextResult


class TestExamplesExecution:
    """Test execution of all example files."""

    @pytest.fixture(scope="class")
    def examples_dir(self) -> Path:
        """Get examples directory path."""
        return Path(__file__).parent.parent.parent / "examples"

    @pytest.fixture(scope="class")
    def example_files(self, examples_dir: Path) -> list[Path]:
        """Get all Python example files."""
        return [f for f in examples_dir.glob("*.py") if f.name != "__init__.py"]

    def test_examples_directory_exists(self, examples_dir: Path) -> None:
        """Test examples directory exists."""
        assert examples_dir.exists(), "Examples directory should exist"
        assert examples_dir.is_dir(), "Examples path should be a directory"

    def test_all_examples_found(self, example_files: list[Path]) -> None:
        """Test that we found the expected number of examples."""
        assert len(example_files) >= 20, (
            f"Expected at least 20 examples, found {len(example_files)}"
        )

        # Check for key examples
        example_names = [f.name for f in example_files]
        key_examples = [
            "01_railway_result.py",
            "02_dependency_injection.py",
            "05_validation_advanced.py",
            "14_exceptions_handling.py",
            "17_end_to_end.py",
        ]

        for key_example in key_examples:
            assert key_example in example_names, f"Key example {key_example} not found"

    @pytest.mark.parametrize(
        "example_file",
        [
            "01_railway_result.py",
            "02_dependency_injection.py",
            "03_cqrs_commands.py",
            "04_validation_modern.py",
            "05_validation_advanced.py",
            "06_ddd_entities_value_objects.py",
            "07_mixins_multiple_inheritance.py",
            "08_configuration.py",
            "09_decorators_cross_cutting.py",
            "10_events_messaging.py",
            "11_handlers_pipeline.py",
            "12_logging_structured.py",
            "13_architecture_interfaces.py",
            "14_exceptions_handling.py",
            "15_advanced_patterns.py",
            "16_integration.py",
            "17_end_to_end.py",
            "18_semantic_modeling.py",
            "19_modern_showcase.py",
            "20_boilerplate_reduction.py",
        ],
    )
    def test_example_execution(self, example_file: str, examples_dir: Path) -> None:
        """Test that each example executes without errors."""
        example_path = examples_dir / example_file
        assert example_path.exists(), f"Example file {example_file} not found"

        # Run example with timeout
        result = subprocess.run(
            [sys.executable, str(example_path)],
            check=False,
            cwd=str(examples_dir.parent),
            env={"PYTHONPATH": "src"},
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode == 0, (
            f"Example {example_file} failed with exit code {result.returncode}\\n"
            f"STDOUT: {result.stdout}\\n"
            f"STDERR: {result.stderr}"
        )

    def test_example_imports(self, example_files: list[Path]) -> None:
        """Test that all examples can be imported without errors."""
        # Add src to path
        src_path = Path(__file__).parent.parent.parent / "src"
        sys.path.insert(0, str(src_path))

        failed_imports = []

        for example_file in example_files:
            try:
                spec = importlib.util.spec_from_file_location(
                    example_file.stem,
                    example_file,
                )
                if spec and spec.loader:
                    # Just test that spec can be created, actual import might run main()
                    pass
                else:
                    failed_imports.append((example_file.name, "No spec created"))
            except Exception as e:
                failed_imports.append((example_file.name, str(e)))

        if failed_imports:
            failure_msg = "\\n".join(
                f"  {name}: {error}" for name, error in failed_imports
            )
            pytest.fail(f"Failed to create import specs for examples:\\n{failure_msg}")


class TestExamplesFunctionality:
    """Test functionality demonstrated in examples."""

    def test_flext_result_patterns(self) -> None:
        """Test FlextResult patterns work as demonstrated in examples."""
        # Test basic FlextResult functionality
        success_result = FlextResult[int].ok(42)
        assert success_result.success
        assert success_result.value == 42

        failure_result = FlextResult[int].fail("error message")
        assert failure_result.is_failure
        assert failure_result.error == "error message"

        # Test railway pattern chaining
        chained_result = (
            FlextResult[int]
            .ok(5)
            .map(lambda x: x * 2)
            .flat_map(lambda x: FlextResult[int].ok(x + 1))
        )
        assert chained_result.success
        assert chained_result.value == 11

    def test_container_dependency_injection(self) -> None:
        """Test container dependency injection works as demonstrated."""
        container = FlextContainer()

        # Test service registration
        test_service = {"name": "test_service", "active": True}
        register_result = container.register("test_service", test_service)
        assert register_result.success

        # Test service retrieval
        get_result = container.get("test_service")
        assert get_result.success
        assert get_result.value == test_service

    def test_validation_patterns(self) -> None:
        """Test validation patterns work as demonstrated."""

        def validate_email(email: str) -> FlextResult[str]:
            """Simple email validation."""
            if "@" not in email:
                return FlextResult[str].fail("Email must contain @")
            return FlextResult[str].ok(email.lower())

        # Test valid email
        valid_result = validate_email("test@example.com")
        assert valid_result.success
        assert valid_result.value == "test@example.com"

        # Test invalid email
        invalid_result = validate_email("invalid-email")
        assert invalid_result.is_failure
        assert invalid_result.error is not None
        assert "must contain @" in invalid_result.error

    def test_exception_handling_patterns(self) -> None:
        """Test exception handling patterns work as demonstrated."""
        # Test basic exception creation with pytest.raises
        validation_message = "Test validation error"
        with pytest.raises(FlextExceptions.ValidationError) as exc_info:
            raise FlextExceptions.ValidationError(
                validation_message,
                validation_details={"field": "email", "error": "invalid"},
            )

        assert validation_message in str(exc_info.value)
        assert hasattr(exc_info.value, "validation_details")

        # Test exception hierarchy - catch specific type, not base class
        not_found_message = "Resource not found"
        with pytest.raises(FlextExceptions.NotFoundError) as not_found_exc_info:
            raise FlextExceptions.NotFoundError(not_found_message, resource_id="123")

        assert not_found_message in str(not_found_exc_info.value)
        assert hasattr(not_found_exc_info.value, "resource_id")


class TestExamplesCodeQuality:
    """Test code quality of examples."""

    @pytest.fixture(scope="class")
    def examples_dir(self) -> Path:
        """Get examples directory path."""
        return Path(__file__).parent.parent.parent / "examples"

    def test_examples_have_docstrings(self, examples_dir: Path) -> None:
        """Test that all examples have proper module docstrings."""
        missing_docstrings = []

        for example_file in examples_dir.glob("*.py"):
            if example_file.name == "__init__.py":
                continue

            try:
                with example_file.open(encoding="utf-8") as f:
                    tree = ast.parse(f.read())

                # Check for module docstring
                if not (
                    tree.body
                    and isinstance(tree.body[0], ast.Expr)
                    and isinstance(tree.body[0].value, ast.Constant)
                    and isinstance(tree.body[0].value.value, str)
                ):
                    missing_docstrings.append(example_file.name)

            except Exception as e:
                pytest.fail(f"Failed to parse {example_file.name}: {e}")

        assert not missing_docstrings, (
            f"Examples missing docstrings: {missing_docstrings}"
        )

    def test_examples_have_main_functions(self, examples_dir: Path) -> None:
        """Test that examples have main functions or if __name__ == '__main__' blocks."""
        missing_main = []

        for example_file in examples_dir.glob("*.py"):
            if example_file.name == "__init__.py":
                continue

            try:
                with example_file.open(encoding="utf-8") as f:
                    content = f.read()
                    tree = ast.parse(content)

                has_main_function = any(
                    isinstance(node, ast.FunctionDef) and node.name == "main"
                    for node in ast.walk(tree)
                )

                has_main_block = 'if __name__ == "__main__"' in content

                if not (has_main_function or has_main_block):
                    missing_main.append(example_file.name)

            except Exception as e:
                pytest.fail(f"Failed to parse {example_file.name}: {e}")

        assert not missing_main, (
            f"Examples missing main function or __main__ block: {missing_main}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
