#!/usr/bin/env python3
"""FLEXT Auto-Workflow Script.

Automates common development workflows based on file changes and patterns.
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path


class AutoWorkflow:
    """Main auto-workflow manager."""

    def __init__(self, project_root: Path) -> None:
        """Initialize AutoWorkflow with project root directory.

        Args:
            project_root: Root directory of the project.

        """
        self.project_root = project_root
        self.hooks_dir = Path(tempfile.gettempdir())

    def run_command(
        self,
        cmd: list[str],
        cwd: Path | None = None,
        timeout: int = 60,
        env: dict[str, str] | None = None,
    ) -> bool:
        """Run a command and return success status."""
        try:
            # Merge environment variables
            run_env = {**os.environ}
            if env:
                run_env.update(env)

            result = subprocess.run(
                cmd,
                check=False,
                cwd=cwd or self.project_root,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=run_env,
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            print(f"⚠️ Command timed out: {' '.join(cmd)}")
            return False
        except Exception as e:
            print(f"⚠️ Command failed: {e}")
            return False

    def validate_code(self) -> bool:
        """Run code validation."""
        print("🔍 Running code validation...")
        success = self.run_command(
            ["make", "lint"], cwd=self.project_root / "flext-core"
        )
        if success:
            print("✅ Validation passed")
        else:
            print("❌ Validation failed")
        return success

    def auto_fix(self) -> bool:
        """Run automatic fixes."""
        print("🔧 Running auto-fixes...")
        fixes_applied = False

        # Run lint auto-fix
        if self.run_command(["make", "fix"], cwd=self.project_root / "flext-core"):
            print("✅ Lint fixes applied")
            fixes_applied = True

        # Run import fixes
        if self.run_command(
            ["python3", str(self.hooks_dir / "auto_import_hook.py"), "--check"],
            cwd=self.project_root / "flext-core",
        ):
            print("✅ Import fixes applied")
            fixes_applied = True

        if not fixes_applied:
            print("ℹ️ No fixes needed")

        return True

    def run_tests(self, test_type: str = "unit") -> bool:
        """Run tests."""
        print(f"🧪 Running {test_type} tests...")

        if test_type == "unit":
            success = self.run_command(
                ["poetry", "run", "pytest", "tests/unit/", "-q"],
                cwd=self.project_root / "flext-core",
                env={"PYTHONPATH": "src"},
            )
        elif test_type == "integration":
            success = self.run_command(
                ["poetry", "run", "pytest", "tests/integration/", "-q"],
                cwd=self.project_root / "flext-core",
                env={"PYTHONPATH": "src"},
            )
        elif test_type == "all":
            success = self.run_command(
                ["make", "test"], cwd=self.project_root / "flext-core"
            )
        else:
            print(f"❌ Unknown test type: {test_type}")
            return False
        if success:
            print(f"✅ {test_type.title()} tests passed")
        else:
            print(f"❌ {test_type.title()} tests failed")
        return success

    def check_dependencies(self) -> bool:
        """Check and install missing dependencies."""
        print("📦 Checking dependencies...")
        # This would check pyproject.toml vs installed packages
        # For now, just run poetry install
        success = self.run_command(
            ["poetry", "install"], cwd=self.project_root / "flext-core"
        )
        if success:
            print("✅ Dependencies updated")
        else:
            print("❌ Dependency update failed")
        return success

    def workflow_validate(self) -> None:
        """Full validation workflow."""
        print("🚀 Starting FLEXT validation workflow...")

        if not self.validate_code():
            print("❌ Validation failed, attempting auto-fix...")
            if self.auto_fix():
                print("🔄 Re-validating after fixes...")
                if not self.validate_code():
                    print("❌ Validation still failing after auto-fix")
                    sys.exit(1)
            else:
                sys.exit(1)

        if not self.run_tests("unit"):
            print("❌ Unit tests failed")
            sys.exit(1)

        print("🎉 All validations passed!")

    def workflow_commit(self) -> None:
        """Commit workflow with validation."""
        print("💾 Starting FLEXT commit workflow...")

        # Stage changes
        if not self.run_command(["git", "add", "."]):
            print("❌ Failed to stage changes")
            sys.exit(1)

        # Validate before commit
        self.workflow_validate()

        # Commit
        commit_msg = input("Enter commit message: ").strip()
        if not commit_msg:
            commit_msg = "Auto-commit: validation passed"

        if self.run_command(["git", "commit", "-m", commit_msg]):
            print("✅ Commit successful")
        else:
            print("❌ Commit failed")
            sys.exit(1)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="FLEXT Auto-Workflow")
    parser.add_argument(
        "action",
        choices=[
            "validate",
            "fix",
            "test",
            "deps",
            "workflow-validate",
            "workflow-commit",
        ],
    )
    parser.add_argument(
        "--test-type",
        choices=["unit", "integration", "all"],
        default="unit",
        help="Test type for test action",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).parent.parent.parent,
        help="FLEXT project root directory",
    )

    args = parser.parse_args()

    workflow = AutoWorkflow(args.project_root)

    if args.action == "validate":
        success = workflow.validate_code()
    elif args.action == "fix":
        success = workflow.auto_fix()
    elif args.action == "test":
        success = workflow.run_tests(args.test_type)
    elif args.action == "deps":
        success = workflow.check_dependencies()
    elif args.action == "workflow-validate":
        workflow.workflow_validate()
        success = True
    elif args.action == "workflow-commit":
        workflow.workflow_commit()
        success = True

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
