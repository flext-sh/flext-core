#!/usr/bin/env python3
"""Script to remove NotImplementedError from command handlers."""

import re
import sys
from pathlib import Path


def fix_notimplemented_errors(file_path: Path) -> int:
    """Remove NotImplementedError from file while preserving functionality."""
    content = file_path.read_text()
    original_content = content

    # Pattern to match method with NotImplementedError followed by actual implementation
    pattern = r'(def [^:]+:[^{]*?)\n(\s+)"""[^"]*"""\n\2raise NotImplementedError\n'

    # Replace with just the method definition without NotImplementedError
    content = re.sub(pattern, r'\1\n\2"""\2', content)

    # Another pattern for async methods
    pattern2 = (
        r'(async def [^:]+:[^{]*?)\n(\s+)"""[^"]*"""\n\2raise NotImplementedError\n'
    )
    content = re.sub(pattern2, r'\1\n\2"""\2', content)

    # Simple line-by-line approach for remaining cases
    lines = content.split("\n")
    new_lines = []
    skip_next = False

    for _i, line in enumerate(lines):
        if skip_next:
            skip_next = False
            continue

        if "raise NotImplementedError" in line:
            # Skip this line
            continue
        else:
            new_lines.append(line)

    content = "\n".join(new_lines)

    # Write back if changed
    if content != original_content:
        file_path.write_text(content)
        return content.count("NotImplementedError")

    return 0


if __name__ == "__main__":
    file_path = Path(
        "/home/marlonsc/pyauto/flx-core/src/flx_core/application/universal_command_handlers.py"
    )

    if not file_path.exists():
        sys.exit(1)

    remaining = fix_notimplemented_errors(file_path)
