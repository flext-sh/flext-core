#!/usr/bin/env python3
"""Remove all fake/alias methods from FlextCore safely."""

import re

# Read the file
with open("src/flext_core/core.py", "r") as f:
    lines = f.readlines()

# Track what to keep and what to remove
keep_lines = []
skip_until = -1

i = 0
while i < len(lines):
    # If we're skipping lines (inside a method to remove), continue
    if i <= skip_until:
        i += 1
        continue
    
    line = lines[i]
    
    # Check if this is the start of a fake method
    if "def " in line and i < len(lines) - 1:
        # Look ahead for docstring
        for j in range(i + 1, min(i + 5, len(lines))):
            if '"""' in lines[j] and ("Simple alias" in lines[j] or "Ultra-simple alias" in lines[j]):
                # This is a fake method, find where it ends
                # Find the next def, class, or dedented line
                indent = len(line) - len(line.lstrip())
                
                # Look for the end of this method
                end_found = False
                for k in range(j + 1, len(lines)):
                    next_line = lines[k]
                    next_indent = len(next_line) - len(next_line.lstrip())
                    
                    # Check if we've reached the next method/class or dedented
                    if next_line.strip() and next_indent <= indent:
                        # Check for decorators that might come before the next method
                        if next_line.strip().startswith('@'):
                            # This is a decorator, back up to include it
                            skip_until = k - 1
                        else:
                            skip_until = k - 1
                        end_found = True
                        break
                
                if not end_found:
                    # Method goes to end of file
                    skip_until = len(lines) - 1
                
                print(f"Removing method at line {i+1}: {line.strip()}")
                i = skip_until + 1
                break
        else:
            # Not a fake method, keep this line
            keep_lines.append(line)
            i += 1
    else:
        # Not a method definition, keep this line
        keep_lines.append(line)
        i += 1

# Clean up excessive blank lines
result = []
prev_blank = False
for line in keep_lines:
    if line.strip() == "":
        if not prev_blank:
            result.append(line)
            prev_blank = True
    else:
        result.append(line)
        prev_blank = False

# Write the result
with open("src/flext_core/core.py", "w") as f:
    f.writelines(result)

print(f"\nOriginal: {len(lines)} lines")
print(f"Result: {len(result)} lines")
print(f"Removed: {len(lines) - len(result)} lines")