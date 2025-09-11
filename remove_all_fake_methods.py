#!/usr/bin/env python3
"""Script to remove ALL fake/alias methods from FlextCore."""

import re

# Read the file
with open("src/flext_core/core.py", "r") as f:
    lines = f.readlines()

# Find all methods with Simple/Ultra-simple alias comments
fake_methods = set()
for i, line in enumerate(lines):
    if "Simple alias" in line or "Ultra-simple alias" in line:
        # Look backwards for the method definition
        for j in range(max(0, i-10), i):
            if "def " in lines[j]:
                method = re.search(r'def (\w+)', lines[j])
                if method:
                    fake_methods.add(method.group(1))
                    break

print(f"Found {len(fake_methods)} fake methods to remove:")
for method in sorted(fake_methods):
    print(f"  - {method}")

# Read content as string for removal
with open("src/flext_core/core.py", "r") as f:
    content = f.read()

# Function to remove a method from the content
def remove_method(content, method_name):
    """Remove a method and its entire body from the content."""
    # Pattern to match property decorators and methods
    patterns = [
        # Property with getter
        rf'(\n\s*)@property\s*\n\s*def {method_name}\([^)]*\)[^:]*:.*?(?=\n\s*(@|def|class|\Z))',
        # Property setter
        rf'(\n\s*)@{method_name}\.setter\s*\n\s*def {method_name}\([^)]*\)[^:]*:.*?(?=\n\s*(@|def|class|\Z))',
        # Static method pattern
        rf'(\n\s*)@staticmethod\s*\n\s*def {method_name}\([^)]*\)[^:]*:.*?(?=\n\s*(@|def|class|\Z))',
        # Instance method pattern  
        rf'(\n\s*)def {method_name}\([^)]*\)[^:]*:.*?(?=\n\s*(@|def|class|\Z))',
        # Class method pattern
        rf'(\n\s*)@classmethod\s*\n\s*def {method_name}\([^)]*\)[^:]*:.*?(?=\n\s*(@|def|class|\Z))',
    ]
    
    for pattern in patterns:
        # Use DOTALL flag to match across multiple lines
        old_len = len(content)
        content = re.sub(pattern, '', content, flags=re.DOTALL)
        if len(content) < old_len:
            print(f"  Removed {method_name}")
            break
    
    return content

# Remove all fake methods
original_length = len(content)
print("\nRemoving methods...")
for method in sorted(fake_methods):
    content = remove_method(content, method)

# Clean up extra blank lines
content = re.sub(r'\n\n\n+', '\n\n', content)

# Remove empty sections headers that might be left
content = re.sub(r'# =+\n\s*# [^\n]+\n\s*# =+\n\s*(?=\n\s*(#|class|\Z))', '', content)

# Write the cleaned content back
with open("src/flext_core/core.py", "w") as f:
    f.write(content)

print(f"\nRemoved {original_length - len(content)} characters")
print(f"File size reduced from {original_length} to {len(content)} characters")
print(f"Total of {len(fake_methods)} fake methods removed")