#!/usr/bin/env python3
"""Clean FlextCore by removing all fake/alias methods while preserving essential ones."""

import re

# Read the file
with open("src/flext_core/core.py", "r") as f:
    content = f.read()

# Essential methods to keep (NOT fake)
essential_methods = {
    "__init__",  # Constructor
    "__repr__",  # String representation
    "__str__",  # String conversion
    "__getattr__",  # Attribute fallback
    "__enter__",  # Context manager enter
    "__exit__",  # Context manager exit
    "get_instance",  # Singleton pattern
    "reset_instance",  # Reset singleton
    "is_valid_config_dict",  # Type guard
    "is_callable_validator",  # Type guard
    "compose",  # Railway pattern (keep if it's using real FlextResult)
}

# Properties to keep (they provide direct access to classes)
essential_properties = {
    "config", "models", "commands", "handlers", "validations", 
    "utilities", "adapters", "services", "decorators", "processors",
    "guards", "fields", "mixins", "protocols", "exceptions",
    "delegation", "result", "container", "context", "logger",
    "constants", "_config"
}

# Find all methods marked as fake/alias
fake_methods = []
lines = content.split('\n')
for i, line in enumerate(lines):
    if "Simple alias" in line or "Ultra-simple alias" in line:
        # Look backwards for the method definition
        for j in range(max(0, i-10), i):
            if "def " in lines[j]:
                method = re.search(r'def (\w+)', lines[j])
                if method:
                    method_name = method.group(1)
                    # Skip essential methods and properties
                    if method_name not in essential_methods and method_name not in essential_properties:
                        fake_methods.append(method_name)
                    break

print(f"Found {len(fake_methods)} fake methods to remove (excluding essentials):")
for method in sorted(set(fake_methods)):
    print(f"  - {method}")

# Now remove only the fake methods
def remove_fake_method(content, method_name):
    """Remove a fake method while preserving structure."""
    # Build a more precise pattern to avoid corruption
    patterns = []
    
    # For property methods
    if method_name in ["logger", "security_config", "database_config", "logging_config", "_context"]:
        # These might be properties, handle specially
        patterns.append(
            rf'(\n    @property\s*\n    def {method_name}\(self\)[^:]*:.*?".*?[Ss]imple alias.*?".*?(?=\n    (@property|def |class |\Z))'
        )
    
    # For static methods
    patterns.append(
        rf'(\n    @staticmethod\s*\n    def {method_name}\([^)]*\)[^:]*:.*?".*?[Ss]imple alias.*?".*?(?=\n    (@|def |class |\Z))'
    )
    
    # For instance methods
    patterns.append(
        rf'(\n    def {method_name}\(self[^)]*\)[^:]*:.*?".*?[Ss]imple alias.*?".*?(?=\n    (@|def |class |\Z))'
    )
    
    # For class methods
    patterns.append(
        rf'(\n    @classmethod\s*\n    def {method_name}\(cls[^)]*\)[^:]*:.*?".*?[Ss]imple alias.*?".*?(?=\n    (@|def |class |\Z))'
    )
    
    for pattern in patterns:
        old_len = len(content)
        content = re.sub(pattern, '\n', content, flags=re.DOTALL)
        if len(content) < old_len:
            return content
    
    return content

# Remove fake methods
original_length = len(content)
for method in sorted(set(fake_methods)):
    content = remove_fake_method(content, method)

# Clean up excessive blank lines
content = re.sub(r'\n\n\n+', '\n\n', content)

# Don't remove section headers - they help with organization

# Write the cleaned content
with open("src/flext_core/core_cleaned.py", "w") as f:
    f.write(content)

print(f"\nOriginal size: {original_length} characters")
print(f"Cleaned size: {len(content)} characters")
print(f"Removed: {original_length - len(content)} characters")
print(f"Removed {len(set(fake_methods))} fake methods")
print("\nCleaned file saved as: src/flext_core/core_cleaned.py")
print("Review the file before replacing the original!")