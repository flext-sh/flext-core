#!/usr/bin/env python3
"""Script to remove all fake/alias methods from FlextCore."""

import re

# Read the file
with open("src/flext_core/core.py", "r") as f:
    content = f.read()

# List of fake methods to remove (identified from grep search)
fake_methods_to_remove = [
    # Static methods with Ultra-simple/Simple alias
    "configure_logging",  # line 245
    "pipe",  # line 276
    "when",  # line 300
    "tap",  # line 315
    "validate_string",  # line 334
    "validate_numeric",  # line 356
    
    # Instance methods with Simple/Ultra-simple alias
    "create_email_address",  # Simple alias
    "validate_field",  # Ultra-simple alias
    "create_config_provider",  # Simple alias
    "configure_core_system",  # Simple alias
    "create_metadata",  # Simple alias
    "create_service_name_value",  # Simple alias
    "create_payload",  # Simple alias
    "create_message",  # Simple alias
    "create_domain_event",  # Simple alias
    "create_factory",  # Simple alias
    "create_standard_validators",  # Simple alias
    "create_validated_model",  # Simple alias
    "configure_decorators_system",  # Simple alias
    "configure_fields_system",  # Simple alias
    "configure_context_system",  # Simple alias
    "create_entity_id",  # Simple alias
    "create_version_number",  # Simple alias
    "health_check",  # Simple alias
    "get_core_system_config",  # Simple alias
    "get_environment_config",  # Simple alias
    "configure_aggregates_system",  # Simple alias
    "get_aggregates_config",  # Simple alias
    "optimize_aggregates_system",  # Simple alias
    
    # More fake methods
    "create_value_object",  # Simple alias
    "create_aggregate",  # Simple alias
    "set_cache_strategy",  # Simple alias
    "cleanup_temp_files",  # Simple alias
    "execute_query",  # Simple alias
    "save_state",  # Simple alias
    "export_config",  # Simple alias
    "import_config",  # Simple alias
    "get_logger",  # Ultra-simple alias (returns logger)
    "context",  # Simple alias
    "get_context_system_config",  # Simple alias
    "validation_engine",  # Simple alias
    "log_info",  # Simple alias
    "log_warning",  # Simple alias
    "log_error",  # Simple alias
    "list_available_methods",  # Simple alias
    "get_all_functionality",  # Simple alias
    "reset_caches",  # Simple alias
    "optimize_core_performance",  # Simple alias
    "optimize_performance",  # Simple alias
    "validation_with_exceptions",  # Simple alias
    "config_manager",  # Simple alias
    "plugin_manager",  # Simple alias
    "load_plugin",  # Simple alias
    "unload_plugin",  # Simple alias
    "list_plugins",  # Simple alias
    "get_plugin_info",  # Simple alias
    
    # Additional fake methods to remove
    "compose",  # Railway pattern alias
]

# Function to remove a method from the content
def remove_method(content, method_name):
    """Remove a method and its entire body from the content."""
    # Pattern to match both static and instance methods
    patterns = [
        # Static method pattern
        rf'(\n\s*)@staticmethod\s*\n\s*def {method_name}\([^)]*\)[^:]*:.*?(?=\n\s*(@|def|class|\Z))',
        # Instance method pattern  
        rf'(\n\s*)def {method_name}\(self[^)]*\)[^:]*:.*?(?=\n\s*(@|def|class|\Z))',
        # Class method pattern
        rf'(\n\s*)@classmethod\s*\n\s*def {method_name}\(cls[^)]*\)[^:]*:.*?(?=\n\s*(@|def|class|\Z))',
    ]
    
    for pattern in patterns:
        # Use DOTALL flag to match across multiple lines
        content = re.sub(pattern, '', content, flags=re.DOTALL)
    
    return content

# Remove all fake methods
original_length = len(content)
for method in fake_methods_to_remove:
    print(f"Removing method: {method}")
    content = remove_method(content, method)

# Clean up extra blank lines
content = re.sub(r'\n\n\n+', '\n\n', content)

# Write the cleaned content back
with open("src/flext_core/core.py", "w") as f:
    f.write(content)

print(f"\nRemoved {original_length - len(content)} characters")
print(f"File size reduced from {original_length} to {len(content)} characters")