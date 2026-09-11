"""FLEXT CLI error string authorities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from flext_cli import t


class FlextCliConstantsErrors:
    """Flat error-message constants authority."""

    ERR_UNKNOWN_ERROR: Final[str] = "unknown error"
    ERR_ENSURE_DIR_FAILED: Final[str] = "ensure_dir: {error}"
    ERR_ENSURE_DIR_GENERIC_FAILED: Final[str] = "ensure_dir failed"
    ERR_ATOMIC_WRITE_TEXT_FILE_FAILED: Final[str] = "atomic_write_text_file: {error}"
    ERR_TEXT_READ_FAILED: Final[str] = "Text read failed: {error}"
    ERR_TEXT_WRITE_FAILED: Final[str] = "Text write failed: {error}"
    ERR_CSV_WRITE_FAILED: Final[str] = "CSV write failed: {error}"
    ERR_CSV_READ_FAILED: Final[str] = "CSV read failed: {error}"
    ERR_BINARY_READ_FAILED: Final[str] = "Binary read failed: {error}"
    ERR_BINARY_WRITE_FAILED: Final[str] = "Binary write failed: {error}"
    ERR_FILE_COPY_FAILED: Final[str] = "File copy failed: {error}"
    ERR_CREATE_PARENT_DIR_FAILED: Final[str] = (
        "failed to create parent dir for {target_path}"
    )
    ERR_ENSURE_SYMLINK_FAILED: Final[str] = (
        "failed to ensure symlink for {target_path}: {error}"
    )
    ERR_FILE_PATH_EMPTY: Final[str] = "File path must be non-empty"
    ERR_AUTO_LOAD_FAILED: Final[str] = "Auto load failed"
    ERR_FILE_DELETION_FAILED: Final[str] = "File deletion failed: {error}"
    ERR_JSON_LOAD_FAILED: Final[str] = "JSON load failed: {error}"

    ERR_INVALID_CREDENTIALS: Final[str] = (
        "Invalid credentials: missing token or username/password"
    )
    ERR_SETTINGS_VALIDATION_FAILED: Final[str] = "Settings validation failed: {error}"
    ERR_AUTH_SAVE_FAILED: Final[str] = "Failed to save token: {error}"
    ERR_AUTH_LOAD_FAILED: Final[str] = "Failed to load token: {error}"
    ERR_AUTH_FILE_NOT_FOUND: Final[str] = "Token file does not exist"
    ERR_INVALID_OUTPUT_FORMAT: Final[str] = "Invalid output format: {format}"
    ERR_SETTINGS_INFO_FAILED: Final[str] = "Settings info failed: {error}"

    CLI_PARAM_ERR_TRACE_REQUIRES_DEBUG: Final[str] = (
        "Trace mode requires debug mode to be enabled"
    )
    CLI_PARAM_ERR_FIELD_NOT_FOUND_FMT: Final[str] = (
        "Field '{field_name}' not found in CLI parameter registry"
    )
    CLI_PARAM_ERR_APPLY_FAILED_FMT: Final[str] = (
        "Failed to apply CLI parameters: {error}"
    )
    CLI_PARAM_ERR_INVALID_WITH_VALID_FMT: Final[str] = (
        "invalid {field_label}: {field_value}. valid: {valid_values}"
    )
    CLI_PARAM_ERR_INVALID_WITH_OPTIONS_FMT: Final[str] = (
        "invalid {field_label}: {field_value}. valid options: {valid_values}"
    )

    ERR_SHOW_SETTINGS_FAILED: Final[str] = "Show settings failed: {error}"

    VALIDATION_MSG_FIELD_CANNOT_BE_EMPTY: Final[str] = "{field_name} cannot be empty"

    ERR_INVALID_CONFIRM_INPUT: Final[str] = (
        "Invalid confirmation input - please enter yes or no"
    )
    ERR_USER_CANCELLED_CONFIRMATION: Final[str] = "User cancelled confirmation"
    ERR_INPUT_STREAM_ENDED: Final[str] = "Input stream ended"
    ERR_CONFIRMATION_FAILED_FMT: Final[str] = "Confirmation failed: {error}"
    ERR_NO_CHOICES: Final[str] = "No choices provided"
    ERR_INTERACTIVE_CHOICE_DISABLED: Final[str] = (
        "Interactive mode disabled for choice prompt"
    )
    ERR_CHOICE_REQUIRED_FMT: Final[str] = "Choice required. Options: {choices}"
    ERR_INVALID_CHOICE_FMT: Final[str] = "Invalid choice: {choice}"
    ERR_INTERACTIVE_PASSWORD_DISABLED: Final[str] = (
        "Interactive mode disabled for password prompt"
    )
    ERR_PASSWORD_TOO_SHORT_FMT: Final[str] = (
        "Password too short: minimum {min_length} characters"
    )
    ERR_PROMPT_FAILED_FMT: Final[str] = "Prompt failed: {error}"
    ERR_CHOICE_PROMPT_FAILED_FMT: Final[str] = "Choice prompt failed: {error}"
    ERR_PASSWORD_PROMPT_FAILED_FMT: Final[str] = "Password prompt failed: {error}"

    ERR_INVALID_COMMAND_NAME: Final[str] = "Invalid command name"
    ERR_COMMAND_FAILED: Final[str] = "Command failed"
    ERR_COMMAND_NOT_FOUND: Final[str] = "Command not found: {name}"
    ERR_HANDLER_NOT_CALLABLE: Final[str] = "Handler not callable for: {name}"
    ERR_COMMAND_EXECUTION_FAILED: Final[str] = "Command execution failed: {error}"
    ERR_COMMAND_NAME_EMPTY: Final[str] = "Command name must be non-empty string"
    ERR_CLI_DEFINITION_INVALID_MODEL: Final[str] = (
        "command '{command}' model '{model}': {reason}"
    )
    ERR_CLI_DEFINITION_FIELD: Final[str] = (
        "command '{command}' model '{model}' field '{field}': {reason}"
    )


__all__: t.MutableSequenceOf[str] = ["FlextCliConstantsErrors"]
