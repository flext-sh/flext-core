#!/usr/bin/env bash
# validate_typings.sh — Enforce TypeAlias rules in typings.py
#
# Non-recursive type aliases MUST use `X: TypeAlias = ...` syntax.
# PEP 695 `type X = ...` creates TypeAliasType which breaks isinstance().
# This script fails (exit 1) if any non-recursive alias uses `type` statement.
#
# Recursive aliases (Serializable, ContainerValue, JsonValue) are ALLOWED
# to use `type X = ...` because they require forward self-references.

set -euo pipefail

TYPINGS_FILE="src/flext_core/typings.py"

if [[ ! -f "$TYPINGS_FILE" ]]; then
	echo "ERROR: $TYPINGS_FILE not found" >&2
	exit 2
fi

# Known recursive aliases that MUST use `type` statement (allowlist)
RECURSIVE_ALLOWLIST="Serializable|ContainerValue|JsonValue|GeneralValueType"

# Known non-recursive aliases that MUST use TypeAlias (blocklist)
# These are the authoritative names from CLAUDE.md and flext-strict-typing skill
NONRECURSIVE_BLOCKED=(
	"Primitives"
	"Scalar"
	"Container"
	"ConfigurationMapping"
	"RegisterableService"
	"FactoryCallable"
	"ResourceCallable"
	"MetadataValue"
	"HandlerCallable"
	"HandlerLike"
	"RegistrablePlugin"
	"ConstantValue"
	"FileContent"
	"SortableObjectType"
	"ConversionMode"
	"TypeHintSpecifier"
	"GenericTypeArgument"
	"MessageTypeSpecifier"
	"IncEx"
	"JsonDict"
	"TYPE_CHECKING"
)

ERRORS=0

for alias in "${NONRECURSIVE_BLOCKED[@]}"; do
	# Check for forbidden `type <Alias> = ...` pattern
	if grep -Pn "^\s+type\s+${alias}\s*=" "$TYPINGS_FILE" >/dev/null 2>&1; then
		LINE=$(grep -Pn "^\s+type\s+${alias}\s*=" "$TYPINGS_FILE")
		echo "VIOLATION: Non-recursive alias '${alias}' uses forbidden 'type' statement:" >&2
		echo "  $LINE" >&2
		echo "  FIX: Change to '${alias}: TypeAlias = ...'" >&2
		echo "" >&2
		ERRORS=$((ERRORS + 1))
	fi
done

# Also check for any `type X =` that is NOT in the allowlist
while IFS= read -r line; do
	# Extract the alias name from `type <Name> =`
	name=$(echo "$line" | grep -oP '(?<=type\s)\w+(?=\s*=)')
	if [[ -n "$name" ]] && ! echo "$name" | grep -qP "^($RECURSIVE_ALLOWLIST)$"; then
        # Check if it's inside the Validation class (Annotated types are OK with `type`)
        lineno=$(echo "$line" | cut -d: -f1)
        # Validation class types use 8+ space indent; FlextTypes-level uses 4 spaces
        indent=$(sed -n "${lineno}p" "$TYPINGS_FILE" | grep -oP '^\s+' | wc -c)
        # 8+ chars of indent = nested class (Validation), skip it
        if [[ $indent -ge 9 ]]; then
            continue
        fi
		echo "WARNING: Unknown 'type' alias '${name}' — verify it is recursive or move to TypeAlias:" >&2
		echo "  $line" >&2
		echo "" >&2
	fi
done < <(grep -Pn '^\s+type\s+\w+\s*=' "$TYPINGS_FILE" 2>/dev/null || true)

if [[ $ERRORS -gt 0 ]]; then
	echo "FAILED: $ERRORS non-recursive alias(es) using forbidden 'type' statement." >&2
	echo "All non-recursive aliases MUST use 'X: TypeAlias = ...' syntax." >&2
	echo "See CLAUDE.md §3 and flext-core/CLAUDE.md Zero Tolerance rules." >&2
	exit 1
fi

echo "OK: All non-recursive aliases use TypeAlias syntax correctly."
exit 0
