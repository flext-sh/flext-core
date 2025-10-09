#!/bin/bash
#
# FLEXT Ecosystem Integration Test Script
# Tests all dependent projects against current flext-core version
#
# Copyright (c) 2025 FLEXT Team. All rights reserved.
# SPDX-License-Identifier: MIT

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLEXT_ROOT="$(dirname "$SCRIPT_DIR")"
WORKSPACE_ROOT="$(dirname "$FLEXT_ROOT")"

echo "====================================="
echo "FLEXT ECOSYSTEM INTEGRATION TESTING"
echo "====================================="
echo ""
echo "flext-core location: $FLEXT_ROOT"
echo "Workspace root: $WORKSPACE_ROOT"
echo ""

# Priority levels for testing
# Tier 1: Core infrastructure (CRITICAL)
TIER1_PROJECTS=(
	"flext-cli"
	"flext-ldap"
	"flext-ldif"
	"flext-api"
	"flext-meltano"
)

# Tier 2: Domain libraries (HIGH)
TIER2_PROJECTS=(
	"flext-auth"
	"flext-db-oracle"
	"flext-web"
	"flext-observability"
	"flext-grpc"
)

# Tier 3: Specialized libraries (MEDIUM)
TIER3_PROJECTS=(
	"flext-plugin"
	"flext-quality"
	"flext-oracle-wms"
	"flext-oracle-oic"
)

# Tier 4: Data pipeline (MEDIUM)
TIER4_PROJECTS=(
	"flext-tap-ldap"
	"flext-tap-ldif"
	"flext-tap-oracle"
	"flext-target-ldap"
	"flext-target-ldif"
	"flext-target-oracle"
	"flext-dbt-ldap"
	"flext-dbt-ldif"
	"flext-dbt-oracle"
)

# Test results tracking
declare -A TEST_RESULTS
PASSED_COUNT=0
FAILED_COUNT=0
SKIPPED_COUNT=0

# Test a single project
test_project() {
	local project="$1"
	local project_path="$WORKSPACE_ROOT/$project"

	echo ""
	echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	echo "Testing: $project"
	echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

	# Check if project exists
	if [ ! -d "$project_path" ]; then
		echo "⚠️  SKIPPED: Project directory not found"
		TEST_RESULTS["$project"]="SKIPPED"
		((SKIPPED_COUNT++))
		return
	fi

	# Check if project has tests
	if [ ! -d "$project_path/tests" ]; then
		echo "⚠️  SKIPPED: No tests directory"
		TEST_RESULTS["$project"]="SKIPPED"
		((SKIPPED_COUNT++))
		return
	fi

	# Change to project directory
	cd "$project_path"

	# Check if poetry.lock exists
	if [ ! -f "poetry.lock" ]; then
		echo "📦 Installing dependencies..."
		if ! poetry install --no-root 2>&1 | tail -5; then
			echo "❌ FAILED: Dependency installation failed"
			TEST_RESULTS["$project"]="FAILED (dependencies)"
			((FAILED_COUNT++))
			return
		fi
	fi

	# Run tests
	echo "🧪 Running tests..."
	if poetry run pytest tests/ -v --tb=short --maxfail=3 2>&1 | tail -20; then
		echo "✅ PASSED: All tests successful"
		TEST_RESULTS["$project"]="PASSED"
		((PASSED_COUNT++))
	else
		echo "❌ FAILED: Tests failed"
		TEST_RESULTS["$project"]="FAILED (tests)"
		((FAILED_COUNT++))
	fi
}

# Test a tier of projects
test_tier() {
	local tier_name="$1"
	shift
	local projects=("$@")

	echo ""
	echo "═══════════════════════════════════════════════════════"
	echo "$tier_name"
	echo "═══════════════════════════════════════════════════════"

	for project in "${projects[@]}"; do
		test_project "$project"
	done
}

# Main execution
main() {
	local tier_arg="${1:-all}"

	echo "Test scope: $tier_arg"
	echo ""

	case "$tier_arg" in
	tier1 | critical)
		test_tier "TIER 1: Core Infrastructure (CRITICAL)" "${TIER1_PROJECTS[@]}"
		;;
	tier2 | high)
		test_tier "TIER 2: Domain Libraries (HIGH)" "${TIER2_PROJECTS[@]}"
		;;
	tier3 | medium)
		test_tier "TIER 3: Specialized Libraries (MEDIUM)" "${TIER3_PROJECTS[@]}"
		;;
	tier4 | pipeline)
		test_tier "TIER 4: Data Pipeline (MEDIUM)" "${TIER4_PROJECTS[@]}"
		;;
	all)
		test_tier "TIER 1: Core Infrastructure (CRITICAL)" "${TIER1_PROJECTS[@]}"
		test_tier "TIER 2: Domain Libraries (HIGH)" "${TIER2_PROJECTS[@]}"
		test_tier "TIER 3: Specialized Libraries (MEDIUM)" "${TIER3_PROJECTS[@]}"
		test_tier "TIER 4: Data Pipeline (MEDIUM)" "${TIER4_PROJECTS[@]}"
		;;
	*)
		echo "Usage: $0 [tier1|tier2|tier3|tier4|all]"
		exit 1
		;;
	esac

	# Print summary
	echo ""
	echo "═══════════════════════════════════════════════════════"
	echo "INTEGRATION TEST SUMMARY"
	echo "═══════════════════════════════════════════════════════"
	echo ""
	echo "✅ Passed:  $PASSED_COUNT"
	echo "❌ Failed:  $FAILED_COUNT"
	echo "⚠️  Skipped: $SKIPPED_COUNT"
	echo ""

	# Detailed results
	echo "Detailed Results:"
	echo "─────────────────────────────────────────────────────"
	for project in "${!TEST_RESULTS[@]}"; do
		printf "%-30s %s\n" "$project" "${TEST_RESULTS[$project]}"
	done | sort
	echo ""

	# Exit code based on failures
	if [ "$FAILED_COUNT" -gt 0 ]; then
		echo "⚠️  WARNING: Some integration tests failed"
		exit 1
	else
		echo "✅ SUCCESS: All integration tests passed!"
		exit 0
	fi
}

# Run main
main "$@"
