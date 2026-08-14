.PHONY: setup-org-libs
.PHONY: ci-pytest

# Optional automation: keep local FLEXT ecosystem repos aligned to current branch.
# Can be disabled with: make setup FLEXT_AUTO_ORG_LIBS=0
FLEXT_AUTO_ORG_LIBS ?= 1

setup-org-libs: venv
	@if [ "$(FLEXT_AUTO_ORG_LIBS)" != "1" ]; then \
		echo "==> Skipping org libs sync (FLEXT_AUTO_ORG_LIBS=$(FLEXT_AUTO_ORG_LIBS))"; \
		exit 0; \
	fi
	@chmod +x scripts/setup_org_libs.sh
	@./scripts/setup_org_libs.sh

# Extend generated setup target without editing generated Makefile directly.
setup: setup-org-libs

ci-pytest: setup-org-libs
	@chmod +x scripts/ci/run_pytest_ci.sh
	@./scripts/ci/run_pytest_ci.sh
