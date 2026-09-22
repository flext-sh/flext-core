# FLEXT Core Integration Tests

<!-- TOC START -->

- No sections found

<!-- TOC END -->

**Reviewed**: 2026-02-17 | **Scope**: Canonical rules alignment and link consistency

These tests verify that configuration, container, dispatcher, and context layers behave
consistently – the core promise of the 1.0.0 modernization plan.

```bash
cd flext-core
make test
```

Highlighted scenarios:

- runtime_bootstrap_options
- runtime_bootstrap_options
- `test_system.py` / `test_service.py` – end-to-end dispatcher flows that exercise
  context propagation and logging.

Support scripts (optional, run manually when reviewing exports or wiring):

```bash
python tests/integration/test_wildcard_exports.py --list
python tests/integration/test_integration.py --detail
```

Keep this document updated when new integration scenarios are introduced during the
modernization rollout.
