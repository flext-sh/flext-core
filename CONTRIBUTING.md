# Contributing to FLEXT-Core


<!-- TOC START -->
- [Quick Workflow](#quick-workflow)
- [Required Local Checks](#required-local-checks)
- [Architecture and Typing Rules](#architecture-and-typing-rules)
- [Pull Request Expectations](#pull-request-expectations)
- [Issue and Feature Requests](#issue-and-feature-requests)
- [Need More Detail?](#need-more-detail)
<!-- TOC END -->

Thanks for contributing.

This file is the fast entrypoint. Canonical contributor guidance lives in:

- `docs/development/contributing.md`
- `README.md`
- `docs/standards/`

## Quick Workflow

1. Create a focused branch from `0.10.0-dev`.
2. Keep changes scoped to one concern per commit.
3. Add or update tests in the same change when behavior changes.
4. Run quality gates before opening a PR.
5. Update docs for API, behavior, or architecture changes.

## Required Local Checks

Run these commands from the repository root:

```bash
make lint
make type-check
make test-fast
```

Before merge, run the full gate:

```bash
make validate
```

## Architecture and Typing Rules

- Follow repository layering and import rules documented in `README.md` and `docs/architecture/`.
- Prefer explicit types and project aliases (`r`, `t`, `p`, `u`) used across `src/flext_core/`.
- Do not add compatibility shims or hidden fallback paths without a documented reason.

## Pull Request Expectations

- Use conventional commit messages (`feat:`, `fix:`, `docs:`, `refactor:`, `chore:`).
- Include a clear problem statement and what changed.
- Include validation evidence (commands run + result).
- Keep PRs small enough for practical review.

## Issue and Feature Requests

- Search existing issues before opening a new one.
- Provide reproducible steps for bugs.
- Explain use-cases and constraints for feature requests.

## Need More Detail?

Use `docs/development/contributing.md` for complete guidance and examples.
