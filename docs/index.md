# FLEXT-Core Documentation Index

<!-- TOC START -->

- [Quick Navigation](#quick-navigation)
- [Repository Structure (docs)](#repository-structure-docs)
- [Notes on Accuracy and Duplication](#notes-on-accuracy-and-duplication)

<!-- TOC END -->

**Reviewed**: 2026-02-17 | **Scope**: Canonical rules alignment and link consistency

Concise navigation for FLEXT-Core reference materials. All documents follow the repository documentation standards and PEP 8/257 guidance.

## Quick Navigation

- **Onboarding:** `quick-start.md`
- **Architecture overview:** `architecture/overview.md`
- **CQRS architecture:** `architecture/cqrs.md`
- **Clean architecture details:** `architecture/clean-architecture.md`
- **API reference by layer:** `api-reference/` (foundation, domain, application, infrastructure)
- **Guides:** `guides/` covering railway-oriented programming, DI, DDD, configuration, error handling, testing, and troubleshooting
- **Service patterns:** `guides/service-patterns.md`
- **Standards:** `standards/` for development, documentation, and templates
- **Contributing:** `development/contributing.md`

## Repository Structure (docs)

````text
docs/
├── INDEX.md              # This file
├── quick-start.md        # Five-minute introduction
├── api-reference/        # Layered API reference
├── architecture/         # System and pattern descriptions
├── development/          # Contribution workflow
├── guides/               # How-to guides and patterns
├── improvements/         # Audit and quality reports
└── standards/            # Coding and documentation standards
```text

## Notes on Accuracy and Duplication

- Prefer linking to authoritative guides instead of repeating the same content across files.
- Align terminology with the dispatcher-centric CQRS architecture: `FlextDispatcher`, handler registry, middleware, and domain-event publishing.
- Update dates and version references only when `pyproject.toml` changes to avoid drift.
````
