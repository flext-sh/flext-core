# FLEXT-Core Documentation Index

Concise navigation for FLEXT-Core reference materials. All documents follow the repository documentation standards and PEP 8/257 guidance.

## Quick Navigation
- **Onboarding:** [`QUICK_START.md`](./QUICK_START.md)
- **Architecture overview:** [`architecture/overview.md`](./architecture/overview.md)
- **Clean architecture details:** [`architecture/clean-architecture.md`](./architecture/clean-architecture.md)
- **API reference by layer:** [`api-reference/`](./api-reference/) (foundation, domain, application, infrastructure)
- **Guides:** [`guides/`](./guides/) covering railway-oriented programming, DI, DDD, configuration, error handling, testing, and troubleshooting
- **Standards:** [`standards/`](./standards/) for development, documentation, and templates
- **Contributing:** [`development/contributing.md`](./development/contributing.md)

## Repository Structure (docs)
```
docs/
├── INDEX.md              # This file
├── QUICK_START.md        # Five-minute introduction
├── api-reference/        # Layered API reference
├── architecture/         # System and pattern descriptions
├── development/          # Contribution workflow
├── guides/               # How-to guides and patterns
├── improvements/         # Audit and quality reports
└── standards/            # Coding and documentation standards
```

## Notes on Accuracy and Duplication
- Prefer linking to authoritative guides instead of repeating the same content across files.
- Align terminology with the dispatcher-centric CQRS architecture: `FlextDispatcher`, handler registry, middleware, and domain-event publishing.
- Update dates and version references only when `pyproject.toml` changes to avoid drift.
