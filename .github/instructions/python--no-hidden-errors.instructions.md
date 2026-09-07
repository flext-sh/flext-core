---
applyTo: '**/*.py'
---

# Python failures remain explicit

Never swallow, mask, demote, or invent a successful result for a failed
operation.

- CLI boundaries, validators, and orchestrators do not catch or translate
  workflow exceptions. Raw traceback and chained cause escape.
- Catch only inside cleanup or rollback. Re-raise the original exception and
  attach any secondary cleanup/rollback exception to its causal evidence.
- Stop validation at the first defect; never aggregate independent errors.
- Do not return `None`, an empty collection, or a default value to conceal an
  error.
- Do not add compatibility accessors, aliases, dual behavior, retry, alternate
  providers, error-triggered or competing defaults, manual exit translation,
  partial results, or silent degradation. A typed canonical default resolved
  before failure is normal owner behavior.
- Tests prove the material error, cancellation, timeout, and should-not-trigger
  behavior through the public surface.

See also: `config-settings-ssot.md` (rule file) — configuration authority owner.
