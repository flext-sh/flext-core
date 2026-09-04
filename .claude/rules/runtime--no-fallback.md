# One authorized path or failure

A workflow selects exactly one typed owner, provider, model, credential source,
algorithm, destination, and execution path during preflight. Failure of that
path terminates the invocation.

Retries, fallback implementations, alternate providers/models/accounts,
cached-success substitution, undeclared or competing defaults, compatibility aliases,
dual reads/writes, deprecated inputs, best-effort branches, partial execution,
and reduced modes are prohibited. Optional behavior exists only as an explicit
typed absence in the canonical schema; it cannot be inferred from a failure.

A deterministic default resolved and validated by the typed owner before any
failure is normal SSOT behavior, not fallback. Consumers omit equal environment
variables, settings, parameters, and arguments; only overrides remain explicit.

See also: `strict-execution.md` (rule file) — aggregate parent policy.
