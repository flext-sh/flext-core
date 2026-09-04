# Evidence-backed review procedure

## Preflight

Load the exact review target, base and head identities, repository law, affected
owners, public contracts, callers, tests, generated surfaces, runtime evidence,
and concurrent changes before issuing a verdict. Validate every command before
execution and preserve its nonzero exit, timeout, or signal as the causal result.

An empty, incomplete, mixed, or inaccessible target blocks review with zero
repository effects and no approval, generic checklist, or alternate range.

## Review order

Inspect only defects supported by changed-line and owner evidence:

1. correctness, boundary values, state transitions, ordering, and concurrency;
2. authentication, authorization, tenant isolation, injection, secret exposure,
   unsafe input/output, dependencies, and data integrity;
3. failure semantics: catches, false success, warning/skip conversion, retries,
   fallback, competing or error-triggered defaults, alternate providers or
   credentials, and partial effects; canonical calculated defaults at one typed
   owner are not defects;
4. public contract, schema, persistence, API, performance, and observability
   regressions;
5. single ownership, speculative scope, duplication, dead code, unrewired
   consumers/tests, and compatibility residue.

Use the canonical `solid`, `ssot`, `yagni`, `dry`, and `simplify` owners for their
respective analysis; do not copy their rules into this review. File size, style,
pattern preference, or hypothetical future use is not a defect.

## Defect contract

For each confirmed defect provide:

- severity derived from concrete exploitability, data loss, incorrect behavior,
  or regression impact;
- exact file and changed-line evidence;
- affected owner, consumer, and runtime path;
- smallest owner-correct remediation and executable proof.

P0 and P1 block landing. A demonstrated lower-impact defect remains required
work in the reviewed change; do not convert it to an optional suggestion or
deferred cleanup. If the review is clean, state exactly what was inspected and
the uncovered boundary without claiming broader safety.

## Output

Lead with `APPROVE` or `REQUEST_CHANGES`, followed by ordered defects and the
bounded evidence. Do not implement, create follow-up work, publish comments, or
change tracker/Git state without separate authorization.
