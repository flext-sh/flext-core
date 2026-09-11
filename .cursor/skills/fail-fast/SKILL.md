---
name: fail-fast
description: 'silent failure, failover removal, error propagation'
license: MIT
metadata:
  aihub.tags: '["policy:atomic-effects","policy:causal-subprocess","policy:fail-loud","policy:no-fallback","policy:preflight-before-effects","policy:strict-execution","policy:zero-residue","provenance:agents-owned","role:reliability","updates:manual","usage:router"]'
  version: 1.0.0
---

# Fail Fast

Activate when an operation may swallow, normalize, retry, aggregate, suppress, or
reroute an error; ignore a child failure; publish stale/partial success; or leave
owned work running.

Read the `failure-path procedure` (skill file). The first exception is
the result: preserve its raw traceback and causal chain, stop immediately, and
remove every false-success or alternate path at the owner.

Do not activate from a keyword alone. A UI loading placeholder, typed optional
business result, or ordinary non-error routing is not failover when its contract
is explicit and tested. A deterministic calculated default resolved by the typed
owner before any failure is likewise normal behavior, not fallback.
