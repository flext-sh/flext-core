# Fail-fast failure-path remediation

## Preflight

Resolve the operation owner, success invariant, prerequisites, first observable
failure, callers, side effects, child processes, publication boundary, runtime,
and native gates. Missing source, owner, or contract blocks mutation; do not infer
a failure path from text alone.

Success is a fresh complete result from the current invocation. Logs, warnings,
findings, skips, empty/neutral values, cached artifacts, partial output, or a
still-running child are not success.

Resolve canonical calculated defaults once at the typed owner during preflight.
Do not externalize them as required environment variables, settings, parameters,
or arguments. Only an alternate selected after missing, invalid, or failed input
is fallback.

## Root-cause cutover

1. Inject the real owner failure and protect its raw exception, traceback, causal
   chain, and zero-effect boundary.
2. Remove catch normalization, retry, error aggregation, alternate provider/model/
   endpoint/database/credential selection, compatibility reads, cached success,
   ignored child status, manual exit mapping, and log-and-continue behavior.
3. Validate all prerequisites before the first effect. A validator stops on its
   first defect; a child nonzero, timeout, signal, or incomplete publication
   propagates unchanged.
4. Produce output at the repository's atomic owner and publish only after producer
   and validator success. Interruption terminates only owned work.
5. Rewire every consumer and remove obsolete flags, tests, fixtures,
   documentation, and old/new paths.

Workflow and CLI boundaries do not catch failures. Only cleanup or rollback may
catch: attach its secondary failure and re-raise the original cause. Cleanup
success never changes the workflow result.

## Proof

Exercise applicable missing/malformed input, dependency absence, child nonzero,
partial/stale output, timeout, signal, cancellation, concurrency, and cleanup
failure. Run the public runtime and affected native gates, then search for removed
sinks and alternatives. Report exact commands, exits, decisive output, first
cause, zero effects, terminated children, atomic publication, and zero residue.
