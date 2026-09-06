---
applyTo: '**'
---

# Child-process failures remain causal

Validate the executable, arguments, environment, working directory, ownership,
timeout contract, and expected publications before spawning. A child nonzero
exit, timeout, signal, spawn failure, malformed output, or incomplete
publication terminates the workflow with the native failure and captured causal
evidence.

Do not use unchecked execution, shell error masking, `|| true`, exit-code
remapping, timeout-to-skip conversion, output heuristics, retries, alternate
commands, or success based on partial output. Cleanup may terminate only the
process group created and owned by this invocation; the original child failure
remains the exception re-raised.

See also: `strict-execution.md` (rule file) — aggregate parent policy.
