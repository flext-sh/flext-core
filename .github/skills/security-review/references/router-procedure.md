# Security Review procedure

# Security Review

Activate for credentials, authentication, authorization, untrusted input,
dependencies, containers, persistence, network boundaries, or scanner output.

## Procedure

1. Resolve project governance, security-triage owner, exact target, current
   scanner configuration, non-derivable external process inputs, runtime path,
   native security gate, and external-token applicability before any edit or
   scan side effect. Owner-calculated defaults require no environment variable,
   setting, parameter, or argument. When a required token is absent before
   selection, do not invoke the dormant workflow; record it as `NOT EXECUTED`
   without a green claim, and keep independently observed findings blocking.
   After direct invocation selects the workflow, a missing or invalid required
   credential raises immediately. Credentials come only from the current
   process environment. Keyring, `secret-tool`, profiles, aliases, and
   credential/provider alternatives are prohibited.
2. Reproduce the named defect through the declared owner. Preserve the first
   scanner, child-process, timeout, signal, or runtime failure and its causal
   chain; never convert it to a finding summary, warning, skip, retry, or success.
3. Trace input, trust boundary, operation, dependency, and affected consumer to
   the canonical configuration or primitive. Use the newest released scanner and
   treat every diagnostic as blocking. A false-positive or compatibility
   classification requires prior operator discussion, reproducible proof that
   the reported data flow cannot occur, and explicit authorization; risk
   acceptance is not a fix.
4. Correct the owner, regenerate owned projections, rewire all consumers, and
   remove the vulnerable or obsolete path atomically. Do not add escaping
   wrappers, hardcoded values, fallback providers, compatibility paths, or
   partial publication.
5. Rerun the reproducer, public runtime, and declared security gates. Record exact
   command, exit status, decisive output, corrected owner, affected consumers,
   and residue search. Completion requires material runtime proof, zero effects
   on failure, and no suppression or superseded path.

Never cap, downgrade, substitute, or suppress a released scanner to avoid its
findings.

Review one ordered defect at a time. Correct and revalidate the first unclosed
defect before advancing; reporting it never closes or transfers the review. Do
not claim the remaining surface clean without executing its declared gates.
