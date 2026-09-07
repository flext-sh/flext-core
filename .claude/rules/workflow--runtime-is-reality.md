# Reality is the running system; tests are checks, not the SSOT

Validate against the real declared runtime (CLI, daemon, config, or public API)
and exercise the actual feature—type and lint green are necessary, not
sufficient.

- A test that only passes by keeping removed or legacy artifacts is wrong: fix or
  delete the test; never restore legacy just to make it pass.
- A config/settings test that breaks when a valid SSOT value changes is defective.
  Test contracts and derivations across arbitrary valid inputs; goldens may lock
  generated structure, never mutable config-owned values.
- A missing facade constant fails only at runtime — import and run the real path.
- Before concluding root cause, prove the running or installed artifact matches
  the declared authoritative revision or release. An editable checkout, local
  cache, generated copy, or stale environment is not evidence of remote/runtime
  behavior until identity is verified.
- Use the newest released version of every required tool. Every diagnostic it
  emits is blocking. A cap, downgrade, substitution, suppression, compatibility
  classification, or false-positive classification requires prior operator
  discussion, reproducible evidence, and explicit authorization; without all
  three, correct the owner and rerun that released version.
