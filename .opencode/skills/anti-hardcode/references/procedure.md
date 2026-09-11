# Operational-value owner migration

## Preflight

Resolve project law, typed configuration/schema facade, generators, runtime
construction, public entry point, native gates, and every producer, consumer,
projection, fixture, and test for the value. Classify the value from semantics,
not a lexical match. Missing owner or consumer evidence blocks the edit.

For each operational value record its type, validation, canonical derivation or
external owner, affected runtime, and missing/invalid behavior. Require an
environment variable, setting, parameter, or argument only when the current
value cannot be derived. Required credentials and other genuinely external
values enter only through the current process environment and the typed owner.
Keyring, `secret-tool`, profiles, aliases, consumer-level `getenv`, lookup
chains, and alternate providers are prohibited.

## Owner cutover

1. Add a boundary test for a different, missing, empty, malformed, conflicting,
   or wrong-environment value.
2. Add or extend the existing typed owner. Declare each deterministic calculated
   default once and omit equal consumer inputs. Make only non-derivable current
   values required; never add a competing or default-on-error path.
3. Validate the complete configuration before the first consumer or publication
   effect, then inject the validated object into every consumer.
4. Regenerate projections through their owner and require an unchanged second
   generation.
5. Remove literals, duplicate constants, direct environment reads, old loaders,
   fallbacks, aliases, fixtures, examples, tests, and copied documentation.

Invalid configuration raises at the owner and preserves its cause through the
public runtime. Do not catch it into a warning, finding, skip, neutral value, or
manually selected success/exit path. No consumer, alternate source, subprocess,
or partial artifact may run after failed preflight.

## Proof

Run the affected public runtime, required environment matrix, focused tests,
generation fixed point, and native static/security/build gates. Search for the
old literal, key, lookup APIs, and alternate sources. Report exact command, exit
status, decisive output, owner, consumers rewired, zero effects, and zero residue.
