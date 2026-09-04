# Workspace toolchain ownership procedure

## Resolve the layer

1. Establish the project contract, requested binary or dependency, current
   consumers, operating scopes, and runtime evidence.
2. Classify exactly one layer:
   - **Host/runtime** — a CLI used across the host: the host owner's typed tool
     inventory owns the release, checksum, installation surface, and global
     selector.
   - **Python runtime entrypoint** — a released wheel exposes the command; the
     host owner pins the tag, never a checkout or editable directory.
   - **Project fleet tool** — the fleet generator's toolchain configuration
     owns the pin and projects it into the workspace.
   - **Project-specific tool** — the project's managed-artifact configuration
     owns an additional tool only when no fleet owner supplies it.
   - **Development dependency** — the project package manifest owns it and the
     project environment installer materializes `.venv`.
   - **Native dependency** — prefer a packaged binary variant; otherwise a
     valid backend in the project's managed-artifact owner supplies it.
3. Reject the request when two layers claim the same binary, the requested
   scope exceeds the owner, the pin cannot be derived from a released version,
   or the owner's generator is unavailable.

## Change through the owner

1. Edit the typed configuration owned by the selected layer. Include only the
   released version, source/backend, binaries, and platform data required by
   that owner's schema.
2. Run the owner's declared generator/installer. It must create or update the
   derived lock, launcher, shim, or environment as one atomic effect.
3. Run the generator a second time and require no diff. Prove the real command
   from the activated workspace or declared host surface, including its
   version/identity when the task requires a pin.
4. Remove obsolete entries, duplicate installers, checkout links, aliases, and
   generated edits in the same cutover.

## Lock ownership inside a workspace

A member of a declared workspace does not own its own lock while the umbrella is
present. The workspace root's member list makes every resolution a workspace
resolution, so locking from inside a member updates the root lock, not the
member's. That is the owner's correct behaviour, not a defect, and no flag
suppresses it.

A member's own lock is therefore resolved only where the umbrella is absent —
its standalone checkout, which is what CI builds. Never conclude that an upgrade
verb is broken because a member pin did not move: prove which lock the run wrote
before treating the pin as stale, and prove the installed revision before
treating a generated artifact as its output.

Never place a manual copy in a user bin directory, invent a tool alias, reuse
another repository's checkout, or resolve a missing credential from a file or
keyring. A failed install, lock, trust, activation, version check, or runtime
command propagates unchanged and leaves no partial owner claim.
