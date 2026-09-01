---
applyTo: '**'
---

# Bash Guard Chaining Precision

Top-level `;`, newline, `&&`, `||`, `|`, and `&` create additional shell execution paths and are denied.

The sole semicolon exception is a pure `export` statement containing one or more `NAME=value` assignments, followed by exactly one top-level `;` and exactly one nonempty governed command.

Leading assignments and configured command wrappers never hide the governed executable. Every Bash guard applies to that executable and its arguments.

Quoted or escaped operator characters are command data, not shell chaining.

Exports that execute command, backtick, or process substitutions are denied. Malformed exports and shell syntax are denied.
