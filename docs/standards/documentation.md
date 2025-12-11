# Documentation Standards

FLEXT-Core documentation follows strict standards for clarity, accuracy, and maintainability.

## Documentation Structure

### Every Feature Needs

- **Overview**: What is it and why use it?
- **Example**: Working code showing basic usage
- **API Reference**: Methods, parameters, return values
- **Best Practices**: Do's and don'ts
- **Common Issues**: Troubleshooting guide

````markdown
# Feature Name

## Overview

One paragraph explaining what this feature does and why use it.

## Basic Usage

```python
# Working example
```
````

## API Reference

### Method Name

- **Parameters**: What it takes
- **Returns**: What it gives back
- **Example**: How to use it

## Best Practices

### ✅ DO

- Recommend this approach

### ❌ DON'T

- Avoid this approach

## Common Issues

### Issue: Something breaks

**Solution**: How to fix it

````

## Code Examples

### Correct Style

```python
# ✅ CORRECT - Complete, working example
from flext_core import FlextResult

def divide(a: float, b: float) -> FlextResult[float]:
    """Divide two numbers safely."""
    if b == 0:
        return FlextResult[float].fail("Division by zero")
    return FlextResult[float].ok(a / b)

result = divide(10, 2)
if result.is_success:
    print(f"Result: {result.unwrap()}")
else:
    print(f"Error: {result.error}")
````

### Import Cleanup

**Rule**: Import ONLY what each example uses.

```python
# ✅ CORRECT - Only FlextResult
from flext_core import FlextResult

# ❌ WRONG - Unnecessary imports
from flext_core import (
    FlextDispatcher,
    FlextSettings,
    FlextConstants,
    FlextContainer,
    FlextContext,
    FlextDecorators,
    FlextExceptions,
    h,
    FlextLogger,
    x,
    FlextModels,
    p,
    FlextRegistry,
    FlextResult,  # Only this one!
    FlextRuntime,
    FlextService,
    t,
    u,
)
```

### Syntax Highlighting

Always use proper code fence language:

```python
# Python code
```

```bash
# Bash commands
```

```toml
# Configuration files
```

## Markdown Style

### Headings

```markdown
# Top Level (Main Title)

## Section

### Subsection

#### Detail
```

### Lists

```markdown
# Unordered

- Item 1
- Item 2
- Item 3

# Ordered

1. First step
2. Second step
3. Third step
```

### Emphasis

```markdown
- **Bold** for important terms
- _Italics_ for stress/emphasis
- `Code` for inline code references
```

### Tables

```markdown
| Header 1 | Header 2 | Header 3 |
| -------- | -------- | -------- |
| Value 1  | Value 2  | Value 3  |
| Value 4  | Value 5  | Value 6  |
```

## Content Accuracy

### Verify Everything

- ✅ Run all code examples before publishing
- ✅ Verify API names match source code
- ✅ Test all command-line examples
- ✅ Update version numbers when needed

### Content Must Be True

```markdown
# ✅ CORRECT - Verified claim

FlextResult has three methods: ok(), fail(), and unwrap().
(Then show all three working)

# ❌ WRONG - Unverified claim

FlextResult makes your code 100% bug-free.
(This is false and unsupported)
```

### Update Frequency

- **Critical fixes**: Same day
- **Minor updates**: Weekly
- **Refresh audit**: Monthly

Keep documentation fresh and accurate.

## Breaking Down Complex Topics

### Short Format (< 1000 words)

Use when documenting:

- Single feature
- Basic usage
- Quick reference

```markdown
# Feature Name (< 200 words)

## Usage (1-2 examples)

## Best Practices (3-5 dos/don'ts)
```

### Medium Format (1000-3000 words)

Use when documenting:

- Feature with variations
- Integration patterns
- Common workflows

```markdown
# Feature Name

## Overview (100-200 words)

## Basic Usage (300-500 words)

## Advanced Usage (400-600 words)

## Best Practices (300-400 words)

## Troubleshooting (300-500 words)
```

### Long Format (3000+ words)

Use for comprehensive guides:

- Complete framework overview
- Architecture guide
- Migration guide

```markdown
# Comprehensive Guide

## Introduction (200 words)

## Fundamentals (800 words)

- Concept 1
- Concept 2
- Concept 3

## Patterns (1000 words)

- Pattern 1
- Pattern 2
- Pattern 3

## Best Practices (800 words)

## Common Issues (500 words)

## Summary (200 words)
```

## Status Badges

Use consistent status indicators:

```markdown
# ✅ Complete - Ready for production

# 🔄 In Progress - Being worked on

# ⚠️ Partial - Incomplete, use with caution

# ❌ Deprecated - Don't use, see alternative
```

## Cross-References

Link to related content properly:

```markdown
# ✅ CORRECT

See [Getting Started](../guides/getting-started.md) for installation.

For advanced patterns, check [Architecture Patterns](../architecture/patterns.md).

# ❌ WRONG

See the getting started guide (file is located in docs/guides/)
```

## Link Reference Standards

**CRITICAL RULE**: Different link types for different contexts

### Cross-Project References (Between FLEXT Projects)

**Use GitHub URLs** for cross-project references:

```markdown
# ✅ CORRECT - GitHub URLs for cross-project

See [flext-core Result Pattern](https://github.com/organization/flext/tree/main/flext-core/docs/guides/railway-oriented-programming.md)

See [flext-ldif Architecture](https://github.com/organization/flext/tree/main/flext-ldif/docs/architecture.md)

# ❌ WRONG - Relative paths for cross-project

See [flext-core](../../flext-core/docs/guides/railway-oriented-programming.md)
```

**Rationale**:

- GitHub URLs remain valid when viewing individual project repos
- Works across different repository setups (monorepo, separate repos)
- Consistent regardless of workspace organization

### Internal References (Within Same Project)

**Use relative paths** for same-project references:

```markdown
# ✅ CORRECT - Relative paths within project

See [Getting Started](./guides/getting-started.md)
See [Architecture Overview](../architecture/overview.md)
See [API Reference](./api-reference/foundation.md)

# ❌ WRONG - GitHub URLs within project

See [Getting Started](https://github.com/organization/flext/tree/main/flext-core/docs/guides/getting-started.md)
```

**Rationale**:

- Works in local development environment
- Faster (no external HTTP requests)
- Works offline
- Survives repository moves/renames

### Link Pattern Matrix

| Reference Type       | Link Format        | Example                                                                  |
| -------------------- | ------------------ | ------------------------------------------------------------------------ |
| **Same project**     | Relative path      | `./guides/getting-started.md`                                            |
| **Parent directory** | Relative path      | `../architecture/overview.md`                                            |
| **Cross-project**    | GitHub URL         | `https://github.com/organization/flext/tree/main/flext-core/docs/api.md` |
| **Workspace docs**   | Relative from root | `../../docs/architecture/README.md`                                      |
| **External**         | Full URL           | `https://docs.python.org/3/`                                             |

### Link Validation Checklist

Before publishing documentation:

- [ ] All cross-project links use GitHub URLs
- [ ] All internal links use relative paths
- [ ] All links resolve correctly
- [ ] No broken links (404 errors)
- [ ] Anchor links (#sections) exist in target files
- [ ] External links use HTTPS

## API Reference Format

### For Functions/Methods

````markdown
### method_name(param1: Type1, param2: Type2) -> ReturnType

**Description**: One sentence explaining purpose.

**Parameters:**

- `param1` (Type1): What is this parameter?
- `param2` (Type2): What is this parameter?

**Returns:**

- (ReturnType): What does it return?

**Example:**

```python
# Working example
```
````

**Raises/Errors:**

- Error1: When this happens
- Error2: When that happens

````

### For Classes

```markdown
## ClassName

**Purpose**: One sentence explaining what this class does.

**Inherits from**: Parent class if applicable

**Usage:**
```python
# Basic usage example
````

### Methods

Each method documented as above.

### Properties

```markdown
### property_name

**Type**: DataType

**Description**: What does this property represent?
```

## Writing Guidelines

### Be Precise

```markdown
# ✅ CORRECT

FlextResult[T] returns either Ok(value) or Fail(error).

# ❌ VAGUE

FlextResult is cool and handles errors.
```

### Be Concise

```markdown
# ✅ CORRECT (1 sentence)

Use ValidationMiddleware to validate incoming messages before processing.

# ❌ VERBOSE (Too much)

The ValidationMiddleware is a powerful tool that you can use in your
application to validate incoming messages. When a message arrives at your
bus, the middleware will check if it's valid. If it's not valid, it will
fail the message...
```

### Be Actionable

```markdown
# ✅ CORRECT - Clear action

To enable debug logging, set `LOG_LEVEL=DEBUG` before running.

# ❌ VAGUE - No action

There is a debug mode available.
```

### Avoid Passive Voice

```markdown
# ✅ ACTIVE

You can register services with the container.

# ❌ PASSIVE

Services can be registered with the container.
```

## Formatting Consistency

### Consistency Checklist

- [ ] Same terminology throughout (not "service" and "handler" interchangeably)
- [ ] Same code style in all examples
- [ ] Same section names across similar documents
- [ ] Same status badges used consistently
- [ ] Same link format throughout

### Example Consistency

```python
# ✅ CONSISTENT - Same style across all examples
result = FlextResult[int].ok(42)
if result.is_success:
    value = result.unwrap()

# ❌ INCONSISTENT - Different styles
result = FlextResult.ok(42)  # First example
res = FlextResult[int].ok(42)  # Second example
r = FlextResult[int].ok(42)  # Third example
```

## Maintenance

### Monthly Review

- [ ] Check all links still valid
- [ ] Verify all examples still work
- [ ] Update version numbers if changed
- [ ] Review for outdated information
- [ ] Check formatting consistency

### Annual Audit

- [ ] Re-verify all technical claims
- [ ] Update deprecated patterns
- [ ] Refresh examples
- [ ] Review for completeness

## Accessibility

### For Code Examples

- Highlight important lines with comments
- Explain what the code does
- Show both correct and wrong examples

```python
# ✅ CORRECT - Commented
result = (
    validate_email(email)      # Validate email format
    .flat_map(check_available) # Check if available
    .map(send_confirmation)    # Send confirmation
)

# ❌ WRONG - No explanation
result = validate_email(email).flat_map(check_available).map(send_confirmation)
```

### For Complex Topics

- Use headings to break up content
- Use lists instead of paragraphs when possible
- Use tables for comparisons
- Use examples liberally

## Quality Gate Checklist

Before publishing documentation:

- [ ] **Accuracy**: All claims verified against source
- [ ] **Completeness**: All major use cases covered
- [ ] **Clarity**: A new user understands it
- [ ] **Examples**: All code examples tested and work
- [ ] **Links**: All cross-references valid
- [ ] **Formatting**: Consistent with other docs
- [ ] **Tone**: Professional, clear, helpful
- [ ] **Updates**: Version numbers and dates current

## Tools and Commands

### Markdown Linting

```bash
# Check markdown style
markdownlint docs/

# Fix common issues
markdownlint --fix docs/
```

### Link Checking

```bash
# Verify all links work
markdown-link-check docs/**/*.md
```

### Spelling

```bash
# Check for spelling errors
spell-check docs/
```

## Examples of Excellent Documentation

- [Flask Documentation](https://flask.palletsprojects.com/) - Clear, concise, well-organized
- [Vue.js Guide](https://vuejs.org/guide/) - Progressive disclosure, examples everywhere
- [Rust Book](https://doc.rust-lang.org/book/) - Comprehensive, well-structured

Use these as inspiration when writing FLEXT-Core documentation.

## Summary

FLEXT-Core documentation standards:

- ✅ Verify everything before publishing
- ✅ Include examples with every feature
- ✅ Keep imports minimal in examples
- ✅ Be precise, concise, and actionable
- ✅ Maintain consistency across documents
- ✅ Update regularly to stay accurate
- ✅ Use accessibility best practices
- ✅ Pass quality gate before publishing

Documentation is part of the product. Maintain the same standards as the code.
