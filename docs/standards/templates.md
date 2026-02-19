# Documentation Templates

<!-- TOC START -->

- [Feature Guide Template](#feature-guide-template)
- [Overview](#overview)
- [Basic Usage](#basic-usage)
  - [Simple Example](#simple-example)
  - [Common Patterns](#common-patterns)
- [API Reference](#api-reference)
  - [Method/Class 1](#methodclass-1)
  - [Method/Class 2](#methodclass-2)
- [Best Practices](#best-practices)
  - [✅ DO](#do)
  - [❌ DON'T](#dont)
- [Common Issues](#common-issues)
  - [Issue: [Problem Description]](#issue-problem-description)
  - [Issue: [Problem Description]](#issue-problem-description)
- [Advanced Topics](#advanced-topics)
- [Summary](#summary)
- [See Also](#see-also)
- [API Reference Template](#api-reference-template)
- [[Class/Module Name]](#classmodule-name)
  - [Class Methods](#class-methods)
  - [Properties](#properties)
- [Quality Metrics](#quality-metrics)
- [Usage Examples](#usage-examples)
  - [Complete Example](#complete-example)
- [See Also](#see-also)
- [Troubleshooting Template](#troubleshooting-template)
- [[Category of Issues]](#category-of-issues)
  - [Issue: [Problem Description]](#issue-problem-description)
- [[Another Category]](#another-category)
  - [Issue: [Problem Description]](#issue-problem-description)
- [Architecture Decision Record (ADR) Template](#architecture-decision-record-adr-template)
- [ADR-\[Number\]: [Title]](#adr-number-title)
  - [Problem](#problem)
  - [Decision](#decision)
  - [Rationale](#rationale)
  - [Alternatives Considered](#alternatives-considered)
  - [Consequences](#consequences)
  - [Related Decisions](#related-decisions)
- [Best Practices Guide Template](#best-practices-guide-template)
- [Principle 1: [Name]](#principle-1-name)
- [Principle 2: [Name]](#principle-2-name)
- [Common Patterns](#common-patterns)
  - [Pattern 1: [Name]](#pattern-1-name)
  - [Pattern 2: [Name]](#pattern-2-name)
- [Common Mistakes](#common-mistakes)
  - [Mistake 1: [What people get wrong]](#mistake-1-what-people-get-wrong)
- [Summary](#summary)
- [Quick Reference Card Template](#quick-reference-card-template)
- [Installation](#installation)
- [Basic Setup](#basic-setup)
- [Common Tasks](#common-tasks)
  - [Task 1](#task-1)
  - [Task 2](#task-2)
- [Common Issues](#common-issues)
- [Resources](#resources)
- [Configuration Documentation Template](#configuration-documentation-template)
- [Overview](#overview)
- [Configuration Methods](#configuration-methods)
  - [Method 1: Configuration File](#method-1-configuration-file)
  - [Method 2: Environment Variables](#method-2-environment-variables)
  - [Method 3: Programmatic](#method-3-programmatic)
- [Configuration Options](#configuration-options)
  - [option_name](#optionname)
- [Environment-Specific](#environment-specific)
  - [Development](#development)
  - [Production](#production)
- [Complete Example](#complete-example)
- [Comparison/Alternatives Template](#comparisonalternatives-template)
- [[Feature A] Details](#feature-a-details)
- [[Feature B] Details](#feature-b-details)
- [[Feature C] Details](#feature-c-details)
- [Decision Matrix](#decision-matrix)
- [Implementation Guide Template](#implementation-guide-template)
- [Step 1: [Name]](#step-1-name)
- [Step 2: [Name]](#step-2-name)
- [Step 3: [Name]](#step-3-name)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)
- [What's Next](#whats-next)
- [Tips for Using Templates](#tips-for-using-templates)
- [Example: Using the Feature Guide Template](#example-using-the-feature-guide-template)
- [Maintenance](#maintenance)

<!-- TOC END -->

**Reviewed**: 2026-02-17 | **Scope**: Canonical rules alignment and link consistency

Reusable templates for creating consistent FLEXT-Core documentation.

## Feature Guide Template

Use for new feature documentation (guides/ directory).

````markdown
# [Feature Name] Guide

[One sentence: What is this and why use it?]

## Overview

[2-3 paragraphs explaining the feature]

**Key Features:**

- Feature 1
- Feature 2
- Feature 3

## Basic Usage

### Simple Example

```python
from flext_core import [Import]

# Your example here
```
````

### Common Patterns

#### Pattern 1: [Name]

```python
# Example
```

#### Pattern 2: [Name]

```python
# Example
```

## API Reference

### Method/Class 1

[Description]

**Usage:**

```python
# Example
```

### Method/Class 2

[Description]

**Usage:**

```python
# Example
```

## Best Practices

### ✅ DO

- Practice 1: Explanation
- Practice 2: Explanation
- Practice 3: Explanation

### ❌ DON'T

- Anti-pattern 1: Why avoid it
- Anti-pattern 2: Why avoid it
- Anti-pattern 3: Why avoid it

## Common Issues

### Issue: [Problem Description]

**Symptom:** What the user sees

**Solution:** How to fix it

```python
# Example fix
```

### Issue: [Problem Description]

**Symptom:** What the user sees

**Solution:** How to fix it

## Advanced Topics

[Optional section for advanced patterns]

## Summary

[One paragraph recap of key points]

## See Also

- Related Guide 1
- Related Guide 2
- API Reference

````

## API Reference Template

Use for API documentation (api-reference/ directory).

```markdown
# [Layer Name] API Reference

[Overview of what this layer provides]

## [Class/Module Name]

[Purpose and use cases]

### Class Methods

#### method_name(param1: Type, param2: Type) -> ReturnType

**Description:**
[What does this method do?]

**Parameters:**
- `param1` (Type): [Explanation]
- `param2` (Type): [Explanation]

**Returns:**
(ReturnType): [What is returned?]

**Example:**
```python
from flext_core import [Import]

# Example usage
````

**See Also:**

- `related_method()`: [Why related]

### Properties

#### property_name

**Type:** DataType

**Description:** [What represents?]

**Example:**

```python
obj.property_name  # Access property
```

## Quality Metrics

| Component  | Coverage | Status | Description |
| ---------- | -------- | ------ | ----------- |
| `file1.py` | XX%      | ✅/🔄  | Description |
| `file2.py` | XX%      | ✅/🔄  | Description |

## Usage Examples

### Complete Example

[Show how multiple components work together]

```python
# Full working example
```

## See Also

- Related API Reference
- How-To Guide

````

## Troubleshooting Template

Use for troubleshooting guides (guides/troubleshooting.md or similar).

```markdown
# [System/Component] Troubleshooting

## [Category of Issues]

### Issue: [Problem Description]

**Symptom:**
[What user observes]

**Common Causes:**
1. Cause 1
2. Cause 2
3. Cause 3

**Solutions:**

**Solution 1: [Approach Name]**
```bash
# Commands or code
````

**Solution 2: [Approach Name]**

```bash
# Commands or code
```

**When to check:**

- When X happens
- When Y is configured as Z

______________________________________________________________________

## [Another Category]

### Issue: [Problem Description]

[Same format as above]

````

## Architecture Decision Record (ADR) Template

Use for architecture/decisions.md.

```markdown
## ADR-[Number]: [Title]

**Status:** PROPOSED | ACCEPTED | DEPRECATED | **Date:** YYYY-MM-DD

### Problem
[What challenge are we facing?]

### Decision
[What choice are we making?]

### Rationale
[Why is this the right choice?]

### Alternatives Considered
- Alternative 1: [Why not this?]
- Alternative 2: [Why not this?]

### Consequences
**Positive:**
- ✅ Good consequence 1
- ✅ Good consequence 2

**Negative:**
- ❌ Bad consequence 1
- ❌ Bad consequence 2

### Related Decisions
- Related ADR
- Related pattern
````

## Best Practices Guide Template

Use for comprehensive guide on patterns/best practices.

````markdown
# [Topic] Best Practices

[Intro: Why is this important?]

## Principle 1: [Name]

**When to use:** [Context]

```python
# ✅ CORRECT
# Example showing best practice
```
````

```python
# ❌ WRONG
# Example showing anti-pattern
```

**Why:** [Explanation of benefits]

## Principle 2: [Name]

[Same structure]

## Common Patterns

### Pattern 1: [Name]

[Description and example]

### Pattern 2: [Name]

[Description and example]

## Common Mistakes

### Mistake 1: [What people get wrong]

```python
# ❌ WRONG - What to avoid
```

**Fix:**

```python
# ✅ CORRECT - How to fix it
```

## Summary

[Key takeaways]

````

## Quick Reference Card Template

Use for cheat sheets or quick reference.

```markdown
# [Feature] Quick Reference

## Installation
```bash
pip install flext-core
````

## Basic Setup

```python
from flext_core import [Import]

# Quick setup
```

## Common Tasks

### Task 1

```python
# Code
```

### Task 2

```python
# Code
```

## Common Issues

| Issue     | Solution |
| --------- | -------- |
| Problem 1 | Fix 1    |
| Problem 2 | Fix 2    |

## Resources

- Full Guide
- API Reference
- Examples

````

## Configuration Documentation Template

Use for documenting configuration options.

```markdown
# [Component] Configuration

## Overview
[What can be configured?]

## Configuration Methods

### Method 1: Configuration File

**File:** `config.toml`

```toml
[section]
key1 = "value"
key2 = 123
````

### Method 2: Environment Variables

```bash
export SETTING_NAME=value
```

### Method 3: Programmatic

```python
from flext_core import [Component]

config = Component
```

## Configuration Options

### option_name

- **Type:** DataType
- **Default:** default_value
- **Required:** Yes/No
- **Description:** [What does it do?]

```python
# Example
```

## Environment-Specific

### Development

```toml
[config.development]
```

### Production

```toml
[config.production]
```

## Complete Example

```toml
# Full working configuration
```

````

## Comparison/Alternatives Template

Use when comparing similar features.

```markdown
# [Feature A] vs [Feature B] vs [Feature C]

| Aspect | [Feature A] | [Feature B] | [Feature C] |
| -------- | ------------- | ------------- | ------------- | 
| Use Case | [Description] | [Description] | [Description] |
| Complexity | Low/Medium/High | Low/Medium/High | Low/Medium/High |
| Performance | Fast/Moderate/Slow | ... | ... |
| Learning Curve | Easy/Moderate/Hard | ... | ... |
| When to Use | [When] | [When] | [When] |

## [Feature A] Details

```python
# Example usage
````

## [Feature B] Details

```python
# Example usage
```

## [Feature C] Details

```python
# Example usage
```

## Decision Matrix

- Choose [Feature A] if you need [criteria]
- Choose [Feature B] if you need [criteria]
- Choose [Feature C] if you need [criteria]

````

## Implementation Guide Template

Use for step-by-step implementation instructions.

```markdown
# Implementing [Feature Name]

**Time Required:** X hours

**Prerequisites:**
- Prerequisite 1
- Prerequisite 2

## Step 1: [Name]

[Description]

```python
# Code for step 1
````

**Verify:**

```bash
# How to verify this step worked
```

## Step 2: [Name]

[Description]

```python
# Code for step 2
```

**Verify:**

```bash
# How to verify this step worked
```

## Step 3: [Name]

[Description]

```python
# Code for step 3
```

## Verification

Run the following to verify complete implementation:

```bash
# Test command
```

Expected output:

```
[Expected output]
```

## Troubleshooting

[Common issues during implementation]

## What's Next

- Next feature
- Related topic

```

## Tips for Using Templates

1. **Copy the template** to create new documentation
2. **Replace brackets** [like this] with your content
3. **Remove optional sections** if not needed
4. **Add examples** liberally
5. **Verify all code examples** work before publishing
6. **Check links** are valid
7. **Pass documentation standards** quality gate

## Example: Using the Feature Guide Template

See Configuration Guide as a real example of the Feature Guide Template.

## Maintenance

- Add new templates as new documentation types emerge
- Update existing templates based on feedback
- Keep examples current and working
- Review template usage quarterly
```
