# Appendix F: Frequently Asked Questions (FAQ)

**Status**: TEAM SUPPORT RESOURCE
**Purpose**: Answer common questions from team members
**Usage**: Quick reference for decision-making

---

## General Questions

### Q: Do we have to migrate everything at once?

**A**: No. Migration is phased:
1. **Phase 1** (Week 1): flext-core foundation
2. **Phase 2** (Week 2): High-priority projects
3. **Phase 3** (Week 3): Remaining ecosystem

Each project can be completed independently, but flext-core must be done first since 32+ projects depend on it.

**Timeline**: 3 weeks total for full ecosystem

---

### Q: Will this break our existing code?

**A**: Not if we follow the deprecation strategy:
- Old APIs remain functional during 2-version transition period (6+ months)
- Deprecation warnings guide developers to new patterns
- Migration tools automate common replacements

**Safety**: All dependent projects get 6+ months to migrate

---

### Q: Why migrate to Pydantic v2?

**A**: Multiple benefits:
1. **Performance**: 50-70% faster JSON parsing
2. **Type Safety**: Better type checking with Pyrefly
3. **Maintainability**: Less custom validation code (~270 lines removed, verified)
4. **Standards**: Industry standard for Python data validation
5. **Community**: Better support and documentation

**Bottom Line**: Modern, maintainable, faster code

---

## Technical Questions

### Q: What's the difference between `.model_dump()` and `.dict()`?

**A**: Functionally identical in most cases, but v2 offers more control:

```python
# Pydantic v1
data = user.dict()

# Pydantic v2 - more flexible
data = user.model_dump()                    # Default
data = user.model_dump(mode="python")       # Python types
data = user.model_dump(mode="json")         # JSON-serializable types
data = user.model_dump(exclude={"password"})  # Exclude fields
data = user.model_dump(by_alias=True)       # Use field aliases
```

**Migration**: Simple find/replace `.dict()` → `.model_dump()`

---

### Q: Why use `model_validate_json()` instead of `json.loads()` + `model_validate()`?

**A**: Performance. Pydantic v2 uses Rust parser:

```python
# ❌ SLOW (two passes: Python JSON → Rust validation)
import json
data = json.loads(json_str)
user = User.model_validate(data)

# ✅ FAST (one pass: Rust JSON+validation)
user = User.model_validate_json(json_str)
```

**Benchmark**: 50-70% faster for JSON operations

---

### Q: When should I use `Annotated` types vs regular types?

**A**: Use `Annotated` for reusable constraints:

```python
# ❌ Repetitive (duplicated across models)
class User(BaseModel):
    age: int = Field(ge=0, le=150)

class Employee(BaseModel):
    age: int = Field(ge=0, le=150)

# ✅ Reusable (single definition)
from typing import Annotated

Age = Annotated[int, Field(ge=0, le=150)]

class User(BaseModel):
    age: Age

class Employee(BaseModel):
    age: Age
```

**Rule**: Use `Annotated` for domain-specific types (PortNumber, Email, etc.)

---

### Q: Should I use `strict=True` in ConfigDict?

**A**: Depends on your needs:

```python
# Strict mode - no automatic coercion
class Strict(BaseModel):
    model_config = ConfigDict(strict=True)
    age: int

Strict(age="30")  # Error!

# Lenient mode - automatic coercion (default)
class Lenient(BaseModel):
    age: int

Lenient(age="30")  # Works! age = 30
```

**FLEXT Pattern**: Don't use strict for public APIs (breaks user experience)

---

## Migration Questions

### Q: How long does migration take per project?

**A**: Depends on project size:

| Project Size | Time | Complexity |
|--------------|------|------------|
| Small (1K lines) | 2-4 hours | Low |
| Medium (5K lines) | 4-8 hours | Medium |
| Large (10K+ lines) | 8-12 hours | High |
| flext-core (4.5K lines) | 8 hours | High (dependency impact) |

**Effort Multiplier**: +2x for projects with complex validators/serialization

---

### Q: What order should projects migrate in?

**A**: Dependency order (lower → higher):

1. **Priority 1**: flext-core (foundation)
2. **Priority 2**: Direct dependents (flext-cli, flext-ldif, flext-ldap, flext-api)
3. **Priority 3**: Domain libraries (flext-auth, flext-web, etc.)
4. **Priority 4**: Singer platform (taps, targets, dbt)
5. **Priority 5**: Enterprise projects

**Why**: Prevent breaking dependent projects during migration

---

### Q: Can we skip migration for some projects?

**A**: Not for foundation projects, but yes for low-priority:

```
✅ MUST migrate:
  - flext-core (dependency for 32+ projects)
  - High-priority projects (flext-cli, flext-ldap, etc.)

⏳ NICE TO HAVE:
  - Enterprise projects (lower impact)
  - Deprecated projects (lower maintenance burden)
```

**Decision Rule**: Must migrate if > 2 other projects depend on it

---

## Troubleshooting Questions

### Q: I'm getting "Extra inputs are not permitted" errors

**A**: Pydantic v2 changed default behavior:

```python
# Solution 1: Allow extra fields
class User(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str

# Solution 2: Strip unknown fields
class User(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str

# Solution 3: Validate your data before passing
user_data = {k: v for k, v in data.items() if k in User.model_fields}
user = User(**user_data)
```

**FLEXT Pattern**: Use `extra="allow"` for public APIs, `extra="forbid"` for internal APIs

---

### Q: Type checker says my validators are wrong

**A**: Missing `@classmethod` and type hints:

```python
# ❌ WRONG
@field_validator("age")
def validate_age(cls, value):
    return value

# ✅ CORRECT
@field_validator("age")
@classmethod
def validate_age(cls, value: int) -> int:
    return value
```

**Fix**: Add `@classmethod` and complete type hints

---

### Q: My tests are failing after migration

**A**: Pydantic v2 behavior changed for some cases:

```python
# Frozen model exception changed
# ❌ OLD
with pytest.raises(ValidationError):
    frozen_model.field = "value"

# ✅ NEW
with pytest.raises(AttributeError, match="frozen"):
    frozen_model.field = "value"

# Computed fields now included in dumps
# ❌ OLD (doesn't expect computed field)
assert user.model_dump() == {"name": "Alice", "age": 30}

# ✅ NEW (includes computed field)
assert user.model_dump() == {"name": "Alice", "age": 30, "full_name": "Alice"}
```

**Solution**: Update test expectations to match v2 behavior

---

## Architecture Questions

### Q: How does this fit with FLEXT's railway pattern?

**A**: Perfectly! Pydantic validation complements FlextResult:

```python
from flext_core import FlextResult

class UserValidator(BaseModel):
    """Pydantic handles field-level validation."""
    name: str
    age: int = Field(ge=0, le=150)

def create_user(data: dict) -> FlextResult[User]:
    """FlextResult handles business logic errors."""
    try:
        validated = UserValidator(**data)
        user = save_to_database(validated)
        return FlextResult[User].ok(user)
    except ValidationError as e:
        return FlextResult[User].fail(str(e))
```

**Pattern**: Pydantic for structural validation, FlextResult for business logic

---

### Q: Should we deprecate custom validators in utilities.py?

**A**: Yes! But gradually:

```python
# Phase 1 (now): Deprecation warning
def validate_port(value: int) -> FlextResult[int]:
    """DEPRECATED: Use PortNumber type instead."""
    warnings.warn("Use PortNumber", DeprecationWarning, stacklevel=2)
    # Still works, but warns users

# Phase 2 (6 months): Document removal
# "Removal in v1.2.0"

# Phase 3 (6+ months later): Remove
# DELETE the function
```

**Timeline**: 2-version deprecation cycle minimum

---

## Performance Questions

### Q: How much faster will JSON operations be?

**A**: Depends on data size:

```
Small objects (< 1KB): 20-30% faster
Medium objects (1-10KB): 40-60% faster
Large objects (> 10KB): 50-70% faster
Large lists (1000+ items): 60-80% faster
```

**Benchmark**: Average 50-70% improvement with `model_validate_json()`

---

### Q: Should we benchmark every project?

**A**: Only if it's performance-critical:

```python
# DO benchmark these projects:
- flext-core (foundation)
- Singer platform taps/targets (high volume)
- Real-time processing services

# OPTIONAL for these:
- CLI tools
- Administrative utilities
- Configuration management
```

**Effort vs Benefit**: Only if potential 10%+ improvement

---

## Team Questions

### Q: Do I need to be Pydantic expert to migrate my project?

**A**: No! We provide:
- Complete migration checklist
- Common error solutions
- Code examples for all patterns
- Migration tools

**Learning**: 30-45 minutes to understand v2 patterns

---

### Q: Who should I ask for help during migration?

**A**: Escalation path:

1. **Check this documentation**: 90% of questions answered
2. **Appendix C (Common Errors)**: Troubleshooting guide
3. **Code Examples**: Reference implementation
4. **Team lead**: Complex architectural decisions
5. **Slack #flext-support**: Real-time help

---

### Q: Will this affect our release schedule?

**A**: 3-week effort:

```
Current Schedule: [unchanged]
Migration: 3 weeks sprint (parallel with other work)
Deployment: Phase 1 (foundation) + Phase 2-3 (gradual)

Impact: Minimal if distributed across team
```

**Mitigation**: Phase into releases, don't delay other features

---

## Validation Questions

### Q: How do I validate complex business rules?

**A**: Use `@model_validator` with mode:

```python
from pydantic import BaseModel, model_validator

class DateRange(BaseModel):
    start: date
    end: date
    name: str

    @model_validator(mode="after")
    def validate_business_rules(self) -> "DateRange":
        """Run after model creation - access full object."""
        if self.end < self.start:
            raise ValueError("End must be after start")
        if (self.end - self.start).days > 365:
            raise ValueError("Max range 1 year")
        return self
```

**Pattern**: `@model_validator` for cross-field rules

---

### Q: Can validators call external services?

**A**: Yes, but consider performance:

```python
# ❌ DON'T - blocks model creation
class User(BaseModel):
    email: str

    @field_validator("email")
    @classmethod
    def verify_email_exists(cls, value: str) -> str:
        # DON'T call external API here - too slow!
        response = verify_email_api(value)
        return value

# ✅ DO - verify after creation
user = User(email="test@example.com")
result = verify_email_api(user.email)
```

**Rule**: Keep validators fast, do slow checks separately

---

## Support Resources

### Where to Find Help

1. **This FAQ**: Most common questions
2. **Appendix C**: Specific error solutions
3. **Appendix E**: Code examples
4. **Appendix D**: Term definitions
5. **Official Docs**: `/home/marlonsc/flext/docs/references/pydantic2/`
6. **Team**: Slack #flext-support

---

### Reporting Issues

If you find an issue:
1. Check this FAQ and Appendix C
2. Verify it's not your code (post code example)
3. Search GitHub issues
4. Report with: Project, Pydantic version, exact error, minimal example

---

**Last Updated**: 2025-01-21  
**Maintainer**: FLEXT Team  
**Questions?**: Post in #flext-support on Slack

