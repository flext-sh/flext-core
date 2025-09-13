# FLEXT-CORE Test Standardization Guide

## Overview

This guide outlines how to fix pytest errors and standardize tests using flext_tests patterns and fixtures. Based on analysis of 68 failing tests, systematic issues have been identified and solutions provided.

## Key Issues Found

### 1. **Protocol Instantiation Errors**
**Problem**: Tests trying to instantiate Protocol interfaces
```python
# ❌ WRONG
validator = config.ConfigValidator()  # ConfigValidator is a Protocol!
```
**Solution**: Use concrete implementations
```python
# ✅ CORRECT  
result = config.validate_runtime_requirements()  # Use instance method
```

### 2. **API Parameter Mismatches**
**Problem**: Using wrong parameter names
```python  
# ❌ WRONG
bus = FlextCommands.Bus(config=config)
```
**Solution**: Use correct parameter names  
```python
# ✅ CORRECT
bus = FlextCommands.Bus(bus_config=config)
```

### 3. **Error Message Assertion Mismatches**
**Problem**: Expecting different error messages than implementation returns
```python
# ❌ WRONG
assert "Expected list" in result.error
```
**Solution**: Use actual error messages
```python  
# ✅ CORRECT
assert "Type mismatch" in result.error  # Actual message from FlextConstants.Messages.TYPE_MISMATCH
```

### 4. **FlextResult Property Access Errors**
**Problem**: Using wrong property names
```python
# ❌ WRONG  
assert result.success  # Wrong property name
```
**Solution**: Use correct FlextResult API
```python
# ✅ CORRECT
assert result.is_success  # Correct property name
FlextTestsMatchers.assert_result_success(result)  # Even better - use fixtures
```

## Standardization Approach

### Step 1: Import FlextTests
```python
from flext_tests import FlextTestsFixtures, FlextTestsMatchers
```

### Step 2: Use FlextTests Fixtures
```python
@pytest.fixture  
def fixtures() -> FlextTestsFixtures:
    return FlextTestsFixtures()

def test_with_fixtures(fixtures: FlextTestsFixtures) -> None:
    # Create test data using fixtures
    test_data = fixtures.create_test_data()
    result = fixtures.create_success_result(test_data)
    
    # Use standardized assertions
    FlextTestsMatchers.assert_result_success(result, expected_data=test_data)
```

### Step 3: Use Actual API Methods
```python
# Check actual implementation before writing tests
def test_config_validation() -> None:
    config = FlextConfig(app_name="test")
    
    # Use actual instance methods, not Protocol instantiation
    result = config.validate_runtime_requirements()  
    FlextTestsMatchers.assert_result_success(result)
```

### Step 4: Use Real Error Messages
```python
def test_validation_with_real_errors() -> None:
    result = FlextValidations.TypeValidators.validate_dict("not_dict")
    FlextTestsMatchers.assert_result_failure(result)
    
    # Use actual error message from FlextConstants.Messages
    assert "Type mismatch" in result.error  # Not "Expected dict"
```

## Common Fixes Required

### FlextConfig Tests
```python
# ❌ WRONG - Protocol instantiation
validator = config.ConfigValidator()
result = validator.validate_runtime_requirements()

# ✅ CORRECT - Instance method
result = config.validate_runtime_requirements()
```

### FlextCommands Tests  
```python
# ❌ WRONG - Wrong parameter name
bus = FlextCommands.Bus(config=config)

# ✅ CORRECT - Correct parameter name
bus = FlextCommands.Bus(bus_config=config) 
```

### FlextValidations Tests
```python
# ❌ WRONG - Wrong error message expectation
assert "Expected list" in result.error

# ✅ CORRECT - Actual error message  
assert "Type mismatch" in result.error
```

### FlextResult Tests
```python
# ❌ WRONG - Wrong property access
assert result.success
assert isinstance(payload, dict)  # payload is FlextResult, not dict

# ✅ CORRECT - Proper FlextResult usage  
FlextTestsMatchers.assert_result_success(result)
payload_data = result.unwrap()  # Extract data first
assert isinstance(payload_data, dict)
```

## Systematic Fix Process

### For Each Failing Test:

1. **Read the actual implementation** of the method being tested
2. **Check parameter names and types** in the actual API  
3. **Verify error messages** by looking at FlextConstants.Messages
4. **Use FlextTestsFixtures** for test data creation
5. **Use FlextTestsMatchers** for assertions
6. **Test the fix** before moving to next test

### Example: Fixing a Validation Test

**Before (Failing)**:
```python
def test_validation() -> None:
    result = FlextValidations.validate_user_data({})
    assert result.is_failure
    assert "name is required" in result.error  # Wrong message
```

**After (Fixed)**:
```python  
def test_validation(self) -> None:
    fixtures = FlextTestsFixtures()
    
    result = FlextValidations.validate_user_data({})
    FlextTestsMatchers.assert_result_failure(result)
    assert "Missing required field: name" in result.error  # Actual message
    
    # Test success case with proper data
    valid_data = {
        "name": "Test User",
        "email": "test@example.com",
        "age": "25"  # Use correct type expected by validation
    }
    success_result = FlextValidations.validate_user_data(valid_data)
    FlextTestsMatchers.assert_result_success(success_result)
```

## Files Requiring Standardization

Priority order (most critical issues first):

1. **test_config_*.py** - Protocol instantiation issues
2. **test_commands_*.py** - API parameter mismatches  
3. **test_validations_*.py** - Error message mismatches
4. **test_result_*.py** - FlextResult property access errors
5. **test_container_*.py** - Missing method errors

## Validation

After fixing tests, run:
```bash
PYTHONPATH=src pytest tests/unit/test_standardized_example.py -v  # Should pass
PYTHONPATH=src pytest tests/ --tb=short -q  # Check overall progress
```

## Success Criteria

- ✅ All tests use `from flext_tests import FlextTestsFixtures, FlextTestsMatchers`
- ✅ All tests use actual API method names and parameters
- ✅ All tests use actual error messages from implementation  
- ✅ All tests use `FlextTestsMatchers.assert_result_success/failure()` instead of manual assertions
- ✅ Zero Protocol instantiation attempts
- ✅ All FlextResult access uses proper `.is_success`, `.unwrap()`, etc.

This approach reduces 68 test failures to systematic, fixable patterns following the flext-core architecture and flext_tests standardization.
