# GPT Function Calling Tests for Driving Behavior Monitoring

This document describes the GPT function calling test implementation for analyzing driving behavior violations, particularly focused on red light and stop sign infractions.

## Overview

The test suite provides:
- **Function calling capabilities** for GPT-4 to analyze driving scenarios
- **Local testing** of driving behavior analysis functions
- **Integration** with existing Azure Content Understanding workflow
- **Comprehensive test scenarios** for traffic violations

## Files Added

### 1. `test_gpt_function_calling.py`
Main GPT function calling test module with:
- `DrivingBehaviorFunctions` class with violation analysis functions
- `GPTFunctionCallingTester` class for GPT integration
- Complete test suite with real-world driving scenarios

### 2. `test_function_calling_basic.py`
Basic testing module for local function validation:
- Tests functions without requiring OpenAI API
- Validates violation detection logic
- Ensures regulation lookup works correctly

### 3. `.env.example`
Example environment configuration showing required variables

## Functions Implemented

### `analyze_red_light_violation()`
Analyzes red light violations based on:
- Vehicle position relative to intersection
- Traffic light status (red/yellow/green)
- Whether intersection was crossed
- Timestamp of event

### `analyze_stop_sign_violation()`
Analyzes stop sign violations based on:
- Vehicle speed at stop sign
- Presence of stop sign
- Whether complete stop was made
- Location of incident

### `get_traffic_regulation()`
Looks up traffic regulations for:
- Violation type (red_light, stop_sign)
- State (currently supports Washington)
- Returns statute, description, penalties

## Usage

### Basic Function Testing (No API Key Required)

```bash
# Test the functions locally
python test_function_calling_basic.py
```

This runs comprehensive tests on all driving behavior analysis functions without requiring external API access.

### GPT Function Calling Tests (Requires OpenAI API Key)

1. **Set up environment:**
```bash
# Copy and configure environment file
cp .env.example .env
# Edit .env and add your OpenAI API key:
# OPENAI_API_KEY=your_api_key_here
```

2. **Run GPT function calling tests:**
```python
from test_gpt_function_calling import GPTFunctionCallingTester

# Initialize tester
tester = GPTFunctionCallingTester()

# Test single scenario
result = tester.test_driving_scenario(
    "A vehicle approaches a red light at 2:30 PM and continues through the intersection"
)

# Run full test suite
results = tester.run_test_suite()
```

3. **Run from command line:**
```bash
python test_gpt_function_calling.py
```

## Test Scenarios

The test suite includes these scenarios:

1. **Red Light Violation**: Vehicle continues through intersection at red light
2. **Rolling Stop**: Vehicle doesn't come to complete stop at stop sign
3. **Legal Stop**: Vehicle properly stops at red light
4. **Complete Stop**: Vehicle makes proper stop at stop sign
5. **Yellow Light**: Vehicle legally enters intersection on yellow

## Expected Output

### Basic Tests
```
🧪 Running Basic Function Tests
==================================================

🚦 Testing Red Light Analysis Functions...
Red Light Tests: 3/3 passed
  ✓ Clear Red Light Violation
  ✓ No Violation - Stopped at Red
  ✓ No Violation - Green Light

🛑 Testing Stop Sign Analysis Functions...
Stop Sign Tests: 3/3 passed
  ✓ Rolling Stop Violation
  ✓ Complete Stop - No Violation
  ✓ No Stop Sign Present

📋 Testing Regulation Lookup Functions...
Regulation Tests: 3/3 passed

📊 Overall Results:
Total Tests: 9
Passed: 9
Failed: 0
Success Rate: 100.0%
```

### GPT Function Calling Tests
```
🚗 GPT Function Calling Test for Driving Behavior Analysis
============================================================
✓ GPT Function Calling Tester initialized

📋 Running test suite...

Testing scenario: A vehicle approaches an intersection with a red traffic...
✓ Success - 1 function calls made

📊 Test Results Summary:
========================================
Total tests: 5
Successful tests: 5
Failed tests: 0
Total function calls made: 7
```

## Integration with Azure Content Understanding

These tests complement the existing Azure Content Understanding workflow:

1. **Video Analysis**: Use Azure Content Understanding to extract driving behavior from video
2. **Document Processing**: Extract traffic regulations from PDF documents
3. **GPT Function Calling**: Analyze extracted content using GPT with the violation detection functions
4. **Violation Detection**: Generate alerts for traffic violations

## Error Handling

The implementation includes comprehensive error handling for:
- Missing OpenAI API key
- Invalid function parameters
- Network connectivity issues
- GPT API rate limits

## Extending the Tests

To add new violation types:

1. Add new analysis function to `DrivingBehaviorFunctions`
2. Add corresponding function schema to `GPTFunctionCallingTester`
3. Update `_call_function()` method to handle new function
4. Add test scenarios to validate the new function

## Dependencies

- `openai>=1.0.0` - OpenAI GPT API client
- `python-dotenv` - Environment variable management
- `azure-identity` - Azure authentication (existing)
- `requests` - HTTP requests (existing)

## Note on GPT-5

The request mentioned "GPT-5" but GPT-5 does not exist yet. This implementation uses GPT-4 which has robust function calling capabilities. When GPT-5 becomes available, simply update the model name in the `test_driving_scenario()` method.