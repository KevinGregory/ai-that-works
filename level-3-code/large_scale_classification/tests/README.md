# Tests Directory

This directory contains all tests for the large-scale classification system.

## Structure

```
tests/
├── README.md                           # This file
├── run_tests.py                        # Main test runner script
├── compare_results.py                  # Utility to compare test results across runs
├── __init__.py                         # Package init
├── data/                              # Test data and fixtures
│   ├── __init__.py
│   └── test_cases.py                  # 25 comprehensive test cases
├── integration/                       # Integration tests
│   ├── __init__.py
│   ├── test_narrowing_accuracy.py     # Narrowing strategy accuracy test
│   ├── test_selection_accuracy.py     # Selection accuracy test
│   └── test_pipeline_accuracy.py      # Complete pipeline accuracy test
└── results/                           # JSON test results (auto-generated)
    ├── narrowing/                     # Narrowing test results
    │   └── narrowing_accuracy_YYYYMMDD_HHMMSS.json
    ├── selection/                     # Selection test results
    │   └── selection_accuracy_YYYYMMDD_HHMMSS.json
    └── pipeline/                      # Pipeline test results
        └── pipeline_accuracy_YYYYMMDD_HHMMSS.json
```

## Available Tests

### 1. Narrowing Accuracy Test (`integration/test_narrowing_accuracy.py`)

**Purpose**: Evaluates how often the correct category is included in the narrowed results for different narrowing strategies.

**What it tests**:
- Embedding-based narrowing strategy
- Hybrid narrowing strategy (embedding + LLM)
- Processing time and performance metrics
- Failure analysis by category hierarchy level

**Metrics provided**:
- Accuracy percentage (% of tests where correct category was in narrowed results)
- Average number of categories returned
- Average processing time
- Detailed failure analysis

### 2. Selection Accuracy Test (`integration/test_selection_accuracy.py`)

**Purpose**: Evaluates how often the correct category is selected by the LLM from the narrowed candidate categories.

**What it tests**:
- LLM-based category selection from narrowed candidates
- Processing time and performance metrics
- Failure analysis for incorrect selections

**Metrics provided**:
- Selection accuracy percentage (% of tests where correct category was selected)
- Average number of candidate categories
- Average processing time
- Detailed failure analysis

### 3. Pipeline Accuracy Test (`integration/test_pipeline_accuracy.py`)

**Purpose**: Evaluates the complete end-to-end classification pipeline by running the full `classify()` method on all test cases.

**What it tests**:
- Complete classification pipeline (narrowing + selection)
- Overall system accuracy
- Performance breakdown (narrowing vs selection time)
- Failure analysis by stage (narrowing vs selection errors)

**Metrics provided**:
- Overall pipeline accuracy percentage
- Performance breakdown by stage
- Failure categorization (narrowing vs selection failures)
- Test type analysis (LLM-generated vs human-generated test cases)

### 4. Test Cases (`data/test_cases.py`)

**Content**: 25 comprehensive test cases covering:
- All categories in the current `categories.txt` (30 categories)
- Different hierarchy levels (appliances → parts → specific parts)
- Realistic product descriptions with model numbers
- Challenging classification scenarios

**Categories covered**:
- French Door Refrigerators
- Built-in/Countertop/Portable/Commercial Dishwashers
- Garbage Disposals
- Various appliance parts (filters, belts, knobs, etc.)

## Running Tests

### Using the Test Runner

```bash
# Run narrowing accuracy test (default)
cd tests
python run_tests.py

# Run specific test types
python run_tests.py --narrowing-accuracy
python run_tests.py --selection-accuracy
python run_tests.py --pipeline-accuracy

# Run all tests
python run_tests.py --all
```

### Running Tests Directly

```bash
# Run tests directly
cd tests/integration
python test_narrowing_accuracy.py
python test_selection_accuracy.py
python test_pipeline_accuracy.py
```

## Test Output Example

```
🚀 Category Narrowing Accuracy Test
============================================================
This test evaluates how often the correct category is included
in the narrowed results for different narrowing strategies.

Loaded 30 categories for testing
Running tests with max_narrowed_categories = 5
------------------------------------------------------------

🧪 Testing Embedding Strategy
==================================================
 1. ✅ Samsung Counter-Depth 17.5-cu ft 3-Door Smart Compatible...
    Expected: /Appliances/Refrigerators/French Door Refrigerators
    Narrowed to 5 categories (245.3ms)

 2. ❌ Whirlpool Dishwasher Upper Dish Rack Assembly W10350375...
    Expected: /Appliances/Appliance Parts/Dishwasher Parts
    Narrowed to 5 categories (198.7ms)
    ⚠️  Correct category NOT found in narrowed results!

📈 Strategy Comparison
============================================================
Strategy        Accuracy   Avg Categories    Avg Time (ms)
------------------------------------------------------------
Embedding          84.0%           5.0             220.5
Hybrid             92.0%           4.8             340.2

🏆 Best Accuracy: Hybrid (92.0%)
⚡ Fastest: Embedding (220.5ms avg)

📁 Results saved to: tests/results/narrowing/narrowing_accuracy_20250916_143022.json
   Use this file for detailed analysis and comparison with other test runs.
```

## JSON Output and Result Analysis

### Automatic JSON Output

All integration tests now automatically save detailed results to JSON files with timestamps:

- **Narrowing tests**: `tests/results/narrowing/narrowing_accuracy_YYYYMMDD_HHMMSS.json`
- **Selection tests**: `tests/results/selection/selection_accuracy_YYYYMMDD_HHMMSS.json`

### JSON Structure

**Narrowing Results**:
```json
{
  "test_info": {
    "test_type": "narrowing_accuracy",
    "timestamp": "2025-09-16T14:30:22.123456",
    "total_categories": 30,
    "max_narrowed_categories": 5,
    "total_test_cases": 25
  },
  "strategies": {
    "Embedding": {
      "strategy_name": "Embedding",
      "total_tests": 25,
      "correct_found": 21,
      "accuracy_percent": 84.0,
      "avg_narrowed_count": 5.0,
      "avg_processing_time_ms": 220.5,
      "results": [...]
    }
  }
}
```

**Selection Results**:
```json
{
  "test_info": {
    "test_type": "selection_accuracy",
    "timestamp": "2025-09-16T14:30:22.123456",
    "total_categories": 30,
    "total_test_cases": 25
  },
  "results": {
    "total_tests": 25,
    "correct_selections": 23,
    "accuracy_percent": 92.0,
    "avg_candidate_count": 4.8,
    "avg_processing_time_ms": 340.2,
    "individual_results": [...]
  }
}
```

**Pipeline Results**:
```json
{
  "test_info": {
    "test_type": "pipeline_accuracy",
    "timestamp": "2025-09-16T14:30:22.123456",
    "total_test_cases": 25,
    "narrowing_strategy": "hybrid",
    "vector_store_enabled": true
  },
  "results": {
    "total_tests": 25,
    "correct_classifications": 23,
    "accuracy_percent": 92.0,
    "avg_narrowed_count": 4.8,
    "avg_processing_time_ms": 520.5,
    "avg_narrowing_time_ms": 340.2,
    "avg_selection_time_ms": 180.3,
    "individual_results": [...]
  }
}
```

### Comparing Results Across Runs

Use the `compare_results.py` utility to analyze changes between test runs:

```bash
# List all available result files
python tests/compare_results.py --list-results

# Compare two narrowing accuracy results
python tests/compare_results.py --narrowing file1.json file2.json

# Compare two selection accuracy results
python tests/compare_results.py --selection file1.json file2.json

# Compare two pipeline accuracy results
python tests/compare_results.py --pipeline file1.json file2.json
```

**Example comparison output**:
```
🧪 Comparing Narrowing Accuracy Results
==================================================
📅 File 1: narrowing_accuracy_20250916_140000.json (2025-09-16T14:00:00)
📅 File 2: narrowing_accuracy_20250916_143000.json (2025-09-16T14:30:00)

Strategy        File 1 Accuracy File 2 Accuracy Change    
------------------------------------------------------------
Embedding             82.0%           84.0%     🟢 +2.0%
Hybrid                90.0%           92.0%     🟢 +2.0%

⏱️  Timing Comparison:
------------------------------
Embedding: 235.2ms → 220.5ms (🟢 -14.7ms)
Hybrid: 355.1ms → 340.2ms (🟢 -14.9ms)
```

### Benefits of JSON Output

1. **Detailed Analysis**: Complete test case results with all narrowed/selected categories
2. **Performance Tracking**: Track accuracy and timing improvements over time
3. **Regression Detection**: Quickly identify when changes hurt performance
4. **Data-Driven Decisions**: Use historical data to make informed optimization choices
5. **Reproducibility**: Full context for understanding test conditions and results

## Adding New Tests

### Integration Tests
1. Create new test file in `tests/integration/`
2. Add import path handling for src modules
3. Follow the existing pattern for test structure
4. Update `run_tests.py` to include the new test

### Test Cases
1. Add new test cases to `tests/data/test_cases.py`
2. Follow the `TestCase` TypedDict structure
3. Ensure realistic product descriptions
4. Cover edge cases and challenging scenarios

## Future Test Types

**Planned test additions**:
- Unit tests for individual components
- Performance benchmarks
- Load testing
- Category hierarchy validation tests
- BAML integration tests
- Regression testing framework

## Dependencies

Tests require the same dependencies as the main application:
- All modules from `src/`
- OpenAI API access (for embedding tests)
- BAML client (for LLM-based tests)

Make sure to set up your `.env` file with required API keys before running tests.
