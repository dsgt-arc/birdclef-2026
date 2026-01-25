# Test Organization

Tests are organized by experiment number for better modularity and maintainability.

## Structure

```
tests/
├── pytest.ini                              # Pytest configuration
├── __init__.py                             # Tests package marker
├── 001-birdset-bioacoustics-benchmark/     # Experiment 001 tests
│   ├── test_environment.py                 # GPU, CUDA, PyTorch validation
│   ├── test_models.py                      # bioacoustics-model-zoo tests
│   ├── test_classifiers.py                 # PyTorch Lightning model tests
│   └── test_datasets.py                    # BirdSet dataset loading tests
├── 002-next-experiment/                    # Future experiment tests
└── common/                                 # Shared test utilities (future)
    ├── fixtures.py                         # Common pytest fixtures
    └── utils.py                            # Test helper functions
```

## Running Tests

**Run all tests:**
```bash
cd user/murilo
pytest tests/
```

**Run tests for specific experiment:**
```bash
pytest tests/001-birdset-bioacoustics-benchmark/
```

**Run specific test file:**
```bash
pytest tests/001-birdset-bioacoustics-benchmark/test_environment.py
```

**Run with markers:**
```bash
# Skip slow tests
pytest tests/ -m "not slow"

# Skip tests that download data
pytest tests/ -m "not download"

# Run only fast tests (no slow, no download)
pytest tests/ -m "not slow and not download"
```

## Test Markers

- `@pytest.mark.slow` - Tests that take significant time (>10s)
- `@pytest.mark.download` - Tests that download data from external sources

## Adding Tests for New Experiments

1. Create a new directory: `tests/XXX-experiment-name/`
2. Add `__init__.py` with experiment description
3. Create test files following naming convention: `test_*.py`
4. Use existing markers or add new ones in `pytest.ini`

## Best Practices

- Keep tests focused and independent
- Use descriptive test names (what is being tested)
- Document complex test setups with comments
- Share common fixtures in `tests/common/fixtures.py`
- Mark slow or download-dependent tests appropriately
