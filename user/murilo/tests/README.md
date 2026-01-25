# Test Organization

Tests are organized by experiment number for better modularity and maintainability.

**Current Status**: 56 tests across 6 test modules for Experiment 001

## Structure

```
tests/
├── pytest.ini                              # Pytest configuration
├── __init__.py                             # Tests package marker
├── 001-birdset-bioacoustics-benchmark/     # Experiment 001 tests (56 tests)
│   ├── test_environment.py                 # GPU, CUDA, PyTorch validation (6 tests)
│   ├── test_models.py                      # bioacoustics-model-zoo tests (12 tests)
│   ├── test_classifiers.py                 # PyTorch Lightning model tests (6 tests)
│   ├── test_datasets.py                    # BirdSet dataset loading tests (6 tests)
│   ├── test_data_pipeline.py               # Data loading and preprocessing (16 tests)
│   └── test_inference.py                   # Direct inference workflow (6 tests)
├── 002-next-experiment/                    # Future experiment tests
└── common/                                 # Shared test utilities (future)
    ├── fixtures.py                         # Common pytest fixtures
    └── utils.py                            # Test helper functions
```

## Test Module Descriptions

### Experiment 001: BirdSet Bioacoustics Benchmark

**test_environment.py** (6 tests)
- GPU availability and CUDA validation
- PyTorch and device configuration
- Environment setup verification

**test_models.py** (12 tests)
- bioacoustics-model-zoo installation and model availability
- BirdNET, Perch, BirdSetConvNeXT, BirdSetEfficientNetB1 loading
- Model prediction API validation
- Embedding extraction (BirdNET, Perch)

**test_classifiers.py** (6 tests)
- PyTorch Lightning LinearProbe model
- Multi-label classification training
- Metric computation

**test_datasets.py** (6 tests)
- BirdSet dataset loading from HuggingFace
- Dataset codes and splits validation
- Basic dataset structure verification

**test_data_pipeline.py** (16 tests) ⭐ NEW
- Audio resampling (upsampling, downsampling)
- Padding and cropping to target duration
- PyTorch Dataset wrapper (BirdSetDataset)
- Multi-hot label encoding from indices
- DataLoader batching and iteration
- Dataset-agnostic design with constant class counts

**test_inference.py** (6 tests) ⭐ NEW
- Metric computation (cmAP, F1, Top-K accuracy)
- Model prediction API with batching
- End-to-end inference workflow
- Output validation (metrics.json, predictions.npz)

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
pytest tests/001-birdset-bioacoustics-benchmark/test_data_pipeline.py
pytest tests/001-birdset-bioacoustics-benchmark/test_inference.py
```

**Run specific test class or function:**
```bash
# Run all tests in a class
pytest tests/001-birdset-bioacoustics-benchmark/test_data_pipeline.py::TestBirdSetDataset

# Run a specific test function
pytest tests/001-birdset-bioacoustics-benchmark/test_inference.py::TestInferenceWorkflow::test_inference_workflow_small_subset
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
  - Model loading and inference tests
  - Dataset loading from HuggingFace
  - Multi-sample processing tests
- `@pytest.mark.download` - Tests that download data from external sources
  - BirdSet dataset loading
  - Model weight downloads

## Test Coverage

**Phase 1 - Environment Setup**: ✅ Complete
- All dependencies validated
- GPU/CUDA availability checked
- Model loading from bioacoustics-model-zoo verified

**Phase 2A - Data Pipeline & Inference**: ✅ Complete
- Audio preprocessing pipeline (resampling, padding, cropping)
- PyTorch Dataset and DataLoader
- Multi-hot label encoding
- Direct inference workflow with BirdSet-native models
- Metric computation (cmAP, F1, Top-K)

**Phase 2B - Transfer Learning**: ⬜ Pending
- Embedding extraction (to be tested)
- Linear probe training (to be tested)
- Taxonomy mapping (to be tested)

**Fast Tests** (no markers): ~10 tests, run in <5 seconds
**Slow Tests** (@pytest.mark.slow): ~46 tests, require model/dataset loading

## Adding Tests for New Experiments

1. Create a new directory: `tests/XXX-experiment-name/`
2. Add `__init__.py` with experiment description
3. Create test files following naming convention: `test_*.py`
4. Use existing markers or add new ones in `pytest.ini`
5. Update this README with test counts and descriptions

## Quick Reference

**Fast validation** (skip slow tests):
```bash
# Fast tests only (~10 tests, <5s)
pytest tests/001-birdset-bioacoustics-benchmark/ -m "not slow"
```

**Full test suite** (all 56 tests, ~2-5 minutes):
```bash
# All tests including model/dataset loading
pytest tests/001-birdset-bioacoustics-benchmark/
```

**Test specific phase**:
```bash
# Phase 2A.1 - Data pipeline
pytest tests/001-birdset-bioacoustics-benchmark/test_data_pipeline.py

# Phase 2A.2 - Inference workflow
pytest tests/001-birdset-bioacoustics-benchmark/test_inference.py
```

**Coverage report**:
```bash
# Run with coverage (configured in pyproject.toml)
pytest tests/001-birdset-bioacoustics-benchmark/ --cov=birdset_benchmark --cov-report=term-missing
```

**Verbose output** (see test names as they run):
```bash
pytest tests/001-birdset-bioacoustics-benchmark/ -v
```

**Expected Test Times** (on GPU node):
- Fast tests only: ~5 seconds
- Data pipeline tests: ~30-60 seconds
- Inference workflow tests: ~60-120 seconds
- Full test suite: ~2-5 minutes

## Best Practices

**Test Organization**:
- Keep tests focused and independent
- Use descriptive test names (what is being tested)
- Group related tests in classes (e.g., `TestBirdSetDataset`, `TestInferenceWorkflow`)
- Document complex test setups with comments

**Performance**:
- Mark slow or download-dependent tests with `@pytest.mark.slow`
- Use small subsets for integration tests (e.g., `subset_size=5-10`)
- Test on minimal data before scaling to full datasets

**Validation**:
- Test edge cases (empty inputs, zero predictions, boundary conditions)
- Validate output shapes and dtypes
- Check metric ranges (e.g., 0.0 <= cmAP <= 1.0)
- Verify file outputs exist and have correct structure

**Examples from Current Tests**:
```python
# Good: Test with small subset
hf_dataset = load_birdset_dataset("HSN", split="test_5s", subset_size=10)

# Good: Validate output shapes
assert y_pred.shape == (n_samples, num_classes)
assert 0.0 <= metrics["cmap"] <= 1.0

# Good: Test edge cases
def test_compute_metrics_zero_predictions():
    """Test metrics with all-zero predictions."""
    y_pred = np.zeros((n_samples, num_classes))
    metrics = compute_metrics(y_true, y_pred, num_classes)
    assert metrics["cmap"] >= 0.0  # Should handle gracefully
```
