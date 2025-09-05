# S2A Testing Suite

This directory contains the comprehensive test suite for the S2A Speech-to-Actions microservice. The tests ensure reliability, performance, and correctness of the ASR pipeline components.

## 📁 Test Structure

```
tests/
├── README.md              # This file - testing documentation
├── conftest.py            # Shared fixtures and pytest configuration
├── pytest.ini            # Pytest settings and markers
├── test_asr_service.py    # Core ASR service functionality tests
├── test_audio_utils.py    # Audio processing utilities tests  
├── test_config.py         # Configuration management tests
└── test_validation.py     # End-to-end integration validation
```

## 🧪 Test Categories

### **Unit Tests**
Fast, isolated tests that mock dependencies and focus on individual components.

- **`test_asr_service.py`** - Core ASR functionality
- **`test_audio_utils.py`** - Audio processing pipeline
- **`test_config.py`** - Configuration management

### **Integration Tests**
- **`test_validation.py`** - Full pipeline validation with real models

## 🚀 Quick Start

### Prerequisites
```bash
# Install test dependencies
pip install pytest pytest-asyncio soundfile numpy

# Optional: For coverage reporting
pip install pytest-cov
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/test_asr_service.py

# Run specific test method
pytest tests/test_asr_service.py::TestNeMoASRService::test_model_info_structure
```

## 🏷️ Test Markers

The test suite uses pytest markers to categorize tests:

### Available Markers
- **`unit`** - Fast unit tests (default for most tests)
- **`integration`** - Integration tests requiring full system
- **`slow`** - Tests that take longer to run (>30 seconds)
- **`gpu`** - Tests requiring CUDA/GPU hardware
- **`validation`** - End-to-end validation tests

### Running by Marker
```bash
# Run only unit tests (fast)
pytest -m "unit"

# Exclude slow tests
pytest -m "not slow"

# Exclude GPU-dependent tests
pytest -m "not gpu"

# Run only integration tests
pytest -m "integration"

# Combine markers
pytest -m "unit and not slow"
```

## 📋 Test Details

### **ASR Service Tests** (`test_asr_service.py`)

Tests the core transcription engine functionality:

```bash
# Key test areas:
- Model initialization and info structure
- Audio preprocessing (validation, duration checks)
- Audio chunking strategies (simple/intelligent)
- Transcription result formatting
- Text stitching from multiple chunks
- Configuration validation
```

**Example:**
```bash
# Test audio preprocessing
pytest tests/test_asr_service.py::TestNeMoASRService::test_preprocess_audio_valid_file

# Test chunking logic
pytest tests/test_asr_service.py::TestNeMoASRService::test_chunk_audio_simple_long
```

### **Audio Processing Tests** (`test_audio_utils.py`)

Tests the audio enhancement and preprocessing pipeline:

```bash
# Key test areas:
- Format conversion (WAV, MP3, FLAC)
- Audio enhancement (noise reduction, filtering)
- Voice activity detection
- Quality metrics calculation
- Normalization and preprocessing
```

**Example:**
```bash
# Test audio enhancement pipeline
pytest tests/test_audio_utils.py::TestAudioProcessor::test_enhance_audio

# Test format conversion
pytest tests/test_audio_utils.py::TestAudioProcessor::test_convert_to_wav_from_mp3
```

### **Configuration Tests** (`test_config.py`)

Tests configuration management and environment variable handling:

```bash
# Key test areas:
- Default configuration values
- Environment variable overrides
- Device auto-detection (CUDA/CPU)
- Performance settings validation
- Boolean/numeric conversions
```

**Example:**
```bash
# Test environment variable override
pytest tests/test_config.py::TestASRConfig::test_environment_variable_override

# Test device detection
pytest tests/test_config.py::TestASRConfig::test_device_auto_detection_cuda_available
```

### **Validation Tests** (`test_validation.py`)

End-to-end integration tests with real models:

```bash
# Key test areas:
- Model loading (NeMo + Whisper fallback)
- Short audio transcription (10 seconds)
- Long audio transcription (25 minutes)
- Performance benchmarking (RTF calculation)
```

**Example:**
```bash
# Run full validation suite
pytest tests/test_validation.py

# Note: Requires actual models and may need GPU
```

## 🔧 Test Configuration

### **Pytest Configuration** (`pytest.ini`)
```ini
[tool:pytest]
testpaths = tests
addopts = -v --tb=short --strict-markers
markers = 
    slow: marks tests as slow
    gpu: marks tests that require GPU
    integration: marks integration tests
timeout = 300
```

### **Test Fixtures** (`conftest.py`)

Shared test utilities and data:

- **`sample_audio_5sec`** - Standard 5-second test audio
- **`sample_audio_short`** - Below minimum duration (2 seconds)
- **`sample_audio_long`** - Long audio for chunking tests (2 minutes)
- **`noisy_audio`** - Audio with noise for enhancement testing
- **`mock_asr_service`** - Mocked ASR service for unit tests
- **`audio_processor`** - AudioProcessor instance

## 📊 Coverage and Reporting

### **Running with Coverage**
```bash
# Install coverage tools
pip install pytest-cov

# Run tests with coverage
pytest tests/ --cov=. --cov-report=html --cov-report=term

# View HTML coverage report
open htmlcov/index.html
```

### **Performance Testing**
```bash
# Run only performance-related tests
pytest tests/ -k "performance"

# Run validation tests for benchmarking
pytest tests/test_validation.py -s -v
```

## 🐛 Debugging Tests

### **Verbose Output**
```bash
# Show detailed test output
pytest tests/ -v -s

# Show local variables on failure
pytest tests/ -v -l

# Drop into debugger on failure
pytest tests/ --pdb
```

### **Running Single Test**
```bash
# Run one specific test with full output
pytest tests/test_asr_service.py::TestNeMoASRService::test_preprocess_audio_valid_file -v -s
```

## 🚫 Common Issues and Solutions

### **Missing Dependencies**
```bash
# Error: ModuleNotFoundError
pip install pytest pytest-asyncio soundfile numpy librosa torch transformers

# For audio format support
pip install pydub ffmpeg
```

### **GPU Tests Failing**
```bash
# Skip GPU tests if no CUDA available
pytest -m "not gpu"

# Or set environment to force CPU
CUDA_VISIBLE_DEVICES="" pytest tests/
```

### **Slow Tests Timing Out**
```bash
# Increase timeout or skip slow tests
pytest -m "not slow" tests/

# Or increase timeout in pytest.ini
timeout = 600  # 10 minutes
```

### **Import Errors**
```bash
# Make sure you're running from project root
cd /path/to/s2a
python -m pytest tests/

# Or add project to PYTHONPATH
PYTHONPATH=/path/to/s2a pytest tests/
```

## 🎯 Best Practices

### **Writing New Tests**

1. **Use appropriate fixtures** from `conftest.py`
2. **Mock external dependencies** (models, GPU, network)
3. **Test edge cases** (empty inputs, invalid formats, errors)
4. **Add appropriate markers** (`@pytest.mark.slow`, etc.)
5. **Keep tests isolated** and independent

### **Test Naming**
- Use descriptive test names: `test_preprocess_audio_with_invalid_format`
- Group related tests in classes: `TestAudioProcessor`
- Follow pattern: `test_[function]_[condition]_[expected_result]`

### **Running in CI/CD**
```bash
# Typical CI command (fast tests only)
pytest tests/ -m "not slow and not gpu" --tb=short
```

## 📈 Continuous Testing

### **Pre-commit Hook**
```bash
# Run fast tests before commits
pytest tests/ -m "not slow and not gpu" -x
```

### **Development Workflow**
```bash
# During development - run relevant tests
pytest tests/test_asr_service.py -v

# Before PR - run full unit test suite  
pytest tests/ -m "not slow and not integration"

# Before release - run everything
pytest tests/
```

---

For questions or issues with the test suite, check the main project documentation or open an issue in the repository.