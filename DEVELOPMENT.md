# MHPMR Development Setup Guide

## Project Structure

```
mhpmr-dev/
├── src/mhpmr/              # Main package source code
│   ├── __init__.py         # Package initialization
│   ├── config.py           # Configuration management
│   ├── utils.py            # Utility functions
│   ├── predictor.py        # Main predictor class
│   ├── video.py            # Video processing
│   ├── export.py           # Mesh export utilities
│   ├── models/             # Model components
│   │   └── __init__.py     # ViT, VertexHead, GNNRefiner
│   ├── datasets/           # Data loading utilities
│   │   └── __init__.py     # Dataset classes
│   ├── training/           # Training module
│   │   └── __init__.py     # Trainer, losses, metrics
│   ├── visualization/      # Visualization tools
│   │   └── __init__.py     # PoseVisualizer, MeshRenderer
│   └── analysis/           # Analysis utilities
│       └── __init__.py     # BodyShapeAnalyzer, metrics
│
├── tests/                  # Test suite
│   ├── conftest.py        # Pytest configuration and fixtures
│   ├── test_predictor.py  # Predictor tests
│   ├── test_models.py     # Model component tests
│   └── test_utils.py      # Utility tests
│
├── notebooks/              # Jupyter notebooks
│   ├── 01_getting_started.ipynb
│   ├── 02_video_processing.ipynb
│   └── ...
│
├── data/                   # Data directories
│   ├── models/            # Character models
│   ├── dataset/           # MESH-DETAIL dataset
│   ├── pretrained/        # Pre-trained weights
│   └── examples/          # Example images
│
├── tools/                  # Command-line tools
│   ├── train_model.py    # Training script
│   ├── benchmark.py      # Benchmarking
│   └── ...
│
├── setup.py               # Package setup
├── pyproject.toml        # Project configuration
├── requirements.txt      # Dependencies
└── README.md             # Project documentation
```

## Installation

### 1. Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n mhpmr python=3.9
conda activate mhpmr
```

### 2. Install Dependencies

```bash
# Install package in development mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"

# Or install from requirements.txt
pip install -r requirements.txt
```

### 3. Verify Installation

```python
import mhpmr
print(mhpmr.__version__)
print(mhpmr.PoseAndMeshPredictor.list_available_models())
```

## Development Workflow

### Code Quality

```bash
# Format code
black src/ tests/
isort src/ tests/

# Check linting
flake8 src/ tests/

# Type checking
mypy src/
```

### Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_predictor.py -v

# Run with coverage
pytest --cov=mhpmr tests/

# Run specific test
pytest tests/test_predictor.py::TestPoseAndMeshPredictor::test_initialization -v
```

### Configuration

Create a `config.yaml` file for your settings:

```yaml
model:
  vit_model: base
  vit_pretrained: true
  input_size: 384
  use_occlusion_handling: true

training:
  batch_size: 32
  learning_rate: 0.0001
  num_epochs: 100

data:
  dataset_root: ./data/dataset
  num_workers: 4

inference:
  device: cuda:0
  batch_size: 1
```

Then load it in your code:

```python
from mhpmr.config import Config

config = Config.from_yaml('config.yaml')
```

## Module Implementation Order

The following is the recommended order for implementing modules:

1. **Core Utilities** (✓ Done)
   - `utils.py` - Basic utilities
   - `config.py` - Configuration management

2. **Model Components** (Partial)
   - `models/__init__.py` - ViT, VertexHead, GNN
   - Implementation needed: ViT layers, attention mechanisms

3. **Data Pipeline** (Partial)
   - `datasets/__init__.py` - Data loaders
   - Implementation needed: TFDataset creation, preprocessing

4. **Training Module** (Partial)
   - `training/__init__.py` - Trainer, losses, metrics
   - Implementation needed: Training loop, loss computation

5. **Main Predictor** (Partial)
   - `predictor.py` - Core inference
   - Implementation needed: Model integration, inference pipeline

6. **Post-Processing**
   - `analysis/__init__.py` - Metrics and analysis
   - `visualization/__init__.py` - Rendering and visualization
   - `export.py` - Mesh export formats

## Key Files Reference

- **Configuration**: `src/mhpmr/config.py`
- **Utilities**: `src/mhpmr/utils.py`
- **Models**: `src/mhpmr/models/__init__.py`
- **Data**: `src/mhpmr/datasets/__init__.py`
- **Training**: `src/mhpmr/training/__init__.py`
- **Tests**: `tests/`

## Git Workflow

```bash
# Create feature branch
git checkout -b feature/component-name

# Make changes and commit
git add .
git commit -m "Implement component functionality"

# Push and create PR
git push origin feature/component-name
```

## Common Issues & Solutions

### ImportError: No module named 'mhpmr'
Solution: Install package in development mode:
```bash
pip install -e .
```

### Missing dependencies
Solution: Install all dependencies:
```bash
pip install -r requirements.txt
```

### CUDA/GPU issues
Solution: Set device to CPU for testing:
```python
predictor = PoseAndMeshPredictor(device='cpu')
```

## Next Steps

1. Review [IMPLEMENT.md](../IMPLEMENT.md) for detailed component specifications
2. Review [RESEARCH_PAPER.md](../RESEARCH_PAPER.md) for algorithm details
3. Start implementing model components following the architecture in IMPLEMENT.md
4. Add unit tests for each component in `tests/`
5. Create notebooks demonstrating usage

## Documentation

- `README.md` - User-facing documentation
- `IMPLEMENT.md` - Implementation specifications
- `RESEARCH_PAPER.md` - Research details and benchmarks
- Docstrings in source code - Detailed function/class documentation

## Support

For implementation questions, refer to:
- `IMPLEMENT.md` for component specifications
- `RESEARCH_PAPER.md` for algorithmic details
- Existing docstrings in source code
