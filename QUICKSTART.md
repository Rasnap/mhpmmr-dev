# MHPMR Quick Reference

## Project Status: ✅ Setup Complete

### Core Files Location
```
Project Root: /Users/sarithd/research/mhpmr-dev/

Source Code:          src/mhpmr/
Tests:                tests/
Configuration:        config.example.yaml
Build Config:         setup.py, pyproject.toml
Dependencies:         requirements.txt
Development Docs:     DEVELOPMENT.md
Implementation Spec:  IMPLEMENT.md
Research Details:     RESEARCH_PAPER.md
Setup Report:         SETUP_REPORT.md
```

## Quick Commands

```bash
# Install
pip install -r requirements.txt
pip install -e ".[dev]"

# Format Code
make format

# Run Tests
make test
make test-cov

# Lint & Type Check
make lint
make type-check

# Clean Up
make clean
```

## Module Structure

| Module | Purpose | Status |
|--------|---------|--------|
| `config.py` | Configuration management | ✅ Core implemented |
| `utils.py` | Utility functions | ✅ Core implemented |
| `predictor.py` | Main predictor class | 🔄 Placeholder |
| `models/` | Model components | 🔄 Placeholder |
| `datasets/` | Data loading | 🔄 Placeholder |
| `training/` | Training pipeline | 🔄 Placeholder |
| `visualization/` | Rendering & display | 🔄 Placeholder |
| `analysis/` | Metrics & analysis | 🔄 Placeholder |

## Key Classes

### Configuration
- `Config` - Main configuration manager
- `ModelConfig` - Model settings
- `TrainingConfig` - Training settings
- `InferenceConfig` - Inference settings
- `DataConfig` - Data settings

### Models
- `ViTBackbone` - Vision Transformer
- `VertexHead` - Vertex prediction
- `GNNRefiner` - Graph refinement

### Core
- `PoseAndMeshPredictor` - Main inference class
- `VideoProcessor` - Video processing
- `MeshExporter` - Export formats

### Utilities
- `AverageMeter` - Metric tracking
- Normalization/denormalization functions
- Device management

## Development Workflow

1. **Setup Environment**
   ```bash
   pip install -r requirements.txt
   pip install -e ".[dev]"
   ```

2. **Create Feature Branch**
   ```bash
   git checkout -b feature/component-name
   ```

3. **Implement Component**
   - Follow specifications in IMPLEMENT.md
   - Add docstrings and type hints
   - Create unit tests in tests/

4. **Code Quality**
   ```bash
   make format      # Format code
   make lint        # Check style
   make type-check  # Type safety
   make test        # Run tests
   ```

5. **Commit & Push**
   ```bash
   git add .
   git commit -m "Implement component"
   git push origin feature/component-name
   ```

## Configuration

Copy and customize:
```bash
cp config.example.yaml config.yaml
# Edit config.yaml with your settings
```

Load in code:
```python
from mhpmr.config import Config
config = Config.from_yaml('config.yaml')
```

## Testing

```bash
# All tests
pytest tests/ -v

# Specific file
pytest tests/test_predictor.py -v

# With coverage
pytest --cov=mhpmr tests/

# Specific test
pytest tests/test_predictor.py::TestPoseAndMeshPredictor::test_initialization -v
```

## Common Issues

| Issue | Solution |
|-------|----------|
| ImportError: No module named 'mhpmr' | `pip install -e .` |
| Missing TensorFlow | `pip install tensorflow>=2.11.0` |
| No module named 'pytest' | `pip install pytest` |
| GPU not detected | Check TensorFlow GPU setup or use `device='cpu'` |

## Implementation Priority

### Phase 1: Foundation (✅ Complete)
- Project structure
- Configuration
- Utilities
- Testing framework

### Phase 2: Models (Next)
1. Vision Transformer backbone
2. Vertex prediction head
3. GNN refinement layer

### Phase 3: Data
1. Dataset loader
2. Data preprocessing
3. Augmentation pipeline

### Phase 4: Training
1. Loss functions
2. Training loop
3. Metrics computation

### Phase 5: Inference
1. Main predictor
2. SMPL fitting
3. Occlusion handling

### Phase 6: Post-Processing
1. Visualization
2. Mesh export
3. Analysis tools

## Documentation Map

| Document | Purpose |
|----------|---------|
| README.md | User guide and overview |
| IMPLEMENT.md | Detailed implementation specs |
| RESEARCH_PAPER.md | Algorithm & research details |
| DEVELOPMENT.md | Development setup guide |
| SETUP_REPORT.md | Setup completion report |
| This file | Quick reference |

## Git Workflow Template

```bash
# Start new feature
git checkout -b feature/vit-backbone

# Make changes and commit
git add .
git commit -m "Implement ViT backbone
- Added ViTBackbone class with 4 model sizes
- Integrated with TensorFlow
- Added comprehensive tests"

# Push and create PR
git push origin feature/vit-backbone

# After review, merge to main
git checkout main
git pull origin main
git merge feature/vit-backbone
git push origin main
```

## Project Statistics

- **Python Files**: 15
- **Test Files**: 4
- **Documentation**: 6 MD files
- **Configuration**: 3 files (setup.py, pyproject.toml, requirements.txt)
- **Total Lines of Code**: ~3,500+ (with documentation)

## Useful Shortcuts

```bash
# Format all code
make format

# Run all checks (lint + type + test)
make lint type-check test

# Clean and reinstall
make clean
pip install -e ".[dev]"

# View help
make help
```

## Support Resources

- **Implementation Specs**: See `IMPLEMENT.md` sections 1-3
- **Algorithm Details**: See `RESEARCH_PAPER.md`
- **Code Examples**: See notebook templates in `notebooks/`
- **API Reference**: See docstrings in source files

---

**Last Updated**: 2026-11-26  
**Project Version**: 0.1.0  
**Framework**: TensorFlow 2.11+  
**Python**: 3.8+
