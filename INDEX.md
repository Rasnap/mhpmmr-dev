# MHPMR Documentation Index

## 📖 Documentation Map

This project has comprehensive documentation organized by purpose. Start with the guide that matches your needs:

### 🎯 Getting Started
- **[QUICKSTART.md](QUICKSTART.md)** ⭐ **START HERE**
  - Quick reference for all commands
  - Module structure overview
  - Common issues and solutions
  - ~5 min read

### 🛠️ Setup & Development
- **[DEVELOPMENT.md](DEVELOPMENT.md)** - Development workflow
  - Detailed installation instructions
  - Project structure explanation
  - Code quality guidelines
  - Git workflow recommendations
  - ~15 min read

- **[setup_dev.py](setup_dev.py)** - Interactive setup script
  - Run: `python3 setup_dev.py`
  - Automatically creates directories
  - Optionally installs dependencies

### 📋 Implementation
- **[IMPLEMENT.md](IMPLEMENT.md)** - Implementation specifications (Existing)
  - Detailed component architecture
  - API specifications
  - Implementation guidelines
  - ~30 min read

- **[RESEARCH_PAPER.md](RESEARCH_PAPER.md)** - Research details (Existing)
  - Algorithm explanations
  - Performance benchmarks
  - Comparison with other methods
  - ~45 min read

### 📊 Project Information
- **[README.md](README.md)** - Main project overview (Existing)
  - Project description
  - Features overview
  - Quick start examples
  - API reference

- **[SETUP_REPORT.md](SETUP_REPORT.md)** - Setup completion report
  - What was created
  - Project statistics
  - Implementation checklist
  - Next steps

## 🏗️ Project Structure

```
mhpmr-dev/
├── 📄 QUICKSTART.md         ← Start here for quick reference
├── 📄 DEVELOPMENT.md        ← Development setup guide  
├── 📄 IMPLEMENT.md          ← Implementation specifications
├── 📄 RESEARCH_PAPER.md     ← Research details
├── 📄 README.md             ← User guide
├── 📄 SETUP_REPORT.md       ← Setup completion info
│
├── 🔧 Configuration Files
│   ├── setup.py             ← Package setup (setuptools)
│   ├── pyproject.toml       ← Modern Python config
│   ├── requirements.txt     ← Dependencies
│   ├── config.example.yaml  ← Example configuration
│   ├── Makefile             ← Development tasks
│   └── .gitignore           ← Git ignore rules
│
├── 📦 Source Code (src/mhpmr/)
│   ├── __init__.py          ← Package initialization
│   ├── config.py            ← Configuration management
│   ├── utils.py             ← Utility functions
│   ├── predictor.py         ← Main predictor class
│   ├── video.py             ← Video processing
│   ├── export.py            ← Mesh export
│   ├── models/              ← Model components
│   ├── datasets/            ← Data loading
│   ├── training/            ← Training pipeline
│   ├── visualization/       ← Rendering
│   └── analysis/            ← Analysis tools
│
├── 🧪 Tests (tests/)
│   ├── conftest.py          ← Fixtures
│   ├── test_predictor.py    ← Predictor tests
│   ├── test_models.py       ← Model tests
│   └── test_utils.py        ← Utility tests
│
└── 📂 Data Directories
    ├── data/models/         ← Character models
    ├── data/dataset/        ← MESH-DETAIL dataset
    ├── data/pretrained/     ← Pre-trained weights
    └── data/examples/       ← Example images
```

## 🚀 Quick Commands Reference

```bash
# Installation
pip install -r requirements.txt
pip install -e ".[dev]"

# Testing
pytest tests/ -v
make test
make test-cov

# Code Quality
make format      # Format code
make lint        # Check style
make type-check  # Type checking
make clean       # Clean up

# Full workflow
make format lint type-check test
```

## 📋 Reading Guide by Role

### 👨‍💻 If you're implementing core components:
1. Read [QUICKSTART.md](QUICKSTART.md) (5 min)
2. Review [DEVELOP.md](DEVELOPMENT.md) (15 min)
3. Study [IMPLEMENT.md](IMPLEMENT.md) (30 min)
4. Check [RESEARCH_PAPER.md](RESEARCH_PAPER.md) for algorithm details
5. Start implementing following the structure in `src/mhpmr/`

### 🔬 If you're understanding the research:
1. Read [README.md](README.md) for project overview
2. Study [RESEARCH_PAPER.md](RESEARCH_PAPER.md) for details
3. Review [IMPLEMENT.md](IMPLEMENT.md) for algorithm specifics
4. Check benchmarks and comparisons in RESEARCH_PAPER.md

### 🏗️ If you're setting up the project:
1. Run `python3 setup_dev.py` for interactive setup
2. Read [DEVELOPMENT.md](DEVELOPMENT.md) for details
3. Install: `pip install -r requirements.txt`
4. Install dev: `pip install -e ".[dev]"`
5. Verify: `pytest tests/ -v`

### 📖 If you just want quick reference:
1. Use [QUICKSTART.md](QUICKSTART.md) - everything condensed
2. Use Makefile: `make help`
3. Check command line: `pytest --help`, `make --help`

## 🔗 Key Files by Category

### Configuration & Setup
- `setup.py` - Package metadata
- `pyproject.toml` - Build configuration  
- `requirements.txt` - Dependencies
- `config.example.yaml` - Configuration template
- `.gitignore` - Git rules

### Source Code Entry Points
- `src/mhpmr/__init__.py` - Main package imports
- `src/mhpmr/config.py` - Configuration management
- `src/mhpmr/utils.py` - Utility functions
- `src/mhpmr/predictor.py` - Main inference class
- `src/mhpmr/models/` - Model components

### Testing
- `tests/conftest.py` - Pytest setup
- `tests/test_*.py` - Test files

### Scripts & Tools
- `setup_dev.py` - Setup script
- `Makefile` - Build automation

## 📚 Documentation by Topic

### Configuration
- See: [config.py](src/mhpmr/config.py)
- Reference: [QUICKSTART.md](QUICKSTART.md) "Configuration" section
- Example: [config.example.yaml](config.example.yaml)

### Model Architecture
- See: [IMPLEMENT.md](IMPLEMENT.md) "Core Components" section
- Code: [models/__init__.py](src/mhpmr/models/__init__.py)
- Research: [RESEARCH_PAPER.md](RESEARCH_PAPER.md)

### Data Pipeline
- See: [IMPLEMENT.md](IMPLEMENT.md) "Data Pipeline" section
- Code: [datasets/__init__.py](src/mhpmr/datasets/__init__.py)

### Training
- See: [IMPLEMENT.md](IMPLEMENT.md) "Model Training" section
- Code: [training/__init__.py](src/mhpmr/training/__init__.py)

### Inference
- See: [IMPLEMENT.md](IMPLEMENT.md) Implementation details
- Code: [predictor.py](src/mhpmr/predictor.py)
- Examples: Notebooks in `notebooks/`

## 💡 Tips for Success

1. **Start Small**: Begin with QUICKSTART.md, don't try to understand everything at once
2. **Follow Structure**: Use the existing skeleton in `src/mhpmr/` for implementation
3. **Write Tests**: Add tests for each component you implement
4. **Use Makefile**: Let `make format`, `make lint` keep code consistent
5. **Read Specs**: Follow IMPLEMENT.md for exact specifications
6. **Check Research**: RESEARCH_PAPER.md has algorithm details

## ⚡ Common Workflows

### Setting up for development
```bash
python3 setup_dev.py        # Interactive setup
pip install -r requirements.txt
pip install -e ".[dev]"
pytest tests/ -v            # Verify installation
```

### Implementing a component
```bash
# 1. Review specifications
cat IMPLEMENT.md | grep -A 50 "Component Name"

# 2. Edit the component file
# 3. Add tests
# 4. Format and test
make format
make lint
make test

# 5. Commit
git add .
git commit -m "Implement component"
```

### Code review before commit
```bash
make format      # Format
make lint        # Style check
make type-check  # Type safety
make test        # Run tests
```

## 📞 Need Help?

- **Installation issues**: See [DEVELOPMENT.md](DEVELOPMENT.md) "Common Issues & Solutions"
- **Implementation questions**: Check [IMPLEMENT.md](IMPLEMENT.md)
- **Algorithm details**: Read [RESEARCH_PAPER.md](RESEARCH_PAPER.md)
- **Quick reference**: Use [QUICKSTART.md](QUICKSTART.md)
- **Code examples**: Check docstrings in source files

## ✅ Verification Checklist

After setup, verify everything works:

```bash
# Check Python version
python3 --version  # Should be 3.8+

# Check project structure
ls src/mhpmr/      # Should show modules

# Check dependencies can be listed
cat requirements.txt

# Try loading package
python3 -c "import sys; sys.path.insert(0, 'src'); from mhpmr import __version__; print(f'MHPMR v{__version__}')"

# Run basic test
pytest tests/test_utils.py -v  # Should pass
```

---

**Last Updated**: 2026-11-26  
**Project Version**: 0.1.0  
**Framework**: TensorFlow 2.11+  
**Status**: Setup Complete - Ready for Implementation

For questions about implementation, refer to the appropriate documentation above or the docstrings in the source code.
