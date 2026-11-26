.PHONY: help install install-dev test test-cov format lint type-check clean docs

help:
	@echo "MHPMR Development Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  install          Install package in production mode"
	@echo "  install-dev      Install package with dev dependencies"
	@echo "  test             Run test suite"
	@echo "  test-cov         Run tests with coverage report"
	@echo "  format           Format code with black and isort"
	@echo "  lint             Check code style with flake8"
	@echo "  type-check       Run type checking with mypy"
	@echo "  clean            Clean up temporary files"
	@echo "  setup-dirs       Create project directories"
	@echo "  verify           Verify installation"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=mhpmr --cov-report=html
	@echo "Coverage report generated: htmlcov/index.html"

format:
	black src/ tests/
	isort src/ tests/

lint:
	flake8 src/ tests/ --max-line-length=100

type-check:
	mypy src/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .eggs/
	rm -rf .mypy_cache/ .pytest_cache/
	rm -rf htmlcov/ .coverage

setup-dirs:
	mkdir -p src/mhpmr/{models,datasets,training,visualization,analysis}
	mkdir -p tests/fixtures
	mkdir -p notebooks
	mkdir -p data/{models,dataset,pretrained,examples}
	mkdir -p tools
	mkdir -p logs

verify:
	python3 -c "import sys; sys.path.insert(0, 'src'); from mhpmr import __version__; print(f'MHPMR v{__version__} loaded successfully')"

.DEFAULT_GOAL := help
