# Makefile for Handwriting Recognition Project

.PHONY: help install test clean run-demo run-web format lint type-check

help: ## Show this help message
	@echo "Handwriting Recognition Project"
	@echo "=============================="
	@echo ""
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	pip install -r requirements.txt

install-dev: ## Install development dependencies
	pip install -r requirements.txt
	pip install pytest pytest-cov black flake8 mypy

test: ## Run tests
	python -m pytest tests/ -v

test-coverage: ## Run tests with coverage
	python -m pytest tests/ --cov=src --cov-report=html --cov-report=term

format: ## Format code with black
	black src/ tests/ web_app/ cli.py demo.py

lint: ## Lint code with flake8
	flake8 src/ tests/ web_app/ cli.py demo.py

type-check: ## Type check with mypy
	mypy src/

clean: ## Clean up generated files
	rm -rf __pycache__/
	rm -rf src/__pycache__/
	rm -rf tests/__pycache__/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf output/*
	rm -rf data/samples/*
	rm -rf data/demo_*

run-demo: ## Run the demo script
	python demo.py

run-web: ## Start the web interface
	streamlit run web_app/app.py

generate-sample: ## Generate a sample image
	python cli.py generate-sample

recognize: ## Recognize text from an image (usage: make recognize IMAGE=path/to/image.jpg)
	python cli.py recognize $(IMAGE)

batch-process: ## Process images in batch (usage: make batch-process DIR=path/to/images)
	python cli.py batch-process $(DIR)

model-info: ## Show model information
	python cli.py model-info

setup: install ## Setup the project
	mkdir -p data models output config
	@echo "Project setup complete!"

all-checks: format lint type-check test ## Run all code quality checks

ci: install-dev all-checks ## Run CI pipeline locally
