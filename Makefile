.PHONY: dev test bench validate run

# Install tools dependencies
install-dev:
	pip install pytest pytest-cov

# Run TDD test suite
test:
	pip install -e .
	PYTHONPATH=. pytest tests/ -v

# Run the performance profiler utility
bench:
	PYTHONPATH=. python scripts/benchmark.py

# Verify the environment complies with OpenEnv specs
validate:
	openenv validate

# Run the baseline LLM inference script
inference:
	python inference.py

# CLI interactive debugger tool
debug:
	python scripts/interactive_debugger.py
