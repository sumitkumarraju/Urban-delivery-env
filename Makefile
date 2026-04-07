.PHONY: dev test bench validate run

install-dev:
	pip install pytest pytest-cov

test:
	pip install -e .
	PYTHONPATH=. pytest tests/ -v

bench:
	PYTHONPATH=. python scripts/benchmark.py

validate:
	openenv validate

inference:
	python inference.py

debug:
	python scripts/interactive_debugger.py

start:
	uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
