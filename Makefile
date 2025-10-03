.PHONY: run server ui test tiny-bench report lint fmt analyze

run:
	python scripts/run_demo.py

server:
	uvicorn ui.server:app --host 0.0.0.0 --port 8000

ui:
	streamlit run ui/streamlit_app.py --server.port 8501

test:
	pytest -q

tiny-bench:
	ragx-run --config experiments/configs/baseline.yaml --repeat 1 --override num_queries=3

report:
	ragx-report --run-dir $$(ls -td runs/* | head -1) --format md --title "RAGX Benchmark"

analyze:
	ragx-analyze --run-dir $$(ls -td runs/* | head -1) --redact

lint:
	ruff check .
	mypy .
	flake8 || true

fmt:
	black .
	isort .
