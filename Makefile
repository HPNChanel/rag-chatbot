.PHONY: run server ui

run:
	python scripts/run_demo.py

server:
	uvicorn ui.server:app --host 0.0.0.0 --port 8000

ui:
	streamlit run ui/streamlit_app.py --server.port 8501
