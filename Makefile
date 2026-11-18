.PHONY: install download dataset labels features evaluate report app test pipeline
install:
	pip install -r requirements.txt
download:
	python -m src.data.download_eia
	python -m src.data.download_weather
dataset:
	python -m src.data.build_dataset
labels:
	python -m src.features.heatwave
features:
	python -m src.features.build_features
evaluate:
	python -m src.evaluation.walk_forward
report:
	python -m src.evaluation.report
app:
	streamlit run src/app/streamlit_app.py
test:
	python -m pytest -q
pipeline: dataset labels features evaluate report
