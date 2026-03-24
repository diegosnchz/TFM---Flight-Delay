.PHONY: setup data features eda train evaluate arbitrage figures tfm all clean

setup:
	python -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	mkdir -p data/raw data/processed data/external
	mkdir -p outputs/figures outputs/tables outputs/models
	mkdir -p docs/tfm/assets
	mkdir -p notebooks
	@echo "Setup completado. Pon los datos crudos en data/raw/ y ejecuta 'make data'."

data:
	python -m src.data.ingest
	python -m src.data.clean

features:
	python -m src.data.features

eda:
	python -m src.visualization.eda_plots

train:
	python -m src.models.train

evaluate:
	python -m src.models.evaluate

arbitrage:
	python -m src.models.predict

figures: eda evaluate arbitrage

tfm:
	python docs/tfm/generate_tfm.py

all: data features eda train evaluate arbitrage figures tfm

clean:
	rm -rf data/processed/* outputs/figures/* outputs/tables/* outputs/models/* .venv
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

# Alias utiles
.PHONY: check-data
check-data:
	@python -c "import pathlib; files = list(pathlib.Path('data/raw').iterdir()); print(f'Archivos en data/raw: {len(files)}'); [print(f'  - {f.name}') for f in files]"
