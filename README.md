# TFM: Analisis Predictivo de Retrasos Aereos y Arbitraje EU261

**Titulo:** Analisis predictivo de retrasos aereos en aerolineas low-cost europeas: Evaluacion de oportunidades de arbitraje regulatorio bajo la normativa EU261/2004

**Autor:** Diego Sanchez
**Master:** Master en Big Data e Inteligencia Artificial - Euroformac
**Fecha:** Junio 2026

---

## Descripcion

Este TFM construye un modelo predictivo de machine learning para identificar vuelos con alta probabilidad de retraso compensable (>=3 horas) bajo la normativa EU261/2004, y evalua si existe un Expected Value positivo en la compra estrategica de billetes de aerolineas low-cost europeas (Ryanair, easyJet, Wizz Air, Vueling).

## Estructura del Proyecto

```
├── src/
│   ├── config.py              # Configuracion global y constantes
│   ├── data/
│   │   ├── ingest.py          # Carga y filtrado de datos crudos
│   │   ├── clean.py           # Limpieza y feature engineering basico
│   │   └── features.py        # Feature engineering avanzado (sin leakage)
│   ├── models/
│   │   ├── train.py           # Entrenamiento de 4 modelos (LR, RF, XGBoost, LGBM)
│   │   ├── evaluate.py        # Evaluacion, metricas y Paradoja de Simpson
│   │   └── predict.py         # Modelo de arbitraje EU261 y Monte Carlo
│   ├── visualization/
│   │   ├── eda_plots.py       # 10 figuras del EDA
│   │   ├── model_plots.py     # Figuras de evaluacion (ROC, PR, SHAP)
│   │   └── arbitrage_plots.py # Figuras del modelo de arbitraje
│   └── utils/
│       ├── eu261.py           # Logica de compensaciones EU261
│       └── geo.py             # Calculo de distancias Haversine
├── docs/tfm/
│   └── generate_tfm.py        # Generador del documento Word
├── tests/
│   └── test_eu261.py          # Tests unitarios
├── data/
│   ├── raw/                   # Datos crudos (NO en git)
│   ├── processed/             # Datos procesados (NO en git)
│   └── external/              # Aeropuertos OpenFlights
├── outputs/
│   ├── figures/               # Graficos PNG (300 DPI)
│   ├── tables/                # Tablas CSV
│   └── models/                # Modelos serializados (NO en git)
├── requirements.txt
└── Makefile
```

## Instalacion y Uso

```bash
# 1. Setup del entorno
make setup

# 2. Coloca los datos en data/raw/ (ver Fuente de Datos)

# 3. Ejecutar el pipeline completo
make all

# O paso a paso:
make data        # Ingesta y limpieza
make features    # Feature engineering
make eda         # Analisis exploratorio
make train       # Entrenamiento de modelos
make evaluate    # Evaluacion y metricas
make arbitrage   # Modelo de arbitraje
make tfm         # Generacion del documento Word
```

## Fuente de Datos

**Principal:** Eurocontrol ADRR (Aviation Data Repository for Research)
- URL: https://www.eurocontrol.int/dashboard/aviation-data-research
- Requiere registro en OneSky Online
- Coloca el archivo descargado en `data/raw/`

**Alternativa (Kaggle):** European Flights Dataset
- Ajustar `KAGGLE_COLUMN_MAP` en `src/data/ingest.py`

## Tests

```bash
python -m pytest tests/ -v
```

## Dependencias Principales

- pandas, numpy, scipy
- scikit-learn, xgboost, lightgbm
- shap (interpretabilidad)
- matplotlib, seaborn, plotly
- python-docx (generacion del TFM)

Ver `requirements.txt` para versiones exactas.
