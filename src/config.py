"""
config.py - Configuracion global del proyecto TFM EU261.

Centraliza todas las constantes, rutas y parametros del proyecto para
garantizar reproducibilidad y facilitar cambios sin tocar multiples archivos.
"""

import logging
from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas del proyecto
# ---------------------------------------------------------------------------

# Raiz del proyecto (la carpeta que contiene este archivo src/)
PROJECT_ROOT = Path(__file__).parent.parent

DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_EXTERNAL_DIR = DATA_DIR / "external"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
MODELS_DIR = OUTPUTS_DIR / "models"

DOCS_DIR = PROJECT_ROOT / "docs"
TFM_DIR = DOCS_DIR / "tfm"

# ---------------------------------------------------------------------------
# Archivos de datos
# ---------------------------------------------------------------------------

# Datos crudos filtrados (resultado de ingest.py)
FLIGHTS_RAW_FILTERED = DATA_PROCESSED_DIR / "flights_raw_filtered.parquet"

# Datos limpios con features basicas (resultado de clean.py)
FLIGHTS_CLEAN = DATA_PROCESSED_DIR / "flights_clean.parquet"

# Datos con todas las features para modelado (resultado de features.py)
FLIGHTS_FEATURES = DATA_PROCESSED_DIR / "flights_features.parquet"

# Coordenadas de aeropuertos (OpenFlights airports.dat)
AIRPORTS_FILE = DATA_EXTERNAL_DIR / "airports.dat"
AIRPORTS_PARQUET = DATA_EXTERNAL_DIR / "airports.parquet"

# URL publica de OpenFlights (se descarga automaticamente si no existe)
OPENFLIGHTS_URL = (
    "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat"
)

# ---------------------------------------------------------------------------
# Aerolíneas low-cost analizadas
# ---------------------------------------------------------------------------

# NOTA: Se usa el dataset de vuelos domesticos de EE.UU. (Bureau of
# Transportation Statistics, via Kaggle) como proxy metodologico.
# Las LCC estadounidenses son analogas operativamente a las europeas:
#   WN (Southwest) ~ Ryanair  |  NK (Spirit) ~ Wizz Air
#   F9 (Frontier)  ~ easyJet  |  G4 (Allegiant) ~ Vueling
# La logica EU261 se aplica identicamente sobre los datos de retraso.
LOW_COST_AIRLINES = {
    "WN": "Southwest Airlines",
    "NK": "Spirit Airlines",
    "F9": "Frontier Airlines",
    "G4": "Allegiant Air",
}

LOW_COST_IATA_CODES = list(LOW_COST_AIRLINES.keys())

# ---------------------------------------------------------------------------
# Normativa EU261/2004
# ---------------------------------------------------------------------------

# Umbral de retraso para compensacion (en minutos)
EU261_DELAY_THRESHOLD_MINUTES = 180  # 3 horas

# Bandas de distancia y compensacion correspondiente (EUR)
EU261_COMPENSATION_BANDS = [
    (0, 1500, 250),       # < 1500 km -> 250 EUR
    (1500, 3500, 400),    # 1500-3500 km -> 400 EUR
    (3500, float("inf"), 600),  # > 3500 km -> 600 EUR
]

# ---------------------------------------------------------------------------
# Modelo de arbitraje
# ---------------------------------------------------------------------------

# Salario minimo en Espana 2025 (EUR/hora) como valor del tiempo
HOURLY_WAGE_EUR = 8.87

# Horas estimadas invertidas en un viaje de arbitraje (ida al aeropuerto,
# espera, vuelo, reclamacion)
ARBITRAGE_HOURS_INVESTED = 8.0

# Coste estimado de transporte al aeropuerto y vuelta (EUR)
TRANSPORT_COST_EUR = 25.0

# Probabilidad estimada de que una reclamacion sea denegada por
# "circunstancias extraordinarias"
P_CLAIM_DENIED = 0.30  # 30% como estimacion central

# ---------------------------------------------------------------------------
# Modelado y reproducibilidad
# ---------------------------------------------------------------------------

# Semilla aleatoria fija para reproducibilidad en TODOS los modelos
RANDOM_SEED = 42

# Proporciones del split train/val/test
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Variable objetivo
TARGET_COL = "is_eu261_eligible"

# ---------------------------------------------------------------------------
# Visualizacion
# ---------------------------------------------------------------------------

# Resolucion de figuras exportadas
FIGURE_DPI = 300

# Tamano por defecto de las figuras
FIGURE_SIZE_DEFAULT = (12, 8)

# Paleta de colores consistente en todo el proyecto
# Basada en seaborn "muted" con colores propios para aerolineas
PALETTE_MUTED = "muted"

AIRLINE_COLORS = {
    "WN": "#FFBF00",  # Amarillo Southwest
    "NK": "#FFA500",  # Naranja Spirit
    "F9": "#006747",  # Verde Frontier
    "G4": "#003DA5",  # Azul Allegiant
}

# Colores para clases (retraso / sin retraso)
COLOR_DELAYED = "#D62728"    # Rojo para vuelos retrasados
COLOR_ON_TIME = "#2CA02C"    # Verde para vuelos puntuales

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
LOG_LEVEL = logging.INFO


def setup_logging(name: str) -> logging.Logger:
    """
    Configura y devuelve un logger estandarizado para el proyecto.

    Args:
        name: Nombre del modulo que solicita el logger (usar __name__).

    Returns:
        Logger configurado con el formato estandar del proyecto.
    """
    logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVEL)
    return logging.getLogger(name)
