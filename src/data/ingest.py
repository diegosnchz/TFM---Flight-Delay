"""
ingest.py - Carga y validacion de datos crudos de vuelos.

Dataset utilizado: "Flight Delay and Cancellation Dataset (2019-2023)"
(robikscube / Bureau of Transportation Statistics, via Kaggle).
3 millones de vuelos domesticos de EE.UU. con retraso en minutos incluido.

Flujo:
1. Detectar y cargar el archivo CSV desde data/raw/
2. Renombrar columnas al schema interno del proyecto
3. Filtrar solo las 4 LCCs objetivo (WN, NK, F9, G4)
4. Registrar estadisticas de filtrado
5. Guardar en data/processed/flights_raw_filtered.parquet

Ejecutar con: python -m src.data.ingest
"""

from __future__ import annotations

import sys

import pandas as pd

from src.config import (
    DATA_RAW_DIR,
    FLIGHTS_RAW_FILTERED,
    LOW_COST_AIRLINES,
    LOW_COST_IATA_CODES,
    setup_logging,
)

logger = setup_logging(__name__)

# ---------------------------------------------------------------------------
# Mapeo de columnas: nombre original del CSV -> nombre interno del proyecto
# ---------------------------------------------------------------------------
BTS_COLUMN_MAP = {
    "FL_DATE":               "flight_date",      # Fecha del vuelo (YYYY-MM-DD)
    "AIRLINE_CODE":          "airline_code",      # Codigo IATA de la aerolinea
    "AIRLINE":               "airline_name",      # Nombre completo de la aerolinea
    "FL_NUMBER":             "flight_number",     # Numero de vuelo
    "ORIGIN":                "origin",            # IATA aeropuerto origen
    "DEST":                  "destination",       # IATA aeropuerto destino
    "ORIGIN_CITY":           "origin_city",
    "DEST_CITY":             "dest_city",
    "CRS_DEP_TIME":          "sched_dep_hhmm",   # Hora salida programada (HHMM)
    "DEP_TIME":              "actual_dep_hhmm",  # Hora salida real (HHMM)
    "DEP_DELAY":             "dep_delay_minutes",
    "CRS_ARR_TIME":          "sched_arr_hhmm",   # Hora llegada programada (HHMM)
    "ARR_TIME":              "actual_arr_hhmm",  # Hora llegada real (HHMM)
    "ARR_DELAY":             "delay_minutes",     # Retraso llegada en minutos (USAR ESTO)
    "CANCELLED":             "cancelled",
    "CANCELLATION_CODE":     "cancellation_code",
    "DIVERTED":              "diverted",
    "DISTANCE":              "distance_miles",   # Distancia en millas (convertir a km)
    "DELAY_DUE_CARRIER":     "delay_carrier",
    "DELAY_DUE_WEATHER":     "delay_weather",
    "DELAY_DUE_NAS":         "delay_nas",
    "DELAY_DUE_SECURITY":    "delay_security",
    "DELAY_DUE_LATE_AIRCRAFT": "delay_late_aircraft",
}

# Columnas minimas para que el pipeline funcione
REQUIRED_COLUMNS = ["airline_code", "origin", "destination", "delay_minutes", "distance_miles"]


def detect_raw_file():
    """Detecta el archivo mas grande en data/raw/ (parquet > csv)."""
    if not DATA_RAW_DIR.exists():
        return None
    for pattern in ["*.parquet", "*.csv"]:
        files = list(DATA_RAW_DIR.glob(pattern))
        if files:
            return max(files, key=lambda p: p.stat().st_size)
    return None


def load_raw_data(filepath) -> pd.DataFrame:
    """Carga el CSV o parquet crudo."""
    logger.info("Cargando datos desde: %s", filepath.name)
    suffix = filepath.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(filepath)
    elif suffix == ".csv":
        df = pd.read_csv(filepath, low_memory=False)
    else:
        raise ValueError(f"Formato no soportado: {suffix}")
    logger.info("Datos cargados: %d filas, %d columnas", len(df), len(df.columns))
    return df


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Renombra columnas del BTS al schema interno. Columnas no mapeadas se conservan."""
    rename = {orig: intern for orig, intern in BTS_COLUMN_MAP.items() if orig in df.columns}
    df = df.rename(columns=rename)
    mapped = list(rename.values())
    logger.info("Columnas renombradas: %d / %d del mapa BTS", len(rename), len(BTS_COLUMN_MAP))
    missing_required = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_required:
        raise ValueError(
            f"Faltan columnas criticas tras el renombrado: {missing_required}\n"
            f"Columnas disponibles: {list(df.columns)}"
        )
    return df


def filter_low_cost_airlines(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra para quedarse solo con las 4 LCC objetivo."""
    n_before = len(df)
    unique = df["airline_code"].value_counts()
    logger.info("Aerolineas en los datos (%d unicas):\n%s", len(unique), unique.to_string())

    df_filtered = df[df["airline_code"].isin(LOW_COST_IATA_CODES)].copy()
    n_after = len(df_filtered)

    logger.info(
        "Filtrado LCC: %d -> %d filas (%.1f%% retenido)",
        n_before, n_after, 100.0 * n_after / n_before if n_before else 0,
    )
    for code, name in LOW_COST_AIRLINES.items():
        count = (df_filtered["airline_code"] == code).sum()
        logger.info("  %s (%s): %d vuelos (%.1f%%)", name, code, count,
                    100.0 * count / n_after if n_after else 0)

    if n_after == 0:
        logger.error(
            "El filtro LCC elimino TODOS los registros. "
            "Codigos buscados: %s | Codigos presentes: %s",
            LOW_COST_IATA_CODES, sorted(df["airline_code"].unique())[:20],
        )
        sys.exit(1)

    return df_filtered


def run() -> pd.DataFrame:
    """Pipeline completo de ingesta."""
    logger.info("=" * 60)
    logger.info("FASE 1: INGESTA DE DATOS")
    logger.info("=" * 60)

    raw_file = detect_raw_file()
    if raw_file is None:
        logger.error(
            "No se encontraron datos en %s. "
            "Coloca el CSV de vuelos en esa carpeta.", DATA_RAW_DIR
        )
        sys.exit(1)

    df = load_raw_data(raw_file)
    df = rename_columns(df)
    df_filtered = filter_low_cost_airlines(df)

    FLIGHTS_RAW_FILTERED.parent.mkdir(parents=True, exist_ok=True)
    df_filtered.to_parquet(FLIGHTS_RAW_FILTERED, index=False)
    logger.info("Guardado: %s (%d filas)", FLIGHTS_RAW_FILTERED, len(df_filtered))

    return df_filtered


if __name__ == "__main__":
    run()
