"""
ingest.py - Carga y validacion de datos crudos de vuelos (ADRR/Kaggle).

Este modulo es el punto de entrada del pipeline de datos. Se encarga de:
1. Detectar y cargar el archivo de datos crudo desde data/raw/
2. Validar el schema (columnas, tipos de datos basicos)
3. Filtrar solo las 4 aerolineas low-cost objetivo
4. Registrar estadisticas de filtrado
5. Guardar el resultado en data/processed/flights_raw_filtered.parquet

IMPORTANTE: Antes de ejecutar este modulo, coloca los datos en data/raw/.
Ejecutar con: python -m src.data.ingest
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

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
# Schema esperado del ADRR de Eurocontrol
# ---------------------------------------------------------------------------
# ATENCION: Estos nombres de columna son los esperados del dataset ADRR.
# Si usas el dataset de Kaggle u otra fuente, el mapeo puede diferir.
# Ajustar COLUMN_MAP segun el formato real de tus datos.

# Columnas MINIMAS requeridas y sus tipos esperados
# Clave: nombre en el archivo fuente | Valor: nombre interno del proyecto
ADRR_COLUMN_MAP = {
    # Ajusta estos nombres segun las columnas reales de tu archivo
    # Formato tipico ADRR:
    "ADEP": "origin",                      # Aeropuerto de salida (ICAO o IATA)
    "ADES": "destination",                  # Aeropuerto de llegada
    "AC_Operator": "airline_code",          # Codigo IATA de la aerolinea
    "AC_Type": "aircraft_type",             # Tipo de aeronave
    "FILED_OFF_BLOCK_TIME": "sched_dep",    # Hora de salida programada
    "ACTUAL_OFF_BLOCK_TIME": "actual_dep",  # Hora de salida real
    "FILED_ARRIVAL_TIME": "sched_arr",      # Hora de llegada programada
    "ACTUAL_ARRIVAL_TIME": "actual_arr",    # Hora de llegada real
    "FLIGHT_DATE": "flight_date",           # Fecha del vuelo
}

# Si los datos tienen diferentes nombres de columna, anadir mapeados aqui
KAGGLE_COLUMN_MAP = {
    # Dataset de Kaggle "European Flights Dataset" (alternativa)
    # Ajustar segun el dataset especifico que uses
    "departure_airport": "origin",
    "arrival_airport": "destination",
    "airline": "airline_code",
    "aircraft_type": "aircraft_type",
    "scheduled_departure": "sched_dep",
    "actual_departure": "actual_dep",
    "scheduled_arrival": "sched_arr",
    "actual_arrival": "actual_arr",
    "date": "flight_date",
}

# Columnas minimas necesarias en el dataset (nombres internos)
REQUIRED_COLUMNS = [
    "origin",
    "destination",
    "airline_code",
    "sched_arr",
    "actual_arr",
]


def detect_raw_file() -> Optional[Path]:
    """
    Detecta automaticamente el archivo de datos crudos en data/raw/.

    Busca archivos en este orden de preferencia:
      1. Archivos .parquet
      2. Archivos .csv con mas de 1 MB
      3. Cualquier archivo .csv

    Returns:
        Ruta al archivo encontrado, o None si no hay datos.
    """
    if not DATA_RAW_DIR.exists():
        return None

    # Buscar parquet primero (mas eficiente)
    parquets = list(DATA_RAW_DIR.glob("*.parquet"))
    if parquets:
        # Si hay varios, tomar el mas grande
        return max(parquets, key=lambda p: p.stat().st_size)

    # Luego CSV
    csvs = list(DATA_RAW_DIR.glob("*.csv"))
    if csvs:
        return max(csvs, key=lambda p: p.stat().st_size)

    return None


def load_raw_data(filepath: Path) -> pd.DataFrame:
    """
    Carga el archivo de datos crudo en un DataFrame de pandas.

    Soporta formatos: .parquet, .csv

    Args:
        filepath: Ruta al archivo de datos.

    Returns:
        DataFrame con los datos crudos sin procesar.

    Raises:
        ValueError: Si el formato del archivo no esta soportado.
        FileNotFoundError: Si el archivo no existe.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {filepath}")

    suffix = filepath.suffix.lower()
    logger.info("Cargando datos desde: %s (formato: %s)", filepath.name, suffix)

    if suffix == ".parquet":
        df = pd.read_parquet(filepath)
    elif suffix == ".csv":
        # Intentar detectar el separador automaticamente
        df = pd.read_csv(filepath, sep=None, engine="python", low_memory=False)
    elif suffix in (".xlsx", ".xls"):
        df = pd.read_excel(filepath)
    else:
        raise ValueError(
            f"Formato no soportado: {suffix}. Usar .parquet, .csv o .xlsx"
        )

    logger.info("Datos cargados: %d filas, %d columnas", len(df), len(df.columns))
    logger.info("Columnas encontradas: %s", list(df.columns))
    return df


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Intenta mapear las columnas del archivo crudo a los nombres internos del
    proyecto.

    Primero intenta el mapeo ADRR, luego el de Kaggle. Si ninguno funciona,
    devuelve el DataFrame original con una advertencia para que el usuario
    ajuste el mapeo manualmente.

    Args:
        df: DataFrame con columnas en formato original.

    Returns:
        DataFrame con columnas renombradas a nombres internos.
    """
    columns_upper = {c.upper(): c for c in df.columns}

    # Intentar mapeo ADRR
    adrr_matches = {
        orig: internal
        for orig, internal in ADRR_COLUMN_MAP.items()
        if orig.upper() in columns_upper
    }
    if len(adrr_matches) >= 4:
        logger.info("Detectado formato ADRR de Eurocontrol.")
        rename_dict = {columns_upper[orig.upper()]: internal
                       for orig, internal in adrr_matches.items()}
        return df.rename(columns=rename_dict)

    # Intentar mapeo Kaggle
    kaggle_matches = {
        orig: internal
        for orig, internal in KAGGLE_COLUMN_MAP.items()
        if orig.upper() in columns_upper
    }
    if len(kaggle_matches) >= 4:
        logger.info("Detectado formato Kaggle.")
        rename_dict = {columns_upper[orig.upper()]: internal
                       for orig, internal in kaggle_matches.items()}
        return df.rename(columns=rename_dict)

    # No se pudo detectar formato
    logger.warning(
        "No se pudo detectar el formato automaticamente. "
        "Columnas actuales: %s\n"
        "Ajusta ADRR_COLUMN_MAP o KAGGLE_COLUMN_MAP en ingest.py "
        "para que coincidan con tus columnas.",
        list(df.columns),
    )
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """
    Valida que el DataFrame contiene las columnas minimas necesarias.

    Args:
        df: DataFrame a validar.

    Raises:
        ValueError: Si faltan columnas criticas.
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Faltan columnas requeridas: {missing}.\n"
            f"Columnas disponibles: {list(df.columns)}\n"
            f"Revisa el mapeo de columnas en ingest.py."
        )
    logger.info("Validacion de schema: OK. Columnas requeridas presentes.")


def filter_low_cost_airlines(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtra el DataFrame para incluir solo las 4 aerolineas low-cost objetivo.

    Las aerolineas objetivo son: Ryanair (FR), easyJet (U2), Wizz Air (W6),
    Vueling (VY).

    Args:
        df: DataFrame con columna 'airline_code'.

    Returns:
        DataFrame filtrado con solo las aerolineas objetivo.
    """
    n_before = len(df)
    unique_airlines = df["airline_code"].unique()
    logger.info(
        "Aerolineas encontradas en los datos: %d -> %s",
        len(unique_airlines),
        sorted(unique_airlines)[:20],
    )

    df_filtered = df[df["airline_code"].isin(LOW_COST_IATA_CODES)].copy()
    n_after = len(df_filtered)

    logger.info(
        "Filtrado por aerolineas low-cost: %d filas -> %d filas (%.1f%% retenido)",
        n_before,
        n_after,
        100.0 * n_after / n_before if n_before > 0 else 0,
    )

    # Log de cuantos vuelos hay por aerolinea
    for code, name in LOW_COST_AIRLINES.items():
        count = (df_filtered["airline_code"] == code).sum()
        pct = 100.0 * count / n_after if n_after > 0 else 0
        logger.info("  %s (%s): %d vuelos (%.1f%%)", name, code, count, pct)

    if n_after == 0:
        logger.warning(
            "ATENCION: El filtro de aerolineas low-cost elimino TODOS los registros. "
            "Verifica que la columna 'airline_code' contiene los codigos IATA "
            "esperados: %s",
            LOW_COST_IATA_CODES,
        )

    return df_filtered


def run() -> pd.DataFrame:
    """
    Ejecuta el pipeline completo de ingesta de datos.

    Flujo:
        1. Detectar archivo de datos en data/raw/
        2. Cargar datos
        3. Normalizar nombres de columnas
        4. Validar schema
        5. Filtrar aerolineas low-cost
        6. Guardar en data/processed/flights_raw_filtered.parquet

    Returns:
        DataFrame con los datos filtrados y guardados.

    Raises:
        FileNotFoundError: Si no hay datos en data/raw/.
        ValueError: Si los datos no tienen el schema esperado.
    """
    logger.info("=" * 60)
    logger.info("FASE 1: INGESTA DE DATOS")
    logger.info("=" * 60)

    # 1. Detectar archivo
    raw_file = detect_raw_file()
    if raw_file is None:
        logger.error(
            "No se encontraron datos en %s.\n"
            "Por favor, descarga los datos de Eurocontrol ADRR o Kaggle "
            "y colocalos en esa carpeta antes de continuar.",
            DATA_RAW_DIR,
        )
        sys.exit(1)

    # 2. Cargar datos
    df = load_raw_data(raw_file)

    # 3. Normalizar columnas
    df = normalize_column_names(df)

    # 4. Validar schema
    validate_schema(df)

    # 5. Filtrar aerolineas
    df_filtered = filter_low_cost_airlines(df)

    # 6. Guardar
    FLIGHTS_RAW_FILTERED.parent.mkdir(parents=True, exist_ok=True)
    df_filtered.to_parquet(FLIGHTS_RAW_FILTERED, index=False)
    logger.info(
        "Datos filtrados guardados en: %s (%d filas)",
        FLIGHTS_RAW_FILTERED,
        len(df_filtered),
    )

    return df_filtered


if __name__ == "__main__":
    run()
