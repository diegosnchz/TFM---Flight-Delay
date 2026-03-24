"""
features.py - Feature engineering para el modelo predictivo de retrasos EU261.

Construye las features finales para el modelado a partir del dataset limpio.
El punto critico de este modulo es el manejo correcto del data leakage:
las tasas historicas (por aerolinea, ruta, aeropuerto) se calculan SOLO
sobre el training set y luego se aplican al test set.

Ejecutar con: python -m src.data.features
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import (
    FLIGHTS_CLEAN,
    FLIGHTS_FEATURES,
    RANDOM_SEED,
    TARGET_COL,
    TABLES_DIR,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
    setup_logging,
)

logger = setup_logging(__name__)


def create_route_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea la feature 'route' concatenando origen y destino.

    Ejemplo: 'STN-BCN' para un vuelo de Stansted a Barcelona.

    Args:
        df: DataFrame con columnas 'origin' y 'destination'.

    Returns:
        DataFrame con columna 'route' anadida.
    """
    df = df.copy()
    df["route"] = df["origin"].str.upper() + "-" + df["destination"].str.upper()
    logger.info("Rutas unicas: %d", df["route"].nunique())
    return df


def temporal_train_test_split(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Divide el dataset en train, validation y test sets.

    Estrategia preferida: split temporal (si los datos tienen rango de fechas
    suficiente). Esto es mas realista para un modelo predictivo: se entrena
    con vuelos pasados y se evalua en vuelos futuros.

    Si no hay columna temporal valida, usa split aleatorio estratificado.

    Args:
        df: DataFrame completo con la feature temporal 'year' o 'flight_date'.
        train_ratio: Proporcion de datos para entrenamiento.
        val_ratio: Proporcion de datos para validacion.
        test_ratio: Proporcion de datos para test.

    Returns:
        Tupla (df_train, df_val, df_test).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9, (
        "Los ratios deben sumar 1.0"
    )

    # Intentar split temporal
    if "year" in df.columns and df["year"].nunique() > 1:
        years_sorted = sorted(df["year"].unique())
        n_years = len(years_sorted)

        # Asignar anos a cada split
        train_end_idx = int(n_years * train_ratio)
        val_end_idx = int(n_years * (train_ratio + val_ratio))

        train_years = years_sorted[:train_end_idx]
        val_years = years_sorted[train_end_idx:val_end_idx]
        test_years = years_sorted[val_end_idx:]

        df_train = df[df["year"].isin(train_years)].copy()
        df_val = df[df["year"].isin(val_years)].copy()
        df_test = df[df["year"].isin(test_years)].copy()

        logger.info(
            "Split temporal por ano: "
            "train=%d filas (anos %s), val=%d filas (anos %s), "
            "test=%d filas (anos %s)",
            len(df_train), train_years,
            len(df_val), val_years,
            len(df_test), test_years,
        )
    else:
        # Split aleatorio estratificado
        logger.info(
            "Usando split aleatorio estratificado (no hay suficientes anos)."
        )
        from sklearn.model_selection import train_test_split

        df_train, df_temp = train_test_split(
            df,
            test_size=(val_ratio + test_ratio),
            stratify=df[TARGET_COL],
            random_state=RANDOM_SEED,
        )
        relative_val_ratio = val_ratio / (val_ratio + test_ratio)
        df_val, df_test = train_test_split(
            df_temp,
            test_size=(1 - relative_val_ratio),
            stratify=df_temp[TARGET_COL],
            random_state=RANDOM_SEED,
        )

        logger.info(
            "Split estratificado: train=%d, val=%d, test=%d",
            len(df_train), len(df_val), len(df_test),
        )

    # Verificar proporciones de la clase positiva
    for name, subset in [("train", df_train), ("val", df_val), ("test", df_test)]:
        rate = 100.0 * subset[TARGET_COL].mean()
        logger.info("  %s: %.4f%% vuelos EU261-elegibles", name, rate)

    return df_train, df_val, df_test


def calculate_historical_rates(
    df_train: pd.DataFrame,
    df_target: pd.DataFrame,
    group_col: str,
    target_col: str = TARGET_COL,
    min_samples: int = 30,
) -> pd.Series:
    """
    Calcula la tasa historica de retrasos EU261 para una columna categorica,
    usando SOLO el training set para evitar data leakage.

    Para grupos con menos de min_samples observaciones, se usa la tasa
    global del training set (suavizado por frecuencia).

    Args:
        df_train: Training set (unica fuente para calcular tasas).
        df_target: Dataset al que se aplica el mapeo (puede ser val o test).
        group_col: Columna por la que agrupar (ej: 'airline_code', 'origin').
        target_col: Variable objetivo binaria.
        min_samples: Minimo de muestras para confiar en la tasa del grupo.

    Returns:
        Serie con la tasa de retraso EU261 para cada fila de df_target.

    Note:
        Esta funcion es la implementacion correcta para evitar data leakage.
        Las tasas se calculan en train y se APLICAN en val/test.
        NUNCA calcular estas tasas sobre el dataset completo antes del split.
    """
    global_rate = df_train[target_col].mean()

    # Calcular tasa por grupo en train
    group_stats = (
        df_train.groupby(group_col)[target_col]
        .agg(["mean", "count"])
        .rename(columns={"mean": "rate", "count": "n_samples"})
    )

    # Para grupos con pocas muestras, usar la tasa global
    group_stats["smoothed_rate"] = np.where(
        group_stats["n_samples"] >= min_samples,
        group_stats["rate"],
        global_rate,
    )

    # Mapear al dataset objetivo
    rate_map = group_stats["smoothed_rate"].to_dict()
    mapped = df_target[group_col].map(rate_map)

    # Para valores desconocidos (grupos no vistos en train), usar tasa global
    n_unknown = mapped.isna().sum()
    if n_unknown > 0:
        logger.warning(
            "  '%s': %d valores desconocidos (no vistos en train) -> "
            "usando tasa global %.4f",
            group_col, n_unknown, global_rate,
        )
    mapped = mapped.fillna(global_rate)

    return mapped


def add_historical_rate_features(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Anade features de tasas historicas de retraso para aerolinea, aeropuertos
    y ruta.

    CRITICO: Las tasas se calculan en df_train y se aplican en df_val y
    df_test para evitar data leakage.

    Features creadas:
      - airline_delay_rate: tasa de retraso EU261 por aerolinea
      - origin_delay_rate: tasa de retraso EU261 por aeropuerto de origen
      - dest_delay_rate: tasa de retraso EU261 por aeropuerto de destino
      - route_delay_rate: tasa de retraso EU261 por ruta

    Args:
        df_train: Training set.
        df_val: Validation set.
        df_test: Test set.

    Returns:
        Tupla (df_train, df_val, df_test) con las nuevas features.
    """
    rate_features = {
        "airline_delay_rate": "airline_code",
        "origin_delay_rate": "origin",
        "dest_delay_rate": "destination",
        "route_delay_rate": "route",
    }

    for feature_name, group_col in rate_features.items():
        if group_col not in df_train.columns:
            logger.warning(
                "Columna '%s' no encontrada. Omitiendo feature '%s'.",
                group_col, feature_name,
            )
            continue

        logger.info("Calculando '%s' por '%s'...", feature_name, group_col)

        # Calcular y asignar para train (sobre si mismo, para consistencia)
        df_train[feature_name] = calculate_historical_rates(
            df_train, df_train, group_col
        )
        df_val[feature_name] = calculate_historical_rates(
            df_train, df_val, group_col
        )
        df_test[feature_name] = calculate_historical_rates(
            df_train, df_test, group_col
        )

    return df_train, df_val, df_test


def encode_aircraft_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea la feature de edad del avion si el dataset incluye informacion
    sobre el tipo o la fecha de fabricacion.

    Si no hay datos de edad disponibles, crea la columna con NaN para
    mantener el schema consistente.

    Args:
        df: DataFrame de vuelos.

    Returns:
        DataFrame con columna 'aircraft_age' (anos desde fabricacion).
    """
    df = df.copy()

    if "aircraft_registration_date" in df.columns:
        # Si hay fecha de registro, calcular edad
        df["aircraft_age"] = (
            pd.to_datetime("today").year
            - pd.to_datetime(df["aircraft_registration_date"], errors="coerce").dt.year
        )
        logger.info("Aircraft age calculado desde 'aircraft_registration_date'.")
    elif "aircraft_age" not in df.columns:
        # Feature no disponible en el dataset
        df["aircraft_age"] = np.nan
        logger.info(
            "Columna 'aircraft_age' no disponible en los datos. "
            "Se crea con NaN (no incluida en el modelo si es todo NaN)."
        )

    return df


def select_modeling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Selecciona las columnas finales para el modelado.

    Incluye solo las features que se usaran en el modelo, descartando
    columnas intermedias, identificadores y datos que causarian leakage.

    Args:
        df: DataFrame completo con todas las columnas.

    Returns:
        DataFrame con solo las features de modelado y la variable objetivo.
    """
    # Features categoricas
    categorical_features = [
        "airline_code",
        "origin",
        "destination",
        "route",
        "aircraft_type",
        "departure_hour_bin",
    ]

    # Features numericas
    numeric_features = [
        "hour",
        "day_of_week",
        "month",
        "day_of_year",
        "is_weekend",
        "is_summer",
        "is_holiday_period",
        "distance_km",
        "aircraft_age",
        "airline_delay_rate",
        "origin_delay_rate",
        "dest_delay_rate",
        "route_delay_rate",
    ]

    # Variable objetivo
    target = [TARGET_COL]

    # Seleccionar solo las columnas disponibles
    available = set(df.columns)
    selected = [
        col for col in categorical_features + numeric_features + target
        if col in available
    ]

    n_missing = len(categorical_features + numeric_features) - (len(selected) - 1)
    if n_missing > 0:
        missing_cols = [
            col for col in categorical_features + numeric_features
            if col not in available
        ]
        logger.warning(
            "%d features no disponibles y seran omitidas: %s",
            n_missing, missing_cols,
        )

    logger.info(
        "Features seleccionadas para modelado: %d features + target",
        len(selected) - 1,
    )

    return df[selected].copy()


def run() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Ejecuta el pipeline completo de feature engineering.

    Flujo:
        1. Cargar datos limpios de clean.py
        2. Crear feature 'route'
        3. Calcular aircraft_age si disponible
        4. Split temporal train/val/test
        5. Calcular tasas historicas (SIN leakage)
        6. Seleccionar features finales
        7. Guardar dataset completo con features en parquet
        8. Guardar tabla de estadisticas del split

    Returns:
        Tupla (df_train, df_val, df_test) con features finales.
    """
    logger.info("=" * 60)
    logger.info("FASE 2: FEATURE ENGINEERING")
    logger.info("=" * 60)

    if not FLIGHTS_CLEAN.exists():
        raise FileNotFoundError(
            f"No se encontro {FLIGHTS_CLEAN}. "
            "Ejecuta primero: python -m src.data.clean"
        )

    df = pd.read_parquet(FLIGHTS_CLEAN)
    logger.info("Datos limpios cargados: %d filas, %d columnas", len(df), len(df.columns))

    # Feature engineering
    df = create_route_feature(df)
    df = encode_aircraft_age(df)

    # Split ANTES de calcular tasas historicas (critico para evitar leakage)
    df_train, df_val, df_test = temporal_train_test_split(df)

    # Tasas historicas (calculadas sobre train, aplicadas a val y test)
    df_train, df_val, df_test = add_historical_rate_features(
        df_train, df_val, df_test
    )

    # Seleccionar features finales
    df_train = select_modeling_features(df_train)
    df_val = select_modeling_features(df_val)
    df_test = select_modeling_features(df_test)

    # Guardar dataset completo con features (train + val + test juntos,
    # con una columna 'split' para identificar cada particion)
    df_train_tagged = df_train.copy()
    df_train_tagged["split"] = "train"
    df_val_tagged = df_val.copy()
    df_val_tagged["split"] = "val"
    df_test_tagged = df_test.copy()
    df_test_tagged["split"] = "test"

    df_all = pd.concat([df_train_tagged, df_val_tagged, df_test_tagged], ignore_index=True)
    FLIGHTS_FEATURES.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_parquet(FLIGHTS_FEATURES, index=False)
    logger.info("Dataset con features guardado en: %s", FLIGHTS_FEATURES)

    # Guardar tabla de estadisticas del split
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    split_stats = pd.DataFrame({
        "split": ["train", "val", "test"],
        "n_samples": [len(df_train), len(df_val), len(df_test)],
        "n_positive": [
            df_train[TARGET_COL].sum(),
            df_val[TARGET_COL].sum(),
            df_test[TARGET_COL].sum(),
        ],
        "positive_rate_pct": [
            round(100.0 * df_train[TARGET_COL].mean(), 4),
            round(100.0 * df_val[TARGET_COL].mean(), 4),
            round(100.0 * df_test[TARGET_COL].mean(), 4),
        ],
    })
    split_stats.to_csv(TABLES_DIR / "split_statistics.csv", index=False)
    logger.info("Estadisticas del split guardadas en outputs/tables/split_statistics.csv")

    return df_train, df_val, df_test


if __name__ == "__main__":
    run()
