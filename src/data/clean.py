"""
clean.py - Limpieza y transformaciones de datos de vuelos.

Toma los datos filtrados de ingest.py y aplica:
1. Calculo del retraso en minutos
2. Creacion de la variable objetivo is_eu261_eligible
3. Eliminacion de vuelos cancelados, duplicados y valores atipicos
4. Tratamiento de missing values (documentado)
5. Creacion de features temporales basicas
6. Merge con coordenadas de aeropuertos
7. Calculo de distancias y banda EU261

Ejecutar con: python -m src.data.clean
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import (
    EU261_DELAY_THRESHOLD_MINUTES,
    FLIGHTS_CLEAN,
    FLIGHTS_RAW_FILTERED,
    setup_logging,
)
from src.utils.eu261 import get_eu261_compensation, is_eu261_eligible
from src.utils.geo import calculate_route_distances, load_airports

logger = setup_logging(__name__)

# ---------------------------------------------------------------------------
# Parametros de limpieza
# ---------------------------------------------------------------------------

# Retrasos maximos razonables en minutos (filtrar outliers extremos)
# Vuelos con retraso > 24 horas se consideran anomalias de datos
MAX_DELAY_MINUTES = 24 * 60  # 1440 minutos

# Retraso minimo (vuelos con adelantos extremos se tratan como anomalias)
MIN_DELAY_MINUTES = -6 * 60  # -360 minutos (6 horas de adelanto maximo)

# Bloques horarios para departure_hour_bin
HOUR_BINS = [0, 6, 12, 18, 24]
HOUR_LABELS = ["madrugada", "manana", "tarde", "noche"]

# Periodos vacacionales europeos (mes, dia_inicio, dia_fin)
# Usados para crear la feature is_holiday_period
HOLIDAY_PERIODS = [
    # Navidad/Ano Nuevo
    (12, 20, 31),
    (1, 1, 7),
    # Semana Santa (variable, se aproxima a marzo-abril)
    (3, 25, 31),
    (4, 1, 10),
    # Verano europeo
    (7, 1, 31),
    (8, 1, 31),
    # Puentes/festivos locales (aproximacion generica)
    (6, 1, 7),
    (9, 1, 7),
]


def parse_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte flight_date a datetime y construye la hora de salida como
    feature temporal a partir de sched_dep_hhmm (entero HHMM, ej: 1430).

    En el dataset BTS, el retraso ya viene calculado en 'delay_minutes'
    (columna ARR_DELAY), por lo que NO se necesita calcular la diferencia
    entre llegada real y programada.

    Args:
        df: DataFrame con columna flight_date (str) y sched_dep_hhmm (int).

    Returns:
        DataFrame con flight_date como datetime y columna 'hour' extraida.
    """
    df = df.copy()

    if "flight_date" in df.columns:
        df["flight_date"] = pd.to_datetime(df["flight_date"], errors="coerce")
        n_nat = df["flight_date"].isna().sum()
        if n_nat > 0:
            logger.warning("flight_date: %d valores no parseables.", n_nat)

    # Extraer hora de salida del entero HHMM (ej: 1430 -> hora 14)
    if "sched_dep_hhmm" in df.columns:
        dep_hhmm = pd.to_numeric(df["sched_dep_hhmm"], errors="coerce")
        df["hour"] = (dep_hhmm // 100).astype("Int64")
        logger.info("Columna 'hour' extraida de sched_dep_hhmm.")

    return df


def calculate_delay(df: pd.DataFrame) -> pd.DataFrame:
    """
    En el dataset BTS, el retraso ya viene en la columna 'delay_minutes'
    (ARR_DELAY del BTS). Solo se valida y se registran estadisticas.

    Args:
        df: DataFrame con columna 'delay_minutes' ya presente.

    Returns:
        DataFrame con 'delay_minutes' validado (sin cambios).
    """
    df = df.copy()

    if "delay_minutes" not in df.columns:
        raise ValueError(
            "Columna 'delay_minutes' no encontrada. "
            "Verifica que ingest.py mapeo correctamente ARR_DELAY."
        )

    df["delay_minutes"] = pd.to_numeric(df["delay_minutes"], errors="coerce")

    logger.info(
        "Estadisticas de retraso (minutos): "
        "media=%.1f, mediana=%.1f, p95=%.1f, p99=%.1f",
        df["delay_minutes"].mean(),
        df["delay_minutes"].median(),
        df["delay_minutes"].quantile(0.95),
        df["delay_minutes"].quantile(0.99),
    )

    return df


def create_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea la variable objetivo binaria is_eu261_eligible.

    La variable vale 1 si el retraso es >= 180 minutos (3 horas), que es el
    umbral de la normativa EU261/2004 para tener derecho a compensacion.

    Args:
        df: DataFrame con columna 'delay_minutes'.

    Returns:
        DataFrame con columna 'is_eu261_eligible' (0 o 1).
    """
    df = df.copy()
    df["is_eu261_eligible"] = df["delay_minutes"].apply(
        lambda x: int(is_eu261_eligible(x)) if pd.notna(x) else np.nan
    )

    n_eligible = df["is_eu261_eligible"].sum()
    n_total = df["is_eu261_eligible"].notna().sum()
    rate = 100.0 * n_eligible / n_total if n_total > 0 else 0

    logger.info(
        "Vuelos con retraso EU261 (>=3h): %d de %d (%.2f%%)",
        int(n_eligible),
        n_total,
        rate,
    )

    return df


def remove_cancelled_flights(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina vuelos cancelados del dataset.

    En el dataset BTS, los vuelos cancelados tienen cancelled=1 y
    delay_minutes=NaN. Se eliminan por ambas condiciones.

    Args:
        df: DataFrame de vuelos.

    Returns:
        DataFrame sin vuelos cancelados.
    """
    n_before = len(df)

    mask_cancelled = pd.Series(False, index=df.index)
    if "cancelled" in df.columns:
        mask_cancelled = pd.to_numeric(df["cancelled"], errors="coerce").fillna(0) == 1

    # Adicionalmente, vuelos sin delay_minutes no son utilizables
    mask_no_delay = df["delay_minutes"].isna()

    df_clean = df[~mask_cancelled & ~mask_no_delay].copy()
    n_removed = n_before - len(df_clean)

    logger.info(
        "Vuelos eliminados (cancelados o sin retraso registrado): %d (%.1f%%)",
        n_removed,
        100.0 * n_removed / n_before if n_before > 0 else 0,
    )

    return df_clean


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina filas con retrasos extremos que probablemente son errores de datos.

    Criterio: retraso > 24 horas o adelanto > 6 horas.
    Estas observaciones son estadisticamente improbables y mas probablemente
    reflejan errores en el registro de datos que vuelos reales.

    Args:
        df: DataFrame con columna 'delay_minutes'.

    Returns:
        DataFrame sin outliers extremos.
    """
    n_before = len(df)
    mask_valid = (
        df["delay_minutes"].between(MIN_DELAY_MINUTES, MAX_DELAY_MINUTES)
        | df["delay_minutes"].isna()
    )
    df_clean = df[mask_valid].copy()
    n_removed = n_before - len(df_clean)

    if n_removed > 0:
        logger.info(
            "Outliers de retraso eliminados (fuera de [%d, %d] min): %d (%.1f%%)",
            MIN_DELAY_MINUTES,
            MAX_DELAY_MINUTES,
            n_removed,
            100.0 * n_removed / n_before,
        )

    return df_clean


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina registros duplicados del dataset.

    Considera duplicado cualquier vuelo con identica combinacion de:
    origin, destination, airline_code, sched_dep (o sched_arr).

    Args:
        df: DataFrame de vuelos.

    Returns:
        DataFrame sin duplicados.
    """
    key_cols = [c for c in ["origin", "destination", "airline_code",
                             "flight_date", "sched_dep_hhmm", "flight_number"]
                if c in df.columns]

    if not key_cols:
        logger.warning("No hay columnas clave para deteccion de duplicados.")
        return df

    n_before = len(df)
    df_clean = df.drop_duplicates(subset=key_cols, keep="first").copy()
    n_removed = n_before - len(df_clean)

    if n_removed > 0:
        logger.info(
            "Duplicados eliminados: %d (%.1f%%)",
            n_removed,
            100.0 * n_removed / n_before,
        )

    return df_clean


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trata los valores faltantes en el dataset.

    Estrategia por columna (documentado para el TFM):
      - delay_minutes: eliminar filas (sin retraso calculable, no podemos
        construir la variable objetivo)
      - airline_code: eliminar filas (filtro de aerolinea requiere codigo)
      - origin / destination: eliminar filas (necesarios para distancia)
      - aircraft_type: rellenar con 'DESCONOCIDO' (feature no critica)
      - sched_dep: no critica para la variable objetivo, rellenar o dejar NaN

    Args:
        df: DataFrame con posibles valores faltantes.

    Returns:
        DataFrame con missing values tratados.
    """
    df = df.copy()
    n_initial = len(df)

    # Columnas criticas: eliminar filas con NaN
    critical_cols = [c for c in ["delay_minutes", "is_eu261_eligible",
                                  "airline_code", "origin", "destination"]
                     if c in df.columns]
    for col in critical_cols:
        n_before = len(df)
        df = df[df[col].notna()]
        n_removed = n_before - len(df)
        if n_removed > 0:
            logger.info(
                "  Filas eliminadas por '%s' nulo: %d", col, n_removed
            )

    # Columnas no criticas: rellenar
    if "aircraft_type" in df.columns:
        n_missing = df["aircraft_type"].isna().sum()
        df["aircraft_type"] = df["aircraft_type"].fillna("DESCONOCIDO")
        if n_missing > 0:
            logger.info(
                "  aircraft_type: %d nulos rellenados con 'DESCONOCIDO'", n_missing
            )

    n_removed_total = n_initial - len(df)
    logger.info(
        "Tratamiento de missing values: %d filas eliminadas (%.1f%% del total)",
        n_removed_total,
        100.0 * n_removed_total / n_initial if n_initial > 0 else 0,
    )

    return df


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea features temporales a partir de flight_date y sched_dep_hhmm.

    En el dataset BTS, flight_date es la fecha del vuelo y 'hour' ya fue
    extraida en parse_datetime_columns desde sched_dep_hhmm.

    Features creadas:
      - hour: hora de salida programada (0-23)
      - day_of_week: dia de la semana (0=lunes, 6=domingo)
      - month: mes (1-12)
      - year: ano
      - day_of_year: dia del ano (1-365)
      - is_weekend: 1 si sabado o domingo
      - is_summer: 1 si junio, julio, agosto o septiembre
      - is_holiday_period: 1 si cae en periodo vacacional
      - departure_hour_bin: categoria por bloques horarios

    Args:
        df: DataFrame con columnas 'flight_date' (datetime) y 'hour' (int).

    Returns:
        DataFrame con features temporales anadidas.
    """
    df = df.copy()

    if "flight_date" not in df.columns or not pd.api.types.is_datetime64_any_dtype(df["flight_date"]):
        logger.warning("Columna 'flight_date' no disponible o no es datetime.")
        return df

    logger.info("Creando features temporales desde 'flight_date'...")

    # Si hour no fue extraida previamente, crearla como NaN
    if "hour" not in df.columns:
        df["hour"] = np.nan

    df["day_of_week"] = df["flight_date"].dt.dayofweek
    df["month"] = df["flight_date"].dt.month
    df["year"] = df["flight_date"].dt.year
    df["day_of_year"] = df["flight_date"].dt.dayofyear

    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_summer"] = df["month"].isin([6, 7, 8, 9]).astype(int)

    # Periodos de alta demanda (vacaciones en EE.UU. y equivalentes globales)
    df["is_holiday_period"] = 0
    for month, day_start, day_end in HOLIDAY_PERIODS:
        mask = (
            (df["month"] == month)
            & (df["flight_date"].dt.day >= day_start)
            & (df["flight_date"].dt.day <= day_end)
        )
        df.loc[mask, "is_holiday_period"] = 1

    # Bloques horarios (solo si hour esta disponible y no es todo NaN)
    hour_col = pd.to_numeric(df["hour"], errors="coerce")
    if hour_col.notna().sum() > 0:
        df["departure_hour_bin"] = pd.cut(
            hour_col,
            bins=HOUR_BINS,
            labels=HOUR_LABELS,
            right=False,
        )
    else:
        df["departure_hour_bin"] = "desconocido"

    logger.info(
        "Features temporales creadas: day_of_week, month, year, "
        "day_of_year, is_weekend, is_summer, is_holiday_period, departure_hour_bin"
    )

    return df


MILES_TO_KM = 1.60934


def add_distance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Anade la distancia en km y la banda de compensacion EU261.

    En el dataset BTS, la distancia ya viene en millas (columna DISTANCE,
    renombrada a distance_miles). Se convierte a kilometros y se asigna
    la banda EU261 correspondiente.

    Si distance_miles no esta disponible, se calcula via Haversine
    desde las coordenadas de OpenFlights.

    Features creadas:
      - distance_km: distancia en km
      - eu261_compensation: 250, 400 o 600 EUR segun la distancia

    Args:
        df: DataFrame con 'distance_miles' o columnas 'origin'/'destination'.

    Returns:
        DataFrame con features de distancia anadidas.
    """
    df = df.copy()

    if "distance_miles" in df.columns:
        df["distance_km"] = pd.to_numeric(df["distance_miles"], errors="coerce") * MILES_TO_KM
        logger.info(
            "Distancia convertida de millas a km. "
            "Rango: %.0f - %.0f km",
            df["distance_km"].min(), df["distance_km"].max(),
        )
    elif "origin" in df.columns and "destination" in df.columns:
        logger.info("Calculando distancias via Haversine desde coordenadas...")
        airports_df = load_airports()
        df["distance_km"] = calculate_route_distances(
            df, origin_col="origin", dest_col="destination", airports_df=airports_df
        )
    else:
        logger.warning("Sin columna de distancia. Se asignara 1000 km por defecto.")
        df["distance_km"] = 1000.0

    # Banda de compensacion EU261 segun distancia
    df["eu261_compensation"] = df["distance_km"].apply(
        lambda x: get_eu261_compensation(x) if pd.notna(x) else np.nan
    )

    n_with_dist = df["distance_km"].notna().sum()
    logger.info(
        "Distancias validas: %d de %d vuelos (%.1f%%)",
        n_with_dist, len(df), 100.0 * n_with_dist / len(df),
    )

    return df


def run() -> pd.DataFrame:
    """
    Ejecuta el pipeline completo de limpieza de datos.

    Flujo:
        1. Cargar datos filtrados de ingest.py
        2. Parsear columnas datetime
        3. Calcular retraso en minutos
        4. Crear variable objetivo
        5. Eliminar vuelos cancelados
        6. Eliminar outliers extremos
        7. Eliminar duplicados
        8. Tratar missing values
        9. Crear features temporales
        10. Anadir distancias y bandas EU261
        11. Guardar en data/processed/flights_clean.parquet

    Returns:
        DataFrame limpio y enriquecido.
    """
    logger.info("=" * 60)
    logger.info("FASE 1b: LIMPIEZA DE DATOS")
    logger.info("=" * 60)

    if not FLIGHTS_RAW_FILTERED.exists():
        raise FileNotFoundError(
            f"No se encontro {FLIGHTS_RAW_FILTERED}. "
            "Ejecuta primero: python -m src.data.ingest"
        )

    df = pd.read_parquet(FLIGHTS_RAW_FILTERED)
    logger.info("Datos cargados: %d filas", len(df))

    df = parse_datetime_columns(df)
    df = calculate_delay(df)
    df = create_target_variable(df)
    df = remove_cancelled_flights(df)
    df = remove_outliers(df)
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df = create_temporal_features(df)
    df = add_distance_features(df)

    # Resumen final
    logger.info("-" * 40)
    logger.info("Dataset limpio final: %d filas, %d columnas", len(df), len(df.columns))
    logger.info(
        "Tasa de retrasos EU261: %.4f%%",
        100.0 * df["is_eu261_eligible"].mean(),
    )

    FLIGHTS_CLEAN.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FLIGHTS_CLEAN, index=False)
    logger.info("Datos limpios guardados en: %s", FLIGHTS_CLEAN)

    return df


if __name__ == "__main__":
    run()
