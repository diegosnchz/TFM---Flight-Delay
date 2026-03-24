"""
geo.py - Calculo de distancias geograficas entre aeropuertos.

Implementa la formula de Haversine para calcular la distancia de gran circulo
entre dos puntos en la superficie terrestre, a partir de sus coordenadas
de latitud y longitud.

La distancia Haversine es la metrica correcta para calcular las bandas de
compensacion EU261, que se basan en la distancia ortodromica entre el
aeropuerto de origen y el de destino del vuelo.
"""

from __future__ import annotations

import logging
from math import asin, cos, radians, sin, sqrt
from typing import Optional

import pandas as pd

from src.config import AIRPORTS_FILE, AIRPORTS_PARQUET, OPENFLIGHTS_URL, setup_logging

logger = setup_logging(__name__)

# Radio medio de la Tierra en kilometros (WGS-84)
EARTH_RADIUS_KM = 6371.0

# Columnas del archivo airports.dat de OpenFlights
OPENFLIGHTS_COLUMNS = [
    "airport_id",
    "name",
    "city",
    "country",
    "iata",
    "icao",
    "latitude",
    "longitude",
    "altitude",
    "timezone",
    "dst",
    "tz_database",
    "type",
    "source",
]


def haversine_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """
    Calcula la distancia de gran circulo entre dos puntos usando la formula
    de Haversine.

    La formula Haversine es numericamente estable para distancias cortas,
    a diferencia de la formula de la ley de cosenos esferica.

    Args:
        lat1: Latitud del punto de origen en grados decimales.
        lon1: Longitud del punto de origen en grados decimales.
        lat2: Latitud del punto de destino en grados decimales.
        lon2: Longitud del punto de destino en grados decimales.

    Returns:
        Distancia en kilometros entre los dos puntos.

    Example:
        >>> haversine_distance(51.4775, -0.4614, 48.3538, 11.7861)  # LHR -> MUC
        1456.0  # aprox.
    """
    # Convertir grados a radianes
    lat1_r, lon1_r = radians(lat1), radians(lon1)
    lat2_r, lon2_r = radians(lat2), radians(lon2)

    # Diferencias
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r

    # Formula de Haversine
    a = sin(dlat / 2) ** 2 + cos(lat1_r) * cos(lat2_r) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))

    return EARTH_RADIUS_KM * c


def load_airports(force_download: bool = False) -> pd.DataFrame:
    """
    Carga el dataset de coordenadas de aeropuertos de OpenFlights.

    Si el archivo parquet procesado existe, lo carga directamente.
    Si no existe pero existe el archivo .dat crudo, lo procesa y guarda.
    Si no existe ninguno (o force_download=True), intenta descargarlo.

    Args:
        force_download: Si True, fuerza la descarga aunque el archivo exista.

    Returns:
        DataFrame con columnas: iata, name, country, latitude, longitude.
        Solo incluye aeropuertos con codigo IATA valido (3 letras).

    Raises:
        FileNotFoundError: Si no se puede descargar ni encontrar el archivo.
    """
    # Usar parquet si ya esta procesado
    if AIRPORTS_PARQUET.exists() and not force_download:
        logger.info("Cargando aeropuertos desde %s", AIRPORTS_PARQUET)
        return pd.read_parquet(AIRPORTS_PARQUET)

    # Procesar desde .dat si existe
    if AIRPORTS_FILE.exists() and not force_download:
        logger.info("Procesando airports.dat desde %s", AIRPORTS_FILE)
        return _process_airports_dat(AIRPORTS_FILE)

    # Intentar descargar
    logger.info(
        "Descargando airports.dat desde OpenFlights: %s", OPENFLIGHTS_URL
    )
    try:
        import urllib.request

        AIRPORTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(OPENFLIGHTS_URL, AIRPORTS_FILE)
        logger.info("airports.dat descargado correctamente.")
        return _process_airports_dat(AIRPORTS_FILE)
    except Exception as e:
        raise FileNotFoundError(
            f"No se pudo descargar airports.dat desde {OPENFLIGHTS_URL}. "
            f"Descargalo manualmente y ponlo en {AIRPORTS_FILE}. "
            f"Error: {e}"
        ) from e


def _process_airports_dat(filepath) -> pd.DataFrame:
    """
    Procesa el archivo airports.dat crudo de OpenFlights y lo guarda como
    parquet para uso futuro.

    Args:
        filepath: Ruta al archivo airports.dat.

    Returns:
        DataFrame limpio con coordenadas de aeropuertos.
    """
    df = pd.read_csv(
        filepath,
        header=None,
        names=OPENFLIGHTS_COLUMNS,
        na_values=["\\N"],
        encoding="utf-8",
    )

    # Filtrar solo aeropuertos con codigo IATA valido
    df = df[
        df["iata"].notna()
        & (df["iata"] != "\\N")
        & (df["iata"].str.len() == 3)
    ].copy()

    # Seleccionar y renombrar columnas relevantes
    airports = df[["iata", "name", "country", "latitude", "longitude"]].copy()
    airports = airports.drop_duplicates(subset=["iata"])
    airports = airports.reset_index(drop=True)

    logger.info(
        "Aeropuertos con IATA valido: %d", len(airports)
    )

    # Guardar como parquet para carga rapida en el futuro
    AIRPORTS_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    airports.to_parquet(AIRPORTS_PARQUET, index=False)
    logger.info("Aeropuertos guardados en %s", AIRPORTS_PARQUET)

    return airports


def build_airport_coord_dict(
    airports_df: pd.DataFrame,
) -> dict[str, tuple[float, float]]:
    """
    Construye un diccionario de coordenadas indexado por codigo IATA.

    Args:
        airports_df: DataFrame con columnas iata, latitude, longitude.

    Returns:
        Diccionario con clave=codigo_IATA, valor=(latitud, longitud).

    Example:
        >>> coords = build_airport_coord_dict(airports_df)
        >>> coords["MAD"]
        (40.4936, -3.56676)
    """
    return {
        row["iata"]: (row["latitude"], row["longitude"])
        for _, row in airports_df.iterrows()
    }


def calculate_route_distances(
    df: pd.DataFrame,
    origin_col: str,
    dest_col: str,
    airports_df: Optional[pd.DataFrame] = None,
) -> pd.Series:
    """
    Calcula la distancia en km para cada fila de un DataFrame de vuelos.

    Utiliza la formula de Haversine sobre las coordenadas del aeropuerto
    de origen y destino de cada vuelo.

    Args:
        df: DataFrame con columnas de aeropuerto origen y destino.
        origin_col: Nombre de la columna con el codigo IATA del origen.
        dest_col: Nombre de la columna con el codigo IATA del destino.
        airports_df: DataFrame de aeropuertos con coordenadas. Si es None,
            se carga automaticamente con load_airports().

    Returns:
        Serie de pandas con la distancia en km para cada vuelo. Los vuelos
        con aeropuertos desconocidos tendran NaN.

    Example:
        >>> distances = calculate_route_distances(flights_df, "origin", "dest")
    """
    if airports_df is None:
        airports_df = load_airports()

    coord_dict = build_airport_coord_dict(airports_df)

    missing_origins = set(df[origin_col].unique()) - set(coord_dict.keys())
    missing_dests = set(df[dest_col].unique()) - set(coord_dict.keys())

    if missing_origins:
        logger.warning(
            "%d aeropuertos de origen sin coordenadas: %s",
            len(missing_origins),
            sorted(missing_origins)[:10],
        )
    if missing_dests:
        logger.warning(
            "%d aeropuertos de destino sin coordenadas: %s",
            len(missing_dests),
            sorted(missing_dests)[:10],
        )

    def _distance(row) -> Optional[float]:
        origin = coord_dict.get(row[origin_col])
        dest = coord_dict.get(row[dest_col])
        if origin is None or dest is None:
            return None
        return haversine_distance(origin[0], origin[1], dest[0], dest[1])

    distances = df.apply(_distance, axis=1)
    n_missing = distances.isna().sum()
    if n_missing > 0:
        logger.warning(
            "%d vuelos (%.1f%%) sin distancia calculable (aeropuerto desconocido).",
            n_missing,
            100.0 * n_missing / len(df),
        )

    return distances
