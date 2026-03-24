"""
predict.py - Modelo de arbitraje EU261: prediccion y simulacion de Expected Value.

Usa el mejor modelo entrenado para:
1. Calcular el EV esperado para cada vuelo del test set
2. Identificar el escenario optimo de arbitraje
3. Rankear las top 10 rutas mas rentables
4. Analisis de sensibilidad del EV
5. Simulacion Monte Carlo del EV (10,000 iteraciones)

Ejecutar con: python -m src.models.predict
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd

from src.config import (
    ARBITRAGE_HOURS_INVESTED,
    FLIGHTS_FEATURES,
    HOURLY_WAGE_EUR,
    MODELS_DIR,
    P_CLAIM_DENIED,
    RANDOM_SEED,
    TABLES_DIR,
    TARGET_COL,
    TRANSPORT_COST_EUR,
    setup_logging,
)
from src.utils.eu261 import (
    breakeven_probability,
    calculate_expected_value,
    get_eu261_compensation,
)

logger = setup_logging(__name__)

np.random.seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# Precio promedio estimado de billetes low-cost por banda de distancia
# Basado en datos publicos (Google Flights, Kiwi) para rutas intraeuropeas
# ---------------------------------------------------------------------------
DEFAULT_TICKET_PRICES = {
    250: 35.0,   # Rutas cortas (<1500 km): precio promedio low-cost 35 EUR
    400: 55.0,   # Rutas medias (1500-3500 km): precio promedio 55 EUR
    600: 90.0,   # Rutas largas (>3500 km): precio promedio 90 EUR
}


def load_best_model() -> tuple[Any, str]:
    """
    Carga el mejor modelo segun la tabla de comparacion de metricas.

    Returns:
        Tupla (modelo, nombre_modelo).
    """
    comparison_file = TABLES_DIR / "model_comparison.csv"

    if comparison_file.exists():
        metrics_df = pd.read_csv(comparison_file)
        best_name = metrics_df.sort_values("auc_pr", ascending=False).iloc[0]["model"]
    else:
        # Si no hay tabla, usar XGBoost por defecto
        best_name = "xgboost"
        logger.warning(
            "No se encontro model_comparison.csv. Usando %s por defecto.",
            best_name,
        )

    model_path = MODELS_DIR / f"{best_name}.joblib"
    if not model_path.exists():
        # Intentar cualquier modelo disponible
        available = list(MODELS_DIR.glob("*.joblib"))
        if not available:
            raise FileNotFoundError(
                f"No hay modelos en {MODELS_DIR}. "
                "Ejecuta primero: python -m src.models.train"
            )
        model_path = available[0]
        best_name = model_path.stem
        logger.warning("Usando modelo alternativo: %s", best_name)

    model = joblib.load(model_path)
    logger.info("Modelo cargado: %s", best_name)
    return model, best_name


def predict_delay_probabilities(
    model: Any,
    X: pd.DataFrame,
) -> np.ndarray:
    """
    Calcula la probabilidad de retraso EU261 para cada vuelo.

    Args:
        model: Pipeline entrenado con metodo predict_proba.
        X: Features de los vuelos.

    Returns:
        Array de probabilidades de retraso (P(retraso >= 3h)).
    """
    probas = model.predict_proba(X)[:, 1]
    logger.info(
        "Probabilidades calculadas: min=%.4f, media=%.4f, max=%.4f, "
        "p95=%.4f",
        probas.min(), probas.mean(), probas.max(), np.percentile(probas, 95),
    )
    return probas


def calculate_ev_for_flights(
    df: pd.DataFrame,
    probas: np.ndarray,
    transport_cost_eur: float = TRANSPORT_COST_EUR,
    hourly_wage_eur: float = HOURLY_WAGE_EUR,
    hours_invested: float = ARBITRAGE_HOURS_INVESTED,
    p_claim_denied: float = P_CLAIM_DENIED,
) -> pd.DataFrame:
    """
    Calcula el Expected Value de arbitraje para cada vuelo.

    Anade columnas de EV al DataFrame:
      - p_delay: probabilidad de retraso EU261 predicha
      - compensation_eur: compensacion EU261 aplicable
      - ticket_price_eur: precio estimado del billete
      - ev: Expected Value neto en EUR
      - ev_profitable: True si EV > 0

    Args:
        df: DataFrame de vuelos con columna 'eu261_compensation' o 'distance_km'.
        probas: Array de probabilidades de retraso predichas.
        transport_cost_eur: Coste de transporte al aeropuerto.
        hourly_wage_eur: Valor del tiempo en EUR/hora.
        hours_invested: Horas invertidas en el viaje de arbitraje.
        p_claim_denied: Probabilidad de que la reclamacion sea denegada.

    Returns:
        DataFrame con columnas de EV anadidas.
    """
    df = df.copy()
    df["p_delay"] = probas

    # Compensacion EU261 (si no esta calculada, usar la banda de distancia)
    if "eu261_compensation" not in df.columns and "distance_km" in df.columns:
        df["eu261_compensation"] = df["distance_km"].apply(
            lambda x: get_eu261_compensation(x) if pd.notna(x) else 250
        )
    elif "eu261_compensation" not in df.columns:
        df["eu261_compensation"] = 250  # Default

    # Precio estimado del billete segun banda
    df["ticket_price_eur"] = df["eu261_compensation"].map(DEFAULT_TICKET_PRICES).fillna(35.0)

    # Calcular EV
    ev_results = df.apply(
        lambda row: calculate_expected_value(
            p_delay=row["p_delay"],
            distance_km=row.get("distance_km", 1000),
            ticket_price_eur=row["ticket_price_eur"],
            transport_cost_eur=transport_cost_eur,
            hourly_wage_eur=hourly_wage_eur,
            hours_invested=hours_invested,
            p_claim_denied=p_claim_denied,
        ),
        axis=1,
    )

    df["ev"] = [r["ev"] for r in ev_results]
    df["ev_profitable"] = [r["profitable"] for r in ev_results]
    df["expected_income"] = [r["expected_income"] for r in ev_results]
    df["total_cost"] = [r["total_cost"] for r in ev_results]

    n_profitable = df["ev_profitable"].sum()
    logger.info(
        "Vuelos con EV positivo: %d de %d (%.2f%%)",
        n_profitable, len(df), 100.0 * n_profitable / len(df),
    )

    return df


def get_optimal_scenario(df: pd.DataFrame) -> dict[str, Any]:
    """
    Identifica el escenario optimo de arbitraje.

    El escenario optimo es el vuelo (o combinacion ruta/aerolinea/horario)
    con el mayor EV esperado.

    Args:
        df: DataFrame con columnas de EV calculadas.

    Returns:
        Diccionario con los datos del escenario optimo.
    """
    if df["ev"].isna().all():
        return {}

    best_idx = df["ev"].idxmax()
    best_row = df.loc[best_idx]

    scenario = {
        "route": best_row.get("route", "N/A"),
        "airline_code": best_row.get("airline_code", "N/A"),
        "departure_hour_bin": best_row.get("departure_hour_bin", "N/A"),
        "p_delay": round(float(best_row["p_delay"]), 4),
        "compensation_eur": float(best_row.get("eu261_compensation", 0)),
        "ticket_price_eur": float(best_row.get("ticket_price_eur", 0)),
        "ev": float(best_row["ev"]),
    }

    logger.info("Escenario optimo de arbitraje:")
    for k, v in scenario.items():
        logger.info("  %s: %s", k, v)

    return scenario


def get_top_routes_by_ev(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    Obtiene las top N rutas con mayor Expected Value medio de arbitraje.

    Agrupa por ruta y calcula estadisticas de EV para identificar las
    mas rentables en promedio.

    Args:
        df: DataFrame con columnas de EV calculadas.
        n: Numero de rutas a devolver.

    Returns:
        DataFrame con las top rutas y sus estadisticas de EV.
    """
    if "route" not in df.columns:
        logger.warning("No hay columna 'route'. No se pueden calcular top rutas.")
        return pd.DataFrame()

    route_stats = (
        df.groupby("route")
        .agg(
            n_vuelos=("ev", "count"),
            p_delay_media=("p_delay", "mean"),
            ev_medio=("ev", "mean"),
            ev_max=("ev", "max"),
            pct_rentables=("ev_profitable", "mean"),
            compensation_eur=("eu261_compensation", "first"),
        )
        .reset_index()
    )

    route_stats["p_delay_media"] = route_stats["p_delay_media"].round(4)
    route_stats["ev_medio"] = route_stats["ev_medio"].round(2)
    route_stats["ev_max"] = route_stats["ev_max"].round(2)
    route_stats["pct_rentables"] = (route_stats["pct_rentables"] * 100).round(1)

    # Filtrar rutas con suficientes vuelos para ser estadisticamente fiables
    route_stats = route_stats[route_stats["n_vuelos"] >= 10]

    top_routes = route_stats.nlargest(n, "ev_medio")

    logger.info("Top %d rutas por EV medio:\n%s", n, top_routes.to_string(index=False))

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    top_routes.to_csv(TABLES_DIR / "top_routes_ev.csv", index=False)

    return top_routes


def sensitivity_analysis(
    base_p_delay: float = 0.10,
    base_distance_km: float = 1200,
) -> pd.DataFrame:
    """
    Analisis de sensibilidad del EV variando cada parametro.

    Muestra como cambia el EV al variar:
      - Precio del billete (10 a 100 EUR)
      - Probabilidad de retraso (0.01 a 0.30)
      - Coste de transporte (10 a 50 EUR)
      - Valor del tiempo (5 a 25 EUR/h)
      - Tasa de rechazo de reclamaciones (0.10 a 0.60)

    Args:
        base_p_delay: Probabilidad base de retraso.
        base_distance_km: Distancia base de la ruta.

    Returns:
        DataFrame con resultados del analisis de sensibilidad.
    """
    results = []

    # Variar precio del billete
    for price in np.linspace(10, 100, 10):
        ev_result = calculate_expected_value(
            p_delay=base_p_delay,
            distance_km=base_distance_km,
            ticket_price_eur=price,
        )
        results.append({
            "parametro": "precio_billete_eur",
            "valor": round(price, 1),
            "ev": ev_result["ev"],
        })

    # Variar probabilidad de retraso
    for p in np.linspace(0.01, 0.40, 15):
        ev_result = calculate_expected_value(
            p_delay=p,
            distance_km=base_distance_km,
            ticket_price_eur=DEFAULT_TICKET_PRICES.get(
                get_eu261_compensation(base_distance_km), 35.0
            ),
        )
        results.append({
            "parametro": "p_delay",
            "valor": round(p, 3),
            "ev": ev_result["ev"],
        })

    # Variar tasa de rechazo
    for p_denied in np.linspace(0.10, 0.70, 10):
        ev_result = calculate_expected_value(
            p_delay=base_p_delay,
            distance_km=base_distance_km,
            ticket_price_eur=DEFAULT_TICKET_PRICES.get(
                get_eu261_compensation(base_distance_km), 35.0
            ),
            p_claim_denied=p_denied,
        )
        results.append({
            "parametro": "p_claim_denied",
            "valor": round(p_denied, 2),
            "ev": ev_result["ev"],
        })

    sensitivity_df = pd.DataFrame(results)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    sensitivity_df.to_csv(TABLES_DIR / "sensitivity_analysis.csv", index=False)
    logger.info("Analisis de sensibilidad guardado en outputs/tables/sensitivity_analysis.csv")

    return sensitivity_df


def monte_carlo_simulation(
    p_delay: float,
    distance_km: float,
    ticket_price_eur: float,
    n_simulations: int = 10_000,
) -> pd.DataFrame:
    """
    Simulacion Monte Carlo del EV para el escenario optimo.

    En lugar de usar valores puntuales, muestrea cada parametro de su
    distribucion de incertidumbre para obtener la distribucion del EV.

    Distribuciones asumidas:
      - p_delay: Normal(p_delay, 0.02) truncada en [0, 1]
      - p_claim_denied: Uniforme(0.15, 0.50)
      - transport_cost: Normal(25, 5) truncada en [5, 60]
      - hours_invested: Normal(8, 1.5) truncada en [4, 16]
      - hourly_wage: Normal(8.87, 2) truncada en [5, 20]

    Args:
        p_delay: Probabilidad central de retraso.
        distance_km: Distancia de la ruta.
        ticket_price_eur: Precio del billete.
        n_simulations: Numero de iteraciones de Monte Carlo.

    Returns:
        DataFrame con los resultados de la simulacion.
    """
    rng = np.random.default_rng(RANDOM_SEED)

    # Muestrar parametros
    p_delays = np.clip(rng.normal(p_delay, 0.02, n_simulations), 0, 1)
    p_denied = rng.uniform(0.15, 0.50, n_simulations)
    transport = np.clip(rng.normal(25, 5, n_simulations), 5, 60)
    hours = np.clip(rng.normal(8, 1.5, n_simulations), 4, 16)
    wage = np.clip(rng.normal(8.87, 2, n_simulations), 5, 20)

    compensation = get_eu261_compensation(distance_km)

    # Calcular EV para cada simulacion
    expected_income = p_delays * compensation * (1 - p_denied)
    time_cost = wage * hours
    total_cost = ticket_price_eur + transport + time_cost
    ev_samples = expected_income - total_cost

    results = pd.DataFrame({
        "p_delay": p_delays,
        "p_claim_denied": p_denied,
        "transport_cost": transport,
        "hours_invested": hours,
        "hourly_wage": wage,
        "ev": ev_samples,
    })

    pct_profitable = 100.0 * (ev_samples > 0).mean()
    logger.info(
        "Monte Carlo (%d iter): EV medio=%.2f EUR, "
        "P(rentable)=%.1f%%, IC95=[%.2f, %.2f]",
        n_simulations,
        ev_samples.mean(),
        pct_profitable,
        np.percentile(ev_samples, 2.5),
        np.percentile(ev_samples, 97.5),
    )

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(TABLES_DIR / "monte_carlo_results.csv", index=False)

    return results


def run() -> dict[str, Any]:
    """
    Ejecuta el pipeline completo del modelo de arbitraje.

    Flujo:
        1. Cargar mejor modelo
        2. Cargar test set con features
        3. Predecir probabilidades de retraso
        4. Calcular EV para cada vuelo
        5. Identificar escenario optimo
        6. Top 10 rutas por EV
        7. Analisis de sensibilidad
        8. Simulacion Monte Carlo
        9. Generar figuras de arbitraje

    Returns:
        Diccionario con resultados del modelo de arbitraje.
    """
    logger.info("=" * 60)
    logger.info("FASE 6: MODELO DE ARBITRAJE EU261")
    logger.info("=" * 60)

    model, model_name = load_best_model()

    df = pd.read_parquet(FLIGHTS_FEATURES)
    df_test = df[df["split"] == "test"].drop(columns=["split"])

    X_test = df_test.drop(columns=[TARGET_COL])
    probas = predict_delay_probabilities(model, X_test)

    df_with_ev = calculate_ev_for_flights(df_test, probas)

    optimal = get_optimal_scenario(df_with_ev)
    top_routes = get_top_routes_by_ev(df_with_ev, n=10)
    sensitivity_df = sensitivity_analysis()

    # Monte Carlo con el escenario optimo
    if optimal:
        mc_results = monte_carlo_simulation(
            p_delay=optimal.get("p_delay", 0.10),
            distance_km=1200,
            ticket_price_eur=optimal.get("ticket_price_eur", 35.0),
        )
    else:
        mc_results = monte_carlo_simulation(p_delay=0.10, distance_km=1200,
                                             ticket_price_eur=35.0)

    # Generar figuras
    try:
        from src.visualization.arbitrage_plots import generate_arbitrage_figures
        generate_arbitrage_figures(
            df_with_ev=df_with_ev,
            top_routes=top_routes,
            sensitivity_df=sensitivity_df,
            mc_results=mc_results,
            optimal_scenario=optimal,
        )
    except Exception as e:
        logger.error("Error generando figuras de arbitraje: %s", e, exc_info=True)

    return {
        "model_name": model_name,
        "df_with_ev": df_with_ev,
        "optimal_scenario": optimal,
        "top_routes": top_routes,
        "sensitivity_df": sensitivity_df,
        "monte_carlo_results": mc_results,
    }


if __name__ == "__main__":
    run()
