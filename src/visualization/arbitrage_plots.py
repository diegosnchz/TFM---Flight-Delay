"""
arbitrage_plots.py - Graficos del modelo de arbitraje EU261.

Genera las figuras 19 a 22 del TFM para la seccion del modelo de arbitraje.

Este modulo es llamado desde predict.py y no debe ejecutarse directamente.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import (
    EU261_COMPENSATION_BANDS,
    FIGURE_DPI,
    FIGURE_SIZE_DEFAULT,
    FIGURES_DIR,
    PALETTE_MUTED,
    setup_logging,
)
from src.utils.eu261 import breakeven_probability

logger = setup_logging(__name__)

sns.set_theme(style="whitegrid", palette=PALETTE_MUTED, font_scale=1.2)


def save_figure(fig: plt.Figure, filename: str) -> None:
    """Guarda figura en outputs/figures/."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    filepath = FIGURES_DIR / filename
    fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    logger.info("Figura guardada: %s", filepath)
    plt.close(fig)


def fig_19_top_routes_ev(top_routes: pd.DataFrame) -> None:
    """
    Barplot horizontal con las top rutas por Expected Value medio.

    Args:
        top_routes: DataFrame con columnas route, ev_medio, pct_rentables.
    """
    if top_routes.empty:
        logger.warning("top_routes vacio. Omitiendo fig_19.")
        return

    top_routes = top_routes.sort_values("ev_medio", ascending=True)

    fig, ax = plt.subplots(figsize=(12, 8))

    colors = [
        sns.color_palette("RdYlGn", 10)[i]
        for i in range(len(top_routes))
    ]

    bars = ax.barh(top_routes["route"], top_routes["ev_medio"],
                   color=colors, alpha=0.85)

    # Linea de EV=0
    ax.axvline(0, color="black", linewidth=1, linestyle="-")

    for bar, pct in zip(bars, top_routes["pct_rentables"]):
        x_pos = bar.get_width()
        offset = 0.3 if x_pos >= 0 else -0.3
        ax.text(
            x_pos + offset,
            bar.get_y() + bar.get_height() / 2,
            f"EV={x_pos:.1f} EUR ({pct:.0f}% rent.)",
            va="center", fontsize=10,
        )

    ax.set_xlabel("Expected Value medio (EUR)")
    ax.set_title(
        "Top Rutas por Expected Value de Arbitraje EU261\n"
        "(Precio billete low-cost estimado, costes incluidos)"
    )

    save_figure(fig, "fig_19_top_routes_ev.png")


def fig_20_sensitivity_analysis(sensitivity_df: pd.DataFrame) -> None:
    """
    Analisis de sensibilidad del EV variando parametros clave.

    Args:
        sensitivity_df: DataFrame con columnas parametro, valor, ev.
    """
    if sensitivity_df.empty:
        logger.warning("sensitivity_df vacio. Omitiendo fig_20.")
        return

    params = sensitivity_df["parametro"].unique()
    n_params = len(params)

    fig, axes = plt.subplots(1, n_params, figsize=(6 * n_params, 5))
    if n_params == 1:
        axes = [axes]

    param_labels = {
        "precio_billete_eur": "Precio del billete (EUR)",
        "p_delay": "Probabilidad de retraso",
        "p_claim_denied": "Tasa de rechazo (%)",
    }

    colors = sns.color_palette(PALETTE_MUTED, n_params)

    for ax, param, color in zip(axes, params, colors):
        data = sensitivity_df[sensitivity_df["parametro"] == param]

        ax.plot(data["valor"], data["ev"], linewidth=2.5, color=color, marker="o")
        ax.axhline(0, color="red", linestyle="--", linewidth=1, alpha=0.7,
                   label="EV = 0")
        ax.fill_between(data["valor"], data["ev"], 0,
                        where=data["ev"] > 0, alpha=0.15, color="green",
                        label="Zona rentable")
        ax.fill_between(data["valor"], data["ev"], 0,
                        where=data["ev"] <= 0, alpha=0.15, color="red",
                        label="Zona no rentable")

        ax.set_xlabel(param_labels.get(param, param))
        ax.set_ylabel("Expected Value (EUR)")
        ax.set_title(f"Sensibilidad del EV\na '{param_labels.get(param, param)}'")
        ax.legend(fontsize=9)

    fig.suptitle("Analisis de Sensibilidad del Modelo de Arbitraje EU261",
                 fontsize=13, fontweight="bold")

    save_figure(fig, "fig_20_sensitivity_analysis.png")


def fig_21_breakeven(
    ticket_prices: Optional[np.ndarray] = None,
    distance_km_options: Optional[list[float]] = None,
) -> None:
    """
    Curva de break-even: probabilidad minima de retraso para EV > 0
    en funcion del precio del billete.

    Args:
        ticket_prices: Array de precios de billetes a evaluar.
        distance_km_options: Lista de distancias (km) para las 3 bandas EU261.
    """
    if ticket_prices is None:
        ticket_prices = np.linspace(5, 120, 100)

    if distance_km_options is None:
        distance_km_options = [800, 2000, 4500]  # Corta, media, larga

    distance_labels = {
        800: "Ruta corta (<1500 km, 250 EUR)",
        2000: "Ruta media (1500-3500 km, 400 EUR)",
        4500: "Ruta larga (>3500 km, 600 EUR)",
    }

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_DEFAULT)

    colors = sns.color_palette(PALETTE_MUTED, len(distance_km_options))

    for dist, color in zip(distance_km_options, colors):
        breakevens = [breakeven_probability(dist, price) for price in ticket_prices]
        breakevens = np.array(breakevens)

        ax.plot(ticket_prices, np.clip(breakevens, 0, 1),
                linewidth=2.5, color=color,
                label=distance_labels.get(dist, f"{dist} km"))

    ax.axhline(0.0062, color="gray", linestyle=":", linewidth=1.5, alpha=0.8,
               label="Tasa media del dataset (0.62%)")

    ax.set_xlabel("Precio del billete (EUR)")
    ax.set_ylabel("Probabilidad minima de retraso para EV > 0")
    ax.set_title(
        "Analisis de Break-Even: Probabilidad Minima de Retraso\n"
        "para que el Arbitraje EU261 sea Rentable"
    )
    ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))
    ax.set_ylim(0, 1.05)

    save_figure(fig, "fig_21_breakeven.png")


def fig_22_ev_simulation(mc_results: pd.DataFrame) -> None:
    """
    Distribucion del EV obtenida por simulacion Monte Carlo.

    Args:
        mc_results: DataFrame con columna 'ev' (resultados de la simulacion).
    """
    if mc_results.empty:
        logger.warning("mc_results vacio. Omitiendo fig_22.")
        return

    ev_samples = mc_results["ev"].values
    pct_profitable = 100.0 * (ev_samples > 0).mean()
    ev_mean = ev_samples.mean()
    ev_p5, ev_p95 = np.percentile(ev_samples, [5, 95])

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_DEFAULT)

    ax.hist(ev_samples, bins=80, density=True, alpha=0.6,
            color=sns.color_palette(PALETTE_MUTED)[3], label="Distribucion EV")

    from scipy.stats import gaussian_kde
    kde = gaussian_kde(ev_samples)
    x_range = np.linspace(ev_samples.min(), ev_samples.max(), 500)
    ax.plot(x_range, kde(x_range), linewidth=2,
            color=sns.color_palette(PALETTE_MUTED)[0])

    ax.axvline(0, color="red", linestyle="--", linewidth=2,
               label="EV = 0 (break-even)")
    ax.axvline(ev_mean, color="green", linestyle="-", linewidth=2,
               label=f"EV medio = {ev_mean:.2f} EUR")
    ax.axvline(ev_p5, color="orange", linestyle=":", linewidth=1.5,
               label=f"IC90%: [{ev_p5:.2f}, {ev_p95:.2f}] EUR")
    ax.axvline(ev_p95, color="orange", linestyle=":", linewidth=1.5)

    ax.fill_between(x_range, kde(x_range), alpha=0.12, color="green",
                    where=x_range > 0, label=f"Escenarios rentables ({pct_profitable:.1f}%)")
    ax.fill_between(x_range, kde(x_range), alpha=0.12, color="red",
                    where=x_range <= 0)

    ax.set_xlabel("Expected Value (EUR)")
    ax.set_ylabel("Densidad de probabilidad")
    ax.set_title(
        f"Simulacion Monte Carlo del Expected Value de Arbitraje EU261\n"
        f"({len(ev_samples):,} iteraciones, escenario optimo)"
    )
    ax.legend()

    save_figure(fig, "fig_22_ev_simulation.png")


def generate_arbitrage_figures(
    df_with_ev: pd.DataFrame,
    top_routes: pd.DataFrame,
    sensitivity_df: pd.DataFrame,
    mc_results: pd.DataFrame,
    optimal_scenario: dict[str, Any],
) -> None:
    """
    Genera todas las figuras del modelo de arbitraje (fig_19 a fig_22).

    Args:
        df_with_ev: DataFrame de vuelos con columnas de EV calculadas.
        top_routes: Top rutas por EV medio.
        sensitivity_df: Resultados del analisis de sensibilidad.
        mc_results: Resultados de la simulacion Monte Carlo.
        optimal_scenario: Datos del escenario optimo.
    """
    logger.info("Generando figuras del modelo de arbitraje...")

    fig_19_top_routes_ev(top_routes)
    fig_20_sensitivity_analysis(sensitivity_df)
    fig_21_breakeven()
    fig_22_ev_simulation(mc_results)

    logger.info("Figuras de arbitraje completadas.")
