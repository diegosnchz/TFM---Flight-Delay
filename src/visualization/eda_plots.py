"""
eda_plots.py - Graficos del Analisis Exploratorio de Datos (EDA).

Genera los 10 graficos principales del EDA y los exporta a outputs/figures/
en formato PNG de alta resolucion (300 DPI).

Ejecutar con: python -m src.visualization.eda_plots
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import (
    AIRLINE_COLORS,
    COLOR_DELAYED,
    COLOR_ON_TIME,
    EU261_DELAY_THRESHOLD_MINUTES,
    FIGURE_DPI,
    FIGURE_SIZE_DEFAULT,
    FIGURES_DIR,
    FLIGHTS_CLEAN,
    LOW_COST_AIRLINES,
    PALETTE_MUTED,
    TARGET_COL,
    setup_logging,
)

logger = setup_logging(__name__)

# Estilo global de matplotlib
sns.set_theme(style="whitegrid", palette=PALETTE_MUTED, font_scale=1.2)
plt.rcParams.update({
    "figure.dpi": FIGURE_DPI,
    "savefig.dpi": FIGURE_DPI,
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})


def save_figure(fig: plt.Figure, filename: str) -> None:
    """
    Guarda una figura en outputs/figures/ con la configuracion estandar.

    Args:
        fig: Figura de matplotlib.
        filename: Nombre del archivo (sin ruta, con extension .png).
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    filepath = FIGURES_DIR / filename
    fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    logger.info("Figura guardada: %s", filepath)
    plt.close(fig)


def fig_01_delay_distribution(df: pd.DataFrame) -> None:
    """
    Distribucion de retrasos en minutos (histograma + KDE).

    Incluye linea vertical en 180 minutos (umbral EU261) y anotacion
    con el porcentaje de vuelos elegibles.

    Args:
        df: DataFrame con columna 'delay_minutes'.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_DEFAULT)

    # Filtrar outliers extremos para visualizacion (mantener -60 a 600 min)
    plot_data = df["delay_minutes"].clip(-60, 600).dropna()

    ax.hist(plot_data, bins=100, density=True, alpha=0.5,
            color=sns.color_palette(PALETTE_MUTED)[0], label="Histograma")

    # KDE
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(plot_data)
    x_range = np.linspace(plot_data.min(), plot_data.max(), 500)
    ax.plot(x_range, kde(x_range), color=sns.color_palette(PALETTE_MUTED)[1],
            linewidth=2, label="Densidad (KDE)")

    # Linea umbral EU261
    ax.axvline(EU261_DELAY_THRESHOLD_MINUTES, color=COLOR_DELAYED,
               linestyle="--", linewidth=2,
               label=f"Umbral EU261 ({EU261_DELAY_THRESHOLD_MINUTES} min)")

    # Porcentaje elegibles
    pct_eligible = 100.0 * (df[TARGET_COL] == 1).mean()
    ax.text(
        EU261_DELAY_THRESHOLD_MINUTES + 10, ax.get_ylim()[1] * 0.8,
        f"Elegibles EU261:\n{pct_eligible:.2f}%",
        color=COLOR_DELAYED, fontsize=11, fontweight="bold",
    )

    ax.set_xlabel("Retraso en llegada (minutos)")
    ax.set_ylabel("Densidad")
    ax.set_title("Distribucion de Retrasos en la Llegada - Aerolineas Low-Cost Europeas")
    ax.legend()

    save_figure(fig, "fig_01_delay_distribution.png")


def fig_02_eu261_rate_by_airline(df: pd.DataFrame) -> None:
    """
    Tasa de retrasos EU261 por aerolinea (barplot horizontal).

    Args:
        df: DataFrame con columnas 'airline_code' y TARGET_COL.
    """
    rates = (
        df.groupby("airline_code")[TARGET_COL]
        .agg(["mean", "count"])
        .rename(columns={"mean": "tasa", "count": "n_vuelos"})
        .reset_index()
    )
    rates["tasa_pct"] = rates["tasa"] * 100
    rates["airline_name"] = rates["airline_code"].map(LOW_COST_AIRLINES)
    rates = rates.sort_values("tasa_pct", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5))

    colors = [AIRLINE_COLORS.get(code, "#888888") for code in rates["airline_code"]]
    bars = ax.barh(rates["airline_name"], rates["tasa_pct"], color=colors, alpha=0.85)

    # Etiquetas con porcentaje exacto
    for bar, val, n in zip(bars, rates["tasa_pct"], rates["n_vuelos"]):
        ax.text(
            bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}% (n={n:,})",
            va="center", fontsize=11,
        )

    ax.set_xlabel("Tasa de retrasos EU261 (%)")
    ax.set_title("Tasa de Retrasos EU261 por Aerolinea Low-Cost")
    ax.set_xlim(0, rates["tasa_pct"].max() * 1.4)

    save_figure(fig, "fig_02_eu261_rate_by_airline.png")


def fig_03_eu261_rate_by_hour(df: pd.DataFrame) -> None:
    """
    Tasa de retrasos EU261 por hora del dia (lineplot).

    Args:
        df: DataFrame con columnas 'hour' y TARGET_COL.
    """
    if "hour" not in df.columns:
        logger.warning("Columna 'hour' no disponible. Omitiendo fig_03.")
        return

    hourly = (
        df.groupby("hour")[TARGET_COL]
        .agg(["mean", "count"])
        .rename(columns={"mean": "tasa", "count": "n"})
        .reset_index()
    )
    hourly["tasa_pct"] = hourly["tasa"] * 100

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_DEFAULT)

    ax.plot(hourly["hour"], hourly["tasa_pct"],
            marker="o", linewidth=2.5, color=sns.color_palette(PALETTE_MUTED)[2])
    ax.fill_between(hourly["hour"], hourly["tasa_pct"], alpha=0.15,
                    color=sns.color_palette(PALETTE_MUTED)[2])

    ax.set_xlabel("Hora de salida programada")
    ax.set_ylabel("Tasa de retrasos EU261 (%)")
    ax.set_title("Tasa de Retrasos EU261 por Hora del Dia")
    ax.set_xticks(range(0, 24))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%02d:00"))

    save_figure(fig, "fig_03_eu261_rate_by_hour.png")


def fig_04_eu261_rate_by_month(df: pd.DataFrame) -> None:
    """
    Tasa de retrasos EU261 por mes del ano (barplot).

    Args:
        df: DataFrame con columnas 'month' y TARGET_COL.
    """
    if "month" not in df.columns:
        logger.warning("Columna 'month' no disponible. Omitiendo fig_04.")
        return

    MONTH_NAMES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
                   "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]

    monthly = (
        df.groupby("month")[TARGET_COL]
        .mean()
        .mul(100)
        .reset_index()
        .rename(columns={TARGET_COL: "tasa_pct"})
    )
    monthly["mes"] = monthly["month"].apply(lambda m: MONTH_NAMES[m - 1])

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_DEFAULT)

    bars = ax.bar(monthly["mes"], monthly["tasa_pct"],
                  color=sns.color_palette(PALETTE_MUTED, 12), alpha=0.85)

    for bar, val in zip(bars, monthly["tasa_pct"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2f}%", ha="center", va="bottom", fontsize=10)

    ax.set_xlabel("Mes")
    ax.set_ylabel("Tasa de retrasos EU261 (%)")
    ax.set_title("Tasa de Retrasos EU261 por Mes")

    save_figure(fig, "fig_04_eu261_rate_by_month.png")


def fig_05_top20_origin_airports(df: pd.DataFrame) -> None:
    """
    Top 20 aeropuertos de origen con mayor tasa de retraso EU261.

    Args:
        df: DataFrame con columnas 'origin' y TARGET_COL.
    """
    if "origin" not in df.columns:
        logger.warning("Columna 'origin' no disponible. Omitiendo fig_05.")
        return

    airport_rates = (
        df.groupby("origin")[TARGET_COL]
        .agg(["mean", "count"])
        .rename(columns={"mean": "tasa", "count": "n"})
        .reset_index()
    )
    airport_rates = airport_rates[airport_rates["n"] >= 50]
    top20 = airport_rates.nlargest(20, "tasa")
    top20["tasa_pct"] = top20["tasa"] * 100
    top20 = top20.sort_values("tasa_pct", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 9))

    ax.barh(top20["origin"], top20["tasa_pct"],
            color=sns.color_palette("YlOrRd_r", 20), alpha=0.85)

    ax.set_xlabel("Tasa de retrasos EU261 (%)")
    ax.set_title("Top 20 Aeropuertos de Origen con Mayor Tasa de Retraso EU261")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())

    save_figure(fig, "fig_05_top20_origin_airports.png")


def fig_06_top20_dest_airports(df: pd.DataFrame) -> None:
    """
    Top 20 aeropuertos de destino con mayor tasa de retraso EU261.

    Args:
        df: DataFrame con columnas 'destination' y TARGET_COL.
    """
    if "destination" not in df.columns:
        logger.warning("Columna 'destination' no disponible. Omitiendo fig_06.")
        return

    airport_rates = (
        df.groupby("destination")[TARGET_COL]
        .agg(["mean", "count"])
        .rename(columns={"mean": "tasa", "count": "n"})
        .reset_index()
    )
    airport_rates = airport_rates[airport_rates["n"] >= 50]
    top20 = airport_rates.nlargest(20, "tasa")
    top20["tasa_pct"] = top20["tasa"] * 100
    top20 = top20.sort_values("tasa_pct", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 9))

    ax.barh(top20["destination"], top20["tasa_pct"],
            color=sns.color_palette("PuBu_r", 20), alpha=0.85)

    ax.set_xlabel("Tasa de retrasos EU261 (%)")
    ax.set_title("Top 20 Aeropuertos de Destino con Mayor Tasa de Retraso EU261")

    save_figure(fig, "fig_06_top20_dest_airports.png")


def fig_07_aircraft_age_paradox(df: pd.DataFrame) -> None:
    """
    Tasa de retraso vs edad del avion (scatterplot con linea de tendencia).

    La "paradoja de la edad del avion" es un hallazgo interesante: aviones
    mas nuevos no siempre tienen menor tasa de retraso, posiblemente porque
    se asignan a rutas mas exigentes o tienen mas rotaciones diarias.

    Args:
        df: DataFrame con columnas 'aircraft_age' y TARGET_COL.
    """
    if "aircraft_age" not in df.columns or df["aircraft_age"].isna().all():
        logger.warning(
            "Columna 'aircraft_age' no disponible o todo NaN. Omitiendo fig_07."
        )
        return

    age_rates = (
        df.dropna(subset=["aircraft_age"])
        .groupby(df["aircraft_age"].round().astype(int))[TARGET_COL]
        .agg(["mean", "count"])
        .rename(columns={"mean": "tasa", "count": "n"})
        .reset_index()
    )
    age_rates = age_rates[age_rates["n"] >= 20]
    age_rates["tasa_pct"] = age_rates["tasa"] * 100

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_DEFAULT)

    scatter = ax.scatter(
        age_rates["aircraft_age"], age_rates["tasa_pct"],
        s=age_rates["n"] / age_rates["n"].max() * 500,
        alpha=0.6, color=sns.color_palette(PALETTE_MUTED)[3],
    )

    # Linea de tendencia
    z = np.polyfit(age_rates["aircraft_age"], age_rates["tasa_pct"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(age_rates["aircraft_age"].min(),
                          age_rates["aircraft_age"].max(), 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label="Tendencia lineal")

    ax.set_xlabel("Edad del avion (anos)")
    ax.set_ylabel("Tasa de retrasos EU261 (%)")
    ax.set_title("Paradoja de la Edad del Avion: Tasa de Retraso vs Edad")
    ax.legend()

    save_figure(fig, "fig_07_aircraft_age_paradox.png")


def fig_08_heatmap_day_hour(df: pd.DataFrame) -> None:
    """
    Heatmap de tasa de retraso por dia de la semana y hora del dia.

    Args:
        df: DataFrame con columnas 'day_of_week', 'hour' y TARGET_COL.
    """
    if "day_of_week" not in df.columns or "hour" not in df.columns:
        logger.warning("Columnas temporales no disponibles. Omitiendo fig_08.")
        return

    DAYS = ["Lunes", "Martes", "Miercoles", "Jueves", "Viernes", "Sabado", "Domingo"]

    pivot = (
        df.groupby(["day_of_week", "hour"])[TARGET_COL]
        .mean()
        .mul(100)
        .reset_index()
        .pivot(index="day_of_week", columns="hour", values=TARGET_COL)
    )
    pivot.index = [DAYS[i] for i in pivot.index]

    fig, ax = plt.subplots(figsize=(16, 6))

    sns.heatmap(
        pivot,
        annot=False,
        fmt=".2f",
        cmap="YlOrRd",
        ax=ax,
        cbar_kws={"label": "Tasa EU261 (%)"},
    )

    ax.set_xlabel("Hora del dia")
    ax.set_ylabel("Dia de la semana")
    ax.set_title("Tasa de Retrasos EU261 por Dia de la Semana y Hora de Salida")

    save_figure(fig, "fig_08_heatmap_day_hour.png")


def fig_09_correlation_matrix(df: pd.DataFrame) -> None:
    """
    Matriz de correlacion de las features numericas.

    Args:
        df: DataFrame con features numericas.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Excluir columnas irrelevantes
    exclude = ["year", "airport_id"]
    numeric_cols = [c for c in numeric_cols if c not in exclude]

    if len(numeric_cols) < 2:
        logger.warning("Menos de 2 columnas numericas. Omitiendo fig_09.")
        return

    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(14, 12))

    mask = np.triu(np.ones_like(corr, dtype=bool))  # Solo triangulo inferior
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        ax=ax, square=True,
        cbar_kws={"label": "Correlacion de Pearson"},
    )

    ax.set_title("Matriz de Correlacion de Features Numericas")

    save_figure(fig, "fig_09_correlation_matrix.png")


def fig_10_class_imbalance(df: pd.DataFrame) -> None:
    """
    Visualizacion del desbalance de clases.

    Args:
        df: DataFrame con columna TARGET_COL.
    """
    counts = df[TARGET_COL].value_counts()
    labels = ["Sin retraso EU261 (<3h)", "Con retraso EU261 (>=3h)"]
    values = [counts.get(0, 0), counts.get(1, 0)]
    colors = [COLOR_ON_TIME, COLOR_DELAYED]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Pie chart
    wedges, texts, autotexts = ax1.pie(
        values, labels=labels, colors=colors, autopct="%1.2f%%",
        startangle=90, pctdistance=0.85,
    )
    for autotext in autotexts:
        autotext.set_fontsize(12)
        autotext.set_fontweight("bold")
    ax1.set_title("Distribucion de Clases (Proporcion)")

    # Bar chart con escala logaritmica para ver ambas clases
    bars = ax2.bar(labels, values, color=colors, alpha=0.85)
    ax2.set_yscale("log")
    ax2.set_ylabel("Numero de vuelos (escala logaritmica)")
    ax2.set_title("Distribucion de Clases (Conteo Absoluto)")

    for bar, val in zip(bars, values):
        ax2.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.1,
            f"{val:,}", ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    fig.suptitle("Desbalance de Clases en el Dataset", fontsize=14, fontweight="bold")

    save_figure(fig, "fig_10_class_imbalance.png")


def run() -> None:
    """
    Genera todos los graficos del EDA y los guarda en outputs/figures/.

    Flujo:
        1. Cargar datos limpios
        2. Generar y guardar cada figura
    """
    logger.info("=" * 60)
    logger.info("FASE 3: ANALISIS EXPLORATORIO DE DATOS")
    logger.info("=" * 60)

    if not FLIGHTS_CLEAN.exists():
        raise FileNotFoundError(
            f"No se encontro {FLIGHTS_CLEAN}. "
            "Ejecuta primero: python -m src.data.clean"
        )

    df = pd.read_parquet(FLIGHTS_CLEAN)
    logger.info("Datos cargados para EDA: %d filas", len(df))

    # Generar todas las figuras
    fig_01_delay_distribution(df)
    fig_02_eu261_rate_by_airline(df)
    fig_03_eu261_rate_by_hour(df)
    fig_04_eu261_rate_by_month(df)
    fig_05_top20_origin_airports(df)
    fig_06_top20_dest_airports(df)
    fig_07_aircraft_age_paradox(df)
    fig_08_heatmap_day_hour(df)
    fig_09_correlation_matrix(df)
    fig_10_class_imbalance(df)

    logger.info("EDA completado. Figuras guardadas en: %s", FIGURES_DIR)


if __name__ == "__main__":
    run()
