"""
model_plots.py - Graficos de evaluacion de modelos (ROC, PR, SHAP, etc.).

Genera las figuras 11 a 18 del TFM para la seccion de resultados del modelado.

Este modulo es llamado desde evaluate.py y no debe ejecutarse directamente.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from src.config import (
    AIRLINE_COLORS,
    FIGURE_DPI,
    FIGURE_SIZE_DEFAULT,
    FIGURES_DIR,
    LOW_COST_AIRLINES,
    PALETTE_MUTED,
    TARGET_COL,
    setup_logging,
)

logger = setup_logging(__name__)

sns.set_theme(style="whitegrid", palette=PALETTE_MUTED, font_scale=1.2)


def save_figure(fig: plt.Figure, filename: str) -> None:
    """Guarda figura en outputs/figures/."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    filepath = FIGURES_DIR / filename
    fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    logger.info("Figura guardada: %s", filepath)
    plt.close(fig)


def fig_11_roc_curves(
    models: dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """
    Curvas ROC de todos los modelos superpuestas en un mismo grafico.

    Args:
        models: Diccionario nombre -> pipeline entrenado.
        X_test: Features del test set.
        y_test: Variable objetivo del test set.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_DEFAULT)

    colors = sns.color_palette(PALETTE_MUTED, len(models))

    for (name, model), color in zip(models.items(), colors):
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, linewidth=2, color=color,
                    label=f"{name} (AUC = {roc_auc:.4f})")
        except Exception as e:
            logger.warning("Error calculando ROC para %s: %s", name, e)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Clasificador aleatorio")
    ax.set_xlabel("Tasa de Falsos Positivos")
    ax.set_ylabel("Tasa de Verdaderos Positivos")
    ax.set_title("Curvas ROC - Comparativa de Modelos")
    ax.legend(loc="lower right")

    save_figure(fig, "fig_11_roc_curves.png")


def fig_12_pr_curves(
    models: dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """
    Curvas Precision-Recall de todos los modelos.

    AUC-PR es mas informativa que AUC-ROC con clases muy desbalanceadas,
    ya que no esta sesgada por la alta tasa de verdaderos negativos.

    Args:
        models: Diccionario nombre -> pipeline entrenado.
        X_test: Features del test set.
        y_test: Variable objetivo del test set.
    """
    baseline_rate = y_test.mean()

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_DEFAULT)

    colors = sns.color_palette(PALETTE_MUTED, len(models))

    for (name, model), color in zip(models.items(), colors):
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            precision, recall, _ = precision_recall_curve(y_test, y_proba)
            pr_auc = auc(recall, precision)
            ax.plot(recall, precision, linewidth=2, color=color,
                    label=f"{name} (AUC-PR = {pr_auc:.4f})")
        except Exception as e:
            logger.warning("Error calculando PR para %s: %s", name, e)

    # Linea de baseline (clasificador aleatorio con la tasa base)
    ax.axhline(baseline_rate, color="k", linestyle="--", linewidth=1,
               label=f"Baseline (tasa={baseline_rate:.4f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Curvas Precision-Recall - Comparativa de Modelos")
    ax.legend(loc="upper right")

    save_figure(fig, "fig_12_pr_curves.png")


def fig_13_confusion_matrix(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    threshold: float,
) -> None:
    """
    Matriz de confusion del mejor modelo con el threshold optimo.

    Args:
        model: Mejor modelo entrenado.
        X_test: Features del test set.
        y_test: Variable objetivo del test set.
        model_name: Nombre del modelo.
        threshold: Threshold optimo de clasificacion.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Sin retraso (<3h)", "Retraso EU261 (>=3h)"],
    )
    disp.plot(
        ax=ax, colorbar=True, cmap="Blues",
        values_format="d",
    )

    ax.set_title(
        f"Matriz de Confusion - {model_name}\n"
        f"(threshold = {threshold:.4f})"
    )

    save_figure(fig, "fig_13_confusion_matrix.png")


def fig_14_shap_summary(
    model: Any,
    X_test: pd.DataFrame,
    model_name: str,
) -> None:
    """
    SHAP summary plot del mejor modelo.

    Muestra la importancia y direccion del efecto de cada feature.

    Args:
        model: Mejor modelo entrenado (pipeline).
        X_test: Features del test set.
        model_name: Nombre del modelo.
    """
    try:
        import shap

        # Obtener el clasificador del pipeline
        classifier = model.named_steps["classifier"]
        preprocessor = model.named_steps["preprocessor"]

        X_processed = preprocessor.transform(X_test)

        # Obtener nombres de features transformadas
        try:
            feature_names = preprocessor.get_feature_names_out()
        except AttributeError:
            feature_names = [f"feature_{i}" for i in range(X_processed.shape[1])]

        # Calcular SHAP values (subsample para eficiencia)
        sample_size = min(1000, X_processed.shape[0])
        idx = np.random.choice(X_processed.shape[0], sample_size, replace=False)
        X_sample = X_processed[idx]

        if hasattr(classifier, "get_booster"):  # XGBoost
            explainer = shap.TreeExplainer(classifier)
        elif hasattr(classifier, "booster_"):  # LightGBM
            explainer = shap.TreeExplainer(classifier)
        else:
            explainer = shap.LinearExplainer(classifier, X_sample)

        shap_values = explainer.shap_values(X_sample)

        # Para clasificacion binaria, usar valores de la clase positiva
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        fig, ax = plt.subplots(figsize=(12, 10))
        shap.summary_plot(
            shap_values,
            X_sample,
            feature_names=feature_names,
            show=False,
            max_display=20,
        )
        fig = plt.gcf()
        fig.suptitle(f"SHAP Summary Plot - {model_name}", fontsize=13)

        save_figure(fig, "fig_14_shap_summary.png")

    except Exception as e:
        logger.error("Error generando SHAP summary: %s", e, exc_info=True)


def fig_16_feature_importance(
    model: Any,
    X_test: pd.DataFrame,
    model_name: str,
) -> None:
    """
    Feature importance del mejor modelo (basada en el modelo, no SHAP).

    Args:
        model: Mejor modelo entrenado.
        X_test: Features del test set.
        model_name: Nombre del modelo.
    """
    try:
        classifier = model.named_steps["classifier"]
        preprocessor = model.named_steps["preprocessor"]

        try:
            feature_names = preprocessor.get_feature_names_out()
        except AttributeError:
            feature_names = [f"feature_{i}" for i in range(
                preprocessor.transform(X_test[:1]).shape[1]
            )]

        if hasattr(classifier, "feature_importances_"):
            importances = classifier.feature_importances_
        elif hasattr(classifier, "coef_"):
            importances = np.abs(classifier.coef_[0])
        else:
            logger.warning("Modelo sin feature_importances_ ni coef_. Omitiendo fig_16.")
            return

        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=True).tail(20)

        fig, ax = plt.subplots(figsize=(10, 10))

        ax.barh(importance_df["feature"], importance_df["importance"],
                color=sns.color_palette(PALETTE_MUTED)[4], alpha=0.85)

        ax.set_xlabel("Importancia (Ganancia de informacion)")
        ax.set_title(f"Top 20 Features por Importancia - {model_name}")

        save_figure(fig, "fig_16_feature_importance.png")

    except Exception as e:
        logger.error("Error generando feature importance: %s", e, exc_info=True)


def fig_17_threshold_analysis(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
) -> None:
    """
    Precision, Recall y F1 en funcion del threshold de clasificacion.

    Args:
        model: Mejor modelo entrenado.
        X_test: Features del test set.
        y_test: Variable objetivo del test set.
        model_name: Nombre del modelo.
    """
    from sklearn.metrics import precision_recall_curve

    y_proba = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

    f1 = np.where(
        (precision + recall) > 0,
        2 * precision * recall / (precision + recall),
        0,
    )

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_DEFAULT)

    ax.plot(thresholds, precision[:-1], linewidth=2,
            color=sns.color_palette(PALETTE_MUTED)[0], label="Precision")
    ax.plot(thresholds, recall[:-1], linewidth=2,
            color=sns.color_palette(PALETTE_MUTED)[1], label="Recall")
    ax.plot(thresholds, f1[:-1], linewidth=2,
            color=sns.color_palette(PALETTE_MUTED)[2], label="F1-Score")

    # Threshold optimo (maximo F1)
    opt_idx = np.argmax(f1[:-1])
    ax.axvline(thresholds[opt_idx], color="red", linestyle="--", alpha=0.7,
               label=f"Threshold optimo ({thresholds[opt_idx]:.3f})")

    ax.set_xlabel("Threshold de clasificacion")
    ax.set_ylabel("Valor de la metrica")
    ax.set_title(f"Precision, Recall y F1 vs Threshold - {model_name}")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)

    save_figure(fig, "fig_17_threshold_analysis.png")


def fig_18_simpson_paradox(simpson_df: pd.DataFrame) -> None:
    """
    Visualizacion de la Paradoja de Simpson.

    Compara la tasa bruta de retrasos por aerolinea vs la probabilidad
    media predicha por el modelo.

    Args:
        simpson_df: DataFrame con columnas airline_code, tasa_bruta_pct,
            prob_media_predicha_pct.
    """
    if simpson_df.empty:
        logger.warning("simpson_df vacio. Omitiendo fig_18.")
        return

    simpson_df = simpson_df.copy()
    simpson_df["airline_name"] = simpson_df["airline_code"].map(LOW_COST_AIRLINES)

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(simpson_df))
    width = 0.35

    bars1 = ax.bar(x - width / 2, simpson_df["tasa_bruta_pct"],
                   width, label="Tasa bruta observada (%)",
                   color=sns.color_palette(PALETTE_MUTED)[0], alpha=0.85)
    bars2 = ax.bar(x + width / 2, simpson_df["prob_media_predicha_pct"],
                   width, label="Probabilidad media predicha (%)",
                   color=sns.color_palette(PALETTE_MUTED)[1], alpha=0.85)

    def autolabel(bars):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=10)

    autolabel(bars1)
    autolabel(bars2)

    ax.set_xticks(x)
    ax.set_xticklabels(simpson_df["airline_name"])
    ax.set_ylabel("Porcentaje (%)")
    ax.set_title(
        "Paradoja de Simpson: Tasa Bruta vs Probabilidad Predicha por Aerolinea\n"
        "(Controlando por ruta, aerolinea y otros factores)"
    )
    ax.legend()

    save_figure(fig, "fig_18_simpson_paradox.png")


def generate_evaluation_figures(
    models: dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    best_model_name: str,
    optimal_threshold: float,
    simpson_df: pd.DataFrame,
) -> None:
    """
    Genera todas las figuras de evaluacion (fig_11 a fig_18).

    Args:
        models: Diccionario con todos los modelos entrenados.
        X_test: Features del test set.
        y_test: Variable objetivo del test set.
        best_model_name: Nombre del mejor modelo.
        optimal_threshold: Threshold optimo del mejor modelo.
        simpson_df: DataFrame del analisis de Simpson.
    """
    logger.info("Generando figuras de evaluacion de modelos...")

    best_model = models[best_model_name]

    fig_11_roc_curves(models, X_test, y_test)
    fig_12_pr_curves(models, X_test, y_test)
    fig_13_confusion_matrix(best_model, X_test, y_test, best_model_name, optimal_threshold)
    fig_14_shap_summary(best_model, X_test, best_model_name)
    fig_16_feature_importance(best_model, X_test, best_model_name)
    fig_17_threshold_analysis(best_model, X_test, y_test, best_model_name)
    fig_18_simpson_paradox(simpson_df)

    logger.info("Figuras de evaluacion completadas.")
