"""
evaluate.py - Evaluacion y comparacion de modelos predictivos.

Calcula metricas exhaustivas para todos los modelos entrenados y genera
las figuras de evaluacion:
  - Curvas ROC y Precision-Recall comparativas
  - Matriz de confusion (mejor modelo)
  - SHAP values para interpretabilidad
  - Analisis del threshold optimo
  - Paradoja de Simpson por aerolinea

Ejecutar con: python -m src.models.evaluate
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import (
    FIGURES_DIR,
    FLIGHTS_FEATURES,
    MODELS_DIR,
    RANDOM_SEED,
    TABLES_DIR,
    TARGET_COL,
    setup_logging,
)

logger = setup_logging(__name__)


def load_trained_models() -> dict[str, Any]:
    """
    Carga todos los modelos serializados desde outputs/models/.

    Returns:
        Diccionario nombre -> pipeline entrenado.
    """
    models = {}
    model_files = list(MODELS_DIR.glob("*.joblib"))

    if not model_files:
        raise FileNotFoundError(
            f"No se encontraron modelos en {MODELS_DIR}. "
            "Ejecuta primero: python -m src.models.train"
        )

    for path in model_files:
        name = path.stem  # Nombre sin extension
        logger.info("Cargando modelo: %s", name)
        models[name] = joblib.load(path)

    return models


def evaluate_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Calcula el conjunto completo de metricas para un modelo.

    Args:
        model: Pipeline entrenado con metodo predict_proba.
        X: Features del dataset de evaluacion.
        y: Variable objetivo verdadera.
        model_name: Nombre del modelo (para logs).
        threshold: Umbral de clasificacion para calcular F1, precision, recall.

    Returns:
        Diccionario con todas las metricas.
    """
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "model": model_name,
        "auc_roc": round(roc_auc_score(y, y_proba), 4),
        "auc_pr": round(average_precision_score(y, y_proba), 4),
        "f1": round(f1_score(y, y_pred, zero_division=0), 4),
        "precision": round(precision_score(y, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y, y_pred, zero_division=0), 4),
        "log_loss": round(log_loss(y, y_proba), 4),
        "threshold": threshold,
    }

    logger.info(
        "%s -> AUC-ROC=%.4f, AUC-PR=%.4f, F1=%.4f, "
        "Precision=%.4f, Recall=%.4f",
        model_name,
        metrics["auc_roc"],
        metrics["auc_pr"],
        metrics["f1"],
        metrics["precision"],
        metrics["recall"],
    )

    return metrics


def find_optimal_threshold(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    metric: str = "f1",
) -> float:
    """
    Encuentra el threshold optimo que maximiza la metrica especificada.

    En problemas con clases muy desbalanceadas (como este, con ~0.62%
    positivos), el threshold por defecto de 0.5 puede ser suboptimo.
    Este metodo explora el espacio de thresholds para encontrar el mejor.

    Args:
        model: Pipeline entrenado.
        X: Features del validation set.
        y: Variable objetivo del validation set.
        metric: Metrica a maximizar ('f1', 'precision', 'recall').

    Returns:
        Threshold optimo.
    """
    y_proba = model.predict_proba(X)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y, y_proba)

    if metric == "f1":
        # F1 = 2 * precision * recall / (precision + recall)
        f1_scores = np.where(
            (precisions + recalls) > 0,
            2 * precisions * recalls / (precisions + recalls),
            0,
        )
        best_idx = np.argmax(f1_scores[:-1])  # thresholds tiene len-1 elementos
    elif metric == "precision":
        best_idx = np.argmax(precisions[:-1])
    elif metric == "recall":
        best_idx = np.argmax(recalls[:-1])
    else:
        raise ValueError(f"Metrica no soportada: {metric}")

    optimal_threshold = thresholds[best_idx]
    logger.info(
        "Threshold optimo para %s: %.4f", metric, optimal_threshold
    )

    return float(optimal_threshold)


def identify_best_model(metrics_df: pd.DataFrame) -> str:
    """
    Identifica el mejor modelo segun AUC-PR (metrica principal para clases
    desbalanceadas).

    Args:
        metrics_df: DataFrame con metricas de todos los modelos.

    Returns:
        Nombre del mejor modelo.
    """
    best_row = metrics_df.loc[metrics_df["auc_pr"].idxmax()]
    best_name = best_row["model"]
    logger.info(
        "Mejor modelo por AUC-PR: %s (AUC-PR=%.4f, AUC-ROC=%.4f)",
        best_name,
        best_row["auc_pr"],
        best_row["auc_roc"],
    )
    return best_name


def simpson_paradox_analysis(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    df_test_full: pd.DataFrame,
) -> pd.DataFrame:
    """
    Analiza la Paradoja de Simpson: compara la tasa bruta de retrasos por
    aerolinea con la probabilidad predicha controlando por ruta.

    La paradoja de Simpson ocurre cuando una aerolinea tiene mayor tasa
    bruta de retrasos pero, al controlar por las rutas que opera, su
    probabilidad intrinseca de retraso es menor.

    Args:
        model: Mejor modelo entrenado.
        X_test: Features del test set.
        y_test: Variable objetivo del test set.
        df_test_full: Test set con todas las columnas (incluye airline_code).

    Returns:
        DataFrame con tasa bruta vs probabilidad media predicha por aerolinea.
    """
    y_proba = model.predict_proba(X_test)[:, 1]

    if "airline_code" not in df_test_full.columns:
        logger.warning(
            "No hay columna 'airline_code' para el analisis de Simpson."
        )
        return pd.DataFrame()

    analysis = pd.DataFrame({
        "airline_code": df_test_full["airline_code"].values,
        "actual": y_test.values,
        "predicted_proba": y_proba,
    })

    result = analysis.groupby("airline_code").agg(
        n_vuelos=("actual", "count"),
        tasa_bruta_pct=("actual", lambda x: round(100.0 * x.mean(), 3)),
        prob_media_predicha_pct=("predicted_proba", lambda x: round(100.0 * x.mean(), 3)),
    ).reset_index()

    result = result.sort_values("tasa_bruta_pct", ascending=False)

    logger.info("Analisis Paradoja de Simpson:\n%s", result.to_string(index=False))

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(TABLES_DIR / "simpson_paradox.csv", index=False)

    return result


def run() -> dict[str, Any]:
    """
    Ejecuta el pipeline completo de evaluacion de modelos.

    Flujo:
        1. Cargar modelos entrenados
        2. Cargar test set
        3. Calcular metricas para cada modelo
        4. Encontrar threshold optimo del mejor modelo
        5. Analisis de Paradoja de Simpson
        6. Guardar tabla comparativa de metricas
        7. Generar figuras (delegado a model_plots.py)

    Returns:
        Diccionario con resultados de evaluacion.
    """
    logger.info("=" * 60)
    logger.info("FASE 5: EVALUACION DE MODELOS")
    logger.info("=" * 60)

    models = load_trained_models()

    if not FLIGHTS_FEATURES.exists():
        raise FileNotFoundError(
            f"No se encontro {FLIGHTS_FEATURES}. "
            "Ejecuta primero: python -m src.data.features"
        )

    df = pd.read_parquet(FLIGHTS_FEATURES)
    df_test = df[df["split"] == "test"].drop(columns=["split"])
    df_val = df[df["split"] == "val"].drop(columns=["split"])

    X_test = df_test.drop(columns=[TARGET_COL])
    y_test = df_test[TARGET_COL].astype(int)
    X_val = df_val.drop(columns=[TARGET_COL])
    y_val = df_val[TARGET_COL].astype(int)

    # Evaluar todos los modelos
    all_metrics = []
    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test, name)
        all_metrics.append(metrics)

    metrics_df = pd.DataFrame(all_metrics).sort_values("auc_pr", ascending=False)

    # Guardar tabla de metricas
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(TABLES_DIR / "model_comparison.csv", index=False)
    logger.info("Tabla de metricas guardada en outputs/tables/model_comparison.csv")
    logger.info("\n%s", metrics_df.to_string(index=False))

    # Mejor modelo
    best_name = identify_best_model(metrics_df)
    best_model = models[best_name]

    # Threshold optimo sobre validation set
    optimal_threshold = find_optimal_threshold(best_model, X_val, y_val, metric="f1")

    # Metricas con threshold optimo
    logger.info(
        "\nEvaluacion del mejor modelo (%s) con threshold optimo (%.4f):",
        best_name, optimal_threshold,
    )
    optimal_metrics = evaluate_model(
        best_model, X_test, y_test, best_name, threshold=optimal_threshold
    )

    # Guardar threshold optimo
    pd.DataFrame([{"model": best_name, "optimal_threshold": optimal_threshold}]).to_csv(
        TABLES_DIR / "optimal_threshold.csv", index=False
    )

    # Analisis Paradoja de Simpson
    simpson_df = simpson_paradox_analysis(best_model, X_test, y_test, df_test)

    # Generar figuras
    try:
        from src.visualization.model_plots import generate_evaluation_figures
        generate_evaluation_figures(
            models=models,
            X_test=X_test,
            y_test=y_test,
            best_model_name=best_name,
            optimal_threshold=optimal_threshold,
            simpson_df=simpson_df,
        )
    except Exception as e:
        logger.error("Error generando figuras de evaluacion: %s", e, exc_info=True)

    return {
        "models": models,
        "metrics_df": metrics_df,
        "best_model_name": best_name,
        "best_model": best_model,
        "optimal_threshold": optimal_threshold,
    }


if __name__ == "__main__":
    run()
