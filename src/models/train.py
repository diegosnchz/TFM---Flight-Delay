"""
train.py - Entrenamiento y comparacion de modelos predictivos de retrasos EU261.

Entrena y compara 4 modelos:
1. Regresion Logistica (baseline interpretable)
2. Random Forest (comparacion con importancia de variables)
3. XGBoost (modelo principal)
4. LightGBM (alternativa eficiente)

Todos los modelos usan pipelines de scikit-learn para garantizar que el
preprocesamiento no cause data leakage.

Ejecutar con: python -m src.models.train
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    FLIGHTS_FEATURES,
    MODELS_DIR,
    RANDOM_SEED,
    TABLES_DIR,
    TARGET_COL,
    setup_logging,
)

logger = setup_logging(__name__)

# ---------------------------------------------------------------------------
# Definicion de features por tipo
# ---------------------------------------------------------------------------

NUMERIC_FEATURES = [
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

CATEGORICAL_FEATURES = [
    "airline_code",
    "aircraft_type",
    "departure_hour_bin",
]

# NOTA: origin, destination y route son categoricas de alta cardinalidad.
# Se encodean con las tasas historicas (ya calculadas en features.py).
# NO se incluyen como one-hot aqui para evitar explosion dimensional.


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """
    Construye el preprocesador de features usando ColumnTransformer.

    Aplica:
      - StandardScaler a features numericas
      - OneHotEncoder a features categoricas (con handle_unknown='ignore'
        para manejar categorias no vistas en test)

    El preprocesador se ajusta en train y se aplica en val/test a traves del
    Pipeline, garantizando que no hay data leakage.

    Args:
        numeric_features: Lista de nombres de features numericas.
        categorical_features: Lista de nombres de features categoricas.

    Returns:
        ColumnTransformer configurado (sin ajustar).
    """
    numeric_transformer = StandardScaler()

    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",  # Categorias no vistas en test -> vector de ceros
        sparse_output=False,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",  # Descartar columnas no especificadas
    )

    return preprocessor


def get_available_features(
    df: pd.DataFrame,
    feature_list: list[str],
) -> list[str]:
    """
    Filtra la lista de features para incluir solo las disponibles en df.

    Args:
        df: DataFrame de datos.
        feature_list: Lista de features deseadas.

    Returns:
        Lista de features disponibles en df.
    """
    available = [f for f in feature_list if f in df.columns]
    missing = [f for f in feature_list if f not in df.columns]
    if missing:
        logger.warning("Features no disponibles y omitidas: %s", missing)
    return available


def build_logistic_regression_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    class_weight: str = "balanced",
) -> Pipeline:
    """
    Construye el pipeline de Regresion Logistica.

    Modelo baseline con alta interpretabilidad. Los coeficientes permiten
    explicar directamente el impacto de cada feature en la probabilidad
    de retraso.

    Args:
        numeric_features: Features numericas.
        categorical_features: Features categoricas.
        class_weight: Manejo del desbalance. 'balanced' ajusta automaticamente.

    Returns:
        Pipeline de scikit-learn con preprocesador + modelo.
    """
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    model = LogisticRegression(
        penalty="l2",
        C=1.0,              # Regularizacion (se puede tunar)
        class_weight=class_weight,
        solver="lbfgs",
        max_iter=1000,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])


def build_random_forest_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    """
    Construye el pipeline de Random Forest.

    Usado principalmente para comparacion y para obtener feature importance
    basada en impureza de Gini.

    Args:
        numeric_features: Features numericas.
        categorical_features: Features categoricas.

    Returns:
        Pipeline de scikit-learn.
    """
    from sklearn.ensemble import RandomForestClassifier

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])


def build_xgboost_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    scale_pos_weight: float = 1.0,
) -> Pipeline:
    """
    Construye el pipeline de XGBoost.

    Modelo principal del TFM. scale_pos_weight maneja el desbalance de clases
    (se calcula como n_negativos / n_positivos).

    Args:
        numeric_features: Features numericas.
        categorical_features: Features categoricas.
        scale_pos_weight: Peso de la clase positiva. Debe ser
            n_negativos / n_positivos del training set.

    Returns:
        Pipeline de scikit-learn.
    """
    from xgboost import XGBClassifier

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",  # AUC-PR mas informativo con clases desbalanceadas
        early_stopping_rounds=50,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=0,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])


def build_lightgbm_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    """
    Construye el pipeline de LightGBM.

    Alternativa a XGBoost, mas eficiente en memoria y generalmente mas rapido
    en datasets grandes. is_unbalance=True maneja el desbalance automaticamente.

    Args:
        numeric_features: Features numericas.
        categorical_features: Features categoricas.

    Returns:
        Pipeline de scikit-learn.
    """
    from lightgbm import LGBMClassifier

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    model = LGBMClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        is_unbalance=True,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])


def calculate_scale_pos_weight(y_train: pd.Series) -> float:
    """
    Calcula el scale_pos_weight para XGBoost.

    Formula: n_negativos / n_positivos

    Args:
        y_train: Serie con la variable objetivo del training set.

    Returns:
        Factor de peso para la clase positiva.
    """
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    if n_pos == 0:
        logger.warning("No hay muestras positivas en el training set.")
        return 1.0
    weight = n_neg / n_pos
    logger.info(
        "scale_pos_weight para XGBoost: %.2f (n_neg=%d / n_pos=%d)",
        weight, n_neg, n_pos,
    )
    return weight


def train_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_name: str,
) -> Pipeline:
    """
    Entrena un pipeline con los datos de entrenamiento.

    Para XGBoost, usa el validation set para early stopping.
    Para los demas modelos, entrena directamente.

    Args:
        pipeline: Pipeline de scikit-learn sin entrenar.
        X_train: Features del training set.
        y_train: Variable objetivo del training set.
        X_val: Features del validation set (para early stopping en XGBoost).
        y_val: Variable objetivo del validation set.
        model_name: Nombre del modelo (para logs).

    Returns:
        Pipeline entrenado.
    """
    logger.info("Entrenando %s...", model_name)

    # XGBoost con early stopping requiere eval_set en el fit del clasificador
    if model_name.lower() == "xgboost":
        # Preprocesar manualmente para pasar eval_set al XGBClassifier
        preprocessor = pipeline.named_steps["preprocessor"]
        X_train_proc = preprocessor.fit_transform(X_train)
        X_val_proc = preprocessor.transform(X_val)

        pipeline.named_steps["classifier"].fit(
            X_train_proc,
            y_train,
            eval_set=[(X_val_proc, y_val)],
            verbose=False,
        )
        # El preprocessor ya esta fitted, solo necesitamos marcar el pipeline
        # Recrear pipeline con preprocessor ya ajustado
        fitted_pipeline = pipeline
    else:
        fitted_pipeline = pipeline.fit(X_train, y_train)

    logger.info("%s entrenado correctamente.", model_name)
    return fitted_pipeline


def run() -> dict[str, Any]:
    """
    Ejecuta el pipeline completo de entrenamiento de modelos.

    Flujo:
        1. Cargar datos con features
        2. Separar en train, val, test
        3. Calcular scale_pos_weight para XGBoost
        4. Entrenar los 4 modelos
        5. Guardar modelos en outputs/models/
        6. Guardar tabla comparativa de configuracion

    Returns:
        Diccionario con los modelos entrenados.
    """
    logger.info("=" * 60)
    logger.info("FASE 4: ENTRENAMIENTO DE MODELOS")
    logger.info("=" * 60)

    if not FLIGHTS_FEATURES.exists():
        raise FileNotFoundError(
            f"No se encontro {FLIGHTS_FEATURES}. "
            "Ejecuta primero: python -m src.data.features"
        )

    df = pd.read_parquet(FLIGHTS_FEATURES)

    # Separar splits
    df_train = df[df["split"] == "train"].drop(columns=["split"])
    df_val = df[df["split"] == "val"].drop(columns=["split"])
    df_test = df[df["split"] == "test"].drop(columns=["split"])

    logger.info(
        "Splits cargados: train=%d, val=%d, test=%d",
        len(df_train), len(df_val), len(df_test),
    )

    # Determinar features disponibles
    numeric_feats = get_available_features(df_train, NUMERIC_FEATURES)
    categorical_feats = get_available_features(df_train, CATEGORICAL_FEATURES)

    X_train = df_train.drop(columns=[TARGET_COL])
    y_train = df_train[TARGET_COL].astype(int)
    X_val = df_val.drop(columns=[TARGET_COL])
    y_val = df_val[TARGET_COL].astype(int)

    scale_pos_weight = calculate_scale_pos_weight(y_train)

    # Definir modelos
    models_config = {
        "logistic_regression": build_logistic_regression_pipeline(
            numeric_feats, categorical_feats
        ),
        "random_forest": build_random_forest_pipeline(
            numeric_feats, categorical_feats
        ),
        "xgboost": build_xgboost_pipeline(
            numeric_feats, categorical_feats, scale_pos_weight
        ),
        "lightgbm": build_lightgbm_pipeline(
            numeric_feats, categorical_feats
        ),
    }

    trained_models = {}
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for name, pipeline in models_config.items():
        try:
            trained = train_model(pipeline, X_train, y_train, X_val, y_val, name)
            trained_models[name] = trained

            # Guardar modelo serializado
            model_path = MODELS_DIR / f"{name}.joblib"
            joblib.dump(trained, model_path)
            logger.info("Modelo guardado en: %s", model_path)

        except Exception as e:
            logger.error("Error entrenando %s: %s", name, e, exc_info=True)

    logger.info(
        "Entrenamiento completado: %d/%d modelos exitosos.",
        len(trained_models), len(models_config),
    )

    return trained_models


if __name__ == "__main__":
    run()
