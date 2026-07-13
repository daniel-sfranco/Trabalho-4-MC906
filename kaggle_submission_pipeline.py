from __future__ import annotations

"""Pipeline final de modelagem para o desafio Predicting Heart Disease.

Esta versao consolida a etapa exploratoria registrada em `Trabalho 4.ipynb`,
reaproveitando a engenharia de atributos inicial e ampliando a comparacao com
modelos adicionais, ensembles e geracao automatica de submissao para o Kaggle.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
TARGET_COLUMN = "Heart Disease"
ID_COLUMN = "id"

BASE_CATEGORICAL_COLUMNS = [
    "Sex",
    "Chest pain type",
    "FBS over 120",
    "EKG results",
    "Exercise angina",
    "Slope of ST",
    "Number of vessels fluro",
    "Thallium",
]

ENGINEERED_CATEGORICAL_COLUMNS = [
    "Risk Cholesterol",
    "Risk BP",
]

NUMERIC_COLUMNS = [
    "Age",
    "BP",
    "Cholesterol",
    "Max HR",
    "ST depression",
    "Proportional HR",
    "Cholesterol_HR",
    "ST_by_Age",
    "BP_HR_Interaction",
]

PASSTHROUGH_COLUMNS = [
    "Severe exertion syndrome",
]

ENSEMBLE_WEIGHTS = {
    "histgb": 0.8,
    "mlp": 0.1,
    "logreg": 0.1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Treina os modelos do Integrante 2, avalia em holdout com ROC-AUC "
            "e gera arquivos de submissao para o Kaggle."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("."),
        help="Diretorio com train.csv, test.csv e sample_submission.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Diretorio onde metricas e submissions serao salvos.",
    )
    parser.add_argument(
        "--holdout-size",
        type=float,
        default=0.2,
        help="Fracao usada para validacao holdout estratificada.",
    )
    parser.add_argument(
        "--skip-submission",
        action="store_true",
        help="Se informado, roda apenas a avaliacao holdout.",
    )
    return parser.parse_args()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    engineered = df.copy()

    engineered["Risk Cholesterol"] = (
        (engineered["Cholesterol"] >= 200).astype(int)
        + (engineered["Cholesterol"] >= 240).astype(int)
    )
    engineered["Risk BP"] = (
        (engineered["BP"] >= 120).astype(int)
        + (engineered["BP"] >= 140).astype(int)
    )
    engineered["Severe exertion syndrome"] = (
        (engineered["Chest pain type"] >= 3)
        & (engineered["Exercise angina"] == 1)
        & (engineered["ST depression"] >= 5)
    ).astype(int)
    engineered["Proportional HR"] = engineered["Max HR"] / (220 - engineered["Age"])
    engineered["Cholesterol_HR"] = engineered["Cholesterol"] / (engineered["Max HR"] + 1)
    engineered["ST_by_Age"] = engineered["ST depression"] / (engineered["Age"] + 1)
    engineered["BP_HR_Interaction"] = engineered["BP"] * engineered["Proportional HR"]

    return engineered


def target_to_numeric(target: pd.Series) -> pd.Series:
    return target.map({"Presence": 1, "Absence": 0}).astype(int)


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32),
                BASE_CATEGORICAL_COLUMNS + ENGINEERED_CATEGORICAL_COLUMNS,
            ),
            ("numeric", StandardScaler(), NUMERIC_COLUMNS),
            ("passthrough", "passthrough", PASSTHROUGH_COLUMNS),
        ],
    )


def transform_split(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    X_other: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_other_transformed = preprocessor.transform(X_other)
    return (
        X_train_transformed.astype(np.float32, copy=False),
        X_other_transformed.astype(np.float32, copy=False),
    )


def build_models() -> dict[str, object]:
    return {
        "logreg": LogisticRegression(
            max_iter=400,
            solver="lbfgs",
        ),
        "rf": RandomForestClassifier(
            n_estimators=180,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            alpha=1e-4,
            batch_size=4096,
            learning_rate_init=1e-3,
            max_iter=25,
            early_stopping=True,
            n_iter_no_change=5,
            random_state=RANDOM_STATE,
        ),
        "histgb": HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=0.05,
            max_depth=8,
            max_iter=250,
            min_samples_leaf=100,
            l2_regularization=0.1,
            random_state=RANDOM_STATE,
        ),
    }


def weighted_ensemble(predictions: dict[str, np.ndarray]) -> np.ndarray:
    return sum(ENSEMBLE_WEIGHTS[name] * predictions[name] for name in ENSEMBLE_WEIGHTS)


def evaluate_holdout(
    train_df: pd.DataFrame,
    holdout_size: float,
) -> tuple[dict[str, float], dict[str, object]]:
    features = add_features(train_df.drop(columns=[TARGET_COLUMN]))
    target = target_to_numeric(train_df[TARGET_COLUMN])

    X_train, X_valid, y_train, y_valid = train_test_split(
        features,
        target,
        test_size=holdout_size,
        stratify=target,
        random_state=RANDOM_STATE,
    )

    preprocessor = build_preprocessor()
    X_train_matrix, X_valid_matrix = transform_split(preprocessor, X_train, X_valid)

    models = build_models()
    valid_predictions: dict[str, np.ndarray] = {}
    holdout_metrics: dict[str, float] = {}

    for name, model in models.items():
        print(f"Treinando {name} no holdout...")
        model.fit(X_train_matrix, y_train)
        valid_predictions[name] = model.predict_proba(X_valid_matrix)[:, 1]
        holdout_metrics[name] = roc_auc_score(y_valid, valid_predictions[name])
        print(f"ROC-AUC holdout {name}: {holdout_metrics[name]:.6f}")

    ensemble_predictions = weighted_ensemble(valid_predictions)
    holdout_metrics["ensemble"] = roc_auc_score(y_valid, ensemble_predictions)
    print(f"ROC-AUC holdout ensemble: {holdout_metrics['ensemble']:.6f}")

    artifacts = {
        "preprocessor": preprocessor,
        "models": models,
        "holdout_predictions": valid_predictions,
    }
    return holdout_metrics, artifacts


def train_full_and_export(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    sample_submission_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    train_features = add_features(train_df.drop(columns=[TARGET_COLUMN]))
    train_target = target_to_numeric(train_df[TARGET_COLUMN])
    test_features = add_features(test_df.copy())

    preprocessor = build_preprocessor()
    X_train_matrix, X_test_matrix = transform_split(preprocessor, train_features, test_features)

    models = build_models()
    test_predictions: dict[str, np.ndarray] = {}

    for name, model in models.items():
        print(f"Treinando {name} em todo o treino para submissao...")
        model.fit(X_train_matrix, train_target)
        test_predictions[name] = model.predict_proba(X_test_matrix)[:, 1]

        submission_df = sample_submission_df.copy()
        submission_df[TARGET_COLUMN] = test_predictions[name]
        submission_path = output_dir / f"submission_{name}.csv"
        submission_df.to_csv(submission_path, index=False)
        print(f"Submissao salva em {submission_path}")

    ensemble_submission = sample_submission_df.copy()
    ensemble_submission[TARGET_COLUMN] = weighted_ensemble(test_predictions)
    ensemble_path = output_dir / "submission_ensemble.csv"
    ensemble_submission.to_csv(ensemble_path, index=False)
    print(f"Submissao ensemble salva em {ensemble_path}")


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    sample_submission_df = pd.read_csv(data_dir / "sample_submission.csv")

    holdout_metrics, _ = evaluate_holdout(train_df, args.holdout_size)

    metrics_path = output_dir / "holdout_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(holdout_metrics, metrics_file, indent=2)
    print(f"Metricas salvas em {metrics_path}")

    if not args.skip_submission:
        train_full_and_export(train_df, test_df, sample_submission_df, output_dir)


if __name__ == "__main__":
    main()
