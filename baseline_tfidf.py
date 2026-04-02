import json
import os
from typing import Dict

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)


TRAIN_PATH = "data/splits/train_split.csv"
TEST_PATH = "data/splits/test_split.csv"
OUTPUT_DIR = "experiments"


def load_data(path: str) -> pd.DataFrame:
    """
    Загружает CSV-файл и проверяет наличие необходимых колонок.

    Args:
        path (str): Путь к CSV-файлу.

    Returns:
        pd.DataFrame: Загруженный датафрейм.
    """
    df = pd.read_csv(path)

    required_columns = {"text", "label"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"В файле {path} отсутствуют колонки: {missing}")

    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.lower()

    return df


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    """
    Вычисляет основные метрики качества.

    Args:
        y_true: Истинные метки.
        y_pred: Предсказанные метки.

    Returns:
        Dict[str, float]: Словарь с метриками.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    return {
        "model": "TF-IDF + Logistic Regression",
        "accuracy": round(float(accuracy), 4),
        "precision_weighted": round(float(precision), 4),
        "recall_weighted": round(float(recall), 4),
        "f1_weighted": round(float(f1), 4),
    }


def main() -> None:
    """
    Обучает бейзлайн-модель TF-IDF + Logistic Regression,
    оценивает её на тестовой выборке и сохраняет результаты.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Загрузка данных...")
    train_df = load_data(TRAIN_PATH)
    test_df = load_data(TEST_PATH)

    print(f"Train size: {len(train_df)}")
    print(f"Test size: {len(test_df)}")

    X_train_text = train_df["text"]
    y_train = train_df["label"]

    X_test_text = test_df["text"]
    y_test = test_df["label"]

    print("TF-IDF векторизация...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        lowercase=True,
    )

    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    print("Обучение Logistic Regression...")
    classifier = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )
    classifier.fit(X_train, y_train)

    print("Предсказание на тестовой выборке...")
    y_pred = classifier.predict(X_test)

    print("Вычисление метрик...")
    metrics = compute_metrics(y_test, y_pred)

    print("\nBaseline metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    # 1. Сохраняем метрики
    metrics_path = os.path.join(OUTPUT_DIR, "baseline_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

    # 2. Сохраняем classification report
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    report_path = os.path.join(OUTPUT_DIR, "baseline_classification_report.csv")
    report_df.to_csv(report_path, index=True)

    # 3. Сохраняем предсказания
    predictions_df = test_df.copy()
    predictions_df["predicted_label"] = y_pred
    predictions_path = os.path.join(OUTPUT_DIR, "baseline_predictions.csv")
    predictions_df.to_csv(predictions_path, index=False)

    # 4. Сохраняем confusion matrix
    labels_order = ["negative", "neutral", "positive"]
    cm = confusion_matrix(y_test, y_pred, labels=labels_order)
    cm_df = pd.DataFrame(cm, index=labels_order, columns=labels_order)
    cm_path = os.path.join(OUTPUT_DIR, "baseline_confusion_matrix.csv")
    cm_df.to_csv(cm_path, index=True)

    print("\nФайлы сохранены:")
    print(metrics_path)
    print(report_path)
    print(predictions_path)
    print(cm_path)


if __name__ == "__main__":
    main()
