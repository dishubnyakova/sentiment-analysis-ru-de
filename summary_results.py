import json
import os
import pandas as pd

EXPERIMENTS_DIR = "experiments"


def load_json(path: str) -> dict:
    """
    Загружает JSON-файл с метриками.

    Args:
        path (str): Путь к файлу.

    Returns:
        dict: Словарь с метриками.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    """
    Собирает сводную таблицу результатов по всем моделям.
    """
    files = [
        "baseline_metrics.json",
        "xlmr_3ep_metrics.json",
        "xlmr_6ep_metrics.json",
        "xlmr_tuned_metrics.json",
    ]

    rows = []
    for filename in files:
        path = os.path.join(EXPERIMENTS_DIR, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Файл не найден: {path}")
        rows.append(load_json(path))

    df = pd.DataFrame(rows)

    df = df.rename(
        columns={
            "model": "Модель",
            "accuracy": "Accuracy",
            "precision_weighted": "Precision",
            "recall_weighted": "Recall",
            "f1_weighted": "F1-score",
        }
    )

    output_path = os.path.join(EXPERIMENTS_DIR, "results_summary.csv")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("Сводная таблица сохранена:")
    print(output_path)
    print("\nТаблица результатов:")
    print(df)


if __name__ == "__main__":
    main()
