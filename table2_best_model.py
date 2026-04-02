import os
import json
import pandas as pd

EXPERIMENTS_DIR = "experiments"


def load_json(path: str) -> dict:
    """
    Загружает JSON-файл.

    Args:
        path (str): Путь к JSON-файлу.

    Returns:
        dict: Содержимое JSON.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    """
    Находит лучшую модель по F1-score и формирует таблицу
    поклассовых метрик для неё.
    """
    metric_files = [
        "baseline_metrics.json",
        "xlmr_3ep_metrics.json",
        "xlmr_6ep_metrics.json",
    ]

    results = []
    for filename in metric_files:
        path = os.path.join(EXPERIMENTS_DIR, filename)
        data = load_json(path)
        data["source_file"] = filename
        results.append(data)

    summary_df = pd.DataFrame(results)

    best_row = summary_df.loc[summary_df["f1_weighted"].idxmax()]
    best_model_name = best_row["model"]
    best_source = best_row["source_file"]

    if best_source == "baseline_metrics.json":
        report_file = "baseline_classification_report.csv"
    elif best_source == "xlmr_3ep_metrics.json":
        report_file = "xlmr_3ep_classification_report.csv"
    else:
        report_file = "xlmr_6ep_classification_report.csv"

    report_path = os.path.join(EXPERIMENTS_DIR, report_file)
    report_df = pd.read_csv(report_path, index_col=0)

    class_rows = ["negative", "neutral", "positive"]
    class_report_df = report_df.loc[class_rows, ["precision", "recall", "f1-score"]].copy()
    class_report_df = class_report_df.reset_index().rename(
        columns={
            "index": "Класс",
            "precision": "Precision",
            "recall": "Recall",
            "f1-score": "F1-score",
        }
    )

    output_path = os.path.join(EXPERIMENTS_DIR, "table2_best_model_class_report.csv")
    class_report_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Лучшая модель: {best_model_name}")
    print(f"Таблица 2 сохранена в: {output_path}")
    print("\nПоклассовые метрики:")
    print(class_report_df)


if __name__ == "__main__":
    main()
