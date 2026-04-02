import os
import json
import pandas as pd

EXPERIMENTS_DIR = "experiments"


def load_json(path: str) -> dict:
    """
    Загружает JSON-файл.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    """
    Находит лучшую модель и формирует файл с примерами
    корректных и некорректных предсказаний для верификации.
    """
    metric_files = [
        "baseline_metrics.json",
        "xlmr_3ep_metrics.json",
        "xlmr_6ep_metrics.json",
        "xlmr_tuned_metrics.json",
    ]

    results = []
    for filename in metric_files:
        path = os.path.join(EXPERIMENTS_DIR, filename)
        data = load_json(path)
        data["source_file"] = filename
        results.append(data)

    results_df = pd.DataFrame(results)
    best_row = results_df.loc[results_df["f1_weighted"].idxmax()]
    best_source = best_row["source_file"]
    best_model_name = best_row["model"]

    if best_source == "baseline_metrics.json":
        pred_file = "baseline_predictions.csv"
    elif best_source == "xlmr_3ep_metrics.json":
        pred_file = "xlmr_3ep_predictions.csv"
    elif best_source == "xlmr_6ep_metrics.json":
        pred_file = "xlmr_6ep_predictions.csv"
    else:
        pred_file = "xlmr_tuned_predictions.csv"

    pred_path = os.path.join(EXPERIMENTS_DIR, pred_file)
    df = pd.read_csv(pred_path)

    df["is_correct"] = df["label"] == df["predicted_label"]

    correct_examples = df[df["is_correct"]].head(5).copy()
    incorrect_examples = df[~df["is_correct"]].head(5).copy()

    correct_examples["example_type"] = "correct"
    incorrect_examples["example_type"] = "incorrect"

    verification_df = pd.concat([correct_examples, incorrect_examples], ignore_index=True)

    output_path = os.path.join(EXPERIMENTS_DIR, "verification_examples.csv")
    verification_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Лучшая модель: {best_model_name}")
    print(f"Файл для верификации сохранён в: {output_path}")
    print("\nПримеры для верификации:")
    print(verification_df[["text", "label", "predicted_label", "example_type"]])


if __name__ == "__main__":
    main()
