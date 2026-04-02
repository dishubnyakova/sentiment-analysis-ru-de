import os
import json
import pandas as pd
import matplotlib.pyplot as plt

EXPERIMENTS_DIR = "experiments"


def load_json(path: str) -> dict:
    """
    Загружает JSON-файл.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    """
    Определяет лучшую модель по F1-score и строит график её матрицы ошибок.
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
        cm_file = "baseline_confusion_matrix.csv"
    elif best_source == "xlmr_3ep_metrics.json":
        cm_file = "xlmr_3ep_confusion_matrix.csv"
    elif best_source == "xlmr_6ep_metrics.json":
        cm_file = "xlmr_6ep_confusion_matrix.csv"
    else:
        cm_file = "xlmr_tuned_confusion_matrix.csv"

    cm_path = os.path.join(EXPERIMENTS_DIR, cm_file)
    output_path = os.path.join(EXPERIMENTS_DIR, "plot3_confusion_matrix.png")

    cm_df = pd.read_csv(cm_path, index_col=0)

    plt.figure(figsize=(7, 6))
    plt.imshow(cm_df.values, interpolation="nearest")
    plt.colorbar()

    plt.xticks(range(len(cm_df.columns)), cm_df.columns)
    plt.yticks(range(len(cm_df.index)), cm_df.index)
    plt.xlabel("Предсказанный класс")
    plt.ylabel("Истинный класс")
    plt.title(f"Матрица ошибок лучшей модели: {best_model_name}")

    for i in range(cm_df.shape[0]):
        for j in range(cm_df.shape[1]):
            plt.text(j, i, str(cm_df.iloc[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()

    print(f"Лучшая модель: {best_model_name}")
    print(f"График сохранён в: {output_path}")


if __name__ == "__main__":
    main()
