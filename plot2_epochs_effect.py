import json
import os
import matplotlib.pyplot as plt

EXPERIMENTS_DIR = "experiments"


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    path_3 = os.path.join(EXPERIMENTS_DIR, "xlmr_3ep_metrics.json")
    path_6 = os.path.join(EXPERIMENTS_DIR, "xlmr_6ep_metrics.json")
    path_tuned = os.path.join(EXPERIMENTS_DIR, "xlmr_tuned_metrics.json")

    output_path = os.path.join(EXPERIMENTS_DIR, "plot2_xlmr_configs.png")

    data_3 = load_json(path_3)
    data_6 = load_json(path_6)
    data_tuned = load_json(path_tuned)

    labels = ["3 epochs", "6 epochs", "tuned"]
    f1_scores = [
        data_3["f1_weighted"],
        data_6["f1_weighted"],
        data_tuned["f1_weighted"],
    ]

    plt.figure(figsize=(8, 5))
    plt.plot(labels, f1_scores, marker="o", linewidth=2, label="XLM-RoBERTa")
    plt.xlabel("Конфигурация модели")
    plt.ylabel("F1-score")
    plt.title("Сравнение конфигураций XLM-RoBERTa")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.show()

    print(f"График сохранён в: {output_path}")


if __name__ == "__main__":
    main()
