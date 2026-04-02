import os
import pandas as pd
import matplotlib.pyplot as plt

EXPERIMENTS_DIR = "experiments"


def main() -> None:
    """
    Строит график сравнения моделей по Accuracy и F1-score.
    """
    input_path = os.path.join(EXPERIMENTS_DIR, "results_summary.csv")
    output_path = os.path.join(EXPERIMENTS_DIR, "plot1_models_comparison.png")

    df = pd.read_csv(input_path)

    models = df["Модель"]
    accuracy = df["Accuracy"]
    f1 = df["F1-score"]

    x = range(len(models))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar([i - width / 2 for i in x], accuracy, width=width, label="Accuracy")
    plt.bar([i + width / 2 for i in x], f1, width=width, label="F1-score")

    plt.xticks(list(x), models, rotation=15)
    plt.ylabel("Значение метрики")
    plt.xlabel("Модель")
    plt.title("Сравнение моделей по Accuracy и F1-score")
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.show()

    print(f"График сохранён в: {output_path}")


if __name__ == "__main__":
    main()
