import os
import json
import argparse
from typing import Dict

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

MODEL_NAME = "xlm-roberta-base"
MAX_LENGTH = 128
LABELS_ORDER = ["negative", "neutral", "positive"]


def load_data(path: str) -> pd.DataFrame:
    """
    Загружает CSV-файл и проверяет наличие необходимых колонок.

    Args:
        path (str): Путь к файлу.

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


def create_label_mappings() -> tuple[dict, dict]:
    """
    Создаёт отображения между строковыми и числовыми метками.

    Returns:
        tuple[dict, dict]: label2id и id2label.
    """
    label2id = {
        "negative": 0,
        "neutral": 1,
        "positive": 2,
    }
    id2label = {v: k for k, v in label2id.items()}
    return label2id, id2label


def tokenize_function(examples, tokenizer):
    """
    Токенизация текстов.

    Args:
        examples: Примеры данных.
        tokenizer: Токенизатор.

    Returns:
        dict: Токенизированный батч.
    """
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )


def compute_metrics(eval_pred) -> Dict[str, float]:
    """
    Вычисляет основные метрики качества.

    Args:
        eval_pred: Логиты и метки.

    Returns:
        Dict[str, float]: Словарь с метриками.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="weighted",
        zero_division=0,
    )
    accuracy = accuracy_score(labels, predictions)

    return {
        "accuracy": round(float(accuracy), 4),
        "precision_weighted": round(float(precision), 4),
        "recall_weighted": round(float(recall), 4),
        "f1_weighted": round(float(f1), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run XLM-RoBERTa experiment on fixed splits.")
    parser.add_argument("--epochs", type=int, required=True, help="Number of training epochs.")
    parser.add_argument("--name", type=str, required=True, help="Experiment name, e.g. xlmr_3ep.")
    args = parser.parse_args()

    experiment_name = args.name
    num_epochs = args.epochs

    os.makedirs("experiments", exist_ok=True)
    os.makedirs("experiments/models", exist_ok=True)

    train_path = "data/splits/train_split.csv"
    val_path = "data/splits/val_split.csv"
    test_path = "data/splits/test_split.csv"

    print("Загрузка данных...")
    train_df = load_data(train_path)
    val_df = load_data(val_path)
    test_df = load_data(test_path)

    label2id, id2label = create_label_mappings()

    train_df["label_id"] = train_df["label"].map(label2id)
    val_df["label_id"] = val_df["label"].map(label2id)
    test_df["label_id"] = test_df["label"].map(label2id)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = Dataset.from_pandas(
        train_df[["text", "label_id"]].rename(columns={"label_id": "label"})
    )
    val_dataset = Dataset.from_pandas(
        val_df[["text", "label_id"]].rename(columns={"label_id": "label"})
    )
    test_dataset = Dataset.from_pandas(
        test_df[["text", "label_id"]].rename(columns={"label_id": "label"})
    )

    train_dataset = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    val_dataset = val_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    test_dataset = test_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    columns_to_return = ["input_ids", "attention_mask", "label"]
    train_dataset.set_format(type="torch", columns=columns_to_return)
    val_dataset.set_format(type="torch", columns=columns_to_return)
    test_dataset.set_format(type="torch", columns=columns_to_return)

    print(f"Загрузка модели {MODEL_NAME}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=id2label,
        label2id=label2id,
    )

    model_output_dir = f"experiments/models/{experiment_name}"

    training_args = TrainingArguments(
        output_dir=model_output_dir,
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        load_best_model_at_end=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print(f"Начало обучения: {experiment_name}")
    trainer.train()

    print("Оценка на тестовой выборке...")
    test_metrics = trainer.evaluate(test_dataset)

    print("Получение предсказаний на тестовой выборке...")
    predictions_output = trainer.predict(test_dataset)
    logits = predictions_output.predictions
    predicted_ids = np.argmax(logits, axis=1)

    predicted_labels = [id2label[int(i)] for i in predicted_ids]
    true_labels = [id2label[int(i)] for i in test_df["label_id"].tolist()]

    metrics = {
        "model": f"XLM-RoBERTa ({num_epochs} epochs)",
        "accuracy": round(float(test_metrics["eval_accuracy"]), 4),
        "precision_weighted": round(float(test_metrics["eval_precision_weighted"]), 4),
        "recall_weighted": round(float(test_metrics["eval_recall_weighted"]), 4),
        "f1_weighted": round(float(test_metrics["eval_f1_weighted"]), 4),
    }

    metrics_path = f"experiments/{experiment_name}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

    report = classification_report(
        true_labels,
        predicted_labels,
        labels=LABELS_ORDER,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose()
    report_path = f"experiments/{experiment_name}_classification_report.csv"
    report_df.to_csv(report_path, index=True)

    predictions_df = test_df.copy()
    predictions_df["predicted_label"] = predicted_labels
    predictions_path = f"experiments/{experiment_name}_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)

    cm = confusion_matrix(true_labels, predicted_labels, labels=LABELS_ORDER)
    cm_df = pd.DataFrame(cm, index=LABELS_ORDER, columns=LABELS_ORDER)
    cm_path = f"experiments/{experiment_name}_confusion_matrix.csv"
    cm_df.to_csv(cm_path, index=True)

    print("\nИтоговые метрики:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\nФайлы сохранены:")
    print(metrics_path)
    print(report_path)
    print(predictions_path)
    print(cm_path)


if __name__ == "__main__":
    main()
