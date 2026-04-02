import os
import json
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


def compute_metrics(eval_pred) -> dict:
    """
    Вычисляет основные метрики качества.

    Args:
        eval_pred: Логиты и метки.

    Returns:
        dict: Словарь с метриками.
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
    """
    Запускает tuned-эксперимент для XLM-RoBERTa:
    обучение на train+val и тестирование на test.
    """
    os.makedirs("experiments", exist_ok=True)
    os.makedirs("experiments/models", exist_ok=True)

    train_path = "data/splits/train_split.csv"
    val_path = "data/splits/val_split.csv"
    test_path = "data/splits/test_split.csv"

    print("Загрузка данных...")
    train_df = load_data(train_path)
    val_df = load_data(val_path)
    test_df = load_data(test_path)

    # Объединяем train и val для tuned-обучения
    trainval_df = pd.concat([train_df, val_df], ignore_index=True)

    label2id, id2label = create_label_mappings()

    trainval_df["label_id"] = trainval_df["label"].map(label2id)
    test_df["label_id"] = test_df["label"].map(label2id)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = Dataset.from_pandas(
        trainval_df[["text", "label_id"]].rename(columns={"label_id": "label"})
    )
    test_dataset = Dataset.from_pandas(
        test_df[["text", "label_id"]].rename(columns={"label_id": "label"})
    )

    train_dataset = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    test_dataset = test_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    columns_to_return = ["input_ids", "attention_mask", "label"]
    train_dataset.set_format(type="torch", columns=columns_to_return)
    test_dataset.set_format(type="torch", columns=columns_to_return)

    print(f"Размер train+val: {len(trainval_df)}")
    print(f"Размер test: {len(test_df)}")

    print(f"Загрузка модели {MODEL_NAME}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=id2label,
        label2id=label2id,
    )

    model_output_dir = "experiments/models/xlmr_tuned"

    training_args = TrainingArguments(
        output_dir=model_output_dir,
        save_strategy="no",
        logging_strategy="epoch",
        learning_rate=1e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=4,
        weight_decay=0.01,
        warmup_ratio=0.1,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        compute_metrics=compute_metrics,
    )

    print("Начало tuned-обучения...")
    trainer.train()

    print("Оценка tuned-модели на тестовой выборке...")
    predictions_output = trainer.predict(test_dataset)
    logits = predictions_output.predictions
    predicted_ids = np.argmax(logits, axis=1)

    predicted_labels = [id2label[int(i)] for i in predicted_ids]
    true_labels = [id2label[int(i)] for i in test_df["label_id"].tolist()]

    accuracy = accuracy_score(true_labels, predicted_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        average="weighted",
        zero_division=0,
    )

    metrics = {
        "model": "XLM-RoBERTa (tuned: train+val, lr=1e-5, bs=4, 4 epochs)",
        "accuracy": round(float(accuracy), 4),
        "precision_weighted": round(float(precision), 4),
        "recall_weighted": round(float(recall), 4),
        "f1_weighted": round(float(f1), 4),
    }

    metrics_path = "experiments/xlmr_tuned_metrics.json"
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
    report_path = "experiments/xlmr_tuned_classification_report.csv"
    report_df.to_csv(report_path, index=True)

    predictions_df = test_df.copy()
    predictions_df["predicted_label"] = predicted_labels
    predictions_path = "experiments/xlmr_tuned_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)

    cm = confusion_matrix(true_labels, predicted_labels, labels=LABELS_ORDER)
    cm_df = pd.DataFrame(cm, index=LABELS_ORDER, columns=LABELS_ORDER)
    cm_path = "experiments/xlmr_tuned_confusion_matrix.csv"
    cm_df.to_csv(cm_path, index=True)

    print("\nИтоговые tuned-метрики:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\nФайлы сохранены:")
    print(metrics_path)
    print(report_path)
    print(predictions_path)
    print(cm_path)


if __name__ == "__main__":
    main()
