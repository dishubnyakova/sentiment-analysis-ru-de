import os
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

MODEL_NAME = "xlm-roberta-base"
DATA_PATH = "data/dataset.csv"
OUTPUT_DIR = "model/sentiment_model"
MAX_LENGTH = 128
RANDOM_STATE = 42


def load_and_prepare_data(data_path: str) -> pd.DataFrame:
    """
    Загружает CSV-файл и проверяет наличие нужных колонок.
    """
    df = pd.read_csv(data_path)

    required_columns = {"text", "language", "label"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"В датасете отсутствуют колонки: {missing}")

    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.lower()

    # Убираем пустые строки
    df = df[df["text"] != ""].copy()

    valid_labels = {"positive", "negative", "neutral"}
    invalid_labels = set(df["label"].unique()) - valid_labels
    if invalid_labels:
        raise ValueError(f"Найдены неподдерживаемые метки: {invalid_labels}")

    return df


def create_label_mappings():
    """
    Создаёт маппинг текстовых меток в числовые.
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
    Токенизация текста для модели.
    """
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )


def compute_metrics(eval_pred):
    """
    Считает основные метрики качества.
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
        "accuracy": accuracy,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
    }


def main():
    print("Загрузка данных...")
    df = load_and_prepare_data(DATA_PATH)

    label2id, id2label = create_label_mappings()
    df["label_id"] = df["label"].map(label2id)

    print("Распределение классов:")
    print(df["label"].value_counts())

    # train / temp
    train_df, temp_df = train_test_split(
        df,
        test_size=0.3,
        random_state=RANDOM_STATE,
        stratify=df["label_id"],
    )

    # val / test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=temp_df["label_id"],
    )

    print(f"Train size: {len(train_df)}")
    print(f"Validation size: {len(val_df)}")
    print(f"Test size: {len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = Dataset.from_pandas(train_df[["text", "label_id"]].rename(columns={"label_id": "label"}))
    val_dataset = Dataset.from_pandas(val_df[["text", "label_id"]].rename(columns={"label_id": "label"}))
    test_dataset = Dataset.from_pandas(test_df[["text", "label_id"]].rename(columns={"label_id": "label"}))

    train_dataset = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    val_dataset = val_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    test_dataset = test_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    columns_to_return = ["input_ids", "attention_mask", "label"]
    train_dataset.set_format(type="torch", columns=columns_to_return)
    val_dataset.set_format(type="torch", columns=columns_to_return)
    test_dataset.set_format(type="torch", columns=columns_to_return)

    print("Загрузка модели...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=7,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,
        report_to="none",
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print("Начало обучения...")
    trainer.train()

    print("Оценка на тестовой выборке...")
    test_metrics = trainer.evaluate(test_dataset)
    print("Test metrics:")
    print(test_metrics)

    print("Сохранение модели и токенизатора...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Сохраняем метрики
    metrics_path = os.path.join(OUTPUT_DIR, "test_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=4)

    # Сохраняем разбиения для отчёта
    train_df.to_csv(os.path.join(OUTPUT_DIR, "train_split.csv"), index=False)
    val_df.to_csv(os.path.join(OUTPUT_DIR, "val_split.csv"), index=False)
    test_df.to_csv(os.path.join(OUTPUT_DIR, "test_split.csv"), index=False)

    print(f"Готово. Модель сохранена в: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
