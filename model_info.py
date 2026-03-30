from torchinfo import summary
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

MODEL_PATH = "model/sentiment_model"


def main() -> None:
    """
    Выводит архитектуру модели с помощью torchinfo.
    """
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    model.eval()

    # создаём пример входа
    text = "Это отличный фильм"
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=128
    )

    # передаём модель с правильными входами
    summary(
        model,
        input_data=(inputs["input_ids"], inputs["attention_mask"]),
        depth=3
    )


if __name__ == "__main__":
    main()
