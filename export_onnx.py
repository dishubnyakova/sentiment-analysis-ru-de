import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "model/sentiment_model"
ONNX_DIR = "artifacts"
ONNX_PATH = os.path.join(ONNX_DIR, "model.onnx")


class WrappedModel(torch.nn.Module):
    """
    Обёртка для экспорта модели в ONNX.
    """

    def __init__(self, model: AutoModelForSequenceClassification) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Возвращает логиты модели.

        Args:
            input_ids (torch.Tensor): Идентификаторы токенов.
            attention_mask (torch.Tensor): Маска внимания.

        Returns:
            torch.Tensor: Логиты классификатора.
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits


def main() -> None:
    """
    Экспортирует обученную модель в формат ONNX для последующей визуализации в Netron.
    """
    os.makedirs(ONNX_DIR, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()

    wrapped_model = WrappedModel(model)

    sample_text = "Это отличный фильм"
    inputs = tokenizer(
        sample_text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128,
    )

    torch.onnx.export(
        wrapped_model,
        (inputs["input_ids"], inputs["attention_mask"]),
        ONNX_PATH,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"},
        },
        opset_version=14,
    )

    print(f"ONNX model saved to: {ONNX_PATH}")


if __name__ == "__main__":
    main()
