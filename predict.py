import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "model/sentiment_model"


class SentimentPredictor:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()

    def predict(self, text: str):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class_id = torch.argmax(probs, dim=-1).item()

        predicted_label = self.model.config.id2label[predicted_class_id]
        probabilities = probs[0].tolist()

        return {
            "label": predicted_label,
            "probabilities": {
                self.model.config.id2label[i]: round(probabilities[i], 4)
                for i in range(len(probabilities))
            },
        }


if __name__ == "__main__":
    predictor = SentimentPredictor(MODEL_PATH)

    while True:
        text = input("Введите текст (или 'exit' для выхода): ").strip()
        if text.lower() == "exit":
            break

        result = predictor.predict(text)
        print("\nРезультат:")
        print(f"Метка: {result['label']}")
        print("Вероятности:")
        for label, prob in result["probabilities"].items():
            print(f"  {label}: {prob}")
        print()
