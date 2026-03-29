import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "model/sentiment_model"


class SentimentApp:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()

    def predict(self, text: str):
        if not text or not text.strip():
            return "Пустой ввод", {}

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

        scores = {
            self.model.config.id2label[i]: float(probabilities[i])
            for i in range(len(probabilities))
        }

        return predicted_label, scores


app_model = SentimentApp(MODEL_PATH)

interface = gr.Interface(
    fn=app_model.predict,
    inputs=gr.Textbox(
        lines=4,
        placeholder="Введите текст на русском или немецком языке...",
        label="Текст"
    ),
    outputs=[
        gr.Textbox(label="Предсказанная тональность"),
        gr.Label(label="Вероятности классов")
    ],
    title="Анализ тональности текста",
    description="Прототип системы классификации эмоциональной окраски русских и немецких текстов."
)

if __name__ == "__main__":
    interface.launch()
