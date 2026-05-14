from __future__ import annotations

from typing import Optional


class BehaviorDiscriminator:
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        self.model_path = model_path
        self.available = bool(model_path)
        if not self.available:
            self.tokenizer = None
            self.model = None
            self.device = None
            return

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.model.eval()

    def predict(self, prompts: list[str], label_texts: list[str]) -> tuple[list[str], list[float]]:
        if not self.available:
            raise RuntimeError("BehaviorDiscriminator was initialized without a model_path.")

        import torch

        assert self.tokenizer is not None and self.model is not None and self.device is not None
        with torch.no_grad():
            batch = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            logits = self.model(**batch).logits
            probs = logits.softmax(-1)
            choices = logits.argmax(-1)
        labels = [label_texts[choice.item()] for choice in choices]
        confidences = [prob.item() for prob in probs.max(dim=-1).values]
        return labels, confidences
