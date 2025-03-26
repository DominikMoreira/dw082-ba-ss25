#  Lädt und verwaltet die ABSA-Modelle von Hugging Face.

from transformers import AutoModelForSequenceClassification, AutoTokenizer

class ABSAModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Lädt das Modell von Hugging Face."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

    def predict(self, texts):
        """Führt Vorhersagen für die gegebenen Texte durch."""
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model(**inputs)
        return outputs.logits.argmax(dim=1).tolist()

def get_model(model_name):
    """Hauptfunktion zum Erstellen und Laden eines Modells."""
    model = ABSAModel(model_name)
    model.load_model()
    return model
