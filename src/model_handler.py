#  Lädt und verwaltet die ABSA-Modelle von Hugging Face.

from transformers import AutoModel, AutoTokenizer
import torch

class Model:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """
        Loads the model from Hugging Face.

        Note: For pre-trained models, it is crucial to use the same tokenizer that was used
        during training. Changing the tokenizer is equivalent to changing the vocabulary,
        which can lead to misinterpretation by the model. AutoTokenizer automatically detects
        and loads the appropriate tokenizer (e.g., WordPiece for DistilBERT-base-uncased).
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)

    def tokenize(self, batch):
            """
            Tokenizes the reviews.

            Args:
                batch: dataframe containing the reviews
                (contains a column named 'cleaned_review')

            Returns:
                Dictionary containing input_ids, attention_mask, etc.
            """
            batch_str = [str(i) for i in batch['cleaned_review'].values]

            if self.tokenizer is None:
                raise ValueError("Tokenizer not loaded. Call load_model() first.")

            return self.tokenizer(
                batch_str,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )

    def predict(self, texts):
        """
        Makes predictions for the given texts.

        Args:
            texts: List of strings to predict

        Returns:
            List of predicted class indices
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model or tokenizer not loaded. Call load_model() first.")

        self.model.eval()  # Set to evaluation mode
        with torch.no_grad():
            inputs = self.tokenize(texts)
            outputs = self.model(**inputs)
            predictions = outputs.logits.argmax(dim=-1)

        return predictions.tolist()

def get_model(model_name):
    """Hauptfunktion zum Erstellen und Laden eines Modells."""
    model = Model(model_name)
    model.load_model()
    return model
