# Orchestriert den gesamten Prozess.
import pandas as pd

from data_loader import load_data, split_data
from aspect_extractor import extract_aspects
from model_handler import get_model
from evaluator import evaluate_model, save_results

def main():
    # Load Pandas DataFrame containing the data from CSV file
    df = load_data()

    # Extract aspects from the reviews
    extract_aspects(df['cleaned_review'].values)
    print("Fertig")

    # # Trainingsdaten aufteilen
    # train_df, test_df, val_df = split_data(df)

    # # Modell laden
    # model = get_model('distilbert/distilbert-base-uncased')

    # print("Starte Tokenisierung ...")
    # # token_val = [str(i) for i in train_df['cleaned_review'].values]
    # train_encodings = model.tokenize(train_df)
    # test_encodings = model.tokenize(test_df)
    # val_encodings = model.tokenize(val_df)
    # print("Tokenisierung abgeschlossen.")

    # # Vorhersagen machen
    # predictions = model.predict(tokenized_reviews)

    # # Evaluieren (angenommen, 'label' ist die Zielvariable in Ihrem DataFrame)
    # results = evaluate_model(predictions, df['label'].tolist())

    # # Ergebnisse speichern
    # save_results(results, 'bert-base-uncased')

    # print("Evaluierung abgeschlossen. Ergebnisse wurden gespeichert.")

if __name__ == "__main__":
    main()
