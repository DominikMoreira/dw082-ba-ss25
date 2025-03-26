# Orchestriert den gesamten Prozess.

from data_loader import get_data
from model_handler import get_model
from evaluator import evaluate_model, save_results

def main():
    # Daten laden
    df = get_data()

    # Modell laden (Beispiel mit BERT)
    model = get_model('bert-base-uncased')

    # Vorhersagen machen
    predictions = model.predict(df['text'].tolist())

    # Evaluieren (angenommen, 'label' ist die Zielvariable in Ihrem DataFrame)
    results = evaluate_model(predictions, df['label'].tolist())

    # Ergebnisse speichern
    save_results(results, 'bert-base-uncased')

    print("Evaluierung abgeschlossen. Ergebnisse wurden gespeichert.")

if __name__ == "__main__":
    main()
