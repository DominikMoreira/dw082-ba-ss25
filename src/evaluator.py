# Führt die Evaluierung der Modelle durch.

from sklearn.metrics import precision_recall_fscore_support

def evaluate_model(predictions, true_labels):
    """Berechnet Evaluierungsmetriken für das Modell."""
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def save_results(results, model_name):
    """Speichert die Evaluierungsergebnisse."""
    with open(f'results/{model_name}_results.txt', 'w') as f:
        for metric, value in results.items():
            f.write(f"{metric}: {value}\n")
