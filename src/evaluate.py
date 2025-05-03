# Führt die Evaluierung der Modelle durch.

from sklearn.metrics import precision_recall_fscore_support

class PerformanceBenchmark:
    def __init__(self, pipeline, dataset, optim_type="BERT baseline"):
        self.pipeline = pipeline
        self.dataset = dataset
        self.optim_type = optim_type

    def compute_accuracy(self):
        pass

    def compute_size(self):
        pass

    def time_pipeline(self):
        pass

    def run_benchmarks(self):
        metrics = {}
        metrics[self.optim_type] = self.compute_size()
        metrics[self.optim_type].update(self.time_pipeline())
        metrics[self.optim_type].update(self.compute_accuracy())
        return metrics

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
