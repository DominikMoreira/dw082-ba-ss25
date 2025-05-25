import ast
import json
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

class ABSAEvaluator:
    def __init__(self, path_csv: str):
        self.df = pd.read_csv(path_csv)
        self.records = []

    def _parse_labels(self, label_str: str):
        """
        Wandelt die String-Darstellung in eine Liste von (aspect, polarity)-Tuples um.
        """
        try:
            return ast.literal_eval(label_str)
        except Exception:
            return []

    def _collect(self):
        """
        Erstellt für jede Zeile:
          - true_pairs und pred_pairs als Sets
          - true_aspects, pred_aspects
          - label-Dicts für Polarity
        """
        for _, row in self.df.iterrows():
            true_list = self._parse_labels(row['true_labels'])
            pred_list = self._parse_labels(row['predicted_labels'])

            true_aspects = {asp for asp, _ in true_list}
            pred_aspects = {asp for asp, _ in pred_list}

            true_pairs = set(true_list)
            pred_pairs = set(pred_list)

            record = {
                'true_aspects': true_aspects,
                'pred_aspects': pred_aspects,
                'true_pairs': true_pairs,
                'pred_pairs': pred_pairs
            }
            self.records.append(record)

    def evaluate(self) -> dict:
        """
        Führt alle drei Evaluationsstufen durch und gibt die Metriken zurück.
        """
        self._collect()

        if not self.records: # Handle empty input
            return {
                'aspect_precision' : 0,
                'aspect_recall'    : 0,
                'aspect_f1'        : 0,
                'aspect_accuracy_emr': 0,
                'end2end_precision': 0,
                'end2end_recall'   : 0,
                'end2end_f1'       : 0,
                'end2end_accuracy_emr': 0,
                'polarity_accuracy': 0,
                'polarity_f1_macro': 0
            }

        # Aspekt-Erkennung
        tp_aspect = sum(len(r['pred_aspects'] & r['true_aspects']) for r in self.records)
        fp_aspect = sum(len(r['pred_aspects'] - r['true_aspects']) for r in self.records)
        fn_aspect = sum(len(r['true_aspects'] - r['pred_aspects']) for r in self.records)

        precision_aspect = tp_aspect / (tp_aspect + fp_aspect) if tp_aspect+fp_aspect else 0
        recall_aspect    = tp_aspect / (tp_aspect + fn_aspect) if tp_aspect+fn_aspect else 0
        f1_aspect = (2 * precision_aspect * recall_aspect / (precision_aspect + recall_aspect)
                     if precision_aspect+recall_aspect else 0)

        # Aspect Exact Match Ratio (Accuracy)
        aspect_emr_correct = sum(1 for r in self.records if r['pred_aspects'] == r['true_aspects'])
        aspect_accuracy_emr = aspect_emr_correct / len(self.records)

        # End-to-End Paare
        tp_pair = sum(len(r['pred_pairs'] & r['true_pairs']) for r in self.records)
        fp_pair = sum(len(r['pred_pairs'] - r['true_pairs']) for r in self.records)
        fn_pair = sum(len(r['true_pairs'] - r['pred_pairs']) for r in self.records)

        precision_pair = tp_pair / (tp_pair + fp_pair) if tp_pair+fp_pair else 0
        recall_pair    = tp_pair / (tp_pair + fn_pair) if tp_pair+fn_pair else 0
        f1_pair = (2 * precision_pair * recall_pair / (precision_pair + recall_pair)
                   if precision_pair+recall_pair else 0)

        # End-to-End Pair Exact Match Ratio (Accuracy)
        pair_emr_correct = sum(1 for r in self.records if r['pred_pairs'] == r['true_pairs'])
        end2end_accuracy_emr = pair_emr_correct / len(self.records)

        # Polarity-Klassifikation über alle Paare
        y_true, y_pred = [], []
        for r in self.records:
            for true_aspect, true_pol in r['true_pairs']:
                y_true.append(true_pol)
                # falls vorhanden: Vorhersagepolarity, sonst "NONE"
                pred_pol = dict(r['pred_pairs']).get(true_aspect, 'NONE')
                y_pred.append(pred_pol)

        accuracy_pol = 0
        if y_true: # Ensure y_true is not empty before calling accuracy_score
            accuracy_pol = accuracy_score(y_true, y_pred)

        prec_pol, rec_pol, f1_pol, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0)

        return {
            'aspect_precision' : precision_aspect,
            'aspect_recall'    : recall_aspect,
            'aspect_f1'        : f1_aspect,
            'aspect_accuracy_emr': aspect_accuracy_emr,
            'end2end_precision': precision_pair,
            'end2end_recall'   : recall_pair,
            'end2end_f1'       : f1_pair,
            'end2end_accuracy_emr': end2end_accuracy_emr,
            'polarity_accuracy': accuracy_pol,
            'polarity_f1_macro': f1_pol
        }

    def save(self, out_path: str):
        """
        Speichert das Evaluations-Resultat als JSON.
        """
        metrics = self.evaluate()
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    evaluator = ABSAEvaluator("data/results/finetuned_openai_results.csv")
    evaluator.save("data/results/absa_evaluation.json")
    print("Evaluation abgeschlossen und gespeichert unter data/results/absa_evaluation.json")
