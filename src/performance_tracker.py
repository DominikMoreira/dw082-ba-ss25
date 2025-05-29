import json
import pandas as pd
from datetime import datetime
import os

class PerformanceTracker:
    def __init__(self, results_dir='data/results'):
        self.results_dir = results_dir
        self.performance_file = os.path.join(results_dir, 'performance_summary.json')
        self.performance_data = self.load_performance_data()

    def load_performance_data(self):
        """Load existing performance data or create new structure"""
        if os.path.exists(self.performance_file):
            with open(self.performance_file, 'r') as f:
                return json.load(f)
        return {"runs": [], "summary": {}}

    def add_run(self, pipeline_name, model_name, evaluation_file, dataset_size, processing_time=None):
        """Add a new pipeline run to tracking"""

        # Load evaluation results
        with open(evaluation_file, 'r') as f:
            eval_data = json.load(f)

        run_data = {
            "timestamp": datetime.now().isoformat(),
            "pipeline": pipeline_name,
            "model": model_name,
            "dataset_size": dataset_size,
            "processing_time": processing_time,
            "metrics": eval_data
        }

        self.performance_data["runs"].append(run_data)
        self.save_performance_data()

        # Update summary
        self.update_summary()

    def update_summary(self):
        """Update performance summary with latest results"""
        summary = {}

        for run in self.performance_data["runs"]:
            pipeline = run["pipeline"]
            if pipeline not in summary:
                summary[pipeline] = {
                    "best_aspect_f1": 0,
                    "best_end2end_f1": 0,
                    "best_aspect_accuracy": 0,
                    "best_end2end_accuracy": 0,
                    "best_polarity_accuracy": 0,
                    "runs_count": 0,
                    "avg_processing_time": 0,
                    "avg_time_per_review": 0,
                    "last_run": None
                }

            summary[pipeline]["runs_count"] += 1
            summary[pipeline]["last_run"] = run["timestamp"]

            # Extract metrics using the correct key names
            metrics = run["metrics"]

            # Update best scores
            summary[pipeline]["best_aspect_f1"] = max(
                summary[pipeline]["best_aspect_f1"],
                metrics.get("aspect_f1", 0)
            )
            summary[pipeline]["best_end2end_f1"] = max(
                summary[pipeline]["best_end2end_f1"],
                metrics.get("end2end_f1", 0)
            )
            summary[pipeline]["best_aspect_accuracy"] = max(
                summary[pipeline]["best_aspect_accuracy"],
                metrics.get("aspect_accuracy_emr", 0)
            )
            summary[pipeline]["best_end2end_accuracy"] = max(
                summary[pipeline]["best_end2end_accuracy"],
                metrics.get("end2end_accuracy_emr", 0)
            )
            summary[pipeline]["best_polarity_accuracy"] = max(
                summary[pipeline]["best_polarity_accuracy"],
                metrics.get("polarity_accuracy", 0)
            )

            # Track processing time
            if run["processing_time"]:
                current_avg = summary[pipeline]["avg_processing_time"]
                count = summary[pipeline]["runs_count"]
                summary[pipeline]["avg_processing_time"] = (current_avg * (count-1) + run["processing_time"]) / count

                # Calculate time per review for this run
                time_per_review = run["processing_time"] / run["dataset_size"] if run["dataset_size"] > 0 else 0
                current_avg_per_review = summary[pipeline]["avg_time_per_review"]
                summary[pipeline]["avg_time_per_review"] = (current_avg_per_review * (count-1) + time_per_review) / count

        self.performance_data["summary"] = summary
        self.save_performance_data()

    def save_performance_data(self):
        """Save performance data to file"""
        os.makedirs(self.results_dir, exist_ok=True)
        with open(self.performance_file, 'w') as f:
            json.dump(self.performance_data, f, indent=2)

    def print_summary(self):
        """Print performance summary"""
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY")
        print("="*60)

        for pipeline, data in self.performance_data["summary"].items():
            print(f"\n{pipeline.upper()}:")
            print(f"  Runs: {data['runs_count']}")
            print(f"  Best Aspect F1: {data['best_aspect_f1']:.4f}")
            print(f"  Best End2End F1: {data['best_end2end_f1']:.4f}")
            print(f"  Best Aspect Accuracy: {data['best_aspect_accuracy']:.4f}")
            print(f"  Best End2End Accuracy: {data['best_end2end_accuracy']:.4f}")
            print(f"  Best Polarity Accuracy: {data['best_polarity_accuracy']:.4f}")
            print(f"  Avg Processing Time: {data['avg_processing_time']:.2f}s")
            print(f"  Avg Time per Review: {data['avg_time_per_review']:.2f}s")
            print(f"  Last Run: {data['last_run']}")

    def get_comparison_report(self):
        """Generate comparison report across pipelines"""
        df_runs = pd.DataFrame(self.performance_data["runs"])

        if df_runs.empty:
            return "No performance data available yet."

        # Create comparison table with correct metric names
        comparison = df_runs.groupby('pipeline').agg({
            'metrics': lambda x: {
                'avg_aspect_f1': sum(m.get('aspect_f1', 0) for m in x) / len(x),
                'avg_end2end_f1': sum(m.get('end2end_f1', 0) for m in x) / len(x),
                'avg_polarity_accuracy': sum(m.get('polarity_accuracy', 0) for m in x) / len(x)
            },
            'processing_time': 'mean',
            'dataset_size': 'last'
        }).reset_index()

        return comparison