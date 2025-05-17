import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast
from collections import defaultdict # Added import


class Recommender:
    def __init__(self, csv_filepath):
        self.csv_filepath = csv_filepath
        self.data = None
        self.aspect_sentiments = defaultdict(lambda: defaultdict(int)) # Changed initialization


    def _load_and_process_data(self, label_column=""):

        # Load the CSV file
        try:
            self.data = pd.read_csv(self.csv_filepath)
        except FileNotFoundError:
            print(f"Error: File not found at {self.csv_filepath}")
            return False
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return False

        if label_column not in self.data.columns:
            print(f"Error: Column '{label_column}' not found in the CSV.")
            return False

        self.aspect_sentiments.clear() # Clear previous data if any

        for index, row in self.data.iterrows():
            try:
                labels_str = row[label_column]
                if pd.isna(labels_str) or not isinstance(labels_str, str):
                    continue

                # ast.literal_eval safely evaluates a string containing a Python literal
                labels_list = ast.literal_eval(labels_str)

                if not isinstance(labels_list, list):
                    print(f"Warning: Parsed data is not a list for row {index}: {labels_list}")
                    continue

                for item in labels_list:
                    if isinstance(item, tuple) and len(item) == 2:
                        aspect, sentiment = item
                        if isinstance(aspect, str) and isinstance(sentiment, str):
                            # Normalize to uppercase for consistency
                            self.aspect_sentiments[aspect.upper()][sentiment.upper()] += 1
                        else:
                            print(f"Warning: Invalid aspect/sentiment type in tuple {item} for row {index}")
                    else:
                        print(f"Warning: Invalid item format {item} in labels_list for row {index}")
            except (ValueError, SyntaxError) as e:
                print(f"Warning: Could not parse labels for row {index}, value: '{row[label_column]}'. Error: {e}")
                continue
            except Exception as e:
                print(f"An unexpected error occurred while processing row {index}: {e}")
                continue

        if not self.aspect_sentiments:
            print(f"No aspect sentiments could be extracted from column '{label_column}'. Check the CSV content.")
            return False
        return True


    def create_diagram(self, label_column='true_labels', output_path=None):
        """
        Creates and displays (or saves) a bar chart of aspect sentiments.

        Args:
            label_column (str): The column to use for sentiment data ('true_labels' or 'predicted_labels').
            output_path (str, optional): If provided, the diagram will be saved to this path.
                                         Otherwise, it will be displayed.
        """
        if not self._load_and_process_data(label_column=label_column):
            print("Data loading and processing failed. Cannot create diagram.")
            return

        aspects = sorted(self.aspect_sentiments.keys())
        if not aspects:
            print("No aspects found to plot.")
            return

        sentiments = ['POSITIVE', 'NEUTRAL', 'NEGATIVE'] # Order for plotting
        sentiment_colors = {'POSITIVE': 'green', 'NEUTRAL': 'blue', 'NEGATIVE': 'red'}

        # Prepare data for plotting
        plot_data = {sentiment: [] for sentiment in sentiments}
        for aspect in aspects:
            for sentiment in sentiments:
                plot_data[sentiment].append(self.aspect_sentiments[aspect].get(sentiment, 0))

        num_aspects = len(aspects)
        bar_width = 0.25

        # Positions of the bars on the X-axis
        r1 = np.arange(num_aspects)
        r2 = [x + bar_width for x in r1]
        r3 = [x + bar_width for x in r2]

        index_positions = np.arange(num_aspects)


        fig, ax = plt.subplots(figsize=(14, 8))

        ax.bar(index_positions - bar_width, plot_data['POSITIVE'], width=bar_width,
               label='Positive', color=sentiment_colors['POSITIVE'])
        ax.bar(index_positions, plot_data['NEUTRAL'], width=bar_width,
               label='Neutral', color=sentiment_colors['NEUTRAL'])
        ax.bar(index_positions + bar_width, plot_data['NEGATIVE'], width=bar_width,
               label='Negative', color=sentiment_colors['NEGATIVE'])

        ax.set_xlabel('Aspect Category', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title(f'Aspect Sentiment Analysis (from {label_column})', fontsize=15, fontweight='bold')

        ax.set_xticks(index_positions)
        ax.set_xticklabels(aspects, rotation=45, ha="right")
        ax.legend()

        ax.grid(axis='y', linestyle='--', alpha=0.7)
        fig.tight_layout() # Adjust layout to make room for labels

        if output_path:
            try:
                plt.savefig(output_path)
                print(f"Diagram saved to {output_path}")
            except Exception as e:
                print(f"Error saving diagram: {e}")
        else:
            plt.show()

if __name__ == '__main__':
    # This is an example of how to use the class.
    # You'll need to adjust the file path to where your CSV is located.
    # The path in the prompt was absolute, so using it directly here for the example.
    csv_file = 'data/results/few_shot_results.csv'

    # Create an instance of the visualizer
    visualizer = Recommender(csv_filepath=csv_file)

    # Create and show the diagram using 'true_labels'
    print("Generating diagram for 'predicted_labels'...")
    visualizer.create_diagram(label_column='predicted_labels')

    # Example: Create and save the diagram using 'predicted_labels' to a file
    # print("\nGenerating diagram for 'predicted_labels' and saving to file...")
    # visualizer.create_diagram(label_column='predicted_labels', output_path='predicted_sentiments_diagram.png')