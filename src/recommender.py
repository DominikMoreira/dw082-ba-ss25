import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast
from collections import defaultdict


class Recommender:
    def __init__(self, csv_filepath):
        self.csv_filepath = csv_filepath
        self.data = None
        self.aspect_sentiments = defaultdict(lambda: defaultdict(int))


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

        # Define sentiment order and colors
        sentiments = ['POSITIVE', 'NEUTRAL', 'NEGATIVE']
        sentiment_colors = {'POSITIVE': '#2E8B57', 'NEUTRAL': '#4682B4', 'NEGATIVE': '#DC143C'}

        # Create figure with better styling
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(max(12, len(aspects) * 1.5), 8))

        # Prepare data for plotting - only include sentiments that exist for each aspect
        bar_positions = []
        bar_values = []
        bar_colors = []
        bar_labels = []
        tick_positions = []
        tick_labels = []

        current_position = 0
        bar_width = 0.6

        for aspect in aspects:
            aspect_sentiments = self.aspect_sentiments[aspect]
            aspect_has_data = False
            aspect_start = current_position

            # Only plot sentiments that exist for this aspect
            for sentiment in sentiments:
                count = aspect_sentiments.get(sentiment, 0)
                if count > 0:
                    bar_positions.append(current_position)
                    bar_values.append(count)
                    bar_colors.append(sentiment_colors[sentiment])
                    bar_labels.append(sentiment.capitalize())
                    current_position += bar_width + 0.1  # Small gap between bars
                    aspect_has_data = True

            if aspect_has_data:
                # Calculate center position for this aspect's label
                aspect_end = current_position - 0.1
                aspect_center = (aspect_start + aspect_end - bar_width) / 2
                tick_positions.append(aspect_center)
                tick_labels.append(aspect)
                current_position += 0.5  # Larger gap between aspects

        if not bar_positions:
            print("No sentiment data found to plot.")
            return

        # Create the bars
        bars = ax.bar(bar_positions, bar_values, width=bar_width,
                     color=bar_colors, alpha=0.8, edgecolor='white', linewidth=0.7)

        # Add value labels on top of bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                   f'{int(height)}', ha='center', va='bottom',
                   fontsize=10, fontweight='bold')

        # Create custom legend
        legend_elements = []
        for sentiment in sentiments:
            if any(self.aspect_sentiments[aspect].get(sentiment, 0) > 0 for aspect in aspects):
                legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=sentiment_colors[sentiment],
                                                   alpha=0.8, label=sentiment.capitalize()))

        ax.legend(handles=legend_elements, loc='upper right', frameon=True,
                 fancybox=True, shadow=True, fontsize=12, title='Sentiment', title_fontsize=13)

        # Styling improvements
        ax.set_xlabel('Aspect Categories', fontsize=14, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=14, fontweight='bold')
        ax.set_title(f'Aspect-Based Sentiment Analysis Results\n({label_column.replace("_", " ").title()})',
                     fontsize=16, fontweight='bold')

        # Set custom tick positions and labels
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=12)

        # Grid improvements
        ax.grid(axis='y', linestyle='--', alpha=0.3, color='gray')
        ax.set_axisbelow(True)

        # Remove top and right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)

        # Set y-axis to start from 0 and add some padding at the top
        ax.set_ylim(0, max(bar_values) * 1.15)

        # Adjust x-axis limits to fit all bars
        ax.set_xlim(-0.5, max(bar_positions) + bar_width + 0.5)

        # Add manual padding using subplots_adjust instead
        plt.subplots_adjust(bottom=0.15, left=0.1, right=0.95, top=0.9)

        # Add subtle background color
        fig.patch.set_facecolor('#f8f9fa')
        ax.set_facecolor('#ffffff')

        if output_path:
            try:
                plt.savefig(output_path, dpi=300, bbox_inches='tight',
                           facecolor=fig.get_facecolor(), edgecolor='none')
                print(f"Diagram saved to {output_path}")
            except Exception as e:
                print(f"Error saving diagram: {e}")
        else:
            plt.show()

        plt.close()  # Free up memory

# if __name__ == '__main__':
#     csv_file = 'data/results/few_shot_results.csv'
#     visualizer = Recommender(csv_filepath=csv_file)
#     print("Generating diagram for 'predicted_labels'...")
#     visualizer.create_diagram(label_column='predicted_labels')

#     # Example: Create and save the diagram using 'predicted_labels' to a file
#     # print("\nGenerating diagram for 'predicted_labels' and saving to file...")
#     # visualizer.create_diagram(label_column='predicted_labels', output_path='data/results/predicted_sentiments_diagram.png')