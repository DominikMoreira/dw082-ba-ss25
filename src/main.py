# Orchestrates the entire process.
import pandas as pd

from pipe_synchain_zero_shot import process_reviews_with_zero_shot
from pipe_synchain_few_shot import process_reviews_with_few_shot
from pipe_finetuned_gpt import process_reviews_with_finetuned
from openaiAPI import OpenAIClient
from data_loader import load_data
from evaluate import ABSAEvaluator
from recommender import Recommender
from performance_tracker import PerformanceTracker
import time

def main():
# ──────────────────────────────────────────────────────────────────────────────
# Start of the program
# ──────────────────────────────────────────────────────────────────────────────
    # Initialize performance tracker
    tracker = PerformanceTracker()

    print(" ============= Starting Programm =============")
    print(" ---=== Load Dataframe start ===--- ")
    # Load Pandas DataFrame containing the data from CSV file
    df = load_data()

    # Create subset with only required columns and first 10 rows
    # df_subset_for_testing = df[['review_id', 'review_body', 'true_labels']][0:3].copy()
    # print("Test subset shape:", df_subset_for_testing.shape)
    print(" ---=== Load Dataframe finished ===--- ")

# ──────────────────────────────────────────────────────────────────────────────
# ZEROSHOT PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

    print(" ---=== Zero-Shot Pipeline start ===--- ")
    start_time = time.time()

    client = OpenAIClient()
    results_df_zero = process_reviews_with_zero_shot(df, client, model="gpt-4.1-nano")

    # Save results
    results_df_zero.to_csv('data/results/zero_shot_results.csv', index=False)

    # Evaluate results
    evaluator = ABSAEvaluator('data/results/zero_shot_results.csv')
    evaluator.save("data/results/eval_synchain_zero_shot.json")

    processing_time = time.time() - start_time

    # Track performance
    tracker.add_run(
        pipeline_name="Zero-Shot SynChain",
        model_name="gpt-4.1-nano",
        evaluation_file="data/results/eval_synchain_zero_shot.json",
        dataset_size=len(df),
        processing_time=processing_time
    )

    print(f"Zero-Shot Pipeline finished in {processing_time:.2f}s")

# ──────────────────────────────────────────────────────────────────────────────
# FEWSHOT PIPELINE
# ──────────────────────────────────────────────────────────────────────────────
    print(" ---=== Few-Shot Pipeline start ===--- ")
    start_time = time.time()

    client = OpenAIClient()
    results_df_few = process_reviews_with_few_shot(df, client, model="gpt-4.1-nano")

    # Save results
    results_df_few.to_csv('data/results/few_shot_results.csv', index=False)

    # Evaluate results
    evaluator = ABSAEvaluator('data/results/few_shot_results.csv')
    evaluator.save("data/results/eval_synchain_few_shot.json")

    processing_time = time.time() - start_time

    # Track performance
    tracker.add_run(
        pipeline_name="Few-Shot SynChain",
        model_name="gpt-4.1-nano",
        evaluation_file="data/results/eval_synchain_few_shot.json",
        dataset_size=len(df),
        processing_time=processing_time
    )

    print(f"Few-Shot Pipeline finished in {processing_time:.2f}s")

# ──────────────────────────────────────────────────────────────────────────────
# FINETUNED OPENAI PIPELINE
# ──────────────────────────────────────────────────────────────────────────────
    print(" ---=== Finetuned Pipeline start ===--- ")
    start_time = time.time()

    client = OpenAIClient()
    results_df_fine = process_reviews_with_finetuned(df, client, model="ft:gpt-4.1-nano-2025-04-14:personal:finetuned:BZIxyrTy")

    # Save results
    results_df_fine.to_csv('data/results/finetuned_openai_results.csv', index=False)

    # Evaluate results
    evaluator = ABSAEvaluator('data/results/finetuned_openai_results.csv')
    evaluator.save("data/results/eval_finetuned.json")

    processing_time = time.time() - start_time

    # Track performance
    tracker.add_run(
        pipeline_name="Fine-Tuned OpenAI",
        model_name="ft:gpt-4.1-nano-2025-04-14:personal:finetuned:BZIxyrTy",
        evaluation_file="data/results/eval_finetuned.json",
        dataset_size=len(df),
        processing_time=processing_time
    )

    print(f"Finetuned Pipeline finished in {processing_time:.2f}s")

    # Print overall performance summary
    tracker.print_summary()

# ──────────────────────────────────────────────────────────────────────────────
# Create diagram
# ──────────────────────────────────────────────────────────────────────────────
    print(" ---=== Create diagrams start ===--- ")
    visualizer = Recommender(csv_filepath="data/results/finetuned_openai_results.csv")
    print("Generating diagram for 'predicted_labels'...")
    visualizer.create_diagram(label_column='predicted_labels')
    # Example: Create and save the diagram using 'predicted_labels' to a file
    # print("\nGenerating diagram for 'predicted_labels' and saving to file...")
    # visualizer.create_diagram(label_column='predicted_labels', output_path='data/results/predicted_sentiments_diagram.png')
    print(" ---=== Create diagrams finished ===--- ")

# ──────────────────────────────────────────────────────────────────────────────
# End of the program
# ──────────────────────────────────────────────────────────────────────────────

    # # Split the data into training, test, and validation sets
    # train_df, test_df, val_df = split_data(df)

    # initlialize model
    # model = get_model('distilbert/distilbert-base-uncased')
    # print(" ============= Model initialized =============")

    # print(" ============= Start tokenizing data =============")
    # train_encodings = model.tokenize(train_df)
    # test_encodings = model.tokenize(test_df)
    # val_encodings = model.tokenize(val_df)
    # print(train_encodings)
    # print(" ============= Finished tokenizing =============")

    # # Vorhersagen machen
    # predictions = model.predict(tokenized_reviews)

    # # Evaluieren (angenommen, 'label' ist die Zielvariable in Ihrem DataFrame)
    # results = evaluate_model(predictions, df['label'].tolist())

    # # Ergebnisse speichern
    # save_results(results, 'bert-base-uncased')

    # print("Evaluierung abgeschlossen. Ergebnisse wurden gespeichert.")

if __name__ == "__main__":
    main()
