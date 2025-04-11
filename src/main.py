# Orchestrates the entire process.
import pandas as pd

from data_loader import load_data, lemmatize_reviews_in_dataframe, split_data
from aspect_extractor import extract_aspects
from model_handler import get_model
from evaluator import evaluate_model, save_results

def main():
    print(" ============= Starting Programm =============")
    # Load Pandas DataFrame containing the data from CSV file
    df_cleaned_reviews = load_data()
    df_subset_to_lemmatize = df_cleaned_reviews[0:3].copy()
    df_lemmatized_reviews = lemmatize_reviews_in_dataframe(df_subset_to_lemmatize) # XXXXXXXXXX Könnte in notebook umgelagert werden.

    # Extract aspects from the reviews
    df = extract_aspects(
        df_lemmatized_reviews[0:3]['cleaned_review'].values, # XXXXXXXXXX [0:3] for testing
        df_lemmatized_reviews[0:3]['Sentiment'].values) # XXXXXXXXXX [0:3] for testing
    print(" ============= DataFrame with review, aspects and sentiment created =============")

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
