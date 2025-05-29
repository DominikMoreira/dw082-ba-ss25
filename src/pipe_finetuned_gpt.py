from prompts import (
    SYSTEM_PROMPT_FINETUNED,
    USER_TEMPLATE_ABSA_FINETUNED
)

from openaiAPI import OpenAIClient
import pandas as pd

def extract_first_elements(list_of_tuples):
    """
    Extracts the first element from each tuple in a list of tuples.

    Args:
        list_of_tuples: A list of tuples, where each tuple has at least one element.
                        Example: [('VERPACKUNG', 'NEGATIVE'), ('GESCHMACK', 'POSITIVE')]

    Returns:
        A list containing the first element of each tuple.
        Example: ['VERPACKUNG', 'GESCHMACK']
    """
    if not isinstance(list_of_tuples, list):
        raise TypeError("Input must be a list.")

    first_elements = []
    for item in list_of_tuples:
        if not isinstance(item, tuple):
            print(f"Warning: Item '{item}' is not a tuple and will be skipped.")
            continue
        if len(item) > 0:
            first_elements.append(item[0])
        else:
            print(f"Warning: Empty tuple found and will be skipped.")
            continue

    return first_elements

def run_finetuned(review: str, client: OpenAIClient, model):
    user = USER_TEMPLATE_ABSA_FINETUNED.substitute(review=review)
    response_labels = client.request(model, user, SYSTEM_PROMPT_FINETUNED) # This is expected to be like [('ASPECT', 'POLARITY'), ...]

    # Ensure response_labels is a list, even if the model returns a string representation of a list
    # This might need more robust parsing depending on actual model output format
    if isinstance(response_labels, str):
        try:
            # A more robust parsing might be needed if the string is not perfectly formatted
            import ast
            parsed_labels = ast.literal_eval(response_labels)
            if not isinstance(parsed_labels, list):
                print(f"Warning: Parsed response_labels is not a list: {parsed_labels}. Treating as empty.")
                parsed_labels = []
        except (ValueError, SyntaxError) as e:
            print(f"Warning: Could not parse response_labels string: '{response_labels}'. Error: {e}. Treating as empty.")
            parsed_labels = []
        response_labels = parsed_labels


    extracted_aspects = extract_first_elements(response_labels)

    return {
        "aspects": extracted_aspects,
        "predicted_labels": response_labels
    }

def process_reviews_with_finetuned(df, client, model, text_column='review_body'):
    """
    Process reviews and add predictions to DataFrame.

    Args:
        df: DataFrame with text column and optionally 'true_labels' column
        client: OpenAIClient instance
        model: Model name to use
        text_column: Name of the column containing the text to analyze (default: 'review_body')

    Returns:
        DataFrame with added 'predicted_labels' column
    """
    results = []

    print(f"\nProcessing reviews from column '{text_column}'...")
    for idx, row in df.iterrows():
        print(f"Processing review {idx + 1}/{len(df)}")
        try:
            result = run_finetuned(row[text_column], client, model)
            results.append(result)
        except Exception as e:
            print(f"Error processing review {idx}: {str(e)}")
            results.append({"aspects": [], "predicted_labels": "[]"})

    # Add results to DataFrame
    df_results = df.copy()
    df_results['predicted_labels'] = [r['predicted_labels'] for r in results]
    df_results['extracted_aspects'] = [r['aspects'] for r in results]

    return df_results


if __name__ == "__main__":
    example_data = {
        'review_body': ["Die Verpackung war beschädigt, aber der Geschmack des Produkts war hervorragend."],
        'true_labels': [('VERPACKUNG', 'NEGATIVE'), ('GESCHMACK', 'POSITIVE')]
    }
    df = pd.DataFrame(example_data)
    client = OpenAIClient()
    result = process_reviews_with_finetuned(df, client)
    print(result)