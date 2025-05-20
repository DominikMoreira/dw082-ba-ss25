from prompts import (
    SYSTEM_PROMPT_ASPECT, USER_TEMPLATE_ASPECT_FEWSHOT,
    SYSTEM_PROMPT_POLARITY, USER_TEMPLATE_POLARITY_FEWSHOT,
    SYSTEM_PROMPT_VAL, USER_TEMPLATE_VAL_FEWSHOT
)

from openaiAPI import OpenAIClient

def run_syn_chain_few_shot(review: str, client: OpenAIClient, model):
    # Step 1: Aspekt Extraction
    user1 = USER_TEMPLATE_ASPECT_FEWSHOT.substitute(review=review)
    aspects_raw = client.request(model, user1, SYSTEM_PROMPT_ASPECT)
    aspects = [aspect.strip() for aspect in aspects_raw.split(",") if aspect.strip()]

    # Step 2: Polarity Extraction
    user2 = USER_TEMPLATE_POLARITY_FEWSHOT.substitute(review=review, aspects=", ".join(aspects))
    aspect_polarity = client.request(model, user2, SYSTEM_PROMPT_POLARITY) # [(GESCHMACK: POSITIV), (PREIS: NEGATIV)]

    # Step 3: Validation of Results
    user3 = USER_TEMPLATE_VAL_FEWSHOT.substitute(
        review=review,
        polarities=aspect_polarity
    )
    justification = client.request(model, user3, SYSTEM_PROMPT_VAL)

    return {
        "aspects": aspects,
        "predicted_labels": aspect_polarity,
        "justification": justification
    }

def process_reviews_with_few_shot(df, client, model):
    """
    Process reviews and add predictions to DataFrame.

    Args:
        df: DataFrame with 'review_body' and 'true_labels' columns
        client: OpenAIClient instance

    Returns:
        DataFrame with added 'predicted_labels' column
    """
    results = []

    print("\nProcessing reviews...")
    for idx, row in df.iterrows():
        print(f"Processing review {idx + 1}/{len(df)}")
        try:
            result = run_syn_chain_few_shot(row['review_body'], client, model)
            results.append(result)
        except Exception as e:
            print(f"Error processing review {idx}: {str(e)}")
            results.append({"aspects": [], "predicted_labels": "[]", "justification": ""})

    # Add results to DataFrame
    df_results = df.copy()
    df_results['predicted_labels'] = [r['predicted_labels'] for r in results]
    df_results['extracted_aspects'] = [r['aspects'] for r in results]
    df_results['justification'] = [r['justification'] for r in results]

    return df_results
