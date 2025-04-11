import spacy
import pandas as pd

def extract_aspects(reviews, sentiments):
    """
    Extracts aspects from a list of reviews using spaCy's noun chunks,
    and adds them along with sentiments to a new DataFrame.

    Args:
        reviews (list): A list of review strings.
        sentiments (list): A list of sentiment values corresponding to the reviews.

    Returns:
        pd.DataFrame: A DataFrame with reviews, their extracted aspects, and sentiments.
    """
    nlp = spacy.load("de_core_news_lg")

    # Create an empty list to store results
    data = []

    for review, sentiment in zip(reviews, sentiments):
        doc = nlp(review)
        # Extract aspects (noun chunks)
        aspects = [chunk.root.text.lower() for chunk in doc.noun_chunks]

        # Append review, its aspects, and sentiment to the data list
        data.append({"review": review, "aspects": aspects, "sentiment": sentiment})

    # Convert the data into a pandas DataFrame
    df = pd.DataFrame(data)
    return df
