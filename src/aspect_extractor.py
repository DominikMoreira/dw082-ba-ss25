import spacy
import pandas as pd

def extract_aspects(reviews):
    """
    Extracts aspects from a list of reviews using spaCy's noun chunks
    and adds them to a new column in a DataFrame.

    Args:
        reviews (list): A list of review strings.

    Returns:
        pd.DataFrame: A DataFrame with reviews and their extracted aspects.
    """
    nlp = spacy.load("de_core_news_sm")

    # Create an empty list to store results
    data = []

    for review in reviews:
        doc = nlp(review)
        # Extract aspects (noun chunks)
        aspects = [chunk.root.text.lower() for chunk in doc.noun_chunks]

        # Append review and its aspects to the data list
        data.append({"review": review, "aspects": aspects})

    # Convert the data into a pandas DataFrame
    df = pd.DataFrame(data)
    return df
