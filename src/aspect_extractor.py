import spacy

def extract_aspects(reviews):
    print("TEST ",type(reviews))
    """
    Extracts aspects from a list of reviews using spaCy's noun chunks.

    Args:
        reviews (list): A list of review strings.

    Returns:
        set: A set of unique aspects extracted from the reviews.
    """

    nlp = spacy.load("de_core_news_sm")
    aspects = set()

    for review in reviews:
        doc = nlp(review)
        for chunk in doc.noun_chunks:
            aspects.add(chunk.root.text.lower())

    print("Extracted Aspects:", aspects)
    return aspects




