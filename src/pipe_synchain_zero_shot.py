from prompts import (
    SYSTEM_PROMPT_ASPECT, USER_TEMPLATE_ASPECT,
    SYSTEM_PROMPT_POLARITY, USER_TEMPLATE_POLARITY,
    SYSTEM_PROMPT_VAL, USER_TEMPLATE_VAL
)

from openaiAPI import OpenAIClient

def run_syn_chain_zero_shot(review: str, client: OpenAIClient):
    # Step 1: Aspekt Extraction
    user1 = USER_TEMPLATE_ASPECT.substitute(review=review)
    aspects_raw = client.request(user1, SYSTEM_PROMPT_ASPECT)
    aspects = [aspect.strip() for aspect in aspects_raw.split(",") if aspect.strip()]

    # Step 2: Polarity Extraction
    user2 = USER_TEMPLATE_POLARITY.substitute(review=review, aspects=", ".join(aspects))
    polarity_str = client.request(user2, SYSTEM_PROMPT_POLARITY)
    polarities = dict(
        item.split(": ")
        for item in polarity_str.strip("{}").split(";")
    )

    # Step 3: Validation of Results
    user3 = USER_TEMPLATE_VAL.substitute(
        review=review,
        polarities="; ".join(f"{k}: {v}" for k,v in polarities.items())
    )
    justification = client.request(user3, SYSTEM_PROMPT_VAL)

    return {
        "aspects": aspects,
        "polarities": polarities,
        "justification": justification
    }

if __name__ == "__main__":
    # Example usage
    review = "Der Salat ist frisch und lecker, aber der Preis ist zu hoch."
    client = OpenAIClient()
    result = run_syn_chain_zero_shot(review, client)
    print(result)