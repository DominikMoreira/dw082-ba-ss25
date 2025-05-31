import streamlit as st
import pandas as pd
from openaiAPI import OpenAIClient
from pipe_synchain_zero_shot import process_reviews_with_zero_shot, run_syn_chain_zero_shot
from pipe_synchain_few_shot import process_reviews_with_few_shot, run_syn_chain_few_shot
from pipe_finetuned_gpt import process_reviews_with_finetuned, run_finetuned
from recommender import Recommender

st.set_page_config(page_title="ABSA with OpenAI", layout="wide")
st.title("Review Analysis with OpenAI")

# Instructions section
st.markdown("## Instructions and Tips")

with st.expander("📋 How to Use This Tool", expanded=False):
    st.markdown("""
    ### Instructions:
    1. **Upload a CSV file** containing customer reviews or text data
    2. **Select the text column** that contains the review content for analysis
    3. **Choose your preferred model** from the sidebar
    4. **Choose your preferred Strategy** (Zero-Shot, Few-Shot, or Fine-Tuned) in the sidebar
    5. **Click "Start ABSA on CSV"** to process all reviews and wait for analysis to complete
    6. **Download your results** with visualized Insights

    ### Notes and Tips:
    - For best results, ensure your text column contains clean, readable review text
    - The analysis extracts aspect categories (e.g., taste, packaging, shipment) and their sentiment polarities
    - You can also analyze single reviews using the text input section below
    - Results include aspect extraction, sentiment classification, and AI justification
    - This tool is optimized for food reviews but works with various review types. Note that the predictions are for
                following categories:
    - **Categories:** Geschmack, Verpackung, Qualität, Preis, Lieferung, Sonstiges
    - **File limit:** Currently limited to reasonable file sizes for processing efficiency
    """)

# Sidebar
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("OpenAI API Key", type="password")
    choosen_model = st.selectbox("Model", ["gpt-4.1-nano", "ft:gpt-4.1-nano-2025-04-14:personal:finetuned:BZIxyrTy", "gpt-4.1", "ft:gpt-4.1-2025-04-14:personal:finetuned-e2e-absa:BbvtNEYh", "gpt-4.1-mini", "ft:gpt-4.1-mini-2025-04-14:personal::BdLZBsRL"])
    strategy = st.selectbox("Strategy", ["Zero-Shot", "Few-Shot", "Fine-Tuned"])
    with st.expander("ℹ️ About", expanded=False):
        st.markdown("""
        This tool is the practical realization of a Bachelor's thesis focused on Aspect-Based Sentiment Analysis (ABSA) using OpenAI models.
        It demonstrates how modern language models can be leveraged to extract aspect categories and sentiment from customer reviews.
        The project was developed as part of the requirements for the Bachelor's degree and showcases applied research in Natural Language Processing.
        """)

# Main content
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())
    text_column = st.selectbox("Select the text column containing the Review", df.columns)
    client = OpenAIClient()
    # client = OpenAIClient(api_key)

    if strategy == "Zero-Shot":
        st.write("You have selected the Zero-Shot strategy.")
        if st.button("Start ABSA on CSV"):
            st.write("Processing CSV file...")
            results = process_reviews_with_zero_shot(df, client, model=choosen_model, text_column=text_column)
            # results = pd.read_csv('data/results/streamlit.csv') # XXX Placeholder for actual processing

            # Save results
            results.to_csv('data/results/streamlit.csv', index=False)

            # Create diagram
            visualizer = Recommender(csv_filepath="data/results/streamlit.csv")
            visualizer.create_diagram(label_column='predicted_labels', output_path='data/results/streamlit_diagram.png')

            # Visualize results in the Streamlit frontend
            st.subheader("ABSA Results")
            st.image('data/results/streamlit_diagram.png')

    elif strategy == "Few-Shot":
        st.write("You have selected the Few-Shot strategy.")
        if st.button("Start ABSA on CSV"):
            st.write("Processing CSV file...")
            results = process_reviews_with_few_shot(df, client, model=choosen_model, text_column=text_column)
            # results = pd.read_csv('data/results/streamlit.csv') # XXX Placeholder for actual processing

            # Save results
            results.to_csv('data/results/streamlit.csv', index=False)

            # Create diagram
            visualizer = Recommender(csv_filepath="data/results/streamlit.csv")
            visualizer.create_diagram(label_column='predicted_labels', output_path='data/results/streamlit_diagram.png')

            # Visualize results in the Streamlit frontend
            st.subheader("ABSA Results")
            st.image('data/results/streamlit_diagram.png')

    elif strategy == "Fine-Tuned":
        st.write("You have selected the Fine-Tuned strategy.")
        if st.button("Start ABSA on CSV"):
            st.write("Processing CSV file...")
            results = process_reviews_with_finetuned(df, client, model=choosen_model, text_column=text_column)
            # results = pd.read_csv('data/results/streamlit.csv') # XXX Placeholder for actual processing

            # Save results
            results.to_csv('data/results/streamlit.csv', index=False)

            # Create diagram
            visualizer = Recommender(csv_filepath="data/results/streamlit.csv")
            visualizer.create_diagram(label_column='predicted_labels', output_path='data/results/streamlit_diagram.png')

            # Visualize results in the Streamlit frontend
            st.subheader("ABSA Results")
            st.image('data/results/streamlit_diagram.png')
    else:
        st.write("Please select a valid strategy from the sidebar.")



# Text input for single review analysis
st.subheader("Or analyze a single review")
review_text = st.text_area("Enter your review text:", placeholder="Type your review here...", height=100)

if review_text:
    if strategy == "Zero-Shot":
        st.write("You have selected the Zero-Shot strategy for single review analysis.")
        if st.button("Start ABSA"):
            st.write("ABSA analysis started...")
            client = OpenAIClient()
            result = run_syn_chain_zero_shot(review_text, client, model=choosen_model)
            st.subheader("ABSA Result")
            st.write("**Extracted Aspect Category:**", result['aspects'])
            st.write("**Predicted Labels:**", result['predicted_labels'])
            st.write("**Justification:**", result['justification'])
    elif strategy == "Few-Shot":
        st.write("You have selected the Few-Shot strategy for single review analysis.")
        if st.button("Start ABSA"):
            st.write("ABSA analysis started...")
            client = OpenAIClient()
            result = run_syn_chain_few_shot(review_text, client, model=choosen_model)
            st.subheader("ABSA Result")
            st.write("**Extracted Aspect Category:**", result['aspects'])
            st.write("**Predicted Labels:**", result['predicted_labels'])
            st.write("**Justification:**", result['justification'])
    elif strategy == "Fine-Tuned":
        st.write("You have selected the Fine-Tuned strategy for single review analysis.")
        if st.button("Start ABSA"):
            st.write("ABSA analysis started...")
            client = OpenAIClient()
            result = run_finetuned(review_text, client, model=choosen_model)
            st.subheader("ABSA Result")
            st.write("**Extracted Aspect Category:**", result['aspects'])
            st.write("**Predicted Labels:**", result['predicted_labels'])
    else:
        st.write("Please select a valid strategy from the sidebar.")
