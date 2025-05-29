import streamlit as st
import pandas as pd
from openaiAPI import OpenAIClient
from pipe_synchain_zero_shot import process_reviews_with_zero_shot, run_syn_chain_zero_shot
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
    choosen_model = st.selectbox("Model", ["gpt-4.1-nano", "ft:gpt-4.1-nano-2025-04-14:personal:finetuned:BZIxyrTy"])

# Main content
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())
    text_column = st.selectbox("Select the text column for ABSA", df.columns)

    if st.button("Start ABSA on CSV"):
        st.write("Processing CSV file...")
        # client = OpenAIClient(api_key)
        client = OpenAIClient()
        # results = process_reviews_with_zero_shot(df, client, model=choosen_model, text_column=text_column)
        # TODO: implement text_column handling in process_reviews_with_zero_shot

        # results = process_reviews_with_zero_shot(df, client, model=choosen_model)
        results = pd.read_csv('data/results/streamlit.csv') # Placeholder for actual processing

        # Save results
        results.to_csv('data/results/streamlit.csv', index=False)

        # Create diagram
        visualizer = Recommender(csv_filepath="data/results/streamlit.csv")
        visualizer.create_diagram(label_column='predicted_labels', output_path='data/results/streamlit_diagram.png')

        # Visualize results in the Streamlit frontend
        st.subheader("ABSA Results")
        st.image('data/results/streamlit_diagram.png')


# Text input for single review analysis
st.subheader("Or analyze a single review")
review_text = st.text_area("Enter your review text:", placeholder="Type your review here...", height=100)

if review_text:
    st.write("**Review to analyze:**")
    st.write(review_text)

    if st.button("Start ABSA"):
        st.write("ABSA analysis started...")
        client = OpenAIClient()
        result = run_syn_chain_zero_shot(review_text, client, model=choosen_model)
        st.subheader("ABSA Result")
        st.write("**Extracted Aspect Category:**", result['aspects'])
        st.write("**Predicted Labels:**", result['predicted_labels'])
        st.write("**Justification:**", result['justification'])
