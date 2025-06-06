import streamlit as st
import pandas as pd
from openaiAPI import OpenAIClient
from pipe_synchain_zero_shot import process_reviews_with_zero_shot, run_syn_chain_zero_shot
from pipe_synchain_few_shot import process_reviews_with_few_shot, run_syn_chain_few_shot
from pipe_finetuned_gpt import process_reviews_with_finetuned, run_finetuned
from recommender import Recommender

st.set_page_config(page_title="ABSA mit OpenAI", layout="wide")
st.title("Aspektbasierte Sentimentanalyse mit OpenAI")

# Instructions section
st.markdown("## Anleitung und Tipps")

with st.expander("📋 So verwenden Sie dieses Tool", expanded=False):
    st.markdown("""
    ### Anleitung:
    1. **Laden Sie eine CSV-Datei hoch**, die Kundenbewertungen enthält
    2. **Wählen Sie die Textspalte aus**, die den zu analysierenden Bewertungsinhalt enthält
    3. **Wählen Sie Ihr bevorzugtes Modell** in der Seitenleiste
    4. **Wählen Sie Ihre bevorzugte Strategie** (Zero-Shot, Few-Shot oder Fine-Tuned) in der Seitenleiste
    5. **Klicken Sie auf "ABSA für CSV starten"**, um alle Bewertungen zu verarbeiten und warten Sie, bis die Analyse abgeschlossen ist
    6. **Laden Sie Ihre Ergebnisse** mit visualisierten Erkenntnissen

    ### Hinweise und Tipps:
    - Für beste Ergebnisse stellen Sie sicher, dass Ihre Textspalte sauberen, lesbaren Bewertungstext enthält
    - Die Analyse extrahiert Aspektkategorien (z.B. Geschmack, Verpackung, Versand) und deren Sentiment-Polaritäten
    - Sie können auch einzelne Bewertungen mit dem Texteingabebereich unten analysieren
    - Die Ergebnisse umfassen Aspektextraktion, Sentimentklassifikation und KI-Begründung
    - Dieses Tool ist für Lebensmittelbewertungen optimiert, funktioniert aber auch mit anderen Bewertungstypen.
      Beachten Sie, dass die Vorhersagen für folgende Kategorien sind:
    - **Kategorien:** Geschmack, Verpackung, Qualität, Preis, Lieferung, Sonstiges
    - **Dateibegrenzung:** Derzeit auf angemessene Dateigrößen für Verarbeitungseffizienz begrenzt
    """)

# Sidebar
with st.sidebar:
    st.header("Einstellungen")
    api_key = st.text_input("OpenAI API Schlüssel", type="password")
    choosen_model = st.selectbox("Modell", ["gpt-4.1-nano", "ft:gpt-4.1-nano-2025-04-14:personal:finetuned:BZIxyrTy", "gpt-4.1", "ft:gpt-4.1-2025-04-14:personal:finetuned-e2e-absa:BbvtNEYh", "gpt-4.1-mini", "ft:gpt-4.1-mini-2025-04-14:personal::BdLZBsRL"])
    strategy = st.selectbox("Strategie", ["Zero-Shot", "Few-Shot", "Fine-Tuned"])
    with st.expander("ℹ️ Über dieses Tool", expanded=False):
        st.markdown("""
        Dieses Tool ist die praktische Umsetzung einer Bachelorarbeit mit Fokus auf aspektbasierte Sentimentanalyse (ABSA)
        unter Verwendung von OpenAI-Modellen. Es demonstriert, wie moderne Sprachmodelle zur Extraktion von Aspektkategorien
        und Sentiment aus Kundenbewertungen eingesetzt werden können. Das Projekt wurde als Teil der Anforderungen für den
        Bachelor-Abschluss entwickelt und zeigt angewandte Forschung im Bereich der natürlichen Sprachverarbeitung auf.
        """)

# Main content
uploaded_file = st.file_uploader("CSV-Datei hochladen", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())
    text_column = st.selectbox("Wählen Sie die Textspalte mit den Bewertungen aus", df.columns)
    client = OpenAIClient()
    # client = OpenAIClient(api_key)

    if strategy == "Zero-Shot":
        st.write("Sie haben die Zero-Shot-Strategie ausgewählt.")
        if st.button("ABSA für CSV starten"):
            st.write("CSV-Datei wird verarbeitet...")
            # results = process_reviews_with_zero_shot(df, client, model=choosen_model, text_column=text_column)
            results = pd.read_csv('data/results/streamlit.csv') # XXX Platzhalter für tatsächliche Verarbeitung

            # Ergebnisse speichern
            results.to_csv('data/results/streamlit.csv', index=False)

            # Diagramm erstellen
            visualizer = Recommender(csv_filepath="data/results/streamlit.csv")
            visualizer.create_diagram(label_column='predicted_labels', output_path='data/results/streamlit_diagram.png')

            # Ergebnisse im Streamlit-Frontend visualisieren
            st.subheader("ABSA-Ergebnisse")
            st.image('data/results/streamlit_diagram.png')
            # Empfehlungen generieren und anzeigen
            recommendations = visualizer.generate_recommendations(client, model=choosen_model)
            st.subheader("Handlungsempfehlungen")
            st.write(recommendations)

    elif strategy == "Few-Shot":
        st.write("Sie haben die Few-Shot-Strategie ausgewählt.")
        if st.button("ABSA für CSV starten"):
            st.write("CSV-Datei wird verarbeitet...")
            # results = process_reviews_with_few_shot(df, client, model=choosen_model, text_column=text_column)
            results = pd.read_csv('data/results/streamlit.csv') # XXX Platzhalter für tatsächliche Verarbeitung

            # Ergebnisse speichern
            results.to_csv('data/results/streamlit.csv', index=False)

            # Diagramm erstellen
            visualizer = Recommender(csv_filepath="data/results/streamlit.csv")
            visualizer.create_diagram(label_column='predicted_labels', output_path='data/results/streamlit_diagram.png')

            # Ergebnisse im Streamlit-Frontend visualisieren
            st.subheader("ABSA-Ergebnisse")
            st.image('data/results/streamlit_diagram.png')

            # Empfehlungen generieren und anzeigen
            recommendations = visualizer.generate_recommendations(client, model=choosen_model)
            st.subheader("Handlungsempfehlungen")
            st.write(recommendations)

    elif strategy == "Fine-Tuned":
        st.write("Sie haben die Fine-Tuned-Strategie ausgewählt.")
        if st.button("ABSA für CSV starten"):
            st.write("CSV-Datei wird verarbeitet...")
            results = process_reviews_with_finetuned(df, client, model=choosen_model, text_column=text_column)
            # results = pd.read_csv('data/results/streamlit.csv') # XXX Platzhalter für tatsächliche Verarbeitung

            # Ergebnisse speichern
            results.to_csv('data/results/streamlit.csv', index=False)

            # Diagramm erstellen
            visualizer = Recommender(csv_filepath="data/results/streamlit.csv")
            visualizer.create_diagram(label_column='predicted_labels', output_path='data/results/streamlit_diagram.png')

            # Ergebnisse im Streamlit-Frontend visualisieren
            st.subheader("ABSA-Ergebnisse")
            st.image('data/results/streamlit_diagram.png')

            # Empfehlungen generieren und anzeigen
            recommendations = visualizer.generate_recommendations(client, model=choosen_model)
            st.subheader("Handlungsempfehlungen")
            st.write(recommendations)
    else:
        st.write("Bitte wählen Sie eine gültige Strategie in der Seitenleiste aus.")



# Texteingabe für Einzelbewertungsanalyse
st.subheader("Oder analysieren Sie eine einzelne Bewertung")
review_text = st.text_area("Geben Sie Ihren Bewertungstext ein:", placeholder="Geben Sie hier Ihre Bewertung ein...", height=100)

if review_text:
    if strategy == "Zero-Shot":
        st.write("Sie haben die Zero-Shot-Strategie für die Einzelbewertungsanalyse ausgewählt.")
        if st.button("ABSA starten"):
            st.write("ABSA-Analyse gestartet...")
            client = OpenAIClient()
            result = run_syn_chain_zero_shot(review_text, client, model=choosen_model)
            st.subheader("ABSA-Ergebnis")
            st.write("**Extrahierte Aspektkategorie:**", result['aspects'])
            st.write("**Vorhergesagte Labels:**", result['predicted_labels'])
            st.write("**Begründung:**", result['justification'])
    elif strategy == "Few-Shot":
        st.write("Sie haben die Few-Shot-Strategie für die Einzelbewertungsanalyse ausgewählt.")
        if st.button("ABSA starten"):
            st.write("ABSA-Analyse gestartet...")
            client = OpenAIClient()
            result = run_syn_chain_few_shot(review_text, client, model=choosen_model)
            st.subheader("ABSA-Ergebnis")
            st.write("**Extrahierte Aspektkategorie:**", result['aspects'])
            st.write("**Vorhergesagte Labels:**", result['predicted_labels'])
            st.write("**Begründung:**", result['justification'])
    elif strategy == "Fine-Tuned":
        st.write("Sie haben die Fine-Tuned-Strategie für die Einzelbewertungsanalyse ausgewählt.")
        if st.button("ABSA starten"):
            st.write("ABSA-Analyse gestartet...")
            client = OpenAIClient()
            result = run_finetuned(review_text, client, model=choosen_model)
            st.subheader("ABSA-Ergebnis")
            st.write("**Extrahierte Aspektkategorie:**", result['aspects'])
            st.write("**Vorhergesagte Labels:**", result['predicted_labels'])
    else:
        st.write("Bitte wählen Sie eine gültige Strategie in der Seitenleiste aus.")
