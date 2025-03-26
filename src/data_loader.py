# Lädt und verarbeitet die Daten aus "reviews.csv".

import pandas as pd

def load_data(file_path='data/raw_reviews.csv'):
    """Lädt die Daten aus der CSV-Datei."""
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    """Führt notwendige Vorverarbeitungsschritte durch."""
    # TODO: Implementierung der Verarbeitungsschritte
    return df

def get_data():
    """Hauptfunktion zum Laden und Vorverarbeiten der Daten."""
    df = load_data()
    df = preprocess_data(df)
    return df
