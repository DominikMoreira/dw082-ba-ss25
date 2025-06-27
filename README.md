# KI-gestützte Sentiment Analyse für Lebensmittelartikel

Eine End-to-End-Pipeline zur aspektbasierten Sentiment-Analyse (ABSA) deutschsprachiger Kundenbewertungen im E-Commerce Bereich, entwickelt mit modernen Large Language Models.

📖 **Über das Projekt** Diese Bachelorarbeit untersucht die Anwendbarkeit von Large Language Models (LLMs) für die aspektbasierte Sentimentanalyse deutschsprachiger Kundenbewertungen. Das System ermöglicht eine vollautomatisierte Analyse von Kundenbewertungen und generiert konkrete Handlungsempfehlungen für Produktverbesserungen, wodurch Unternehmen wertvolle Einblicke in die Wahrnehmung einzelner Produktaspekte gewinnen können.

## Kernfeatures

*   **Automatische Aspekt-Extraktion:** Identifizierung relevanter Produktaspekte (Geschmack, Verpackung, Qualität, Preis, Lieferung, Sonstiges)
*   **Sentiment-Klassifikation:** Bestimmung der Polarität (positiv, negativ, neutral) für jeden erkannten Aspekt
*   **End-to-End Pipeline:** Vollständige Verarbeitung von CSV-Dateien bis zur Visualisierung
*   **Multi-Modell Support:** Vergleich verschiedener GPT-Varianten mit Zero-Shot, Few-Shot und Finetuning-Strategien
*   **Handlungsempfehlungen:** Automatische Generierung konkreter Verbesserungsvorschläge
*   **Web-Interface:** Benutzerfreundliche Streamlit-basierte GUI

## 📊 Ergebnisse
Das feingetunte GPT-4.1-nano Modell erreichte einen F1-Score von 0,78 und übertraf damit andere getestete Modelle bei der End-to-End ABSA-Aufgabe. Die Evaluation zeigt:

*   Finetuning übertrifft Few-Shot und Zero-Shot Strategien deutlich
*   Kleinere Modelle (GPT-4.1-nano) zeigen überraschend starke Performance
*   Kosteneffizienz: GPT-4.1-nano bietet das beste Preis-Leistungs-Verhältnis (9823.68 F1/USD)
*   Vergleichbare Leistung zu etablierten BERT-basierten Ansätzen (gbert-base: 0.74, gbert-large: 0.82)

## 🚀 Installation
### Voraussetzungen

*   Python 3.11 oder höher
*   OpenAI API-Schlüssel
*   UV

### Setup

1. **Repository klonen**
    ```bash
    git clone https://github.com/DominikWunderlich/dw082-ba-ss25
    cd dw082-ba-ss25
    ```

2. **Python-Umgebung einrichten**
    ```
    uv venv
    source .venv/bin/activate  # Linux/macOS
    ```


3. **Abhängigkeiten installieren**
    ```bash
    uv sync
    ```

4. **OpenAI API-Schlüssel konfigurieren**
    Lege eine `.env`-Datei im Projektverzeichnis an:
    ```
    OPENAI_API_KEY=dein-api-schluessel
    ```

5. **Streamlit-App starten**
    ```bash
    streamlit run src/streamlit.py
    ```

## 🤝 Mitwirkende

*   **Autor:** Dominik Abilio Wunderlich
*   **Betreuer:** Prof. Dr. Jan Kirenz, Prof. Dr. Hendrik Meth
*   **Institution:** Hochschule der Medien Stuttgart
*   **Studiengang:** Wirtschaftsinformatik und digitale Medien

## 📝 Zitation

```
@thesis{wunderlich2025absa,
  title={KI-gestützte Sentiment Analyse für Lebensmittelartikel: Wie NLP Kundenmeinungen entschlüsselt und Produktverbesserungen ermöglicht},
  author={Wunderlich, Dominik Abilio},
  year={2025},
  school={Hochschule der Medien Stuttgart},
  type={Bachelorarbeit}
}
```

## 🆘 Support
Bei Fragen oder Problemen:

*   Prüfen Sie die Issues
*   Erstellen Sie ein neues Issue mit detaillierter Beschreibung


> **Hinweis:** Diese Arbeit wurde im Rahmen einer Bachelorarbeit an der Hochschule der Medien Stuttgart erstellt und dient primär Forschungszwecken. Für den produktiven Einsatz sollten die Limitationen (siehe Kapitel 6.2 der Arbeit) berücksichtigt werden.