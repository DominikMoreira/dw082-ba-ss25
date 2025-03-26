# hdm-dw082-ba-ss25
Repository zur Bachelor Arbeit

## Projektstruktur
```
hdm-dw082-ba-ss25/
├── data/
│   └── reviews.csv
├── src/
│   ├── main.py - Orchestriert den gesamten Prozess.
│   ├── data_loader.py - Lädt und verarbeitet die Daten aus "reviews.csv".
│   ├── model_handler.py - Lädt und verwaltet die ABSA-Modelle von Hugging Face.
│   └── evaluator.py - Führt die Evaluierung der Modelle durch.
├── results/
├── requirements.txt
└── README.md
```

## Beispiel für einen Evaluierungslauf
So könnte ein typischer Evaluierungslauf aussehen:
