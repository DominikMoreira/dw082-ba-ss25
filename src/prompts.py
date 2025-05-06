from string import Template

# ──────────────────────────────────────────────────────────────────────────────
# System-Prompts
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_ASPECT = (
    "Du bist ein Experte für aspektbasierte Stimmungsanalyse. "
    "Extrahiere alle Aspekte die in der Rezension erwähnt werden und ordne sie jeweils einem der folgenden "
    "Aspektkateogrien zu: GESCHMACK, VERPACKUNG, QUALITÄT, PREIS, LIEFERUNG, SONSTIGES. "
    "Antworte mit einer kommagetrennten Liste."
)

SYSTEM_PROMPT_POLARITY = (
    "Du bist Experte für die Klassifizierung von Stimmungspolaritäten. "
    "Weise anhand der Rezension und der extrahierten Aspekte jedem Aspekt eine Stimmungspolarität "
    "zu (POSITIVE, NEGATIVE, NEUTRAL). "
    "Antworte in dem Format [('Aspekt1', 'Polarität1'), ('Aspekt2', 'Polarität2')]."
)

SYSTEM_PROMPT_VAL = (
    "Du bist ein Spezialist für Stimmungsanalyse. "
    "Überprüfe die in Aufforderung 2 zugewiesenen Polaritäten anhand der Rezension. "
    "Gebe eine kurze Begründung für jedes Paar „Aspekt: Polarität. "
)

# ──────────────────────────────────────────────────────────────────────────────
# User-Prompt-Templates
# ──────────────────────────────────────────────────────────────────────────────

# Schritt 1: Aspekt Extraction
USER_TEMPLATE_ASPECT = Template("Review:\n$review")

# Schritt 2: Polarity Extraction
USER_TEMPLATE_POLARITY = Template("Review:\n$review\nAspects: $aspects")

# Schritt 3: Validation of Results
USER_TEMPLATE_VAL = Template("Review:\n$review\nPolarities: $polarities")
