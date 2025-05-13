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
# User-Prompt-Templates Zero-Shot
# ──────────────────────────────────────────────────────────────────────────────

# Schritt 1: Aspekt Extraction
USER_TEMPLATE_ASPECT_ZEROSHOT = Template("Review:\n$review")

# Schritt 2: Polarity Extraction
USER_TEMPLATE_POLARITY_ZEROSHOT = Template("Review:\n$review\nAspects: $aspects")

# Schritt 3: Validation of Results
USER_TEMPLATE_VAL_ZEROSHOT = Template("Review:\n$review\nPolarities: $polarities")

# ──────────────────────────────────────────────────────────────────────────────
# User-Prompt-Templates Few-Shot
# ──────────────────────────────────────────────────────────────────────────────

# Schritt 1: Aspekt Extraction
USER_TEMPLATE_ASPECT_FEWSHOT = Template("""\
Beispiel 1:
Review: "Die Verpackung war beschädigt, aber der Geschmack des Produkts war hervorragend."
Antwort: VERPACKUNG, GESCHMACK

Beispiel 2:
Review: "Ich finde den Preis zu hoch und die Qualität enttäuschend."
Antwort: PREIS, QUALITÄT

Beispiel 3:
Review: "Ich habe bisher Meersalz verwendet und möchte dieses für das Jod untermischen."
Antwort: SONSTIGES

Jetzt du:
Review: "$review"
Antwort:""")

# Schritt 2: Polarity Extraction
USER_TEMPLATE_POLARITY_FEWSHOT = Template("""\
Beispiel 1:
Review: "Die Verpackung war beschädigt, aber der Geschmack des Produkts war hervorragend."
Aspekte: VERPACKUNG, GESCHMACK
Antwort: [('VERPACKUNG', 'NEGATIVE'), ('GESCHMACK', 'POSITIVE')]

Beispiel 2:
Review: "Ich finde den Preis zu hoch und die Qualität enttäuschend."
Aspekte: PREIS, QUALITÄT
Antwort: [('PREIS', 'NEGATIVE'), ('QUALITÄT', 'NEGATIVE')]

Beispiel 3:
Review: "Ich habe bisher Meersalz verwendet und möchte dieses für das Jod untermischen."
Aspekte: SONSTIGES
Antwort: [('SONSTIGES', 'NEUTRAL')]

Jetzt du:
Review: "$review"
Aspekte: $aspects
Antwort:""")

# Schritt 3: Validation of Results
USER_TEMPLATE_VAL_FEWSHOT = Template("""\
Beispiel 1:
Review: "Die Verpackung war beschädigt, aber der Geschmack des Produkts war hervorragend."
Polarities: [('VERPACKUNG', 'NEGATIVE'), ('GESCHMACK', 'POSITIVE')]
Antwort:
VERPACKUNG: NEGATIVE – Die Rezension beschreibt eine beschädigte Verpackung.
GESCHMACK: POSITIVE – Der Geschmack wird als hervorragend bezeichnet.

Beispiel 2:
Review: "Ich finde den Preis zu hoch und die Qualität enttäuschend."
Polarities: [('PREIS', 'NEGATIVE'), ('QUALITÄT', 'NEGATIVE')]
Antwort:
PREIS: NEGATIVE – Der Preis wird als zu hoch kritisiert.
QUALITÄT: NEGATIVE – Die Qualität wird als enttäuschend beschrieben.

Beispiel 3:
Review: "Ich habe bisher Meersalz verwendet und möchte dieses für das Jod untermischen."
Polarities: [('SONSTIGES', 'NEUTRAL')]
Antwort:
SONSTIGES: NEUTRAL – Es wird keine klare positive oder negative Meinung geäußert, sondern eine neutrale Information gegeben.

Jetzt du:
Review: "$review"
Polarities: $polarities
Antwort:""")
