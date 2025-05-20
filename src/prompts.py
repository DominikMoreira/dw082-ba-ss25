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

SYSTEM_PROMPT_FINETUNED = (
    "Du bist ein Experte für aspektbasierte Stimmungsanalyse und die Klassifizierung von Stimmungspolaritäten. "
    "Extrahiere alle Aspekte, die in der Rezension erwähnt werden und ordne sie jeweils einer der folgenden Aspektkategorien zu: GESCHMACK, VERPACKUNG, QUALITÄT, PREIS, LIEFERUNG, SONSTIGES."
    "Weise anschließend anhand der Rezension und der extrahierten Aspekte jedem Aspekt eine Stimmungspolarität zu (POSITIVE, NEGATIVE, NEUTRAL)."
    "Antworte im Format:"
    "[('Aspektkategorie1', 'Polarität1'), ('Aspektkategorie2', 'Polarität2'), ...]"
    "Beispiel:"
    "[('KUNDENSERVICE', 'NEGATIVE'), ('VERPACKUNG', 'NEGATIVE')]"
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

Beispiel 4:
Review: "Riecht und schmeckt fischig - trinken kann ich das nicht...."
Antwort: GESCHMACK

Beispiel 5:
Review: "Scan-Etikett liest Chicken Pho auf der Spicy Beef Box. Als ich das Klebeband abnahm, war die erste Packung (vermutlich aus Versehen) mit einem Kartonschneider aufgeschnitten worden. Amazon sagt, dass dieser Artikel nicht für eine Rückerstattung oder einen Ersatz in Frage kommt und ich stecke in einer Schleife fest, die mich nicht weiterbringt. Es gibt keine Telefonnummer, die ich anrufen kann, und ich koche dieses Wochenende PHO-Suppe für 50 Leute und habe Angst, mehr zu bestellen und weitere 22 Dollar auszugeben. Total deprimiert."
Antwort: KUNDENSERVICE, VERPACKUNG

Beispiel 6:
Review: "der Tee erzielte bei mir nicht die Wirkung, wie hier in einigen Rezessionen beschrieben, als Tee aber ok. Ein Stoffwechsel, oder abnehmen hat bei mir nicht stattgefunden."
Antwort: QUALITÄT

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

Beispiel 4:
Review: "Riecht und schmeckt fischig - trinken kann ich das nicht...."
Aspekte: GESCHMACK
Antwort: [('GESCHMACK', 'NEGATIVE')]

Beispiel 5:
Review: "Scan-Etikett liest Chicken Pho auf der Spicy Beef Box. Als ich das Klebeband abnahm, war die erste Packung (vermutlich aus Versehen) mit einem Kartonschneider aufgeschnitten worden. Amazon sagt, dass dieser Artikel nicht für eine Rückerstattung oder einen Ersatz in Frage kommt und ich stecke in einer Schleife fest, die mich nicht weiterbringt. Es gibt keine Telefonnummer, die ich anrufen kann, und ich koche dieses Wochenende PHO-Suppe für 50 Leute und habe Angst, mehr zu bestellen und weitere 22 Dollar auszugeben. Total deprimiert."
Aspekte: KUNDENSERVICE, VERPACKUNG
Antwort: [('KUNDENSERVICE', 'NEGATIVE'), ('VERPACKUNG', 'NEGATIVE')]

Beispiel 6:
Review: "der Tee erzielte bei mir nicht die Wirkung, wie hier in einigen Rezessionen beschrieben, als Tee aber ok. Ein Stoffwechsel, oder abnehmen hat bei mir nicht stattgefunden."
Aspekte: QUALITÄT
Antwort: [('QUALITÄT', 'NEUTRAL')]

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

Beispiel 4:
Review: "Riecht und schmeckt fischig - trinken kann ich das nicht...."
Polarities: [('GESCHMACK', 'NEGATIVE')]
Antwort:
GESCHMACK: NEGATIVE – Der Geschmack wird als fischig und ungenießbar beschrieben, daher negativ bewertet.

Beispiel 5:
Review: "Scan-Etikett liest Chicken Pho auf der Spicy Beef Box. Als ich das Klebeband abnahm, war die erste Packung (vermutlich aus Versehen) mit einem Kartonschneider aufgeschnitten worden. Amazon sagt, dass dieser Artikel nicht für eine Rückerstattung oder einen Ersatz in Frage kommt und ich stecke in einer Schleife fest, die mich nicht weiterbringt. Es gibt keine Telefonnummer, die ich anrufen kann, und ich koche dieses Wochenende PHO-Suppe für 50 Leute und habe Angst, mehr zu bestellen und weitere 22 Dollar auszugeben. Total deprimiert."
Polarities: [('KUNDENSERVICE', 'NEGATIVE'), ('VERPACKUNG', 'NEGATIVE')]
Antwort:
KUNDENSERVICE: NEGATIVE – Amazon lehnt Rückerstattung oder Ersatz ab, bietet keine erreichbare Telefonnummer und hält den Kunden in einer nicht endenden Schleife gefangen, was zu Frust und Hilflosigkeit führt.
VERPACKUNG: NEGATIVE – Die erste Packung war offenbar versehentlich mit einem Kartonschneider aufgeschnitten und damit beschädigt angekommen.

Beispiel 6:
Review: "der Tee erzielte bei mir nicht die Wirkung, wie hier in einigen Rezessionen beschrieben, als Tee aber ok. Ein Stoffwechsel, oder abnehmen hat bei mir nicht stattgefunden."
Polarities: [('QUALITÄT', 'NEUTRAL')]
Antwort:
QUALITÄT: NEUTRAL – Der Tee wird als „ok“ beschrieben, aber es wird keine klare positive oder negative Meinung geäußert. Die Wirkung ist nicht wie erwartet, was neutral bewertet wird.

Jetzt du:
Review: "$review"
Polarities: $polarities
Antwort:""")

# ──────────────────────────────────────────────────────────────────────────────
# User-Prompt-Templates Fine-Tuned
# ──────────────────────────────────────────────────────────────────────────────

USER_TEMPLATE_ABSA_FINETUNED = Template("Review:\n$review")