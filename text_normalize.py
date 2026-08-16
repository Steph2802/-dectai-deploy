"""
text_normalize.py

Shared text preprocessing used by BOTH train_model.py and moderate_api.py.
It's critical this stays identical in both places — if training normalizes
text one way and the live API normalizes it differently, the model sees
different-looking input at prediction time than it learned from, and
accuracy silently degrades.

This targets a specific evasion pattern: people beating word-based filters
by swapping a letter for a symbol or number, e.g. "p*nis", "sl*t", "h3ll".
"""

import re

# Common leetspeak / symbol substitutions used to sneak words past
# simple filters. Applied before the text ever reaches the vectorizer.
_SUBSTITUTIONS = {
    "0": "o",
    "1": "i",
    "3": "e",
    "4": "a",
    "5": "s",
    "7": "t",
    "@": "a",
    "$": "s",
}


def normalize_text(text: str) -> str:
    """Lowercase, undo common letter/symbol substitutions, and collapse
    stray punctuation people insert mid-word to dodge filters."""
    text = text.lower()

    for symbol, letter in _SUBSTITUTIONS.items():
        text = text.replace(symbol, letter)

    # Symbols like * ! # are common word-breakers ("p*nis", "sh!t").
    # Strip them ONLY when they sit between two letters (mid-word), so we
    # don't destroy normal punctuation like "wait... really?" or emoticons.
    text = re.sub(r"(?<=[a-z])[*!#_]+(?=[a-z])", "", text)

    # Collapse 3+ repeated letters some people use to dodge exact-match
    # filters too ("shiiiit" -> "shiit"), without wrecking normal words.
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    return text
