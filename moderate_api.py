"""
moderate_api.py

A tiny local API that loads model.pkl (produced by train_model.py) and
exposes POST /moderate for the DECTAI app to call.

Usage:
    python moderate_api.py

Then leave this running in its own terminal while you run `npm run dev`
in the dectai folder.
"""

import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS
from better_profanity import profanity
from text_normalize import normalize_text

app = Flask(__name__)
CORS(app)  # allow requests from the Vite dev server (localhost:5173)

with open("model.pkl", "rb") as f:
    saved = pickle.load(f)

model = saved["model"]
vectorizer = saved["vectorizer"]
labels = saved["labels"]

# --- Rule-based safety net ---
# The trained model handles broad semantic toxicity (insults, threats,
# harassment), but single-character substitution on an explicit word
# ("p*nis", "sl*t") can slip past even a retrained model if there weren't
# many such examples in the training data. better-profanity is a
# maintained library built specifically to catch that pattern (including
# leetspeak and one-letter censoring), so we run it as a second,
# independent check: if EITHER the model OR this wordlist flags a
# comment, it's blocked.
#
# Its default wordlist includes some words that are completely normal in
# benign contexts (health class, HR policy, news) and shouldn't be
# blanket-blocked on their own — we remove those before loading it.
_SAFE_EXCEPTIONS = {"sex", "sexual", "gay"}

import os
_wordlist_path = os.path.join(os.path.dirname(__import__("better_profanity").__file__), "profanity_wordlist.txt")
with open(_wordlist_path, "r", encoding="utf-8") as f:
    _custom_words = [w.strip() for w in f if w.strip() and w.strip() not in _SAFE_EXCEPTIONS]

profanity.load_censor_words(custom_words=_custom_words)

# --- Known-safe phrase allowlist ---
# The underlying training data statistically over-associates certain
# words (like "sex") with obscene content because of how they appear in
# the original dataset, even in clearly benign contexts. Rather than
# fight that with more training data under time pressure, we explicitly
# allowlist phrases that should always pass, checked before anything
# else. Add more lowercase phrases here as you find more false positives.
SAFE_PHRASES = [
    "sex education",
    "safe sex",
    "sex of the baby",
    "sexual health",
    "sex ed",
]

# Deterministic pattern-based override for the identity_hate category
# specifically. Probability thresholds alone weren't reliably fixing
# false positives on plain identity self-statements ("i am gay", "my
# sister is gay"), because the bias in the training data is severe
# enough that the model stays overconfident even at high thresholds. A
# regex match is guaranteed to work regardless of model calibration.
# These patterns only describe/identify — they don't match combined
# hostility like "you gay idiot", which still gets caught normally
# through the toxic/insult categories.
import re as _re

IDENTITY_NEUTRAL_PATTERNS = [
    r"\bi\s*(?:'?m|\s+am)\s+(?:gay|lesbian|bisexual|transgender|muslim|jewish|christian|black|white|disabled)\b",
    r"\b(?:he|she|they)\s+(?:is|are)\s+(?:gay|lesbian|bisexual|transgender|muslim|jewish|christian|black|white|disabled)\b",
    r"\bidentif(?:y|ies)\s+as\s+(?:gay|lesbian|bisexual|transgender)\b",
    r"\bcame\s+out\s+as\s+(?:gay|lesbian|bisexual|transgender)\b",
    r"\b(?:my\s+\w+\s+is|our\s+\w+\s+is)\s+(?:gay|lesbian|bisexual|transgender|muslim|jewish|christian|black|white|disabled)\b",
]
_IDENTITY_NEUTRAL_RE = [_re.compile(p, _re.IGNORECASE) for p in IDENTITY_NEUTRAL_PATTERNS]

def _is_identity_neutral(text: str) -> bool:
    return any(p.search(text) for p in _IDENTITY_NEUTRAL_RE)

# Per-category thresholds instead of one global number. identity_hate is
# set much stricter because the Jigsaw dataset has documented bias on
# this specific category — it over-associates identity words (e.g.
# "gay", "muslim", "black") with hate regardless of actual context,
# more so than any other category. Other categories stay more sensitive
# since they don't share this specific bias to the same degree.
THRESHOLDS = {
    "toxic": 0.82,
    "severe_toxic": 0.8,
    "obscene": 0.8,
    "threat": 0.65,       # threats are rare and serious — keep sensitive
    "insult": 0.8,
    "identity_hate": 0.92,
}
DEFAULT_THRESHOLD = 0.8

@app.route("/moderate", methods=["POST"])
def moderate():
    data = request.get_json(force=True)
    text = (data or {}).get("text", "").strip()

    if not text:
        return jsonify({"isSafe": True})

    lower_text = text.lower()
    if any(phrase in lower_text for phrase in SAFE_PHRASES):
        return jsonify({"isSafe": True})

    # Checked unconditionally, before anything else — the real trained
    # model can flag identity self-statements ("i am gay") under MULTIPLE
    # categories at once (not just identity_hate alone), so a check that
    # only fired when identity_hate was the sole flag wasn't reliable.
    if _is_identity_neutral(text):
        return jsonify({"isSafe": True})

    # Rule-based check first (fast, catches symbol-censored profanity the
    # model might miss).
    if profanity.contains_profanity(text):
        return jsonify({
            "isSafe": False,
            "reason": "This comment was flagged for inappropriate language.",
        })

    normalized = normalize_text(text)
    vec = vectorizer.transform([normalized])
    probs = model.predict_proba(vec)[0]

    flagged = [
        (label, float(prob))
        for label, prob in zip(labels, probs)
        if prob >= THRESHOLDS.get(label, DEFAULT_THRESHOLD)
    ]

    # identity_hate is the category most affected by the dataset's
    # documented bias (identity words like "gay" statistically
    # over-associated with hate regardless of context). Rather than trust
    # it in isolation, only let it block a comment if EITHER another
    # category also flagged the same comment (real hostility signal
    # alongside the identity word), OR its own confidence is extreme
    # (>=0.97), which still catches identity-only hate speech that
    # doesn't happen to trip any other category.
    other_flags = [f for f in flagged if f[0] != "identity_hate"]
    identity_flag = next((f for f in flagged if f[0] == "identity_hate"), None)

    if identity_flag and not other_flags and identity_flag[1] < 0.97:
        flagged = other_flags

    if flagged:
        # report the category the model is most confident about
        top_label, top_prob = max(flagged, key=lambda x: x[1])
        return jsonify({
            "isSafe": False,
            "reason": f"This comment was flagged for {top_label.replace('_', ' ')}.",
        })

    return jsonify({"isSafe": True})


if __name__ == "__main__":
    print("DECTAI moderation API running on http://localhost:5000")
    app.run(port=5000, debug=False)
