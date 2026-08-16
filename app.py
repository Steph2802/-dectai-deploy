"""
app.py

Deployment version of DECTAI for hosting on a real server (e.g. Render).
Serves the built React app AND the moderation API from ONE Python
process. Unlike run_dectai.py (which is for your own machine), this
version reads the port from the environment and doesn't try to open a
browser, since there's no browser on a server.

Requires, in the same folder:
  - model.pkl (the trained model)
  - frontend_dist/ (the built React app — `npm run build` in dectai,
    then copy dist/'s contents here)
  - text_normalize.py
"""

import os
import pickle

from flask import Flask, request, jsonify, send_from_directory
from better_profanity import profanity
from text_normalize import normalize_text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend_dist")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
PORT = int(os.environ.get("PORT", 5000))

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")

# --- Load the trained model ---
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "model.pkl not found. Run 'python train_model.py \"path\\to\\train.csv\"' "
        "first, then try again."
    )

with open(MODEL_PATH, "rb") as f:
    saved = pickle.load(f)

model = saved["model"]
vectorizer = saved["vectorizer"]
labels = saved["labels"]

# --- Rule-based safety net (same as moderate_api.py) ---
_SAFE_EXCEPTIONS = {"sex", "sexual", "gay"}
_wordlist_path = os.path.join(os.path.dirname(__import__("better_profanity").__file__), "profanity_wordlist.txt")
with open(_wordlist_path, "r", encoding="utf-8") as f:
    _custom_words = [w.strip() for w in f if w.strip() and w.strip() not in _SAFE_EXCEPTIONS]
profanity.load_censor_words(custom_words=_custom_words)

SAFE_PHRASES = [
    "sex education",
    "safe sex",
    "sex of the baby",
    "sexual health",
    "sex ed",
]

# Deterministic pattern-based override for the identity_hate category
# specifically — see moderate_api.py for full explanation.
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

THRESHOLDS = {
    "toxic": 0.82,
    "severe_toxic": 0.8,
    "obscene": 0.8,
    "threat": 0.65,
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

    if _is_identity_neutral(text):
        return jsonify({"isSafe": True})

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

    other_flags = [f for f in flagged if f[0] != "identity_hate"]
    identity_flag = next((f for f in flagged if f[0] == "identity_hate"), None)

    if identity_flag and not other_flags and identity_flag[1] < 0.97:
        flagged = other_flags

    if flagged:
        top_label, top_prob = max(flagged, key=lambda x: x[1])
        return jsonify({
            "isSafe": False,
            "reason": f"This comment was flagged for {top_label.replace('_', ' ')}.",
        })

    return jsonify({"isSafe": True})


# --- Serve the built React app for every other route ---
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    full_path = os.path.join(FRONTEND_DIR, path)
    if path and os.path.exists(full_path):
        return send_from_directory(FRONTEND_DIR, path)
    return send_from_directory(FRONTEND_DIR, "index.html")


if __name__ == "__main__":
    if not os.path.isdir(FRONTEND_DIR):
        raise FileNotFoundError(
            f"'{FRONTEND_DIR}' not found. Build the app first (npm run build "
            "in the dectai folder) and copy its dist/ contents here as "
            "'frontend_dist'."
        )

    app.run(host="0.0.0.0", port=PORT, debug=False)
