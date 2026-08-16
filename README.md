# DECTAI — training your own model on the Jigsaw dataset

This folder trains a real classifier on your Jigsaw `train.csv` and serves
it to the DECTAI app locally, replacing the wordlist stub.

## 1. Install Python packages

Open a terminal in this folder and run:

```
pip install -r requirements.txt
```

## 2. Train the model

```
python train_model.py "C:\Users\STEPH\Downloads\train.csv"
```

(Replace the path with wherever your CSV actually is.)

This will take well under a minute even on the full ~160k-row dataset,
print an accuracy/precision/recall report per category, and save a
`model.pkl` file in this folder.

## 3. Start the local moderation API

```
python moderate_api.py
```

Leave this terminal window open — it needs to keep running. You should see:

```
DECTAI moderation API running on http://localhost:5000
```

## 4. Point the app at it

Copy `ModerationService.js` from this folder into your DECTAI app,
replacing the one at `dectai/src/ModerationService.js`.

## 5. Run the app as before

In a **separate** terminal, in the `dectai` folder:

```
npm run dev
```

Open the printed URL. Comments now get scored by your trained model
instead of the wordlist stub — try a few borderline examples (mild
swearing vs. actual insults) to see the difference from a simple
keyword filter.

## Notes

- **Two terminals running at once**: one for `moderate_api.py`, one for
  `npm run dev`. Both need to stay open while you use the app.
- **Threshold**: in `moderate_api.py`, `THRESHOLD = 0.5` controls how
  confident the model needs to be before flagging a comment. Raise it
  if too much gets flagged, lower it if too much gets through.
- **If the app shows every comment as flagged with "Could not reach the
  moderation service"**: the API server isn't running, or it's on a
  different port — check the `moderate_api.py` terminal for errors.
