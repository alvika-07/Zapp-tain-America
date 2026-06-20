# Q3B Streamlit App

This app implements the Q3 audio fingerprinting system:

- indexes the provided songs once from `songs/`
- identifies one uploaded query clip
- displays the spectrogram, constellation of peaks, and offset histogram
- supports batch upload
- exports `results.csv` with exactly `filename,prediction`

## Setup

1. Copy the provided course MP3 files into `songs/` without renaming them.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run locally:

```bash
streamlit run app.py
```

4. In the sidebar, click `Rebuild index` once. The app writes `index/fingerprint_db.pkl`.

## Deployment

For Streamlit Community Cloud, push this folder to GitHub with the `songs/` folder included, or push the prebuilt `index/fingerprint_db.pkl` plus the song files if your instructor wants the full app to work immediately. The appendix says the indexed song database must ship with the deployed app and the live link must work.

The batch output uses the exact required columns:

```csv
filename,prediction
query1.mp3,Let It Be
```
