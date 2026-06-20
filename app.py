from pathlib import Path
import tempfile

import pandas as pd
import streamlit as st

from fingerprint import (
    build_database,
    identify,
    load_database,
    plot_constellation,
    plot_offset_histogram,
    plot_spectrogram,
)


APP_DIR = Path(__file__).parent
SONG_DIR = APP_DIR / "songs"
INDEX_DIR = APP_DIR / "index"
DB_PATH = INDEX_DIR / "fingerprint_db.pkl"


st.set_page_config(page_title="Magical Mystery Tune", layout="wide")
st.title("Magical Mystery Tune")


def get_database():
    if DB_PATH.exists():
        return load_database(DB_PATH)
    with st.spinner("Indexing the provided song database..."):
        return build_database(SONG_DIR, DB_PATH)


def save_uploaded_file(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix or ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getbuffer())
    tmp.close()
    return Path(tmp.name)


with st.sidebar:
    st.header("Database")
    song_count = len(list(SONG_DIR.glob("*.*"))) if SONG_DIR.exists() else 0
    st.write(f"Songs found: {song_count}")
    st.write(f"Index exists: {'yes' if DB_PATH.exists() else 'no'}")
    if st.button("Rebuild index", use_container_width=True):
        with st.spinner("Rebuilding fingerprints from songs/ ..."):
            build_database(SONG_DIR, DB_PATH)
        st.success("Index rebuilt.")

    st.header("Matching")
    use_pairs = st.toggle("Use paired-peak hashes", value=True)


single_tab, batch_tab, status_tab = st.tabs(["Single clip", "Batch mode", "Index status"])


with single_tab:
    uploaded = st.file_uploader(
        "Upload a query clip", type=["mp3", "wav", "flac", "ogg"], accept_multiple_files=False
    )
    if uploaded is not None:
        query_path = save_uploaded_file(uploaded)
        st.audio(uploaded)
        db = get_database()
        with st.spinner("Matching query clip..."):
            result = identify(query_path, db, use_pairs=use_pairs)

        metric_cols = st.columns(2)
        metric_cols[0].metric("Recognised song", result.prediction)
        metric_cols[1].metric("Top offset votes", result.score)

        st.subheader("Top offset matches")
        st.dataframe(result.top_matches, use_container_width=True, hide_index=True)

        left, right = st.columns(2)
        with left:
            st.pyplot(plot_spectrogram(result.spectrogram_db), clear_figure=True)
        with right:
            st.pyplot(plot_constellation(result.spectrogram_db, result.peaks), clear_figure=True)
        st.pyplot(plot_offset_histogram(result.histogram, result.prediction), clear_figure=True)


with batch_tab:
    uploads = st.file_uploader(
        "Upload query clips for batch prediction",
        type=["mp3", "wav", "flac", "ogg"],
        accept_multiple_files=True,
    )
    if uploads:
        db = get_database()
        rows = []
        progress = st.progress(0)
        for i, uploaded_file in enumerate(uploads, start=1):
            query_path = save_uploaded_file(uploaded_file)
            result = identify(query_path, db, use_pairs=True)
            rows.append(
                {
                    "filename": uploaded_file.name,
                    "prediction": Path(result.prediction).stem,
                }
            )
            progress.progress(i / len(uploads))

        results = pd.DataFrame(rows, columns=["filename", "prediction"])
        st.dataframe(results, use_container_width=True, hide_index=True)
        csv_bytes = results.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download results.csv",
            csv_bytes,
            file_name="results.csv",
            mime="text/csv",
            use_container_width=True,
        )


with status_tab:
    st.subheader("Expected project layout")
    st.code(
        """q3_streamlit_app/
  app.py
  fingerprint.py
  requirements.txt
  songs/
    A Day In The Life.mp3
    ...
  index/
    fingerprint_db.pkl  # created by the app
""",
        language="text",
    )
    if DB_PATH.exists():
        db = load_database(DB_PATH)
        st.write("Fingerprint parameters")
        st.json(db["params"])
        st.write("Indexed songs")
        st.dataframe(pd.DataFrame(db["songs"]), use_container_width=True, hide_index=True)
    else:
        st.info("Build the index after placing the provided MP3 files inside songs/.")
