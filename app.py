from pathlib import Path
import tempfile

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from fingerprint import (
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


@st.cache_resource(show_spinner=False)
def get_database():
    if not DB_PATH.exists():
        raise FileNotFoundError(
            "Missing index/fingerprint_db.pkl. Build the index locally once, commit/upload it, "
            "and redeploy the app."
        )
    return load_database(DB_PATH)


def save_uploaded_file(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix or ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getbuffer())
    tmp.close()
    return Path(tmp.name)


def count_songs() -> int:
    if not SONG_DIR.exists():
        return 0
    return sum(
        len(list(SONG_DIR.glob(pattern)))
        for pattern in ("*.mp3", "*.wav", "*.flac", "*.ogg")
    )


with st.sidebar:
    st.header("Database")
    st.write(f"Songs found: {count_songs()}")
    st.write(f"Index exists: {'yes' if DB_PATH.exists() else 'no'}")
    st.caption("The deployed app uses the prebuilt index. Rebuild it locally before deployment.")

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
        try:
            db = get_database()
            with st.spinner("Matching query clip..."):
                result = identify(query_path, db, use_pairs=use_pairs)

            metric_cols = st.columns(2)
            metric_cols[0].metric("Recognised song", result.prediction)
            metric_cols[1].metric("Top offset votes", result.score)

            st.subheader("Top offset matches")
            st.dataframe(result.top_matches, width="stretch", hide_index=True)

            left, right = st.columns(2)
            with left:
                fig = plot_spectrogram(result.spectrogram_db)
                st.pyplot(fig, clear_figure=True)
                plt.close(fig)
            with right:
                fig = plot_constellation(result.spectrogram_db, result.peaks)
                st.pyplot(fig, clear_figure=True)
                plt.close(fig)
            fig = plot_offset_histogram(result.histogram, result.prediction)
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)
        except Exception as exc:
            st.error(f"Could not identify this clip: {exc}")
        finally:
            query_path.unlink(missing_ok=True)


with batch_tab:
    uploads = st.file_uploader(
        "Upload query clips for batch prediction",
        type=["mp3", "wav", "flac", "ogg"],
        accept_multiple_files=True,
    )
    if uploads:
        try:
            db = get_database()
        except Exception as exc:
            st.error(f"Could not load fingerprint database: {exc}")
            st.stop()

        rows = []
        progress = st.progress(0)
        for i, uploaded_file in enumerate(uploads, start=1):
            query_path = save_uploaded_file(uploaded_file)
            try:
                result = identify(query_path, db, use_pairs=True)
                prediction = Path(result.prediction).stem
            except Exception as exc:
                prediction = "unknown"
                st.warning(f"{uploaded_file.name}: {exc}")
            finally:
                query_path.unlink(missing_ok=True)

            rows.append({"filename": uploaded_file.name, "prediction": prediction})
            progress.progress(i / len(uploads))

        results = pd.DataFrame(rows, columns=["filename", "prediction"])
        st.dataframe(results, width="stretch", hide_index=True)
        csv_bytes = results.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download results.csv",
            csv_bytes,
            file_name="results.csv",
            mime="text/csv",
            width="stretch",
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
    fingerprint_db.pkl  # prebuilt before deployment
""",
        language="text",
    )
    if DB_PATH.exists():
        try:
            db = get_database()
            st.write("Fingerprint parameters")
            st.json(db["params"])
            st.write("Indexed songs")
            st.dataframe(pd.DataFrame(db["songs"]), width="stretch", hide_index=True)
        except Exception as exc:
            st.error(f"Could not load fingerprint database: {exc}")
    else:
        st.error("Missing index/fingerprint_db.pkl. Build the index locally and redeploy.")
