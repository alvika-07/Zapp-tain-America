from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import pickle
import tempfile
from typing import BinaryIO

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter


SR = 22050
N_FFT = 2048
HOP_LENGTH = 512
PEAK_NEIGHBORHOOD = (21, 21)
PEAK_PERCENTILE = 99.4
MAX_PEAKS = 2500
FANOUT = 12
MIN_DT = 1
MAX_DT = 80


@dataclass(frozen=True)
class FingerprintResult:
    prediction: str
    score: int
    top_matches: pd.DataFrame
    spectrogram_db: np.ndarray
    peaks: np.ndarray
    histogram: Counter


def label_from_path(path: Path) -> str:
    return path.stem


def load_audio(path: Path | str, sr: int = SR) -> tuple[np.ndarray, int]:
    y, loaded_sr = librosa.load(str(path), sr=sr, mono=True)
    if y.size == 0:
        raise ValueError(f"No audio samples found in {path}")
    y = librosa.util.normalize(y)
    return y.astype(np.float32), loaded_sr


def save_upload(upload: BinaryIO, suffix: str) -> Path:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(upload.read())
    tmp.close()
    return Path(tmp.name)


def compute_spectrogram(y: np.ndarray) -> np.ndarray:
    stft = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH, window="hann")
    mag = np.abs(stft)
    return librosa.amplitude_to_db(mag, ref=np.max)


def find_constellation_peaks(s_db: np.ndarray) -> np.ndarray:
    local_max = s_db == maximum_filter(s_db, size=PEAK_NEIGHBORHOOD)
    threshold = np.percentile(s_db, PEAK_PERCENTILE)
    mask = local_max & (s_db >= threshold)
    freq_bins, time_bins = np.where(mask)
    amps = s_db[freq_bins, time_bins]

    peaks = np.column_stack([time_bins, freq_bins, amps])
    if len(peaks) > MAX_PEAKS:
        keep = np.argsort(peaks[:, 2])[-MAX_PEAKS:]
        peaks = peaks[keep]
    order = np.lexsort((peaks[:, 1], peaks[:, 0]))
    return peaks[order].astype(np.int32)


def make_pair_hashes(peaks: np.ndarray) -> list[tuple[tuple[int, int, int], int]]:
    hashes: list[tuple[tuple[int, int, int], int]] = []
    for i, anchor in enumerate(peaks):
        t1, f1 = int(anchor[0]), int(anchor[1])
        for target in peaks[i + 1 : i + 1 + FANOUT]:
            t2, f2 = int(target[0]), int(target[1])
            dt = t2 - t1
            if dt < MIN_DT:
                continue
            if dt > MAX_DT:
                break
            hashes.append(((f1, f2, dt), t1))
    return hashes


def make_single_peak_hashes(peaks: np.ndarray) -> list[tuple[int, int]]:
    return [(int(freq_bin), int(time_bin)) for time_bin, freq_bin, _ in peaks]


def fingerprint_audio(path: Path | str) -> tuple[np.ndarray, np.ndarray, list, list]:
    y, _ = load_audio(path)
    s_db = compute_spectrogram(y)
    peaks = find_constellation_peaks(s_db)
    pair_hashes = make_pair_hashes(peaks)
    single_hashes = make_single_peak_hashes(peaks)
    return s_db, peaks, pair_hashes, single_hashes


def build_database(song_dir: Path | str, db_path: Path | str) -> dict:
    song_dir = Path(song_dir)
    db_path = Path(db_path)
    pair_db: dict[tuple[int, int, int], list[tuple[str, int]]] = defaultdict(list)
    single_db: dict[int, list[tuple[str, int]]] = defaultdict(list)
    songs = sorted(
        list(song_dir.glob("*.mp3"))
        + list(song_dir.glob("*.wav"))
        + list(song_dir.glob("*.flac"))
        + list(song_dir.glob("*.ogg"))
    )
    if not songs:
        raise FileNotFoundError(f"No audio files found in {song_dir}")

    metadata = []
    for song_path in songs:
        label = label_from_path(song_path)
        _, peaks, pair_hashes, single_hashes = fingerprint_audio(song_path)
        for h, anchor_time in pair_hashes:
            pair_db[h].append((label, anchor_time))
        for h, anchor_time in single_hashes:
            single_db[h].append((label, anchor_time))
        metadata.append(
            {
                "filename": song_path.name,
                "label": label,
                "peaks": int(len(peaks)),
                "paired_hashes": int(len(pair_hashes)),
            }
        )

    db = {
        "params": {
            "sr": SR,
            "n_fft": N_FFT,
            "hop_length": HOP_LENGTH,
            "peak_percentile": PEAK_PERCENTILE,
            "fanout": FANOUT,
            "min_dt": MIN_DT,
            "max_dt": MAX_DT,
        },
        "pair_db": dict(pair_db),
        "single_db": dict(single_db),
        "songs": metadata,
    }
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with db_path.open("wb") as f:
        pickle.dump(db, f)
    return db


def load_database(db_path: Path | str) -> dict:
    with Path(db_path).open("rb") as f:
        return pickle.load(f)


def match_hashes(query_hashes: list, db_hashes: dict) -> tuple[str, int, Counter, pd.DataFrame]:
    histogram: Counter = Counter()
    for h, query_time in query_hashes:
        for song_label, song_time in db_hashes.get(h, []):
            histogram[(song_label, song_time - query_time)] += 1

    if not histogram:
        empty = pd.DataFrame(columns=["song", "offset_frames", "votes"])
        return "unknown", 0, histogram, empty

    (best_song, _), best_score = histogram.most_common(1)[0]
    rows = [
        {"song": song, "offset_frames": offset, "votes": votes}
        for (song, offset), votes in histogram.most_common(10)
    ]
    return best_song, int(best_score), histogram, pd.DataFrame(rows)


def identify(path: Path | str, db: dict, use_pairs: bool = True) -> FingerprintResult:
    s_db, peaks, pair_hashes, single_hashes = fingerprint_audio(path)
    if use_pairs:
        prediction, score, histogram, top_matches = match_hashes(pair_hashes, db["pair_db"])
    else:
        prediction, score, histogram, top_matches = match_hashes(single_hashes, db["single_db"])
    return FingerprintResult(prediction, score, top_matches, s_db, peaks, histogram)


def plot_spectrogram(s_db: np.ndarray):
    fig, ax = plt.subplots(figsize=(9, 4))
    img = librosa.display.specshow(
        s_db, sr=SR, hop_length=HOP_LENGTH, x_axis="time", y_axis="hz", ax=ax, cmap="magma"
    )
    ax.set_title("Spectrogram")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    fig.tight_layout()
    return fig


def plot_constellation(s_db: np.ndarray, peaks: np.ndarray):
    fig, ax = plt.subplots(figsize=(9, 4))
    librosa.display.specshow(
        s_db, sr=SR, hop_length=HOP_LENGTH, x_axis="time", y_axis="hz", ax=ax, cmap="gray_r"
    )
    times = librosa.frames_to_time(peaks[:, 0], sr=SR, hop_length=HOP_LENGTH)
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)[peaks[:, 1]]
    ax.scatter(times, freqs, s=10, facecolors="none", edgecolors="#00b894", linewidths=0.8)
    ax.set_title("Constellation of Peaks")
    fig.tight_layout()
    return fig


def plot_offset_histogram(histogram: Counter, prediction: str):
    matched_offsets = [
        (offset, count)
        for (song, offset), count in histogram.items()
        if song == prediction
    ]
    matched_offsets = sorted(matched_offsets, key=lambda item: item[1], reverse=True)[:40]
    matched_offsets = sorted(matched_offsets, key=lambda item: item[0])
    offsets = [offset for offset, _ in matched_offsets]
    votes = [count for _, count in matched_offsets]

    fig, ax = plt.subplots(figsize=(9, 3.2))
    if offsets:
        spread = max(offsets) - min(offsets) if len(offsets) > 1 else 1
        bar_width = max(1.0, spread / 120)
        ax.bar(offsets, votes, width=bar_width, color="#2d7dd2")
        best_offset = offsets[int(np.argmax(votes))]
        ax.axvline(best_offset, color="#d62828", linewidth=1.2, linestyle="--")
        ax.annotate(
            f"best offset = {best_offset}",
            xy=(best_offset, max(votes)),
            xytext=(8, -18),
            textcoords="offset points",
            fontsize=8,
            color="#d62828",
        )
    ax.set_title(f"Offset Histogram for {prediction}")
    ax.set_xlabel("Offset in spectrogram frames")
    ax.set_ylabel("Matching hash votes")
    fig.tight_layout()
    return fig
