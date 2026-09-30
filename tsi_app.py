
# ===============================================================================
# Application Overview (Annotated)
# -------------------------------------------------------------------------------
# This file implements a Tkinter desktop application for biosignal analysis.
# Key capabilities:
# - Setup/Load data and configure channels (ACC vs EMG, left/right)
# - Visualization of time-domain signals with selection tools
# - FFT analysis of processed data (single-sided magnitude spectrum)
# - Additional analysis tabs (placeholders may exist for future modules)
#
# About these comments:
# - Added for readability and maintainability.
# - Placed as BLOCK comments above classes/functions/sections.
# - The code itself is UNCHANGED (no logic or behavior modified).
# ===============================================================================
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TSI App — Visualization + Selection + Frequency-domain

What’s included
---------------
• Setup tab:
  - Load .txt/.csv/.tsv/.dat (auto delimiter; optional decimal comma).
  - Set sampling frequency (Fs).
  - Map up to 8 channels: Type (ACC/EMG) and Side (Left/Right).
  - Bottom log shows file info (path, delimiter, rows × cols, preview),
    AND a full “Current channel config (pre-save)” snapshot of the 8 rows.

• Visualization tab:
  - Plot Raw or Processed signals (Processed = ACC band-pass, EMG band-pass + rectified).
  - Split by Left/Right in 2 columns of subplots.
  - If a selection exists (from the Selection tab), the selected time span is shaded.

• Selection tab:
  - Auto-plots a guidance channel (prefers Left/ACC; else first enabled) with a Combobox to switch.
  - Select time [t0, t1] via mouse (SpanSelector) or by typing Start/End and Apply.
  - Stores selection in global AppState: selection_t0, selection_t1.

• Frequency domain (FFT):
  - FFT for up to 6 channels in the same order as Visualization.
  - Uses selected segment if present; else full record.
  - Plots default x-range 2–20 Hz but is fully interactive.
  - X-axis synchronized across ALL subplots; Y-axis synchronized BY TYPE (ACC together, EMG together).
  - Robust to short selections / Fs/2 < 20 Hz (no more blank panel).

• Coherence/Cumulant:
  - Magnitude-squared coherence between two selectable channels (default Left ACC vs Right ACC).
  - Window (s) configurable (default 1.0 s). Welch with nperseg=pow2(round(win*Fs)), noverlap=nperseg/2.
  - 95% significance line (Halliday threshold) and 95% confidence interval ribbon (jackknife on coherency).
  - Time-domain cumulant density (Halliday et al. 1995b) between the same
    channel pair and window, plotted side by side with coherence: shows the
    lag (ms) at which the two signals correlate best, with a constant
    +/- 95% confidence band under the assumption of independence.
    Max lag (ms) is configurable (default 100 ms).

• Spectrogram:
  - Spectrograms for up to 6 channels (same Visualization order) in a 3×2 grid.
  - ~1 s window; 50% overlap; dB scaling; y-axis forced to 2–20 Hz.
  - Uses selected segment if present; else full.

Requirements
------------
• Python 3.9+, NumPy, Matplotlib; SciPy recommended (filters/Welch/spectrogram).
"""


# Imports
# -------------------------------------------------------------------------------
# External libraries: numpy, matplotlib (TkAgg backend), and tkinter/ttk provide
# numerics, plotting, and GUI. Some imports may be wrapped in try/except to allow
# the app to degrade gracefully if optional packages are missing.
from __future__ import annotations

import os
import sys
import io
import re
import csv
import math
import shutil
import subprocess
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

# -----------------------------
# Small helper: file log + exit
# -----------------------------
def _script_dir() -> str:
    # PyInstaller (--onefile) unpacks to a throwaway temp folder at runtime,
    # so __file__ there points somewhere that's gone as soon as the app
    # closes — wrong for a log file or cache meant to persist. sys.frozen is
    # set by PyInstaller/similar tools; sys.executable is the actual .exe's
    # (stable) location in that case.
    if getattr(sys, "frozen", False):
        try:
            return os.path.dirname(os.path.abspath(sys.executable))
        except Exception:
            pass
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except Exception:
        return os.getcwd()

LOG_PATH = os.path.join(_script_dir(), "tsi_app.log")

def _log_file(msg: str) -> None:
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{ts}] {msg}\n")
    except Exception:
        pass

def _fatal(msg: str, exc: Exception | None = None, title: str = "TSI App — Fatal"):
    full = msg if exc is None else f"{msg}\n\n{exc}"
    _log_file("FATAL: " + full)
    try:
        if sys.platform.startswith("win"):
            import ctypes
            ctypes.windll.user32.MessageBoxW(0, full, title, 0x10)
        else:
            print(full, file=sys.stderr)
    except Exception:
        pass
    sys.exit(1)

if sys.version_info < (3, 9):
    _fatal("Python 3.9+ is required.")

# -------------
# GUI and libs
# -------------
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox, simpledialog
    from tkinter.scrolledtext import ScrolledText
except Exception as e:
    _fatal("tkinter is not available.", e)

# -----------------------------------------------------------------------
# Auto-install missing core dependencies into THIS interpreter.
# -----------------------------------------------------------------------
# Whichever Python launches this script (the system one, a venv, or the
# Python 3.9 some users were told to use just for sonpy) may be missing
# numpy/scipy/matplotlib. Unlike sonpy, these have wheels for essentially
# every supported Python version/OS, so a plain "pip install" into the
# running interpreter is enough to self-heal — no version juggling needed.
#
# Frozen exception: inside a PyInstaller/py2app bundle there is no pip, no
# writable site-packages, and often no internet guarantee — everything must
# already be baked in at build time. sys.frozen is set by PyInstaller (and
# similar tools) at runtime, so this is a no-op there instead of a confusing
# failure.
def _ensure_pip_package(pip_name: str, timeout: int = 300) -> bool:
    if getattr(sys, "frozen", False):
        _log_file(f"[DEPS] Running as a frozen/standalone build — skipping "
                   f"'pip install {pip_name}' (it must be bundled at build time).")
        return False
    try:
        r = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet",
             "--disable-pip-version-check", pip_name],
            capture_output=True, text=True, timeout=timeout,
        )
        if r.returncode != 0:
            _log_file(f"[DEPS] pip install {pip_name} failed:\n{r.stdout}\n{r.stderr}")
        return r.returncode == 0
    except Exception as e:
        _log_file(f"[DEPS] pip install {pip_name} raised: {e}")
        return False


try:
    import numpy as np
    HAS_NUMPY = True
except Exception as e:
    HAS_NUMPY = False
    _log_file(f"NumPy import failed: {e}; attempting automatic install...")
    if _ensure_pip_package("numpy"):
        try:
            import importlib
            importlib.invalidate_caches()
            import numpy as np
            HAS_NUMPY = True
            _log_file("[DEPS] NumPy installed and imported successfully.")
        except Exception as e2:
            _log_file(f"[DEPS] NumPy still failed to import after install: {e2}")
    if not HAS_NUMPY:
        _log_file("[DEPS] NumPy is unavailable in this Python — data loading/processing will fail.")

try:
    import scipy.signal as sps
    from scipy.signal import butter, filtfilt, welch
    from scipy.stats import t as t_dist
    HAS_SCIPY = True
except Exception as e:
    HAS_SCIPY = False
    butter = filtfilt = welch = None  # type: ignore[assignment]
    _log_file(f"SciPy import failed: {e}; attempting automatic install...")
    if _ensure_pip_package("scipy"):
        try:
            import importlib
            importlib.invalidate_caches()
            import scipy.signal as sps
            from scipy.signal import butter, filtfilt, welch
            from scipy.stats import t as t_dist
            HAS_SCIPY = True
            _log_file("[DEPS] SciPy installed and imported successfully.")
        except Exception as e2:
            _log_file(f"[DEPS] SciPy still failed to import after install: {e2}")
    if not HAS_SCIPY:
        _log_file("[DEPS] SciPy unavailable — falling back to FIR filters where possible.")

MATPLOTLIB_OK = True
try:
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    import matplotlib.pyplot as plt
    from matplotlib.widgets import SpanSelector
except Exception as e:
    MATPLOTLIB_OK = False
    _log_file(f"Matplotlib import failed: {e}; attempting automatic install...")
    if _ensure_pip_package("matplotlib"):
        try:
            import importlib
            importlib.invalidate_caches()
            import matplotlib
            matplotlib.use("TkAgg")
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
            import matplotlib.pyplot as plt
            from matplotlib.widgets import SpanSelector
            MATPLOTLIB_OK = True
            _log_file("[DEPS] Matplotlib installed and imported successfully.")
        except Exception as e2:
            _log_file(f"[DEPS] Matplotlib still failed to import after install: {e2}")
    if not MATPLOTLIB_OK:
        _log_file("[DEPS] Matplotlib unavailable — plotting tabs will be disabled.")

# Word (.docx) export — used by the Comparison tab's session report.
try:
    import docx
    HAS_DOCX = True
except Exception as e:
    HAS_DOCX = False
    _log_file(f"python-docx import failed: {e}; attempting automatic install...")
    if _ensure_pip_package("python-docx"):
        try:
            import importlib
            importlib.invalidate_caches()
            import docx
            HAS_DOCX = True
            _log_file("[DEPS] python-docx installed and imported successfully.")
        except Exception as e2:
            _log_file(f"[DEPS] python-docx still failed to import after install: {e2}")
    if not HAS_DOCX:
        _log_file("[DEPS] python-docx unavailable — Word export will be disabled.")

# SON64 / SMRX support — sonpy requires Python 3.9.x (not 3.10+)
try:
    import sonpy
    HAS_SONPY = True
except ImportError:
    HAS_SONPY = False
except Exception as e:
    HAS_SONPY = False
    _log_file(f"sonpy import failed unexpectedly: {e}")

# -------------------------
# Data model / App state
# -------------------------

# Configuration/Data Containers
# -------------------------------------------------------------------------------
# These @dataclass structures hold configuration and state for the application,
# such as sample rate (Fs), channel mapping (left/right, kind), and selection
# intervals. They make it easy to pass structured state between tabs.
@dataclass
class ChannelConfig:
    enabled: bool = True
    col_1based: int = 1          # human-facing 1-based column index
    kind: str = "ACC"            # "ACC" or "EMG"
    side: str = "Left"           # "Left" or "Right"
    name: str = ""               # free-text channel name (defaults to the
                                  # column header as it appears in the file;
                                  # user-editable in Setup). Used everywhere
                                  # a channel needs to be identified — see
                                  # _channel_display_name().


def _channel_display_name(cfg: "ChannelConfig", i: int) -> str:
    """
    The name used to identify a channel everywhere in the UI: the user's
    custom name if they've set one, else a sensible fallback so nothing is
    ever blank.
    """
    name = (cfg.name or "").strip()
    if name:
        return name
    return f"{cfg.kind}-{cfg.side}" if (cfg.kind or cfg.side) else f"Ch{i+1}"

@dataclass
class AppState:
    raw_data: Optional[Any] = None
    fs: Optional[float] = None
    channel_config: Dict[int, ChannelConfig] = field(default_factory=dict)
    options: Dict[str, Any] = field(default_factory=lambda: {
        "decimal_comma": False,
        "skip_rows": 0,
        "source_path": None,
        "detected_delimiter": None,
    })

    # Processing defaults
    acc_bp_low: float = 0.5
    acc_bp_high: float = 20.0
    emg_hp: float = 20.0
    emg_lp: float = 450.0

    # Global selection (seconds) — set by Selection tab
    selection_t0: Optional[float] = None
    selection_t1: Optional[float] = None

    def summary(self) -> str:
        lines = [
            f"File: {self.options.get('source_path')}",
            f"Decimal comma: {self.options.get('decimal_comma')}",
            f"Skip rows: {self.options.get('skip_rows')}",
            f"Sampling Frequency (Hz): {self.fs}",
        ]
        if self.raw_data is None:
            lines.append("Raw data: None")
        else:
            try:
                n = len(self.raw_data)
                m = len(self.raw_data[0]) if n > 0 else 0
            except Exception:
                shp = getattr(self.raw_data, "shape", (0, 0))
                n, m = shp[0], (shp[1] if len(shp) > 1 else 0)
            lines.append(f"Raw shape: {n} x {m}")
        lines.append("Enabled channels:")
        for i in range(8):
            cfg = self.channel_config.get(i, ChannelConfig(enabled=False, col_1based=i+1))
            if cfg.enabled:
                lines.append(f"  Ch{i+1} \"{_channel_display_name(cfg, i)}\": "
                             f"col={cfg.col_1based} {cfg.kind}-{cfg.side}")
        lines.append(f"Filters: ACC {self.acc_bp_low}-{self.acc_bp_high} Hz | EMG {self.emg_hp}-{self.emg_lp} Hz (rectified)")
        if self.selection_t0 is not None and self.selection_t1 is not None:
            t0, t1 = sorted((self.selection_t0, self.selection_t1))
            lines.append(f"Selection: {t0:.6f} – {t1:.6f} s")
        else:
            lines.append("Selection: (none)")
        return "\n".join(lines)

# --------------------
# Parser utilities
# --------------------
def _normalize_decimal_commas(text: str) -> str:
    """Turn decimal commas into dots (e.g., 12,34 → 12.34)."""
    return re.sub(r'(?P<a>\d),(?P<b>\d)', r'\g<a>.\g<b>', text)

def _detect_delimiter(sample_text: str) -> str:
    """Try to guess delimiter among comma/semicolon/tab/space."""
    candidates = [',', ';', '\t', ' ']
    try:
        dialect = csv.Sniffer().sniff(sample_text, delimiters="".join(candidates))
        if dialect.delimiter in candidates:
            return dialect.delimiter
    except Exception:
        pass
    # fallback by counting appearances
    lines = sample_text.splitlines()[:10]
    counts = {c: 0 for c in candidates}
    for ln in lines:
        for c in candidates:
            counts[c] += ln.count(c)
    best = max(counts, key=counts.get)
    # avoid ' ' if we also see structured delimiters
    if best == ' ' and (counts[','] > 0 or counts[';'] > 0 or counts['\t'] > 0):
        best = max([',', ';', '\t'], key=lambda d: counts[d])
    return best

def parse_txt_anycols(filepath: str, skip_rows: int = 0, decimal_comma: bool = False):
    """
    Parse a numeric table with an auto-detected delimiter.
    Returns (data, delimiter, skipped, col_titles) where:
    - skipped     = total header rows skipped
    - col_titles  = list of column name strings if a text header was found, else []
    """
    if not os.path.isfile(filepath):
        raise ValueError("File not found")
    with io.open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        raw = f.read()
    raw = raw.replace('\r\n', '\n').replace('\r', '\n')
    lines = raw.splitlines()
    if skip_rows > 0:
        lines = lines[skip_rows:]
    if not lines:
        raise ValueError("No lines remain after applying 'Skip rows'.")

    if decimal_comma:
        lines = [_normalize_decimal_commas(ln) for ln in lines]
    sample = "\n".join(lines[:50])
    delim = _detect_delimiter(sample)

    parsed: List[List[float]] = []
    auto_skipped = 0
    found_data   = False
    col_titles: List[str] = []   # captured from the last text-only header row

    reader = csv.reader(lines, delimiter=delim)
    for i, row in enumerate(reader, 1):
        if not row:
            if not found_data:
                auto_skipped += 1
            continue
        if delim == ' ':
            row = re.split(r'\s+', " ".join(row).strip())
        row = [c.strip() for c in row if c.strip() != ""]
        if not any(re.search(r'[-+]?\d', c) for c in row):
            if not found_data:
                auto_skipped += 1
                # Save as potential column titles (strip quotes)
                col_titles = [c.strip('"\'') for c in row]
            continue
        try:
            nums = [float(c) for c in row]
        except Exception:
            if not found_data:
                auto_skipped += 1
                col_titles = [c.strip('"\'') for c in row]
                continue
            continue
        found_data = True
        parsed.append(nums)

    if not parsed:
        raise ValueError("No numeric rows found.")

    maxc = max(len(r) for r in parsed)
    for r in parsed:
        if len(r) < maxc:
            r += [float('nan')] * (maxc - len(r))

    data = np.asarray(parsed, dtype=float) if HAS_NUMPY else parsed
    return data, delim, skip_rows + auto_skipped, col_titles

# --------------------
# Basic DSP helpers
# --------------------
def _validate_band(fs: float, lo: float | None, hi: float | None) -> Tuple[float, float]:
    nyq = fs / 2.0
    if lo is None: lo = 0.0
    if hi is None: hi = nyq * 0.99
    lo = max(0.0, float(lo))
    hi = max(lo + 1e-6, float(hi))
    if hi >= nyq: hi = nyq * 0.99
    return lo, hi

def _butter_bandpass_sos(fs: float, lo: float, hi: float, order: int = 4):
    nyq = fs * 0.5
    return sps.butter(order, [lo/nyq, hi/nyq], btype='band', output='sos')

def _apply_scipy_sos(x, fs, lo, hi):
    sos = _butter_bandpass_sos(fs, lo, hi)
    return sps.sosfiltfilt(sos, x, axis=0)

def _fir_band(window_len: int, fs: float, lo: float, hi: float):
    if window_len % 2 == 0: window_len += 1
    n = np.arange(window_len) - (window_len - 1)/2
    def lp(fc):
        fc_n = fc/fs
        h = np.sinc(2*fc_n*n)
        win = 0.5 - 0.5*np.cos(2*np.pi*(np.arange(window_len)/(window_len-1)))
        return h*win
    h_hi = lp(hi); h_lo = lp(lo)
    h = h_hi - h_lo
    # Use sum of absolute values: a band-pass FIR sums to ~0 by design,
    # so normalizing by np.sum(h) would divide by near-zero.
    h_abs_sum = np.sum(np.abs(h))
    h /= h_abs_sum if h_abs_sum > 1e-12 else 1.0
    return h

def _filtfilt_fir(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    y = np.apply_along_axis(lambda v: np.convolve(v, h, mode='same'), axis=0, arr=x)
    y = y[::-1, ...]
    y = np.apply_along_axis(lambda v: np.convolve(v, h, mode='same'), axis=0, arr=y)
    return y[::-1, ...]


# Signal Processing Pipeline
# -------------------------------------------------------------------------------
# Function: process_signals_matrix(...)
# Role: Given raw data, sample rate, and per-channel metadata (ACC/EMG and side),
# apply the configured filters (e.g., band-pass for ACC, high-/low-pass for EMG)
# and return a processed matrix. Analysis tabs (like FFT) consume these signals.
def process_signals_matrix(
    raw: Any, fs: float,
    active_cols_0based: List[int], per_channel_kind: List[str],
    acc_lo: float, acc_hi: float,
    emg_lo: float, emg_hi: float,
) -> Tuple[np.ndarray, List[int]]:
    """
    Build a processed matrix with selected columns:
      - ACC: band-pass (acc_lo..acc_hi) Hz
      - EMG: band-pass (emg_lo..emg_hi) Hz, rectified (abs)
    Returns (Y, valid_idx_map) where Y has shape [N, n_selected].
    """
    if not HAS_NUMPY:
        raise RuntimeError("NumPy is required for processing.")
    X = np.asarray(raw, dtype=float)
    N, M = X.shape
    take = []
    valid = []
    for i, col in enumerate(active_cols_0based):
        if 0 <= col < M:
            take.append(X[:, col:col+1])
            valid.append(i)
    if not take:
        return np.zeros((N, 0)), []
    data = np.hstack(take)
    kinds = [per_channel_kind[i] for i in valid]
    mask_acc = np.array([k == "ACC" for k in kinds])
    mask_emg = ~mask_acc

    if HAS_SCIPY:
        loA, hiA = _validate_band(fs, acc_lo, acc_hi)
        loE, hiE = _validate_band(fs, emg_lo, emg_hi)
        Y = data.copy()
        if np.any(mask_acc):
            Y[:, mask_acc] = _apply_scipy_sos(data[:, mask_acc], fs, loA, hiA)
        if np.any(mask_emg):
            Y[:, mask_emg] = _apply_scipy_sos(data[:, mask_emg], fs, loE, hiE)
            Y[:, mask_emg] = np.abs(Y[:, mask_emg])
        return Y, valid

    # FIR fallback if SciPy not available
    loA, hiA = _validate_band(fs, acc_lo, acc_hi)
    loE, hiE = _validate_band(fs, emg_lo, emg_hi)
    min_fc = max(loA if loA > 0 else 0.1, loE if loE > 0 else 0.1)
    L = int(min(2001, max(101, round(fs * 3 / max(min_fc, 0.1)))))
    hA = _fir_band(L, fs, loA, hiA)
    hE = _fir_band(L, fs, loE, hiE)
    Y = data.copy()
    if np.any(mask_acc):
        Y[:, mask_acc] = _filtfilt_fir(data[:, mask_acc], hA)
    if np.any(mask_emg):
        Y[:, mask_emg] = _filtfilt_fir(data[:, mask_emg], hE)
        Y[:, mask_emg] = np.abs(Y[:, mask_emg])
    return Y, valid

# ==========================================================
# Utility helpers shared by Frequency/Coherence/Spectrogram
# ==========================================================
@dataclass
class ChannelMeta:
    kinds: List[str]
    sides: List[str]
    labels: List[str]
    names: List[str] = field(default_factory=list)   # user-facing channel names (pre-dedup)

def _get_enabled_columns_and_meta(app_state: AppState) -> Tuple[List[int], List[str], List[str], List[str]]:
    """
    Return lists aligned by selected channel order from Setup:
      - cols0: 0-based column indexes
      - kinds: "ACC"/"EMG"
      - sides: "Left"/"Right"
      - names: user-facing channel name (custom, or a sensible fallback)
    Only includes channels with enabled=True.
    """
    cols0: List[int] = []
    kinds: List[str] = []
    sides: List[str] = []
    names: List[str] = []
    for i in range(8):
        cfg = app_state.channel_config.get(i, ChannelConfig(enabled=False, col_1based=i+1))
        if not cfg.enabled:
            continue
        cols0.append(cfg.col_1based - 1)
        kinds.append(cfg.kind)
        sides.append(cfg.side)
        names.append(_channel_display_name(cfg, i))
    return cols0, kinds, sides, names


# Selection Handling
# -------------------------------------------------------------------------------
# Helper to translate a (t0, t1) selection window into sample indices based on Fs.
# If no selection exists or t0==t1, the helper typically returns the full span.
# Downstream consumers (e.g., FFT) slice the processed matrix with these indices.
def _get_selection_indices(app_state: AppState, n: int) -> Tuple[int, int]:
    """
    Compute sample indices [i0, i1) for the currently selected segment.
    If no selection, returns the full range [0, n).
    """
    if not app_state.fs or app_state.fs <= 0:
        return 0, n
    if app_state.selection_t0 is None or app_state.selection_t1 is None or app_state.selection_t0 == app_state.selection_t1:
        return 0, n
    fs = app_state.fs
    t0, t1 = sorted((app_state.selection_t0, app_state.selection_t1))
    i0 = max(0, min(int(round(t0 * fs)), n - 1))
    i1 = max(i0 + 1, min(int(round(t1 * fs)), n))
    return i0, i1

def _next_pow2(n: int) -> int:
    """Return next power of 2 ≥ n."""
    if n <= 1:
        return 1
    return 1 << (int(n - 1).bit_length())

def _get_visualization_order_matrix(app_state: AppState, use_processed: bool = True) -> Tuple[np.ndarray, ChannelMeta]:
    """
    Build a (N × K) matrix for K enabled channels in the same Left/Right order used by Visualization,
    and return (Y, ChannelMeta). If use_processed is True, applies ACC/EMG filters like Visualization.
    """
    if app_state.raw_data is None:
        raise RuntimeError("No data loaded.")
    if not HAS_NUMPY:
        raise RuntimeError("NumPy is required.")
    X = np.asarray(app_state.raw_data, dtype=float)
    fs = app_state.fs or 1.0

    cols0, kinds_all, sides_all, names_all = _get_enabled_columns_and_meta(app_state)
    if not cols0:
        return np.zeros((len(X), 0)), ChannelMeta([], [], [], [])

    if use_processed:
        Y, valid = process_signals_matrix(
            raw=X, fs=fs, active_cols_0based=cols0, per_channel_kind=kinds_all,
            acc_lo=app_state.acc_bp_low, acc_hi=app_state.acc_bp_high,
            emg_lo=app_state.emg_hp, emg_hi=app_state.emg_lp
        )
        kinds = [kinds_all[i] for i in valid]
        sides = [sides_all[i] for i in valid]
        names = [names_all[i] for i in valid]
    else:
        # raw slice
        take = []
        valid = []
        for i, c in enumerate(cols0):
            if 0 <= c < X.shape[1]:
                take.append(X[:, c:c+1])
                valid.append(i)
        if not take:
            return np.zeros((len(X), 0)), ChannelMeta([], [], [], [])
        Y = np.hstack(take)
        kinds = [kinds_all[i] for i in valid]
        sides = [sides_all[i] for i in valid]
        names = [names_all[i] for i in valid]

    # Reorder to match Visualization's left/right two-column layout
    idx_left = [i for i, s in enumerate(sides) if s == "Left"]
    idx_right = [i for i, s in enumerate(sides) if s == "Right"]
    order = []
    for r in range(max(len(idx_left), len(idx_right))):
        if r < len(idx_left): order.append(idx_left[r])
        if r < len(idx_right): order.append(idx_right[r])
    if not order:
        return np.zeros((len(Y), 0)), ChannelMeta([], [], [], [])
    Yv = Y[:, order]
    kinds_v = [kinds[i] for i in order]
    sides_v = [sides[i] for i in order]
    names_v = [names[i] for i in order]
    # Disambiguate duplicate names (e.g. user left two channels with the
    # same custom name) by appending a counter, so combobox/label lookups
    # by string stay unique.
    seen: Dict[str, int] = {}
    labels: List[str] = []
    for nm in names_v:
        seen[nm] = seen.get(nm, 0) + 1
        labels.append(nm if seen[nm] == 1 else f"{nm} (#{seen[nm]})")
    return Yv, ChannelMeta(kinds_v, sides_v, labels, names_v)

def _get_flat_order_matrix(app_state: AppState, use_processed: bool = True) -> Tuple[np.ndarray, ChannelMeta]:
    """
    Like _get_visualization_order_matrix, but keeps the Setup channel order
    as-is (no Left/Right interleaving into two columns). Used by tabs that
    want every enabled channel listed in a single column — e.g. the
    Explore/Welch tab — instead of grouped by side.
    """
    if app_state.raw_data is None:
        raise RuntimeError("No data loaded.")
    if not HAS_NUMPY:
        raise RuntimeError("NumPy is required.")
    X = np.asarray(app_state.raw_data, dtype=float)
    fs = app_state.fs or 1.0

    cols0, kinds_all, sides_all, names_all = _get_enabled_columns_and_meta(app_state)
    if not cols0:
        return np.zeros((len(X), 0)), ChannelMeta([], [], [], [])

    if use_processed:
        Y, valid = process_signals_matrix(
            raw=X, fs=fs, active_cols_0based=cols0, per_channel_kind=kinds_all,
            acc_lo=app_state.acc_bp_low, acc_hi=app_state.acc_bp_high,
            emg_lo=app_state.emg_hp, emg_hi=app_state.emg_lp
        )
        kinds = [kinds_all[i] for i in valid]
        sides = [sides_all[i] for i in valid]
        names = [names_all[i] for i in valid]
    else:
        take = []
        valid = []
        for i, c in enumerate(cols0):
            if 0 <= c < X.shape[1]:
                take.append(X[:, c:c+1])
                valid.append(i)
        if not take:
            return np.zeros((len(X), 0)), ChannelMeta([], [], [], [])
        Y = np.hstack(take)
        kinds = [kinds_all[i] for i in valid]
        sides = [sides_all[i] for i in valid]
        names = [names_all[i] for i in valid]

    # Disambiguate duplicate names, same convention as the interleaved matrix.
    seen: Dict[str, int] = {}
    labels: List[str] = []
    for nm in names:
        seen[nm] = seen.get(nm, 0) + 1
        labels.append(nm if seen[nm] == 1 else f"{nm} (#{seen[nm]})")
    return Y, ChannelMeta(kinds, sides, labels, names)

def _limit_to_6(Y: np.ndarray, meta: ChannelMeta) -> Tuple[np.ndarray, ChannelMeta]:
    """Take at most the first 6 channels in visualization order."""
    k = min(Y.shape[1], 6)
    return Y[:, :k], ChannelMeta(meta.kinds[:k], meta.sides[:k], meta.labels[:k], meta.names[:k])

# --------------------
# Setup tab
# --------------------

# Setup Tab
# -------------------------------------------------------------------------------
# Purpose: Load raw data files (e.g., CSV), configure sample rate (Fs), choose
# which columns map to channels, and specify channel kinds (ACC vs EMG) and sides
# (Left vs Right). Also provides filter parameters for each kind.
# Outputs: Updates the shared AppState so other tabs can read current settings.
class SetupTab(ttk.Frame):
    def __init__(self, parent, app_state: AppState, log_cb, viz=None):
        super().__init__(parent)
        self.app_state = app_state
        self._log_cb = log_cb
        self._viz = viz  # VisualizationTab reference for auto-refresh

        # Tk variables
        self.var_path = tk.StringVar(value="")
        self.var_fs = tk.StringVar(value="1000")
        self.var_decimal = tk.BooleanVar(value=False)
        self.var_skip = tk.IntVar(value=0)

        self.ch_enabled: List[tk.BooleanVar] = []
        self.ch_col: List[tk.StringVar] = []
        self.ch_kind: List[tk.StringVar] = []
        self.ch_side: List[tk.StringVar] = []
        self.ch_name: List[tk.StringVar] = []

        self._build()

        # Defaults for quick start
        for i in range(8):
            self.ch_enabled[i].set(True if i < 6 else False)
            self.ch_col[i].set(str(i + 1))
            self.ch_kind[i].set("ACC" if i < 2 else "EMG")
            self.ch_side[i].set("Left" if i % 2 == 0 else "Right")
            self.ch_name[i].set(f"Ch{i+1}")

    def log(self, msg: str) -> None:
        self._log_cb(msg); _log_file(msg)

    def _build(self) -> None:
        # File picker + parser options
        lf = ttk.LabelFrame(self, text="Data file (.txt / .csv / .dat / .tsv / .smrx)")
        lf.pack(fill="x", padx=10, pady=10)

        ttk.Label(lf, text="Path:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(lf, textvariable=self.var_path, width=72).grid(row=0, column=1, sticky="we", padx=5, pady=5)
        ttk.Button(lf, text="Browse", command=self._browse).grid(row=0, column=2, sticky="e", padx=5, pady=5)
        lf.columnconfigure(1, weight=1)

        row = ttk.Frame(lf); row.grid(row=1, column=0, columnspan=3, sticky="we", padx=5, pady=4)
        ttk.Checkbutton(row, text="Decimal comma (e.g., 12,34 → 12.34)", variable=self.var_decimal).pack(side="left", padx=5)
        ttk.Label(row, text="Skip rows:").pack(side="left", padx=(12, 4))
        ttk.Spinbox(row, from_=0, to=100000, textvariable=self.var_skip, width=8).pack(side="left")

        # SMRX status label (kept up to date by _refresh_smrx_status)
        self._smrx_status_label = ttk.Label(lf, text="", foreground="gray")
        self._smrx_status_label.grid(
            row=2, column=0, columnspan=3, sticky="w", padx=5, pady=(0, 4))
        self._refresh_smrx_status()

        ttk.Button(lf, text="Load file", command=self._load).grid(
            row=3, column=0, columnspan=3, sticky="we", padx=5, pady=6)

        # Sampling frequency
        ffs = ttk.LabelFrame(self, text="Sampling frequency")
        ffs.pack(fill="x", padx=10, pady=5)
        ttk.Label(ffs, text="Fs (Hz):").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(ffs, textvariable=self.var_fs, width=10).grid(row=0, column=1, sticky="w", padx=5, pady=5)

        # Channel configuration
        fmap = ttk.LabelFrame(self, text="Channel configuration (up to 8)")
        fmap.pack(fill="x", padx=10, pady=5)
        headers = ("Use", "Ch", "Column (1-based)", "Type", "Side", "Name")
        for j, h in enumerate(headers):
            ttk.Label(fmap, text=h, font=("Segoe UI", 9, "bold")).grid(row=0, column=j, padx=6, pady=4, sticky="w")

        for i in range(8):
            v_en = tk.BooleanVar(value=False)
            v_col = tk.StringVar(value=str(i + 1))
            v_kind = tk.StringVar(value="ACC")
            v_side = tk.StringVar(value="Left")
            v_name = tk.StringVar(value=f"Ch{i+1}")
            self.ch_enabled.append(v_en)
            self.ch_col.append(v_col)
            self.ch_kind.append(v_kind)
            self.ch_side.append(v_side)
            self.ch_name.append(v_name)

            ttk.Checkbutton(fmap, variable=v_en).grid(row=i+1, column=0, padx=6, pady=2)
            ttk.Label(fmap, text=f"Ch {i+1}").grid(row=i+1, column=1, padx=6, pady=2, sticky="w")
            ttk.Spinbox(fmap, from_=1, to=256, textvariable=v_col, width=8).grid(row=i+1, column=2, padx=6, pady=2, sticky="w")
            ttk.Combobox(fmap, values=["ACC", "EMG"], textvariable=v_kind, state="readonly", width=8).grid(row=i+1, column=3, padx=6, pady=2, sticky="w")
            ttk.Combobox(fmap, values=["Left", "Right"], textvariable=v_side, state="readonly", width=8).grid(row=i+1, column=4, padx=6, pady=2, sticky="w")
            ttk.Entry(fmap, textvariable=v_name, width=18).grid(row=i+1, column=5, padx=6, pady=2, sticky="w")
        ttk.Label(fmap, text="← defaults to the column's name in the file; edit freely, used everywhere to identify this channel",
                  foreground="gray", font=("Segoe UI", 8)).grid(
            row=9, column=5, padx=6, pady=(0, 4), sticky="w")

        # Actions — "Apply changes" is only needed when manually editing channel config or Fs.
        # Loading a file triggers save automatically.
        actions = ttk.Frame(self); actions.pack(fill="x", padx=10, pady=10)
        ttk.Button(actions, text="Apply channel config", command=self._save).pack(side="left", padx=5)
        ttk.Label(actions, text="← only needed after manually editing Channel Configuration or Fs",
                  foreground="gray").pack(side="left", padx=6)

        # Session shortcut
        sep = ttk.Frame(self); sep.pack(fill="x", padx=10, pady=(0, 8))
        ttk.Separator(sep, orient="horizontal").pack(fill="x", pady=4)
        sess_row = ttk.Frame(sep); sess_row.pack(fill="x")
        ttk.Label(sess_row, text="Session:").pack(side="left", padx=(0, 8))
        self._var_sess_label = tk.StringVar()
        ttk.Entry(sess_row, textvariable=self._var_sess_label,
                  width=24, ).pack(side="left", padx=(0, 6))
        ttk.Label(sess_row, text="← maneuver name",
                  foreground="gray").pack(side="left")
        ttk.Button(sess_row, text="➕ Add to session",
                   command=self._add_to_session).pack(side="right", padx=4)

    def _refresh_smrx_status(self) -> None:
        """Update the sonpy/.smrx status label based on current availability."""
        if HAS_SONPY:
            ver = getattr(sonpy, "__version__", "installed")
            self._smrx_status_label.config(
                text=f"sonpy {ver} detected — .smrx files supported",
                foreground="dark green")
        elif os.path.isfile(_SONPY_CACHE_FILE):
            self._smrx_status_label.config(
                text="sonpy configured via a secondary Python runtime — .smrx files supported",
                foreground="dark green")
        else:
            self._smrx_status_label.config(
                text="sonpy not found — it will be installed automatically the first "
                     "time you load a .smrx file",
                foreground="gray")

    def _add_to_session(self) -> None:
        """Add current recording to the Comparison session."""
        if self.app_state.raw_data is None:
            messagebox.showinfo("No data", "Load a file first.")
            return
        # Get or generate label
        label = self._var_sess_label.get().strip()
        if not label:
            path  = self.app_state.options.get("source_path", "")
            label = path.split("/")[-1].split("\\")[-1].rsplit(".", 1)[0] or \
                    f"Maneuver {1}"

        # Find the session/comparison tab via the top-level window
        app = self.winfo_toplevel()
        session = getattr(app, "session", None)
        if session is None:
            messagebox.showerror("Error", "Session tab not found.")
            return
        session._add_with_label(label)
        # Suggest next label
        self._var_sess_label.set("")

    def _browse(self) -> None:
        path = filedialog.askopenfilename(
            title="Select file",
            filetypes=[
                ("All supported", "*.txt *.csv *.dat *.tsv *.smrx"),
                ("CED Spike2",    "*.smrx"),
                ("Text / CSV",    "*.txt *.csv *.dat *.tsv"),
                ("All files",     "*.*"),
            ]
        )
        if path:
            self.var_path.set(path)

    def _current_ui_channel_snapshot(self) -> str:
        """Build a human-readable snapshot of the 8 UI channel rows (pre-save)."""
        lines = ["Current channel config (pre-save):"]
        for i in range(8):
            en = bool(self.ch_enabled[i].get())
            col = self.ch_col[i].get()
            kind = self.ch_kind[i].get()
            side = self.ch_side[i].get()
            name = self.ch_name[i].get() if i < len(self.ch_name) else ""
            lines.append(f"  Ch{i+1} \"{name}\": use={en}  col={col}  {kind}-{side}")
        return "\n".join(lines)

    def _load(self) -> None:
        """Load the file and report rich info in the bottom log (incl. channels)."""
        path = self.var_path.get().strip()
        if not path:
            messagebox.showwarning("Warning", "Please select a file first.")
            return

        is_smrx = path.lower().endswith(".smrx")

        if is_smrx:
            self._load_smrx(path)
        else:
            self._load_text(path)

    def _load_smrx(self, path: str) -> None:
        """Load a CED Spike2 .smrx file.

        sonpy (only needed to read channel metadata) is located or installed
        automatically the first time it's required — see _resolve_sonpy_python.
        This can take a little while on the very first .smrx load (searching
        for / installing / possibly downloading a compatible Python); it is
        instant on every load after that.
        """
        self.log("[SMRX] Loading .smrx file (setting up sonpy automatically if needed)...")
        self.config(cursor="watch")
        self.update_idletasks()
        try:
            data, fs, titles = _read_smrx(path, log_cb=self.log)
        except Exception as e:
            tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            self.log(f"[SMRX ERROR]\n{tb}")
            messagebox.showerror("SMRX load error", str(e))
            return
        finally:
            self.config(cursor="")
            self._refresh_smrx_status()

        self.app_state.raw_data = data
        self.app_state.fs = fs
        self.var_fs.set(str(round(fs, 2)))
        self.app_state.options.update({"source_path": path, "format": "smrx"})

        n_ch = data.shape[1]
        self.log(
            f"[OK] SMRX loaded\n"
            f"  Path: {path}\n"
            f"  Channels ({n_ch}): {', '.join(titles)}\n"
            f"  Fs: {fs:.2f} Hz\n"
            f"  Duration: {data.shape[0]/fs:.2f} s\n"
            f"  Shape: {data.shape[0]} × {data.shape[1]}"
        )

        # Auto-configure channel UI using robust name inference
        inferred = _infer_channel_config(titles, min(n_ch, 8), first_signal_col_1based=1)
        for i, cfg in enumerate(inferred):
            self.ch_enabled[i].set(True)
            self.ch_col[i].set(str(i + 1))
            self.ch_kind[i].set(cfg["kind"])
            self.ch_side[i].set(cfg["side"])
            self.ch_name[i].set(cfg.get("title") or f"Ch{i+1}")
        for i in range(len(inferred), 8):
            self.ch_enabled[i].set(False)

        self.log("[SMRX] Channel UI auto-configured from titles. Loading visualization...")

        # Suggest a session label from filename
        fname = path.split("/")[-1].split("\\")[-1].rsplit(".", 1)[0]
        self._var_sess_label.set(fname)

        # Auto-save setup so visualization appears immediately
        self._save()

    def _load_text(self, path: str) -> None:
        """Load a plain-text / CSV file."""
        dec = bool(self.var_decimal.get())
        skip = int(self.var_skip.get())

        try:
            data, delim, total_skipped, col_titles = parse_txt_anycols(
                path, skip_rows=skip, decimal_comma=dec)
        except Exception as e:
            self.log("[ERROR] Load failed:\n" + "".join(traceback.format_exception_only(type(e), e)))
            messagebox.showerror("Load error", str(e))
            return

        auto_skipped = total_skipped - skip
        if auto_skipped > 0:
            self.var_skip.set(total_skipped)
            self.log(f"[Info] Auto-detected and skipped {auto_skipped} header row(s). "
                     f"'Skip rows' updated to {total_skipped}.")

        self.app_state.raw_data = data
        self.app_state.options.update({
            "source_path": path,
            "decimal_comma": dec,
            "skip_rows": skip,
            "detected_delimiter": delim
        })

        # shape + preview
        try:
            nrows = len(data)
            ncols = len(data[0]) if nrows > 0 else 0
        except Exception:
            shp = getattr(data, "shape", (0, 0))
            nrows, ncols = shp[0], (shp[1] if len(shp) > 1 else 0)

        head_rows = min(5, nrows if nrows else 5)
        head_cols = min(8, ncols if ncols else 8)
        preview_lines = []
        for r in range(head_rows):
            row = data[r]
            as_list = row.tolist() if HAS_NUMPY else row
            preview_lines.append(", ".join(f"{x:.6g}" for x in as_list[:head_cols]))

        self.log(
            "[OK] File loaded\n"
            f"  Path: {path}\n"
            f"  Decimal comma: {dec}\n"
            f"  Skip rows: {total_skipped} (manual={skip}, auto-detected={auto_skipped})\n"
            f"  Detected delimiter: {repr(delim)}\n"
            f"  Shape: {nrows} rows × {ncols} columns\n"
            f"  Loaded channels: {ncols}\n"
            f"  Preview (first {head_rows} rows, first {head_cols} cols):\n    " +
            "\n    ".join(preview_lines)
        )
        self.log(self._current_ui_channel_snapshot())

        if ncols > 8:
            self.log("[Note] You loaded more than 8 columns; the UI maps at most 8.")

        # ── Auto-configure channel UI ────────────────────────────────────
        # Detect if column 0 is a time axis
        col0 = data[:, 0] if ncols > 0 else np.array([])
        is_time_col = (ncols > 1 and
                       np.all(np.diff(col0[np.isfinite(col0)]) > 0) and
                       float(col0[0]) >= 0 and float(col0[0]) < 1.0)
        first_signal_col = 2 if is_time_col else 1   # 1-based

        n_signal_cols = ncols - (1 if is_time_col else 0)
        n_ch = min(n_signal_cols, 8)

        # Strip the time column title (if present) to get signal titles
        signal_titles: List[str] = []
        if col_titles:
            start = 1 if is_time_col else 0
            signal_titles = col_titles[start : start + n_ch]

        # Use robust inference (names from header if available, else positional)
        inferred = _infer_channel_config(signal_titles, n_ch, first_signal_col)

        for i, cfg in enumerate(inferred):
            self.ch_enabled[i].set(True)
            self.ch_col[i].set(str(cfg["col_1based"]))
            self.ch_kind[i].set(cfg["kind"])
            self.ch_side[i].set(cfg["side"])
            self.ch_name[i].set(cfg.get("title") or f"Ch{i+1}")
        for i in range(n_ch, 8):
            self.ch_enabled[i].set(False)

        src = "header names" if signal_titles else "column position"
        if is_time_col:
            self.log(f"[Info] Column 1 detected as Time axis — "
                     f"signals: cols {first_signal_col}–{first_signal_col+n_ch-1}.")
        self.log(f"[Info] Auto-configured {n_ch} channel(s) using {src}. "
                 f"Review Type/Side in Setup if needed.")

        # Suggest a session label from filename
        fname = path.split("/")[-1].split("\\")[-1].rsplit(".", 1)[0]
        self._var_sess_label.set(fname)

        # Auto-save setup so visualization appears immediately
        self._save()

    def _save(self) -> None:
        """Validate Fs and store channel mapping into AppState; log a concise summary."""
        try:
            fs = float(self.var_fs.get())
            if fs <= 0:
                raise ValueError("Fs must be positive.")
        except Exception:
            messagebox.showwarning("Sampling frequency", "Fs must be a positive number.")
            return
        self.app_state.fs = fs

        cfg: Dict[int, ChannelConfig] = {}
        for i in range(8):
            en = bool(self.ch_enabled[i].get())
            try:
                col = int(float(self.ch_col[i].get()))
            except Exception:
                col = i + 1
            col = max(1, col)
            kind = self.ch_kind[i].get() or "ACC"
            side = self.ch_side[i].get() or "Left"
            name = self.ch_name[i].get().strip() if i < len(self.ch_name) else ""
            cfg[i] = ChannelConfig(enabled=en, col_1based=col, kind=kind, side=side, name=name)
        self.app_state.channel_config = cfg

        enabled = [f"Ch{i+1} \"{_channel_display_name(c, i)}\"(col={c.col_1based},{c.kind}-{c.side})"
                   for i, c in cfg.items() if c.enabled]
        self.log(
            "[OK] Setup saved\n"
            f"  Fs: {self.app_state.fs} Hz\n"
            f"  Enabled channels ({len(enabled)}): " + (", ".join(enabled) if enabled else "(none)")
        )
        # Auto-refresh Visualization so data appears immediately after saving
        if self._viz is not None:
            try:
                self._viz.redraw()
            except Exception:
                pass
        # Auto-compute FFT so it's ready without needing Refresh
        if getattr(self, '_fft', None) is not None:
            try:
                self._fft.redraw()
            except Exception:
                pass
        # Auto-refresh Explore/Welch (same reasoning as Visualization/FFT)
        if getattr(self, '_explore', None) is not None:
            try:
                self._explore.redraw()
            except Exception:
                pass

# --------------------
# Visualization tab
# --------------------

# Visualization Tab (Time Domain)
# -------------------------------------------------------------------------------
# Purpose: Plot time-domain signals for enabled channels. Layout typically mirrors
# left/right sides in two columns. Supports interactive zoom and a selection
# mechanism to choose a time segment (t0–t1) used by analysis tabs.
# Important: Y-axis may be synchronized across channels of the same type (ACC/EMG)
# for fair comparison, depending on your implementation.
class VisualizationTab(ttk.Frame):
    SIGNAL_VIEW = ["Processed", "Raw"]

    def __init__(self, parent, app_state: AppState, log_cb):
        super().__init__(parent)
        self.app_state = app_state
        self._log_cb = log_cb

        # UI variables
        self.var_signal = tk.StringVar(value=self.SIGNAL_VIEW[0])
        self.var_t0 = tk.StringVar(value="0")
        self.var_tw = tk.StringVar(value="")  # empty = until end
        self.var_acc_lo = tk.StringVar(value=str(self.app_state.acc_bp_low))
        self.var_acc_hi = tk.StringVar(value=str(self.app_state.acc_bp_high))
        self.var_emg_lo = tk.StringVar(value=str(self.app_state.emg_hp))
        self.var_emg_hi = tk.StringVar(value=str(self.app_state.emg_lp))

        # Fit Y-axis toggle state: False = shared scale by type (default), True = fit to view
        self._fit_y_active: bool = False

        # Cache last drawn segment/meta for re-scaling without full redraw
        self._last_seg: Optional[Any] = None
        self._last_meta: Optional[Any] = None

        # Matplotlib artifacts
        self.fig = None
        self.canvas = None
        self._axes: List[Any] = []

        self._build()

        # Auto-redraw when Signal type changes
        self.var_signal.trace_add("write", lambda *_: self.redraw())

    def log(self, msg: str) -> None:
        self._log_cb(msg); _log_file(msg)

    def _build(self) -> None:
        # Controls row
        ctrl = ttk.Frame(self); ctrl.pack(fill="x", padx=10, pady=6)
        ttk.Label(ctrl, text="Signal:").grid(row=0, column=0, sticky="w")
        ttk.Combobox(ctrl, values=self.SIGNAL_VIEW, textvariable=self.var_signal,
                     state="readonly", width=12).grid(row=0, column=1, sticky="w", padx=4)

        ttk.Label(ctrl, text="Start (s):").grid(row=0, column=2, sticky="e", padx=(12, 2))
        ttk.Entry(ctrl, textvariable=self.var_t0, width=8).grid(row=0, column=3, sticky="w")
        ttk.Label(ctrl, text="Window (s):").grid(row=0, column=4, sticky="e", padx=(12, 2))
        ttk.Entry(ctrl, textvariable=self.var_tw, width=8).grid(row=0, column=5, sticky="w")

        ttk.Button(ctrl, text="Refresh", command=self.redraw).grid(row=0, column=6, sticky="w", padx=10)

        # "Fit Y-axis" toggle button — scales each subplot to its own visible peak, reversible
        self._btn_fit = ttk.Button(ctrl, text="Fit Y-axis", command=self._toggle_fit_y)
        self._btn_fit.grid(row=0, column=7, sticky="w", padx=(4, 10))

        # Filter row
        row = ttk.Frame(self); row.pack(fill="x", padx=10, pady=4)
        ttk.Label(row, text="ACC BP (Hz):").grid(row=0, column=0, sticky="w")
        ttk.Label(row, text="Low").grid(row=0, column=1, sticky="e")
        ttk.Entry(row, textvariable=self.var_acc_lo, width=7).grid(row=0, column=2, sticky="w", padx=4)
        ttk.Label(row, text="High").grid(row=0, column=3, sticky="e")
        ttk.Entry(row, textvariable=self.var_acc_hi, width=7).grid(row=0, column=4, sticky="w", padx=4)

        ttk.Label(row, text="EMG band (Hz):").grid(row=0, column=5, sticky="e", padx=(12, 2))
        ttk.Label(row, text="HP").grid(row=0, column=6, sticky="e")
        ttk.Entry(row, textvariable=self.var_emg_lo, width=7).grid(row=0, column=7, sticky="w", padx=4)
        ttk.Label(row, text="LP").grid(row=0, column=8, sticky="e")
        ttk.Entry(row, textvariable=self.var_emg_hi, width=7).grid(row=0, column=9, sticky="w", padx=4)

        ttk.Button(row, text="Apply filters", command=self._apply_filters).grid(row=0, column=10, sticky="w", padx=10)

        # Horizontal scrollbar — below the plot, always visible. Packed
        # BEFORE the expanding plot area (side="bottom" reserves its strip
        # first) so it always keeps its space regardless of how tall the
        # figure/canvas above ends up being.
        scroll_frame = ttk.Frame(self, relief="groove", borderwidth=1)
        scroll_frame.pack(side="bottom", fill="x", padx=8, pady=4)

        ttk.Label(scroll_frame, text="Pan:",
                  width=4).pack(side="left", padx=(6, 2))
        self._hscroll = ttk.Scale(scroll_frame, from_=0.0, to=1.0,
                                   orient="horizontal",
                                   command=self._on_hscroll)
        self._hscroll.pack(side="left", fill="x", expand=True, padx=4, pady=3)
        ttk.Label(scroll_frame,
                  text="scroll wheel = zoom",
                  foreground="gray", font=("TkDefaultFont", 8)
                  ).pack(side="right", padx=8)

        # Plot area
        plot = ttk.Frame(self); plot.pack(fill="both", expand=True, padx=8, pady=(2,0))
        if MATPLOTLIB_OK:
            self.fig = plt.Figure(figsize=(10, 6), dpi=100)
            self.canvas = FigureCanvasTkAgg(self.fig, master=plot)

            # Toolbar at TOP of plot area (more visible)
            toolbar_frame = ttk.Frame(plot)
            toolbar_frame.pack(side="top", fill="x")
            NavigationToolbar2Tk(self.canvas, toolbar_frame).update()

            self.canvas.get_tk_widget().pack(fill="both", expand=True)

            # Mouse scroll zoom
            self.canvas.mpl_connect("scroll_event", self._on_scroll)
        else:
            ttk.Label(plot, text="Matplotlib not available").pack(pady=20)

    def _on_scroll(self, event) -> None:
        """Zoom in/out with mouse wheel, centered on cursor position."""
        axes = self.fig.get_axes()
        if not axes: return
        ax0 = axes[0]
        xlo, xhi = ax0.get_xlim()
        xrange = xhi - xlo
        if xrange <= 0: return

        factor = 0.75 if event.button == "up" else 1.25
        x_center = event.xdata if event.xdata is not None else (xlo + xhi) / 2
        x_center = max(xlo, min(xhi, x_center))

        new_lo = x_center - (x_center - xlo) * factor
        new_hi = x_center + (xhi - x_center) * factor

        t_total = getattr(self, '_total_duration', None) or (xhi * 2)
        new_lo = max(0.0, new_lo)
        new_hi = min(t_total, new_hi)
        if new_hi - new_lo < 0.5: return

        for ax in axes:
            ax.set_xlim(new_lo, new_hi)
        self._sync_scrollbar(new_lo, new_hi)
        self.canvas.draw_idle()

    def _on_hscroll(self, val) -> None:
        """Pan the time window using the horizontal scrollbar."""
        axes = self.fig.get_axes()
        if not axes: return
        ax0 = axes[0]
        xlo, xhi = ax0.get_xlim()
        win = max(xhi - xlo, 0.5)
        t_total = getattr(self, '_total_duration', None) or win
        if t_total <= win:
            return

        pos = float(val) * (t_total - win)
        new_lo = max(0.0, min(t_total - win, pos))
        new_hi = new_lo + win
        for ax in axes:
            ax.set_xlim(new_lo, new_hi)
        self.canvas.draw_idle()

    def _sync_scrollbar(self, xlo: float, xhi: float) -> None:
        """Update scrollbar position to match current view."""
        t_total = getattr(self, '_total_duration', None)
        if not t_total or t_total <= 0: return
        win = xhi - xlo
        if win >= t_total:
            self._hscroll.set(0.0)
        else:
            pos = xlo / max(t_total - win, 1e-9)
            self._hscroll.set(max(0.0, min(1.0, pos)))

    def _apply_filters(self) -> None:
        """Refresh processing cutoffs and redraw."""
        try:
            self.app_state.acc_bp_low = float(self.var_acc_lo.get())
            self.app_state.acc_bp_high = float(self.var_acc_hi.get())
            self.app_state.emg_hp = float(self.var_emg_lo.get())
            self.app_state.emg_lp = float(self.var_emg_hi.get())
        except Exception:
            messagebox.showwarning("Filters", "Numeric values are required.")
            return
        self.redraw()

    def _toggle_fit_y(self) -> None:
        """Toggle between per-subplot Y fit and shared-by-type Y scale."""
        self._fit_y_active = not self._fit_y_active
        # Update button appearance to reflect state
        if self._fit_y_active:
            self._btn_fit.config(text="Fit Y-axis ✓")
        else:
            self._btn_fit.config(text="Fit Y-axis")
        # Re-apply scale to existing axes without a full redraw
        if self._axes:
            self._apply_y_scaling(self._axes, self._last_seg, self._last_meta)
            self.canvas.draw_idle()

    def _apply_y_scaling(self, axes: List[Any], seg: Any, meta: Any) -> None:
        """
        Apply Y-axis scaling to all active axes.
        - Fit Y-axis ON : each subplot independently scaled to its visible data peak.
        - Fit Y-axis OFF: shared scale within ACC group and within EMG group.
        """
        if seg is None or meta is None:
            return

        idx_left  = [j for j, s in enumerate(meta.sides) if s == "Left"]
        idx_right = [j for j, s in enumerate(meta.sides) if s == "Right"]
        nrows = max(len(idx_left), len(idx_right))

        # Build mapping: axis index → channel index in seg
        ax_chan: List[Optional[int]] = []
        for r in range(nrows):
            ax_chan.append(idx_left[r]  if r < len(idx_left)  else None)
            ax_chan.append(idx_right[r] if r < len(idx_right) else None)

        active_axes = [ax for ax in axes if ax.get_visible() and ax.lines]

        if self._fit_y_active:
            # Per-subplot: fit to the data currently visible on screen (x range)
            for ax, ch in zip(axes, ax_chan):
                if ch is None or not ax.lines:
                    continue
                xlo, xhi = ax.get_xlim()
                y = seg[:, ch]
                # Build x array matching seg (same logic as _time_indices result)
                # We stored x values in the line's xdata
                try:
                    xdata = ax.lines[0].get_xdata()
                    ydata = ax.lines[0].get_ydata()
                    mask = (xdata >= xlo) & (xdata <= xhi)
                    visible_y = ydata[mask]
                except Exception:
                    visible_y = y
                if visible_y.size == 0:
                    continue
                ymin = float(np.nanmin(visible_y))
                ymax = float(np.nanmax(visible_y))
                if not np.isfinite(ymin) or not np.isfinite(ymax) or ymin == ymax:
                    margin = max(1.0, abs(ymax) * 0.05)
                else:
                    margin = (ymax - ymin) * 0.05
                ax.set_ylim(ymin - margin, ymax + margin)
        else:
            # Shared scale by type (ACC together, EMG together)
            acc_ylims: List[Tuple[float, float]] = []
            emg_ylims: List[Tuple[float, float]] = []
            for ax, ch in zip(axes, ax_chan):
                if ch is None or not ax.lines:
                    continue
                kind = meta.kinds[ch]
                y = seg[:, ch]
                if not np.any(np.isfinite(y)):
                    continue
                lo = float(np.nanmin(y)); hi = float(np.nanmax(y))
                if kind == "ACC":
                    acc_ylims.append((lo, hi))
                else:
                    emg_ylims.append((lo, hi))

            def _group_lim(ylims):
                if not ylims:
                    return None
                lo = min(v[0] for v in ylims)
                hi = max(v[1] for v in ylims)
                if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                    return (lo - 1.0, hi + 1.0)
                m = (hi - lo) * 0.05
                return (lo - m, hi + m)

            acc_lim = _group_lim(acc_ylims)
            emg_lim = _group_lim(emg_ylims)
            for ax, ch in zip(axes, ax_chan):
                if ch is None or not ax.lines:
                    continue
                lim = acc_lim if meta.kinds[ch] == "ACC" else emg_lim
                if lim is not None:
                    ax.set_ylim(*lim)

    def _time_indices(self, n: int) -> Tuple[int, int, List[float]]:
        """Convert (Start, Window) seconds into sample indices and x-axis in seconds."""
        fs = self.app_state.fs or 1.0
        try:
            t0 = max(0.0, float(self.var_t0.get()))
        except Exception:
            t0 = 0.0
        raw = self.var_tw.get().strip()
        if raw == "":
            tw = max((n / fs) - t0, 0.01)
        else:
            try:
                tw = float(raw)
                if tw <= 0:
                    tw = max((n / fs) - t0, 0.01)
            except Exception:
                tw = max((n / fs) - t0, 0.01)
        i0 = int(round(t0 * fs))
        i1 = int(round((t0 + tw) * fs))
        i0 = max(0, min(i0, n - 1))
        i1 = max(i0 + 1, min(i1, n))
        x = [(i - i0) / fs + t0 for i in range(i0, i1)]
        return i0, i1, x

    def redraw(self) -> None:
        """Rebuild the figure: columns grouped by Left/Right (2 columns)."""
        if not MATPLOTLIB_OK:
            return
        raw = self.app_state.raw_data
        if raw is None:
            messagebox.showinfo("No data", "Load a file in the Setup tab first.")
            return
        if not self.app_state.fs or self.app_state.fs <= 0:
            messagebox.showinfo("Missing Fs", "Set Fs in Setup, then Save.")
            return

        # Choose raw or processed to match UI
        use_processed = (self.var_signal.get() == "Processed")
        Yv, meta = _get_visualization_order_matrix(self.app_state, use_processed=use_processed)
        if Yv.shape[1] == 0:
            messagebox.showinfo("No channels", "Enable channels in Setup.")
            return

        data_arr = np.asarray(raw, dtype=float)
        n = len(data_arr)
        i0, i1, x = self._time_indices(n)

        # Store total duration for scroll/zoom bounds
        self._total_duration = n / self.app_state.fs

        # Clip to window
        seg = Yv[i0:i1, :]

        # Cache for _toggle_fit_y (re-scaling without full redraw)
        self._last_seg = seg
        self._last_meta = meta

        # Split by side indexes in Yv order
        idx_left = [j for j, s in enumerate(meta.sides) if s == "Left"]
        idx_right = [j for j, s in enumerate(meta.sides) if s == "Right"]
        nrows = max(len(idx_left), len(idx_right))
        if nrows == 0:
            return

        self.fig.clf()
        # Tight vertical spacing (hspace) + per-panel channel name as the
        # y-label instead of a title bar, with the time axis only labeled on
        # each column's last populated row — frees up the vertical space a
        # per-row title/x-label used to take, so each trace gets more height
        # to actually be seen (closer to how Spike2 stacks channels).
        axes = self.fig.subplots(nrows, 2, squeeze=False, sharex=True,
                                  gridspec_kw={'hspace': 0.08})
        self._axes = []

        # Build axis→channel mapping in same order used by _apply_y_scaling
        ax_list: List[Any] = []
        ch_list: List[Optional[int]] = []

        def add(ax, y, title, k, ch_idx, show_xlabel):
            ax.plot(x, y, linewidth=0.8, color=("green" if k == "ACC" else None))
            ax.set_ylabel(title, fontsize=8)
            ax.grid(True, alpha=0.3)
            if show_xlabel:
                ax.set_xlabel("Time (s)")
            else:
                ax.tick_params(labelbottom=False)
            self._axes.append(ax)
            ax_list.append(ax)
            ch_list.append(ch_idx)

        for r in range(nrows):
            axL = axes[r, 0]
            if r < len(idx_left):
                j = idx_left[r]
                add(axL, seg[:, j], f"Left — {meta.names[j]}", meta.kinds[j], j,
                    show_xlabel=(r == len(idx_left) - 1))
            else:
                axL.axis("off")
                self._axes.append(axL)
                ax_list.append(axL)
                ch_list.append(None)
            axR = axes[r, 1]
            if r < len(idx_right):
                j = idx_right[r]
                add(axR, seg[:, j], f"Right — {meta.names[j]}", meta.kinds[j], j,
                    show_xlabel=(r == len(idx_right) - 1))
            else:
                axR.axis("off")
                self._axes.append(axR)
                ax_list.append(axR)
                ch_list.append(None)

        # Apply Y scaling (fit per-subplot or shared by type)
        self._apply_y_scaling(ax_list, seg, meta)

        self.fig.tight_layout(h_pad=0.3)
        self.fig.subplots_adjust(hspace=0.08)
        self.canvas.draw_idle()

        # Reset scrollbar to beginning after full redraw
        if hasattr(self, '_hscroll'):
            self._hscroll.set(0.0)

        # Shade selected segment if exists (global absolute time)
        if self.app_state.selection_t0 is not None and self.app_state.selection_t1 is not None:
            t0s, t1s = sorted((self.app_state.selection_t0, self.app_state.selection_t1))
            for ax in self._axes:
                try:
                    ax.axvspan(t0s, t1s, alpha=0.2, lw=0)
                except Exception:
                    pass
            self.canvas.draw_idle()

# --------------------
# Selection tab
# --------------------

# Selection Tab
# -------------------------------------------------------------------------------
# Purpose: Provide a dedicated interface to view and edit the selection interval
# (t0–t1). The selection can be created in Visualization and refined here.
# Effect: The selection bounds are stored in AppState and used by analysis tabs.
class SelectionTab(ttk.Frame):
    """
    Displays one guidance channel with:
      • Combobox to choose which channel to plot (default Left/ACC if available,
        else the first enabled channel).
      • Horizontal SpanSelector for manual time selection.
      • Manual inputs (Start/End in seconds) + Apply button.
    Saves selection to global AppState: selection_t0, selection_t1.
    """

    def __init__(self, parent, app_state: AppState, viz: VisualizationTab, log_cb):
        super().__init__(parent)
        self.app_state = app_state
        self.viz = viz           # to remain consistent with 'Processed' vs 'Raw'
        self._log_cb = log_cb

        # UI variables
        self.var_channel_label = tk.StringVar(value="")
        self.var_method = tk.StringVar(value="mouse")    # "mouse" or "time"
        self.var_t0 = tk.StringVar(value="0")
        self.var_t1 = tk.StringVar(value="1")

        # Internal caches for guidance data
        self._t: Optional[np.ndarray] = None
        self._Y: Optional[np.ndarray] = None
        self._labels: List[str] = []
        self._kind: List[str] = []
        self._side: List[str] = []

        # Matplotlib
        self.fig = None
        self.ax = None
        self.canvas = None
        self._span: Optional[SpanSelector] = None

        self._build()

    def log(self, msg: str) -> None:
        self._log_cb(msg); _log_file(msg)

    def _build(self) -> None:
        top = ttk.Frame(self); top.pack(fill="x", padx=10, pady=8)

        # Channel chooser
        ttk.Label(top, text="Channel:").pack(side="left")
        self.cb = ttk.Combobox(top, textvariable=self.var_channel_label, values=[],
                               state="readonly", width=36)
        self.cb.pack(side="left", padx=6)
        ttk.Button(top, text="Refresh list", command=self._prepare_guidance_data).pack(side="left", padx=(6, 12))

        # Method toggles
        ttk.Label(top, text="Method:").pack(side="left")
        ttk.Radiobutton(top, text="Manual (mouse drag)", variable=self.var_method, value="mouse",
                        command=self._toggle_method).pack(side="left", padx=(6, 0))
        ttk.Radiobutton(top, text="By time (s)", variable=self.var_method, value="time",
                        command=self._toggle_method).pack(side="left", padx=(8, 0))

        # Manual time inputs
        row = ttk.Frame(self); row.pack(fill="x", padx=10, pady=(4, 0))
        ttk.Label(row, text="Start (s):").pack(side="left")
        ttk.Entry(row, textvariable=self.var_t0, width=10).pack(side="left", padx=(4, 10))
        ttk.Label(row, text="End (s):").pack(side="left")
        ttk.Entry(row, textvariable=self.var_t1, width=10).pack(side="left", padx=(4, 10))
        ttk.Button(row, text="Apply", command=self._apply_manual_times).pack(side="left", padx=(6, 0))
        ttk.Button(row, text="Clear", command=self._clear_selection).pack(side="left", padx=6)

        # Plot
        plot = ttk.Frame(self); plot.pack(fill="both", expand=True, padx=8, pady=8)
        if MATPLOTLIB_OK:
            self.fig = plt.Figure(figsize=(10, 4), dpi=100)
            self.ax = self.fig.add_subplot(1, 1, 1)
            self.canvas = FigureCanvasTkAgg(self.fig, master=plot)
            self.canvas.get_tk_widget().pack(fill="both", expand=True)
            NavigationToolbar2Tk(self.canvas, plot).update()
        else:
            ttk.Label(plot, text="Matplotlib not available").pack(pady=20)

        # Events
        self.cb.bind("<<ComboboxSelected>>", lambda e: self._plot_guidance())
        # Initial fill
        self._prepare_guidance_data()
        self._toggle_method()

    # ---------- Data prep ----------
    def _pick_default_left_acc(self, kinds: List[str], sides: List[str]) -> int:
        for idx, (k, s) in enumerate(zip(kinds, sides)):
            if k == "ACC" and s == "Left":
                return idx
        return 0

    def _prepare_guidance_data(self) -> None:
        data = self.app_state.raw_data
        fs = self.app_state.fs
        if data is None or not fs or fs <= 0:
            self.cb['values'] = []; self.var_channel_label.set("")
            return

        # Use processed to guide selection, consistent with Visualization default
        Yv, meta = _get_visualization_order_matrix(self.app_state, use_processed=True)
        if Yv.shape[1] == 0:
            self.cb['values'] = []; self.var_channel_label.set("")
            return

        t = np.arange(Yv.shape[0]) / fs
        self._t = t
        self._Y = Yv
        self._labels = meta.labels
        self._kind = meta.kinds
        self._side = meta.sides

        self.cb['values'] = self._labels
        if not self.var_channel_label.get():
            self.var_channel_label.set(self._labels[self._pick_default_left_acc(meta.kinds, meta.sides)])
        self._plot_guidance()

    # ---------- Plot + interactions ----------
    def _plot_guidance(self) -> None:
        if not MATPLOTLIB_OK or self.ax is None or self._Y is None or self._t is None:
            return
        label = self.var_channel_label.get()
        if label not in self._labels:
            self.var_channel_label.set(self._labels[0]); label = self._labels[0]
        idx = self._labels.index(label)

        self.ax.clear()
        k = self._kind[idx]
        self.ax.plot(self._t, self._Y[:, idx], linewidth=0.9, color=("green" if k == "ACC" else None))
        self.ax.set_title(f"Selection (guidance): {label}")
        self.ax.set_xlabel("Time (s)")
        self.ax.grid(True, alpha=0.3)

        self._draw_local_shading()
        self.fig.tight_layout()
        self.canvas.draw_idle()

        if self.var_method.get() == "mouse":
            self._arm_span_selector()

    def _toggle_method(self) -> None:
        if self.var_method.get() == "mouse":
            self._arm_span_selector()
        else:
            self._disarm_span_selector()

    def _arm_span_selector(self) -> None:
        if not MATPLOTLIB_OK or self.ax is None:
            return
        self._disarm_span_selector()

        def on_select(xmin, xmax):
            if xmin == xmax:
                return
            lo, hi = sorted((xmin, xmax))
            self.var_t0.set(f"{lo:.6f}")
            self.var_t1.set(f"{hi:.6f}")
            self._apply_selection(lo, hi)

        self._span = SpanSelector(self.ax, on_select, "horizontal",
                                  useblit=True, interactive=True,
                                  props=dict(alpha=0.15), grab_range=5)

    def _disarm_span_selector(self) -> None:
        if self._span is not None:
            try:
                self._span.disconnect_events()
            except Exception:
                pass
            self._span = None

    def _draw_local_shading(self) -> None:
        t0, t1 = self.app_state.selection_t0, self.app_state.selection_t1
        if t0 is None or t1 is None or t0 == t1 or self.ax is None:
            return
        lo, hi = sorted((t0, t1))
        try:
            self.ax.axvspan(lo, hi, alpha=0.2, lw=0)
        except Exception:
            pass

    def _clear_selection(self) -> None:
        self.app_state.selection_t0 = None
        self.app_state.selection_t1 = None
        self.var_t0.set("0"); self.var_t1.set("1")
        if self.canvas:
            self.canvas.draw_idle()
        self.log("[Selection] Cleared")

    def _apply_manual_times(self) -> None:
        try:
            t0 = float(self.var_t0.get())
            t1 = float(self.var_t1.get())
        except Exception:
            messagebox.showerror("Selection", "Start/End must be numeric seconds.")
            return
        if t0 == t1:
            messagebox.showerror("Selection", "Start and End cannot be equal.")
            return
        self._apply_selection(*sorted((t0, t1)))

    def _apply_selection(self, t0: float, t1: float) -> None:
        self.app_state.selection_t0 = float(t0)
        self.app_state.selection_t1 = float(t1)
        if self.canvas:
            self.canvas.draw_idle()
        self.log(f"[Selection] {t0:.6f} – {t1:.6f} s")

# ====================================
# Frequency domain tab (FFT of 6 chans)
# ====================================
# NOTE: FrequencyTab below is UNUSED (superseded by FFTTab registered in the Notebook).
# It is kept here only for reference. Do not instantiate it directly.
class FrequencyTab(ttk.Frame):
    """
    FFT magnitude for up to 6 channels (Visualization order).
    • Interactive toolbar for pan/zoom.
    • X-axis synchronized across ALL subplots (sharex=True).
    • Y-axis synchronized BY TYPE (ACC together, EMG together).
    • Default view 2–20 Hz; robust to short selections or low Fs/2 (no blank panel).
    """

    def __init__(self, parent, app_state: AppState, viz: VisualizationTab, log_cb):
        super().__init__(parent)
        self.app_state = app_state; self.viz = viz; self._log_cb = log_cb
        self.fig=None; self.canvas=None
        self._axes: List[Any] = []
        self._build()

    def log(self, msg: str) -> None:
        self._log_cb(msg); _log_file(msg)

    def _build(self) -> None:
        ctrl = ttk.Frame(self); ctrl.pack(fill="x", padx=10, pady=6)
        ttk.Button(ctrl, text="Refresh", command=self.redraw).pack(side="left", padx=5)
        ttk.Label(ctrl, text="Note: x-axis synced across plots; y-axis synced by type (ACC vs EMG).").pack(side="left", padx=8)

        plot = ttk.Frame(self); plot.pack(fill="both", expand=True, padx=8, pady=6)
        if MATPLOTLIB_OK:
            self.fig = plt.Figure(figsize=(10, 6), dpi=100)
            self.canvas = FigureCanvasTkAgg(self.fig, master=plot)
            self.canvas.get_tk_widget().pack(fill="both", expand=True)
            NavigationToolbar2Tk(self.canvas, plot).update()
        else:
            ttk.Label(plot, text="Matplotlib not available").pack(pady=20)

    def _compute_fft(self, x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Hann-windowed rFFT magnitude (linear units).
        Returns (freqs, mag). No unit enforcement (relative spectrum).
        """
        n = len(x)
        if n < 2:
            return np.array([0.0]), np.array([0.0])
        w = np.hanning(n)
        xw = (x - np.nanmean(x)) * w
        X = np.fft.rfft(xw, n=n)
        freqs = np.fft.rfftfreq(n, d=1.0/fs)
        # normalize by window RMS to keep relative scale reasonable
        w_norm = np.sqrt(np.mean(w**2))
        mag = np.abs(X) / (n * (w_norm if w_norm > 0 else 1.0))
        mag *= 2.0  # single-sided
        return freqs, mag

    def _sync_y_by_type(self, axes: List[Any], kinds: List[str]):
        """
        Join y-axis groups so interactive y-zoom stays synchronized within ACC and within EMG.
        """
        acc_axes = [ax for ax, k in zip(axes, kinds) if k == "ACC"]
        emg_axes = [ax for ax, k in zip(axes, kinds) if k == "EMG"]
        if len(acc_axes) > 1:
            root = acc_axes[0]
            for ax in acc_axes[1:]:
                root.get_shared_y_axes().join(root, ax)
        if len(emg_axes) > 1:
            root = emg_axes[0]
            for ax in emg_axes[1:]:
                root.get_shared_y_axes().join(root, ax)

    def _safe_minmax(self, f: np.ndarray, m: np.ndarray, fmin: float, fmax: float) -> Tuple[float, float]:
        """Return finite (ymin, ymax), using fmin/fmax mask if available; else full-band."""
        mask = (f >= fmin) & (f <= fmax)
        mm = m[mask] if mask.any() else m
        mm = mm[np.isfinite(mm)]
        if mm.size == 0:
            return 0.0, 1.0
        lo, hi = float(np.nanmin(mm)), float(np.nanmax(mm))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return 0.0, max(1.0, abs(lo) + abs(hi) + 1.0)
        return lo, hi

    def redraw(self) -> None:
        if not MATPLOTLIB_OK: return
        try:
            if self.app_state.raw_data is None or not self.app_state.fs or self.app_state.fs <= 0:
                messagebox.showinfo("Info", "Load data and set Fs in Setup; then Save.")
                return

            use_processed = (self.viz.var_signal.get() == "Processed")
            Yv, meta = _get_visualization_order_matrix(self.app_state, use_processed=use_processed)
            if Yv.shape[1] == 0:
                messagebox.showinfo("No channels", "Enable channels in Setup."); return
            Yv, meta = _limit_to_6(Yv, meta)

            fs = self.app_state.fs
            n = Yv.shape[0]
            i0, i1 = _get_selection_indices(self.app_state, n)
            seg = Yv[i0:i1, :]

            self.fig.clf()
            rows, cols = 3, 2
            # sharex=True → interactive x-zoom/pan sync across all subplots
            axes = self.fig.subplots(rows, cols, squeeze=False, sharex=True)
            flat_axes: List[Any] = []
            xlims = (2.0, 20.0)  # default visible window

            # Collect per-type y-lims
            acc_ylims = []
            emg_ylims = []

            k = seg.shape[1]
            for i in range(rows * cols):
                r, c = divmod(i, cols)
                ax = axes[r, c]
                if i < k:
                    y = seg[:, i]
                    f, m = self._compute_fft(y, fs)
                    ax.plot(f, m, linewidth=0.9, color=("green" if meta.kinds[i] == "ACC" else None))
                    ax.set_title(f"{meta.sides[i]} — {meta.kinds[i]}")
                    if r == rows - 1: ax.set_xlabel("Frequency (Hz)")
                    if c == 0: ax.set_ylabel("Magnitude (a.u.)")
                    ax.grid(True, alpha=0.3)
                    # If Fs/2 < 20 Hz you can still pan/zoom; we just start at 2–20
                    ax.set_xlim(*xlims)

                    # record candidate y-lims per type (robust to empty 2–20 Hz slice)
                    ymin, ymax = self._safe_minmax(f, m, *xlims)
                    if meta.kinds[i] == "ACC":
                        acc_ylims.append((ymin, ymax))
                    else:
                        emg_ylims.append((ymin, ymax))
                else:
                    ax.axis("off")
                flat_axes.append(ax)

            # Initial y equalization by type (use median across panels to avoid outlier panels)
            def apply_group_ylims(group_axes: List[Any], ylims: List[Tuple[float, float]]):
                if not group_axes or not ylims:
                    return
                lows = np.array([a for a, _ in ylims])
                highs = np.array([b for _, b in ylims])
                ymin = float(np.nanmedian(lows))
                ymax = float(np.nanmedian(highs))
                if not (np.isfinite(ymin) and np.isfinite(ymax)) or ymax <= ymin:
                    ymin, ymax = 0.0, 1.0
                for ax in group_axes:
                    try:
                        ax.set_ylim(ymin, ymax)
                    except Exception:
                        pass

            # Map axes to kinds (only the first k are “real”)
            real_axes = [ax for i, ax in enumerate(flat_axes) if i < k]
            acc_axes = [ax for i, ax in enumerate(real_axes) if meta.kinds[i] == "ACC"]
            emg_axes = [ax for i, ax in enumerate(real_axes) if meta.kinds[i] == "EMG"]

            apply_group_ylims(acc_axes, acc_ylims)
            apply_group_ylims(emg_axes, emg_ylims)

            # Persist y-sharing interactivity
            self._sync_y_by_type(real_axes, meta.kinds[:k])

            self.fig.tight_layout()
            self.canvas.draw_idle()
            self._axes = real_axes
            self._log_cb("[FFT] Updated (robust y-sync).")

        except Exception as e:
            tb = "".join(traceback.format_exception_only(type(e), e)).strip()
            self._log_cb(f"[ERROR][FFT] {tb}")

# ==========================
# Coherence tab (pairwise)
# ==========================
class CoherenceTab(ttk.Frame):
    """
    Magnitude-squared coherence between two selectable channels.
    - Welch with nperseg=pow2(round(win*Fs)), noverlap=nperseg/2.
    - 95% significance line (Halliday threshold).
    - 95% confidence interval ribbon:
      Jackknife on coherency magnitude using Fisher z, then square endpoints to get MSC CI.
    """

    def __init__(self, parent, app_state: AppState, viz: VisualizationTab, log_cb):
        super().__init__(parent)
        self.app_state = app_state; self.viz = viz; self._log_cb = log_cb

        self.var_win = tk.StringVar(value="1.0")
        self.var_maxlag = tk.StringVar(value="100")
        self.var_ch1 = tk.StringVar(value="")
        self.var_ch2 = tk.StringVar(value="")

        self.fig = None
        self.canvas = None

        self._labels: List[str] = []
        self._Y: Optional[np.ndarray] = None
        self._kinds: List[str] = []
        self._sides: List[str] = []

        self._build()

    def log(self, msg: str) -> None:
        self._log_cb(msg); _log_file(msg)

    def _build(self) -> None:
        ctrl = ttk.Frame(self); ctrl.pack(fill="x", padx=10, pady=6)

        ttk.Label(ctrl, text="Window (s):").pack(side="left")
        ttk.Entry(ctrl, textvariable=self.var_win, width=8).pack(side="left", padx=(4, 10))
        ttk.Label(ctrl, text="Max lag (ms):").pack(side="left")
        ttk.Entry(ctrl, textvariable=self.var_maxlag, width=8).pack(side="left", padx=(4, 10))
        ttk.Label(ctrl, text="Channel 1:").pack(side="left")
        self.cb1 = ttk.Combobox(ctrl, textvariable=self.var_ch1, values=[], state="readonly", width=28)
        self.cb1.pack(side="left", padx=4)
        ttk.Label(ctrl, text="Channel 2:").pack(side="left", padx=(10, 0))
        self.cb2 = ttk.Combobox(ctrl, textvariable=self.var_ch2, values=[], state="readonly", width=28)
        self.cb2.pack(side="left", padx=4)
        ttk.Button(ctrl, text="Refresh", command=self.redraw).pack(side="left", padx=10)

        plot = ttk.Frame(self); plot.pack(fill="both", expand=True, padx=8, pady=6)
        if MATPLOTLIB_OK:
            self.fig = plt.Figure(figsize=(12, 4.5), dpi=100)
            self.canvas = FigureCanvasTkAgg(self.fig, master=plot)
            self.canvas.get_tk_widget().pack(fill="both", expand=True)
            NavigationToolbar2Tk(self.canvas, plot).update()
        else:
            ttk.Label(plot, text="Matplotlib not available").pack(pady=20)

        # Prime channel list
        self._prepare_matrix_and_labels()

    def _prepare_matrix_and_labels(self) -> None:
        if self.app_state.raw_data is None or not self.app_state.fs or self.app_state.fs <= 0:
            self._labels = []; self._Y = None; self.cb1['values']=[]; self.cb2['values']=[]
            return
        use_processed = (self.viz.var_signal.get() == "Processed")
        Yv, meta = _get_visualization_order_matrix(self.app_state, use_processed=use_processed)
        if Yv.shape[1] == 0:
            self._labels = []; self._Y = None; self.cb1['values']=[]; self.cb2['values']=[]
            return
        # Note: unlike the grid-based tabs (which need a fixed 2xN layout and
        # therefore cap at 6 channels via _limit_to_6), this tab only fills two
        # dropdowns, so all enabled channels stay selectable here (no cap) —
        # otherwise ACC channels could be silently excluded when more than 6
        # channels are enabled with EMG channels ordered first.
        self._Y = Yv
        self._kinds = meta.kinds
        self._sides = meta.sides
        labels = meta.labels
        self._labels = labels
        self.cb1['values'] = labels
        self.cb2['values'] = labels

        # Default: Left ACC vs Right ACC if available
        def find_idx(k: str, s: str) -> Optional[int]:
            for i, (ki, si) in enumerate(zip(meta.kinds, meta.sides)):
                if ki == k and si == s:
                    return i
            return None

        i1 = find_idx("ACC", "Left")
        i2 = find_idx("ACC", "Right")
        if i1 is None or i2 is None or i1 == i2:
            # fallback: first two channels
            i1, i2 = 0, min(1, len(labels)-1)
        self.var_ch1.set(labels[i1]); self.var_ch2.set(labels[i2])

    @staticmethod
    def _halliday_threshold(neff: int, alpha: float = 0.05) -> float:
        """
        Halliday significance level for magnitude-squared coherence:
        C_sig = 1 - (alpha)^(1/(Neff - 1)).
        With Welch averaging, degrees of freedom ≈ 2 * (#segments) → Neff here.
        """
        if neff is None or neff <= 1:
            return 1.0
        try:
            return 1.0 - (alpha ** (1.0 / (neff - 1.0)))
        except Exception:
            return 1.0

    def _segment_fft(self, x: np.ndarray, fs: float, nperseg: int, noverlap: int):
        """Yield windowed FFTs (rFFT) of overlapping segments."""
        step = max(1, nperseg - noverlap)
        w = np.hanning(nperseg)
        idx = 0
        while idx + nperseg <= len(x):
            seg = x[idx:idx+nperseg]
            seg = (seg - np.mean(seg)) * w
            X = np.fft.rfft(seg)
            yield X
            idx += step

    def _coherence_with_jackknife(self, x: np.ndarray, y: np.ndarray, fs: float, win_s: float):
        """
        Compute:
          - f: frequency axis
          - Cxy: Welch MSC
          - Csig: 95% significance threshold (Halliday)
          - (Clow, Cup): 95% confidence interval via jackknife on coherency magnitude
        """
        n = len(x)
        nperseg = max(16, int(round(win_s * fs))); nperseg = _next_pow2(nperseg); nperseg = min(nperseg, n)
        noverlap = nperseg // 2
        step = max(1, nperseg - noverlap)
        if nperseg < 8: nperseg = min(8, n)

        # Collect per-segment spectra
        X_list=[]; Y_list=[]
        for Xk, Yk in zip(self._segment_fft(x, fs, nperseg, noverlap),
                          self._segment_fft(y, fs, nperseg, noverlap)):
            X_list.append(Xk); Y_list.append(Yk)
        K = len(X_list)
        f = np.fft.rfftfreq(nperseg, d=1/fs)
        if K == 0:
            return f, np.zeros_like(f), 1.0, np.zeros_like(f), np.zeros_like(f)

        X_arr = np.vstack(X_list)     # [K, F]
        Y_arr = np.vstack(Y_list)     # [K, F]

        # Autospectra and cross-spectra per segment (Hann-normalized density)
        w2 = np.mean(np.hanning(nperseg)**2)
        Sxx_k = (np.abs(X_arr)**2) / (w2 * fs)
        Syy_k = (np.abs(Y_arr)**2) / (w2 * fs)
        Sxy_k = (X_arr * np.conj(Y_arr)) / (w2 * fs)

        # Welch averages
        Sxx = np.mean(Sxx_k, axis=0)
        Syy = np.mean(Syy_k, axis=0)
        Sxy = np.mean(Sxy_k, axis=0)
        coherency = Sxy / np.sqrt(Sxx * Syy)          # complex coherency γ(f)
        Cxy = np.clip(np.abs(coherency)**2, 0, 1)     # MSC

        # Halliday significance threshold (Neff ≈ 2*K)
        neff = max(2, 2 * K)
        Csig = self._halliday_threshold(neff, alpha=0.05)

        # ---- Jackknife CI on |coherency| (then square to MSC CI) ----
        abs_gamma = np.clip(np.abs(coherency), 1e-8, 1-1e-8)  # |γ|
        if K > 1:
            jk_vals = np.empty((K, abs_gamma.size))
            Sxx_sum = np.sum(Sxx_k, axis=0)
            Syy_sum = np.sum(Syy_k, axis=0)
            Sxy_sum = np.sum(Sxy_k, axis=0)
            for k in range(K):
                Sxx_lo = (Sxx_sum - Sxx_k[k]) / (K - 1)
                Syy_lo = (Syy_sum - Syy_k[k]) / (K - 1)
                Sxy_lo = (Sxy_sum - Sxy_k[k]) / (K - 1)
                coh_lo = Sxy_lo / np.sqrt(Sxx_lo * Syy_lo)
                jk_vals[k, :] = np.clip(np.abs(coh_lo), 1e-8, 1-1e-8)
            # Fisher z of |γ|
            z_full = np.arctanh(abs_gamma)
            z_jk = np.arctanh(jk_vals)
            z_bar = np.mean(z_jk, axis=0)
            se = np.sqrt((K - 1) * np.mean((z_jk - z_bar)**2, axis=0))
            tcrit = t_dist.ppf(0.975, df=K-1) if HAS_SCIPY else 1.96
            z_lo = z_full - tcrit * se
            z_hi = z_full + tcrit * se
            g_lo = np.tanh(z_lo); g_hi = np.tanh(z_hi)
            Clow = np.clip(g_lo**2, 0.0, 1.0)
            Cup  = np.clip(g_hi**2, 0.0, 1.0)
        else:
            Clow = np.zeros_like(Cxy); Cup = np.ones_like(Cxy)

        return f, Cxy, Csig, Clow, Cup

    def _cumulant_density(self, x: np.ndarray, y: np.ndarray, fs: float, win_s: float,
                           max_lag_ms: float = 100.0):
        """
        Time-domain cumulant density q_hat_xy(u) (Halliday, Rosenberg, Conway,
        Farmer & Rosenberg 1995b "A framework for the analysis of mixed time
        series/point process data"; Rosenberg et al. 1989). This is the
        Fourier-pair counterpart of coherence: the same segments are used, but
        instead of comparing spectra, the (biased) cross-covariance of each
        segment is computed and averaged across segments. A peak away from
        lag 0 shows the time delay (lag) at which the two signals line up
        best; a peak at/near lag 0 shows near-simultaneous coupling.

        Segments here are NOT tapered (only mean-removed): a Hann/other taper
        would multiply the cross-covariance by the window's own
        autocorrelation and distort its shape and lag position, which is not
        an issue for coherence (a frequency-domain quantity) but matters here.

        Returns:
          lags_ms : lag axis (ms), positive lag = x leads y, restricted to
                    +/- max_lag_ms
          q       : cumulant density estimate at each lag
          q_ci    : scalar 95% confidence limit (+/-) under the assumption
                    of independence (Halliday et al. 1995b) — constant across
                    lag, matching the horizontal confidence lines in Halliday's
                    figures
          K       : number of segments averaged
        """
        n = len(x)
        nperseg = max(16, int(round(win_s * fs))); nperseg = _next_pow2(nperseg); nperseg = min(nperseg, n)
        # Disjoint (non-overlapping) segments: the variance/CI formula below
        # assumes independent segments, as in Halliday's original framework.
        # Coherence (a separate estimate) uses 50%-overlapping Welch segments
        # for a smoother spectrum — fine there, but it would violate the
        # independence assumption a confidence interval on q_hat(u) relies on.
        step = nperseg
        if nperseg < 8:
            nperseg = min(8, n); step = max(1, nperseg)

        cc_list: List[np.ndarray] = []
        Pxx_list: List[np.ndarray] = []
        Pyy_list: List[np.ndarray] = []
        idx = 0
        while idx + nperseg <= n:
            xs = x[idx:idx + nperseg]; xs = xs - np.mean(xs)
            ys = y[idx:idx + nperseg]; ys = ys - np.mean(ys)
            Xf = np.fft.fft(xs)
            Yf = np.fft.fft(ys)
            # Biased circular cross-covariance for this segment
            cc = np.fft.ifft(Xf * np.conj(Yf)).real / nperseg
            cc_list.append(cc)
            Pxx_list.append(np.abs(Xf) ** 2)
            Pyy_list.append(np.abs(Yf) ** 2)
            idx += step

        K = len(cc_list)
        if K == 0:
            return np.array([0.0]), np.array([0.0]), 0.0, 0

        q_full = np.mean(np.vstack(cc_list), axis=0)
        Pxx_avg = np.mean(np.vstack(Pxx_list), axis=0)
        Pyy_avg = np.mean(np.vstack(Pyy_list), axis=0)

        # Var{q_hat(u)} = (1/(K*N^4)) * sum_f Pxx(f)*Pyy(f), constant across
        # lag u (Bartlett-type formula for cross-covariance under
        # independence, averaged over K segments — see Halliday et al. 1995b).
        # Verified numerically against the known Var ~= var_x*var_y/N result
        # for a single segment of white noise.
        var_q = float(np.sum(Pxx_avg * Pyy_avg) / (K * (nperseg ** 4)))
        tcrit = t_dist.ppf(0.975, df=K - 1) if (HAS_SCIPY and K > 1) else 1.96
        q_ci = float(tcrit * math.sqrt(max(var_q, 0.0)))

        # Re-center circular lags 0..N-1 onto -N/2..N/2-1, then flip sign so
        # that "lag" follows the conventional reading "positive = x (Ch.1)
        # leads y (Ch.2)": q_full[u] = mean_n x(n)*y(n-u) is maximal at
        # u = -delay when y is a delayed copy of x, so displayed_lag = -u
        # puts that peak at +delay.
        lag_idx = np.arange(nperseg)
        lag_centered = ((lag_idx + nperseg // 2) % nperseg) - nperseg // 2
        disp_lag = -lag_centered
        order = np.argsort(disp_lag)
        lags = disp_lag[order]
        q = q_full[order]

        lags_ms = lags / fs * 1000.0
        mask = np.abs(lags_ms) <= max_lag_ms
        if not np.any(mask):
            mask = np.ones_like(lags_ms, dtype=bool)
        return lags_ms[mask], q[mask], q_ci, K

    def redraw(self) -> None:
        if not MATPLOTLIB_OK:
            return
        if self.app_state.raw_data is None or not self.app_state.fs or self.app_state.fs <= 0:
            messagebox.showinfo("Info", "Load data and set Fs in Setup; then Save.")
            return

        if self._Y is None:
            self._prepare_matrix_and_labels()
            if self._Y is None:
                messagebox.showinfo("No channels", "Enable channels in Setup.")
                return

        fs = self.app_state.fs
        Yv = self._Y

        # Resolve channel indices from combobox labels
        lab1 = self.var_ch1.get(); lab2 = self.var_ch2.get()
        if lab1 not in self._labels or lab2 not in self._labels:
            self._prepare_matrix_and_labels()
            if self._labels:
                lab1 = self._labels[0]
                lab2 = self._labels[min(1, len(self._labels)-1)]
                self.var_ch1.set(lab1); self.var_ch2.set(lab2)
            else:
                messagebox.showinfo("No channels", "Enable channels in Setup.")
                return
        i1 = self._labels.index(lab1)
        i2 = self._labels.index(lab2)
        if i1 == i2:
            messagebox.showwarning("Coherence/Cumulant", "Choose two different channels.")
            return

        # Selection slice
        n = Yv.shape[0]
        i0, i1s = _get_selection_indices(self.app_state, n)
        seg1 = Yv[i0:i1s, i1]
        seg2 = Yv[i0:i1s, i2]

        # Window in seconds
        try:
            win_s = float(self.var_win.get())
            if win_s <= 0:
                win_s = 1.0
        except Exception:
            win_s = 1.0
            self.var_win.set("1.0")

        # Max lag (ms) for the cumulant density panel
        try:
            max_lag_ms = float(self.var_maxlag.get())
            if max_lag_ms <= 0:
                max_lag_ms = 100.0
        except Exception:
            max_lag_ms = 100.0
            self.var_maxlag.set("100")

        # Frequency domain: coherence (Halliday significance + jackknife CI)
        f, Cxy, Csig, Clow, Cup = self._coherence_with_jackknife(seg1, seg2, fs, win_s)
        # Time domain: cumulant density (Halliday et al. 1995b) — computed
        # every time this tab redraws, alongside coherence, using the same
        # window length and channel pair.
        lags_ms, q, q_ci, Kc = self._cumulant_density(seg1, seg2, fs, win_s, max_lag_ms=max_lag_ms)

        self.fig.clf()
        ax1, ax2 = self.fig.subplots(1, 2)

        # --- Left: Coherence (frequency domain) ---
        ax1.plot(f, Cxy, linewidth=1.0, label="Coherence")
        ax1.fill_between(f, Clow, Cup, alpha=0.25, label="95% CI")
        ax1.axhline(Csig, linestyle="--", linewidth=1.0, color="red", alpha=0.7,
                    label=f"95% sig ≈ {Csig:.2f}")
        ax1.set_xlim(2, 20); ax1.set_ylim(0, 1.0)
        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_ylabel("Coherence")
        ax1.set_title(f"Coherence: {lab1} vs {lab2} (win={win_s:.2f}s)")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="upper right", fontsize=8)

        # --- Right: Cumulant density (time domain) ---
        ax2.plot(lags_ms, q, linewidth=1.0, color="tab:purple", label="Cumulant density")
        if q.size > 0:
            ax2.axhline(q_ci, linestyle="-", linewidth=1.0, color="red", alpha=0.7,
                        label="95% limit (independence)")
            ax2.axhline(-q_ci, linestyle="-", linewidth=1.0, color="red", alpha=0.7)
        ax2.axvline(0.0, linestyle=":", linewidth=1.0, color="gray", alpha=0.8)
        # Mark the lag of the largest-magnitude peak (best-aligning time shift)
        if q.size > 0:
            pk = int(np.argmax(np.abs(q)))
            ax2.plot(lags_ms[pk], q[pk], marker="o", markersize=4, color="tab:purple")
            ax2.annotate(f"{lags_ms[pk]:.1f} ms",
                         xy=(lags_ms[pk], q[pk]), xycoords="data",
                         xytext=(0, 8 if q[pk] >= 0 else -8),
                         textcoords="offset points",
                         ha="center", va="bottom" if q[pk] >= 0 else "top",
                         fontsize=7.5, color="tab:purple", fontweight="bold")
        ax2.set_xlim(-max_lag_ms, max_lag_ms)
        ax2.set_xlabel("Lag (ms)  [+ : Ch.1 leads Ch.2]")
        ax2.set_ylabel("Cumulant density")
        ax2.set_title(f"Cumulant density: {lab1} vs {lab2} (K={Kc} segs)")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="upper right", fontsize=8)

        self.fig.suptitle(f"Coherence / Cumulant density — {lab1} vs {lab2}", fontsize=10)
        self.fig.tight_layout(rect=(0, 0, 1, 0.95))
        self.canvas.draw_idle()
        self._log_cb("[Coherence/Cumulant] Updated (coherence CI/significance + cumulant density CI).")

# ==========================
# Spectrogram tab (STFT + Wavelet toggle)
# ==========================
class SpectrogramTab(ttk.Frame):
    """
    Unified time-frequency tab with two modes selectable via a toggle button:

    STFT mode (default)
    -------------------
    • Welch/STFT spectrogram, ~1 s window, 50% overlap, dB scale.
    • Uniform frequency resolution (Hz/bin constant).
    • Y-axis 2–20 Hz, linear scale.

    Wavelet mode
    ------------
    • CWT with complex Morlet (ω₀ = 6), FFT-based convolution.
    • Log-spaced frequencies 2–20 Hz, ≤ 0.5 Hz bin spacing at 2 Hz.
    • dB relative to per-channel peak, linear Y-axis.
    • Better for detecting transient or frequency-modulated tremor.

    Both modes use the Selection segment if active, else the full record.
    """

    # ---- Wavelet constants ----
    _OMEGA0: float = 10.0   # increased from 6→10 for better freq resolution
    _F_MIN:  float = 2.0
    _F_MAX:  float = 20.0
    _DF_MAX: float = 0.5

    def __init__(self, parent, app_state: AppState, viz: VisualizationTab, log_cb):
        super().__init__(parent)
        self.app_state = app_state
        self.viz = viz
        self._log_cb = log_cb
        self.fig = None
        self.canvas = None
        # Mode: "stft" (default) or "wavelet"
        self._mode: str = "stft"
        self.var_win = tk.StringVar(value="1.0")
        self._build()

    def log(self, msg: str) -> None:
        self._log_cb(msg); _log_file(msg)

    def _build(self) -> None:
        ctrl = ttk.Frame(self); ctrl.pack(fill="x", padx=10, pady=6)

        ttk.Button(ctrl, text="Refresh", command=self.redraw).pack(side="left", padx=5)

        # Mode toggle button
        self._btn_mode = ttk.Button(ctrl, text="Switch to Wavelet", command=self._toggle_mode)
        self._btn_mode.pack(side="left", padx=(4, 12))

        # STFT window entry
        ttk.Label(ctrl, text="Window (s):").pack(side="left")
        self._win_entry = ttk.Entry(ctrl, textvariable=self.var_win, width=6)
        self._win_entry.pack(side="left", padx=(2, 8))

        # Wavelet ω₀ slider (only used in wavelet mode)
        ttk.Separator(ctrl, orient="vertical").pack(side="left", fill="y", padx=6, pady=3)
        ttk.Label(ctrl, text="ω₀:").pack(side="left")
        self._var_omega0 = tk.DoubleVar(value=self._OMEGA0)
        self._omega0_scale = ttk.Scale(ctrl, from_=4, to=20,
                                        variable=self._var_omega0,
                                        orient="horizontal", length=100)
        self._omega0_scale.pack(side="left", padx=2)
        self._omega0_label = ttk.Label(ctrl, text=f"{self._OMEGA0:.0f}", width=3)
        self._omega0_label.pack(side="left", padx=(0, 4))
        self._var_omega0.trace_add("write",
            lambda *_: self._omega0_label.config(
                text=f"{self._var_omega0.get():.0f}"))
        ttk.Label(ctrl, text="← freq. resolution  (wavelet only)",
                  foreground="gray").pack(side="left", padx=2)

        self._info_label = ttk.Label(ctrl, text=self._mode_info())
        self._info_label.pack(side="left", padx=4)

        plot = ttk.Frame(self); plot.pack(fill="both", expand=True, padx=8, pady=6)
        if MATPLOTLIB_OK:
            self.fig = plt.Figure(figsize=(10, 8), dpi=100)
            self.canvas = FigureCanvasTkAgg(self.fig, master=plot)
            self.canvas.get_tk_widget().pack(fill="both", expand=True)
            NavigationToolbar2Tk(self.canvas, plot).update()
        else:
            ttk.Label(plot, text="Matplotlib not available").pack(pady=20)

    def _mode_info(self) -> str:
        if self._mode == "stft":
            return "STFT • dB • 2–20 Hz • linear Y"
        w0 = float(getattr(self, '_var_omega0', None) and
                   self._var_omega0.get() or self._OMEGA0)
        return f"CWT Morlet ω₀={w0:.0f} • dB (rel. peak) • 2–20 Hz"

    def _toggle_mode(self) -> None:
        self._mode = "wavelet" if self._mode == "stft" else "stft"
        if self._mode == "wavelet":
            self._btn_mode.config(text="Switch to STFT")
            self._win_entry.config(state="disabled")
        else:
            self._btn_mode.config(text="Switch to Wavelet")
            self._win_entry.config(state="normal")
        self._info_label.config(text=self._mode_info())
        self.redraw()

    # ------------------------------------------------------------------
    # Common Y-axis helper
    # ------------------------------------------------------------------
    def _style_freq_ax(self, ax, r: int, rows: int, is_last_col: bool = False) -> None:
        """Apply uniform Y-axis style (linear, 2–20 Hz, ticks every 2 Hz)."""
        ax.set_ylim(self._F_MIN, self._F_MAX)
        ax.set_yticks(np.arange(2, 21, 2))
        ax.set_ylabel("Freq (Hz)")
        if r == rows - 1:
            ax.set_xlabel("Time (s)")

    # ------------------------------------------------------------------
    # STFT helpers
    # ------------------------------------------------------------------
    def _do_stft(self, x: np.ndarray, fs: float, win_s: float
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (t, f, Sxx_dB) for the STFT spectrogram."""
        nperseg = max(16, int(round(win_s * fs)))
        nperseg = _next_pow2(nperseg)
        nperseg = min(nperseg, len(x)) if len(x) > 0 else nperseg
        noverlap = nperseg // 2
        if HAS_SCIPY:
            f, t, Sxx = sps.spectrogram(
                x, fs=fs, nperseg=nperseg, noverlap=noverlap,
                scaling="density", detrend="constant"
            )
        else:
            step = max(1, nperseg - noverlap)
            nwin = 1 + max(0, (len(x) - nperseg) // step)
            t = np.arange(nwin) * (step / fs)
            f = np.fft.rfftfreq(nperseg, d=1.0 / fs)
            Sxx = np.zeros((len(f), nwin))
            w = np.hanning(nperseg)
            for i in range(nwin):
                seg = x[i*step: i*step + nperseg]
                if len(seg) < nperseg:
                    seg = np.pad(seg, (0, nperseg - len(seg)))
                X = np.fft.rfft((seg - np.mean(seg)) * w)
                Sxx[:, i] = (np.abs(X) ** 2) / (np.sum(w ** 2) * fs)
        Sxx_dB = 10.0 * np.log10(np.maximum(Sxx, np.finfo(float).eps))
        return t, f, Sxx_dB

    # ------------------------------------------------------------------
    # Wavelet helpers
    # ------------------------------------------------------------------
    def _build_freq_axis(self, fs: float) -> np.ndarray:
        f_top = min(self._F_MAX, fs / 2.0 - 0.5)
        if f_top <= self._F_MIN:
            return np.array([self._F_MIN])
        n_voices = max(int(math.ceil(self._F_MIN * math.log(2.0) / self._DF_MAX)), 8)
        n_octaves = math.log2(f_top / self._F_MIN)
        n_freqs = max(int(round(n_octaves * n_voices)), 2)
        return np.logspace(math.log10(self._F_MIN), math.log10(f_top), num=n_freqs)

    def _morlet_cwt_fft(self, x: np.ndarray, fs: float, freqs: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float); x = x - x.mean()
        N = len(x)
        Nfft = _next_pow2(N)
        Xf = np.fft.fft(x, n=Nfft)
        omega = 2.0 * math.pi * np.fft.fftfreq(Nfft, d=1.0 / fs)
        W = np.empty((len(freqs), N), dtype=complex)
        # Read ω₀ from slider if available, else use class default
        w0 = float(getattr(self, '_var_omega0', None) and
                   self._var_omega0.get() or self._OMEGA0)
        if w0 < 1.0: w0 = self._OMEGA0
        for k, f in enumerate(freqs):
            s = w0 / (2.0 * math.pi * f)
            arg = s * omega - w0
            Psi = (math.pi ** -0.25) * np.exp(-0.5 * arg * arg)
            Psi[omega < 0] = 0.0
            norm = math.sqrt(2.0 * math.pi * s / (1.0 / fs))
            W[k, :] = np.fft.ifft(Xf * Psi * norm)[:N]
        return W

    def _cwt_to_db(self, W: np.ndarray) -> np.ndarray:
        amp = np.abs(W)
        peak = amp.max()
        if peak == 0.0:
            return np.zeros_like(amp)
        return 20.0 * np.log10(np.maximum(amp, peak * 1e-6) / peak)

    # ------------------------------------------------------------------
    # Redraw dispatcher
    # ------------------------------------------------------------------
    def redraw(self) -> None:
        if not MATPLOTLIB_OK:
            return
        if self.app_state.raw_data is None or not self.app_state.fs or self.app_state.fs <= 0:
            messagebox.showinfo("Info", "Load data and set Fs in Setup; then Save.")
            return
        try:
            if self._mode == "stft":
                self._draw_stft()
            else:
                self._draw_wavelet()
        except Exception as e:
            tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            self.log(f"[Spectrogram ERROR]\n{tb}")
            messagebox.showerror("Spectrogram error", str(e))

    # ------------------------------------------------------------------
    # STFT draw
    # ------------------------------------------------------------------
    def _draw_stft(self) -> None:
        try:
            win_s = float(self.var_win.get())
            if win_s <= 0: win_s = 1.0
        except Exception:
            win_s = 1.0; self.var_win.set("1.0")

        fs = self.app_state.fs
        use_processed = (self.viz.var_signal.get() == "Processed")
        Yv, meta = _get_visualization_order_matrix(self.app_state, use_processed=use_processed)
        if Yv.shape[1] == 0:
            messagebox.showinfo("No channels", "Enable channels in Setup."); return
        Yv, meta = _limit_to_6(Yv, meta)

        n = Yv.shape[0]
        i0, i1 = _get_selection_indices(self.app_state, n)
        seg = Yv[i0:i1, :]

        self.fig.clf()
        rows, cols = 3, 2
        axes = self.fig.subplots(rows, cols, squeeze=False, sharex=False, sharey=False)

        for idx in range(rows * cols):
            r, c = divmod(idx, cols)
            ax = axes[r, c]
            if idx >= seg.shape[1]:
                ax.axis("off"); continue

            x = seg[:, idx]
            t, f, Sxx_dB = self._do_stft(x, fs, win_s)
            fmask = (f >= self._F_MIN) & (f <= self._F_MAX)
            im = ax.pcolormesh(t, f[fmask], Sxx_dB[fmask, :],
                               shading="auto", rasterized=True)
            ax.set_title(f"{meta.sides[idx]} — {meta.names[idx]}")
            self._style_freq_ax(ax, r, rows)
            ax.grid(False)
            self.fig.colorbar(im, ax=ax, pad=0.01, fraction=0.046, label="dB")

        self.fig.tight_layout()
        self.canvas.draw_idle()
        self.log(f"[Spectrogram/STFT] Updated  (window={win_s:.2f} s).")

    # ------------------------------------------------------------------
    # Wavelet draw
    # ------------------------------------------------------------------
    def _draw_wavelet(self) -> None:
        fs = self.app_state.fs
        use_processed = (self.viz.var_signal.get() == "Processed")
        Yv, meta = _get_visualization_order_matrix(self.app_state, use_processed=use_processed)
        if Yv.shape[1] == 0:
            messagebox.showinfo("No channels", "Enable channels in Setup."); return
        Yv, meta = _limit_to_6(Yv, meta)

        n = Yv.shape[0]
        i0, i1 = _get_selection_indices(self.app_state, n)
        seg = Yv[i0:i1, :]
        n_seg = seg.shape[0]

        if n_seg < 4:
            messagebox.showwarning("Wavelet", "Segment too short (< 4 samples)."); return

        freqs = self._build_freq_axis(fs)
        t_axis = np.arange(n_seg) / fs
        n_ch = seg.shape[1]

        self.log(f"[Spectrogram/Wavelet] CWT: {n_ch} ch, {n_seg} samples, "
                 f"{len(freqs)} freq bins ({freqs[0]:.2f}–{freqs[-1]:.2f} Hz)…")
        self.update_idletasks()

        self.fig.clf()
        rows, cols = 3, 2
        axes = self.fig.subplots(rows, cols, squeeze=False)

        for idx in range(rows * cols):
            r, c = divmod(idx, cols)
            ax = axes[r, c]
            if idx >= n_ch:
                ax.axis("off"); continue

            x = seg[:, idx].copy()
            finite = np.isfinite(x)
            if not finite.any():
                ax.text(0.5, 0.5, "All NaN/Inf", ha="center", va="center",
                        transform=ax.transAxes); continue
            if not finite.all():
                ii = np.arange(n_seg)
                x[~finite] = np.interp(ii[~finite], ii[finite], x[finite])

            W = self._morlet_cwt_fft(x, fs, freqs)
            Wdb = self._cwt_to_db(W)

            im = ax.pcolormesh(t_axis, freqs, Wdb,
                               shading="auto", cmap="inferno",
                               vmin=-40, vmax=0, rasterized=True)
            ax.set_title(f"{meta.sides[idx]} — {meta.names[idx]}")
            self._style_freq_ax(ax, r, rows)
            ax.grid(True, alpha=0.2, linestyle="--")
            self.fig.colorbar(im, ax=ax, pad=0.01, fraction=0.046, label="dB (rel. peak)")

        self.fig.tight_layout()
        self.canvas.draw_idle()
        self.log("[Spectrogram/Wavelet] Done.")

# ─────────────────────────────────────────────────────────────────────────────
# SON64 / SMRX support — automatic sonpy provisioning
# -----------------------------------------------------------------------------
# sonpy only publishes pre-built wheels for two CPython ABIs: 3.9.x (its
# original line) and 3.14.x (its current line). Any other interpreter
# (3.10-3.13, or <3.9) cannot "pip install sonpy" directly. Instead of asking
# the user to install and juggle a second Python by hand, we resolve this
# automatically, in order:
#   1. If THIS interpreter is already 3.9.x/3.14.x, pip-install sonpy into it.
#   2. Otherwise, look for another Python already on the machine that is
#      3.9.x/3.14.x (Windows "py" launcher, PATH, common install folders)
#      and pip-install sonpy into that one.
#   3. If nothing compatible exists anywhere, download a self-contained
#      ("embeddable") Python 3.9 into a folder next to this script and set
#      sonpy up there — no admin rights, no separate installer needed.
#
# Only the small amount of *metadata* sonpy exposes (channel titles, scale,
# offset, sampling rate) is ever needed — the actual waveform samples are
# parsed directly from the file's binary blocks in THIS process (see below),
# so when a different interpreter is used we only run a tiny helper script in
# it and read back a few KB of JSON; the whole GUI never has to be relaunched
# under another Python.
# ─────────────────────────────────────────────────────────────────────────────

SONPY_PY_VERSIONS = {(3, 9), (3, 14)}   # (major, minor) with published sonpy wheels
SONPY_PINNED_VERSION = {(3, 9): "sonpy==1.9.5", (3, 14): "sonpy==1.9.11"}
_SONPY_RUNTIME_DIR = os.path.join(_script_dir(), "_runtime_py39")
_SONPY_CACHE_FILE = os.path.join(_script_dir(), ".sonpy_python.txt")
_SONPY_EMBED_URL = "https://www.python.org/ftp/python/3.9.13/python-3.9.13-embed-amd64.zip"
_GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"


def _python_version_tuple(python_exe: str) -> Optional[Tuple[int, int]]:
    """Query (major, minor) of a Python interpreter without importing it here."""
    try:
        r = subprocess.run(
            [python_exe, "-c", "import sys; print(sys.version_info[0], sys.version_info[1])"],
            capture_output=True, text=True, timeout=15,
        )
        if r.returncode == 0:
            parts = r.stdout.split()
            return int(parts[0]), int(parts[1])
    except Exception:
        pass
    return None


def _python_has_sonpy(python_exe: str) -> bool:
    try:
        r = subprocess.run([python_exe, "-c", "import sonpy"],
                            capture_output=True, text=True, timeout=20)
        return r.returncode == 0
    except Exception:
        return False


def _pip_install(python_exe: str, packages: List[str], timeout: int = 240) -> Tuple[bool, str]:
    try:
        r = subprocess.run(
            [python_exe, "-m", "pip", "install", "--quiet",
             "--disable-pip-version-check", *packages],
            capture_output=True, text=True, timeout=timeout,
        )
        return r.returncode == 0, (r.stdout + r.stderr)
    except Exception as e:
        return False, str(e)


def _pip_install_sonpy(python_exe: str, ver: Tuple[int, int]) -> Tuple[bool, str]:
    """Try a plain 'pip install sonpy' first; fall back to a pinned version
    known to work for this Python line if PyPI's resolver picks something
    incompatible."""
    ok, out = _pip_install(python_exe, ["sonpy"])
    if ok and _python_has_sonpy(python_exe):
        return True, out
    pinned = SONPY_PINNED_VERSION.get(ver)
    if pinned:
        ok2, out2 = _pip_install(python_exe, [pinned])
        if ok2 and _python_has_sonpy(python_exe):
            return True, out2
        return False, out + "\n" + out2
    return False, out


def _candidate_pythons() -> List[str]:
    """Every Python interpreter we can find on this machine, current one first."""
    seen: set = set()
    out: List[str] = []

    def add(p: Optional[str]) -> None:
        if p and os.path.isfile(p) and p not in seen:
            seen.add(p)
            out.append(p)

    add(sys.executable)

    if sys.platform.startswith("win"):
        for tag in ("-3.9", "-3.14"):
            try:
                r = subprocess.run(["py", tag, "-c", "import sys; print(sys.executable)"],
                                    capture_output=True, text=True, timeout=15)
                if r.returncode == 0:
                    add(r.stdout.strip())
            except Exception:
                pass
        for base in (os.environ.get("LOCALAPPDATA", ""), "C:\\\\", "C:\\\\Program Files"):
            if not base:
                continue
            for ver_dir in ("Python39", "Python314"):
                add(os.path.join(base, "Programs", "Python", ver_dir, "python.exe"))
                add(os.path.join(base, ver_dir, "python.exe"))
    else:
        for name in ("python3.9", "python3.14"):
            add(shutil.which(name))

    # A portable interpreter bootstrapped by us on a previous run
    add(os.path.join(_SONPY_RUNTIME_DIR, "python.exe"))

    # The interpreter that worked last time, cached to skip re-scanning
    try:
        if os.path.isfile(_SONPY_CACHE_FILE):
            with open(_SONPY_CACHE_FILE, "r", encoding="utf-8") as f:
                add(f.read().strip())
    except Exception:
        pass

    return out


def _cache_sonpy_python(path: str) -> None:
    try:
        with open(_SONPY_CACHE_FILE, "w", encoding="utf-8") as f:
            f.write(path)
    except Exception:
        pass


def _reload_sonpy_in_process() -> None:
    """If sonpy just got installed into the CURRENT interpreter, import it now
    so the rest of this run can use it in-process (no subprocess needed)."""
    global HAS_SONPY, sonpy
    try:
        import importlib
        importlib.invalidate_caches()
        import sonpy  # noqa: F401
        HAS_SONPY = True
    except Exception as e:
        _log_file(f"[SMRX] sonpy installed but failed to import in-process: {e}")


def _bootstrap_portable_python(log_cb) -> Optional[str]:
    """
    Download a self-contained Python 3.9 ("embeddable" build) into a folder
    next to this script, then install pip + numpy + sonpy into it.
    Windows-only (the embeddable zip is a Windows-specific distribution
    format); on other platforms we report why automatic setup can't proceed.
    Returns the path to the new python.exe, or None on failure.
    """
    if getattr(sys, "frozen", False):
        # A packaged build should already ship a working sonpy in-process
        # (built with a 3.9.x/3.14.x interpreter) — reaching this point means
        # that bundling didn't work as expected. Silently downloading a whole
        # separate Python runtime onto a colleague's machine, from inside a
        # shipped clinical tool, is exactly the kind of surprise behavior a
        # standalone build should avoid — report it instead.
        log_cb("[SMRX] .smrx support isn't working in this packaged build "
               "(sonpy did not load). Please report this — it should have "
               "been bundled in. Meanwhile, .txt/.csv/.dat files work normally.")
        return None

    if not sys.platform.startswith("win"):
        log_cb("[SMRX] No compatible Python (3.9.x/3.14.x) found on this machine, and "
               "automatic portable-runtime setup is only implemented for Windows. "
               "Install a Python 3.9 or 3.14 and 'pip install sonpy' in it, or ask for "
               "this to be extended to your OS.")
        return None

    import zipfile
    import urllib.request

    py_exe = os.path.join(_SONPY_RUNTIME_DIR, "python.exe")
    if os.path.isfile(py_exe) and _python_has_sonpy(py_exe):
        return py_exe

    try:
        os.makedirs(_SONPY_RUNTIME_DIR, exist_ok=True)
        log_cb("[SMRX] No compatible Python found on this machine — downloading a "
               "self-contained Python 3.9 runtime (~9 MB) just for .smrx support...")
        zip_path = os.path.join(_SONPY_RUNTIME_DIR, "_python39_embed.zip")
        urllib.request.urlretrieve(_SONPY_EMBED_URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(_SONPY_RUNTIME_DIR)
        os.remove(zip_path)

        # The embeddable build ships a ._pth file that disables site-packages
        # (and therefore pip-installed packages) by default — re-enable it.
        for fn in os.listdir(_SONPY_RUNTIME_DIR):
            if fn.endswith("._pth"):
                pth_path = os.path.join(_SONPY_RUNTIME_DIR, fn)
                with open(pth_path, "r", encoding="utf-8") as f:
                    content = f.read()
                content = content.replace("#import site", "import site")
                with open(pth_path, "w", encoding="utf-8") as f:
                    f.write(content)

        log_cb("[SMRX] Installing pip into the portable runtime...")
        get_pip_path = os.path.join(_SONPY_RUNTIME_DIR, "get-pip.py")
        urllib.request.urlretrieve(_GET_PIP_URL, get_pip_path)
        r = subprocess.run([py_exe, get_pip_path, "--no-warn-script-location"],
                            capture_output=True, text=True, timeout=180)
        if r.returncode != 0:
            log_cb(f"[SMRX ERROR] get-pip failed:\n{r.stdout}\n{r.stderr}")
            return None

        log_cb("[SMRX] Installing numpy + sonpy into the portable runtime...")
        ok, out = _pip_install(py_exe, ["numpy"])
        if not ok:
            log_cb(f"[SMRX ERROR] numpy install failed:\n{out}")
            return None
        ok, out = _pip_install_sonpy(py_exe, (3, 9))
        if not ok:
            log_cb(f"[SMRX ERROR] sonpy install failed:\n{out}")
            return None

        log_cb(f"[SMRX] Portable Python ready at: {py_exe}")
        return py_exe
    except Exception as e:
        log_cb(f"[SMRX ERROR] Portable Python setup failed: {e}")
        return None


def _resolve_sonpy_python(log_cb=None) -> Optional[str]:
    """
    Return the path to a Python interpreter that has (or can get) sonpy
    installed, trying in order: current interpreter, other interpreters
    already on the machine, then a freshly bootstrapped portable one.
    Caches the winning path so future loads skip straight to it.
    """
    global HAS_SONPY
    log_cb = log_cb or _log_file

    if HAS_SONPY:
        return sys.executable

    for cand in _candidate_pythons():
        ver = _python_version_tuple(cand)
        if ver not in SONPY_PY_VERSIONS:
            continue
        if _python_has_sonpy(cand):
            _cache_sonpy_python(cand)
            if cand == sys.executable:
                _reload_sonpy_in_process()
            return cand
        log_cb(f"[SMRX] Found a compatible Python {ver[0]}.{ver[1]} at {cand} — "
               f"installing sonpy automatically...")
        ok, out = _pip_install_sonpy(cand, ver)
        if ok:
            log_cb(f"[SMRX] sonpy installed successfully in {cand}")
            _cache_sonpy_python(cand)
            if cand == sys.executable:
                _reload_sonpy_in_process()
            return cand
        log_cb(f"[SMRX] sonpy install failed in {cand}:\n{out}")

    portable = _bootstrap_portable_python(log_cb)
    if portable:
        _cache_sonpy_python(portable)
    return portable


# Small helper script run inside a *different* Python interpreter (via
# subprocess) purely to extract sonpy metadata as JSON — never the sample data.
_SMRX_META_HELPER = r"""
import sys, json
from math import floor
from sonpy import lib as sp

path = sys.argv[1]
try:
    fil = sp.SonFile(path, True)
    err = fil.GetOpenError()
    if err != 0:
        print(json.dumps({"error": "sonpy could not open file (error %d): %s" % (err, sp.GetErrorString(err))}))
        sys.exit(1)

    ADC_KINDS = (sp.DataType.Adc, sp.DataType.RealWave)
    chans = []
    for ch in range(fil.MaxChannels()):
        kind = fil.ChannelType(ch)
        if kind in ADC_KINDS and fil.ChannelMaxTime(ch) > 0:
            chans.append(ch)
    if not chans:
        print(json.dumps({"error": "No waveform channels found in the .smrx file."}))
        sys.exit(1)

    time_base = fil.GetTimeBase()
    sample_ticks = fil.ChannelDivide(chans[0])
    fs = 1.0 / (sample_ticks * time_base)

    ch_meta = []
    for ch in chans:
        title = (fil.GetChannelTitle(ch) or "").strip() or ("Ch%d" % ch)
        scale = fil.GetChannelScale(ch)
        offset = fil.GetChannelOffset(ch)
        max_time = fil.ChannelMaxTime(ch)
        divide = fil.ChannelDivide(ch)
        t_from = fil.FirstTime(ch, 0, max_time)
        n_samp = floor((max_time - t_from) / divide) + 1
        ch_meta.append({"title": title, "scale": scale, "offset": offset,
                         "n_samples": n_samp, "t_from": t_from,
                         "t_end": max_time, "divide": divide})
    print(json.dumps({"fs": fs, "channels": ch_meta}))
except Exception as e:
    print(json.dumps({"error": str(e)}))
    sys.exit(1)
"""


def _smrx_metadata_via_subprocess(python_exe: str, path: str) -> dict:
    import json as _json
    import tempfile as _tmp
    helper_path = os.path.join(_tmp.gettempdir(), "_tsi_smrx_meta_helper.py")
    with open(helper_path, "w", encoding="utf-8") as f:
        f.write(_SMRX_META_HELPER)
    r = subprocess.run([python_exe, helper_path, path],
                        capture_output=True, text=True, timeout=120)
    out_lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
    if not out_lines:
        raise RuntimeError(f"sonpy helper produced no output: {r.stderr.strip()}")
    try:
        payload = _json.loads(out_lines[-1])
    except Exception:
        raise RuntimeError(f"Could not parse sonpy helper output: {r.stdout}\n{r.stderr}")
    if "error" in payload:
        raise RuntimeError(payload["error"])
    return payload


def _smrx_metadata_inprocess(path: str) -> dict:
    from sonpy import lib as sp
    from math import floor

    fil = sp.SonFile(path, True)
    err = fil.GetOpenError()
    if err != 0:
        raise RuntimeError(f"sonpy could not open file (error {err}): {sp.GetErrorString(err)}")

    ADC_KINDS = (sp.DataType.Adc, sp.DataType.RealWave)
    chans: List[int] = []
    for ch in range(fil.MaxChannels()):
        kind = fil.ChannelType(ch)
        if kind in ADC_KINDS and fil.ChannelMaxTime(ch) > 0:
            chans.append(ch)
    if not chans:
        raise RuntimeError("No waveform channels found in the .smrx file.")

    time_base = fil.GetTimeBase()
    sample_ticks = fil.ChannelDivide(chans[0])
    fs = 1.0 / (sample_ticks * time_base)

    ch_meta = []
    for ch in chans:
        title = fil.GetChannelTitle(ch).strip() or f"Ch{ch}"
        scale = fil.GetChannelScale(ch)
        offset = fil.GetChannelOffset(ch)
        max_time = fil.ChannelMaxTime(ch)
        divide = fil.ChannelDivide(ch)
        t_from = fil.FirstTime(ch, 0, max_time)
        n_samp = floor((max_time - t_from) / divide) + 1
        ch_meta.append({
            "title": title, "scale": scale, "offset": offset,
            "n_samples": n_samp, "t_from": t_from,
            "t_end": max_time, "divide": divide,
        })
    return {"fs": fs, "channels": ch_meta}


def _read_smrx(path: str, log_cb=None) -> Tuple[np.ndarray, float, List[str]]:
    """
    Read a CED Spike2 .smrx (SON64) file.
    Metadata (titles/scale/offset/fs) comes from sonpy, which is located or
    installed automatically if it's missing (see _resolve_sonpy_python).
    The waveform samples themselves are parsed directly from the file's
    binary blocks below and never need sonpy at all.
    """
    log_cb = log_cb or _log_file
    import os as _os, shutil as _sh, tempfile as _tmp

    if not HAS_NUMPY:
        raise RuntimeError(
            "NumPy is not available in this Python environment and could not be "
            "installed automatically (see tsi_app.log for details). Reading .smrx "
            "files needs it. Try running:\n"
            f'  "{sys.executable}" -m pip install numpy'
        )

    path = _os.path.abspath(_os.path.normpath(path))
    if not _os.path.isfile(path):
        raise RuntimeError(f"File not found: {path}")

    needs_copy = any(ord(c) > 127 or c in ' ()[]{}' for c in path)
    tmp_dir    = None
    open_path  = path
    if needs_copy:
        tmp_dir   = _tmp.mkdtemp()
        open_path = _os.path.join(tmp_dir, "data.smrx")
        _sh.copy2(path, open_path)

    try:
        if HAS_SONPY:
            meta = _smrx_metadata_inprocess(open_path)
        else:
            python_exe = _resolve_sonpy_python(log_cb)
            if python_exe is None:
                raise ImportError(
                    "sonpy is not available and automatic setup could not complete.\n\n"
                    "You can install it manually in a Python 3.9 or 3.14 environment:\n"
                    "  py -3.9 -m pip install sonpy\n"
                    "Then run this application with that interpreter:\n"
                    "  py -3.9 app5_corregido.py\n\n"
                    "See the Setup log for details on what was tried automatically."
                )
            if python_exe == sys.executable and HAS_SONPY:
                meta = _smrx_metadata_inprocess(open_path)
            else:
                meta = _smrx_metadata_via_subprocess(python_exe, open_path)
        fs = meta["fs"]
        ch_meta = meta["channels"]
    finally:
        if tmp_dir and _os.path.isdir(tmp_dir):
            try:
                _sh.rmtree(tmp_dir)
            except Exception:
                pass

    # ── Direct binary read ──────────────────────────────────────────────────
    # SON64 block layout:
    #   Each block = 65536 bytes. Block 0 = file header (skip).
    #   Sentinel block  (blk_hdr[1]==2, sub[4]==SENTINEL):
    #     Multiple per channel for long recordings — all concatenated.
    #     Channel index = blk_hdr[0] // 4096
    #   Continuation block (blk_hdr[1]==2, sub[4] not in {0, SENTINEL}):
    #     sub[4] = sample count; data at sub[8].
    #   Index/metadata block (blk_hdr[1] != 2): skip.

    BLOCK_SIZE = 65536
    BLOCK_HDR  = 16
    SENTINEL   = 32752  # 0x7FF0

    with open(path, "rb") as fh:
        raw_bytes = fh.read()

    all_i16  = np.frombuffer(raw_bytes, dtype="<i2")
    n_blocks = len(raw_bytes) // BLOCK_SIZE

    sentinel_data:     dict = {}   # ch_idx → list of int16 arrays
    continuation_data: dict = {}   # ch_idx → int16 array

    for blk in range(1, n_blocks):
        base      = blk * BLOCK_SIZE // 2
        blk_hdr   = all_i16[base : base + BLOCK_HDR // 2]
        data_base = base + BLOCK_HDR // 2
        sub       = all_i16[data_base : data_base + 8]

        if len(blk_hdr) < 4 or int(blk_hdr[1]) != 2:
            continue

        ch_idx = int(blk_hdr[0]) // 4096

        if len(sub) >= 5 and int(sub[4]) == SENTINEL:
            adc_start = data_base + 8
            adc_arr   = all_i16[adc_start : base + BLOCK_SIZE // 2]
            if ch_idx not in sentinel_data:
                sentinel_data[ch_idx] = []
            sentinel_data[ch_idx].append(adc_arr)

        elif len(sub) >= 5 and int(sub[4]) not in (0, SENTINEL):
            count = int(sub[4])
            if 0 < count < BLOCK_SIZE // 2:
                adc_start  = data_base + 8
                existing   = continuation_data.get(ch_idx, np.array([], dtype="<i2"))
                new_data   = all_i16[adc_start : adc_start + count]
                continuation_data[ch_idx] = np.concatenate([existing, new_data])

    if not sentinel_data and not continuation_data:
        raise RuntimeError("No waveform data blocks found — file format may be unsupported.")
    # NOTE: short recordings (a channel's entire signal fits in a single
    # block) never emit a "sentinel" block at all — CED only uses the
    # sentinel marker when a channel's data spans MULTIPLE blocks. In that
    # case the one and only block for that channel is classified above as a
    # "continuation" block (sub[4] holds its sample count directly), so
    # continuation_data alone is enough; sentinel_data can legitimately be
    # empty for such files.

    calibrated: List[np.ndarray] = []
    titles: List[str]            = []

    for i, meta in enumerate(ch_meta):
        ch_idx  = i
        n_sonpy = meta["n_samples"]

        parts  = sentinel_data.get(ch_idx, [])
        cont   = continuation_data.get(ch_idx, np.array([], dtype="<i2"))
        raw_ch = np.concatenate(parts + [cont]).astype(np.float64)

        n_binary = len(raw_ch)
        if abs(n_binary - n_sonpy) > max(10, n_sonpy * 0.01):
            _log_file(
                f"[SMRX WARNING] ch{ch_idx} ({meta['title']}): "
                f"binary={n_binary}, sonpy={n_sonpy} (diff={n_binary-n_sonpy})"
            )

        seg = raw_ch[:n_sonpy]
        if len(seg) < n_sonpy:
            seg = np.pad(seg, (0, n_sonpy - len(seg)))

        cal = seg * (meta["scale"] / 6553.6) + meta["offset"]
        calibrated.append(cal)
        titles.append(meta["title"])

    n_min = min(len(a) for a in calibrated)
    data  = np.column_stack([a[:n_min] for a in calibrated])
    data -= data.mean(axis=0, keepdims=True)

    return data, fs, titles


def _infer_channel_config(titles: List[str],
                          n_signal_cols: int,
                          first_signal_col_1based: int = 1
                          ) -> List[dict]:
    """
    Infer channel kind (ACC/EMG) and side (Left/Right) from channel titles.

    Robust to:
    - Mixed case  (FCR_R, fcr_r, FCR_Right)
    - Multiple separators  (FCR_r, FCR-r, FCR.r, FCR r, FCRr)
    - Side anywhere in the string  (R_FCR, FCR_R, RightFCR, FCR_Right)
    - Known muscle/sensor abbreviations
    - Positional fallback when name gives no information

    Returns a list of dicts:
        [{"col_1based": int, "kind": "ACC"|"EMG", "side": "Left"|"Right",
          "enabled": bool}, ...]
    """
    import re

    # ── Known name patterns ──────────────────────────────────────────────
    # These override everything else when matched.
    ACC_NAMES = {
        "acc", "accel", "accelero", "accelerom", "accelerometer",
        "gyr", "gyro", "gyroscope", "kin", "kinem", "kinematic",
        "mov", "motion", "imu",
    }
    EMG_NAMES = {
        "emg", "muscle", "electro", "musc",
        # Common wrist/forearm muscles
        "fcr", "ecr", "fcu", "ecu", "fds", "fpl",
        "bb", "tb", "br",               # biceps, triceps, brachioradialis
        "fdi", "apb", "adm",            # intrinsics
        "ta", "sol", "gm", "gl",        # lower limb
        "adc", "adb",
    }

    # ── Side patterns ────────────────────────────────────────────────────
    # Matched anywhere in the normalised token list
    LEFT_TOKENS  = {"l", "left", "izq", "izquierda", "izquierdo", "izda", "izdo",
                     "i", "gauche", "links"}
    RIGHT_TOKENS = {"r", "right", "der", "derecha", "derecho", "dcha", "dcho",
                     "d", "droit", "rechts"}

    def _normalise(s: str) -> List[str]:
        """Lowercase, replace separators with spaces, split into tokens."""
        s = s.lower()
        s = re.sub(r"[\-\.\s_/\\]+", " ", s)
        return [t for t in s.split() if t]

    def _infer_kind(tokens: List[str]) -> str:
        for t in tokens:
            if t in ACC_NAMES:
                return "ACC"
            if t in EMG_NAMES:
                return "EMG"
        return None   # unknown

    def _infer_side(tokens: List[str]) -> str:
        for t in tokens:
            if t in LEFT_TOKENS:
                return "Left"
            if t in RIGHT_TOKENS:
                return "Right"
        return None   # unknown

    result = []
    kind_unknown_indices = []   # indices needing positional fallback for kind
    side_unknown_indices = []   # indices needing positional fallback for side

    for i in range(n_signal_cols):
        col_1based = first_signal_col_1based + i
        title      = titles[i] if i < len(titles) else f"Ch{i+1}"
        tokens     = _normalise(title)

        kind = _infer_kind(tokens)
        side = _infer_side(tokens)

        entry = {"col_1based": col_1based, "kind": kind,
                 "side": side, "enabled": True, "title": title}
        result.append(entry)

        if kind is None:
            kind_unknown_indices.append(i)
        if side is None:
            side_unknown_indices.append(i)

    # ── Positional fallback for KIND ─────────────────────────────────────
    # If some channels have unknown kind, use position:
    # channels whose kind IS known tell us the layout.
    # If still nothing, assume last 2 = ACC, rest = EMG.
    if kind_unknown_indices:
        known_kinds = [(i, result[i]["kind"])
                       for i in range(len(result))
                       if result[i]["kind"] is not None]
        if not known_kinds:
            # No information at all — last 2 = ACC, rest = EMG
            for i in kind_unknown_indices:
                result[i]["kind"] = "ACC" if i >= n_signal_cols - 2 else "EMG"
        else:
            # Inherit from nearest known channel
            for i in kind_unknown_indices:
                nearest = min(known_kinds, key=lambda x: abs(x[0] - i))
                result[i]["kind"] = nearest[1]

    # ── Positional fallback for SIDE ─────────────────────────────────────
    # Alternate Right/Left in pairs among channels of the same kind.
    if side_unknown_indices:
        for kind_val in ("EMG", "ACC"):
            group = [i for i in range(len(result))
                     if result[i]["kind"] == kind_val]
            pair_idx = 0
            for i in group:
                if result[i]["side"] is None:
                    result[i]["side"] = "Right" if pair_idx % 2 == 0 else "Left"
                pair_idx += 1

    # Final safety net
    for entry in result:
        if entry["kind"] is None: entry["kind"] = "EMG"
        if entry["side"] is None: entry["side"] = "Right"

    return result



    """
    Read a CED Spike2 .smrx (SON64) file.

    Uses a direct binary parser based on the SON64 specification.
    sonpy is used only to extract channel metadata (titles, scale, offset, fs).
    Raw ADC data is read directly from the binary file, bypassing sonpy's
    ReadInts/ReadFloats which incorrectly return repeated values for this
    file format version.
    """
    if not HAS_SONPY:
        raise ImportError(
            "sonpy is not installed.\n"
            "Install it in a Python 3.9 environment:\n"
            "  py -3.9 -m pip install sonpy\n"
            "Then run this application with that interpreter:\n"
            "  py -3.9 app5_corregido.py"
        )

    from sonpy import lib as sp
    from math import floor
    import struct as _struct
    import os as _os, shutil as _sh, tempfile as _tmp

    # Normalize path
    path = _os.path.abspath(_os.path.normpath(path))

    if not _os.path.isfile(path):
        raise RuntimeError(f"File not found: {path}")

    # sonpy can fail with paths containing spaces or non-ASCII characters.
    # Copy to a short temp path if needed.
    needs_copy = any(ord(c) > 127 or c in ' ()[]{}' for c in path)
    tmp_dir    = None
    open_path  = path
    if needs_copy:
        tmp_dir   = _tmp.mkdtemp()
        open_path = _os.path.join(tmp_dir, "data.smrx")
        _sh.copy2(path, open_path)

    try:
        fil = sp.SonFile(open_path, True)
        err = fil.GetOpenError()
        if err != 0:
            raise RuntimeError(
                f"sonpy could not open file (error {err}): "
                f"{sp.GetErrorString(err)}"
            )

        ADC_KINDS = (sp.DataType.Adc, sp.DataType.RealWave)

        # Collect active waveform channels and their metadata via sonpy
        waveform_chans: List[Tuple[int, object]] = []
        for ch in range(fil.MaxChannels()):
            kind = fil.ChannelType(ch)
            if kind in ADC_KINDS and fil.ChannelMaxTime(ch) > 0:
                waveform_chans.append((ch, kind))

        if not waveform_chans:
            raise RuntimeError("No waveform channels found in the .smrx file.")

        # Sampling rate from first channel
        time_base    = fil.GetTimeBase()
        ch0, _       = waveform_chans[0]
        sample_ticks = fil.ChannelDivide(ch0)
        fs           = 1.0 / (sample_ticks * time_base)

        # Collect metadata for each channel
        ch_meta = []
        for ch, kind in waveform_chans:
            title    = fil.GetChannelTitle(ch).strip() or f"Ch{ch}"
            scale    = fil.GetChannelScale(ch)
            offset   = fil.GetChannelOffset(ch)
            max_time = fil.ChannelMaxTime(ch)
            divide   = fil.ChannelDivide(ch)
            t_from   = fil.FirstTime(ch, 0, max_time)
            n_samp   = floor((max_time - t_from) / divide) + 1
            ch_meta.append({
                "title": title, "kind": kind,
                "scale": scale, "offset": offset,
                "n_samples": n_samp,
                "t_from": t_from,
                "t_end": max_time,
                "divide": divide,
            })

    finally:
        # Remove temp copy now that all metadata is collected
        if tmp_dir and _os.path.isdir(tmp_dir):
            try:
                _sh.rmtree(tmp_dir)
            except Exception:
                pass

    # Binary read uses the original path (not temp copy which is already deleted)
    # ── Direct binary read ──────────────────────────────────────────────────
    # SON64 block layout (fully reverse-engineered):
    #
    # Each block is 65536 bytes. Block 0 = file header (skip).
    # Remaining blocks classified by their 16-byte header:
    #
    #   Sentinel block  (blk_hdr[1] == 2, sub[4] == 32752):
    #     [blk_hdr(16B)][4-sample preamble][sentinel 0x7FF0][3 zeros][ADC data...]
    #     Contains first 32752 samples of a channel.
    #     Channel index = blk_hdr[0] // 4096
    #
    #   Continuation block (blk_hdr[1] == 2, sub[1] == 49):
    #     [blk_hdr(16B)][sub: x, 49, 0, 0, count, 0, 0, 0][ADC data (count samples)]
    #     Channel index = blk_hdr[0] // 4096
    #
    #   Index/metadata block (blk_hdr[1] != 2): skip entirely.
    #
    # Channel index is encoded in blk_hdr[0]: 0→ch0, 4096→ch1, 8192→ch2, etc.
    # This mapping is read dynamically — works for any channel count and length.

    BLOCK_SIZE = 65536
    BLOCK_HDR  = 16
    SENTINEL   = 32752  # 0x7FF0

    with open(path, "rb") as fh:
        raw_bytes = fh.read()

    all_i16  = np.frombuffer(raw_bytes, dtype="<i2")
    n_blocks = len(raw_bytes) // BLOCK_SIZE

    sentinel_data:     dict = {}   # ch_idx → list of int16 arrays (one per sentinel block)
    continuation_data: dict = {}   # ch_idx → int16 array of continuation ADC data

    for blk in range(1, n_blocks):
        base      = blk * BLOCK_SIZE // 2
        blk_hdr   = all_i16[base : base + BLOCK_HDR // 2]
        data_base = base + BLOCK_HDR // 2
        sub       = all_i16[data_base : data_base + 8]

        if len(blk_hdr) < 4 or int(blk_hdr[1]) != 2:
            continue  # index/metadata block — skip

        ch_idx = int(blk_hdr[0]) // 4096

        if len(sub) >= 5 and int(sub[4]) == SENTINEL:
            # Sentinel block: ADC data starts at sub[8]
            # Multiple sentinel blocks per channel for long recordings
            adc_start = data_base + 8
            adc_arr   = all_i16[adc_start : base + BLOCK_SIZE // 2]
            if ch_idx not in sentinel_data:
                sentinel_data[ch_idx] = []
            sentinel_data[ch_idx].append(adc_arr)

        elif len(sub) >= 5 and int(sub[4]) not in (0, SENTINEL):
            # Continuation block: identified by having a plausible sample count
            # at sub[4] (not 0, not SENTINEL). sub[1] varies (49, 199, or other)
            # across different file versions — so we don't filter on sub[1].
            count = int(sub[4])
            # Sanity check: count must be a plausible number of samples
            # (positive, less than one full block worth)
            if 0 < count < BLOCK_SIZE // 2:
                adc_start = data_base + 8
                # If multiple continuation blocks exist for a channel, concatenate
                existing = continuation_data.get(ch_idx, np.array([], dtype="<i2"))
                new_data = all_i16[adc_start : adc_start + count]
                continuation_data[ch_idx] = np.concatenate([existing, new_data])

    if not sentinel_data:
        raise RuntimeError("No sentinel blocks found — file format may be unsupported.")

    # ── Extract and calibrate each channel ───────────────────────────────
    # Cross-validate binary read against sonpy's authoritative sample count.
    # If counts don't match, log a warning but continue with what we have.
    calibrated: List[np.ndarray] = []
    titles: List[str]            = []

    for i, meta in enumerate(ch_meta):
        ch_idx  = i
        n_sonpy = meta["n_samples"]

        # Concatenate all sentinel blocks (in file order) + continuation
        parts   = sentinel_data.get(ch_idx, [])
        cont    = continuation_data.get(ch_idx, np.array([], dtype="<i2"))
        raw_ch  = np.concatenate(parts + [cont]).astype(np.float64)

        n_binary = len(raw_ch)

        # Cross-validate
        if abs(n_binary - n_sonpy) > max(10, n_sonpy * 0.01):
            # More than 1% or 10 samples difference — log warning
            _log_file(
                f"[SMRX WARNING] ch{ch_idx} ({meta['title']}): "
                f"binary={n_binary} samples, sonpy={n_sonpy} samples "
                f"(diff={n_binary - n_sonpy}). "
                f"Using sonpy count as authoritative."
            )

        # Use sonpy's count as the authoritative recording window
        seg = raw_ch[:n_sonpy]
        if len(seg) < n_sonpy:
            seg = np.pad(seg, (0, n_sonpy - len(seg)))

        # Calibrate: V = int16 × (scale / 6553.6) + offset
        cal = seg * (meta["scale"] / 6553.6) + meta["offset"]
        calibrated.append(cal)
        titles.append(meta["title"])

    n_min = min(len(a) for a in calibrated)
    data  = np.column_stack([a[:n_min] for a in calibrated])

    # Remove per-channel DC offset before filtering
    data -= data.mean(axis=0, keepdims=True)

    return data, fs, titles

# Logging Helper
# -------------------------------------------------------------------------------
# LoggerMixin standardizes how tabs emit log messages. Tabs inherit from this
# mixin to call self.log(...) and route messages to the main application's log
# panel and/or stdout. This keeps UI code cleaner.
class LoggerMixin:
    """
    Mixin that provides a log() method routing to both a GUI callback and the log file.
    Subclasses must set self._log_cb before calling log().
    """
    def log(self, msg: str) -> None:
        try:
            if callable(getattr(self, "_log_cb", None)):
                self._log_cb(msg)
            else:
                print(msg)
        except Exception:
            pass
        _log_file(msg)


# --------------------
# Main window
# --------------------
class TSIApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("TSI App — Visualization + Selection + Frequency-domain")
        self.geometry("1250x900")
        self.state = AppState()
        self._build_ui()

    def _build_ui(self) -> None:
        # Top toolbar — always visible
        toolbar = ttk.Frame(self, relief="ridge")
        toolbar.pack(side="top", fill="x", padx=0, pady=0)
        ttk.Button(toolbar, text="📄  Export Report (PDF)",
                   command=self._export_report).pack(side="right", padx=8, pady=4)
        ttk.Label(toolbar, text="Tremor Analysis",
                  font=("TkDefaultFont", 10, "bold")).pack(side="left", padx=8, pady=4)

        # Maneuver selector — switches which session recording is displayed
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=6, pady=4)
        ttk.Label(toolbar, text="Viewing:").pack(side="left", padx=(0, 4), pady=4)
        self._var_maneuver = tk.StringVar(value="— current —")
        self._cb_maneuver  = ttk.Combobox(toolbar, textvariable=self._var_maneuver,
                                           state="readonly", width=22)
        self._cb_maneuver.pack(side="left", pady=4)
        self._cb_maneuver.bind("<<ComboboxSelected>>", self._on_maneuver_select)

        # Top notebook + bottom log
        top = ttk.Frame(self)
        top.pack(side="top", fill="both", expand=True)
        bottom = ttk.Frame(self)
        bottom.pack(side="bottom", fill="x")

        # Tabs
        self.nb = ttk.Notebook(top)
        self.nb.pack(fill="both", expand=True, padx=8, pady=8)

        self.viz = VisualizationTab(self.nb, self.state, log_cb=self.log)
        self.setup = SetupTab(self.nb, self.state, log_cb=self.log, viz=self.viz)
# -------------------------------------------------------------------------------
# The main Notebook collects individual tabs in a specific order. This determines
# the left-to-right order of tabs visible to the user.
        self.nb.add(self.setup, text="Setup")
        self.nb.add(self.viz, text="Visualization")

        # Selection TAB
        self.sel = SelectionTab(self.nb, self.state, self.viz, log_cb=self.log)
        self.nb.add(self.sel, text="Selection")

        # Explore/Welch tab — all channels stacked in one column, 2 movable
        # cursors, per-channel Welch of the cursor-selected segment.
        self.explore = ExploreWelchTab(self.nb, self.state, log_cb=self.log)
        self.nb.add(self.explore, text="Explore/Welch")
        self.setup._explore = self.explore

        # FFT tab
        self.fft = FFTTab(self.nb, self.state, log_cb=self.log)
        self.nb.add(self.fft, text="FFT")
        # Wire FFT auto-redraw into SetupTab (same pattern as viz)
        self.setup._fft = self.fft

        self.coh = CoherenceTab(self.nb, self.state, self.viz, log_cb=self.log)
        self.nb.add(self.coh, text="Coherence/Cumulant")

        self.spec = SpectrogramTab(self.nb, self.state, self.viz, log_cb=self.log)
        self.nb.add(self.spec, text="Spectrogram")

        self.tsi_tab = TSITab(self.nb, self.state, log_cb=self.log)
        self.nb.add(self.tsi_tab, text="TSI")

        # Comparison tab (formerly Session)
        self.session = SessionTab(self.nb, self.state, self.viz, self.fft,
                                  self.tsi_tab, log_cb=self.log,
                                  spectrogram_tab=self.spec, coherence_tab=self.coh)
        self.nb.add(self.session, text="Comparison")

        # Export button — always visible above the log console
        # Bottom log console
        ttk.Label(bottom, text="Log:").pack(anchor="w", padx=8, pady=(6, 0))
        self.log_text = ScrolledText(bottom, height=10, wrap="word")
        self.log_text.pack(fill="both", expand=True, padx=8, pady=6)

        # Status bar
        self.status = tk.StringVar(value="Ready.")
        ttk.Label(self, textvariable=self.status, relief="sunken", anchor="w").pack(fill="x", side="bottom")

        # Shortcuts
        self.bind("<Control-l>", lambda e: self._clear_log())
        self.bind("<Control-s>", lambda e: self._save_state_to_log())

        if not HAS_SCIPY:
            self.log("[INFO] SciPy not found — using FIR fallbacks; coherence CI uses a simplified jackknife.")

    # bottom log helpers
    def log(self, message: str) -> None:
        try:
            self.log_text.insert("end", message.rstrip() + "\n")
            self.log_text.see("end")
            self.status.set(message.splitlines()[0][:100])
        except Exception as e:
            _log_file(f"GUI log write failed: {e}")

    def _export_report(self) -> None:
        if self.state.raw_data is None:
            messagebox.showinfo("No data", "Load a file first.")
            return
        ReportDialog(self, self.state,
                     self.viz, self.fft, self.spec, self.tsi_tab,
                     log_cb=self.log)

    def refresh_maneuver_selector(self) -> None:
        """Called by SessionTab when the maneuver list changes."""
        options = ["— current —"] + [m.label for m in self.session.maneuvers]
        self._cb_maneuver["values"] = options
        if self._var_maneuver.get() not in options:
            self._var_maneuver.set("— current —")

    def _on_maneuver_select(self, event=None) -> None:
        """Load selected maneuver into app_state and refresh all tabs."""
        sel = self._var_maneuver.get()
        if sel == "— current —":
            return
        maneuver = next(
            (m for m in self.session.maneuvers if m.label == sel), None)
        if maneuver is None:
            return

        import copy
        self.state.raw_data       = maneuver.data.copy()
        self.state.fs             = maneuver.fs
        self.state.channel_config = copy.deepcopy(maneuver.channel_config)
        self.state.selection_t0   = None
        self.state.selection_t1   = None

        self.log(f"[Session] Viewing: '{maneuver.label}'  "
                 f"({maneuver.data.shape[0]/maneuver.fs:.1f}s)")

        # Update Fs entry in Setup
        self.setup.var_fs.set(str(round(maneuver.fs, 2)))

        # Refresh all main views
        for tab in [self.viz, self.fft, self.spec, self.explore]:
            try: tab.redraw()
            except Exception: pass

        # Coherence/Cumulant caches its channel matrix (self._Y) and channel
        # labels across redraws for performance, and its own redraw() only
        # rebuilds that cache when the currently selected channel label is no
        # longer found in the list — NOT whenever the underlying data changes.
        # So when switching "Viewing" to a different maneuver, if the new
        # maneuver happens to reuse the same channel labels (e.g. same
        # Setup/channel-name configuration across maneuvers of one session,
        # which is the common case), the cache looked "valid" and the tab
        # kept silently computing on the PREVIOUS maneuver's raw data. Force a
        # fresh rebuild here so it always reflects the maneuver now selected.
        try:
            self.coh._prepare_matrix_and_labels()
            self.coh.redraw()
        except Exception:
            pass

    def _clear_log(self) -> None:
        self.log_text.delete("1.0", "end")
        self.status.set("Log cleared.")

    def _save_state_to_log(self) -> None:
        self.log(self.state.summary())



# --------------------
# FFT tab (drop-in from v4)
# --------------------

# FFT Tab (Frequency Domain of Processed Signals)
# -------------------------------------------------------------------------------
# Purpose: Compute a single-sided magnitude FFT per enabled channel using the
# PROCESSED signals. If a selection is active, the FFT is computed only over the
# selected segment; otherwise it uses the full duration.
# Layout: Mirrors Visualization (two columns for Left/Right). All plots share the
# same X-axis; Y-axis is often shared within type groups (ACC vs EMG). Default
# X-range is usually 2–20 Hz (adjust via pan/zoom).
# Robustness: Implementations commonly interpolate NaN/Inf in the selected
# segment so plots never vanish due to non-finite values.
class FFTTab(ttk.Frame, LoggerMixin):
    """
    Computes magnitude FFT for the *processed* signals only.
    • Uses the Selection (t0–t1) if present; otherwise entire duration.
    • Layout mirrors Visualization: two columns (Left | Right), rows by count.
    • All subplots share the same X-axis; Y is shared within type groups (ACC vs EMG)
      by default, or fit independently per subplot with "Fit Y-axis".
    • Default X-range = 2–20 Hz (you can pan/zoom).
    • Draggable peak markers: each subplot shows a vertical line at the FFT peak.
      Drag the line to any frequency; the summary table updates live.
    """
    def __init__(self, parent, app_state: AppState, log_cb):
        super().__init__(parent)
        self.app_state = app_state
        self._log_cb = log_cb
        self.fig = None
        self.canvas = None
        self._axes = []
        self._fit_y_active: bool = False
        self._last_fft_data: List[Tuple[np.ndarray, np.ndarray, str]] = []

        # Draggable marker state
        # _markers[i] = {'vline': Line2D, 'annot': Annotation, 'f': float, 'm': float,
        #                 'fdata': array, 'mdata': array, 'label': str}
        self._markers: List[dict] = []
        self._drag_idx: Optional[int] = None   # which marker is being dragged
        self._drag_cid_press   = None
        self._drag_cid_release = None
        self._drag_cid_motion  = None

        self._build()

    def _build(self) -> None:
        ctrl = ttk.Frame(self); ctrl.pack(fill="x", padx=10, pady=6)
        ttk.Button(ctrl, text="Refresh", command=self.redraw).pack(side="left", padx=5)
        self._btn_fit = ttk.Button(ctrl, text="Fit Y-axis", command=self._toggle_fit_y)
        self._btn_fit.pack(side="left", padx=(4, 10))

        # Welch / FFT toggle
        ttk.Label(ctrl, text="Method:").pack(side="left", padx=(6, 2))
        self._var_method = tk.StringVar(value="Welch")
        ttk.Radiobutton(ctrl, text="Welch (PSD)", variable=self._var_method,
                        value="Welch").pack(side="left", padx=2)
        ttk.Radiobutton(ctrl, text="FFT", variable=self._var_method,
                        value="FFT").pack(side="left", padx=2)

        ttk.Label(ctrl, text="• Default 2–20 Hz • drag markers to adjust",
                  foreground="gray").pack(side="left", padx=8)

        # Main area: plot left, summary table right
        main = ttk.Frame(self); main.pack(fill="both", expand=True, padx=8, pady=4)
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=0)
        main.rowconfigure(0, weight=1)

        plot_frame = ttk.Frame(main)
        plot_frame.grid(row=0, column=0, sticky="nsew")

        if MATPLOTLIB_OK:
            self.fig = plt.Figure(figsize=(10, 6), dpi=100)
            self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
            self.canvas.get_tk_widget().pack(fill="both", expand=True)
            NavigationToolbar2Tk(self.canvas, plot_frame).update()
        else:
            ttk.Label(plot_frame, text="Matplotlib not available").pack(pady=20)

        # Summary table (right panel)
        tbl_frame = ttk.LabelFrame(main, text="Peak markers", width=220)
        tbl_frame.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        tbl_frame.grid_propagate(False)

        self._tbl_label = ttk.Label(tbl_frame, text="Press 'Refresh'",
                                     font=("Courier New", 9), justify="left",
                                     anchor="nw")
        self._tbl_label.pack(fill="both", expand=True, padx=4, pady=4)

    # --------- helpers ---------
    def _selected_channels(self):
        cols0 = []; kinds = []; sides = []; names = []
        for i in range(8):
            cfg = self.app_state.channel_config.get(i, ChannelConfig(enabled=False, col_1based=i+1))
            if not cfg.enabled:
                continue
            cols0.append(cfg.col_1based - 1)
            kinds.append(cfg.kind)
            sides.append(cfg.side)
            names.append(_channel_display_name(cfg, i))
        return cols0, kinds, sides, names

    def _get_selection_indices(self, n: int) -> Tuple[int, int]:
        return _get_selection_indices(self.app_state, n)

    def _compute_fft(self, x, fs):
        """
        Compute power spectral density using Welch's method.

        Uses fixed 2-second segments regardless of recording duration,
        so PSD amplitude (units²/Hz) is comparable across recordings
        of different lengths.

        Returns
        -------
        f   : frequency array (Hz)
        psd : PSD in (units²/Hz)  — density scaling
        """
        from scipy import signal as _sps
        x = np.asarray(x, dtype=float).ravel()
        n = x.size
        if n < 2 or not np.isfinite(fs) or fs <= 0:
            return np.array([0.0]), np.array([0.0])
        finite = np.isfinite(x)
        if not finite.any():
            return np.array([0.0, fs/2.0]), np.array([0.0, 0.0])
        if not finite.all():
            idx = np.arange(n)
            x[~finite] = np.interp(idx[~finite], idx[finite], x[finite])

        # Fixed 2-second segment → resolution = 0.5 Hz, comparable across durations
        # Use 4-second segments for better frequency resolution (0.25 Hz)
        # while maintaining enough segments for stable PSD estimate
        nperseg = min(int(fs * 4), n)
        f, psd = _sps.welch(x - x.mean(), fs=fs, nperseg=nperseg,
                            noverlap=nperseg // 2,
                            window="hann", scaling="density")
        psd = np.nan_to_num(psd, nan=0.0, posinf=0.0, neginf=0.0)
        return f, psd

    def _significance_threshold(self, f: np.ndarray, psd: np.ndarray,
                                 flo: float = 2.0, fhi: float = 20.0):
        """
        Estimate the noise floor and significance threshold within [flo, fhi] Hz.

        Method (robust, band-restricted):
          - noise_floor = 25th percentile of PSD in [flo, fhi]
            (resistant to peaks — reflects the spectral background)
          - noise_sd    = IQR / 1.35  (robust SD estimator)
          - threshold   = noise_floor + 2 × noise_sd

        Returns threshold value (same units as psd).
        """
        mask = (f >= flo) & (f <= fhi)
        if not mask.any():
            return float(np.max(psd)) * 2.0
        band = psd[mask]
        band = band[np.isfinite(band)]
        if len(band) < 4:
            return float(np.max(psd)) * 2.0
        q25 = float(np.percentile(band, 25))
        q75 = float(np.percentile(band, 75))
        iqr = q75 - q25
        noise_sd = iqr / 1.35   # robust SD
        return q25 + 2.0 * noise_sd

    def _compute_fft_simple(self, x, fs):
        """Plain single-window FFT (Hanning). Returns (f, magnitude)."""
        x = np.asarray(x, dtype=float).ravel()
        n = x.size
        if n < 2 or not np.isfinite(fs) or fs <= 0:
            return np.array([0.0]), np.array([0.0])
        finite = np.isfinite(x)
        if not finite.any():
            return np.array([0.0, fs/2.0]), np.array([0.0, 0.0])
        if not finite.all():
            idx = np.arange(n)
            x[~finite] = np.interp(idx[~finite], idx[finite], x[finite])
        w    = np.hanning(n)
        xw   = (x - x.mean()) * w
        X    = np.fft.rfft(xw)
        f    = np.fft.rfftfreq(n, d=1.0/fs)
        wrms = float(np.sqrt(np.mean(w**2))) or 1.0
        mag  = np.abs(X) * 2.0 / (n * wrms)
        mag  = np.nan_to_num(mag, nan=0.0, posinf=0.0, neginf=0.0)
        return f, mag

    def _safe_minmax(self, f, m, xlo, xhi):
        sl = (f >= xlo) & (f <= xhi)
        if not np.any(sl):
            return 0.0, 1.0
        y = m[sl]; y = y[np.isfinite(y)]
        if y.size == 0: return 0.0, 1.0
        lo = float(np.min(y)); hi = float(np.max(y))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            hi = lo + (1.0 if not np.isfinite(hi) else max(1.0, abs(lo) * 0.1))
        return lo, hi

    def _toggle_fit_y(self) -> None:
        self._fit_y_active = not self._fit_y_active
        self._btn_fit.config(text="Fit Y-axis ✓" if self._fit_y_active else "Fit Y-axis")
        if self._axes and self._last_fft_data:
            self._apply_y_scaling()
            self.canvas.draw_idle()

    def _apply_y_scaling(self) -> None:
        xlims = (2.0, 20.0)
        real_axes = [ax for ax in self._axes if ax.lines]
        fft_entries = self._last_fft_data
        if self._fit_y_active:
            for ax, (f, m, _) in zip(real_axes, fft_entries):
                try: xlo, xhi = ax.get_xlim()
                except Exception: xlo, xhi = xlims
                lo, hi = self._safe_minmax(f, m, xlo, xhi)
                margin = (hi - lo) * 0.05 if hi > lo else max(hi * 0.05, 0.01)
                ax.set_ylim(lo - margin, hi + margin)
        else:
            acc_lims = [self._safe_minmax(f, m, *xlims) for f, m, k in fft_entries if k == "ACC"]
            emg_lims = [self._safe_minmax(f, m, *xlims) for f, m, k in fft_entries if k == "EMG"]
            def group_lim(lims):
                if not lims: return None
                lo = min(v[0] for v in lims); hi = max(v[1] for v in lims)
                if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                    return (lo - 1.0, hi + 1.0)
                # Headroom above the group max so the tallest peak's marker
                # line and "★ freq Hz" label have room to sit above the
                # curve instead of touching (or clipping against) the top.
                pad = (hi - lo) * 0.12
                return (lo, hi + pad)
            acc_lim = group_lim(acc_lims); emg_lim = group_lim(emg_lims)
            for ax, (f, m, kind) in zip(real_axes, fft_entries):
                lim = acc_lim if kind == "ACC" else emg_lim
                if lim: ax.set_ylim(*lim)

    # --------- marker helpers ---------
    def _peak_in_view(self, f: np.ndarray, m: np.ndarray, ax) -> Tuple[float, float]:
        """Return (freq, magnitude) of the FFT peak within the current X-axis view."""
        try:
            xlo, xhi = ax.get_xlim()
        except Exception:
            xlo, xhi = 2.0, 20.0
        mask = (f >= xlo) & (f <= xhi) & np.isfinite(m)
        if not mask.any():
            return float(f[np.argmax(m)]), float(np.max(m))
        idx = np.argmax(m[mask])
        f_vis = f[mask]; m_vis = m[mask]
        return float(f_vis[idx]), float(m_vis[idx])

    def _interp_mag(self, marker: dict, freq: float) -> float:
        """Interpolate magnitude at an arbitrary frequency from stored FFT data."""
        return float(np.interp(freq, marker["fdata"], marker["mdata"]))

    def _compute_bandwidth(self, f_arr: np.ndarray, m_arr: np.ndarray,
                            f_peak: float, m_peak: float,
                            flo: float = 2.0, fhi: float = 20.0
                            ) -> tuple:
        """
        Compute half-power bandwidth (BW) around f_peak.

        Finds the frequencies where PSD = m_peak / 2 on each side of the peak.
        Returns (bw, f_lo, f_hi) in Hz.  Returns (None, None, None) if the
        half-power crossing cannot be found on either side.
        """
        mask = (f_arr >= flo) & (f_arr <= fhi)
        f_v  = f_arr[mask]
        m_v  = m_arr[mask]
        if f_v.size < 4 or m_peak <= 0:
            return None, None, None

        half = m_peak / 2.0
        pk_idx = int(np.argmin(np.abs(f_v - f_peak)))

        # Left crossing: scan left from peak
        f_left = None
        for i in range(pk_idx, 0, -1):
            if m_v[i - 1] <= half:
                # Linear interpolation
                slope = (m_v[i] - m_v[i-1]) / (f_v[i] - f_v[i-1])
                if slope != 0:
                    f_left = f_v[i-1] + (half - m_v[i-1]) / slope
                else:
                    f_left = (f_v[i] + f_v[i-1]) / 2
                break

        # Right crossing: scan right from peak
        f_right = None
        for i in range(pk_idx, len(f_v) - 1):
            if m_v[i + 1] <= half:
                slope = (m_v[i+1] - m_v[i]) / (f_v[i+1] - f_v[i])
                if slope != 0:
                    f_right = f_v[i] + (half - m_v[i]) / slope
                else:
                    f_right = (f_v[i] + f_v[i+1]) / 2
                break

        if f_left is None or f_right is None:
            return None, None, None

        bw = float(f_right - f_left)
        return bw, float(f_left), float(f_right)

    @staticmethod
    def _annot_corner(ax, freq: float) -> tuple:
        """
        Pick the top corner (axes-fraction x, ha) farthest from `freq`
        within the axis' current x-range, so a peak-info box never lands
        on top of the peak it is describing.
        """
        try:
            xlo, xhi = ax.get_xlim()
        except Exception:
            xlo, xhi = 2.0, 20.0
        frac = (freq - xlo) / (xhi - xlo) if xhi > xlo else 0.5
        if frac <= 0.5:
            return 0.98, "right"   # peak on the left half -> box top-right
        return 0.02, "left"        # peak on the right half -> box top-left

    def _add_marker(self, ax, f_arr, m_arr, label: str, color: str) -> None:
        """Place a draggable vertical-line marker at the FFT peak + half-power BW line."""
        f_peak, m_peak = self._peak_in_view(f_arr, m_arr, ax)

        # Vertical line at peak frequency
        vline = ax.axvline(f_peak, color=color, linewidth=1.5,
                           linestyle="--", alpha=0.85, picker=6)

        # Half-power bandwidth — shaded span only (no horizontal line)
        bw, f_lo, f_hi = self._compute_bandwidth(f_arr, m_arr, f_peak, m_peak)
        if bw is not None:
            span = ax.axvspan(f_lo, f_hi, alpha=0.12, color=color)
        else:
            span = None

        # Annotation — placed in whichever top corner is farthest from the
        # peak's x-position, so the box doesn't sit on top of the curve.
        # (The vertical dashed line already marks the exact peak location.)
        ann_x, ha = self._annot_corner(ax, f_peak)
        bw_str = f"\nBW={bw:.2f}Hz" if bw is not None else ""
        annot = ax.text(ann_x, 0.97,
                        f"{f_peak:.2f} Hz\n{m_peak:.3e}{bw_str}",
                        transform=ax.transAxes,
                        ha=ha, va="top",
                        fontsize=7.5,
                        bbox=dict(boxstyle="round,pad=0.3",
                                  fc="white", alpha=0.8, ec=color),
                        color=color)

        marker = {"vline": vline, "annot": annot,
                  "span": span,
                  "f": f_peak, "m": m_peak,
                  "bw": bw, "f_lo": f_lo, "f_hi": f_hi,
                  "fdata": f_arr, "mdata": m_arr,
                  "label": label, "ax": ax, "color": color}
        self._markers.append(marker)

    def _update_marker(self, idx: int, freq: float) -> None:
        """Move marker to freq, update annotation, BW line and table."""
        mk = self._markers[idx]
        try:
            xlo, xhi = mk["ax"].get_xlim()
        except Exception:
            xlo, xhi = 0.5, 25.0
        freq = max(xlo, min(xhi, freq))
        mag  = self._interp_mag(mk, freq)

        # Update vertical line
        mk["vline"].set_xdata([freq, freq])

        # Recalculate bandwidth at new cursor position
        bw, f_lo, f_hi = self._compute_bandwidth(
            mk["fdata"], mk["mdata"], freq, mag)
        mk["bw"] = bw; mk["f_lo"] = f_lo; mk["f_hi"] = f_hi

        # Update half-power shaded span (no horizontal line)
        if mk.get("span") is not None:
            try:
                mk["span"].remove()
            except Exception:
                pass
        if bw is not None:
            mk["span"] = mk["ax"].axvspan(
                f_lo, f_hi, alpha=0.12, color=mk["color"])
        else:
            mk["span"] = None

        # Update annotation (reposition to whichever corner stays clear of
        # the curve as the marker moves)
        ann_x, ha = self._annot_corner(mk["ax"], freq)
        mk["annot"].set_position((ann_x, 0.97))
        mk["annot"].set_ha(ha)
        bw_str = f"\nBW={bw:.2f}Hz" if bw is not None else ""
        mk["annot"].set_text(f"{freq:.2f} Hz\n{mag:.3e}{bw_str}")

        mk["f"] = freq; mk["m"] = mag
        self._refresh_table()
        self.canvas.draw_idle()

    def _refresh_table(self) -> None:
        """Update the peak-markers summary panel."""
        lines = [f"{'Channel':<14} {'Freq':>7} {'BW':>7} {'Sig':>4}",
                 "-" * 36]
        for mk in self._markers:
            thresh = self._significance_threshold(
                mk["fdata"], mk["mdata"], 2.0, 20.0)
            sig  = "★" if mk["m"] > thresh else "—"
            bw_s = f"{mk['bw']:.2f}Hz" if mk.get("bw") is not None else "n/a"
            lines.append(
                f"{mk['label']:<14} {mk['f']:>6.2f}Hz {bw_s:>7} {sig:>4}")
        text = "\n".join(lines)
        self._tbl_label.config(text=text)

    def get_marker_table_text(self) -> str:
        """Return current marker positions as a formatted string for PDF reports."""
        use_welch = getattr(self, '_var_method', None)
        method = "Welch PSD" if (use_welch is None or use_welch.get() == "Welch") else "FFT"
        lines = [
            f"Spectral Peak Markers  [{method}]",
            "=" * 50,
            f"{'Channel':<16} {'Freq (Hz)':>10} {'BW (Hz)':>9} {'Sig':>14}",
            "-" * 50,
        ]
        for mk in self._markers:
            thresh = self._significance_threshold(
                mk["fdata"], mk["mdata"], 2.0, 20.0)
            sig  = "★ significant" if mk["m"] > thresh else "— n.s."
            bw_s = f"{mk['bw']:.3f}" if mk.get("bw") is not None else "n/a"
            lines.append(
                f"{mk['label']:<16} {mk['f']:>10.3f} {bw_s:>9} {sig:>14}")
        return "\n".join(lines)

    # --------- drag event handlers ---------
    def _on_press(self, event) -> None:
        if event.inaxes is None: return
        # Find closest marker line to the click (within 0.5 Hz)
        best_idx, best_dist = None, 0.5
        for i, mk in enumerate(self._markers):
            if mk["ax"] is not event.inaxes: continue
            dist = abs(mk["f"] - event.xdata)
            if dist < best_dist:
                best_dist = dist; best_idx = i
        self._drag_idx = best_idx

    def _on_release(self, event) -> None:
        self._drag_idx = None

    def _on_motion(self, event) -> None:
        if self._drag_idx is None or event.inaxes is None: return
        if self._markers[self._drag_idx]["ax"] is not event.inaxes: return
        self._update_marker(self._drag_idx, event.xdata)

    def _connect_drag(self) -> None:
        if self.canvas is None: return
        for cid in [self._drag_cid_press, self._drag_cid_release, self._drag_cid_motion]:
            if cid is not None:
                try: self.canvas.mpl_disconnect(cid)
                except Exception: pass
        self._drag_cid_press   = self.canvas.mpl_connect("button_press_event",   self._on_press)
        self._drag_cid_release = self.canvas.mpl_connect("button_release_event", self._on_release)
        self._drag_cid_motion  = self.canvas.mpl_connect("motion_notify_event",  self._on_motion)

    # --------- main redraw ---------
    def redraw(self) -> None:
        if not MATPLOTLIB_OK: return
        data = self.app_state.raw_data
        fs   = self.app_state.fs
        if data is None:
            messagebox.showinfo("No data", "Load a file in the Setup tab first."); return
        if not fs or fs <= 0:
            messagebox.showinfo("Missing Fs", "Set Fs in Setup, then Save."); return

        cols0, kinds, sides, names = self._selected_channels()
        if not cols0:
            messagebox.showinfo("No channels", "Enable channels in Setup."); return

        X = np.asarray(data, dtype=float)
        Yfull, valid = process_signals_matrix(
            raw=X, fs=fs, active_cols_0based=cols0, per_channel_kind=kinds,
            acc_lo=self.app_state.acc_bp_low, acc_hi=self.app_state.acc_bp_high,
            emg_lo=self.app_state.emg_hp, emg_hi=self.app_state.emg_lp
        )
        kinds = [kinds[i] for i in valid]
        sides = [sides[i] for i in valid]
        names = [names[i] for i in valid]

        n = len(Yfull)
        i0, i1 = self._get_selection_indices(n)
        seg = Yfull[i0:i1, :]

        left  = [j for j, s in enumerate(sides) if s == "Left"]
        right = [j for j, s in enumerate(sides) if s == "Right"]
        nrows = max(len(left), len(right))
        if nrows == 0:
            self.log("[INFO] No channels after left/right split."); return

        self.fig.clf()
        axes = self.fig.subplots(nrows, 2, squeeze=False, sharex=True)
        self._axes = []
        self._last_fft_data = []
        self._markers = []

        xlims = (2.0, 20.0)
        # Marker colours: alternate so left/right are visually distinct
        COLORS = ["#d62728", "#1f77b4", "#e377c2", "#2ca02c",
                  "#ff7f0e", "#9467bd", "#8c564b", "#17becf"]

        method = getattr(self, '_var_method', None)
        use_welch = (method is None or method.get() == "Welch")

        def add_fft(ax, y, title, kind, label, color):
            if use_welch:
                f, spec = self._compute_fft(y, fs)
                ylabel  = "PSD (u²/Hz)"
            else:
                f, spec = self._compute_fft_simple(y, fs)
                ylabel  = "Magnitude (a.u.)"

            # Restrict to 2-20 Hz
            mask  = (f >= xlims[0]) & (f <= xlims[1])
            f_v   = f[mask]
            s_v   = spec[mask]

            # Significance threshold (only meaningful for Welch PSD)
            sig_peak_f = sig_peak_p = None
            if use_welch and s_v.size > 0:
                thresh = self._significance_threshold(f, spec, *xlims)
                pk_idx = int(np.argmax(s_v))
                if s_v[pk_idx] > thresh:
                    sig_peak_f = float(f_v[pk_idx])
                    sig_peak_p = float(s_v[pk_idx])
                # Draw threshold line
                ax.axhline(thresh, color="gray", linewidth=0.8,
                           linestyle="--", alpha=0.7)
            else:
                thresh = None

            # Plot spectrum
            line_color = "green" if kind == "ACC" else color
            ax.plot(f_v, s_v, linewidth=0.9, color=line_color)

            # Mark significant peak (Welch only) — anchor the label to the
            # peak's *actual* height and nudge it a few pixels above the tip,
            # instead of a fixed spot near the bottom of the axes. A fixed
            # position sits on top of the curve whenever this channel shares
            # a Y-scale with a much bigger one and ends up looking flat.
            if sig_peak_f is not None:
                ax.axvline(sig_peak_f, color=line_color,
                           linewidth=1.0, linestyle=":", alpha=0.6)
                ax.annotate(f"★ {sig_peak_f:.2f} Hz",
                            xy=(sig_peak_f, s_v[pk_idx]), xycoords="data",
                            xytext=(0, 5), textcoords="offset points",
                            ha="center", va="bottom",
                            fontsize=7, color=line_color, fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.2",
                                      fc="white", ec=line_color, alpha=0.8),
                            annotation_clip=False)

            ax.set_title(title)
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(*xlims)
            self._axes.append(ax)
            self._last_fft_data.append((f, spec, kind))
            self._add_marker(ax, f_v, s_v, label, color)

        # Disambiguate duplicate names (e.g. two un-renamed channels both
        # falling back to the same "ACC-Left") so table/legend labels stay unique.
        _seen_lbl: Dict[str, int] = {}
        def _uniq(nm: str) -> str:
            _seen_lbl[nm] = _seen_lbl.get(nm, 0) + 1
            return nm if _seen_lbl[nm] == 1 else f"{nm} (#{_seen_lbl[nm]})"

        color_idx = 0
        for r in range(nrows):
            axL = axes[r, 0]
            if r < len(left):
                j = left[r]
                lbl = _uniq(names[j])
                add_fft(axL, seg[:, j], f"Left — {names[j]}", kinds[j],
                        lbl, COLORS[color_idx % len(COLORS)])
                color_idx += 1
            else:
                axL.axis("off"); self._axes.append(axL)
            axR = axes[r, 1]
            if r < len(right):
                j = right[r]
                lbl = _uniq(names[j])
                add_fft(axR, seg[:, j], f"Right — {names[j]}", kinds[j],
                        lbl, COLORS[color_idx % len(COLORS)])
                color_idx += 1
            else:
                axR.axis("off"); self._axes.append(axR)

        self._apply_y_scaling()
        self.fig.tight_layout()
        self.canvas.draw_idle()
        self._refresh_table()
        self._connect_drag()
        self.log("[FFT] Updated")

# ==========================
# Explore/Welch tab
# ==========================
# Purpose: Show every enabled channel in the time domain, processed per its
# kind (ACC band-pass; EMG band-pass + rectified) — same exploration controls
# as Visualization (Signal Processed/Raw, Start/Window, scroll-zoom, pan,
# Fit Y-axis, ACC/EMG band entries) — but all channels stacked in a single
# column (no Left/Right split), each row paired with a Welch PSD panel to its
# right. Two draggable vertical cursors span all time-domain rows at once;
# dragging them updates a live Δt/1/Δt readout and recomputes each channel's
# Welch PSD over the segment between them. This selection is local to this
# tab only — it does NOT touch app_state.selection_t0/t1 (the shared
# selection used by FFT/Spectrogram/TSI).
class ExploreWelchTab(ttk.Frame, LoggerMixin):
    SIGNAL_VIEW = ["Processed", "Raw"]
    CURSOR_COLORS = ("crimson", "darkorange")

    def __init__(self, parent, app_state: AppState, log_cb):
        super().__init__(parent)
        self.app_state = app_state
        self._log_cb = log_cb

        self.var_signal = tk.StringVar(value=self.SIGNAL_VIEW[0])
        self.var_t0 = tk.StringVar(value="0")
        self.var_tw = tk.StringVar(value="")  # empty = until end
        self.var_acc_lo = tk.StringVar(value=str(self.app_state.acc_bp_low))
        self.var_acc_hi = tk.StringVar(value=str(self.app_state.acc_bp_high))
        self.var_emg_lo = tk.StringVar(value=str(self.app_state.emg_hp))
        self.var_emg_hi = tk.StringVar(value=str(self.app_state.emg_lp))
        self.var_dist = tk.StringVar(value="Cursors: —")

        self._fit_y_active: bool = False

        # Local (independent) 2-cursor segment, in absolute seconds within
        # the currently loaded time window. None until placed.
        self._cursor_t: List[Optional[float]] = [None, None]
        self._dragging: Optional[int] = None

        self.fig = None
        self.canvas = None
        self._axes_time: List[Any] = []
        self._axes_welch: List[Any] = []
        self._cursor_lines: List[List[Any]] = [[], []]
        self._t: Optional[np.ndarray] = None
        self._last_seg: Optional[np.ndarray] = None
        self._last_meta: Optional[ChannelMeta] = None
        self._total_duration: float = 0.0

        self._build()
        self.var_signal.trace_add("write", lambda *_: self.redraw())

    def log(self, msg: str) -> None:
        self._log_cb(msg); _log_file(msg)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build(self) -> None:
        ctrl = ttk.Frame(self); ctrl.pack(fill="x", padx=10, pady=6)
        ttk.Label(ctrl, text="Signal:").grid(row=0, column=0, sticky="w")
        ttk.Combobox(ctrl, values=self.SIGNAL_VIEW, textvariable=self.var_signal,
                     state="readonly", width=12).grid(row=0, column=1, sticky="w", padx=4)

        ttk.Label(ctrl, text="Start (s):").grid(row=0, column=2, sticky="e", padx=(12, 2))
        ttk.Entry(ctrl, textvariable=self.var_t0, width=8).grid(row=0, column=3, sticky="w")
        ttk.Label(ctrl, text="Window (s):").grid(row=0, column=4, sticky="e", padx=(12, 2))
        ttk.Entry(ctrl, textvariable=self.var_tw, width=8).grid(row=0, column=5, sticky="w")

        ttk.Button(ctrl, text="Refresh", command=self.redraw).grid(row=0, column=6, sticky="w", padx=10)
        self._btn_fit = ttk.Button(ctrl, text="Fit Y-axis", command=self._toggle_fit_y)
        self._btn_fit.grid(row=0, column=7, sticky="w", padx=(4, 10))
        ttk.Button(ctrl, text="Clear cursors", command=self._clear_cursors).grid(row=0, column=8, sticky="w", padx=(4, 10))

        row = ttk.Frame(self); row.pack(fill="x", padx=10, pady=4)
        ttk.Label(row, text="ACC BP (Hz):").grid(row=0, column=0, sticky="w")
        ttk.Label(row, text="Low").grid(row=0, column=1, sticky="e")
        ttk.Entry(row, textvariable=self.var_acc_lo, width=7).grid(row=0, column=2, sticky="w", padx=4)
        ttk.Label(row, text="High").grid(row=0, column=3, sticky="e")
        ttk.Entry(row, textvariable=self.var_acc_hi, width=7).grid(row=0, column=4, sticky="w", padx=4)

        ttk.Label(row, text="EMG band (Hz):").grid(row=0, column=5, sticky="e", padx=(12, 2))
        ttk.Label(row, text="HP").grid(row=0, column=6, sticky="e")
        ttk.Entry(row, textvariable=self.var_emg_lo, width=7).grid(row=0, column=7, sticky="w", padx=4)
        ttk.Label(row, text="LP").grid(row=0, column=8, sticky="e")
        ttk.Entry(row, textvariable=self.var_emg_hi, width=7).grid(row=0, column=9, sticky="w", padx=4)
        ttk.Button(row, text="Apply filters", command=self._apply_filters).grid(row=0, column=10, sticky="w", padx=10)

        info = ttk.Frame(self); info.pack(fill="x", padx=10, pady=(0, 4))
        ttk.Label(info, textvariable=self.var_dist, font=("TkDefaultFont", 9, "bold")).pack(side="left")
        ttk.Label(info, text="   Arrastra las líneas verticales (izquierda) para mover los cursores.",
                  foreground="gray").pack(side="left", padx=10)

        # Packed BEFORE the expanding plot area so it always keeps its strip
        # of space, no matter how tall the figure above ends up being (it
        # grows with the number of channels).
        scroll_frame = ttk.Frame(self, relief="groove", borderwidth=1)
        scroll_frame.pack(side="bottom", fill="x", padx=8, pady=4)
        ttk.Label(scroll_frame, text="Pan:", width=4).pack(side="left", padx=(6, 2))
        self._hscroll = ttk.Scale(scroll_frame, from_=0.0, to=1.0,
                                   orient="horizontal", command=self._on_hscroll)
        self._hscroll.pack(side="left", fill="x", expand=True, padx=4, pady=3)
        ttk.Label(scroll_frame, text="scroll wheel = zoom",
                  foreground="gray", font=("TkDefaultFont", 8)).pack(side="right", padx=8)

        plot = ttk.Frame(self); plot.pack(fill="both", expand=True, padx=8, pady=(2, 0))
        if MATPLOTLIB_OK:
            # Same height as Visualization's figure (6in) at construction
            # time — a taller initial figure here made ttk.Notebook size the
            # whole app window to fit this tab even before any data was
            # loaded, pushing the bottom "Pan" scrollbar off-screen.
            self.fig = plt.Figure(figsize=(13, 6), dpi=100)
            self.canvas = FigureCanvasTkAgg(self.fig, master=plot)

            toolbar_frame = ttk.Frame(plot)
            toolbar_frame.pack(side="top", fill="x")
            NavigationToolbar2Tk(self.canvas, toolbar_frame).update()

            self.canvas.get_tk_widget().pack(fill="both", expand=True)

            self.canvas.mpl_connect("scroll_event", self._on_scroll)
            self.canvas.mpl_connect("button_press_event", self._on_press)
            self.canvas.mpl_connect("motion_notify_event", self._on_motion)
            self.canvas.mpl_connect("button_release_event", self._on_release)
        else:
            ttk.Label(plot, text="Matplotlib not available").pack(pady=20)

    # ------------------------------------------------------------------
    # Pan / zoom (time-domain column only)
    # ------------------------------------------------------------------
    def _on_scroll(self, event) -> None:
        if event.inaxes not in self._axes_time:
            return
        ax0 = self._axes_time[0]
        xlo, xhi = ax0.get_xlim()
        xrange = xhi - xlo
        if xrange <= 0:
            return
        factor = 0.75 if event.button == "up" else 1.25
        x_center = event.xdata if event.xdata is not None else (xlo + xhi) / 2
        x_center = max(xlo, min(xhi, x_center))
        new_lo = x_center - (x_center - xlo) * factor
        new_hi = x_center + (xhi - x_center) * factor
        t0 = self._t[0] if self._t is not None and len(self._t) else 0.0
        t1 = self._t[-1] if self._t is not None and len(self._t) else (xhi * 2)
        new_lo = max(t0, new_lo)
        new_hi = min(t1, new_hi)
        if new_hi - new_lo < 0.05:
            return
        for ax in self._axes_time:
            ax.set_xlim(new_lo, new_hi)
        self._sync_scrollbar(new_lo, new_hi)
        self.canvas.draw_idle()

    def _on_hscroll(self, val) -> None:
        if not self._axes_time:
            return
        ax0 = self._axes_time[0]
        xlo, xhi = ax0.get_xlim()
        win = max(xhi - xlo, 0.05)
        t0 = self._t[0] if self._t is not None and len(self._t) else 0.0
        t1 = self._t[-1] if self._t is not None and len(self._t) else win
        t_total = t1 - t0
        if t_total <= win:
            return
        pos = float(val) * (t_total - win)
        new_lo = max(t0, min(t1 - win, t0 + pos))
        new_hi = new_lo + win
        for ax in self._axes_time:
            ax.set_xlim(new_lo, new_hi)
        self.canvas.draw_idle()

    def _sync_scrollbar(self, xlo: float, xhi: float) -> None:
        if self._t is None or not len(self._t):
            return
        t0, t1 = self._t[0], self._t[-1]
        t_total = t1 - t0
        if t_total <= 0:
            return
        win = xhi - xlo
        if win >= t_total:
            self._hscroll.set(0.0)
        else:
            pos = (xlo - t0) / max(t_total - win, 1e-9)
            self._hscroll.set(max(0.0, min(1.0, pos)))

    # ------------------------------------------------------------------
    # Cursor dragging (shared across all time-domain rows)
    # ------------------------------------------------------------------
    def _on_press(self, event) -> None:
        if event.inaxes not in self._axes_time or event.xdata is None:
            self._dragging = None
            return
        xlo, xhi = event.inaxes.get_xlim()
        tol = 0.02 * max(xhi - xlo, 1e-9)
        dists = [abs(event.xdata - c) if c is not None else float("inf") for c in self._cursor_t]
        idx = int(np.argmin(dists))
        self._dragging = idx if dists[idx] <= tol else None

    def _on_motion(self, event) -> None:
        if self._dragging is None:
            return
        if event.inaxes not in self._axes_time or event.xdata is None or self._t is None or not len(self._t):
            return
        x = max(float(self._t[0]), min(float(self._t[-1]), event.xdata))
        self._cursor_t[self._dragging] = x
        self._update_cursor_visuals()
        self.canvas.draw_idle()

    def _on_release(self, event) -> None:
        if self._dragging is None:
            return
        self._dragging = None
        self._recompute_welch()
        self.canvas.draw_idle()

    def _clear_cursors(self) -> None:
        self._cursor_t = [None, None]
        self._update_cursor_visuals()
        self._recompute_welch()
        if self.canvas:
            self.canvas.draw_idle()

    def _update_cursor_visuals(self) -> None:
        for k in (0, 1):
            x = self._cursor_t[k]
            for ln in self._cursor_lines[k]:
                ln.set_xdata([x, x] if x is not None else [np.nan, np.nan])
        c0, c1 = self._cursor_t
        if c0 is not None and c1 is not None:
            dist = abs(c1 - c0)
            hz = (1.0 / dist) if dist > 1e-9 else float("nan")
            self.var_dist.set(f"Cursors: Δt = {dist:.4f} s ({dist*1000:.1f} ms)   |   1/Δt ≈ {hz:.2f} Hz")
        else:
            self.var_dist.set("Cursors: —")

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------
    def _apply_filters(self) -> None:
        try:
            self.app_state.acc_bp_low = float(self.var_acc_lo.get())
            self.app_state.acc_bp_high = float(self.var_acc_hi.get())
            self.app_state.emg_hp = float(self.var_emg_lo.get())
            self.app_state.emg_lp = float(self.var_emg_hi.get())
        except Exception:
            messagebox.showwarning("Filters", "Numeric values are required.")
            return
        self.redraw()

    def _toggle_fit_y(self) -> None:
        self._fit_y_active = not self._fit_y_active
        self._btn_fit.config(text="Fit Y-axis ✓" if self._fit_y_active else "Fit Y-axis")
        if self._axes_time:
            self._apply_y_scaling_time()
            self.canvas.draw_idle()

    def _apply_y_scaling_time(self) -> None:
        seg = self._last_seg; meta = self._last_meta
        if seg is None or meta is None or not self._axes_time:
            return
        if self._fit_y_active:
            for ax in self._axes_time:
                ax.relim(); ax.autoscale(axis='y')
            return
        for kind in ("ACC", "EMG"):
            idxs = [i for i, k in enumerate(meta.kinds) if k == kind]
            if not idxs:
                continue
            vals = seg[:, idxs]
            finite = vals[np.isfinite(vals)]
            if finite.size == 0:
                continue
            lo = float(np.min(finite)); hi = float(np.max(finite))
            if lo == hi:
                lo, hi = lo - 1.0, hi + 1.0
            pad = (hi - lo) * 0.08
            for i in idxs:
                self._axes_time[i].set_ylim(lo - pad, hi + pad)

    # ------------------------------------------------------------------
    # Welch (per channel, over the cursor-selected segment)
    # ------------------------------------------------------------------
    def _compute_welch(self, x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
        """Welch PSD — same convention as FFTTab._compute_fft (fixed
        ≤4-second segments, Hann window, density scaling), applied here to
        whatever segment currently sits between the two cursors."""
        x = np.asarray(x, dtype=float).ravel()
        n = x.size
        if n < 2 or not np.isfinite(fs) or fs <= 0:
            return np.array([0.0]), np.array([0.0])
        finite = np.isfinite(x)
        if not finite.any():
            return np.array([0.0, fs / 2.0]), np.array([0.0, 0.0])
        if not finite.all():
            idx = np.arange(n)
            x[~finite] = np.interp(idx[~finite], idx[finite], x[finite])
        if HAS_SCIPY:
            nperseg = max(8, min(int(fs * 4), n))
            f, psd = sps.welch(x - x.mean(), fs=fs, nperseg=nperseg,
                                noverlap=nperseg // 2, window="hann", scaling="density")
        else:
            nperseg = _next_pow2(min(int(fs * 4), n)); nperseg = min(nperseg, n)
            w = np.hanning(nperseg)
            seg = x[:nperseg] - x[:nperseg].mean()
            X = np.fft.rfft(seg * w)
            f = np.fft.rfftfreq(nperseg, d=1 / fs)
            psd = (np.abs(X) ** 2) / (fs * np.sum(w ** 2))
        psd = np.nan_to_num(psd, nan=0.0, posinf=0.0, neginf=0.0)
        return f, psd

    def _recompute_welch(self) -> None:
        if not self._axes_welch:
            return
        n_rows = len(self._axes_welch)
        c0, c1 = self._cursor_t
        if c0 is None or c1 is None or self._last_seg is None or self._t is None:
            for row, ax_w in enumerate(self._axes_welch):
                ax_w.clear()
                ax_w.grid(True, alpha=0.3)
                if row == 0:
                    ax_w.set_title("Welch (1–20 Hz)", fontsize=8)
                    ax_w.set_ylabel("PSD", fontsize=7)
                if row == n_rows - 1:
                    ax_w.set_xlabel("Frequency (Hz)")
                else:
                    ax_w.tick_params(labelbottom=False)
                ax_w.text(0.5, 0.5, "Place the\n2 cursors",
                          ha="center", va="center", transform=ax_w.transAxes,
                          fontsize=7, color="gray")
            self.fig.tight_layout()
            self.fig.subplots_adjust(hspace=0.08)
            return

        lo, hi = sorted((c0, c1))
        t = self._t
        j0 = int(np.searchsorted(t, lo)); j1 = int(np.searchsorted(t, hi))
        j0 = max(0, min(j0, len(t) - 1)); j1 = max(j0 + 1, min(j1, len(t)))
        fs = self.app_state.fs
        seg = self._last_seg; meta = self._last_meta
        # No per-row title/x-label repeated on every panel (the matching
        # time-domain row already names the channel) — only the top row
        # gets a column header, and only the bottom row gets the frequency
        # axis labeled, so the vertical space goes to the traces instead.
        for row, ax_w in enumerate(self._axes_welch):
            ax_w.clear()
            x = seg[j0:j1, row]
            f, psd = self._compute_welch(x, fs)
            mask = (f >= 1.0) & (f <= 20.0)
            if not np.any(mask):
                mask = np.ones_like(f, dtype=bool)
            color = "green" if meta.kinds[row] == "ACC" else None
            ax_w.plot(f[mask], psd[mask], linewidth=0.9, color=color)
            ax_w.set_xlim(1.0, 20.0)
            ax_w.grid(True, alpha=0.3)
            if row == 0:
                ax_w.set_title("Welch (1–20 Hz)", fontsize=8)
                ax_w.set_ylabel("PSD", fontsize=7)
            if row == n_rows - 1:
                ax_w.set_xlabel("Frequency (Hz)")
            else:
                ax_w.tick_params(labelbottom=False)
        self.fig.tight_layout()
        self.fig.subplots_adjust(hspace=0.08)

    # ------------------------------------------------------------------
    # Main redraw
    # ------------------------------------------------------------------
    def redraw(self) -> None:
        if not MATPLOTLIB_OK:
            return
        if self.app_state.raw_data is None or not self.app_state.fs or self.app_state.fs <= 0:
            messagebox.showinfo("Info", "Load data and set Fs in Setup; then Save.")
            return

        use_processed = (self.var_signal.get() == "Processed")
        try:
            Yv, meta = _get_flat_order_matrix(self.app_state, use_processed=use_processed)
        except Exception as e:
            messagebox.showerror("Explore/Welch", str(e))
            return
        if Yv.shape[1] == 0:
            messagebox.showinfo("No channels", "Enable channels in Setup.")
            return

        fs = self.app_state.fs
        n = Yv.shape[0]
        t_total_all = n / fs

        try:
            t0 = float(self.var_t0.get())
        except Exception:
            t0 = 0.0
        tw_str = self.var_tw.get().strip()
        try:
            tw = float(tw_str) if tw_str else None
        except Exception:
            tw = None
        t0 = max(0.0, min(t0, max(t_total_all - 1e-6, 0.0)))
        t1 = t_total_all if (tw is None or tw <= 0) else min(t_total_all, t0 + tw)
        i0 = int(round(t0 * fs)); i1 = int(round(t1 * fs))
        i0 = max(0, min(i0, n - 1)); i1 = max(i0 + 1, min(i1, n))

        t = np.arange(i0, i1) / fs
        seg = Yv[i0:i1, :]

        self._t = t
        self._last_seg = seg
        self._last_meta = meta
        self._total_duration = float(t[-1] - t[0]) if len(t) else 0.0

        n_ch = Yv.shape[1]
        # Capped (not proportional to n_ch without bound): letting this
        # figure grow past ~6-7in tall made ttk.Notebook size the whole app
        # window to fit it (Notebook sizes itself to its largest tab, even
        # one that isn't currently shown), which pushed the bottom "Pan"
        # scrollbar off-screen in every tab, not just this one.
        fig_h = min(7.0, max(6.0, 0.8 * n_ch))
        self.fig.set_size_inches(13, fig_h, forward=False)
        self.fig.clf()
        # Time-domain column gets 80% of the width, Welch column the
        # remaining 20% (width_ratios=[4, 1]). Rows are packed tight
        # (hspace) and only the bottom row carries a time-axis label, so
        # the vertical space that used to go to per-row chrome goes to the
        # traces instead — closer to how Spike2 stacks channels.
        axes = self.fig.subplots(n_ch, 2, squeeze=False,
                                  gridspec_kw={'width_ratios': [4, 1], 'hspace': 0.08})
        self._axes_time = []
        self._axes_welch = []
        self._cursor_lines = [[], []]

        COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e",
                  "#9467bd", "#8c564b", "#e377c2", "#17becf"]

        for row in range(n_ch):
            ax_t = axes[row, 0]; ax_w = axes[row, 1]
            kind = meta.kinds[row]; name = meta.labels[row]
            color = "green" if kind == "ACC" else COLORS[row % len(COLORS)]
            ax_t.plot(t, seg[:, row], linewidth=0.8, color=color)
            ax_t.set_ylabel(name, fontsize=8)
            ax_t.grid(True, alpha=0.3)
            if len(t):
                ax_t.set_xlim(t[0], t[-1] if t[-1] > t[0] else t[0] + 1.0)
            if row == n_ch - 1:
                ax_t.set_xlabel("Time (s)")
            else:
                ax_t.tick_params(labelbottom=False)
            self._axes_time.append(ax_t)

            # Welch axes' title/labels are (re)drawn by _recompute_welch()
            # right after this loop, which clear()s each one first — only
            # bookkeeping happens here.
            self._axes_welch.append(ax_w)

            for k in (0, 1):
                ln = ax_t.axvline(np.nan, color=self.CURSOR_COLORS[k], linewidth=1.3, linestyle="--")
                self._cursor_lines[k].append(ln)

        # Keep existing cursor positions if still within the new window;
        # otherwise (first draw, or the window moved past them) place two
        # sensible defaults at 25%/75% so there's always something to drag.
        valid_range = len(t) > 0
        c0, c1 = self._cursor_t
        in_range = (valid_range and c0 is not None and c1 is not None and
                    t[0] <= c0 <= t[-1] and t[0] <= c1 <= t[-1])
        if valid_range and not in_range:
            span = t[-1] - t[0]
            self._cursor_t = [t[0] + 0.25 * span, t[0] + 0.75 * span]
        elif not valid_range:
            self._cursor_t = [None, None]

        self._apply_y_scaling_time()
        self._update_cursor_visuals()
        self._recompute_welch()
        self.fig.tight_layout()
        self.fig.subplots_adjust(hspace=0.08)
        self.canvas.draw_idle()
        if len(t):
            self._sync_scrollbar(t[0], t[-1] if t[-1] > t[0] else t[0] + 1.0)

# ==========================
# TSI tab (Tremor Stability Index)
# ==========================
class TSITab(ttk.Frame):
    """
    Tremor Stability Index — di Biase et al., Brain 2017.

    Algorithm (ACC channel, per paper):
      1. High-pass filter at 0.1 Hz (remove drift)
      2. Find dominant peak fc in 2–9 Hz (Welch PSD)
      3. Band-pass filter at [fc-2, fc+2] Hz (3rd-order Butterworth, zero-phase)
      4. Detect positive zero-crossings → intervals Tn
      5. Instantaneous frequency: fn = 1/Tn
      6. Cycle-by-cycle frequency change: Δf = fn − fn+1
      7. TSI = IQR(Δf) = Q3 − Q1

    Plots (replicating Figure 1B of the paper):
      • Left  : histogram of Δf distribution
      • Right : scatter plot Δf vs f (instantaneous frequency)
        The IQR band is shown as horizontal dashed lines.

    Cut-off (paper): TSI ≤ 1.05 → Parkinson's disease
                     TSI > 1.05 → Essential tremor
    Only the numeric value is reported; clinical interpretation is left to the user.
    """

    F_PEAK_RANGE = (2.0, 9.0)   # Hz — search range for dominant tremor frequency
    HP_FC        = 0.1           # Hz — high-pass cutoff for drift removal
    BP_HW        = 2.0           # Hz — half-width of band-pass around fc
    TSI_CUTOFF   = 1.05          # paper cut-off value (informational only)

    def __init__(self, parent, app_state: AppState, log_cb):
        super().__init__(parent)
        self.app_state = app_state
        self._log_cb   = log_cb
        self.fig       = None
        self.canvas    = None
        self._build()

    def log(self, msg: str) -> None:
        self._log_cb(msg); _log_file(msg)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build(self) -> None:
        ctrl = ttk.Frame(self); ctrl.pack(fill="x", padx=10, pady=6)

        ttk.Button(ctrl, text="Compute TSI", command=self.redraw).pack(side="left", padx=5)

        ttk.Label(ctrl, text="Channel:").pack(side="left", padx=(10, 2))
        self._var_chan = tk.StringVar(value="ACC (all)")
        self._chan_cb  = ttk.Combobox(ctrl, textvariable=self._var_chan,
                                       state="readonly", width=18)
        self._chan_cb.pack(side="left", padx=2)

        ttk.Label(ctrl, text="  Uses Selection if active",
                  foreground="gray").pack(side="left", padx=8)

        # Results panel (right of plot)
        main = ttk.Frame(self); main.pack(fill="both", expand=True, padx=8, pady=4)
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=0)
        main.rowconfigure(0, weight=1)

        plot_frame = ttk.Frame(main)
        plot_frame.grid(row=0, column=0, sticky="nsew")

        if MATPLOTLIB_OK:
            self.fig    = plt.Figure(figsize=(10, 4), dpi=100)
            self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
            self.canvas.get_tk_widget().pack(fill="both", expand=True)
            NavigationToolbar2Tk(self.canvas, plot_frame).update()
        else:
            ttk.Label(plot_frame, text="Matplotlib not available").pack(pady=20)

        # Results table
        res_frame = ttk.LabelFrame(main, text="TSI results", width=200)
        res_frame.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        res_frame.grid_propagate(False)

        self._res_text = tk.Text(res_frame, width=24, font=("Courier New", 9),
                                  state="disabled", relief="flat", bg="#f5f5f5",
                                  wrap="none")
        self._res_text.pack(fill="both", expand=True, padx=4, pady=4)

    def _update_channel_list(self) -> None:
        """Populate channel combobox from current app_state channel config."""
        options = ["ACC (all)", "EMG (all)"]
        for i in range(8):
            cfg = self.app_state.channel_config.get(
                i, ChannelConfig(enabled=False, col_1based=i+1))
            if cfg.enabled:
                options.append(f"Ch{i+1} — {_channel_display_name(cfg, i)}")
        self._chan_cb["values"] = options
        if self._var_chan.get() not in options:
            self._var_chan.set("ACC (all)")

    # ------------------------------------------------------------------
    # Signal processing
    # ------------------------------------------------------------------
    def _butter_bp(self, x: np.ndarray, fs: float,
                   lo: float, hi: float, order: int = 3) -> np.ndarray:
        """Zero-phase Butterworth band-pass."""
        if not HAS_SCIPY:
            raise RuntimeError("SciPy required for TSI computation.")
        nyq = fs / 2.0
        lo  = max(lo, 0.05)
        hi  = min(hi, nyq * 0.99)
        if lo >= hi:
            raise ValueError(f"Invalid band [{lo:.2f}, {hi:.2f}] Hz.")
        sos = sps.butter(order, [lo / nyq, hi / nyq], btype="band", output="sos")
        return sps.sosfiltfilt(sos, x)

    def _butter_hp(self, x: np.ndarray, fs: float,
                   fc: float, order: int = 3) -> np.ndarray:
        """Zero-phase Butterworth high-pass."""
        nyq = fs / 2.0
        sos = sps.butter(order, fc / nyq, btype="high", output="sos")
        return sps.sosfiltfilt(sos, x)

    def _peak_frequency(self, x: np.ndarray, fs: float) -> float:
        """Dominant frequency in F_PEAK_RANGE using Welch PSD."""
        nperseg = min(len(x), max(256, int(fs * 2)))
        f, psd  = sps.welch(x, fs=fs, nperseg=nperseg)
        mask    = (f >= self.F_PEAK_RANGE[0]) & (f <= self.F_PEAK_RANGE[1])
        if not mask.any():
            return float(np.mean(self.F_PEAK_RANGE))
        return float(f[mask][np.argmax(psd[mask])])

    def _compute_tsi(self, x: np.ndarray, fs: float) -> dict:
        """
        Full TSI pipeline for one signal.
        Returns dict with keys: tsi, fc, fn, delta_f, q1, q3, n_cycles
        """
        x = np.asarray(x, dtype=float)
        # 1. Remove NaN/Inf
        bad = ~np.isfinite(x)
        if bad.any():
            idx = np.arange(len(x))
            x[bad] = np.interp(idx[bad], idx[~bad], x[~bad])

        # 2. High-pass to remove drift
        x_hp = self._butter_hp(x, fs, self.HP_FC)

        # 3. Find dominant frequency fc
        fc = self._peak_frequency(x_hp, fs)

        # 4. Band-pass around fc ±2 Hz
        lo = max(0.5, fc - self.BP_HW)
        hi = min(fs / 2.0 - 0.5, fc + self.BP_HW)
        x_bp = self._butter_bp(x_hp, fs, lo, hi)

        # 5. Positive zero-crossings (threshold at 0 with positive gradient)
        signs    = np.sign(x_bp)
        crossings = np.where((signs[:-1] <= 0) & (signs[1:] > 0))[0]
        if len(crossings) < 3:
            raise ValueError("Too few zero-crossings — check signal/filter settings.")

        # 6. Instantaneous frequencies fn = 1/Tn
        intervals = np.diff(crossings) / fs          # seconds
        fn        = 1.0 / intervals                  # Hz

        # 7. Δf = fn − fn+1
        delta_f = fn[:-1] - fn[1:]

        # 8. TSI = IQR(Δf)
        q1  = float(np.percentile(delta_f, 25))
        q3  = float(np.percentile(delta_f, 75))
        tsi = q3 - q1

        # Mid-cycle frequency (average of consecutive fn pairs) for scatter plot
        f_mid = (fn[:-1] + fn[1:]) / 2.0

        return {
            "tsi":     tsi,
            "fc":      fc,
            "fn":      fn,
            "f_mid":   f_mid,
            "delta_f": delta_f,
            "q1":      q1,
            "q3":      q3,
            "n_cycles": len(delta_f),
        }

    # ------------------------------------------------------------------
    # Channel selection
    # ------------------------------------------------------------------
    def _get_channels(self) -> List[Tuple[str, np.ndarray]]:
        """
        Return list of (label, signal_array) according to the combobox selection.
        Uses processed signals (HP/BP already applied) for EMG; raw for ACC
        (paper uses raw ACC, only trend-corrected internally by _compute_tsi).
        """
        raw = self.app_state.raw_data
        fs  = self.app_state.fs
        if raw is None or not fs:
            return []

        X   = np.asarray(raw, dtype=float)
        n   = X.shape[0]
        i0, i1 = _get_selection_indices(self.app_state, n)
        X   = X[i0:i1, :]

        sel = self._var_chan.get()
        channels = []

        for i in range(8):
            cfg = self.app_state.channel_config.get(
                i, ChannelConfig(enabled=False, col_1based=i+1))
            if not cfg.enabled:
                continue
            col = cfg.col_1based - 1
            if col >= X.shape[1]:
                continue
            label = f"Ch{i+1} — {_channel_display_name(cfg, i)}"

            include = (
                sel == "ACC (all)"  and cfg.kind == "ACC" or
                sel == "EMG (all)"  and cfg.kind == "EMG" or
                sel.startswith(f"Ch{i+1} ")
            )
            if include:
                channels.append((label, X[:, col]))

        return channels

    # ------------------------------------------------------------------
    # Redraw
    # ------------------------------------------------------------------
    def redraw(self) -> None:
        if not MATPLOTLIB_OK:
            return
        if self.app_state.raw_data is None or not self.app_state.fs:
            messagebox.showinfo("No data", "Load a file in Setup first.")
            return

        self._update_channel_list()
        channels = self._get_channels()
        if not channels:
            messagebox.showinfo("No channels",
                                "No channels match the selection. "
                                "Check Setup channel configuration.")
            return

        fs = self.app_state.fs

        # Compute TSI for each channel
        results = []
        for label, sig in channels:
            try:
                r = self._compute_tsi(sig, fs)
                r["label"] = label
                results.append(r)
                self.log(f"[TSI] {label}: TSI={r['tsi']:.4f}  fc={r['fc']:.2f} Hz  "
                         f"n_cycles={r['n_cycles']}")
            except Exception as e:
                self.log(f"[TSI] {label}: ERROR — {e}")

        if not results:
            messagebox.showerror("TSI", "Could not compute TSI for any channel.")
            return

        # Store for session snapshot
        self._last_results = results

        # ------ Plot ------
        self.fig.clf()
        n_ch  = len(results)
        # 2 columns per channel: histogram (left) + scatter (right)
        # Layout: n_ch rows × 2 columns
        axes  = self.fig.subplots(n_ch, 2, squeeze=False)

        COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e",
                  "#9467bd", "#8c564b", "#e377c2", "#17becf"]

        for row, r in enumerate(results):
            color = COLORS[row % len(COLORS)]
            df    = r["delta_f"]
            fm    = r["f_mid"]
            tsi   = r["tsi"]
            q1, q3 = r["q1"], r["q3"]

            # ---- Left: Δf histogram (replicating paper Fig 2) ----
            ax_h = axes[row, 0]
            ax_h.hist(df, bins=40, color=color, alpha=0.75, edgecolor="none")
            ax_h.axvline(q1, color="k", linestyle="--", linewidth=1, alpha=0.7)
            ax_h.axvline(q3, color="k", linestyle="--", linewidth=1, alpha=0.7)
            ax_h.axvspan(q1, q3, alpha=0.15, color="k")
            ax_h.set_xlabel("Δf (Hz)")
            ax_h.set_ylabel("Count")
            ax_h.set_title(f"{r['label']}  —  Δf distribution")
            ax_h.text(0.97, 0.95, f"TSI = {tsi:.3f}", transform=ax_h.transAxes,
                      ha="right", va="top", fontsize=9,
                      bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
            ax_h.grid(True, alpha=0.25)

            # ---- Right: Δf vs f scatter (replicating paper Fig 1B) ----
            ax_s = axes[row, 1]
            ax_s.scatter(fm, df, s=8, color=color, alpha=0.5, linewidths=0)
            ax_s.axhline(q1, color="k", linestyle="--", linewidth=1, alpha=0.7)
            ax_s.axhline(q3, color="k", linestyle="--", linewidth=1, alpha=0.7)
            ax_s.axhspan(q1, q3, alpha=0.1, color="k")
            ax_s.axhline(0, color="gray", linewidth=0.8, alpha=0.5)
            ax_s.set_xlabel("f  (Hz)")
            ax_s.set_ylabel("Δf (Hz)")
            ax_s.set_title(f"{r['label']}  —  Δf vs f")
            ax_s.text(0.97, 0.95, f"fc = {r['fc']:.2f} Hz", transform=ax_s.transAxes,
                      ha="right", va="top", fontsize=9,
                      bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
            ax_s.grid(True, alpha=0.25)

        self.fig.tight_layout()
        self.canvas.draw_idle()

        # ------ Results table ------
        lines = [
            f"{'Channel':<22} {'TSI':>8}",
            "-" * 32,
        ]
        for r in results:
            lines.append(f"{r['label']:<22} {r['tsi']:>8.4f}")
        self._res_text.config(state="normal")
        self._res_text.delete("1.0", "end")
        self._res_text.insert("1.0", "\n".join(lines))
        self._res_text.config(state="disabled")


# --------------------
# Entry point
# ==========================
# PDF Report Export
# ==========================
class ReportDialog(tk.Toplevel):
    """
    Modal dialog for exporting a PDF clinical report.

    Patient info  : Name, Age, ID/DNI, Maneuver
    Sections      : checkboxes — Time Series, FFT, Spectrogram, TSI
    Output format : PDF via matplotlib.backends.backend_pdf.PdfPages
                    (no extra dependencies beyond matplotlib)
    """

    SECTIONS = [
        ("time_series", "Time Series"),
        ("fft",         "FFT"),
        ("spectrogram", "Spectrogram"),
        ("tsi",         "TSI"),
    ]

    def __init__(self, parent, app_state, viz, fft_tab, spec_tab, tsi_tab, log_cb):
        super().__init__(parent)
        self.title("Export Report")
        self.resizable(False, False)
        self.grab_set()          # modal
        self.transient(parent)

        self.app_state = app_state
        self.viz       = viz
        self.fft_tab   = fft_tab
        self.spec_tab  = spec_tab
        self.tsi_tab   = tsi_tab
        self.log_cb    = log_cb

        self._build()
        self.wait_window(self)

    # ------------------------------------------------------------------
    def _build(self) -> None:
        P = dict(padx=10, pady=5)

        # Patient information
        info_lf = ttk.LabelFrame(self, text="Patient information")
        info_lf.pack(fill="x", padx=14, pady=(12, 6))

        self._vars = {}
        for row, (label, key) in enumerate([
            ("Patient name", "name"),
            ("Age",          "age"),
            ("ID / DNI",     "pid"),
            ("Maneuver",     "maneuver"),
        ]):
            ttk.Label(info_lf, text=label + ":").grid(
                row=row, column=0, sticky="w", **P)
            var = tk.StringVar()
            self._vars[key] = var
            ttk.Entry(info_lf, textvariable=var, width=34).grid(
                row=row, column=1, sticky="we", **P)
        info_lf.columnconfigure(1, weight=1)

        # Report sections
        sec_lf = ttk.LabelFrame(self, text="Report sections")
        sec_lf.pack(fill="x", padx=14, pady=6)

        self._checks = {}
        defaults = {"time_series": True, "fft": True,
                    "spectrogram": False, "tsi": False}
        for i, (key, label) in enumerate(self.SECTIONS):
            var = tk.BooleanVar(value=defaults.get(key, False))
            self._checks[key] = var
            ttk.Checkbutton(sec_lf, text=label, variable=var).grid(
                row=i // 2, column=i % 2, sticky="w", padx=14, pady=4)

        # Status + buttons
        self._status = ttk.Label(self, text="", foreground="gray")
        self._status.pack(padx=14, pady=(4, 0))

        btn = ttk.Frame(self)
        btn.pack(fill="x", padx=14, pady=(6, 14))
        ttk.Button(btn, text="Export PDF",
                   command=self._do_export).pack(side="right", padx=4)
        ttk.Button(btn, text="Cancel",
                   command=self.destroy).pack(side="right")

    # ------------------------------------------------------------------
    def _do_export(self) -> None:
        if not MATPLOTLIB_OK:
            messagebox.showerror("Error",
                                 "Matplotlib is required for PDF export.")
            return

        sections = {k: v.get() for k, v in self._checks.items()}
        if not any(sections.values()):
            messagebox.showwarning("Nothing selected",
                                   "Select at least one section.")
            return

        info = {k: v.get().strip() for k, v in self._vars.items()}

        import datetime
        stem = (info.get("name") or "report").replace(" ", "_")
        default = f"{stem}_{datetime.date.today():%Y%m%d}.pdf"

        path = filedialog.asksaveasfilename(
            title="Save PDF report",
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
            initialfile=default,
        )
        if not path:
            return

        self._status.config(text="Generating PDF…", foreground="gray")
        self.update_idletasks()

        try:
            self._generate(path, info, sections)
            self._status.config(text=f"Saved: {path}", foreground="dark green")
            self.log_cb(f"[Report] PDF saved → {path}")
            messagebox.showinfo("Done", f"Report saved:\n{path}")
            self.destroy()
        except Exception as exc:
            import traceback as _tb
            self.log_cb("[Report ERROR]\n" +
                        "".join(_tb.format_exception(type(exc), exc,
                                                     exc.__traceback__)))
            self._status.config(text=str(exc), foreground="red")
            messagebox.showerror("Export failed", str(exc))

    # ------------------------------------------------------------------
    def _make_cover(self, info: dict):
        """Return a matplotlib Figure containing patient info (A4 portrait)."""
        import datetime
        fig = plt.Figure(figsize=(8.5, 14))    # Legal portrait
        ax  = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")

        lines = [
            "TREMOR ANALYSIS REPORT",
            "",
            f"Patient name : {info.get('name') or '—'}",
            f"Age          : {info.get('age')  or '—'}",
            f"ID / DNI     : {info.get('pid')  or '—'}",
            f"Maneuver     : {info.get('maneuver') or '—'}",
            "",
            f"Date         : {datetime.date.today().strftime('%d %B %Y')}",
        ]
        ax.text(0.5, 0.65, "\n".join(lines),
                ha="center", va="center",
                transform=ax.transAxes,
                fontsize=14, family="monospace",
                linespacing=1.8,
                bbox=dict(boxstyle="round,pad=1", fc="#f0f4f8", ec="#888"))
        return fig

    def _capture_fig(self, source_fig, title: str):
        """Snapshot a live matplotlib Figure into a new A4-landscape Figure."""
        import io
        import matplotlib.image as mpimg

        if source_fig is None:
            return None

        buf = io.BytesIO()
        try:
            source_fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        except Exception:
            return None
        buf.seek(0)

        img = mpimg.imread(buf)
        fig = plt.Figure(figsize=(14, 8.5))    # Legal landscape
        ax  = fig.add_axes([0.01, 0.05, 0.98, 0.88])
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(title, fontsize=13, pad=6)
        return fig

    def _text_to_fig(self, text: str, title: str = ""):
        """Render a plain-text table as a Legal-portrait figure page."""
        fig = plt.Figure(figsize=(8.5, 14))
        ax  = fig.add_axes([0.08, 0.1, 0.84, 0.82])
        ax.axis("off")
        if title:
            ax.set_title(title, fontsize=12, pad=10)
        ax.text(0.0, 1.0, text,
                transform=ax.transAxes,
                ha="left", va="top",
                fontsize=9, family="monospace",
                linespacing=1.6)
        return fig

    def _generate(self, path: str, info: dict, sections: dict) -> None:
        from matplotlib.backends.backend_pdf import PdfPages

        tab_figs = {
            "time_series": (self.viz.fig,      "Time Series"),
            "fft":         (self.fft_tab.fig,  "FFT — Processed signals"),
            "spectrogram": (self.spec_tab.fig, "Spectrogram"),
            "tsi":         (self.tsi_tab.fig,  "Tremor Stability Index (TSI)"),
        }

        with PdfPages(path) as pdf:
            # Cover page
            cover = self._make_cover(info)
            pdf.savefig(cover, bbox_inches="tight")
            plt.close(cover)

            # Selected sections
            for key, (label, _) in [(k, v) for k, v in
                                     [(s[0], (s[0], s[1])) for s in self.SECTIONS]]:
                if not sections.get(key):
                    continue
                src_fig, title = tab_figs.get(key, (None, key))
                page = self._capture_fig(src_fig, title)
                if page is None:
                    self.log_cb(f"[Report] '{title}' skipped — "
                                f"no figure (run that tab first).")
                    continue
                pdf.savefig(page, bbox_inches="tight")
                plt.close(page)

                # After FFT figure: add marker frequency table
                if key == "fft" and hasattr(self.fft_tab, 'get_marker_table_text'):
                    marker_text = self.fft_tab.get_marker_table_text()
                    if marker_text:
                        tbl_page = self._text_to_fig(
                            marker_text, "Spectral Peak Markers")
                        pdf.savefig(tbl_page, bbox_inches="tight")
                        plt.close(tbl_page)

            # Metadata
            d = pdf.infodict()
            d["Title"]   = "Tremor Analysis Report"
            d["Subject"] = (f"Patient: {info.get('name') or '—'} | "
                            f"Maneuver: {info.get('maneuver') or '—'}")


# ==========================
# Session Tab (multi-maneuver)
# ==========================
class Maneuver:
    """Holds data + metadata for one maneuver recording."""
    def __init__(self, label: str, path: str,
                 data: np.ndarray, fs: float,
                 channel_config: dict,
                 fig_viz=None, fig_fft=None, fig_tsi=None,
                 tsi_results=None):
        self.label          = label
        self.path           = path
        self.data           = data
        self.fs             = fs
        self.channel_config = channel_config   # copy of AppState.channel_config
        self.fig_viz        = fig_viz
        self.fig_fft        = fig_fft
        self.fig_tsi        = fig_tsi
        self.tsi_results    = tsi_results      # list of TSI result dicts


class SessionTab(ttk.Frame):
    """
    Multi-maneuver session manager.

    Workflow:
      1. Load each maneuver via the Setup tab as usual.
      2. Click "Add current as maneuver" — snapshots the current data + figures.
      3. Repeat for each maneuver (rest, posture, kinetic, etc.).
      4. Reorder or remove maneuvers with the Up/Down/Delete buttons.
      5. Click "Export Session (Word)" to generate a multi-maneuver report.

    The session report (Word .docx) includes:
      - The Comparison tab's cross-maneuver overlay view (optional)
      - Per maneuver: Time series (always), FFT (always, auto-computed when
        the maneuver is added), TSI (optional, if Compute TSI was run)
    """

    def __init__(self, parent, app_state: AppState,
                 viz, fft_tab, tsi_tab, log_cb,
                 spectrogram_tab=None, coherence_tab=None):
        super().__init__(parent)
        self.app_state      = app_state
        self.viz            = viz
        self.fft_tab        = fft_tab
        self.tsi_tab        = tsi_tab
        self.spectrogram_tab = spectrogram_tab
        self.coherence_tab   = coherence_tab
        self._log_cb   = log_cb
        self.maneuvers: List[Maneuver] = []
        self._build()

    def log(self, msg: str) -> None:
        self._log_cb(msg); _log_file(msg)

    def _build(self) -> None:
        # Top controls
        ctrl = ttk.Frame(self); ctrl.pack(fill="x", padx=10, pady=6)

        ttk.Button(ctrl, text="➕  Add current",
                   command=self._add_current).pack(side="left", padx=4)
        ttk.Button(ctrl, text="🗑  Remove",
                   command=self._remove_selected).pack(side="left", padx=4)
        ttk.Button(ctrl, text="▲",
                   command=lambda: self._move(-1)).pack(side="left", padx=2)
        ttk.Button(ctrl, text="▼",
                   command=lambda: self._move(1)).pack(side="left", padx=2)
        ttk.Button(ctrl, text="📄  Export Session (Word)",
                   command=self._export).pack(side="right", padx=8)

        # Main area: list (left) + comparison plot (right)
        main = ttk.Frame(self); main.pack(fill="both", expand=True, padx=8, pady=(0, 4))
        main.columnconfigure(0, weight=0, minsize=340)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        # ── Left: maneuver list ──────────────────────────────────
        list_frame = ttk.LabelFrame(main, text="Maneuvers")
        list_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        cols = ("label", "duration", "tsi")
        self._tree = ttk.Treeview(list_frame, columns=cols,
                                   show="headings", selectmode="browse",
                                   height=20)
        self._tree.heading("label",    text="Maneuver")
        self._tree.heading("duration", text="Duration")
        self._tree.heading("tsi",      text="TSI")
        self._tree.column("label",    width=160)
        self._tree.column("duration", width=72, anchor="center")
        self._tree.column("tsi",      width=80, anchor="center")

        sb = ttk.Scrollbar(list_frame, orient="vertical",
                           command=self._tree.yview)
        self._tree.configure(yscrollcommand=sb.set)
        self._tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self._tree.bind("<Double-1>", self._rename_selected)

        ttk.Label(list_frame, text="Double-click to rename",
                  foreground="gray").pack(anchor="w", padx=4, pady=2)

        # ── Right: comparison plot ───────────────────────────────
        plot_frame = ttk.LabelFrame(main, text="Comparison view")
        plot_frame.grid(row=0, column=1, sticky="nsew")

        if MATPLOTLIB_OK:
            self.fig    = plt.Figure(figsize=(9, 6), dpi=100)
            self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
            self.canvas.get_tk_widget().pack(fill="both", expand=True)
            NavigationToolbar2Tk(self.canvas, plot_frame).update()
        else:
            self.fig = self.canvas = None
            ttk.Label(plot_frame, text="Matplotlib not available").pack(pady=20)

    # ------------------------------------------------------------------
    def _draw_comparison(self) -> None:
        """
        Welch PSD comparison — stacked layout:
        rows = one per maneuver, cols = one per channel (individual, not deduplicated).
        Each maneuver gets its own row of Welch PSD subplots.
        """
        if not MATPLOTLIB_OK or not self.maneuvers or self.fig is None:
            return

        COLORS = ["#1f77b4","#d62728","#2ca02c","#ff7f0e",
                  "#9467bd","#8c564b","#e377c2","#17becf"]

        def _active_channels(m: Maneuver):
            """Return all enabled channels as (col_0based, kind, side, name)."""
            return [(cfg.col_1based-1, cfg.kind, cfg.side,
                     (cfg.name.strip() if getattr(cfg, "name", "").strip() else f"{cfg.kind}-{cfg.side}"))
                    for cfg in sorted(m.channel_config.values(),
                                      key=lambda c: c.col_1based)
                    if cfg.enabled]

        def _welch_psd(sig, fs):
            """Welch PSD — same method as FFT tab."""
            from scipy import signal as _sps
            sig = np.asarray(sig, dtype=float)
            n   = len(sig)
            if n < 4: return None, None
            finite = np.isfinite(sig)
            if not finite.all():
                ii = np.arange(n)
                sig[~finite] = np.interp(ii[~finite], ii[finite], sig[finite])
            nperseg = min(int(fs * 4), n)
            f, psd = _sps.welch(sig - sig.mean(), fs=fs, nperseg=nperseg,
                                noverlap=nperseg // 2,
                                window="hann", scaling="density")
            mask = (f >= 2.0) & (f <= 20.0)
            return f[mask], psd[mask]

        # Build channel layout from first maneuver — ALL channels, not deduplicated
        # Use (col_0based, kind, side) as unique key so FCR_r and ECR_r both show
        ref_chans = _active_channels(self.maneuvers[0])
        # ch_layout: list of (kind, side, label) in column order — the label
        # uses the reference maneuver's channel name (side kept as a small
        # prefix since columns are still grouped by side).
        ch_layout = [(kind, side, f"{side}\n{name}")
                     for _, kind, side, name in ref_chans]

        n_rows = len(self.maneuvers)
        n_cols = len(ch_layout)
        if n_cols == 0: return

        self.fig.clf()
        axes = self.fig.subplots(n_rows, n_cols, squeeze=False,
                                  sharey="col", sharex=True)
        self.fig.subplots_adjust(left=0.10, right=0.97,
                                  top=0.92, bottom=0.10,
                                  hspace=0.12, wspace=0.25)

        for row, m in enumerate(self.maneuvers):
            color = COLORS[row % len(COLORS)]
            chans = _active_channels(m)

            for col, (kind, side, col_label) in enumerate(ch_layout):
                ax = axes[row, col]

                # Column header (top row only)
                if row == 0:
                    ax.set_title(col_label, fontsize=7, pad=3)

                # Row label (leftmost column only)
                if col == 0:
                    ax.set_ylabel(m.label, fontsize=8, rotation=0,
                                  ha="right", va="center", labelpad=60)

                # Match by position — nth channel of same kind+side
                # Count how many channels of this kind+side precede col in ch_layout
                nth = sum(1 for k2, s2, _ in ch_layout[:col]
                          if k2 == kind and s2 == side)
                candidates = [(c, k, s) for c, k, s, _ in chans
                              if k == kind and s == side]
                if nth >= len(candidates):
                    ax.axis("off"); continue
                col0 = candidates[nth][0]
                if col0 >= m.data.shape[1]:
                    ax.axis("off"); continue

                sig = m.data[:, col0]

                # Process signal
                try:
                    Yf, _ = process_signals_matrix(
                        raw=m.data, fs=m.fs,
                        active_cols_0based=[col0],
                        per_channel_kind=[kind],
                        acc_lo=self.app_state.acc_bp_low,
                        acc_hi=self.app_state.acc_bp_high,
                        emg_lo=self.app_state.emg_hp,
                        emg_hi=self.app_state.emg_lp,
                    )
                    sig = Yf[:, 0] if Yf.shape[1] > 0 else sig
                except Exception:
                    pass

                f, psd = _welch_psd(sig, m.fs)
                if f is None: continue

                line_color = "#2ca02c" if kind == "ACC" else color
                ax.plot(f, psd, linewidth=1.0, color=line_color, alpha=0.9)
                ax.set_xlim(2, 20)
                ax.grid(True, alpha=0.25)
                ax.tick_params(labelsize=6)

                # Maneuver label inside the plot (top-right corner)
                ax.text(0.97, 0.93, m.label,
                        transform=ax.transAxes,
                        ha="right", va="top", fontsize=7.5,
                        color=line_color, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.2",
                                  fc="white", ec=line_color, alpha=0.8))

                # X label on bottom row only
                if row == n_rows - 1:
                    ax.set_xlabel("Frequency (Hz)", fontsize=7)
                else:
                    ax.tick_params(labelbottom=False)

        self.fig.suptitle("FFT Comparison", fontsize=11)
        self.canvas.draw_idle()

    def _add_with_label(self, label: str) -> None:
        """Add current app_state as a maneuver with given label (called from Setup)."""
        if self.app_state.raw_data is None:
            messagebox.showinfo("No data", "Load a file first.")
            return

        # Auto-refresh FFT
        try:
            self.fft_tab.redraw()
        except Exception:
            pass

        import io, copy
        def _snapshot(fig):
            if fig is None: return None
            try:
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
                buf.seek(0)
                import matplotlib.image as mpimg
                return mpimg.imread(buf)
            except Exception:
                return None

        tsi_results = getattr(self.tsi_tab, '_last_results', None)
        path = self.app_state.options.get("source_path", "")

        m = Maneuver(
            label          = label,
            path           = path,
            data           = self.app_state.raw_data.copy(),
            fs             = self.app_state.fs,
            channel_config = copy.deepcopy(self.app_state.channel_config),
            fig_viz        = _snapshot(self.viz.fig),
            fig_fft        = _snapshot(self.fft_tab.fig),
            fig_tsi        = _snapshot(self.tsi_tab.fig),
            tsi_results    = tsi_results,
        )
        self.maneuvers.append(m)
        self._refresh_tree()
        self.log(f"[Comparison] Added: '{label}'  "
                 f"({m.data.shape[0]/m.fs:.1f}s)")

    # ------------------------------------------------------------------
    def _add_current(self) -> None:
        """Called from the Comparison tab's own button."""
        if self.app_state.raw_data is None:
            messagebox.showinfo("No data", "Load a file in Setup first.")
            return
        path = self.app_state.options.get("source_path", "")
        default = path.split("/")[-1].split("\\")[-1].rsplit(".", 1)[0] or \
                  f"Maneuver {len(self.maneuvers)+1}"
        label = simpledialog.askstring(
            "Maneuver name", "Name for this recording:",
            initialvalue=default, parent=self)
        if label:
            self._add_with_label(label.strip() or default)

    def _refresh_tree(self) -> None:
        self._tree.delete(*self._tree.get_children())
        for i, m in enumerate(self.maneuvers):
            dur = f"{m.data.shape[0]/m.fs:.1f}s"
            tsi_str = ""
            if m.tsi_results:
                vals = [f"{r['tsi']:.2f}" for r in m.tsi_results]
                tsi_str = ", ".join(vals)
            self._tree.insert("", "end", iid=str(i),
                              values=(m.label, dur, tsi_str))
        self._draw_comparison()
        # Update the toolbar maneuver selector
        app = self.winfo_toplevel()
        if hasattr(app, "refresh_maneuver_selector"):
            app.refresh_maneuver_selector()

    def _selected_idx(self) -> Optional[int]:
        sel = self._tree.selection()
        if not sel: return None
        return int(sel[0])

    def _remove_selected(self) -> None:
        idx = self._selected_idx()
        if idx is None:
            messagebox.showinfo("Nothing selected", "Select a maneuver first.")
            return
        removed = self.maneuvers.pop(idx)
        self._refresh_tree()
        self.log(f"[Session] Removed maneuver: '{removed.label}'")

    def _move(self, direction: int) -> None:
        idx = self._selected_idx()
        if idx is None: return
        new_idx = idx + direction
        if new_idx < 0 or new_idx >= len(self.maneuvers): return
        self.maneuvers[idx], self.maneuvers[new_idx] = \
            self.maneuvers[new_idx], self.maneuvers[idx]
        self._refresh_tree()
        self._tree.selection_set(str(new_idx))

    def _rename_selected(self, event=None) -> None:
        idx = self._selected_idx()
        if idx is None: return
        m = self.maneuvers[idx]
        new_label = simpledialog.askstring(
            "Rename maneuver",
            "New name:",
            initialvalue=m.label,
            parent=self,
        )
        if new_label and new_label.strip():
            m.label = new_label.strip()
            self._refresh_tree()
            self._tree.selection_set(str(idx))

    # ------------------------------------------------------------------
    def _export(self) -> None:
        if not self.maneuvers:
            messagebox.showinfo("Empty session",
                                "Add at least one maneuver before exporting.")
            return
        SessionReportDialog(self, self.maneuvers, log_cb=self.log,
                             comparison_fig=self.fig,
                             app_state=self.app_state,
                             spectrogram_tab=self.spectrogram_tab,
                             coherence_tab=self.coherence_tab)


# ==========================
# Session PDF Report
# ==========================
class SessionReportDialog(tk.Toplevel):
    """Export a multi-maneuver Word (.docx) report."""

    def __init__(self, parent, maneuvers: List[Maneuver], log_cb, comparison_fig=None,
                 app_state=None, spectrogram_tab=None, coherence_tab=None):
        super().__init__(parent)
        self.title("Export Session Report")
        self.resizable(False, False)
        self.grab_set()
        self.transient(parent)

        self.maneuvers      = maneuvers
        self.log_cb         = log_cb
        self.comparison_fig = comparison_fig   # SessionTab's cross-maneuver comparison figure
        self.app_state       = app_state        # shared AppState (temporarily repointed per maneuver)
        self.spectrogram_tab = spectrogram_tab  # live SpectrogramTab, reused to render each maneuver
        self.coherence_tab   = coherence_tab    # live CoherenceTab, reused to render each maneuver
        self._build()
        self.wait_window(self)

    def _build(self) -> None:
        P = dict(padx=10, pady=5)

        # Patient info
        info_lf = ttk.LabelFrame(self, text="Patient information")
        info_lf.pack(fill="x", padx=14, pady=(12, 6))

        self._vars = {}
        for row, (label, key) in enumerate([
            ("Patient name", "name"),
            ("Age",          "age"),
            ("ID / DNI",     "pid"),
        ]):
            ttk.Label(info_lf, text=label + ":").grid(
                row=row, column=0, sticky="w", **P)
            var = tk.StringVar()
            self._vars[key] = var
            ttk.Entry(info_lf, textvariable=var, width=34).grid(
                row=row, column=1, sticky="we", **P)
        info_lf.columnconfigure(1, weight=1)

        # Session-level section (cross-maneuver comparison view)
        cmp_lf = ttk.LabelFrame(self, text="Include session-level")
        cmp_lf.pack(fill="x", padx=14, pady=(6, 0))
        self._include_comparison = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            cmp_lf, text="Comparison view (cross-maneuver overlay, from the Comparison tab)",
            variable=self._include_comparison,
            state="normal" if self.comparison_fig is not None else "disabled"
        ).pack(anchor="w", padx=14, pady=6)

        # Section options per maneuver
        sec_lf = ttk.LabelFrame(self, text="Include per maneuver")
        sec_lf.pack(fill="x", padx=14, pady=6)

        self._checks = {}
        defaults = {"viz": True, "fft": True, "tsi": False,
                    "spectrogram": False, "coherence": False}
        labels   = {"viz": "Time Series", "fft": "FFT", "tsi": "TSI",
                    "spectrogram": "Spectrogram", "coherence": "Coherence/Cumulant (ACC L vs R)"}
        for i, (key, label) in enumerate(labels.items()):
            var = tk.BooleanVar(value=defaults[key])
            self._checks[key] = var
            state = "normal"
            if key == "spectrogram" and self.spectrogram_tab is None:
                state = "disabled"
            if key == "coherence" and self.coherence_tab is None:
                state = "disabled"
            ttk.Checkbutton(sec_lf, text=label, variable=var, state=state).grid(
                row=i // 3, column=i % 3, sticky="w", padx=14, pady=6)
        ttk.Label(sec_lf,
                  text="Coherence/Cumulant is recomputed per maneuver, defaulting to Left ACC vs Right ACC "
                       "(same rule as the Coherence/Cumulant tab).",
                  foreground="gray", wraplength=420).grid(
            row=2, column=0, columnspan=3, sticky="w", padx=14, pady=(0, 4))

        # Maneuver list preview
        prev_lf = ttk.LabelFrame(self, text="Maneuvers to include")
        prev_lf.pack(fill="x", padx=14, pady=6)
        for m in self.maneuvers:
            ttk.Label(prev_lf, text=f"• {m.label}").pack(
                anchor="w", padx=10, pady=1)

        # Status + buttons
        self._status = ttk.Label(self, text="", foreground="gray")
        self._status.pack(padx=14, pady=(4, 0))

        btn = ttk.Frame(self)
        btn.pack(fill="x", padx=14, pady=(6, 14))
        ttk.Button(btn, text="Export Word (.docx)",
                   command=self._do_export).pack(side="right", padx=4)
        ttk.Button(btn, text="Cancel",
                   command=self.destroy).pack(side="right")

    def _do_export(self) -> None:
        if not MATPLOTLIB_OK:
            messagebox.showerror("Error", "Matplotlib is required to render the figures.")
            return
        if not HAS_DOCX:
            messagebox.showerror(
                "Error",
                "python-docx is required for Word export and could not be installed "
                "automatically (see tsi_app.log). Try:\n"
                f'  "{sys.executable}" -m pip install python-docx'
            )
            return

        info     = {k: v.get().strip() for k, v in self._vars.items()}
        sections = {k: v.get() for k, v in self._checks.items()}
        include_comparison = bool(self._include_comparison.get()) and self.comparison_fig is not None

        import datetime
        stem    = (info.get("name") or "session").replace(" ", "_")
        default = f"{stem}_{datetime.date.today():%Y%m%d}.docx"

        path = filedialog.asksaveasfilename(
            title="Save Session Report",
            defaultextension=".docx",
            filetypes=[("Word document", "*.docx"), ("All files", "*.*")],
            initialfile=default,
        )
        if not path: return

        self._status.config(text="Generating Word document…", foreground="gray")
        self.update_idletasks()

        try:
            self._generate_docx(path, info, sections, include_comparison)
            self._status.config(text=f"Saved: {path}", foreground="dark green")
            self.log_cb(f"[Session] Word report saved → {path}")
            messagebox.showinfo("Done", f"Session report saved:\n{path}")
            self.destroy()
        except Exception as exc:
            import traceback as _tb
            self.log_cb("[Session ERROR]\n" +
                        "".join(_tb.format_exception(type(exc), exc,
                                                     exc.__traceback__)))
            self._status.config(text=str(exc), foreground="red")
            messagebox.showerror("Export failed", str(exc))
        finally:
            # Always put the borrowed Spectrogram/Coherence tabs back to the
            # currently loaded recording, whether the export succeeded or not.
            self._restore_live_tabs()

    def _add_fig_picture(self, document, fig_or_arr, is_array: bool) -> bool:
        """Render a matplotlib Figure or a snapshot image array to PNG and
        embed it in the Word document. Returns True on success."""
        from docx.shared import Inches
        import io as _io
        if fig_or_arr is None:
            return False
        buf = _io.BytesIO()
        try:
            if is_array:
                plt.imsave(buf, fig_or_arr, format="png", dpi=150)
            else:
                fig_or_arr.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            buf.seek(0)
            document.add_picture(buf, width=Inches(6.5))
            return True
        except Exception as e:
            self.log_cb(f"[Session] Could not embed image: {e}")
            return False

    # ------------------------------------------------------------------
    # Per-maneuver Spectrogram / Coherence rendering
    # ------------------------------------------------------------------
    # Spectrogram and Coherence are NOT snapshotted when a maneuver is added
    # (unlike Time Series / FFT / TSI) — they're recomputed here, on export,
    # directly from the maneuver's own stored data/channel_config. This is
    # done by temporarily pointing the shared AppState at that maneuver's
    # data and reusing the live SpectrogramTab/CoherenceTab redraw() methods,
    # so the math is identical to what those tabs show interactively. For
    # Coherence, re-running _prepare_matrix_and_labels() also resets the
    # channel pair to its own default (Left ACC vs Right ACC when available)
    # for EACH maneuver, regardless of whatever was last selected on-screen.
    def _snapshot_current_tab_fig(self, tab):
        import io as _io
        if tab is None or tab.fig is None:
            return None
        buf = _io.BytesIO()
        try:
            tab.fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
            buf.seek(0)
            import matplotlib.image as mpimg
            return mpimg.imread(buf)
        except Exception as e:
            self.log_cb(f"[Session] Could not snapshot {tab.__class__.__name__}: {e}")
            return None

    def _render_tab_for_maneuver(self, tab, m: "Maneuver"):
        """Temporarily repoint self.app_state at maneuver m, redraw `tab`,
        snapshot the result, then restore the original app_state. Returns
        an image array (or None if rendering failed)."""
        if tab is None or self.app_state is None:
            return None

        saved = (self.app_state.raw_data, self.app_state.fs,
                 self.app_state.channel_config,
                 self.app_state.selection_t0, self.app_state.selection_t1)
        try:
            self.app_state.raw_data       = m.data
            self.app_state.fs             = m.fs
            self.app_state.channel_config = m.channel_config
            self.app_state.selection_t0   = None
            self.app_state.selection_t1   = None

            if hasattr(tab, "_prepare_matrix_and_labels"):
                try:
                    tab._prepare_matrix_and_labels()
                except Exception as e:
                    self.log_cb(f"[Session] {tab.__class__.__name__} setup failed for "
                                f"'{m.label}': {e}")
                    return None
            try:
                tab.redraw()
            except Exception as e:
                self.log_cb(f"[Session] {tab.__class__.__name__} render failed for "
                            f"'{m.label}': {e}")
                return None

            return self._snapshot_current_tab_fig(tab)
        finally:
            (self.app_state.raw_data, self.app_state.fs,
             self.app_state.channel_config,
             self.app_state.selection_t0, self.app_state.selection_t1) = saved

    def _restore_live_tabs(self) -> None:
        """After borrowing the live Spectrogram/Coherence tabs to render each
        maneuver, refresh them once more against the ORIGINAL app_state so
        they don't keep showing the last maneuver processed during export."""
        if self.app_state is None or self.app_state.raw_data is None:
            return
        for tab in (self.spectrogram_tab, self.coherence_tab):
            if tab is None:
                continue
            try:
                if hasattr(tab, "_prepare_matrix_and_labels"):
                    tab._prepare_matrix_and_labels()
                tab.redraw()
            except Exception:
                pass

    def _generate_docx(self, path: str, info: dict, sections: dict,
                        include_comparison: bool) -> None:
        import datetime
        document = docx.Document()

        # ── Cover ────────────────────────────────────────────────────────
        document.add_heading("Tremor Analysis Report", level=0)
        document.add_heading("Session — Multiple Maneuvers", level=2)

        table = document.add_table(rows=0, cols=2)
        table.style = "Light Grid Accent 1"
        rows = [
            ("Patient name", info.get("name") or "—"),
            ("Age",          info.get("age")  or "—"),
            ("ID / DNI",     info.get("pid")  or "—"),
            ("Date",         datetime.date.today().strftime("%d %B %Y")),
        ]
        for label, value in rows:
            cells = table.add_row().cells
            cells[0].text = label
            cells[1].text = str(value)

        document.add_heading("Maneuvers", level=2)
        for i, m in enumerate(self.maneuvers):
            dur = f"{m.data.shape[0]/m.fs:.1f} s"
            document.add_paragraph(f"{i+1}. {m.label}  ({dur})", style="List Bullet")

        # ── Session-level comparison view (from the Comparison tab) ───────
        if include_comparison and self.comparison_fig is not None:
            document.add_page_break()
            document.add_heading("Comparison (cross-maneuver)", level=1)
            if not self._add_fig_picture(document, self.comparison_fig, is_array=False):
                document.add_paragraph("(Comparison view could not be rendered.)")

        # ── Per-maneuver sections ──────────────────────────────────────────
        for m in self.maneuvers:
            document.add_page_break()
            document.add_heading(m.label, level=1)
            document.add_paragraph(f"Duration: {m.data.shape[0]/m.fs:.1f} s")

            if sections.get("viz") and m.fig_viz is not None:
                document.add_heading(f"{m.label} — Time Series", level=2)
                self._add_fig_picture(document, m.fig_viz, is_array=True)

            if sections.get("fft") and m.fig_fft is not None:
                document.add_heading(f"{m.label} — FFT", level=2)
                self._add_fig_picture(document, m.fig_fft, is_array=True)

            if sections.get("tsi") and m.fig_tsi is not None:
                document.add_heading(f"{m.label} — TSI", level=2)
                self._add_fig_picture(document, m.fig_tsi, is_array=True)

            if sections.get("spectrogram") and self.spectrogram_tab is not None:
                document.add_heading(f"{m.label} — Spectrogram", level=2)
                arr = self._render_tab_for_maneuver(self.spectrogram_tab, m)
                if arr is None or not self._add_fig_picture(document, arr, is_array=True):
                    document.add_paragraph("(Spectrogram could not be generated for this maneuver.)")

            if sections.get("coherence") and self.coherence_tab is not None:
                document.add_heading(f"{m.label} — Coherence/Cumulant (default: Left ACC vs Right ACC)", level=2)
                arr = self._render_tab_for_maneuver(self.coherence_tab, m)
                if arr is None or not self._add_fig_picture(document, arr, is_array=True):
                    document.add_paragraph("(Coherence/Cumulant could not be generated for this maneuver — "
                                           "needs at least two enabled channels.)")

        document.core_properties.title   = "Tremor Analysis — Session Report"
        document.core_properties.subject = f"Patient: {info.get('name') or '—'}"
        document.save(path)


# --------------------
def main():
    if sys.platform.startswith("win"):
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    app = TSIApp()
    app.mainloop()


# Program Entry Point
# -------------------------------------------------------------------------------
# main() builds the TSIApp and starts Tk's event loop. If anything fails during
# startup, check your console/log panel for error messages. Running from a
# terminal helps expose exceptions that may be hidden when double-clicking.
if __name__ == "__main__":
    main()


# ========================= TSI MODULE (Drop-in) =========================
# This block is self-contained and can be pasted at the end of your main script.
# It implements the Tremor Stability Index (TSI) pipeline for a single 1-D signal.
# The pipeline is:
#   1) High-pass at 0.1 Hz (to remove drift/DC)
#   2) Estimate dominant peak in 2–9 Hz using Welch PSD
#   3) Band-pass centered at f_peak ± 2 Hz (clipped to valid band)
#   4) Positive zero-crossings (ZC+)
#   5) Instantaneous frequency from complete cycles (every two ZC+)
#   6) Δf = consecutive differences of instantaneous frequency
#   7) TSI = IQR(Δf) = Q3 - Q1
#
# Public API:
#   - class TSIResult: container for outputs
#   - compute_tsi(signal, fs, hp_fc=0.1, peak_range=(2.0, 9.0), bp_halfwidth=2.0) -> TSIResult
#
# Typical usage in your GUI:
#   res_left  = compute_tsi(acc_left_segment, fs)
#   res_right = compute_tsi(acc_right_segment, fs)
#   # Then display:
#   #   res_left.tsi, res_left.q1, res_left.q3
#   #   res_left.bp (band-passed signal), res_left.zc_idx (ZC positions)
#   #   histogram over res_left.delta_f with vertical lines at res_left.q1 and res_left.q3
#
# NOTE: numpy (np) and scipy.signal (butter, filtfilt, welch) are imported globally
# at the top of this file. No redundant imports are needed here.
# =======================================================================

class TSIResult:
    """
    Container for all TSI-related outputs.

    Attributes
    ----------
    tsi : float
        Tremor Stability Index (IQR of Δf).
    q1 : float
        First quartile (25th percentile) of Δf.
    q3 : float
        Third quartile (75th percentile) of Δf.
    delta_f : np.ndarray
        Consecutive differences of instantaneous frequency (Hz).
    inst_freq : np.ndarray
        Instantaneous frequency estimates (Hz) computed from full cycles.
    bp : np.ndarray
        Band-passed version of the input signal (used for ZC and inst. freq).
    zc_idx : np.ndarray
        Indices (samples) of positive zero-crossings in the band-passed signal.
    f_peak : float
        Dominant peak frequency (Hz) found between peak_range.
    """
    def __init__(self, tsi, q1, q3, delta_f, inst_freq, bp, zc_idx, f_peak):
        self.tsi = float(tsi)
        self.q1 = float(q1)
        self.q3 = float(q3)
        self.delta_f = np.asarray(delta_f, dtype=float)
        self.inst_freq = np.asarray(inst_freq, dtype=float)
        self.bp = np.asarray(bp, dtype=float)
        self.zc_idx = np.asarray(zc_idx, dtype=int)
        self.f_peak = float(f_peak)


def _butter_filter(sig, fs, kind, fc_low=None, fc_high=None, order=4):
    """
    Helper for high-pass and band-pass Butterworth filtering with filtfilt.

    Parameters
    ----------
    sig : array-like
        Input 1-D signal.
    fs : float
        Sampling rate (Hz).
    kind : {"hp", "bp"}
        Filter type: "hp" (high-pass) or "bp" (band-pass).
    fc_low : float or None
        Cutoff (hp) or lower cutoff (bp), in Hz.
    fc_high : float or None
        Upper cutoff (bp), in Hz.
    order : int
        Filter order.

    Returns
    -------
    np.ndarray
        Filtered signal.
    """
    if not HAS_SCIPY:
        raise RuntimeError("SciPy is required for TSI filtering (butter/filtfilt).")

    sig = np.asarray(sig, dtype=float)
    nyq = fs / 2.0

    if kind == "hp":
        if fc_low is None or fc_low <= 0 or fc_low >= nyq:
            raise ValueError("Invalid hp cutoff. Must be within (0, fs/2).")
        b, a = butter(order, fc_low / nyq, btype="highpass")

    elif kind == "bp":
        if (fc_low is None) or (fc_high is None):
            raise ValueError("Band-pass requires fc_low and fc_high.")
        if not (0 < fc_low < fc_high < nyq):
            raise ValueError("Invalid bp band. Must satisfy 0 < fc_low < fc_high < fs/2.")
        b, a = butter(order, [fc_low / nyq, fc_high / nyq], btype="bandpass")

    else:
        raise ValueError("kind must be 'hp' or 'bp'.")

    return filtfilt(b, a, sig)


def _estimate_peak_freq(sig, fs, fmin=2.0, fmax=9.0):
    """
    Estimate the dominant spectral peak in [fmin, fmax] using Welch.

    Notes
    -----
    - Uses adaptive nperseg to balance short/long signals.
    - Returns None if a valid peak cannot be identified.

    Returns
    -------
    float or None
        Peak frequency in Hz, or None if not found.
    """
    if not HAS_SCIPY:
        raise RuntimeError("SciPy is required for TSI peak estimation (welch).")

    n = len(sig)

    # Choose nperseg conservatively: prefer >= 1 s, cap to keep PSD smooth
    if n < fs:
        # Very short segment: use ~1/2 length but at least 256
        nperseg = max(256, int(n // 2))
    else:
        # Longer segments: around 2 s, clamped between 512 and 4096
        nperseg = min(4096, max(512, int(fs * 2)))

    f, Pxx = welch(sig, fs=fs, nperseg=nperseg)

    # Mask to the band of interest
    mask = (f >= fmin) & (f <= fmax)
    if not np.any(mask):
        return None

    p_sub = Pxx[mask]
    if np.all(p_sub == 0):
        return None

    f_sub = f[mask]
    peak_idx = int(np.argmax(p_sub))
    return float(f_sub[peak_idx])


def _positive_zero_crossings(x):
    """
    Find positive zero-crossings (ZC+): indices i where x[i-1] <= 0 and x[i] > 0.

    Returns
    -------
    np.ndarray of int
        1-D array of indices (sample positions) of ZC+ events.
    """
    x = np.asarray(x, dtype=float)

    # Convert exact zeros to a small negative to avoid ambiguous sign changes
    s = np.sign(x)
    s[s == 0] = -1.0

    # ZC+ occurs where sign goes from <= 0 to > 0
    return np.where((s[:-1] <= 0) & (s[1:] > 0))[0] + 1


def compute_tsi(signal, fs, hp_fc=0.1, peak_range=(2.0, 9.0), bp_halfwidth=2.0):
    """
    Compute the Tremor Stability Index (TSI) from a 1-D signal.

    Parameters
    ----------
    signal : array-like
        1-D array with the input signal (e.g., accelerometer trace).
    fs : float
        Sampling rate (Hz).
    hp_fc : float, optional
        High-pass cutoff frequency in Hz (default 0.1 Hz).
    peak_range : tuple(float, float), optional
        (fmin, fmax) in Hz to search for the dominant peak (default (2.0, 9.0)).
    bp_halfwidth : float, optional
        Half-width for the band-pass around the peak (default ±2.0 Hz).

    Returns
    -------
    TSIResult
        Container with TSI and all intermediate results.

    Raises
    ------
    ValueError
        If the input is not a valid 1-D signal or is too short.
    RuntimeError
        If no reliable peak is found, or there are too few ZC to estimate
        instantaneous frequency and Δf.
    """
    sig = np.asarray(signal, dtype=float)

    # Basic validation: require a 1-D vector and at least ~0.5 s of data
    if sig.ndim != 1 or len(sig) < int(fs * 0.5):
        raise ValueError("Input signal must be 1-D and at least ~0.5 s long.")

    # 1) High-pass filter to remove slow drift/DC
    hp = _butter_filter(sig, fs, kind="hp", fc_low=hp_fc)

    # 2) Find dominant peak within [fmin, fmax]
    fmin, fmax = float(peak_range[0]), float(peak_range[1])
    f_peak = _estimate_peak_freq(hp, fs, fmin=fmin, fmax=fmax)
    if (f_peak is None) or (not np.isfinite(f_peak)):
        raise RuntimeError("Could not find a clear spectral peak in the specified range.")

    # 3) Build a band-pass around f_peak ± bp_halfwidth and clip to a safe range
    #    We keep a small guard (0.2 Hz) away from 0 Hz and Nyquist
    lo = max(0.2, f_peak - float(bp_halfwidth))
    hi = min(fs / 2.0 - 0.2, f_peak + float(bp_halfwidth))
    if hi <= lo:
        raise RuntimeError("Invalid band-pass range. Check fs or estimated f_peak.")

    bp = _butter_filter(hp, fs, kind="bp", fc_low=lo, fc_high=hi)

    # 4) Positive zero-crossings on the band-passed signal
    zc_idx = _positive_zero_crossings(bp)
    if zc_idx.size < 3:
        raise RuntimeError("Too few zero-crossings to estimate instantaneous frequency.")

    # 5) Instantaneous frequency from full cycles:
    #    Using every two *consecutive* positive ZC approximates whole periods.
    #    Convert ZC sample indices to times, then take diffs every two crossings.
    zc_times = zc_idx / float(fs)

    # If the number of ZC+ is odd, the last one will be dropped implicitly by slicing
    # Use every two crossings to get full-cycle periods
    full_cycle_times = zc_times[::2]
    periods = np.diff(full_cycle_times)
    periods = periods[periods > 0]  # keep only positive periods

    if periods.size < 2:
        raise RuntimeError("Not enough full cycles to compute Δf robustly.")

    inst_freq = 1.0 / periods  # Hz

    # 6) Δf: consecutive differences of the instantaneous frequency
    delta_f = np.diff(inst_freq)
    if delta_f.size == 0:
        raise RuntimeError("Could not compute Δf (insufficient cycles).")

    # 7) TSI = IQR(Δf) = Q3 - Q1
    q1 = float(np.percentile(delta_f, 25))
    q3 = float(np.percentile(delta_f, 75))
    tsi = float(q3 - q1)

    return TSIResult(
        tsi=tsi,
        q1=q1,
        q3=q3,
        delta_f=delta_f,
        inst_freq=inst_freq,
        bp=bp,
        zc_idx=zc_idx,
        f_peak=f_peak
    )
# ======================= END OF TSI MODULE (Drop-in) =======================
