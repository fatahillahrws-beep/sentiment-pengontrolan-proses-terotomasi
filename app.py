# -*- coding: utf-8 -*-
"""
SentiScope - Dashboard Analisis Sentimen IndoBERT (Klik Indomaret & Alfagift)
=============================================================================
Versi Streamlit dari notebook `versi_3.ipynb`.

Alur yang sama dengan notebook:
  1. Data  -> 2. Pelabelan lexicon InSet -> 3. Lexicon vs rating (Kappa)
  4. Latih IndoBERT (dataset x sumber label) -> 5. Evaluasi -> 6. Control chart p

Jalankan:
    pip install -r requirements.txt
    streamlit run app.py

Semua konfigurasi (data, lexicon, hyper-parameter, p-chart, tampilan) bisa diatur
lewat sidebar, dan bisa diekspor / diimpor sebagai file JSON.
"""
from __future__ import annotations

import html as _html
import hashlib
import importlib.util
import inspect
import io
import json
import math
import os
import re
import shutil
import time
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots
from sklearn.metrics import (accuracy_score, classification_report, cohen_kappa_score,
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.model_selection import train_test_split

# =============================================================================
# KONSTANTA
# =============================================================================
APP_NAME = "SentiScope"
LABELS = ["negatif", "netral", "positif"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}
LABEL_COLORS = {"negatif": "#F43F5E", "netral": "#94A3B8", "positif": "#10B981"}
INSET_URL = "https://raw.githubusercontent.com/fajri91/InSet/master/{}.tsv"
STAR_OPT = "(konversi dari kolom bintang)"
NONE_OPT = "(tidak ada)"
FONT = "'Plus Jakarta Sans', 'Segoe UI', system-ui, -apple-system, sans-serif"

PALETTES = {
    "Samudra":    ("#6366F1", "#06B6D4"),
    "Senja":      ("#F97316", "#EC4899"),
    "Hutan":      ("#10B981", "#0EA5E9"),
    "Ungu Malam": ("#8B5CF6", "#F472B6"),
    "Emas":       ("#F59E0B", "#EF4444"),
}

# ---- kamus (sama persis dengan notebook) ------------------------------------
SLANG = {
    "ga": "tidak", "gak": "tidak", "gk": "tidak", "g": "tidak", "nggak": "tidak", "ngga": "tidak",
    "enggak": "tidak", "engga": "tidak", "tdk": "tidak", "ndak": "tidak", "kagak": "tidak", "tak": "tidak",
    "gabisa": "tidak bisa", "gbs": "tidak bisa", "gajelas": "tidak jelas", "blm": "belum", "blum": "belum",
    "bgt": "banget", "bngt": "banget", "udh": "sudah", "udah": "sudah", "sdh": "sudah", "dah": "sudah",
    "yg": "yang", "tp": "tapi", "tpi": "tapi", "krn": "karena", "karna": "karena", "klo": "kalau",
    "kalo": "kalau", "sampe": "sampai", "pake": "pakai", "apk": "aplikasi", "aplikasinya": "aplikasi",
    "eror": "error", "lemod": "lemot", "trims": "terima kasih", "makasih": "terima kasih",
    "makasi": "terima kasih", "thanks": "terima kasih", "tq": "terima kasih", "mantab": "mantap",
    "mantep": "mantap", "mantul": "mantap", "rekomen": "rekomendasi", "recommended": "rekomendasi",
    "kesel": "kesal", "nyesel": "menyesal", "males": "malas", "bener": "benar", "dateng": "datang",
    "telat": "terlambat", "cancel": "batal", "dicancel": "dibatalkan", "lelet": "lambat",
}
NEGATORS_DEFAULT = ["tidak", "bukan", "belum", "jangan", "kurang", "tanpa"]

DOMAIN_LEX = {
    # positif
    "bagus": 3, "mantap": 4, "membantu": 3, "mudah": 3, "memudahkan": 3, "cepat": 3, "murah": 3,
    "baik": 3, "suka": 3, "puas": 3, "terbaik": 4, "keren": 3, "lengkap": 2, "ramah": 3, "hemat": 3,
    "oke": 2, "ok": 2, "rekomendasi": 3, "praktis": 3, "aman": 2, "nyaman": 3, "senang": 3,
    "lancar": 3, "mantap jiwa": 4, "terima kasih": 2, "tepat waktu": 3, "sesuai": 2, "top": 3,
    "worth": 3, "untung": 2, "gratis": 2, "promo": 1, "diskon": 1, "membantu sekali": 4,
    "sangat membantu": 4, "sukses": 2, "joss": 3, "jos": 3, "good": 3, "nice": 3, "best": 4,
    # negatif
    "kecewa": -4, "mengecewakan": -4, "lama": -3, "lambat": -3, "lemot": -4, "error": -4, "ribet": -3,
    "susah": -3, "sulit": -3, "jelek": -4, "buruk": -4, "parah": -4, "gagal": -3, "kosong": -2,
    "gangguan": -3, "batal": -3, "dibatalkan": -3, "terlambat": -3, "mahal": -3, "aneh": -2, "bug": -3,
    "komplain": -2, "kesal": -3, "menyesal": -3, "rugi": -3, "tipu": -4, "penipuan": -4, "bohong": -4,
    "kapok": -4, "malas": -3, "zonk": -3, "masalah": -2, "bermasalah": -3, "tidak bisa": -3,
    "tidak jelas": -3, "tidak sesuai": -3, "force close": -3, "hilang": -2,
    "lambat sekali": -4, "lama sekali": -4, "bad": -3,
    # netral di domain ini (InSet memberi bobot besar padahal netral)
    "aplikasi": 0, "saya": 0, "yang": 0, "dari": 0, "lagi": 0, "kalau": 0, "itu": 0, "harus": 0,
    "masuk": 0, "apa": 0, "karena": 0, "stok": 0, "sistem": 0, "saja": 0, "semua": 0, "katanya": 0,
    "pilih": 0, "cuma": 0, "lain": 0, "datang": 0, "tau": 0, "sebelah": 0, "jauh": 0, "pihak": 0,
    "pun": 0, "balik": 0, "hanya": 0, "seperti": 0, "ada": 0, "mau": 0, "sudah": 0, "belanja": 0,
    "sampai": 0, "pesan": 0, "tolong": 0, "beli": 0, "jadi": 0, "jelas": 0, "bayar": 0, "aku": 0,
    "minta": 0, "lewat": 0, "mulu": 0, "kasih": 0, "habis": 0, "buka": 0, "pagi": 0, "biar": 0,
    "daftar": 0, "benar": 0, "satu": 0, "ganti": 0, "coba": 0, "muncul": 0, "memang": 0, "dapat": 0,
    "mohon": 0, "kaya": 0, "tinggal": 0, "pengiriman": 0, "sekali": 0, "sangat": 0, "banyak": 0,
    "beda": 0, "mending": 0, "banget": 0, "aja": 0, "sering": 0, "sama": 0, "lebih": 0, "buat": 0,
    "dikirim": 0, "chat": 0, "pas": 0, "ya": 0, "pelayanan": 0, "harga": 0, "kirim": 0, "respon": 0,
    "langsung": 0, "produk": 0, "nunggu": 0, "konfirmasi": 0, "dana": 0, "rumah": 0, "makin": 0,
    "proses": 0, "saldo": 0, "dalam": 0, "poin": 0, "kadang": 0, "transaksi": 0, "pelanggan": 0,
    "layanan": 0, "tetap": 0, "karyawan": 0, "voucher": 0, "pilihan": 0, "guna": 0, "punya": 0,
    "pasti": 0, "tersedia": 0, "kurir": 0, "barang": 0, "pesanan": 0, "toko": 0, "jam": 0, "hari": 0,
}

LABEL_SYNONYMS = {
    "positive": "positif", "pos": "positif", "positif": "positif",
    "negative": "negatif", "neg": "negatif", "negatif": "negatif",
    "neutral": "netral", "neu": "netral", "netral": "netral",
}

INSET_MODES = ["Otomatis (cache lokal → GitHub)", "Unggah file TSV", "Hanya kamus domain (tanpa InSet)"]
LR_OPTIONS = [5e-6, 1e-5, 2e-5, 3e-5, 5e-5, 1e-4]
FREQS = ["Harian", "Mingguan", "Bulanan"]
P_SOURCES_ALL = ["prediksi_lexicon", "prediksi_rating", "label_lexicon", "label_rating"]

# ---- nilai bawaan seluruh konfigurasi (semua bisa diubah di sidebar) --------
DEFAULTS = {
    # tampilan
    "palette": "Samudra", "chart_theme": "Otomatis", "animate": True, "autoplay": True,
    "frame_ms": 60, "chart_h": 460,
    # data
    "use_demo": True, "ds2_enabled": True,
    "ds1_name": "Klik Indomaret", "ds1_path": "Klick_Indomaret.csv",
    "ds1_text": "ulasan", "ds1_label": "sentimen", "ds1_rating": "rating", "ds1_date": "tanggal",
    "ds2_name": "Alfagift", "ds2_path": "ulasan_alfagift_fix.csv",
    "ds2_text": "ulasan", "ds2_label": "label_sentimen_rating", "ds2_rating": "rating", "ds2_date": "tanggal_ulasan",
    # lexicon
    "inset_mode": INSET_MODES[0], "use_domain": True, "neutral_margin": 0, "neg_window": 2,
    "negators": ", ".join(NEGATORS_DEFAULT), "extra_lex": "",
    # model
    "model_name": "indobenchmark/indobert-base-p1", "max_len": 128, "epochs": 4, "batch": 16,
    "lr": 2e-5, "seed": 42, "class_weight": True, "patience": 2, "weight_decay": 0.01, "warmup": 0.1,
    "split_train": 8, "split_val": 1, "split_test": 1, "label_sources": ["lexicon", "rating"],
    # control chart p
    "p_defect": "negatif", "p_sources": ["prediksi_lexicon"], "p_k": 3.0, "p_freq": "Harian", "p_min_n": 1,
    # output
    "output_dir": "hasil_indobert",
}


def K(name: str) -> str:
    return f"cfg_{name}"


def C(name: str):
    return st.session_state.get(K(name), DEFAULTS[name])


# =============================================================================
# UTILITAS UMUM
# =============================================================================
def _stretch_kw(fn) -> dict:
    """Kompatibel lintas versi Streamlit: width='stretch' (baru) atau use_container_width (lama)."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {}
    if "width" in params:
        return {"width": "stretch"}
    if "use_container_width" in params:
        return {"use_container_width": True}
    return {}


def html_block(s: str):
    st.markdown(s, unsafe_allow_html=True)


def hex_to_rgba(h: str, a: float) -> str:
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a})"


def accents():
    return PALETTES.get(C("palette"), PALETTES["Samudra"])


def is_dark() -> bool:
    mode = C("chart_theme")
    if mode == "Gelap":
        return True
    if mode == "Terang":
        return False
    try:
        t = st.context.theme.type
        if t in ("dark", "light"):
            return t == "dark"
    except Exception:
        pass
    try:
        return (st.get_option("theme.base") or "dark") == "dark"
    except Exception:
        return True


def slugify(s: str) -> str:
    return re.sub(r"\W+", "_", s.lower()).strip("_")


def pct(x, d=1):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:.{d}f}%"


def kappa_label(k: float) -> str:
    if k < 0.20:
        return "lemah"
    if k <= 0.40:
        return "cukup"
    if k <= 0.60:
        return "sedang"
    if k <= 0.80:
        return "kuat"
    return "sangat kuat"


# =============================================================================
# DATA CONTOH (sintetis) - hanya agar aplikasi bisa dicoba tanpa file asli
# =============================================================================
@st.cache_data(show_spinner=False)
def demo_csv(i: int) -> bytes:
    rng = np.random.default_rng(100 + i)
    pos = ["aplikasinya bagus banget dan mudah dipakai", "pengiriman cepat barang sesuai pesanan mantap",
           "promo banyak harga murah sangat membantu", "pelayanan ramah kurirnya tepat waktu terima kasih",
           "puas belanja di sini rekomendasi banget", "fitur lengkap proses checkout lancar",
           "suka banget hemat dan praktis", "barang sampai aman pelayanan baik", "keren dan nyaman dipakai belanja"]
    neg = ["aplikasi error terus tidak bisa login kecewa", "pengiriman lama banget sampai terlambat parah",
           "barang tidak sesuai pesanan mengecewakan", "lemot dan sering force close sangat ribet",
           "pesanan dibatalkan sepihak saldo hilang kesal", "respon customer service lambat tidak jelas",
           "stok sering kosong dan harga mahal", "gagal bayar terus bug parah menyesal", "kurang bagus dan susah dipakai"]
    neu = ["saya pesan barang lewat aplikasi kemarin", "stok di toko sebelah jam buka pagi",
           "mau tanya cara pakai voucher", "sudah coba daftar lagi", "pesanan sampai hari ini",
           "minta tolong konfirmasi pengiriman", "poin belanja masuk ke saldo"]
    suffix = ["", "", " bgt", " sih", " ya", " gak nyesel", " tolong diperbaiki", " semoga makin baik"]
    n_days, n = 120, 1800
    base = pd.Timestamp("2026-06-01")
    day_w = np.ones(n_days)
    day_w[78:82] = 2.6                       # lonjakan ulasan
    day_w = day_w / day_w.sum()
    days = rng.choice(n_days, size=n, p=day_w)
    rows = []
    for dday in days:
        p_neg = 0.78 if 78 <= dday < 82 else 0.32
        p = np.array([p_neg, 0.14, 1 - p_neg - 0.14])
        s = rng.choice(["negatif", "netral", "positif"], p=p)
        pool = {"negatif": neg, "netral": neu, "positif": pos}[s]
        text = " ".join(rng.choice(pool, size=rng.integers(1, 3), replace=False)) + rng.choice(suffix)
        star = {"negatif": rng.choice([1, 2]), "netral": 3, "positif": rng.choice([4, 5])}[s]
        if rng.random() < 0.13:                              # sebagian rating tidak cocok dengan teks
            star = int(rng.integers(1, 6))
        lab = "negatif" if star <= 2 else ("netral" if star == 3 else "positif")
        ts = base + pd.Timedelta(days=int(dday), hours=int(rng.integers(7, 23)), minutes=int(rng.integers(0, 60)))
        rows.append((text, int(star), lab, ts))
    if i == 1:
        df = pd.DataFrame({"ulasan": [r[0] for r in rows], "rating": [r[1] for r in rows],
                           "sentimen": [r[2] for r in rows],
                           "tanggal": [r[3].strftime("%d/%m/%Y %H:%M") for r in rows]})
    else:
        df = pd.DataFrame({"ulasan": [r[0] for r in rows], "rating": [r[1] for r in rows],
                           "label_sentimen_rating": [r[2] for r in rows],
                           "tanggal_ulasan": [r[3].strftime("%m/%d/%Y %H:%M") for r in rows]})
    return df.to_csv(index=False).encode("utf-8")


# =============================================================================
# LEXICON (port dari notebook, bagian 5)
# =============================================================================
@st.cache_data(show_spinner="Memuat kamus InSet…")
def load_inset(mode: str, out_dir: str, up_pos: bytes | None, up_neg: bytes | None):
    """Mengembalikan (pos_df, neg_df). Melempar exception bila gagal (agar tidak ikut di-cache)."""
    def one(name, up):
        cache_path = os.path.join(out_dir, f"inset_{name}.tsv")
        if mode == INSET_MODES[1]:
            if not up:
                raise ValueError("File TSV InSet belum diunggah.")
            return pd.read_csv(io.BytesIO(up), sep="\t")
        if os.path.exists(cache_path):
            return pd.read_csv(cache_path, sep="\t")
        with urllib.request.urlopen(INSET_URL.format(name), timeout=25) as r:
            data = r.read()
        os.makedirs(out_dir, exist_ok=True)
        with open(cache_path, "wb") as f:
            f.write(data)
        return pd.read_csv(io.BytesIO(data), sep="\t")

    return one("positive", up_pos), one("negative", up_neg)


def parse_extra_lex(text: str) -> dict:
    out = {}
    for line in str(text).splitlines():
        m = re.match(r"^\s*(.+?)\s*[:=,]\s*(-?\d+)\s*$", line)
        if m:
            out[m.group(1).strip().lower()] = int(m.group(2))
    return out


def build_lexicon():
    """Bangun LEX dari pengaturan sidebar. Hasil disimpan di session_state['_lex']."""
    mode = C("inset_mode")
    out_dir = C("output_dir")
    pos_df = neg_df = None
    status, inset_ok = "", False
    if mode != INSET_MODES[2]:
        try:
            up_pos = st.session_state.get("cfg_inset_pos")
            up_neg = st.session_state.get("cfg_inset_neg")
            pos_df, neg_df = load_inset(mode, out_dir,
                                        up_pos.getvalue() if up_pos else None,
                                        up_neg.getvalue() if up_neg else None)
            inset_ok = True
            status = f"InSet dimuat ({len(pos_df):,} kata positif, {len(neg_df):,} kata negatif)"
        except Exception as e:                       # tanpa internet / file belum ada
            status = f"InSet gagal dimuat ({type(e).__name__}); memakai kamus domain saja"
    else:
        status = "Mode kamus domain saja"

    use_domain = bool(C("use_domain")) or not inset_ok
    lex: dict = {}
    if inset_ok:
        for w, s in zip(pos_df["word"], pos_df["weight"]):
            lex[str(w).strip()] = lex.get(str(w).strip(), 0) + int(s)
        for w, s in zip(neg_df["word"], neg_df["weight"]):
            lex[str(w).strip()] = lex.get(str(w).strip(), 0) + int(s)
    if use_domain:
        lex.update(DOMAIN_LEX)
    lex.update(parse_extra_lex(C("extra_lex")))
    negators = {w.strip().lower() for w in re.split(r"[,\n]", C("negators")) if w.strip()}
    for n in negators:                      # negasi ditangani lewat aturan, bukan bobot kata
        lex.pop(n, None)
    sig = hashlib.md5(json.dumps(sorted(lex.items()), ensure_ascii=False).encode()).hexdigest()
    st.session_state["_lex"] = {"lex": lex, "negators": negators, "sig": sig, "status": status,
                                "inset_ok": inset_ok, "n": len(lex)}


def normalize_tokens(text) -> list[str]:
    t = str(text).lower()
    t = re.sub(r"https?://\S+|www\.\S+|@\w+", " ", t)
    t = re.sub(r"[^a-z\s]", " ", t)
    t = re.sub(r"(.)\1{2,}", r"\1", t)           # "bagusss" -> "bagus"
    toks: list[str] = []
    for w in t.split():
        toks.extend(SLANG.get(w, w).split())
    return toks


def score_tokens(toks: list[str], lex: dict, negators, window: int = 2):
    """Jumlah bobot kata (frasa 1-3 kata); negasi membalik tanda kata sentimen berikutnya."""
    score, i, negate_until, hits = 0, 0, -1, []
    while i < len(toks):
        matched = False
        for n in (3, 2):                         # frasa dulu (mis. "tidak bisa")
            seg = toks[i:i + n]
            if len(seg) == n:
                ph = " ".join(seg)
                if ph in lex:
                    val = lex[ph]
                    score += val
                    if val:
                        hits.append((ph, val, i, i + n))
                    i += n
                    matched = True
                    break
        if matched:
            continue
        w = toks[i]
        if w in negators:
            negate_until = i + window            # negasi berlaku untuk N kata berikutnya
        elif w in lex and lex[w] != 0:
            val = -lex[w] if i <= negate_until else lex[w]
            if i <= negate_until:
                negate_until = -1
            score += val
            hits.append((w, val, i, i + 1))
        i += 1
    return score, hits


def lexicon_score(text, lex, negators, window=2):
    toks = normalize_tokens(text)
    score, hits = score_tokens(toks, lex, negators, window)
    return score, ", ".join(f"{p}({v:+d})" for p, v, _, _ in hits)


def label_from_score(score: int, margin: int) -> str:
    if score > margin:
        return "positif"
    if score < -margin:
        return "negatif"
    return "netral"


# =============================================================================
# PEMBACAAN & PERSIAPAN DATA (bagian 4, 5, 7 notebook)
# =============================================================================
@st.cache_data(show_spinner=False)
def read_csv_bytes(raw: bytes) -> pd.DataFrame:
    # sep=None mendeteksi pemisah otomatis (Indomaret ';', Alfagift ',')
    return pd.read_csv(io.BytesIO(raw), sep=None, engine="python", encoding="utf-8-sig")


def clean_text(t) -> str:
    t = str(t).lower()
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)
    t = re.sub(r"@\w+", " ", t)
    t = re.sub(r"[^0-9a-z\s]", " ", t)
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)
    return re.sub(r"\s+", " ", t).strip()


def parse_dates(s: pd.Series) -> pd.Series:
    """Coba beberapa format tanggal, pilih yang paling banyak berhasil."""
    formats = ["%d/%m/%Y %H:%M", "%m/%d/%y %H:%M", "%m/%d/%Y %H:%M", "%d/%m/%y %H:%M",
               "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%d/%m/%Y", "%Y-%m-%d"]
    best = None
    for fmt in formats:
        parsed = pd.to_datetime(s, format=fmt, errors="coerce")
        if best is None or parsed.notna().sum() > best.notna().sum():
            best = parsed
    if best.notna().mean() < 0.9:
        best = pd.to_datetime(s, errors="coerce", dayfirst=True)
    return best


def star_to_label(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "tidak diketahui"
    return "negatif" if v <= 2 else ("netral" if v < 4 else "positif")


@st.cache_data(show_spinner="Memproses & melabeli ulasan dengan lexicon…")
def process_dataset(raw: bytes, name: str, text_col: str, label_col: str, rating_col: str, date_col: str,
                    lex_sig: str, margin: int, window: int, _lex: dict, _negators: set):
    df = read_csv_bytes(raw).copy()
    texts = df[text_col].astype(str)
    sc = [lexicon_score(t, _lex, _negators, window) for t in texts]
    df["skor_lexicon"] = [s for s, _ in sc]
    df["kata_sentimen"] = [h for _, h in sc]
    df["label_lexicon"] = df["skor_lexicon"].map(lambda s: label_from_score(s, margin))

    if label_col == STAR_OPT and rating_col in df.columns:
        df["label_rating"] = df[rating_col].map(star_to_label)
    elif label_col in df.columns:
        lab = df[label_col].astype(str).str.strip().str.lower()
        df["label_rating"] = lab.map(lambda x: LABEL_SYNONYMS.get(x, x))
    else:
        df["label_rating"] = "tidak diketahui"

    dates = parse_dates(df[date_col].astype(str)) if date_col in df.columns else pd.Series(pd.NaT, index=df.index)
    d = pd.DataFrame({
        "tanggal": dates,
        "text_asli": df[text_col],
        "text": df[text_col].map(clean_text),
        "label_rating_str": df["label_rating"],
        "label_lexicon_str": df["label_lexicon"],
        "skor": df["skor_lexicon"],
        "kata_sentimen": df["kata_sentimen"],
    })
    if rating_col in df.columns:
        d["rating"] = df[rating_col]
    before = len(d)
    d = d[d["label_rating_str"].isin(LABELS) & (d["text"].str.len() > 0)].reset_index(drop=True)
    d["label_rating"] = d["label_rating_str"].map(LABEL2ID)
    d["label_lexicon"] = d["label_lexicon_str"].map(LABEL2ID)
    info = {"dibuang": before - len(d), "n_raw": before, "text_col": text_col, "rating_col": rating_col,
            "tgl_gagal": int(d["tanggal"].isna().sum())}
    return {"raw": df, "data": d, "info": info}


def get_ctx() -> dict:
    cfgs = st.session_state.get("_ds_cfgs") or []
    lx = st.session_state.get("_lex")
    if not cfgs or not lx:
        empty_state()
        st.stop()
    ctx = {}
    for c in cfgs:
        ctx[c["name"]] = process_dataset(c["raw"], c["name"], c["text"], c["label"], c["rating"], c["date"],
                                         lx["sig"], int(C("neutral_margin")), int(C("neg_window")),
                                         _lex=lx["lex"], _negators=lx["negators"])
        ctx[c["name"]]["origin"] = c["origin"]
    return ctx


def split_data(d: pd.DataFrame, ratios: dict, seed: int):
    total = sum(ratios.values())
    p_val, p_test = ratios["val"] / total, ratios["test"] / total
    try:
        train, temp = train_test_split(d, test_size=p_val + p_test, stratify=d["label_lexicon"], random_state=seed)
        val, test = train_test_split(temp, test_size=p_test / (p_val + p_test),
                                     stratify=temp["label_lexicon"], random_state=seed)
    except ValueError:                              # kelas terlalu kecil untuk stratify
        train, temp = train_test_split(d, test_size=p_val + p_test, random_state=seed)
        val, test = train_test_split(temp, test_size=p_test / (p_val + p_test), random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


# =============================================================================
# CONTROL CHART p (bagian 14 notebook)
# =============================================================================
def p_table(dates: pd.Series, values: pd.Series, defect: str, k: float, freq: str, min_n: int = 1):
    tmp = pd.DataFrame({"tanggal": dates.values, "v": values.values}).dropna(subset=["tanggal"])
    if tmp.empty:
        return None, None
    if freq == "Harian":
        tmp["periode"] = tmp["tanggal"].dt.normalize()
    elif freq == "Mingguan":
        tmp["periode"] = tmp["tanggal"].dt.to_period("W").dt.start_time
    else:
        tmp["periode"] = tmp["tanggal"].dt.to_period("M").dt.start_time
    tmp["cacat"] = (tmp["v"] == defect).astype(int)
    t = tmp.groupby("periode").agg(n=("cacat", "size"), d=("cacat", "sum")).reset_index()
    t = t[t["n"] >= max(1, int(min_n))].reset_index(drop=True)
    if t.empty:
        return None, None
    t["p"] = t["d"] / t["n"]
    p_bar = t["d"].sum() / t["n"].sum()
    sigma = np.sqrt(p_bar * (1 - p_bar) / t["n"])
    t["CL"] = p_bar
    t["UCL"] = (p_bar + k * sigma).clip(upper=1)
    t["LCL"] = (p_bar - k * sigma).clip(lower=0)
    t["status"] = np.where(t["p"] > t["UCL"], "di atas UCL",
                           np.where(t["p"] < t["LCL"], "di bawah LCL", "terkendali"))
    return t, float(p_bar)


def source_series(d: pd.DataFrame, ds: str, source: str):
    if source.startswith("label_"):
        return d[f"{source}_str"]
    src = source.replace("prediksi_", "")
    r = st.session_state["results"].get((ds, src))
    if r is None or len(r["pred_all"]) != len(d):
        return None
    return pd.Series(r["pred_all"], index=d.index)


def available_sources(d, ds) -> list[str]:
    return [s for s in P_SOURCES_ALL if source_series(d, ds, s) is not None]


# =============================================================================
# PELATIHAN INDOBERT (bagian 9-12 notebook) - impor torch/transformers dimuat malas
# =============================================================================
def torch_ready() -> bool:
    return importlib.util.find_spec("torch") is not None and importlib.util.find_spec("transformers") is not None


def build_hist(log_history) -> pd.DataFrame:
    tr = pd.DataFrame([{"epoch": h["epoch"], "train_loss": h["loss"]}
                       for h in log_history if "loss" in h and "eval_loss" not in h])
    ev = pd.DataFrame([{"epoch": h["epoch"], "val_loss": h["eval_loss"], "val_accuracy": h["eval_accuracy"],
                        "val_f1_macro": h["eval_f1_macro"]} for h in log_history if "eval_f1_macro" in h])
    if ev.empty:
        return pd.DataFrame(columns=["epoch", "train_loss", "val_loss", "val_accuracy", "val_f1_macro"])
    if tr.empty:
        tr = pd.DataFrame({"epoch": ev["epoch"], "train_loss": np.nan})
    return ev.merge(tr, on="epoch", how="left")[["epoch", "train_loss", "val_loss", "val_accuracy", "val_f1_macro"]]


def run_experiment(ds_name, label_source, data, splits, ui):
    import gc
    import torch
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments,
                              DataCollatorWithPadding, EarlyStoppingCallback, TrainerCallback)

    cfg = {k: C(k) for k in DEFAULTS}
    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    train_df, val_df, test_df = splits
    col = f"label_{label_source}"
    slug = slugify(f"{ds_name}_{label_source}")

    def set_seed(seed):
        import random
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    set_seed(cfg["seed"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])

    class ReviewDataset(torch.utils.data.Dataset):
        def __init__(self, texts, labels):
            self.enc = tokenizer(list(texts), truncation=True, max_length=int(cfg["max_len"]))
            self.labels = list(labels)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
            item["labels"] = torch.tensor(self.labels[i])
            return item

    class WeightedTrainer(Trainer):
        def __init__(self, *args, class_weights=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.class_weights = class_weights

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            w = self.class_weights.to(outputs.logits.device) if self.class_weights is not None else None
            loss = torch.nn.functional.cross_entropy(outputs.logits, labels, weight=w)
            return (loss, outputs) if return_outputs else loss

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
        return {"accuracy": accuracy_score(labels, preds), "precision_macro": p, "recall_macro": r, "f1_macro": f1}

    class UICallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kw):
            frac = state.global_step / max(state.max_steps, 1)
            ui["prog"].progress(min(frac, 1.0), text=f"{ds_name} · label {label_source} · "
                                                     f"langkah {state.global_step}/{state.max_steps} · epoch {state.epoch or 0:.2f}")

        def on_evaluate(self, args, state, control, metrics=None, **kw):
            h = build_hist(state.log_history)
            if not h.empty:
                ui["counter"][0] += 1
                ui["table"].dataframe(h.round(4), hide_index=True, **_stretch_kw(st.dataframe))

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"], num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID)
    train_ds = ReviewDataset(train_df["text"], train_df[col])
    val_ds = ReviewDataset(val_df["text"], val_df[col])
    test_ds = ReviewDataset(test_df["text"], test_df[col])

    steps_per_epoch = int(np.ceil(len(train_ds) / cfg["batch"]))
    kw = dict(output_dir=os.path.join(out_dir, f"checkpoint_{slug}"), save_strategy="epoch",
              learning_rate=cfg["lr"], per_device_train_batch_size=int(cfg["batch"]),
              per_device_eval_batch_size=int(cfg["batch"]) * 2, num_train_epochs=int(cfg["epochs"]),
              weight_decay=float(cfg["weight_decay"]),
              warmup_steps=int(float(cfg["warmup"]) * steps_per_epoch * int(cfg["epochs"])),
              load_best_model_at_end=True, metric_for_best_model="f1_macro", greater_is_better=True,
              save_total_limit=1, logging_strategy="epoch", disable_tqdm=True, report_to="none",
              seed=int(cfg["seed"]), fp16=torch.cuda.is_available())
    try:
        args = TrainingArguments(eval_strategy="epoch", **kw)
    except TypeError:                                # transformers lama
        args = TrainingArguments(evaluation_strategy="epoch", **kw)

    cw = None
    if cfg["class_weight"]:
        counts = np.bincount(train_df[col].values, minlength=len(LABELS)).astype(float)
        cw = torch.tensor(len(train_df) / (len(LABELS) * np.maximum(counts, 1)), dtype=torch.float)

    trainer = WeightedTrainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                              data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=compute_metrics,
                              callbacks=[EarlyStoppingCallback(early_stopping_patience=int(cfg["patience"])),
                                         UICallback()],
                              class_weights=cw)
    train_out = trainer.train()

    hist = build_hist(trainer.state.log_history)
    best_epoch = float(hist.loc[hist["val_f1_macro"].idxmax(), "epoch"])

    pred = trainer.predict(test_ds)
    y_true = test_df[col].values
    y_pred = pred.predictions.argmax(axis=-1)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))
    rep = classification_report(y_true, y_pred, labels=list(range(len(LABELS))), target_names=LABELS,
                                output_dict=True, zero_division=0)
    per_class = pd.DataFrame({l: rep[l] for l in LABELS}).T[["precision", "recall", "f1-score", "support"]]
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)

    out = test_df[["text_asli", "text", "label_rating_str", "label_lexicon_str"]].copy()
    out[f"prediksi_model_{label_source}"] = [ID2LABEL[i] for i in y_pred]
    out["benar"] = out[f"label_{label_source}_str"] == out[f"prediksi_model_{label_source}"]

    full_pred = trainer.predict(ReviewDataset(data["text"], data[col])).predictions.argmax(axis=-1)
    pred_all = [ID2LABEL[i] for i in full_pred]

    summary = {"dataset": ds_name, "label": label_source, "n_train": len(train_df), "n_val": len(val_df),
               "n_test": len(test_df), "accuracy": float(accuracy_score(y_true, y_pred)),
               "precision_macro": float(p), "recall_macro": float(r), "f1_macro": float(f1),
               "best_epoch": best_epoch, "train_seconds": float(train_out.metrics.get("train_runtime", 0))}

    del trainer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    shutil.rmtree(kw["output_dir"], ignore_errors=True)       # checkpoint besar tidak perlu disimpan
    return {"summary": summary, "cm": cm, "hist": hist, "per_class": per_class, "test_pred": out,
            "pred_all": pred_all}


def persist_results():
    out_dir = C("output_dir")
    os.makedirs(out_dir, exist_ok=True)
    store = {}
    for (ds, src), r in st.session_state["results"].items():
        slug = slugify(f"{ds}_{src}")
        r["hist"].to_csv(os.path.join(out_dir, f"riwayat_training_{slug}.csv"), index=False, encoding="utf-8-sig")
        r["test_pred"].to_csv(os.path.join(out_dir, f"prediksi_test_{slug}.csv"), index=False, encoding="utf-8-sig")
        pd.Series(r["pred_all"], name="prediksi").to_csv(os.path.join(out_dir, f"prediksi_semua_{slug}.csv"),
                                                          index=False, encoding="utf-8-sig")
        store[f"{ds}||{src}"] = {"summary": r["summary"], "cm": np.asarray(r["cm"]).tolist(),
                                 "hist": r["hist"].to_dict("records"),
                                 "per_class": r["per_class"].reset_index().to_dict("records"), "slug": slug}
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, indent=1)
    summ = pd.DataFrame([r["summary"] for r in st.session_state["results"].values()])
    if not summ.empty:
        summ.to_csv(os.path.join(out_dir, "ringkasan_evaluasi.csv"), index=False, encoding="utf-8-sig")


def load_saved_results() -> int:
    out_dir = C("output_dir")
    path = os.path.join(out_dir, "results.json")
    if not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8") as f:
        store = json.load(f)
    n = 0
    for key, v in store.items():
        ds, src = key.split("||")
        slug = v["slug"]
        try:
            pred_all = pd.read_csv(os.path.join(out_dir, f"prediksi_semua_{slug}.csv"))["prediksi"].tolist()
            test_pred = pd.read_csv(os.path.join(out_dir, f"prediksi_test_{slug}.csv"))
        except Exception:
            continue
        st.session_state["results"][(ds, src)] = {
            "summary": v["summary"], "cm": np.array(v["cm"]), "hist": pd.DataFrame(v["hist"]),
            "per_class": pd.DataFrame(v["per_class"]).set_index("index"), "test_pred": test_pred,
            "pred_all": pred_all}
        n += 1
    return n


# =============================================================================
# GAYA GRAFIK & ANIMASI PLOTLY
# =============================================================================
def themed(fig, height=None, title=None, legend=True):
    dark = is_dark()
    txt = "#E2E8F0" if dark else "#1E293B"
    grid = "rgba(148,163,184,.18)"
    fig.update_layout(
        height=height or int(C("chart_h")),
        title=dict(text=title, x=0.01, xanchor="left", y=0.985, yanchor="top", font=dict(size=16)) if title else None,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=txt, size=13),
        margin=dict(l=12, r=12, t=100 if title else 48, b=12),
        hoverlabel=dict(bgcolor="#0F172A" if dark else "#FFFFFF", font=dict(family=FONT, color="#F8FAFC" if dark else "#0F172A"),
                        bordercolor=accents()[0]),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0, bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor=grid, zeroline=False, linecolor=grid)
    fig.update_yaxes(gridcolor=grid, zeroline=False, linecolor=grid)
    return fig


def add_play_button(fig, frame_ms: int, redraw=True, x=0.0, y=1.0):
    dark = is_dark()
    fig.update_layout(updatemenus=[dict(
        type="buttons", direction="left", x=x, y=y, xanchor="left", yanchor="bottom",
        pad=dict(r=4, t=0, b=6), showactive=False,
        bgcolor="rgba(148,163,184,.18)", bordercolor="rgba(148,163,184,.35)",
        font=dict(color="#E2E8F0" if dark else "#1E293B", size=12),
        buttons=[
            dict(label="▶ Putar", method="animate",
                 args=[None, dict(frame=dict(duration=frame_ms, redraw=redraw), fromcurrent=False,
                                  transition=dict(duration=frame_ms, easing="cubic-in-out"), mode="immediate")]),
            dict(label="⏸ Jeda", method="animate",
                 args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate",
                                    transition=dict(duration=0))]),
        ])])


def add_grow(fig, horizontal=False, steps=14, frame_ms=45):
    """Animasi batang tumbuh dari 0 ke nilai akhir."""
    if not C("animate"):
        return fig
    key = "x" if horizontal else "y"
    finals = [list(getattr(tr, key)) for tr in fig.data if tr.type == "bar"]
    idx = [i for i, tr in enumerate(fig.data) if tr.type == "bar"]
    if not finals:
        return fig
    frames = []
    for s in range(1, steps + 1):
        f = 1 - (1 - s / steps) ** 3                       # ease-out cubic
        frames.append(go.Frame(name=str(s), traces=idx,
                               data=[go.Bar(**{key: [v * f for v in fin]}) for fin in finals]))
    fig.frames = frames
    add_play_button(fig, frame_ms)
    return fig


def show(fig, key=None, autoplay=False, animation_opts=None):
    """Tampilkan figure. Bila ada animasi + autoplay aktif, render lewat HTML agar langsung bergerak."""
    height = int(fig.layout.height or C("chart_h"))
    cfg = {"displaylogo": False, "responsive": True,
           "modeBarButtonsToRemove": ["lasso2d", "select2d"]}
    if autoplay and C("animate") and C("autoplay") and len(fig.frames) > 0:
        opts = animation_opts or {"frame": {"duration": int(C("frame_ms")), "redraw": True},
                                  "transition": {"duration": int(C("frame_ms")), "easing": "cubic-in-out"},
                                  "fromcurrent": False, "mode": "immediate"}
        html = fig.to_html(include_plotlyjs="cdn", full_html=False, auto_play=True,
                           animation_opts=opts, config=cfg)
        components.html(html, height=height + 8)
    else:
        st.plotly_chart(fig, theme=None, key=key, config=cfg, **_stretch_kw(st.plotly_chart))


# =============================================================================
# KOMPONEN UI: CSS, HERO, KPI ANIMASI
# =============================================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
html, body, [class*="st-"], .stApp { font-family: 'Plus Jakarta Sans', 'Segoe UI', system-ui, sans-serif; }
.block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1280px; }
header[data-testid="stHeader"] { background: transparent; backdrop-filter: blur(10px); }
#MainMenu, footer { visibility: hidden; }

/* ---------- hero ---------- */
.hero { position: relative; overflow: hidden; border-radius: 26px; padding: 44px 48px; margin-bottom: 26px; color: #F8FAFC;
  background: radial-gradient(900px 420px at 88% -15%, __A2__66, transparent 60%),
              radial-gradient(760px 420px at -5% 115%, __A1__88, transparent 62%), #0B1020;
  border: 1px solid rgba(255,255,255,.08); }
.hero .blob { position: absolute; border-radius: 50%; filter: blur(60px); opacity: .45; animation: floaty 14s ease-in-out infinite; }
.hero .b1 { width: 260px; height: 260px; background: __A1__; right: 12%; top: -90px; }
.hero .b2 { width: 220px; height: 220px; background: __A2__; left: 30%; bottom: -120px; animation-delay: -6s; }
@keyframes floaty { 0%,100% { transform: translate(0,0) scale(1); } 50% { transform: translate(24px,-18px) scale(1.12); } }
.hero-grid { position: relative; display: flex; flex-wrap: wrap; gap: 28px; align-items: center; justify-content: space-between; }
.hero-text { flex: 1 1 430px; max-width: 640px; }
.hero h1 { font-size: 2.55rem; line-height: 1.12; font-weight: 800; margin: 0 0 14px 0; letter-spacing: -.02em; color: #fff; padding: 0; }
.hero p { font-size: 1.04rem; line-height: 1.65; color: rgba(226,232,240,.88); margin: 0 0 20px 0; }
.chips { display: flex; flex-wrap: wrap; gap: 8px; }
.chip { font-size: .8rem; font-weight: 600; padding: 6px 13px; border-radius: 999px; background: rgba(255,255,255,.10);
  border: 1px solid rgba(255,255,255,.18); color: #E2E8F0; backdrop-filter: blur(6px); }
.hero-art { flex: 0 1 420px; min-width: 280px; }
.hero-art svg { width: 100%; height: auto; display: block; }
.hero-art .ln { stroke-dasharray: 1; stroke-dashoffset: 1; animation: draw 3.4s cubic-bezier(.4,.1,.2,1) .5s forwards; }
.hero-art .dot { opacity: 0; animation: pop .4s ease 3.6s forwards; }
.hero-art .ring { opacity: 0; transform-origin: 375px 22px; animation: ring 1.8s ease-out 3.7s infinite; }
@keyframes draw { to { stroke-dashoffset: 0; } }
@keyframes pop { to { opacity: 1; } }
@keyframes ring { 0% { opacity: .9; transform: scale(.4); } 100% { opacity: 0; transform: scale(2.6); } }

/* ---------- header halaman ---------- */
.phead { position: relative; overflow: hidden; display: flex; align-items: center; gap: 18px; padding: 22px 28px; margin-bottom: 22px;
  border-radius: 20px; color: #F8FAFC; border: 1px solid rgba(255,255,255,.08);
  background: radial-gradient(520px 200px at 100% 0%, __A2__55, transparent 65%),
              radial-gradient(520px 220px at 0% 100%, __A1__66, transparent 65%), #0B1020; }
.phead .ic { font-size: 1.7rem; width: 54px; height: 54px; display: grid; place-items: center; border-radius: 16px;
  background: rgba(255,255,255,.10); border: 1px solid rgba(255,255,255,.16); flex: 0 0 auto; }
.phead h2 { margin: 0; padding: 0; font-size: 1.5rem; font-weight: 800; color: #fff; letter-spacing: -.01em; }
.phead p { margin: 3px 0 0 0; color: rgba(226,232,240,.85); font-size: .95rem; }

/* ---------- kartu & seksi ---------- */
.sec { display: flex; align-items: center; gap: 10px; margin: 26px 0 6px 0; }
.sec .bar { width: 5px; height: 24px; border-radius: 4px; background: linear-gradient(180deg, __A1__, __A2__); }
.sec h3 { margin: 0; padding: 0; font-size: 1.2rem; font-weight: 700; }
.sec-sub { opacity: .72; font-size: .92rem; margin: 0 0 12px 15px; }
.card { background: rgba(148,163,184,.09); border: 1px solid rgba(148,163,184,.22); border-radius: 18px; padding: 18px 20px; height: 100%; }
.card h4 { margin: 0 0 6px 0; padding: 0; font-size: 1.02rem; font-weight: 700; }
.card p { margin: 0; font-size: .9rem; line-height: 1.55; opacity: .82; }
.step-no { display: inline-grid; place-items: center; width: 28px; height: 28px; border-radius: 9px; font-weight: 800; font-size: .85rem;
  color: #fff; background: linear-gradient(135deg, __A1__, __A2__); margin-bottom: 10px; }
.note { border-left: 4px solid __A1__; background: rgba(148,163,184,.09); padding: 12px 16px; border-radius: 0 12px 12px 0; font-size: .92rem; line-height: 1.6; }
.pill { display: inline-block; font-size: .78rem; font-weight: 600; padding: 3px 11px; border-radius: 999px; border: 1px solid rgba(148,163,184,.35); background: rgba(148,163,184,.12); margin-right: 6px; }
.pill.ok { color: #10B981; border-color: rgba(16,185,129,.45); background: rgba(16,185,129,.10); }
.pill.warn { color: #F59E0B; border-color: rgba(245,158,11,.45); background: rgba(245,158,11,.10); }
.pill.bad { color: #F43F5E; border-color: rgba(244,63,94,.45); background: rgba(244,63,94,.10); }
.empty { text-align: center; padding: 70px 20px; border: 1.5px dashed rgba(148,163,184,.4); border-radius: 20px; }
.empty .big { font-size: 2.6rem; }

/* ---------- meteran kappa ---------- */
.kmeter { margin: 10px 0 4px 0; }
.kbar { position: relative; height: 12px; border-radius: 999px;
  background: linear-gradient(90deg, #F43F5E 0%, #F59E0B 25%, #EAB308 45%, #84CC16 65%, #10B981 85%, #059669 100%); }
.kmark { position: absolute; top: -6px; left: var(--to); width: 24px; height: 24px; margin-left: -12px; border-radius: 50%;
  background: #fff; border: 4px solid __A1__; box-shadow: 0 2px 10px rgba(0,0,0,.35); animation: mv 1.6s cubic-bezier(.2,.8,.2,1) both; }
@keyframes mv { from { left: 0%; } to { left: var(--to); } }
.kscale { display: flex; justify-content: space-between; font-size: .72rem; opacity: .65; margin-top: 8px; }

/* ---------- chip token lexicon ---------- */
.tok { display: inline-block; margin: 3px 4px 3px 0; padding: 4px 10px; border-radius: 9px; font-size: .9rem; font-weight: 600;
  border: 1px solid rgba(148,163,184,.3); background: rgba(148,163,184,.10); }
.tok.p { background: rgba(16,185,129,.16); border-color: rgba(16,185,129,.55); color: #10B981; }
.tok.n { background: rgba(244,63,94,.16); border-color: rgba(244,63,94,.55); color: #F43F5E; }
.tok.g { background: rgba(245,158,11,.16); border-color: rgba(245,158,11,.55); color: #F59E0B; }
.tok small { opacity: .8; margin-left: 4px; font-weight: 700; }

/* ---------- widget Streamlit ---------- */
button[data-baseweb="tab"] { border-radius: 999px; padding: .3rem 1rem; }
div[role="radiogroup"] > label { background: rgba(148,163,184,.10); border: 1px solid rgba(148,163,184,.25);
  padding: .22rem .8rem; border-radius: 999px; margin-right: .2rem; transition: background .2s, border-color .2s; }
div[role="radiogroup"] > label:hover { border-color: __A1__; background: __A1__22; }
button[kind="primary"], button[data-testid="stBaseButton-primary"] { background: linear-gradient(135deg, __A1__, __A2__) !important;
  border: 0 !important; color: #fff !important; font-weight: 700; border-radius: 12px; }
button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover { filter: brightness(1.1); transform: translateY(-1px); }
section[data-testid="stSidebar"] { border-right: 1px solid rgba(148,163,184,.2); }
div[data-testid="stExpander"] { border-radius: 14px; }
@media (max-width: 760px) { .hero { padding: 28px 22px; } .hero h1 { font-size: 1.8rem; } }
@media (prefers-reduced-motion: reduce) {
  .hero .blob, .kmark, .hero-art .ring { animation: none !important; }
  .hero-art .ln { animation: none !important; stroke-dashoffset: 0; } .hero-art .dot { animation: none !important; opacity: 1; } }
</style>
"""

HERO_SVG = (
    '<svg viewBox="0 0 420 220" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Ilustrasi control chart">'
    '<rect x="1" y="1" width="418" height="218" rx="18" fill="rgba(255,255,255,.04)" stroke="rgba(255,255,255,.14)"/>'
    '<line x1="14" y1="42" x2="406" y2="42" stroke="#FB7185" stroke-width="1.4" stroke-dasharray="5 5"/>'
    '<line x1="14" y1="112" x2="406" y2="112" stroke="#34D399" stroke-width="1.6"/>'
    '<line x1="14" y1="182" x2="406" y2="182" stroke="#FB7185" stroke-width="1.4" stroke-dasharray="5 5"/>'
    '<text x="22" y="36" fill="#FDA4AF" font-size="10" font-weight="700">UCL</text>'
    '<text x="22" y="106" fill="#6EE7B7" font-size="10" font-weight="700">CL</text>'
    '<text x="22" y="196" fill="#FDA4AF" font-size="10" font-weight="700">LCL</text>'
    '<path class="ln" pathLength="1" d="M20,128 L62,104 L104,134 L146,98 L188,120 L230,92 L272,124 L314,100 L346,76 L375,22" '
    'fill="none" stroke="#fff" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/>'
    '<circle class="dot" cx="375" cy="22" r="6.5" fill="#F43F5E" stroke="#fff" stroke-width="2"/>'
    '<circle class="ring" cx="375" cy="22" r="9" fill="none" stroke="#F43F5E" stroke-width="2"/>'
    '</svg>'
)


def inject_css():
    a1, a2 = accents()
    html_block(CSS.replace("__A1__", a1).replace("__A2__", a2))


def hero():
    chips = "".join(f'<span class="chip">{c}</span>' for c in
                    ["IndoBERT", "Lexicon InSet", "Cohen's Kappa", "Control chart p", "Interaktif"])
    html_block(
        '<div class="hero"><div class="blob b1"></div><div class="blob b2"></div><div class="hero-grid">'
        '<div class="hero-text"><h1>Pantau sentimen ulasan aplikasi belanja, dari teks sampai batas kendali.</h1>'
        '<p>Bandingkan label lexicon dengan rating bintang, latih IndoBERT, dan deteksi hari ketika ulasan negatif '
        'melonjak di luar kewajaran. Untuk ulasan Klik Indomaret dan Alfagift.</p>'
        f'<div class="chips">{chips}</div></div>'
        f'<div class="hero-art">{HERO_SVG}</div></div></div>')


def page_header(icon, title, sub):
    html_block(f'<div class="phead"><div class="ic">{icon}</div><div><h2>{title}</h2><p>{sub}</p></div></div>')


def section(title, sub=None):
    html_block(f'<div class="sec"><div class="bar"></div><h3>{title}</h3></div>')
    if sub:
        html_block(f'<div class="sec-sub">{sub}</div>')


def note(text):
    html_block(f'<div class="note">{text}</div>')


def empty_state():
    html_block('<div class="empty"><div class="big">🗂️</div><h3>Belum ada data</h3>'
               '<p>Buka sidebar → <b>Data</b>, lalu unggah CSV, isi path file, atau aktifkan <b>data contoh</b>.</p></div>')


KPI_HTML = """
<style>
* { box-sizing: border-box; font-family: 'Plus Jakarta Sans','Segoe UI',system-ui,sans-serif; }
body { margin: 0; background: transparent; }
.row { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 14px; }
.k { background: __BG__; border: 1px solid __BR__; border-radius: 18px; padding: 16px 18px; color: __TX__;
     position: relative; overflow: hidden; }
.k::before { content: ''; position: absolute; left: 0; top: 0; bottom: 0; width: 4px; background: linear-gradient(180deg, __A1__, __A2__); }
.lab { font-size: 12.5px; font-weight: 600; opacity: .75; display: flex; gap: 8px; align-items: center; }
.val { font-size: 31px; font-weight: 800; margin: 6px 0 2px 0; letter-spacing: -.02em; line-height: 1.15;
       background: linear-gradient(90deg, __A1__, __A2__); -webkit-background-clip: text; background-clip: text; color: transparent; }
.sub { font-size: 12px; opacity: .65; }
</style>
<div class="row">__CARDS__</div>
<script>
const reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
document.querySelectorAll('.num').forEach(function(el){
  const t = parseFloat(el.dataset.t), d = parseInt(el.dataset.d || '0');
  const fmt = function(v){ return v.toLocaleString('id-ID', {minimumFractionDigits: d, maximumFractionDigits: d}); };
  if (isNaN(t) || reduce) { el.textContent = isNaN(t) ? '—' : fmt(t); return; }
  const s = performance.now(), dur = 1500;
  (function f(now){ const p = Math.min((now - s) / dur, 1), e = 1 - Math.pow(1 - p, 3);
    el.textContent = fmt(t * e); if (p < 1) requestAnimationFrame(f); })(s);
});
</script>
"""


def kpi_row(items: list[dict], height=128):
    """items: label, value (angka / None), decimals, prefix, suffix, sub, icon"""
    dark = is_dark()
    a1, a2 = accents()
    cards = []
    for it in items:
        v = it.get("value")
        shown = "" if v is None else f"{v}"
        txt = "—" if v is None else ""
        cards.append(
            f'<div class="k"><div class="lab"><span>{it.get("icon", "")}</span>{_html.escape(it["label"])}</div>'
            f'<div class="val">{it.get("prefix", "")}<span class="num" data-t="{shown}" data-d="{it.get("decimals", 0)}">{txt}</span>'
            f'{it.get("suffix", "")}</div><div class="sub">{_html.escape(it.get("sub", ""))}</div></div>')
    page = (KPI_HTML.replace("__CARDS__", "".join(cards)).replace("__A1__", a1).replace("__A2__", a2)
            .replace("__TX__", "#F1F5F9" if dark else "#0F172A")
            .replace("__BG__", "rgba(148,163,184,.12)").replace("__BR__", "rgba(148,163,184,.28)"))
    components.html(page, height=height)


def kappa_meter(kappa: float):
    pos = max(0.0, min(1.0, kappa)) * 100
    html_block(
        f'<div class="kmeter"><div class="kbar"><div class="kmark" style="--to:{pos:.1f}%"></div></div>'
        '<div class="kscale"><span>0 lemah</span><span>0,2</span><span>0,4 sedang</span><span>0,6</span>'
        '<span>0,8</span><span>1 sangat kuat</span></div></div>')


# =============================================================================
# SIDEBAR - SEMUA KONFIGURASI
# =============================================================================
def _import_cfg():
    f = st.session_state.get("cfg_import_file")
    if f is None:
        return
    try:
        data = json.load(f)
        for k, v in data.items():
            if k in DEFAULTS and (v is None or isinstance(v, type(DEFAULTS[k])) or
                                  (isinstance(DEFAULTS[k], float) and isinstance(v, (int, float)))):
                st.session_state[K(k)] = v
        st.session_state["_cfg_msg"] = "Konfigurasi berhasil diimpor."
    except Exception as e:
        st.session_state["_cfg_msg"] = f"Gagal impor: {e}"


def _reset_cfg():
    for k, v in DEFAULTS.items():
        st.session_state[K(k)] = v


def _col_select(label, key, options, guesses):
    if st.session_state.get(key) not in options:
        st.session_state[key] = next((g for g in guesses if g in options), options[0])
    st.selectbox(label, options, key=key)


def dataset_slot(i: int):
    p = f"ds{i}_"
    st.text_input("Nama dataset", key=K(p + "name"))
    up = st.file_uploader("Unggah CSV", type=["csv", "tsv", "txt"], key=f"cfg_up{i}")
    st.text_input("…atau path file lokal", key=K(p + "path"), help="Dipakai bila tidak ada file yang diunggah.")
    raw, origin = None, None
    path = C(p + "path")
    if up is not None:
        raw, origin = up.getvalue(), f"unggahan · {up.name}"
    elif path and os.path.isfile(path):
        with open(path, "rb") as f:
            raw = f.read()
        origin = f"file lokal · {path}"
    elif C("use_demo"):
        raw, origin = demo_csv(i), "data contoh (sintetis)"
    if raw is None:
        st.warning("Belum ada data untuk dataset ini.")
        return None
    try:
        df = read_csv_bytes(raw)
    except Exception as e:
        st.error(f"CSV tidak bisa dibaca: {e}")
        return None
    cols = list(df.columns)
    st.caption(f"✅ {origin} · {len(df):,} baris · {len(cols)} kolom")
    _col_select("Kolom teks ulasan", K(p + "text"), cols, ["ulasan", "review", "teks", "text", "content"])
    star_cols = [NONE_OPT] + cols
    _col_select("Kolom rating bintang (opsional)", K(p + "rating"), star_cols, ["rating", "score", "bintang"])
    lab_opts = cols + [STAR_OPT]
    _col_select("Kolom label rating", K(p + "label"), lab_opts,
                ["sentimen", "label_sentimen_rating", "label_rating", "label", "sentiment"])
    _col_select("Kolom tanggal", K(p + "date"), [NONE_OPT] + cols,
                ["tanggal", "tanggal_ulasan", "date", "at", "created_at", "waktu"])
    return {"name": C(p + "name"), "raw": raw, "origin": origin, "text": C(p + "text"), "label": C(p + "label"),
            "rating": C(p + "rating"), "date": C(p + "date")}


def build_sidebar():
    with st.sidebar:
        a1, a2 = accents()
        html_block(f'<div style="font-weight:800;font-size:1.35rem;letter-spacing:-.02em;margin:2px 0 2px 0;'
                   f'background:linear-gradient(90deg,{a1},{a2});-webkit-background-clip:text;background-clip:text;'
                   f'color:transparent;">◢ {APP_NAME}</div>'
                   '<div style="opacity:.65;font-size:.82rem;margin-bottom:10px;">Panel konfigurasi</div>')

        with st.expander("🎨 Tampilan & animasi"):
            st.selectbox("Palet warna", list(PALETTES), key=K("palette"))
            st.radio("Tema grafik", ["Otomatis", "Gelap", "Terang"], horizontal=True, key=K("chart_theme"),
                     help="Otomatis mengikuti tema Streamlit (menu ⋮ → Settings).")
            st.toggle("Animasi grafik", key=K("animate"))
            st.toggle("Putar otomatis saat dibuka", key=K("autoplay"),
                      help="Aktif: grafik beranimasi langsung (memuat plotly.js dari CDN). "
                           "Nonaktif: tekan tombol ▶ Putar pada grafik.")
            st.slider("Kecepatan animasi (ms/frame)", 20, 300, step=10, key=K("frame_ms"))
            st.slider("Tinggi grafik (px)", 320, 760, step=20, key=K("chart_h"))

        cfgs = []
        with st.expander("🗂️ Data", expanded=True):
            st.toggle("Pakai data contoh bila file tidak ada", key=K("use_demo"))
            st.toggle("Aktifkan dataset ke-2", key=K("ds2_enabled"))
            tabs = st.tabs(["Dataset 1", "Dataset 2"] if C("ds2_enabled") else ["Dataset 1"])
            with tabs[0]:
                c1 = dataset_slot(1)
            if c1:
                cfgs.append(c1)
            if C("ds2_enabled"):
                with tabs[1]:
                    c2 = dataset_slot(2)
                if c2:
                    if cfgs and c2["name"] == cfgs[0]["name"]:
                        c2["name"] += " (2)"
                    cfgs.append(c2)
        st.session_state["_ds_cfgs"] = cfgs

        with st.expander("🔤 Lexicon (InSet)"):
            st.selectbox("Sumber kamus", INSET_MODES, key=K("inset_mode"))
            if C("inset_mode") == INSET_MODES[1]:
                st.file_uploader("positive.tsv", type=["tsv", "txt"], key="cfg_inset_pos")
                st.file_uploader("negative.tsv", type=["tsv", "txt"], key="cfg_inset_neg")
            st.toggle("Penyesuaian domain belanja online", key=K("use_domain"),
                      help="Matikan untuk memakai InSet murni.")
            st.slider("Margin netral (skor −m … +m dianggap netral)", 0, 5, key=K("neutral_margin"))
            st.slider("Jangkauan negasi (jumlah kata)", 1, 5, key=K("neg_window"))
            st.text_area("Kata negasi (pisahkan koma)", key=K("negators"), height=70)
            st.text_area("Tambahan / koreksi kata (satu baris: kata: bobot)", key=K("extra_lex"), height=90,
                         placeholder="lemot: -4\nmantul: 3")

        with st.expander("🧠 Model IndoBERT"):
            st.text_input("Nama model (Hugging Face)", key=K("model_name"))
            c1, c2 = st.columns(2)
            c1.number_input("Max token", 16, 512, step=16, key=K("max_len"))
            c2.number_input("Epoch", 1, 30, key=K("epochs"))
            c1.number_input("Batch size", 2, 128, key=K("batch"))
            c2.number_input("Seed", 0, 100000, key=K("seed"))
            st.select_slider("Learning rate", options=LR_OPTIONS, format_func=lambda x: f"{x:.0e}", key=K("lr"))
            st.number_input("Weight decay", 0.0, 0.5, step=0.005, format="%.3f", key=K("weight_decay"))
            st.slider("Warm-up (porsi langkah)", 0.0, 0.5, step=0.01, key=K("warmup"))
            st.slider("Early stopping patience", 1, 6, key=K("patience"))
            st.toggle("Bobot kelas pada loss", key=K("class_weight"))
            st.multiselect("Sumber label yang dilatih", ["lexicon", "rating"], key=K("label_sources"))
            st.caption("Pembagian data (rasio, otomatis dinormalisasi)")
            s1, s2, s3 = st.columns(3)
            s1.number_input("Train", 1, 98, key=K("split_train"))
            s2.number_input("Val", 1, 98, key=K("split_val"))
            s3.number_input("Test", 1, 98, key=K("split_test"))

        with st.expander("📉 Control chart p"):
            st.selectbox("Kelas 'cacat'", LABELS, key=K("p_defect"))
            st.multiselect("Sumber sentimen", P_SOURCES_ALL, key=K("p_sources"),
                           help="prediksi_* membutuhkan model yang sudah dilatih.")
            st.radio("Subgroup", FREQS, horizontal=True, key=K("p_freq"))
            st.slider("Lebar batas kendali (k sigma)", 1.0, 4.0, step=0.5, key=K("p_k"))
            st.number_input("Minimal ulasan per subgroup", 1, 200, key=K("p_min_n"))

        with st.expander("💾 Output & konfigurasi"):
            st.text_input("Folder output", key=K("output_dir"))
            st.download_button("⬇️ Ekspor konfigurasi (JSON)",
                               json.dumps({k: st.session_state.get(K(k), v) for k, v in DEFAULTS.items()},
                                          indent=2, ensure_ascii=False, default=str),
                               file_name="konfigurasi_sentiscope.json", mime="application/json")
            st.file_uploader("⬆️ Impor konfigurasi", type=["json"], key="cfg_import_file", on_change=_import_cfg)
            st.button("↺ Reset semua ke bawaan", on_click=_reset_cfg)
            if st.session_state.get("_cfg_msg"):
                st.caption(st.session_state.pop("_cfg_msg"))
            if st.button("🗑️ Kosongkan cache data"):
                st.cache_data.clear()
                st.rerun()


# =============================================================================
# HALAMAN
# =============================================================================
def pick_dataset(ctx, key):
    names = list(ctx)
    if len(names) == 1:
        return names[0]
    return st.radio("Dataset", names, horizontal=True, key=key)


def agreement(d):
    a, b = d["label_rating_str"], d["label_lexicon_str"]
    return float(accuracy_score(a, b)), float(cohen_kappa_score(a, b))


# ----------------------------------------------------------------------------- Beranda
def page_home():
    ctx = get_ctx()
    hero()
    results = st.session_state["results"]
    total = sum(len(v["data"]) for v in ctx.values())
    ag = [agreement(v["data"]) for v in ctx.values()]
    f1s = [r["summary"]["f1_macro"] for r in results.values()]
    kpi_row([
        {"icon": "💬", "label": "Total ulasan dianalisis", "value": total, "sub": f"{len(ctx)} dataset aktif"},
        {"icon": "🤝", "label": "Kesesuaian lexicon vs rating", "value": round(np.mean([a for a, _ in ag]) * 100, 1),
         "decimals": 1, "suffix": "%", "sub": f"Kappa rata-rata {np.mean([k for _, k in ag]):.3f}"},
        {"icon": "🧠", "label": "F1-macro IndoBERT terbaik", "value": round(max(f1s) * 100, 1) if f1s else None,
         "decimals": 1, "suffix": "%" if f1s else "", "sub": f"{len(f1s)} model dilatih" if f1s else "Belum ada model dilatih"},
        {"icon": "📚", "label": "Entri kamus lexicon", "value": st.session_state["_lex"]["n"],
         "sub": st.session_state["_lex"]["status"][:48]},
    ])

    section("Alur kerja", "Sama dengan notebook, dari data mentah sampai pemantauan harian.")
    steps = [("Data & Lexicon", "Ulasan diberi label dari isi teks memakai kamus InSet, lengkap dengan normalisasi kata gaul dan negasi."),
             ("Lexicon vs Rating", "Label dari teks dibandingkan dengan label bintang memakai kesesuaian dan Cohen's Kappa."),
             ("Latih IndoBERT", "Dua sumber label dilatih pada tiap dataset, lalu dievaluasi dengan confusion matrix pada data test."),
             ("Control chart p", "Proporsi ulasan negatif per hari dipantau; hari di luar UCL/LCL ditandai untuk ditelusuri.")]
    cols = st.columns(4)
    for i, (c, (t, s)) in enumerate(zip(cols, steps), 1):
        with c:
            html_block(f'<div class="card"><div class="step-no">{i}</div><h4>{t}</h4><p>{s}</p></div>')

    section("Komposisi sentimen", "Persentase tiap kelas menurut label lexicon dan label rating.")
    fig = go.Figure()
    ylabels = []
    for name, v in ctx.items():
        for src in ("lexicon", "rating"):
            ylabels.append(f"{name} · {src}")
    for lab in LABELS:
        xs = []
        for name, v in ctx.items():
            for src in ("lexicon", "rating"):
                s = v["data"][f"label_{src}_str"]
                xs.append(float((s == lab).mean() * 100))
        fig.add_trace(go.Bar(y=ylabels, x=xs, name=lab, orientation="h", marker=dict(color=LABEL_COLORS[lab]),
                             text=[f"{x:.1f}%" for x in xs], textposition="inside", insidetextanchor="middle",
                             hovertemplate="%{y}<br>" + lab + ": %{x:.1f}%<extra></extra>"))
    fig.update_layout(barmode="stack")
    themed(fig, height=max(260, 90 * len(ylabels)))
    fig.update_xaxes(range=[0, 100], ticksuffix="%")
    fig.update_yaxes(autorange="reversed")
    add_grow(fig, horizontal=True, frame_ms=int(C("frame_ms")))
    show(fig, key="home_comp", autoplay=True)

    section("Konfigurasi aktif", "Ubah semuanya lewat sidebar. Pengaturan bisa diekspor sebagai JSON.")
    c1, c2, c3, c4 = st.columns(4)
    r = C("split_train") + C("split_val") + C("split_test")
    for c, (t, s) in zip([c1, c2, c3, c4], [
        ("Model", f"{C('model_name').split('/')[-1]}<br>max {C('max_len')} token · {C('epochs')} epoch"),
        ("Optimasi", f"batch {C('batch')} · lr {C('lr'):.0e}<br>bobot kelas: {'ya' if C('class_weight') else 'tidak'}"),
        ("Pembagian data", f"train {C('split_train') / r:.0%} · val {C('split_val') / r:.0%} · test {C('split_test') / r:.0%}"),
        ("Control chart", f"{C('p_freq').lower()} · {C('p_k'):g}σ · kelas {C('p_defect')}")]):
        with c:
            html_block(f'<div class="card"><h4>{t}</h4><p>{s}</p></div>')


# ----------------------------------------------------------------------------- Data & Lexicon
def page_data():
    page_header("🗂️", "Data & Lexicon", "Telusuri ulasan, distribusi label, kata pemicu sentimen, dan uji kamus secara langsung.")
    ctx = get_ctx()
    lx = st.session_state["_lex"]
    ds = pick_dataset(ctx, "p2_ds")
    d, raw, info = ctx[ds]["data"], ctx[ds]["raw"], ctx[ds]["info"]
    st.markdown(f'<span class="pill ok">{ctx[ds]["origin"]}</span>'
                f'<span class="pill {"ok" if lx["inset_ok"] else "warn"}">{lx["status"]}</span>', unsafe_allow_html=True)
    if info["dibuang"]:
        st.caption(f"{info['dibuang']:,} baris dibuang karena label/teks kosong · {info['tgl_gagal']:,} tanggal gagal dibaca.")

    view = st.radio("Bagian", ["Ringkasan", "Perlombaan sentimen", "Kata kunci", "Uji lexicon", "Tabel data"],
                    horizontal=True, key="p2_view", label_visibility="collapsed")

    if view == "Ringkasan":
        dmin, dmax = d["tanggal"].min(), d["tanggal"].max()
        rng = f"{dmin:%d %b %Y} – {dmax:%d %b %Y}" if pd.notna(dmin) else "—"
        neg_share = float((d["label_lexicon_str"] == "negatif").mean() * 100)
        kpi_row([
            {"icon": "💬", "label": "Ulasan dipakai", "value": len(d), "sub": f"dari {info['n_raw']:,} baris"},
            {"icon": "🗓️", "label": "Rentang tanggal", "value": None, "sub": rng},
            {"icon": "🔻", "label": "Negatif (lexicon)", "value": round(neg_share, 1), "decimals": 1, "suffix": "%",
             "sub": f"rating: {(d['label_rating_str'] == 'negatif').mean() * 100:.1f}% negatif"},
            {"icon": "📐", "label": "Rata-rata skor lexicon", "value": round(float(d["skor"].mean()), 2), "decimals": 2,
             "sub": f"median {d['skor'].median():.0f}"},
        ])
        c1, c2 = st.columns([1, 1.15])
        with c1:
            fig = make_subplots(rows=1, cols=2, specs=[[{"type": "domain"}, {"type": "domain"}]],
                                subplot_titles=("Lexicon", "Rating"))
            for j, src in enumerate(("lexicon", "rating"), 1):
                vc = d[f"label_{src}_str"].value_counts().reindex(LABELS, fill_value=0)
                fig.add_trace(go.Pie(labels=vc.index, values=vc.values, hole=.62, sort=False,
                                     marker=dict(colors=[LABEL_COLORS[l] for l in vc.index],
                                                 line=dict(color="rgba(0,0,0,0)", width=2)),
                                     textinfo="percent", hovertemplate="%{label}: %{value:,} (%{percent})<extra></extra>"),
                              row=1, col=j)
            themed(fig, height=340, title="Distribusi label")
            fig.update_traces(pull=0.02)
            show(fig, key="d_pie")
        with c2:
            fig = go.Figure()
            for lab in LABELS:
                vals = [int((d[f"label_{s}_str"] == lab).sum()) for s in ("lexicon", "rating")]
                fig.add_trace(go.Bar(x=["Lexicon", "Rating"], y=vals, name=lab, marker=dict(color=LABEL_COLORS[lab]),
                                     text=[f"{v:,}" for v in vals], textposition="outside",
                                     hovertemplate=lab + ": %{y:,}<extra></extra>"))
            fig.update_layout(barmode="group", bargap=.25)
            themed(fig, height=340, title="Jumlah ulasan per label")
            add_grow(fig, frame_ms=int(C("frame_ms")))
            show(fig, key="d_bar", autoplay=True)

        dd = d.dropna(subset=["tanggal"])
        if not dd.empty:
            section("Volume ulasan harian", "Garis tebal = rata-rata bergerak 7 hari. Seret untuk zoom, klik dua kali untuk reset.")
            daily = (dd.assign(hari=dd["tanggal"].dt.normalize())
                     .groupby(["hari", "label_lexicon_str"]).size().unstack(fill_value=0)
                     .reindex(columns=LABELS, fill_value=0))
            full_idx = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
            daily = daily.reindex(full_idx, fill_value=0)
            fig = go.Figure()
            for lab in LABELS:
                fig.add_trace(go.Scatter(x=daily.index, y=daily[lab], name=lab, stackgroup="one", mode="lines",
                                         line=dict(width=.6, color=LABEL_COLORS[lab]),
                                         fillcolor=hex_to_rgba(LABEL_COLORS[lab], .55),
                                         hovertemplate="%{x|%d %b %Y}<br>" + lab + ": %{y}<extra></extra>"))
            ma = daily.sum(axis=1).rolling(7, min_periods=1).mean()
            fig.add_trace(go.Scatter(x=daily.index, y=ma, name="rata-rata 7 hari", mode="lines",
                                     line=dict(color=accents()[1], width=3, dash="dot"),
                                     hovertemplate="MA7: %{y:.1f}<extra></extra>"))
            themed(fig, height=int(C("chart_h")) - 60)
            fig.update_xaxes(rangeslider=dict(visible=True, thickness=.07))
            fig.update_layout(hovermode="x unified")
            show(fig, key="d_daily")

            if (dd["tanggal"].dt.hour != 0).any():
                section("Pola waktu", "Kapan ulasan paling banyak masuk (hari × jam).")
                pv = (dd.assign(hr=dd["tanggal"].dt.dayofweek, jam=dd["tanggal"].dt.hour)
                      .pivot_table(index="hr", columns="jam", values="text", aggfunc="size", fill_value=0)
                      .reindex(index=range(7), columns=range(24), fill_value=0))
                fig = go.Figure(go.Heatmap(
                    z=pv.values, x=[f"{h:02d}" for h in range(24)],
                    y=["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"], xgap=3, ygap=3,
                    colorscale=[[0, hex_to_rgba(accents()[0], .06)], [1, accents()[0]]],
                    hovertemplate="%{y} pukul %{x}:00<br>%{z} ulasan<extra></extra>", colorbar=dict(thickness=10)))
                themed(fig, height=320, legend=False)
                fig.update_yaxes(autorange="reversed")
                fig.update_xaxes(title_text="Jam")
                show(fig, key="d_heat")

        section("Sebaran skor lexicon", "Skor 0 dianggap netral (bila margin = 0).")
        fig = px.histogram(d, x="skor", color="label_lexicon_str", color_discrete_map=LABEL_COLORS,
                           category_orders={"label_lexicon_str": LABELS}, nbins=40, marginal="box", opacity=.85)
        themed(fig, height=380)
        fig.update_layout(barmode="overlay", legend_title_text="")
        show(fig, key="d_hist")

    elif view == "Perlombaan sentimen":
        section("Perlombaan sentimen kumulatif", "Tekan ▶ Putar: jumlah ulasan tiap kelas bertambah mengikuti waktu.")
        dd = d.dropna(subset=["tanggal"])
        if dd.empty:
            st.info("Kolom tanggal tidak tersedia / tidak terbaca, jadi animasi waktu tidak bisa dibuat.")
        else:
            span = (dd["tanggal"].max() - dd["tanggal"].min()).days
            freq, fmt = ("D", "%d %b %Y") if span <= 21 else (("W", "%d %b %Y") if span <= 180 else ("M", "%b %Y"))
            per = dd["tanggal"].dt.to_period(freq).dt.start_time
            ct = (pd.crosstab(per, dd["label_lexicon_str"]).reindex(columns=LABELS, fill_value=0).sort_index().cumsum())
            long = ct.reset_index(names="periode").melt(id_vars="periode", var_name="sentimen", value_name="jumlah")
            long["frame"] = long["periode"].dt.strftime(fmt)
            fig = px.bar(long, x="sentimen", y="jumlah", color="sentimen", color_discrete_map=LABEL_COLORS,
                         animation_frame="frame", range_y=[0, float(ct.values.max()) * 1.15 + 1],
                         category_orders={"sentimen": LABELS}, text="jumlah")
            fig.update_traces(textposition="outside", hovertemplate="%{x}: %{y:,}<extra></extra>")
            themed(fig, height=int(C("chart_h")) + 40, legend=False)
            slow = max(250, int(C("frame_ms")) * 8)
            try:
                fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = slow
                fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = int(slow * .7)
                fig.layout.sliders[0].currentvalue.prefix = "Periode: "
            except Exception:
                pass
            show(fig, key="d_race", autoplay=True,
                 animation_opts={"frame": {"duration": slow, "redraw": True},
                                 "transition": {"duration": int(slow * .7), "easing": "cubic-in-out"},
                                 "fromcurrent": False, "mode": "immediate"})
            st.caption(f"Satu frame = satu {'hari' if freq == 'D' else 'minggu' if freq == 'W' else 'bulan'}; {len(ct)} frame.")

    elif view == "Kata kunci":
        section("Kata pemicu sentimen", "Dihitung dari kata yang benar-benar menyumbang skor (setelah negasi diterapkan).")
        rows = []
        for s in d["kata_sentimen"]:
            if not s:
                continue
            for h in s.split(", "):
                m = re.match(r"^(.*)\(([+-]\d+)\)$", h)
                if m:
                    rows.append((m.group(1), int(m.group(2))))
        if not rows:
            st.info("Belum ada kata sentimen yang terdeteksi.")
        else:
            hits = pd.DataFrame(rows, columns=["kata", "bobot"])
            topn = st.slider("Jumlah kata teratas", 5, 30, 12, key="p2_topn")
            c1, c2 = st.columns(2)
            for c, sign, lab, title in [(c1, 1, "positif", "Kata positif terbanyak"), (c2, -1, "negatif", "Kata negatif terbanyak")]:
                sub = hits[(hits["bobot"] * sign) > 0]
                g = sub.groupby("kata").agg(frekuensi=("bobot", "size"), total=("bobot", "sum")).sort_values("frekuensi", ascending=False).head(topn)
                with c:
                    fig = go.Figure(go.Bar(
                        y=g.index, x=g["frekuensi"], orientation="h", marker=dict(color=g["frekuensi"], colorscale=[[0, hex_to_rgba(LABEL_COLORS[lab], .35)], [1, LABEL_COLORS[lab]]]),
                        customdata=g["total"], text=g["frekuensi"], textposition="outside",
                        hovertemplate="<b>%{y}</b><br>muncul %{x}×<br>total bobot %{customdata:+d}<extra></extra>"))
                    themed(fig, height=max(320, 30 * len(g) + 90), title=title, legend=False)
                    fig.update_yaxes(autorange="reversed")
                    add_grow(fig, horizontal=True, frame_ms=int(C("frame_ms")))
                    show(fig, key=f"d_words_{lab}", autoplay=True)

    elif view == "Uji lexicon":
        section("Uji lexicon langsung", "Ketik ulasan; token diwarnai menurut sumbangan skornya.")
        examples = {"(ketik sendiri)": "",
                    "Bagus tapi pengiriman lama": "aplikasinya bagus tapi pengiriman lama banget",
                    "Negasi": "kurang bagus dan tidak ada masalah",
                    "Kata gaul": "gabisa login mulu, udh kesel bgt",
                    "Netral": "saya pesan barang kemarin lewat aplikasi"}
        ex = st.selectbox("Contoh", list(examples), key="p2_ex")
        st.session_state.setdefault("p2_txt", "aplikasinya bagus tapi pengiriman lama banget")
        if st.session_state.get("_last_ex") != ex:           # isi textarea hanya saat contoh berganti
            st.session_state["_last_ex"] = ex
            if examples[ex]:
                st.session_state["p2_txt"] = examples[ex]
        txt = st.text_area("Teks ulasan", key="p2_txt", height=100)
        toks = normalize_tokens(txt)
        score, hits_ = score_tokens(toks, lx["lex"], lx["negators"], int(C("neg_window")))
        label = label_from_score(score, int(C("neutral_margin")))
        cover = {}
        for ph, val, a, b in hits_:
            for t in range(a, b):
                cover[t] = val
        chips = []
        for i, t in enumerate(toks):
            if i in cover:
                cls = "p" if cover[i] > 0 else "n"
                chips.append(f'<span class="tok {cls}">{_html.escape(t)}<small>{cover[i]:+d}</small></span>')
            elif t in lx["negators"]:
                chips.append(f'<span class="tok g">{_html.escape(t)}<small>negasi</small></span>')
            else:
                chips.append(f'<span class="tok">{_html.escape(t)}</span>')
        c1, c2 = st.columns([3, 1])
        with c1:
            html_block("".join(chips) or "<i>Tidak ada token.</i>")
        with c2:
            col = LABEL_COLORS[label]
            html_block(f'<div class="card" style="text-align:center"><div style="font-size:.8rem;opacity:.7">Skor total</div>'
                       f'<div style="font-size:2.3rem;font-weight:800;color:{col}">{score:+d}</div>'
                       f'<span class="pill" style="color:{col};border-color:{col}">{label}</span></div>')

    else:
        section("Tabel data berlabel")
        c1, c2, c3 = st.columns([1, 1, 1.4])
        f_lex = c1.multiselect("Label lexicon", LABELS, default=LABELS, key="p2_flex")
        f_rat = c2.multiselect("Label rating", LABELS, default=LABELS, key="p2_frat")
        q = c3.text_input("Cari kata dalam ulasan", key="p2_q")
        only_diff = st.toggle("Hanya yang labelnya berbeda", key="p2_diff")
        t = d[d["label_lexicon_str"].isin(f_lex) & d["label_rating_str"].isin(f_rat)]
        if only_diff:
            t = t[t["label_lexicon_str"] != t["label_rating_str"]]
        if q:
            t = t[t["text_asli"].astype(str).str.contains(q, case=False, na=False)]
        cols = ["tanggal", "text_asli", "label_rating_str", "skor", "label_lexicon_str", "kata_sentimen"]
        if "rating" in t.columns:
            cols.insert(2, "rating")
        st.caption(f"{len(t):,} ulasan cocok.")
        st.dataframe(t[cols].head(2000), hide_index=True, **_stretch_kw(st.dataframe),
                     column_config={"tanggal": st.column_config.DatetimeColumn("Tanggal", format="DD MMM YYYY HH:mm"),
                                    "text_asli": st.column_config.TextColumn("Ulasan", width="large"),
                                    "label_rating_str": "Label rating", "label_lexicon_str": "Label lexicon",
                                    "skor": st.column_config.NumberColumn("Skor"),
                                    "kata_sentimen": st.column_config.TextColumn("Kata sentimen", width="medium")})


# ----------------------------------------------------------------------------- Lexicon vs Rating
def page_compare():
    page_header("🤝", "Lexicon vs Rating", "Seberapa sering label dari isi teks sama dengan label dari bintang?")
    ctx = get_ctx()
    rows = []
    for name, v in ctx.items():
        a, k = agreement(v["data"])
        rows.append({"dataset": name, "jumlah": len(v["data"]), "kesesuaian": a, "cohen_kappa": k, "tafsir": kappa_label(k)})
    comp = pd.DataFrame(rows)

    kpi_row([{"icon": "📊", "label": f"{r['dataset']} · kesesuaian", "value": round(r["kesesuaian"] * 100, 1), "decimals": 1,
              "suffix": "%", "sub": f"Kappa {r['cohen_kappa']:.3f} ({r['tafsir']})"} for r in rows])

    ds = pick_dataset(ctx, "p3_ds")
    d = ctx[ds]["data"]
    a, k = agreement(d)
    section(f"{ds}: Cohen's Kappa", "Kesesuaian setelah dikoreksi faktor kebetulan.")
    kappa_meter(k)
    html_block(f'<div style="margin-top:10px">Kappa <b>{k:.3f}</b> berarti kesesuaian <b>{kappa_label(k)}</b>.</div>')

    c1, c2 = st.columns(2)
    with c1:
        norm = st.toggle("Tampilkan persen per baris", value=True, key="p3_norm")
        cm = confusion_matrix(d["label_rating_str"], d["label_lexicon_str"], labels=LABELS)
        pctm = cm / cm.sum(axis=1, keepdims=True).clip(min=1) * 100
        z = pctm if norm else cm
        text = [[f"{cm[i, j]:,}<br>({pctm[i, j]:.1f}%)" for j in range(3)] for i in range(3)]
        fig = go.Figure(go.Heatmap(z=z, x=LABELS, y=LABELS, text=text, texttemplate="%{text}", xgap=4, ygap=4,
                                   colorscale=[[0, hex_to_rgba(accents()[0], .06)], [1, accents()[0]]], showscale=False,
                                   textfont=dict(size=14),
                                   hovertemplate="Rating: %{y}<br>Lexicon: %{x}<br>%{text}<extra></extra>"))
        themed(fig, height=400, title="Confusion matrix (baris = rating)", legend=False)
        fig.update_yaxes(autorange="reversed", title_text="Label rating")
        fig.update_xaxes(title_text="Label lexicon")
        show(fig, key="p3_cm")
    with c2:
        ct = pd.crosstab(d["label_rating_str"], d["label_lexicon_str"]).reindex(index=LABELS, columns=LABELS, fill_value=0)
        s, t, v, colr = [], [], [], []
        for i, la in enumerate(LABELS):
            for j, lb in enumerate(LABELS):
                if ct.loc[la, lb] > 0:
                    s.append(i); t.append(3 + j); v.append(int(ct.loc[la, lb])); colr.append(hex_to_rgba(LABEL_COLORS[la], .38))
        fig = go.Figure(go.Sankey(arrangement="snap",
                                  node=dict(label=[f"rating · {l}" for l in LABELS] + [f"lexicon · {l}" for l in LABELS],
                                            color=[LABEL_COLORS[l] for l in LABELS] * 2, pad=18, thickness=18,
                                            line=dict(color="rgba(0,0,0,0)", width=0)),
                                  link=dict(source=s, target=t, value=v, color=colr,
                                            hovertemplate="%{source.label} → %{target.label}<br>%{value:,} ulasan<extra></extra>")))
        themed(fig, height=400, title="Aliran label: rating → lexicon", legend=False)
        show(fig, key="p3_sankey")

    if len(comp) > 1:
        section("Perbandingan antar dataset")
        fig = go.Figure()
        for m, nm, col in [("kesesuaian", "Kesesuaian", accents()[0]), ("cohen_kappa", "Cohen's Kappa", accents()[1])]:
            fig.add_trace(go.Bar(x=comp["dataset"], y=comp[m], name=nm, marker=dict(color=col),
                                 text=[f"{x:.3f}" for x in comp[m]], textposition="outside",
                                 hovertemplate="%{x}: %{y:.3f}<extra>" + nm + "</extra>"))
        fig.update_layout(barmode="group", bargap=.35)
        themed(fig, height=340)
        fig.update_yaxes(range=[0, 1.1])
        add_grow(fig, frame_ms=int(C("frame_ms")))
        show(fig, key="p3_cmp", autoplay=True)

    section("Ulasan yang labelnya berbeda", "Bahan pembahasan untuk laporan.")
    diff = d[d["label_rating_str"] != d["label_lexicon_str"]]
    c1, c2, c3 = st.columns([1, 1, 1])
    fr = c1.selectbox("Label rating", ["(semua)"] + LABELS, key="p3_fr")
    fl = c2.selectbox("Label lexicon", ["(semua)"] + LABELS, key="p3_fl")
    nshow = c3.slider("Jumlah contoh", 5, 100, 15, key="p3_n")
    if fr != "(semua)":
        diff = diff[diff["label_rating_str"] == fr]
    if fl != "(semua)":
        diff = diff[diff["label_lexicon_str"] == fl]
    st.caption(f"{len(diff):,} ulasan berbeda label.")
    if len(diff):
        cols = ["text_asli", "label_rating_str", "label_lexicon_str", "skor", "kata_sentimen"]
        if "rating" in diff.columns:
            cols.insert(1, "rating")
        st.dataframe(diff[cols].sample(min(nshow, len(diff)), random_state=int(C("seed"))), hide_index=True,
                     **_stretch_kw(st.dataframe),
                     column_config={"text_asli": st.column_config.TextColumn("Ulasan", width="large"),
                                    "label_rating_str": "Label rating", "label_lexicon_str": "Label lexicon",
                                    "kata_sentimen": "Kata sentimen"})
    note("Ketidaksesuaian itu wajar: rating mencerminkan kepuasan umum, sedangkan lexicon membaca kata-kata dalam teks. "
         "Contoh: bintang 5 dengan ulasan <i>pengiriman lama tapi barang bagus</i> bisa berlabel netral atau negatif menurut lexicon.")


# ----------------------------------------------------------------------------- Latih IndoBERT
def learning_curve_fig(hist: pd.DataFrame, title: str):
    if hist.empty:
        return None
    ep = list(hist["epoch"])
    best = float(hist.loc[hist["val_f1_macro"].idxmax(), "epoch"])
    a1, a2 = accents()
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Loss per epoch", "Metrik validasi per epoch"), horizontal_spacing=.1)
    series = [("train loss", "train_loss", a1, 1), ("validation loss", "val_loss", "#F43F5E", 1),
              ("val accuracy", "val_accuracy", a2, 2), ("val F1-macro", "val_f1_macro", "#F59E0B", 2)]
    for nm, colname, color, c in series:
        fig.add_trace(go.Scatter(x=ep, y=hist[colname], name=nm, mode="lines+markers", line=dict(color=color, width=3),
                                 marker=dict(size=9), hovertemplate="epoch %{x}<br>" + nm + ": %{y:.4f}<extra></extra>"),
                      row=1, col=c)
    fig.add_vline(x=best, line_dash="dash", line_color="rgba(148,163,184,.8)", row=1, col=2,
                  annotation_text="epoch terbaik", annotation_position="top")
    themed(fig, height=400, title=title)
    fig.update_xaxes(tickmode="array", tickvals=ep, title_text="Epoch")
    fig.update_yaxes(range=[0, 1.02], row=1, col=2)
    if C("animate") and len(ep) > 1:
        frames = []
        for kk in range(1, len(ep) + 1):
            frames.append(go.Frame(name=str(kk), traces=[0, 1, 2, 3],
                                   data=[go.Scatter(x=ep[:kk], y=list(hist[c][:kk])) for _, c, _, _ in series]))
        fig.frames = frames
        add_play_button(fig, max(350, int(C("frame_ms")) * 8))
    return fig


def page_train():
    page_header("🧠", "Latih IndoBERT", "Setiap dataset × sumber label dilatih terpisah; hasilnya dipakai halaman Evaluasi dan p-chart.")
    ctx = get_ctx()
    results = st.session_state["results"]
    sources = C("label_sources") or []
    jobs_all = [(ds, s) for ds in ctx for s in sources]
    ratios = {"train": C("split_train"), "val": C("split_val"), "test": C("split_test")}

    ready = torch_ready()
    dev = "—"
    if ready:
        import torch
        dev = f"GPU · {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU (lambat)"
    st.markdown(f'<span class="pill {"ok" if ready else "bad"}">{"PyTorch & Transformers terpasang" if ready else "PyTorch/Transformers belum terpasang"}</span>'
                f'<span class="pill {"ok" if "GPU" in dev else "warn"}">Perangkat: {dev}</span>'
                f'<span class="pill">{len(jobs_all)} eksperimen</span>', unsafe_allow_html=True)
    if not ready:
        note("Untuk melatih model, pasang dahulu: <code>pip install torch transformers accelerate</code>. "
             "Halaman lain tetap berfungsi penuh memakai label lexicon/rating.")
    if not sources:
        st.warning("Pilih minimal satu sumber label di sidebar → Model IndoBERT.")

    section("Pembagian data", "Stratified menurut label lexicon, sama untuk kedua sumber label.")
    ds_prev = pick_dataset(ctx, "p4_ds")
    tr, va, te = split_data(ctx[ds_prev]["data"], ratios, int(C("seed")))
    fig = go.Figure()
    for lab in LABELS:
        fig.add_trace(go.Bar(y=["train", "val", "test"], orientation="h", name=lab, marker=dict(color=LABEL_COLORS[lab]),
                             x=[int((p["label_lexicon_str"] == lab).sum()) for p in (tr, va, te)],
                             hovertemplate="%{y} · " + lab + ": %{x:,}<extra></extra>"))
    fig.update_layout(barmode="stack")
    themed(fig, height=260, title=f"{ds_prev}: label lexicon per bagian ({len(tr):,} / {len(va):,} / {len(te):,})")
    fig.update_yaxes(autorange="reversed")
    add_grow(fig, horizontal=True, frame_ms=int(C("frame_ms")))
    show(fig, key="p4_split", autoplay=True)

    section("Jalankan pelatihan")
    labels_jobs = [f"{ds} · label {s}" for ds, s in jobs_all]
    if "p4_jobs" in st.session_state:
        st.session_state["p4_jobs"] = [x for x in st.session_state["p4_jobs"] if x in labels_jobs]
    else:
        st.session_state["p4_jobs"] = list(labels_jobs)
    chosen = st.multiselect("Eksperimen yang dijalankan", labels_jobs, key="p4_jobs")
    c1, c2, _ = st.columns([1, 1, 2])
    go_train = c1.button("🚀 Mulai pelatihan", type="primary", disabled=not (ready and chosen))
    if c2.button("📂 Muat hasil tersimpan"):
        n = load_saved_results()
        st.success(f"{n} hasil dimuat dari '{C('output_dir')}'.") if n else st.warning("Tidak ada hasil tersimpan.")
        results = st.session_state["results"]

    if go_train:
        sel = [j for j, l in zip(jobs_all, labels_jobs) if l in chosen]
        overall = st.progress(0.0, text="Menyiapkan…")
        ui = {"prog": st.progress(0.0, text="Memuat model…"), "table": st.empty(), "counter": [0]}
        t0 = time.time()
        try:
            for n, (ds, src) in enumerate(sel):
                overall.progress(n / len(sel), text=f"Eksperimen {n + 1}/{len(sel)}: {ds} · {src}")
                ui["table"].empty()
                res = run_experiment(ds, src, ctx[ds]["data"], split_data(ctx[ds]["data"], ratios, int(C("seed"))), ui)
                st.session_state["results"][(ds, src)] = res
                persist_results()
            overall.progress(1.0, text=f"Selesai dalam {(time.time() - t0) / 60:.1f} menit")
            ui["prog"].progress(1.0, text="Semua eksperimen selesai")
            st.balloons()
            st.success("Pelatihan selesai. Lihat halaman **Evaluasi**.")
        except Exception as e:
            st.error(f"Pelatihan berhenti: {type(e).__name__}: {e}")
        results = st.session_state["results"]

    if results:
        section("Kurva belajar", "Tekan ▶ Putar untuk melihat kurva tergambar epoch demi epoch.")
        keys = list(results)
        sel_k = st.selectbox("Eksperimen", keys, format_func=lambda k: f"{k[0]} · label {k[1]}", key="p4_curve")
        r = results[sel_k]
        fig = learning_curve_fig(r["hist"], f"{sel_k[0]} · label {sel_k[1]}")
        if fig is not None:
            show(fig, key="p4_curve_fig", autoplay=True,
                 animation_opts={"frame": {"duration": 500, "redraw": True},
                                 "transition": {"duration": 350, "easing": "cubic-in-out"}, "fromcurrent": False, "mode": "immediate"})
        st.dataframe(r["hist"].round(4), hide_index=True, **_stretch_kw(st.dataframe))


# ----------------------------------------------------------------------------- Evaluasi
def page_eval():
    page_header("📏", "Evaluasi model", "Akurasi, F1-macro, dan confusion matrix IndoBERT pada data test.")
    results = st.session_state["results"]
    if not results:
        html_block('<div class="empty"><div class="big">🧪</div><h3>Belum ada hasil pelatihan</h3>'
                   '<p>Latih model di halaman <b>Latih IndoBERT</b>, atau klik <b>Muat hasil tersimpan</b>.</p></div>')
        return
    summ = pd.DataFrame([r["summary"] for r in results.values()])
    best = summ.loc[summ["f1_macro"].idxmax()]
    kpi_row([
        {"icon": "🏆", "label": "F1-macro terbaik", "value": round(best["f1_macro"] * 100, 2), "decimals": 2, "suffix": "%",
         "sub": f"{best['dataset']} · label {best['label']}"},
        {"icon": "🎯", "label": "Akurasi tertinggi", "value": round(summ["accuracy"].max() * 100, 2), "decimals": 2, "suffix": "%",
         "sub": "pada data test"},
        {"icon": "🧪", "label": "Model dievaluasi", "value": len(summ), "sub": f"{summ['n_test'].sum():,} ulasan test"},
        {"icon": "⏱️", "label": "Total waktu latih", "value": round(summ["train_seconds"].sum() / 60, 1), "decimals": 1,
         "suffix": " mnt", "sub": "semua eksperimen"},
    ])

    section("Ringkasan metrik")
    st.dataframe(summ.set_index(["dataset", "label"]), **_stretch_kw(st.dataframe),
                 column_config={m: st.column_config.ProgressColumn(m, min_value=0, max_value=1, format="%.4f")
                                for m in ["accuracy", "precision_macro", "recall_macro", "f1_macro"]})

    names = [f"{r['dataset']} · {r['label']}" for _, r in summ.iterrows()]
    fig = go.Figure()
    palette = [accents()[0], accents()[1], "#F59E0B", "#10B981"]
    for m, col, nm in zip(["accuracy", "precision_macro", "recall_macro", "f1_macro"], palette,
                          ["Akurasi", "Precision", "Recall", "F1-macro"]):
        fig.add_trace(go.Bar(x=names, y=summ[m], name=nm, marker=dict(color=col),
                             text=[f"{v:.3f}" for v in summ[m]], textposition="outside",
                             hovertemplate="%{x}<br>" + nm + ": %{y:.4f}<extra></extra>"))
    fig.update_layout(barmode="group", bargap=.22)
    themed(fig, title="Metrik per eksperimen")
    fig.update_yaxes(range=[0, 1.12])
    add_grow(fig, frame_ms=int(C("frame_ms")))
    show(fig, key="e_metrics", autoplay=True)

    c1, c2 = st.columns([1, 1])
    with c1:
        section("Radar per kelas")
        metric = st.radio("Metrik", ["f1-score", "precision", "recall"], horizontal=True, key="e_radar_m")
        fig = go.Figure()
        cols = [accents()[0], accents()[1], "#F59E0B", "#10B981"]
        for i, ((ds, src), r) in enumerate(results.items()):
            vals = [float(r["per_class"].loc[l, metric]) for l in LABELS]
            fig.add_trace(go.Scatterpolar(r=vals + vals[:1], theta=LABELS + LABELS[:1], fill="toself", name=f"{ds} · {src}",
                                          line=dict(color=cols[i % 4], width=2.5),
                                          fillcolor=hex_to_rgba(cols[i % 4], .18),
                                          hovertemplate="%{theta}: %{r:.3f}<extra>" + f"{ds} · {src}" + "</extra>"))
        themed(fig, height=420)
        fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(range=[0, 1], gridcolor="rgba(148,163,184,.25)"),
                                     angularaxis=dict(gridcolor="rgba(148,163,184,.25)")))
        show(fig, key="e_radar")
    with c2:
        section("Laporan per kelas")
        k = st.selectbox("Eksperimen", list(results), format_func=lambda k: f"{k[0]} · label {k[1]}", key="e_pc")
        st.dataframe(results[k]["per_class"].round(4), **_stretch_kw(st.dataframe),
                     column_config={c: st.column_config.ProgressColumn(c, min_value=0, max_value=1, format="%.3f")
                                    for c in ["precision", "recall", "f1-score"]})
        st.caption("Persentase di confusion matrix dihitung per baris, jadi sama dengan recall tiap kelas.")

    section("Confusion matrix IndoBERT", "Baris = label aktual, kolom = prediksi. Diagonal = prediksi benar.")
    norm = st.toggle("Tampilkan persen per baris", value=True, key="e_norm")
    keys = list(results)
    n = len(keys)
    ncols = 2 if n > 1 else 1
    nrows = math.ceil(n / ncols)
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=[f"{k[0]} · label {k[1]}" for k in keys],
                        horizontal_spacing=.12, vertical_spacing=.2 if nrows > 1 else .1)
    for i, key in enumerate(keys):
        cm = np.asarray(results[key]["cm"])
        pm = cm / cm.sum(axis=1, keepdims=True).clip(min=1) * 100
        text = [[f"{cm[a, b]:,}<br>({pm[a, b]:.1f}%)" for b in range(3)] for a in range(3)]
        fig.add_trace(go.Heatmap(z=pm if norm else cm, x=LABELS, y=LABELS, text=text, texttemplate="%{text}", xgap=4, ygap=4,
                                 colorscale=[[0, hex_to_rgba(accents()[0], .06)], [1, accents()[0]]], showscale=False,
                                 hovertemplate="Aktual: %{y}<br>Prediksi: %{x}<br>%{text}<extra></extra>"),
                      row=i // ncols + 1, col=i % ncols + 1)
    themed(fig, height=max(420, 400 * nrows), legend=False)
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title_text="Prediksi IndoBERT", title_font=dict(size=11))
    fig.update_yaxes(title_text="Aktual", title_font=dict(size=11))
    show(fig, key="e_cm")

    section("Prediksi pada data test")
    k2 = st.selectbox("Eksperimen", keys, format_func=lambda k: f"{k[0]} · label {k[1]}", key="e_pred")
    tp = results[k2]["test_pred"]
    only_wrong = st.toggle("Hanya prediksi yang salah", key="e_wrong")
    st.dataframe(tp[~tp["benar"]] if only_wrong else tp, hide_index=True, **_stretch_kw(st.dataframe))


# ----------------------------------------------------------------------------- Control chart p
def p_chart_figure(t, p_bar, defect, k, title):
    a1, a2 = accents()
    x = t["periode"]
    n = len(t)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.05, row_heights=[.78, .22])
    fig.add_trace(go.Scatter(x=x, y=t["LCL"], mode="lines", name="LCL",
                             line=dict(color="rgba(244,63,94,.9)", width=1.4, dash="dash", shape="hvh"),
                             hovertemplate="LCL %{y:.3f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=t["UCL"], mode="lines", name="UCL",
                             line=dict(color="#F43F5E", width=1.6, shape="hvh"), fill="tonexty",
                             fillcolor=hex_to_rgba(a1, .08), hovertemplate="UCL %{y:.3f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[x.min(), x.max()], y=[p_bar, p_bar], mode="lines", name=f"CL = p̄ = {p_bar:.3f}",
                             line=dict(color="#10B981", width=2.2), hoverinfo="skip"), row=1, col=1)
    cd = np.stack([t["n"], t["d"], t["UCL"], t["LCL"]], axis=-1)
    fig.add_trace(go.Scatter(x=x, y=t["p"], mode="lines+markers", name="p per subgroup", customdata=cd,
                             line=dict(color=a1, width=2.4), marker=dict(size=6, color=a1),
                             hovertemplate="<b>%{x|%d %b %Y}</b><br>p = %{y:.3f}<br>n = %{customdata[0]} · cacat = %{customdata[1]}"
                                           "<br>LCL %{customdata[3]:.3f} · UCL %{customdata[2]:.3f}<extra></extra>"),
                  row=1, col=1)
    out = t[t["status"] != "terkendali"]
    fig.add_trace(go.Scatter(x=out["periode"], y=out["p"], mode="markers", name=f"di luar batas ({len(out)})",
                             marker=dict(size=14, color="rgba(244,63,94,.18)", line=dict(color="#F43F5E", width=2.4)),
                             customdata=out["status"],
                             hovertemplate="<b>%{x|%d %b %Y}</b><br>p = %{y:.3f}<br>%{customdata}<extra></extra>"),
                  row=1, col=1)
    bar_col = ["#F43F5E" if s != "terkendali" else hex_to_rgba(a2, .75) for s in t["status"]]
    fig.add_trace(go.Bar(x=x, y=t["n"], marker=dict(color=bar_col), name="n ulasan", showlegend=False,
                         hovertemplate="%{x|%d %b %Y}<br>n = %{y} ulasan<extra></extra>"), row=2, col=1)
    themed(fig, height=int(C("chart_h")) + 80, title=title)
    ymax = min(1.02, max(float(t["UCL"].max()), float(t["p"].max())) * 1.12 + 0.01)
    pad = pd.Timedelta(days=1) if C("p_freq") == "Harian" else pd.Timedelta(days=4)
    fig.update_yaxes(range=[-0.01, ymax], title_text=f"Proporsi ulasan {defect}", row=1, col=1)
    fig.update_yaxes(title_text="n", row=2, col=1)
    fig.update_xaxes(range=[x.min() - pad, x.max() + pad])
    fig.update_layout(hovermode="closest", bargap=.15)

    if C("animate") and n > 2:
        steps = np.unique(np.linspace(1, n, min(n, 70)).astype(int))
        frames = []
        for kk in steps:
            sub_out = out[out["periode"] <= x.iloc[kk - 1]]
            frames.append(go.Frame(name=str(kk), traces=[3, 4],
                                   data=[go.Scatter(x=x.iloc[:kk], y=t["p"].iloc[:kk]),
                                         go.Scatter(x=sub_out["periode"], y=sub_out["p"])]))
        fig.frames = frames
        add_play_button(fig, int(C("frame_ms")))
    return fig, out


def page_pchart():
    page_header("📉", "Control chart p", "Proporsi ulasan negatif per subgroup, dengan batas kendali yang berubah mengikuti jumlah ulasan.")
    ctx = get_ctx()
    ds = pick_dataset(ctx, "p6_ds")
    d = ctx[ds]["data"]
    avail = available_sources(d, ds)
    wanted = [s for s in C("p_sources") if s in avail] or [s for s in avail if s.startswith("label_")][:1]
    c1, c2, c3 = st.columns([1.3, 1, 1.4])
    source = c1.selectbox("Sumber sentimen", avail, index=avail.index(wanted[0]) if wanted else 0, key="p6_src",
                          help="prediksi_* tersedia setelah model dilatih/dimuat.")
    k_sig = c2.number_input("k sigma", 1.0, 5.0, float(C("p_k")), 0.5, key="p6_k")
    valid_dates = d["tanggal"].dropna()
    if valid_dates.empty:
        st.warning("Dataset ini tidak punya tanggal yang terbaca, jadi control chart tidak bisa dibuat. Periksa kolom tanggal di sidebar.")
        return
    dmin, dmax = valid_dates.min().date(), valid_dates.max().date()
    if dmin < dmax:
        rng = c3.slider("Rentang tanggal", dmin, dmax, (dmin, dmax), key="p6_rng")
    else:
        rng = (dmin, dmax)
    mask = (d["tanggal"].dt.date >= rng[0]) & (d["tanggal"].dt.date <= rng[1])
    d_f = d[mask]
    series = source_series(d, ds, source)[mask]
    defect = C("p_defect")
    t, p_bar = p_table(d_f["tanggal"], series, defect, float(k_sig), C("p_freq"), int(C("p_min_n")))
    if t is None:
        st.info("Tidak ada subgroup yang memenuhi syarat (periksa minimal ulasan per subgroup).")
        return
    out = t[t["status"] != "terkendali"]
    unit = {"Harian": "hari", "Mingguan": "minggu", "Bulanan": "bulan"}[C("p_freq")]
    kpi_row([
        {"icon": "🎯", "label": f"p̄ (rata-rata proporsi {defect})", "value": round(p_bar * 100, 2), "decimals": 2, "suffix": "%",
         "sub": f"{int(t['d'].sum()):,} dari {int(t['n'].sum()):,} ulasan"},
        {"icon": "🗓️", "label": f"Jumlah {unit} dipantau", "value": len(t), "sub": f"n per {unit}: median {t['n'].median():.0f}, maks {t['n'].max()}"},
        {"icon": "🚨", "label": "Di atas UCL", "value": int((out["status"] == "di atas UCL").sum()), "sub": "lonjakan tidak wajar"},
        {"icon": "🟢", "label": "Di bawah LCL", "value": int((out["status"] == "di bawah LCL").sum()), "sub": "penurunan tidak wajar"},
    ])
    fig, _ = p_chart_figure(t, p_bar, defect, k_sig, f"p-Chart proporsi ulasan {defect} · {ds} · {source}")
    show(fig, key="p6_chart", autoplay=True,
         animation_opts={"frame": {"duration": int(C("frame_ms")), "redraw": True},
                         "transition": {"duration": int(C("frame_ms")) // 2, "easing": "linear"},
                         "fromcurrent": False, "mode": "immediate"})
    if len(out) == 0:
        note("Semua titik berada dalam batas kendali: variasi harian masih tergolong wajar (terkendali secara statistik).")
    else:
        section(f"{unit.capitalize()} di luar batas kendali", "Telusuri apa yang terjadi pada tanggal-tanggal ini.")
        show_t = out.assign(periode=out["periode"].dt.strftime("%d-%m-%Y"))[["periode", "n", "d", "p", "LCL", "UCL", "status"]]
        st.dataframe(show_t.round(4), hide_index=True, **_stretch_kw(st.dataframe),
                     column_config={"p": st.column_config.ProgressColumn("p", min_value=0, max_value=1, format="%.3f")})

        section("Baca ulasan pada tanggal terpilih")
        pick = st.selectbox("Periode", list(out["periode"]), format_func=lambda x: f"{x:%d %b %Y}", key="p6_pick")
        if C("p_freq") == "Harian":
            sel = d_f["tanggal"].dt.normalize() == pick
        elif C("p_freq") == "Mingguan":
            sel = d_f["tanggal"].dt.to_period("W").dt.start_time == pick
        else:
            sel = d_f["tanggal"].dt.to_period("M").dt.start_time == pick
        sub = d_f[sel].copy()
        sub["sumber"] = series[sel]
        only = st.toggle(f"Hanya ulasan {defect}", value=True, key="p6_only")
        if only:
            sub = sub[sub["sumber"] == defect]
        cols = ["tanggal", "text_asli", "sumber", "label_rating_str", "label_lexicon_str"]
        st.caption(f"{len(sub):,} ulasan.")
        st.dataframe(sub[cols], hide_index=True, **_stretch_kw(st.dataframe),
                     column_config={"text_asli": st.column_config.TextColumn("Ulasan", width="large"),
                                    "sumber": "Sentimen (sumber p-chart)", "label_rating_str": "Label rating",
                                    "label_lexicon_str": "Label lexicon"})
    with st.expander("Cara membaca p-chart"):
        st.markdown(
            "- **CL (p̄)**: rata-rata proporsi ulasan negatif selama periode pengamatan.\n"
            "- **Di atas UCL**: lonjakan ulasan negatif yang tidak wajar (*special cause*), misalnya gangguan aplikasi, "
            "update bermasalah, atau keterlambatan pengiriman massal.\n"
            "- **Di bawah LCL**: ulasan negatif jauh lebih sedikit dari biasanya, misalnya saat promo besar.\n"
            "- Batas kendali bertangga karena dihitung per subgroup mengikuti jumlah ulasan *nᵢ*.\n\n"
            r"$$p_i=\frac{d_i}{n_i},\quad \bar p=\frac{\sum d_i}{\sum n_i},\quad UCL_i,LCL_i=\bar p\pm k\sqrt{\frac{\bar p(1-\bar p)}{n_i}}$$")
        if source.startswith("prediksi"):
            st.caption("Prediksi seluruh ulasan juga mencakup data train yang sudah 'dilihat' model; wajar untuk tujuan pemantauan.")


# ----------------------------------------------------------------------------- Ekspor & Panduan
def page_export():
    page_header("📦", "Ekspor & Panduan", "Unduh semua hasil sebagai ZIP dan baca cara menafsirkannya.")
    ctx = get_ctx()
    results = st.session_state["results"]

    def build_zip() -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for name, v in ctx.items():
                z.writestr(f"{slugify(name)}_berlabel.csv", v["raw"].to_csv(index=False, encoding="utf-8-sig"))
                for source in P_SOURCES_ALL:
                    ser = source_series(v["data"], name, source)
                    if ser is None:
                        continue
                    t, _ = p_table(v["data"]["tanggal"], ser, C("p_defect"), float(C("p_k")), C("p_freq"), int(C("p_min_n")))
                    if t is not None:
                        z.writestr(f"p_chart_{slugify(name)}_{source}.csv",
                                   t.assign(periode=t["periode"].dt.date).to_csv(index=False, encoding="utf-8-sig"))
            if results:
                summ = pd.DataFrame([r["summary"] for r in results.values()])
                z.writestr("ringkasan_evaluasi.csv", summ.to_csv(index=False, encoding="utf-8-sig"))
                for (ds, src), r in results.items():
                    sl = slugify(f"{ds}_{src}")
                    z.writestr(f"riwayat_training_{sl}.csv", r["hist"].to_csv(index=False, encoding="utf-8-sig"))
                    z.writestr(f"prediksi_test_{sl}.csv", r["test_pred"].to_csv(index=False, encoding="utf-8-sig"))
            z.writestr("konfigurasi.json", json.dumps({k: C(k) for k in DEFAULTS}, indent=2, ensure_ascii=False, default=str))
        return buf.getvalue()

    c1, c2, c3 = st.columns(3)
    with c1:
        html_block('<div class="card"><h4>Paket lengkap</h4><p>Data berlabel, tabel p-chart, ringkasan evaluasi, '
                   'riwayat training, prediksi test, dan konfigurasi aktif.</p></div>')
        st.download_button("⬇️ Unduh hasil (ZIP)", build_zip(), file_name="hasil_sentiscope.zip", mime="application/zip", type="primary")
    with c2:
        ds = pick_dataset(ctx, "p7_ds")
        html_block('<div class="card"><h4>Data berlabel</h4><p>Seluruh ulasan beserta skor, label lexicon, dan kata sentimen.</p></div>')
        st.download_button("⬇️ CSV data berlabel", ctx[ds]["raw"].to_csv(index=False, encoding="utf-8-sig"),
                           file_name=f"{slugify(ds)}_berlabel.csv", mime="text/csv")
    with c3:
        html_block('<div class="card"><h4>Simpan ke folder</h4><p>Menulis hasil model ke folder output di server tempat aplikasi berjalan.</p></div>')
        if st.button("💾 Simpan hasil model"):
            if results:
                persist_results()
                st.success(f"Tersimpan di '{C('output_dir')}'.")
            else:
                st.info("Belum ada hasil model.")

    section("Cara membaca hasil")
    with st.expander("Perbandingan label (Lexicon vs Rating)", expanded=True):
        st.markdown("- **Kesesuaian**: persentase ulasan yang labelnya sama antara lexicon dan rating.\n"
                    "- **Cohen's Kappa**: kesesuaian setelah dikoreksi faktor kebetulan; lebih tepat dilaporkan daripada persentase biasa.\n"
                    "- Panduan Kappa: <0,20 lemah · 0,21–0,40 cukup · 0,41–0,60 sedang · 0,61–0,80 kuat · >0,80 sangat kuat.")
    with st.expander("Confusion matrix IndoBERT"):
        st.markdown("- **Baris = label aktual**, **kolom = prediksi**. Diagonal = prediksi benar.\n"
                    "- Persentase dihitung per baris, sama dengan **recall** tiap kelas.\n"
                    "- Bandingkan **F1-macro**, bukan hanya akurasi, karena kelas netral jauh lebih sedikit.\n"
                    "- Bila F1 model berlabel lexicon lebih tinggi daripada berlabel rating, label lexicon lebih konsisten dengan isi teks: IndoBERT hanya membaca teks, tidak melihat bintang.")
    with st.expander("Catatan metodologi"):
        st.markdown("Label lexicon dihasilkan aturan otomatis, jadi IndoBERT pada dasarnya belajar meniru aturan kamus. "
                    "Untuk ground truth paling kuat, lakukan validasi manual pada sampel kecil (mis. 100–200 ulasan) "
                    "lalu laporkan kesesuaiannya dengan label lexicon.")
    with st.expander("Referensi"):
        st.markdown("Koto & Adriani (2017), *InSet: Indonesia Sentiment Lexicon* · "
                    "Wilie et al. (2020), *IndoNLU* (model `indobenchmark/indobert-base-p1`).")


# =============================================================================
# MAIN
# =============================================================================
def main():
    st.set_page_config(page_title=f"{APP_NAME} · Sentimen IndoBERT", page_icon="📊", layout="wide",
                       initial_sidebar_state="expanded")
    for k, v in DEFAULTS.items():
        st.session_state.setdefault(K(k), v)
    st.session_state.setdefault("results", {})

    build_sidebar()
    build_lexicon()
    inject_css()

    pages = [
        st.Page(page_home, title="Beranda", icon="🏠", default=True),
        st.Page(page_data, title="Data & Lexicon", icon="🗂️", url_path="data"),
        st.Page(page_compare, title="Lexicon vs Rating", icon="🤝", url_path="perbandingan"),
        st.Page(page_train, title="Latih IndoBERT", icon="🧠", url_path="latih"),
        st.Page(page_eval, title="Evaluasi", icon="📏", url_path="evaluasi"),
        st.Page(page_pchart, title="Control Chart p", icon="📉", url_path="pchart"),
        st.Page(page_export, title="Ekspor & Panduan", icon="📦", url_path="ekspor"),
    ]
    try:
        nav = st.navigation(pages, position="top")
    except TypeError:                      # Streamlit lama tanpa navigasi atas
        nav = st.navigation(pages)
    nav.run()
    html_block(f'<div style="text-align:center;opacity:.5;font-size:.8rem;margin-top:36px;">{APP_NAME} · '
               'IndoBERT × InSet × control chart p</div>')


if __name__ == "__main__":
    main()
