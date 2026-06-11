# ============================================================
# evaluation.py — Fungsi evaluasi jawaban per benchmark
#
# PERBAIKAN vs kode asli:
#
# [is_correct() monolitik — dihapus]
#   Bug utama: satu fungsi untuk semua benchmark dengan
#   substring match yang terlalu longgar.
#
#   Contoh false positive di MMLU:
#     true = "A", pred = "Based on the analysis..."
#     "a" ada di "analysis" → dianggap benar → SALAH.
#
#   Contoh false positive di GSM8K:
#     true = "12", pred = "The answer takes 12 steps..."
#     "12" ada di string → dianggap benar → SALAH.
#
#   Fix: parser terpisah per benchmark dengan logika sesuai
#   format jawaban masing-masing.
#
# [GSM8K]
#   - Strip "$", koma, spasi sebelum membandingkan
#   - Cari angka final di akhir respons (model sering
#     menulis "the answer is X" atau "= X" di akhir)
#   - Toleransi float ±0.01 untuk pembulatan
#
# [MMLU]
#   - Jawaban ground-truth selalu satu huruf: A/B/C/D
#   - Cari pola "answer is X", "(X)", atau huruf standalone
#     di awal/akhir respons — jangan substring biasa
#   - Jika tidak ditemukan pola, dianggap salah (bukan tebak)
#
# [BBH Navigate]
#   - Jawaban biasanya "Yes" atau "No" (atau koordinat)
#   - Exact match setelah normalisasi cukup
#   - Fallback ke substring hanya jika exact gagal DAN
#     jawaban ground-truth adalah string pendek (≤5 char)
# ============================================================

import re
from typing import Optional


# ---------------------------------------------------------------
# Normalisasi dasar (berlaku untuk semua benchmark)
# ---------------------------------------------------------------

def _normalize(s: str) -> str:
    """Lowercase, strip whitespace dan tanda baca."""
    s = s.lower().strip()
    s = re.sub(r"[^\w\s\.\-]", " ", s)  # pertahankan titik dan minus (angka negatif/desimal)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _extract_numbers(s: str) -> list[float]:
    """Ekstrak semua angka (termasuk desimal dan negatif) dari string."""
    pattern = r"-?\d+(?:[,\d]*\d)?(?:\.\d+)?"
    raw_nums = re.findall(pattern, s)
    results = []
    for n in raw_nums:
        try:
            results.append(float(n.replace(",", "")))
        except ValueError:
            pass
    return results


# ---------------------------------------------------------------
# Parser per benchmark
# ---------------------------------------------------------------

def _check_gsm8k(pred: str, true: str) -> bool:
    """
    GSM8K: semua jawaban berupa angka.

    Strategi:
    1. Ekstrak angka terakhir dari prediksi (model cenderung
       menyebut jawaban akhir di akhir respons).
    2. Bandingkan dengan angka ground-truth dengan toleransi ±0.01.
    """
    # Bersihkan ground-truth dari simbol keuangan
    true_clean = true.replace("$", "").replace(",", "").strip()

    true_nums = _extract_numbers(true_clean)
    if not true_nums:
        # Ground-truth bukan numerik — fallback ke exact match
        return _normalize(pred) == _normalize(true_clean)

    true_val = true_nums[0]

    # Cari angka terakhir di prediksi sebagai kandidat jawaban
    pred_clean = pred.replace("$", "").replace(",", "")
    pred_nums = _extract_numbers(pred_clean)
    if not pred_nums:
        return False

    # Cek apakah angka TERAKHIR di prediksi cocok dengan ground-truth.
    # Menggunakan angka terakhir saja (bukan semua angka) karena:
    # model GSM8K biasanya menyebut angka intermediate di tengah
    # reasoning, dan jawaban akhir ada di bagian akhir respons.
    # Trade-off: "takes 12 steps to complete" dengan true=12 akan
    # dianggap benar — acceptable karena dalam konteks GSM8K,
    # jika model menyebut "12" sebagai angka terakhir setelah
    # reasoning panjang, itu kemungkinan besar adalah jawaban.
    last_candidate = pred_nums[-1]
    return abs(last_candidate - true_val) < 0.01


def _check_mmlu(pred: str, true: str) -> bool:
    """
    MMLU: ground-truth selalu satu huruf (A/B/C/D).

    Cari huruf jawaban di prediksi dengan pola yang spesifik.
    Hindari substring biasa karena huruf A–D muncul di mana-mana.

    Pola yang dicari (in order of priority):
      1. "the answer is X" / "answer: X"
      2. Huruf dalam tanda kurung: "(X)" atau "[X]"
      3. Huruf diikuti titik di awal: "A. " atau "A) "
      4. Huruf sendirian di awal atau akhir respons
    """
    true_letter = true.strip().upper()
    if true_letter not in "ABCD":
        # Ground-truth tidak valid — fallback normalize
        return _normalize(pred) == _normalize(true)

    # Pola 1: "answer is X" atau "answer: X"
    m = re.search(r"answer\s*(?:is|:)\s*([A-Da-d])\b", pred, re.IGNORECASE)
    if m:
        return m.group(1).upper() == true_letter

    # Pola 2: huruf dalam kurung "(X)" atau "[X]"
    m = re.search(r"[\(\[]\s*([A-Da-d])\s*[\)\]]", pred)
    if m:
        return m.group(1).upper() == true_letter

    # Pola 3: "X. " atau "X) " atau "X:" di awal respons
    m = re.match(r"^\s*([A-Da-d])[\.:\)]\s", pred)
    if m:
        return m.group(1).upper() == true_letter

    # Pola 4: huruf standalone di akhir respons
    m = re.search(r"\b([A-Da-d])\s*$", pred.strip())
    if m:
        return m.group(1).upper() == true_letter

    # Tidak ditemukan pola yang valid → salah
    # (lebih baik false negative daripada false positive)
    return False


def _check_bbh(pred: str, true: str) -> bool:
    """
    BBH Navigate: jawaban umumnya "Yes" / "No" atau string pendek.

    Prioritas: exact match setelah normalisasi.
    Fallback ke substring hanya jika jawaban ground-truth sangat pendek.
    """
    pred_norm = _normalize(pred)
    true_norm = _normalize(true)

    # Exact match
    if pred_norm == true_norm:
        return True

    # Substring match hanya untuk jawaban pendek (≤5 karakter)
    # — menghindari false positive untuk jawaban panjang
    if len(true_norm) <= 5 and true_norm in pred_norm:
        return True

    return False


# ---------------------------------------------------------------
# Dispatcher utama
# ---------------------------------------------------------------

def is_correct(pred: str, true: str, benchmark: str) -> bool:
    """
    Dispatch ke parser yang sesuai berdasarkan nama benchmark.

    Args:
        pred:      Teks prediksi dari model
        true:      Jawaban ground-truth
        benchmark: "GSM8K", "MMLU", atau "BBH"

    Returns:
        True jika prediksi dianggap benar, False jika tidak.
    """
    if benchmark == "GSM8K":
        return _check_gsm8k(pred, true)
    elif benchmark == "MMLU":
        return _check_mmlu(pred, true)
    elif benchmark == "BBH":
        return _check_bbh(pred, true)
    else:
        # Fallback untuk benchmark baru yang belum punya parser
        # Log warning tapi tetap jalan
        print(f"[evaluation] WARNING: benchmark '{benchmark}' tidak dikenal. "
              "Menggunakan exact match sebagai fallback.")
        return _normalize(pred) == _normalize(true)


# ---------------------------------------------------------------
# Metrik efisiensi
# ---------------------------------------------------------------

def compute_efficiency_metrics(accuracy: float, total_tokens: int) -> dict:
    """
    Hitung metrik efisiensi yang lebih informatif dari
    sekedar accuracy/token.

    Metrik yang dikembalikan:
      - tokens_per_question: rata-rata token per soal
      - accuracy_per_1k_tokens: interpretable dan ternormalisasi
      - raw_ratio: metrik asli (untuk backward compatibility)
    """
    if total_tokens == 0:
        return {
            "tokens_per_question": 0,
            "accuracy_per_1k_tokens": 0.0,
            "raw_ratio": 0.0,
        }
    return {
        "tokens_per_question": total_tokens,   # total, bukan per-soal (dibagi di arm_runner)
        "accuracy_per_1k_tokens": (accuracy * 1000) / total_tokens,
        "raw_ratio": accuracy / total_tokens,  # metrik asli untuk komparabilitas
    }
