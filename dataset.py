# ============================================================
# dataset.py — Loading dan filtering benchmark dataset
#
# PERBAIKAN vs kode asli:
#
# [GSM8K "Hard" filter]
#   Bug: panjang jawaban (dalam kata) ≠ kesulitan soal.
#   Soal 3-step bisa lebih sulit dari soal dengan kalimat panjang.
#   Fix: gunakan jumlah LANGKAH kalkulasi (baris numerik dalam
#   reasoning chain) sebagai proxy kesulitan — lebih valid
#   secara construct daripada panjang string jawaban.
#
# [MMLU]
#   Tidak ada bug kritis. Tambah metadata domain label ("Gc")
#   supaya routing accuracy bisa dievaluasi.
#
# [BBH]
#   Bug: try/except diam-diam mengembalikan [] jika dataset
#   gagal dimuat. Eksperimen lanjut tanpa BBH tanpa warning.
#   Fix: raise RuntimeError eksplisit jika BBH tidak bisa dimuat.
#   Peneliti harus tahu jika satu benchmark hilang.
#
# [Ground-truth domain label]
#   Ditambahkan field "domain_label" ("Gf"/"Gc"/"Gq") ke tiap
#   sample. Ini digunakan untuk mengevaluasi routing accuracy ARM 5.
#   Labelnya deterministik berdasarkan sumber benchmark:
#     GSM8K → Gq (semua soal matematika)
#     MMLU college_mathematics → Gq
#     BBH navigate → Gf (spatial/logical reasoning)
# ============================================================

import random
from typing import List, Dict
from config import GLOBAL_SEED


# ---------------------------------------------------------------
# GSM8K — Mathematical Reasoning
# ---------------------------------------------------------------

def _count_calculation_steps(reasoning_chain: str) -> int:
    """
    Hitung jumlah baris yang mengandung angka dan operator
    sebagai proxy kesulitan soal GSM8K.

    Lebih valid daripada panjang string karena mengukur
    kedalaman operasi, bukan verbositas teks.
    """
    lines = reasoning_chain.split("\n")
    step_count = 0
    for line in lines:
        # Baris yang mengandung angka + setidaknya satu operator
        has_number   = any(c.isdigit() for c in line)
        has_operator = any(op in line for op in ["=", "+", "-", "*", "/", "×", "÷"])
        if has_number and has_operator:
            step_count += 1
    return step_count


def load_gsm8k(n: int = 10, seed: int = GLOBAL_SEED) -> List[Dict]:
    """
    Ambil n soal GSM8K paling sulit berdasarkan jumlah langkah kalkulasi.

    Menggunakan 'train' split (lebih banyak pilihan untuk filter).
    """
    from datasets import load_dataset
    rng = random.Random(seed)

    ds = load_dataset("openai/gsm8k", "main", split="test")

    items_scored = []
    for item in ds:
        full_answer   = item["answer"]
        final_answer  = full_answer.split("####")[-1].strip()
        # Hapus simbol mata uang dan koma agar numerik konsisten
        final_answer_clean = final_answer.replace("$", "").replace(",", "").strip()
        steps = _count_calculation_steps(full_answer)
        items_scored.append((steps, item, final_answer_clean))

    # Urutkan descending berdasarkan jumlah langkah kalkulasi
    items_scored.sort(key=lambda x: x[0], reverse=True)

    # Ambil top 2n lalu sample n secara acak dari yang "sulit"
    # (menghindari dataset terlalu deterministik/homogen)
    pool = items_scored[:max(n * 2, 20)]
    chosen = rng.sample(pool, min(n, len(pool)))

    samples = []
    for steps, item, answer_clean in chosen:
        samples.append({
            "benchmark":    "GSM8K",
            "question":     item["question"],
            "answer":       answer_clean,
            "domain_label": "Gq",       # semua GSM8K = quantitative
            "difficulty_proxy": steps,
        })
    return samples


# ---------------------------------------------------------------
# MMLU — Factual & Domain Knowledge
# ---------------------------------------------------------------

def load_mmlu(n: int = 10, seed: int = GLOBAL_SEED) -> List[Dict]:
    """
    Ambil n soal dari MMLU college_mathematics.

    Jawaban ground-truth adalah HURUF (A/B/C/D), bukan teks opsi.
    domain_label = "Gq" karena subset college_mathematics.
    """
    from datasets import load_dataset
    rng = random.Random(seed)

    ds = load_dataset("cais/mmlu", "college_mathematics", split="test")
    total = len(ds)

    if total == 0:
        raise RuntimeError("MMLU college_mathematics: dataset kosong.")

    indices = rng.sample(range(total), min(n, total))

    samples = []
    for idx in indices:
        item = ds[idx]
        options_text = "\n".join(
            f"{chr(65 + j)}. {item['choices'][j]}" for j in range(4)
        )
        question_full = f"{item['question']}\n{options_text}"
        answer_letter = chr(65 + item["answer"])   # int 0–3 → "A"–"D"

        samples.append({
            "benchmark":    "MMLU",
            "question":     question_full,
            "answer":       answer_letter,
            "domain_label": "Gq",
            "difficulty_proxy": None,
        })
    return samples


# ---------------------------------------------------------------
# BBH Navigate — Spatial / Logical Reasoning
# ---------------------------------------------------------------

def load_bbh(n: int = 10, seed: int = GLOBAL_SEED) -> List[Dict]:
    """
    Ambil n soal BBH Navigate.

    BBH Navigate mengukur kemampuan mengikuti instruksi arah dan
    menentukan posisi akhir — merupakan domain Gf (spatial reasoning).

    Mengganti try/except diam-diam dengan RuntimeError eksplisit.
    Peneliti harus tahu jika benchmark tidak bisa dimuat.
    """
    from datasets import load_dataset
    rng = random.Random(seed)

    try:
        ds = load_dataset(
            "lukaemon/bbh",
            "navigate",
            split="test",
        )
    except Exception as e:
        raise RuntimeError(
            f"Gagal memuat BBH navigate: {e}\n"
            "Coba: pip install datasets>=2.14 atau periksa koneksi internet."
        ) from e

    total = len(ds)
    if total == 0:
        raise RuntimeError("BBH navigate: dataset kosong setelah dimuat.")

    indices = rng.sample(range(total), min(n, total))

    samples = []
    for idx in indices:
        item = ds[idx]
        question = item["input"].strip()
        answer   = item["target"].strip()

        if not question or not answer:
            continue  # skip item malformed

        samples.append({
            "benchmark":    "BBH",
            "question":     question,
            "answer":       answer,
            "domain_label": "Gf",   # navigate = spatial/logical reasoning
            "difficulty_proxy": None,
        })
    return samples


# ---------------------------------------------------------------
# Builder utama
# ---------------------------------------------------------------

def build_dataset(n_per_benchmark: int = 10) -> List[Dict]:
    """
    Bangun dataset gabungan dari tiga benchmark.

    Menambahkan validasi hard setelah loading:
    - Jika salah satu benchmark gagal, eksperimen berhenti.
    - Tidak ada silent failure seperti di kode asli.
    """
    print(f"[dataset] Memuat {n_per_benchmark} soal dari GSM8K...")
    gsm8k = load_gsm8k(n_per_benchmark)
    print(f"  → {len(gsm8k)} soal dimuat.")

    print(f"[dataset] Memuat {n_per_benchmark} soal dari MMLU (college_math)...")
    mmlu = load_mmlu(n_per_benchmark)
    print(f"  → {len(mmlu)} soal dimuat.")

    print(f"[dataset] Memuat {n_per_benchmark} soal dari BBH (navigate)...")
    bbh = load_bbh(n_per_benchmark)
    print(f"  → {len(bbh)} soal dimuat.")

    # Validasi: tidak boleh ada benchmark yang kosong
    for name, ds in [("GSM8K", gsm8k), ("MMLU", mmlu), ("BBH", bbh)]:
        if len(ds) == 0:
            raise RuntimeError(
                f"[dataset] {name} menghasilkan 0 sample. "
                "Hentikan eksperimen dan periksa loading function."
            )

    dataset = gsm8k + mmlu + bbh
    print(f"\n[dataset] Total: {len(dataset)} soal "
          f"({len(gsm8k)} GSM8K + {len(mmlu)} MMLU + {len(bbh)} BBH)\n")
    return dataset
