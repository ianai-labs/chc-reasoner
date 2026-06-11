# CHC-Reasoner

Eksperimen benchmark multi-ARM untuk menguji apakah routing agen berbasis domain kognitif (Cattell-Horn-Carroll) meningkatkan akurasi dan efisiensi dibanding baseline monolitik, menggunakan Qwen2.5 3B Instruct via Ollama.

---

## Latar Belakang

Teori CHC (Cattell-Horn-Carroll) membagi kecerdasan menjadi beberapa domain. Proyek ini menguji hipotesis bahwa model LLM kecil dapat diarahkan ke spesialisasi domain melalui system prompt, dan bahwa routing otomatis ke domain yang tepat meningkatkan performa dibanding model tunggal tanpa spesialisasi.

Tiga domain yang diuji:

| Label | Domain | Benchmark Proxy |
|-------|--------|-----------------|
| **Gf** | Fluid reasoning — logika, deduksi, pola | BBH Navigate |
| **Gc** | Crystallized knowledge — fakta, definisi | MMLU college_mathematics |
| **Gq** | Quantitative reasoning — matematika | GSM8K |

---

## Struktur File

```
chc_reasoner/
├── config.py            Konfigurasi sentral (URL Ollama, model, seed, timeout)
├── ollama_client.py     Wrapper inferensi dengan token counting akurat dari eval_count
├── agents.py            Definisi agen Gf / Gc / Gq dan monolitik
├── routing.py           Tiga strategi routing (random, heuristic, LLM) + RoutingStats
├── dataset.py           Loading GSM8K / MMLU / BBH dengan validasi hard
├── evaluation.py        Parser jawaban per-benchmark (terpisah per format)
├── arm_runner.py        Runner eksperimen per ARM + majority voting
├── stats.py             McNemar test + Bonferroni correction + effect size
├── main.py              Entry point dengan checkpointing
└── logs/                Output JSON per checkpoint dan laporan final
```

---

## Desain Eksperimen: 5 ARM

| ARM | Nama | Strategi |
|-----|------|----------|
| 1 | `Arm1_Monolithic` | Satu model, satu system prompt kosong — baseline |
| 2 | `Arm2_GenericMultiAgent` | 3 agen generik identik → majority voting |
| 3 | `Arm3_CHC_RandomRouting` | CHC agents, agen dipilih acak (seed per-sample) |
| 4 | `Arm4_CHC_HeuristicRouting` | CHC agents, routing via keyword matching |
| 5 | `Arm5_FullCHC` | CHC agents, routing via LLM classifier |

---

## Cara Menjalankan

### Prasyarat

```bash
pip install requests datasets scipy
```

### Setup Ollama

```bash
# Jalankan server
ollama serve

# Download model
ollama pull qwen2.5:3b-instruct
```

### Jalankan Eksperimen

```bash
python main.py
```

Output tersimpan di `logs/`:

```
logs/
├── {run_id}_Arm1_Monolithic.json          # checkpoint per ARM
├── {run_id}_Arm2_GenericMultiAgent.json
├── ...
└── {run_id}_final_report.json             # laporan gabungan
```

### Konfigurasi

Edit `config.py` untuk menyesuaikan:

```python
SAMPLES_PER_BENCHMARK = 100   # jumlah soal per benchmark
GLOBAL_SEED           = 42    # seed reproducibility
INFERENCE_TIMEOUT     = 120   # timeout per call (detik)
MODEL_NAME            = "qwen2.5:3b-instruct"
```

---

## Metrik yang Dilaporkan

| Metrik | Deskripsi |
|--------|-----------|
| `accuracy` | Exact match accuracy keseluruhan |
| `accuracy_per_bench` | Accuracy terpisah per GSM8K / MMLU / BBH |
| `total_tokens` | Total token (prompt + completion) seluruh dataset |
| `avg_tokens` | Rata-rata token per soal |
| `avg_latency` | Rata-rata latency per soal (detik) |
| `routing_accuracy` | Seberapa sering router memilih domain yang tepat (ARM 3–5) |
| `fallback_rate` | Frekuensi fallback ke Gc karena parse gagal (ARM 5) |
| `accuracy_per_1k_tokens` | Metrik efisiensi utama: akurasi ternormalisasi per 1000 token |
| `routing_overhead_tokens` | Token ekstra untuk routing call (ARM 5, dicatat terpisah) |

---

## Analisis Statistik

Setelah semua ARM selesai, `main.py` otomatis menjalankan:

- **McNemar Test** — uji berpasangan untuk data biner benar/salah pada soal yang sama
- **Bonferroni Correction** — koreksi untuk 10 perbandingan simultan (C(5,2))
- **Odds Ratio** — effect size sebagai suplemen p-value

```
Alpha original:   0.05
Alpha Bonferroni: 0.005   (0.05 / 10 pasangan)
```

Membutuhkan `scipy`. Jika tidak tersedia, eksperimen tetap berjalan dan analisis statistik dilewati.

---

## Perbaikan dari Versi Asli

### Bug Kritis

**ARM 2 — bukan multi-agent yang valid**

Versi lama menjalankan 3 iterasi tapi hanya menyimpan output terakhir — efektif single-agent dengan biaya 3×. Diperbaiki dengan majority voting yang benar: 3 agen menjawab independen, jawaban final dipilih berdasarkan konsensus.

**`is_correct()` — substring match terlalu longgar**

Versi lama menggunakan satu fungsi untuk semua benchmark dengan `if true in pred`. Ini menyebabkan false positive seperti `true="A"` cocok dengan kata "analysis", atau `true="12"` cocok dengan "takes 12 steps". Diperbaiki dengan parser terpisah per benchmark:

- **GSM8K** — ekstrak angka terakhir, bandingkan numerik dengan toleransi ±0.01
- **MMLU** — regex pola spesifik: `"answer is X"`, `"(X)"`, huruf standalone di awal/akhir
- **BBH** — exact match setelah normalisasi; substring hanya untuk jawaban ≤5 karakter

**Token counting — word count bukan token count**

Versi lama menggunakan `len(text.split())` yang bisa meleset 30–50%. Diperbaiki menggunakan `eval_count` dan `prompt_eval_count` dari respons JSON Ollama.

**ARM 3 — random routing tidak reproducible**

`random.choice()` tanpa seed per-sample membuat hasil bergantung pada state RNG global. Diperbaiki dengan `seed = GLOBAL_SEED + sample_index` sehingga soal ke-N selalu mendapat agen yang sama di setiap run.

**ARM 5 — routing decision tidak di-log**

Tidak ada cara untuk mengaudit akurasi router. Diperbaiki dengan `RoutingStats` yang mencatat setiap keputusan routing beserta label prediksi, label ground-truth, apakah fallback digunakan, raw output, dan token overhead.

**ARM 5 — parsing label tidak robust**

`if "GF" in decision.upper()` bisa false-match pada string seperti "GF reasoning requires...". Diperbaiki dengan regex `\b(Gf|Gc|Gq)\b` yang menggunakan word boundary.

**BBH — silent failure**

`try/except: return []` membuat eksperimen lanjut tanpa BBH tanpa peringatan apapun. Diperbaiki dengan `raise RuntimeError(...)` eksplisit.

### Fitur Tambahan

| Fitur | Keterangan |
|-------|------------|
| `domain_label` per sample | Ground-truth label untuk evaluasi routing accuracy |
| Checkpointing per ARM | Hasil disimpan langsung; crash di tengah tidak kehilangan data |
| `RoutingStats` class | Distribusi label, fallback rate, routing accuracy teragregasi |
| `accuracy_per_1k_tokens` | Menggantikan `accuracy/token` yang tidak interpretable |
| GSM8K difficulty proxy | Jumlah langkah kalkulasi (bukan panjang string) sebagai filter kesulitan |
| `routing_overhead_tokens` | Token routing ARM 5 dicatat terpisah untuk analisis cost-benefit |

---

## Yang Tidak Diubah

- System prompt untuk setiap agen (Gf, Gc, Gq, Monolithic)
- Backbone model: Qwen2.5 3B Instruct via Ollama
- Desain 5 ARM (Arm1–Arm5)
- Benchmark yang digunakan: GSM8K, MMLU college_mathematics, BBH Navigate
- Metrik primer: Exact Match Accuracy
- Metode statistik: McNemar + Bonferroni (α = 0.05)

---
