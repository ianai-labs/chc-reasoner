# ============================================================
# main.py — Entry point eksperimen CHC-Reasoner
#
# Urutan eksekusi:
#   1. Validasi koneksi Ollama
#   2. Load dataset (dengan validasi hard)
#   3. Jalankan semua ARM secara berurutan
#   4. Simpan hasil per-ARM segera setelah selesai (checkpointing)
#   5. Cetak ringkasan tabel
#   6. Jalankan analisis statistik
#   7. Simpan laporan final
# ============================================================

import os
import json
import time
from datetime import datetime
from config import SAMPLES_PER_BENCHMARK, LOG_DIR
from ollama_client import check_ollama_alive
from dataset import build_dataset
from arm_runner import evaluate_arm, ARM_FUNCTIONS
from stats import run_statistical_analysis, print_statistical_summary


def _ensure_log_dir():
    os.makedirs(LOG_DIR, exist_ok=True)


def _save_checkpoint(arm_result: dict, run_id: str):
    """Simpan hasil satu ARM segera setelah selesai."""
    _ensure_log_dir()
    arm_name = arm_result["arm_name"]
    path = os.path.join(LOG_DIR, f"{run_id}_{arm_name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(arm_result, f, indent=2, ensure_ascii=False, default=str)
    print(f"  [checkpoint] Disimpan: {path}")


def _print_summary_table(results: list[dict], dataset_size: int):
    """Cetak tabel ringkasan semua ARM."""
    print("\n" + "=" * 90)
    print(f"RINGKASAN EKSPERIMEN  ({dataset_size} soal total)")
    print("=" * 90)

    header = (
        f"{'ARM':<28} {'Acc':>6} {'GSM8K':>7} {'MMLU':>7} {'BBH':>6} "
        f"{'Tokens':>8} {'Lat(s)':>7} {'Acc/1KTok':>10} {'RoutAcc':>9}"
    )
    print(header)
    print("-" * 90)

    for r in results:
        pb = r.get("accuracy_per_bench", {})
        gsm = f"{pb.get('GSM8K', float('nan')):.0%}" if "GSM8K" in pb else "  N/A"
        mml = f"{pb.get('MMLU',  float('nan')):.0%}" if "MMLU"  in pb else "  N/A"
        bbh = f"{pb.get('BBH',   float('nan')):.0%}" if "BBH"   in pb else "  N/A"
        r_acc = (
            f"{r['routing_accuracy']:.0%}"
            if r.get("routing_accuracy") is not None else "  N/A"
        )
        eff = r["efficiency"]["accuracy_per_1k_tokens"]

        print(
            f"{r['arm_name']:<28} "
            f"{r['accuracy']:>6.1%} "
            f"{gsm:>7} {mml:>7} {bbh:>6} "
            f"{r['total_tokens']:>8} "
            f"{r['avg_latency']:>7.2f} "
            f"{eff:>10.4f} "
            f"{r_acc:>9}"
        )

    print()


def main():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n{'='*60}")
    print(f"  CHC-Reasoner Experiment — Run ID: {run_id}")
    print(f"{'='*60}\n")

    # ---- Step 1: Validasi Ollama ----
    print("[1/5] Memeriksa koneksi Ollama...")
    if not check_ollama_alive():
        print(
            "\n[ERROR] Tidak bisa terhubung ke Ollama.\n"
            "Jalankan: ollama serve\n"
            "Pastikan model tersedia: ollama pull qwen2.5:3b-instruct"
        )
        return
    print("  ✓ Ollama aktif.\n")

    # ---- Step 2: Load dataset ----
    print(f"[2/5] Memuat dataset ({SAMPLES_PER_BENCHMARK} soal per benchmark)...")
    try:
        dataset = build_dataset(SAMPLES_PER_BENCHMARK)
    except RuntimeError as e:
        print(f"\n[ERROR] Gagal memuat dataset:\n{e}")
        return
    dataset_size = len(dataset)

    # ---- Step 3 & 4: Jalankan ARM satu per satu ----
    print(f"[3/5] Menjalankan {len(ARM_FUNCTIONS)} ARM...\n")

    arm_order = [
        "Arm1_Monolithic",
        "Arm2_GenericMultiAgent",
        "Arm3_CHC_RandomRouting",
        "Arm4_CHC_HeuristicRouting",
        "Arm5_FullCHC",
    ]

    all_results = []
    for arm_name in arm_order:
        t0 = time.perf_counter()
        try:
            result = evaluate_arm(arm_name, dataset, verbose=True)
        except Exception as e:
            print(f"\n[ERROR] ARM '{arm_name}' gagal: {e}")
            print("Melanjutkan ke ARM berikutnya...\n")
            continue

        elapsed = time.perf_counter() - t0
        print(f"  Selesai dalam {elapsed:.1f}s\n")

        _save_checkpoint(result, run_id)
        all_results.append(result)

    if not all_results:
        print("[ERROR] Tidak ada ARM yang berhasil dijalankan.")
        return

    # ---- Step 5: Ringkasan tabel ----
    print("[4/5] Mencetak ringkasan...")
    _print_summary_table(all_results, dataset_size)

    # ---- Step 6: Analisis statistik ----
    print("[5/5] Analisis statistik (McNemar + Bonferroni)...")
    if len(all_results) >= 2:
        try:
            analysis = run_statistical_analysis(all_results, alpha=0.05)
            print_statistical_summary(analysis)
        except ImportError as e:
            print(f"  [SKIP] scipy tidak tersedia: {e}")
            print("  Install: pip install scipy")
            analysis = None
    else:
        print("  [SKIP] Butuh minimal 2 ARM untuk analisis statistik.")
        analysis = None

    # ---- Simpan laporan final ----
    _ensure_log_dir()
    final_path = os.path.join(LOG_DIR, f"{run_id}_final_report.json")
    report = {
        "run_id":        run_id,
        "config": {
            "samples_per_benchmark": SAMPLES_PER_BENCHMARK,
            "total_questions":       dataset_size,
        },
        "results":       all_results,
        "statistical_analysis": analysis,
    }
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nLaporan final disimpan: {final_path}")
    print(f"Checkpoint per-ARM ada di direktori: {LOG_DIR}/\n")


if __name__ == "__main__":
    main()
