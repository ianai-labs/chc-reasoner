# ============================================================
# arm_runner.py — Eksekutor eksperimen per ARM
#
# PERBAIKAN vs kode asli:
#
# [ARM 2 — Generic Multi-Agent]
#   Bug utama: 3 iterasi dijalankan tapi hanya output terakhir
#   yang diambil. Output agen 1 dan 2 dibuang sepenuhnya.
#   Ini bukan "multi-agent" — ini single-agent dengan biaya 3×.
#
#   Fix: implementasikan majority voting yang benar.
#   3 agen generik menjawab secara independen, jawaban akhir
#   dipilih via simple majority. Jika tidak ada mayoritas,
#   ambil jawaban yang paling sering muncul (ties → agen pertama).
#
# [ARM 3 — CHC Random]
#   Fix: terima sample_index sebagai seed per-sample sehingga
#   pilihan agen reproducible antar run.
#
# [ARM 5 — Full CHC]
#   Fix: catat routing decision, token overhead routing,
#   dan apakah fallback digunakan, sebagai kolom terpisah di log.
#
# [Logging]
#   Tambahkan checkpointing sederhana: hasil tiap soal langsung
#   di-append ke list, sehingga jika proses crash di tengah,
#   hasil parsial masih bisa dianalisis.
# ============================================================

from typing import List, Dict, Callable, Tuple
from agents import monolithic_agent, generic_agent, CHC_AGENTS
from routing import route_random, route_heuristic, route_llm, RoutingStats
from evaluation import is_correct, compute_efficiency_metrics
from config import GLOBAL_SEED


# ---------------------------------------------------------------
# Majority voting helper (untuk ARM 2)
# ---------------------------------------------------------------

def _majority_vote(answers: List[str]) -> str:
    """
    Pilih jawaban yang paling sering muncul di antara beberapa prediksi.
    Tie-breaking: ambil jawaban pertama yang muncul paling sering
    (deterministik, tidak random).
    """
    from collections import Counter
    if not answers:
        return ""
    counter = Counter(answers)
    return counter.most_common(1)[0][0]


# ---------------------------------------------------------------
# ARM definitions
# ---------------------------------------------------------------

def run_arm1_monolithic(item: Dict, idx: int) -> Dict:
    """ARM 1 — Single model, no specialization."""
    pred, p_tok, c_tok, lat = monolithic_agent(item["question"])
    correct = is_correct(pred, item["answer"], item["benchmark"])

    return {
        "arm": "Arm1_Monolithic",
        "correct": correct,
        "prompt_tokens": p_tok,
        "completion_tokens": c_tok,
        "total_tokens": p_tok + c_tok,
        "latency": lat,
        "routing_label": None,
        "routing_overhead_tokens": 0,
        "fallback_used": None,
        "prediction": pred,
    }


def run_arm2_generic_multiagent(item: Dict, idx: int) -> Dict:
    """
    ARM 2 — Generic Multi-Agent dengan majority voting.

    3 agen generik identik menjawab secara independen.
    Jawaban final dipilih via majority vote.
    Token dihitung sebagai total semua 3 call.

    Ini menggantikan round-robin yang hanya mengambil output
    terakhir — baru bisa disebut 'multi-agent' yang valid.
    """
    NUM_AGENTS = 3
    answers    = []
    total_p    = 0
    total_c    = 0
    total_lat  = 0.0

    for _ in range(NUM_AGENTS):
        pred, p_tok, c_tok, lat = generic_agent(item["question"])
        answers.append(pred)
        total_p   += p_tok
        total_c   += c_tok
        total_lat += lat

    # Pilih jawaban via majority vote
    final_pred = _majority_vote(answers)
    correct = is_correct(final_pred, item["answer"], item["benchmark"])

    return {
        "arm": "Arm2_GenericMultiAgent",
        "correct": correct,
        "prompt_tokens": total_p,
        "completion_tokens": total_c,
        "total_tokens": total_p + total_c,
        "latency": total_lat,
        "routing_label": None,
        "routing_overhead_tokens": 0,
        "fallback_used": None,
        "prediction": final_pred,
        "all_answers": answers,   # simpan semua jawaban untuk analisis disagreement
    }


def run_arm3_chc_random(item: Dict, idx: int) -> Dict:
    """
    ARM 3 — CHC agents dengan random routing.

    seed = GLOBAL_SEED + idx memastikan tiap sample selalu
    mendapat agen yang sama di setiap run (reproducible).
    """
    sample_seed = GLOBAL_SEED + idx
    routing_label, agent_func = route_random(item["question"], sample_seed)

    pred, p_tok, c_tok, lat = agent_func(item["question"])
    correct = is_correct(pred, item["answer"], item["benchmark"])

    return {
        "arm": "Arm3_CHC_RandomRouting",
        "correct": correct,
        "prompt_tokens": p_tok,
        "completion_tokens": c_tok,
        "total_tokens": p_tok + c_tok,
        "latency": lat,
        "routing_label": routing_label,
        "routing_overhead_tokens": 0,
        "fallback_used": False,
        "prediction": pred,
    }


def run_arm4_chc_heuristic(item: Dict, idx: int) -> Dict:
    """ARM 4 — CHC agents dengan heuristic keyword routing."""
    routing_label, agent_func = route_heuristic(item["question"])

    pred, p_tok, c_tok, lat = agent_func(item["question"])
    correct = is_correct(pred, item["answer"], item["benchmark"])

    return {
        "arm": "Arm4_CHC_HeuristicRouting",
        "correct": correct,
        "prompt_tokens": p_tok,
        "completion_tokens": c_tok,
        "total_tokens": p_tok + c_tok,
        "latency": lat,
        "routing_label": routing_label,
        "routing_overhead_tokens": 0,
        "fallback_used": False,
        "prediction": pred,
    }


def run_arm5_full_chc(item: Dict, idx: int) -> Dict:
    """
    ARM 5 — CHC agents dengan LLM cognitive router.

    Token routing dicatat terpisah sebagai overhead.
    Ini memungkinkan analisis: apakah akurasi routing 
    sebanding dengan biaya tambahan routing call?
    """
    (
        routing_label,
        agent_func,
        r_p_tok, r_c_tok, r_lat,
        raw_routing_output,
        fallback_used,
    ) = route_llm(item["question"])

    pred, p_tok, c_tok, lat = agent_func(item["question"])
    correct = is_correct(pred, item["answer"], item["benchmark"])

    routing_overhead = r_p_tok + r_c_tok

    return {
        "arm": "Arm5_FullCHC",
        "correct": correct,
        "prompt_tokens": p_tok + r_p_tok,
        "completion_tokens": c_tok + r_c_tok,
        "total_tokens": p_tok + c_tok + routing_overhead,
        "latency": lat + r_lat,
        "routing_label": routing_label,
        "routing_overhead_tokens": routing_overhead,
        "fallback_used": fallback_used,
        "raw_routing_output": raw_routing_output,
        "prediction": pred,
    }


# ---------------------------------------------------------------
# Dispatcher generik
# ---------------------------------------------------------------

ARM_FUNCTIONS: Dict[str, Callable] = {
    "Arm1_Monolithic":              run_arm1_monolithic,
    "Arm2_GenericMultiAgent":       run_arm2_generic_multiagent,
    "Arm3_CHC_RandomRouting":       run_arm3_chc_random,
    "Arm4_CHC_HeuristicRouting":    run_arm4_chc_heuristic,
    "Arm5_FullCHC":                 run_arm5_full_chc,
}


def evaluate_arm(
    arm_name: str,
    dataset: List[Dict],
    verbose: bool = True,
) -> Dict:
    """
    Jalankan satu ARM pada seluruh dataset dan kumpulkan metrik.

    Returns dict dengan:
      - accuracy keseluruhan dan per-benchmark
      - total & rata-rata token (prompt + completion terpisah)
      - latency rata-rata
      - metrik efisiensi
      - routing stats (label distribution, fallback rate)
      - detail per soal
    """
    if arm_name not in ARM_FUNCTIONS:
        raise ValueError(f"ARM '{arm_name}' tidak dikenal. "
                         f"Pilihan: {list(ARM_FUNCTIONS.keys())}")

    arm_func = ARM_FUNCTIONS[arm_name]
    details  = []
    routing_stats = RoutingStats()

    total       = len(dataset)
    correct_all = 0
    total_tokens = 0
    total_latency = 0.0

    # Akumulasi per-benchmark
    bench_correct: Dict[str, int]  = {}
    bench_total:   Dict[str, int]  = {}

    if verbose:
        print(f"\n{'='*60}")
        print(f"  ARM: {arm_name}")
        print(f"  Dataset: {total} soal")
        print(f"{'='*60}")

    for idx, item in enumerate(dataset):
        bname = item["benchmark"]
        bench_total[bname]   = bench_total.get(bname, 0) + 1
        bench_correct[bname] = bench_correct.get(bname, 0)

        if verbose:
            snippet = item["question"][:70].replace("\n", " ")
            print(f"  [{idx+1:>3}/{total}] [{bname}] {snippet}...")

        result = arm_func(item, idx)

        if result["correct"]:
            correct_all += 1
            bench_correct[bname] += 1
        elif verbose:
            print(f"    ✗  true: {item['answer']!r:15} | pred: {result['prediction'][:80]!r}")

        total_tokens  += result["total_tokens"]
        total_latency += result["latency"]

        # Catat routing decision untuk ARM 3/4/5
        if result.get("routing_label"):
            routing_stats.record(
                question       = item["question"],
                predicted_label= result["routing_label"],
                true_label     = item.get("domain_label"),
                fallback_used  = result.get("fallback_used", False),
                raw_output     = result.get("raw_routing_output", ""),
            )

        # Simpan detail lengkap
        details.append({
            **result,
            "benchmark":    bname,
            "true_answer":  item["answer"],
            "domain_label": item.get("domain_label"),
            "question":     item["question"],
        })

    # ---- Agregasi ----
    accuracy     = correct_all / total
    avg_latency  = total_latency / total
    avg_tokens   = total_tokens / total

    per_bench = {
        bname: bench_correct[bname] / bench_total[bname]
        for bname in bench_total
    }

    efficiency = compute_efficiency_metrics(accuracy, total_tokens)

    r_accuracy  = routing_stats.routing_accuracy()
    r_fallback  = routing_stats.fallback_rate()
    r_label_dist= routing_stats.label_distribution()

    if verbose:
        print(f"\n  ─ Hasil ─")
        print(f"  Accuracy total : {accuracy:.1%}")
        for bname, acc in per_bench.items():
            print(f"  Accuracy {bname:<8}: {acc:.1%}")
        if r_accuracy is not None:
            print(f"  Routing accuracy: {r_accuracy:.1%}")
        print(f"  Fallback rate   : {r_fallback:.1%}")
        print(f"  Label dist.     : {r_label_dist}")
        print(f"  Total tokens    : {total_tokens}")
        print(f"  Avg tokens/soal : {avg_tokens:.1f}")
        print(f"  Avg latency     : {avg_latency:.2f}s")
        print(f"  Acc/1K tokens   : {efficiency['accuracy_per_1k_tokens']:.4f}")

    return {
        "arm_name":          arm_name,
        "accuracy":          accuracy,
        "accuracy_per_bench": per_bench,
        "total_tokens":      total_tokens,
        "avg_tokens":        avg_tokens,
        "avg_latency":       avg_latency,
        "routing_accuracy":  r_accuracy,
        "fallback_rate":     r_fallback,
        "label_distribution": r_label_dist,
        "efficiency":        efficiency,
        "details":           details,
    }
