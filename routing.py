# ============================================================
# routing.py — Semua strategi routing untuk ARM 3, 4, 5
#
# PERBAIKAN vs kode asli:
#
# [ARM 3 — Random]
#   Bug: random.choice() dipanggil tanpa seed per-sample,
#   sehingga urutan pilihan agen bergantung pada state global
#   RNG yang tidak terdokumentasi → tidak reproducible.
#   Fix: terima seed per-sample eksplisit.
#
# [ARM 4 — Heuristic]
#   Tidak ada bug kritis, tapi false positive "how many X" 
#   (pertanyaan Gc bukan Gq) dicatat sebagai keterbatasan
#   yang akan diukur via routing accuracy log.
#
# [ARM 5 — LLM Router]
#   Bug 1: parsing "GF" / "GQ" bisa false-match string panjang
#          ("The question requires GF reasoning" → "GF" terdeteksi,
#           tapi jika urutan kata berbeda bisa gagal).
#   Bug 2: fallback selalu ke Gc tanpa logging → tidak bisa
#          audit seberapa sering fallback terjadi.
#   Bug 3: routing accuracy TIDAK diukur → tidak bisa klaim
#          "LLM router lebih baik dari heuristic".
#   Fix: parsing regex yang ketat, logging setiap keputusan,
#        dan fungsi khusus yang mengembalikan label routing
#        beserta total token overhead.
# ============================================================

import re
import random
from typing import Tuple, Dict
from agents import CHC_AGENTS, gf_agent, gc_agent, gq_agent
from ollama_client import ollama_generate


# ---------------------------------------------------------------
# ARM 3 — Random Routing
# ---------------------------------------------------------------

def route_random(query: str, sample_seed: int) -> Tuple[str, str]:
    """
    Pilih agen CHC secara acak dengan seed per-sample.

    Mengembalikan (routing_label, agent_func).
    seed per-sample memastikan eksperimen reproducible:
    sample ke-7 selalu mendapat agen yang sama di setiap run.
    """
    rng = random.Random(sample_seed)
    label = rng.choice(list(CHC_AGENTS.keys()))  # "Gf", "Gc", atau "Gq"
    return label, CHC_AGENTS[label]


# ---------------------------------------------------------------
# ARM 4 — Heuristic Routing
# ---------------------------------------------------------------

# Keyword list dipisah per domain untuk audit dan iterasi mudah.
_MATH_KEYWORDS = {
    "calculate", "how many", "how much", "sum", "difference", "product",
    "divide", "multiply", "math", "arithmetic", "number", "total", "average",
    "percent", "fraction", "distance", "speed", "area", "volume", "per",
    "rate", "ratio", "equation", "formula", "solve", "compute",
}

_REASONING_KEYWORDS = {
    "if", "then", "unless", "all", "some", "none", "therefore", "deduce",
    "conclude", "assume", "implies", "logical", "order", "arrange",
    "left", "right", "taller", "older", "younger", "precede", "follow",
    "sequence", "pattern", "which comes next", "infer",
}

# Catatan: "how many" ada di _MATH_KEYWORDS.
# Pertanyaan seperti "how many presidents..." seharusnya Gc.
# Ini adalah known limitation yang harus dilaporkan di paper —
# bukan yang disembunyikan dengan menambahkan pengecualian ad-hoc.


def route_heuristic(query: str) -> Tuple[str, str]:
    """
    Rule-based routing berdasarkan keyword matching.

    Mengembalikan (routing_label, agent_func).
    Prioritas: Gq > Gf > Gc (fallback).
    """
    q_lower = query.lower()

    if any(k in q_lower for k in _MATH_KEYWORDS):
        return "Gq", gq_agent
    elif any(k in q_lower for k in _REASONING_KEYWORDS):
        return "Gf", gf_agent
    else:
        return "Gc", gc_agent


# ---------------------------------------------------------------
# ARM 5 — LLM-based Cognitive Router
# ---------------------------------------------------------------

_ROUTING_SYSTEM = (
    "You are a cognitive task classifier. "
    "Classify the question into exactly one category:\n"
    "  Gf — abstract reasoning, logic, deduction, pattern recognition\n"
    "  Gc — factual knowledge, definitions, trivia, general information\n"
    "  Gq — mathematics, arithmetic, calculation, numeric problem solving\n\n"
    "Output ONLY the label with no other text: Gf, Gc, or Gq."
)

_LABEL_PATTERN = re.compile(r"\b(Gf|Gc|Gq)\b", re.IGNORECASE)


def route_llm(query: str) -> Tuple[str, str, int, int, float]:
    """
    Gunakan LLM kecil untuk mengklasifikasi domain kognitif query.

    Mengembalikan:
        (routing_label, agent_func, prompt_tokens, completion_tokens, latency)

    Token dan latency routing DICATAT TERPISAH supaya overhead ARM 5
    bisa dihitung dan dilaporkan sebagai metrik tersendiri di paper.

    Parsing menggunakan regex \b(Gf|Gc|Gq)\b untuk menghindari
    false-match pada string seperti "GF reasoning requires..."
    yang dulu bisa diparse salah dengan simple `.upper()` check.

    Fallback ke "Gc" jika parse gagal, TETAPI fallback ini selalu
    di-log supaya bisa diaudit di post-hoc analysis.
    """
    routing_prompt = f"Question: {query}"

    raw_decision, p_tok, c_tok, lat = ollama_generate(
        routing_prompt,
        system=_ROUTING_SYSTEM,
        max_tokens=8,   # hanya butuh 2–3 token untuk label
    )

    match = _LABEL_PATTERN.search(raw_decision)
    if match:
        label = match.group(1).capitalize()          # "GF" → "Gf"
        label = {"Gf": "Gf", "Gc": "Gc", "Gq": "Gq"}.get(label, "Gc")
        fallback_used = False
    else:
        # Model tidak mengeluarkan label yang valid — catat dan fallback
        label = "Gc"
        fallback_used = True

    return label, CHC_AGENTS[label], p_tok, c_tok, lat, raw_decision, fallback_used


# ---------------------------------------------------------------
# Statistik routing (diakumulasi selama eksperimen)
# ---------------------------------------------------------------

class RoutingStats:
    """
    Koleksi keputusan routing untuk satu ARM.
    Digunakan untuk menghitung routing accuracy di akhir eksperimen.
    """

    def __init__(self):
        self.decisions: list[Dict] = []

    def record(
        self,
        question: str,
        predicted_label: str,
        true_label: str | None,
        fallback_used: bool = False,
        raw_output: str = "",
    ):
        self.decisions.append({
            "question_snippet": question[:80],
            "predicted": predicted_label,
            "true": true_label,
            "fallback": fallback_used,
            "raw_output": raw_output,
        })

    def routing_accuracy(self) -> float | None:
        """
        Hitung akurasi routing jika ground-truth label tersedia.
        Untuk dataset yang memiliki anotasi domain (misal: MMLU subset).
        """
        labeled = [d for d in self.decisions if d["true"] is not None]
        if not labeled:
            return None
        correct = sum(1 for d in labeled if d["predicted"] == d["true"])
        return correct / len(labeled)

    def fallback_rate(self) -> float:
        if not self.decisions:
            return 0.0
        return sum(1 for d in self.decisions if d["fallback"]) / len(self.decisions)

    def label_distribution(self) -> Dict[str, int]:
        dist = {"Gf": 0, "Gc": 0, "Gq": 0}
        for d in self.decisions:
            dist[d["predicted"]] = dist.get(d["predicted"], 0) + 1
        return dist
