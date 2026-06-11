# ============================================================
# agents.py — Definisi agen CHC + agen generik
#
# Tidak ada perubahan logika agen itu sendiri — system prompt
# untuk Gf/Gc/Gq dipertahankan persis seperti proposal.
# Perubahan hanya pada signature: fungsi kini mengembalikan
# 4-tuple (text, prompt_tokens, completion_tokens, latency)
# supaya konsisten dengan ollama_client yang baru.
# ============================================================

from ollama_client import ollama_generate

# ------------------------------------------------------------------
# Sistem prompt per agen (sesuai proposal CHC-Reasoner)
# ------------------------------------------------------------------

_SYSTEM_GF = "Gf: prefer logical reasoning."
_SYSTEM_GC = "Gc: prefer factual knowledge."
_SYSTEM_GQ = "Gq: prefer mathematical calculation."
_SYSTEM_GENERIC = ""


# ------------------------------------------------------------------
# Agen CHC
# ------------------------------------------------------------------

def gf_agent(query: str):
    """Fluid Intelligence — logika, pola, deduksi."""
    return ollama_generate(query, system=_SYSTEM_GF)


def gc_agent(query: str):
    """Crystallized Intelligence — pengetahuan faktual."""
    return ollama_generate(query, system=_SYSTEM_GC)


def gq_agent(query: str):
    """Quantitative Reasoning — matematika dan kalkulasi."""
    return ollama_generate(query, system=_SYSTEM_GQ)


# ------------------------------------------------------------------
# Agen generik (untuk ARM 1 dan ARM 2)
# ------------------------------------------------------------------

def monolithic_agent(query: str):
    """ARM 1 — baseline monolitik tunggal."""
    return ollama_generate(query, system=_SYSTEM_GENERIC)


def generic_agent(query: str):
    """Satu agen generik tanpa spesialisasi domain."""
    return ollama_generate(query, system=_SYSTEM_GENERIC)


# ------------------------------------------------------------------
# Peta agen CHC (digunakan oleh router)
# ------------------------------------------------------------------

CHC_AGENTS = {
    "Gf": gf_agent,
    "Gc": gc_agent,
    "Gq": gq_agent,
}
