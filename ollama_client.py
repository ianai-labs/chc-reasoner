# ============================================================
# ollama_client.py — Wrapper inferensi Ollama
#
# PERBAIKAN vs kode asli:
#   - Token count diambil dari eval_count (respons JSON Ollama),
#     bukan word-count manual yang bisa meleset 30–50%.
#   - prompt_eval_count (token prompt) juga dicatat terpisah
#     supaya ARM 5 bisa menghitung overhead routing secara akurat.
#   - Timeout dikonfigurasi dari config.py, bukan hardcoded.
#   - Semua error di-raise (bukan ditelan), biar eksperimen
#     tidak lanjut dengan data korup secara diam-diam.
# ============================================================

import time
import requests
from typing import Tuple
from config import OLLAMA_URL, MODEL_NAME, INFERENCE_TIMEOUT


def ollama_generate(
    prompt: str,
    system: str = "You are a helpful assistant.",
    max_tokens: int = 512,
) -> Tuple[str, int, int, float]:
    """
    Kirim prompt ke Ollama, kembalikan:
        (response_text, prompt_tokens, completion_tokens, latency_seconds)

    Menggunakan eval_count dari respons JSON Ollama — bukan word split —
    sehingga token count akurat untuk perbandingan antar ARM.
    """
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": max_tokens,
        },
    }

    start = time.perf_counter()
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=INFERENCE_TIMEOUT)
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        raise RuntimeError(
            f"Ollama timeout setelah {INFERENCE_TIMEOUT}s "
            f"untuk model '{MODEL_NAME}'. "
            "Coba kurangi max_tokens atau restart Ollama."
        )
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Tidak bisa terhubung ke Ollama. "
            "Pastikan `ollama serve` sudah berjalan."
        )

    latency = time.perf_counter() - start
    data = resp.json()

    text = data.get("response", "").strip()

    # Ollama melaporkan eval_count (completion tokens) dan
    # prompt_eval_count (prompt tokens) secara native.
    # Fallback ke word-count hanya jika field tidak ada
    # (versi Ollama lama).
    completion_tokens = data.get("eval_count", len(text.split()))
    prompt_tokens     = data.get("prompt_eval_count", len(prompt.split()))

    return text, prompt_tokens, completion_tokens, latency


def check_ollama_alive() -> bool:
    """Ping cepat untuk validasi koneksi sebelum eksperimen dimulai."""
    try:
        ollama_generate("ping", max_tokens=4)
        return True
    except RuntimeError:
        return False
