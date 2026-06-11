# ============================================================
# config.py — Konfigurasi sentral CHC-Reasoner
# ============================================================

OLLAMA_URL   = "http://localhost:11434/api/generate"
MODEL_NAME   = "qwen2.5:3b-instruct"

# Jumlah sample per benchmark (ubah ke 10+ untuk full eval)
SAMPLES_PER_BENCHMARK = 100

# Random seed global (reproducibility)
GLOBAL_SEED = 42

# Batas timeout inferensi (detik)
INFERENCE_TIMEOUT = 120

# Direktori output log
LOG_DIR = "logs"
