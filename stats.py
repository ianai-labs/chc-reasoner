# ============================================================
# stats.py — Analisis statistik hasil eksperimen
#
# Mengimplementasikan metodologi dari proposal:
#   - Paired McNemar Test (untuk perbandingan biner benar/salah)
#   - Bonferroni Correction untuk multiple comparisons
#   - Effect size (odds ratio) sebagai suplemen p-value
#
# Catatan: McNemar test dipilih karena data berpasangan
# (soal yang sama diujikan ke semua ARM), bukan independent.
# ============================================================

import json
from typing import Dict, List, Tuple, Optional
from itertools import combinations


def _mcnemar_statistic(
    results_a: List[bool],
    results_b: List[bool],
) -> Tuple[float, float, int, int]:
    """
    Hitung statistik McNemar untuk dua ARM pada soal yang sama.

    Contingency table:
        b=correct  b=wrong
    a=correct  [ n11      n10  ]
    a=wrong    [ n01      n00  ]

    McNemar test fokus pada discordant pairs: n10 dan n01.
    H0: kedua metode memiliki akurasi yang sama (n10 == n01).

    Returns: (chi2, p_value, n10, n01)
    """
    try:
        from scipy.stats import chi2 as chi2_dist
        import math
    except ImportError:
        raise ImportError(
            "scipy diperlukan untuk McNemar test. "
            "Install: pip install scipy"
        )

    if len(results_a) != len(results_b):
        raise ValueError("Kedua ARM harus dievaluasi pada dataset yang sama.")

    n10 = sum(1 for a, b in zip(results_a, results_b) if a and not b)
    n01 = sum(1 for a, b in zip(results_a, results_b) if not a and b)

    # Jika discordant pairs terlalu sedikit, test tidak reliable
    if n10 + n01 < 5:
        return float("nan"), float("nan"), n10, n01

    # McNemar dengan continuity correction (Edwards)
    numerator = (abs(n10 - n01) - 1) ** 2
    denominator = n10 + n01
    chi2 = numerator / denominator

    # p-value dari chi-square distribution (df=1)
    p_value = 1 - chi2_dist.cdf(chi2, df=1)

    return chi2, p_value, n10, n01


def _odds_ratio(n10: int, n01: int) -> Optional[float]:
    """
    Odds ratio sebagai effect size untuk McNemar.
    OR > 1: ARM A lebih baik dari B pada discordant pairs.
    """
    if n01 == 0:
        return float("inf") if n10 > 0 else float("nan")
    return n10 / n01


def run_statistical_analysis(
    results: List[Dict],
    alpha: float = 0.05,
) -> Dict:
    """
    Jalankan McNemar test untuk semua pasangan ARM.
    Terapkan Bonferroni correction untuk multiple comparisons.

    Args:
        results: list output dari evaluate_arm()
        alpha:   significance level sebelum correction (default 0.05)

    Returns:
        Dict berisi semua pairwise comparisons dan corrected p-values.
    """
    num_arms   = len(results)
    num_pairs  = num_arms * (num_arms - 1) // 2
    alpha_adj  = alpha / num_pairs if num_pairs > 0 else alpha  # Bonferroni

    comparisons = []

    for arm_a, arm_b in combinations(results, 2):
        name_a = arm_a["arm_name"]
        name_b = arm_b["arm_name"]

        # Ekstrak vektor benar/salah dari details
        correct_a = [d["correct"] for d in arm_a["details"]]
        correct_b = [d["correct"] for d in arm_b["details"]]

        chi2, p_val, n10, n01 = _mcnemar_statistic(correct_a, correct_b)
        or_val = _odds_ratio(n10, n01)

        # Tentukan arah: siapa yang lebih baik?
        acc_a = arm_a["accuracy"]
        acc_b = arm_b["accuracy"]
        if acc_a > acc_b:
            direction = f"{name_a} > {name_b}"
        elif acc_b > acc_a:
            direction = f"{name_b} > {name_a}"
        else:
            direction = "tie"

        significant = (
            not (p_val != p_val)  # not NaN
            and p_val < alpha_adj
        )

        comparisons.append({
            "arm_a":           name_a,
            "arm_b":           name_b,
            "accuracy_a":      acc_a,
            "accuracy_b":      acc_b,
            "direction":       direction,
            "n10":             n10,    # A benar, B salah
            "n01":             n01,    # A salah, B benar
            "mcnemar_chi2":    round(chi2, 4) if chi2 == chi2 else None,
            "p_value":         round(p_val, 4) if p_val == p_val else None,
            "alpha_adjusted":  round(alpha_adj, 4),
            "significant":     significant,
            "odds_ratio":      round(or_val, 3) if or_val not in (float("inf"), float("nan")) else str(or_val),
        })

    return {
        "num_comparisons": num_pairs,
        "alpha_original":  alpha,
        "alpha_bonferroni": round(alpha_adj, 4),
        "comparisons":     comparisons,
    }


def print_statistical_summary(analysis: Dict):
    """Pretty-print hasil analisis statistik ke console."""
    print("\n" + "=" * 70)
    print("ANALISIS STATISTIK — McNemar Test + Bonferroni Correction")
    print("=" * 70)
    print(f"Jumlah perbandingan: {analysis['num_comparisons']}")
    print(f"Alpha (original):    {analysis['alpha_original']}")
    print(f"Alpha (Bonferroni):  {analysis['alpha_bonferroni']}")
    print()

    header = f"{'Perbandingan':<42} {'p-val':>7} {'Sig':>5} {'OR':>7} {'Arah'}"
    print(header)
    print("-" * 75)

    for c in analysis["comparisons"]:
        pair = f"{c['arm_a']} vs {c['arm_b']}"
        p_str  = f"{c['p_value']:.4f}" if c["p_value"] is not None else "  N/A"
        sig    = "  *" if c["significant"] else "   "
        or_str = str(c["odds_ratio"])
        print(f"{pair:<42} {p_str:>7} {sig:>5} {or_str:>7}  {c['direction']}")

    print()
    sig_count = sum(1 for c in analysis["comparisons"] if c["significant"])
    print(f"Perbandingan signifikan: {sig_count}/{analysis['num_comparisons']}")
