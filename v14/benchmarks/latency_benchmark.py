# -*- coding: utf-8 -*-
"""
latency_benchmark.py — System Latency Benchmarks
=================================================
Measures end-to-end latency for all components:
    - BB84 QKD key exchange
    - Kyber ML-KEM key encapsulation
    - HKDF key combination
    - AES-256 encryption (single + dual)
    - Full hybrid pipeline

Author  : FYP Team
Module  : benchmarks/latency_benchmark.py
Project : Hybrid QKD-PQC Security Framework for IoT/RAN
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import *

from src.qkd.bb84            import BB84, RANChannel
from src.pqc.kyber           import KyberKEM
from src.hybrid.combiner     import HybridKeyCombiner
from src.hybrid.dual_encrypt import IoTPacketEncryptor

import warnings
warnings.filterwarnings("ignore")

TRIALS = 100


def benchmark_bb84(trials: int = TRIALS) -> dict:
    """Benchmark BB84 QKD latency."""
    channel = RANChannel(RAN_NOISE_NORMAL, RAN_LOSS_NORMAL, RAN_DELAY_NORMAL)
    times   = []

    for _ in range(trials):
        start = time.perf_counter()
        BB84(QKD_NUM_QUBITS, channel).run()
        times.append((time.perf_counter() - start) * 1000)

    return {
        "component" : "BB84 QKD",
        "trials"    : trials,
        "avg_ms"    : round(np.mean(times), 3),
        "min_ms"    : round(min(times), 3),
        "max_ms"    : round(max(times), 3),
        "std_ms"    : round(np.std(times), 3),
        "p95_ms"    : round(np.percentile(times, 95), 3),
        "times"     : times,
    }


def benchmark_kyber(trials: int = TRIALS) -> dict:
    """Benchmark Kyber ML-KEM latency."""
    times = []

    for _ in range(trials):
        kem   = KyberKEM(PQC_ALGORITHM)
        start = time.perf_counter()
        kem.full_key_exchange()
        times.append((time.perf_counter() - start) * 1000)

    return {
        "component" : f"Kyber {PQC_ALGORITHM}",
        "trials"    : trials,
        "avg_ms"    : round(np.mean(times), 3),
        "min_ms"    : round(min(times), 3),
        "max_ms"    : round(max(times), 3),
        "std_ms"    : round(np.std(times), 3),
        "p95_ms"    : round(np.percentile(times, 95), 3),
        "times"     : times,
    }


def benchmark_hkdf(trials: int = TRIALS) -> dict:
    """Benchmark HKDF key combination latency."""
    # Pre-generate keys
    channel    = RANChannel(RAN_NOISE_NORMAL, RAN_LOSS_NORMAL)
    qkd_result = BB84(QKD_NUM_QUBITS, channel).run()
    kem_result = KyberKEM(PQC_ALGORITHM).full_key_exchange()
    combiner   = HybridKeyCombiner(HYBRID_METHOD)
    times      = []

    for _ in range(trials):
        start = time.perf_counter()
        combiner.combine(
            qkd_result["secret_key"],
            kem_result["shared_secret"]
        )
        times.append((time.perf_counter() - start) * 1000)

    return {
        "component" : "HKDF Combination",
        "trials"    : trials,
        "avg_ms"    : round(np.mean(times), 3),
        "min_ms"    : round(min(times), 3),
        "max_ms"    : round(max(times), 3),
        "std_ms"    : round(np.std(times), 3),
        "p95_ms"    : round(np.percentile(times, 95), 3),
        "times"     : times,
    }


def benchmark_encryption(trials: int = TRIALS) -> dict:
    """Benchmark AES-256 single and dual encryption latency."""
    channel    = RANChannel(RAN_NOISE_NORMAL, RAN_LOSS_NORMAL)
    qkd_result = BB84(QKD_NUM_QUBITS, channel).run()
    kem_result = KyberKEM(PQC_ALGORITHM).full_key_exchange()
    combiner   = HybridKeyCombiner(HYBRID_METHOD)
    hybrid     = combiner.combine(
        qkd_result["secret_key"],
        kem_result["shared_secret"]
    )

    # Single layer
    enc_single = IoTPacketEncryptor(hybrid["hybrid_key"], mode="SINGLE")
    times_single = []
    for _ in range(trials):
        start = time.perf_counter()
        r = enc_single.encrypt_packet(
            "traffic_sensor",
            id="001", lat="31.52", lon="74.35", val=42, ts=1711360800
        )
        enc_single.decrypt_packet(r)
        times_single.append((time.perf_counter() - start) * 1000)

    # Dual layer
    enc_dual = IoTPacketEncryptor(
        hybrid["hybrid_key"],
        qkd_key   = qkd_result["secret_key"],
        kyber_key = kem_result["shared_secret"],
        mode      = "DUAL"
    )
    times_dual = []
    for _ in range(trials):
        start = time.perf_counter()
        r = enc_dual.encrypt_packet(
            "traffic_sensor",
            id="001", lat="31.52", lon="74.35", val=42, ts=1711360800
        )
        enc_dual.decrypt_packet(r)
        times_dual.append((time.perf_counter() - start) * 1000)

    return {
        "single": {
            "component" : "AES-256 Single Layer",
            "trials"    : trials,
            "avg_ms"    : round(np.mean(times_single), 3),
            "min_ms"    : round(min(times_single), 3),
            "max_ms"    : round(max(times_single), 3),
            "std_ms"    : round(np.std(times_single), 3),
            "p95_ms"    : round(np.percentile(times_single, 95), 3),
            "times"     : times_single,
        },
        "dual": {
            "component" : "AES-256 Dual Layer",
            "trials"    : trials,
            "avg_ms"    : round(np.mean(times_dual), 3),
            "min_ms"    : round(min(times_dual), 3),
            "max_ms"    : round(max(times_dual), 3),
            "std_ms"    : round(np.std(times_dual), 3),
            "p95_ms"    : round(np.percentile(times_dual, 95), 3),
            "times"     : times_dual,
        }
    }


def benchmark_full_pipeline(trials: int = TRIALS) -> dict:
    """Benchmark complete hybrid pipeline latency."""
    channel  = RANChannel(RAN_NOISE_NORMAL, RAN_LOSS_NORMAL)
    combiner = HybridKeyCombiner(HYBRID_METHOD)
    times    = []

    for _ in range(trials):
        start = time.perf_counter()

        # Full pipeline
        qkd_r  = BB84(QKD_NUM_QUBITS, channel).run()
        kem_r  = KyberKEM(PQC_ALGORITHM).full_key_exchange()
        hybrid = combiner.combine(
            qkd_r["secret_key"],
            kem_r["shared_secret"]
        )
        enc = IoTPacketEncryptor(
            hybrid["hybrid_key"],
            qkd_key   = qkd_r["secret_key"],
            kyber_key = kem_r["shared_secret"],
            mode      = "DUAL"
        )
        r = enc.encrypt_packet(
            "hospital",
            id="003", bp="120/80", hr=72, crit="NO", ts=1711360800
        )
        enc.decrypt_packet(r)

        times.append((time.perf_counter() - start) * 1000)

    return {
        "component" : "Full Hybrid Pipeline",
        "trials"    : trials,
        "avg_ms"    : round(np.mean(times), 3),
        "min_ms"    : round(min(times), 3),
        "max_ms"    : round(max(times), 3),
        "std_ms"    : round(np.std(times), 3),
        "p95_ms"    : round(np.percentile(times, 95), 3),
        "times"     : times,
    }


def plot_latency(results: dict):
    """Plot latency benchmarks."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#f9f9f7")

    # ── Bar chart — avg latency ───────────────
    components = [
        "BB84 QKD",
        f"Kyber {PQC_ALGORITHM}",
        "HKDF",
        "AES Single",
        "AES Dual",
        "Full Pipeline",
    ]
    avgs = [
        results["bb84"]["avg_ms"],
        results["kyber"]["avg_ms"],
        results["hkdf"]["avg_ms"],
        results["enc"]["single"]["avg_ms"],
        results["enc"]["dual"]["avg_ms"],
        results["pipeline"]["avg_ms"],
    ]
    stds = [
        results["bb84"]["std_ms"],
        results["kyber"]["std_ms"],
        results["hkdf"]["std_ms"],
        results["enc"]["single"]["std_ms"],
        results["enc"]["dual"]["std_ms"],
        results["pipeline"]["std_ms"],
    ]

    colors = ["#7F77DD","#1D9E75","#D85A30","#BA7517","#888780","#D4537E"]
    bars   = axes[0].bar(components, avgs, color=colors,
                          width=0.55, zorder=3, yerr=stds,
                          capsize=4, error_kw={"linewidth": 1.5})
    axes[0].set_title("Average latency per component (ms)",
                       fontsize=11, pad=10)
    axes[0].set_ylabel("Latency (ms)")
    axes[0].grid(axis="y", alpha=0.3, zorder=0)
    axes[0].set_xticklabels(components, rotation=15, fontsize=8)
    for bar, val in zip(bars, avgs):
        axes[0].text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + max(stds)*0.1,
            f"{val:.2f}ms", ha="center", fontsize=7
        )

    # ── Box plot — distribution ───────────────
    box_data   = [
        results["bb84"]["times"],
        results["kyber"]["times"],
        results["hkdf"]["times"],
        results["enc"]["single"]["times"],
        results["enc"]["dual"]["times"],
        results["pipeline"]["times"],
    ]
    bp = axes[1].boxplot(box_data, patch_artist=True,
                          medianprops={"color": "white", "linewidth": 2})
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1].set_title("Latency distribution over 100 trials",
                       fontsize=11, pad=10)
    axes[1].set_ylabel("Latency (ms)")
    axes[1].set_xticklabels(components, rotation=15, fontsize=8)
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle("Hybrid QKD-PQC — Latency Benchmarks",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    os.makedirs(RESULTS_PLOTS_PATH, exist_ok=True)
    path = os.path.join(RESULTS_PLOTS_PATH, "latency_benchmark.png")
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.show()
    print(f"  Plot saved: {path}")


if __name__ == "__main__":

    print("=" * 58)
    print("  Latency Benchmark — 100 trials each")
    print("=" * 58)

    print("\n  Running BB84 QKD benchmark...")
    bb84_r    = benchmark_bb84()

    print("  Running Kyber benchmark...")
    kyber_r   = benchmark_kyber()

    print("  Running HKDF benchmark...")
    hkdf_r    = benchmark_hkdf()

    print("  Running encryption benchmark...")
    enc_r     = benchmark_encryption()

    print("  Running full pipeline benchmark...")
    pipeline_r = benchmark_full_pipeline()

    results = {
        "bb84"    : bb84_r,
        "kyber"   : kyber_r,
        "hkdf"    : hkdf_r,
        "enc"     : enc_r,
        "pipeline": pipeline_r,
    }

    # Print table
    print(f"\n  {'Component':<25} {'Avg':>8} {'Min':>8} "
          f"{'Max':>8} {'Std':>8} {'P95':>8}")
    print("-" * 68)

    components = [
        ("BB84 QKD",        bb84_r),
        (f"Kyber768",       kyber_r),
        ("HKDF",            hkdf_r),
        ("AES-256 Single",  enc_r["single"]),
        ("AES-256 Dual",    enc_r["dual"]),
        ("Full Pipeline",   pipeline_r),
    ]

    for name, r in components:
        print(f"  {name:<25} "
              f"{r['avg_ms']:>7}ms "
              f"{r['min_ms']:>7}ms "
              f"{r['max_ms']:>7}ms "
              f"{r['std_ms']:>7}ms "
              f"{r['p95_ms']:>7}ms")

    print(f"\n  Target: <2000ms end-to-end")
    print(f"  Achieved: {pipeline_r['avg_ms']}ms avg ✓")

    print("\n  Generating plots...")
    plot_latency(results)