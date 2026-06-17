# -*- coding: utf-8 -*-
"""
energy_benchmark.py — System Energy Consumption Benchmarks
===========================================================
Estimates energy consumption for all components using
CPU time as a proxy for energy on IoT devices.

Method:
    Energy (Joules) = Power (Watts) × Time (seconds)

    For IoT devices (e.g. ARM Cortex-M based):
        Active CPU power ≈ 10-50 mW typical
        We use 25mW as conservative IoT estimate

    This gives relative energy comparison between
    components — useful for IoT battery life analysis.

Components measured:
    - BB84 QKD
    - Kyber ML-KEM
    - HKDF combination
    - AES-256 single layer
    - AES-256 dual layer
    - Full hybrid pipeline

Author  : FYP Team
Module  : benchmarks/energy_benchmark.py
Project : Hybrid QKD-PQC Security Framework for IoT/RAN
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import *

from src.qkd.bb84 import BB84, RANChannel
from src.pqc.kyber import KyberKEM
from src.hybrid.combiner import HybridKeyCombiner
from src.hybrid.dual_encrypt import IoTPacketEncryptor

import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  IoT Device Power Profiles
#  Based on real IoT hardware datasheets
# ─────────────────────────────────────────────

IOT_POWER_PROFILES = {
    "high_end": {"name": "Raspberry Pi Zero 2W", "power_mw": 350},
    "mid_range": {"name": "ESP32", "power_mw": 80},
    "low_end": {"name": "ARM Cortex-M4", "power_mw": 25},
    "ultra_low": {"name": "ARM Cortex-M0+", "power_mw": 5},
}

TRIALS = 50


# ─────────────────────────────────────────────
#  Energy Estimator — imported from src.metrics
# ─────────────────────────────────────────────
# Single source of truth: src/metrics/energy.py
# Wrapper keeps estimate() / estimate_daily() names
# used throughout this benchmark file.

from src.metrics.energy import EnergyEstimator as _BaseEnergyEstimator

class EnergyEstimator(_BaseEnergyEstimator):
    """
    Thin wrapper around src.metrics.EnergyEstimator that
    preserves the estimate() / estimate_daily() API used
    in this benchmark script.
    """

    def estimate(self, time_ms: float) -> float:
        """Returns energy in millijoules (mJ)."""
        return self.from_latency_ms(time_ms)

    def estimate_daily(self, time_ms: float, operations_per_day: int) -> float:
        """Estimate daily energy in mJ for N operations per day."""
        return self.daily_energy_mj(time_ms, operations_per_day)


def measure_component(func, trials: int = TRIALS) -> dict:
    """Run a function N times and collect timing + energy stats."""
    estimator = EnergyEstimator(power_mw=25.0)
    times_ms = []
    energy_mj = []

    for _ in range(trials):
        start = time.perf_counter()
        func()
        elapsed_ms = (time.perf_counter() - start) * 1000
        times_ms.append(elapsed_ms)
        energy_mj.append(estimator.estimate(elapsed_ms))

    return {
        "trials": trials,
        "avg_ms": round(np.mean(times_ms), 3),
        "avg_mj": round(np.mean(energy_mj), 6),
        "min_mj": round(min(energy_mj), 6),
        "max_mj": round(max(energy_mj), 6),
        "std_mj": round(np.std(energy_mj), 6),
        "times_ms": times_ms,
        "energy_mj": energy_mj,
    }


if __name__ == "__main__":

    print("=" * 62)
    print("  Energy Consumption Benchmark")
    print("  IoT Device: ARM Cortex-M4 (25mW active power)")
    print("=" * 62)

    # Pre-generate shared objects
    channel = RANChannel(RAN_NOISE_NORMAL, RAN_LOSS_NORMAL)
    qkd_result = BB84(QKD_NUM_QUBITS, channel).run()
    kem_result = KyberKEM(PQC_ALGORITHM).full_key_exchange()
    combiner = HybridKeyCombiner(HYBRID_METHOD)
    hybrid = combiner.combine(
        qkd_result["secret_key"],
        kem_result["shared_secret"]
    )

    # ── Measure each component ────────────────
    print(f"\n  Running {TRIALS} trials each...")
    print("-" * 62)

    print("  [1/6] BB84 QKD...")
    bb84_r = measure_component(
        lambda: BB84(QKD_NUM_QUBITS, channel).run()
    )

    print("  [2/6] Kyber ML-KEM...")
    kyber_r = measure_component(
        lambda: KyberKEM(PQC_ALGORITHM).full_key_exchange()
    )

    print("  [3/6] HKDF combination...")
    hkdf_r = measure_component(
        lambda: combiner.combine(
            qkd_result["secret_key"],
            kem_result["shared_secret"]
        )
    )

    print("  [4/6] AES-256 single layer...")
    enc_single = IoTPacketEncryptor(hybrid["hybrid_key"], mode="SINGLE")
    single_r = measure_component(
        lambda: enc_single.decrypt_packet(
            enc_single.encrypt_packet(
                "hospital",
                id="003", bp="120/80", hr=72, crit="NO", ts=1711360800
            )
        )
    )

    print("  [5/6] AES-256 dual layer...")
    enc_dual = IoTPacketEncryptor(
        hybrid["hybrid_key"],
        qkd_key=qkd_result["secret_key"],
        kyber_key=kem_result["shared_secret"],
        mode="DUAL"
    )
    dual_r = measure_component(
        lambda: enc_dual.decrypt_packet(
            enc_dual.encrypt_packet(
                "hospital",
                id="003", bp="120/80", hr=72, crit="NO", ts=1711360800
            )
        )
    )

    print("  [6/6] Full hybrid pipeline...")


    def full_pipeline():
        qr = BB84(QKD_NUM_QUBITS, channel).run()
        kr = KyberKEM(PQC_ALGORITHM).full_key_exchange()
        h = combiner.combine(qr["secret_key"], kr["shared_secret"])
        e = IoTPacketEncryptor(
            h["hybrid_key"],
            qkd_key=qr["secret_key"],
            kyber_key=kr["shared_secret"],
            mode="DUAL"
        )
        e.decrypt_packet(e.encrypt_packet(
            "hospital",
            id="003", bp="120/80", hr=72, crit="NO", ts=1711360800
        ))


    pipeline_r = measure_component(full_pipeline)

    # ── Results table ─────────────────────────
    components = [
        ("BB84 QKD", bb84_r),
        ("Kyber768", kyber_r),
        ("HKDF", hkdf_r),
        ("AES-256 Single", single_r),
        ("AES-256 Dual", dual_r),
        ("Full Pipeline", pipeline_r),
    ]

    print(f"\n  {'Component':<20} {'Avg Time':>10} {'Avg Energy':>12} "
          f"{'Min':>10} {'Max':>10}")
    print("-" * 66)

    for name, r in components:
        print(f"  {name:<20} "
              f"{r['avg_ms']:>9.3f}ms "
              f"{r['avg_mj']:>11.6f}mJ "
              f"{r['min_mj']:>9.6f}mJ "
              f"{r['max_mj']:>9.6f}mJ")

    # ── Daily energy estimates ─────────────────
    print(f"\n  Daily energy estimate (key refresh every 30s = 2880 ops/day)")
    print("-" * 62)
    estimator = EnergyEstimator(25.0)
    ops = 2880

    for name, r in components:
        daily_mj = estimator.estimate_daily(r["avg_ms"], ops)
        daily_j = daily_mj / 1000
        print(f"  {name:<20} {daily_j:>10.4f} J/day  "
              f"({daily_mj:>10.2f} mJ/day)")

    # ── Multi device profile ──────────────────
    print(f"\n  Full pipeline energy across IoT device types")
    print("-" * 62)
    print(f"  {'Device':<25} {'Power':>8} {'Per op':>12} {'Daily':>12}")
    print("-" * 62)

    for key, profile in IOT_POWER_PROFILES.items():
        est = EnergyEstimator(profile["power_mw"])
        per_op = est.estimate(pipeline_r["avg_ms"])
        daily = est.estimate_daily(pipeline_r["avg_ms"], ops) / 1000
        print(f"  {profile['name']:<25} "
              f"{profile['power_mw']:>6}mW "
              f"{per_op:>10.4f}mJ "
              f"{daily:>10.4f}J/day")

    # ── Plots ─────────────────────────────────
    print("\n  Generating plots...")

    names = [c[0] for c in components]
    avgs = [c[1]["avg_mj"] for c in components]
    stds = [c[1]["std_mj"] for c in components]
    colors = ["#7F77DD", "#1D9E75", "#D85A30", "#BA7517", "#888780", "#D4537E"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#f9f9f7")

    # Bar chart
    bars = axes[0].bar(names, avgs, color=colors,
                       width=0.55, zorder=3,
                       yerr=stds, capsize=4,
                       error_kw={"linewidth": 1.5})
    axes[0].set_title("Energy per operation (mJ)\nIoT device: ARM Cortex-M4 @ 25mW",
                      fontsize=10, pad=10)
    axes[0].set_ylabel("Energy (mJ)")
    axes[0].grid(axis="y", alpha=0.3, zorder=0)
    axes[0].set_xticklabels(names, rotation=15, fontsize=8)
    for bar, val in zip(bars, avgs):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(stds) * 0.05,
            f"{val:.4f}", ha="center", fontsize=7
        )

    # Multi device comparison for full pipeline
    device_names = [p["name"] for p in IOT_POWER_PROFILES.values()]
    device_powers = [p["power_mw"] for p in IOT_POWER_PROFILES.values()]
    device_energy = [
        EnergyEstimator(p).estimate(pipeline_r["avg_ms"])
        for p in device_powers
    ]
    dev_colors = ["#D85A30", "#7F77DD", "#1D9E75", "#BA7517"]

    bars2 = axes[1].bar(device_names, device_energy,
                        color=dev_colors, width=0.55, zorder=3)
    axes[1].set_title("Full pipeline energy by IoT device type (mJ)",
                      fontsize=10, pad=10)
    axes[1].set_ylabel("Energy (mJ)")
    axes[1].grid(axis="y", alpha=0.3, zorder=0)
    axes[1].set_xticklabels(device_names, rotation=15, fontsize=8)
    for bar, val in zip(bars2, device_energy):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(device_energy) * 0.02,
            f"{val:.4f}mJ", ha="center", fontsize=8
        )

    fig.suptitle("Hybrid QKD-PQC — Energy Consumption Benchmarks",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    os.makedirs(RESULTS_PLOTS_PATH, exist_ok=True)
    path = os.path.join(RESULTS_PLOTS_PATH, "energy_benchmark.png")
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.show()
    print(f"  Plot saved: {path}")

    print(f"\n  Target: <10 J/day for full pipeline")
    daily_pipeline = estimator.estimate_daily(
        pipeline_r["avg_ms"], ops) / 1000
    status = "✓ PASS" if daily_pipeline < 10 else "✗ FAIL"
    print(f"  Achieved: {daily_pipeline:.4f} J/day [{status}]")