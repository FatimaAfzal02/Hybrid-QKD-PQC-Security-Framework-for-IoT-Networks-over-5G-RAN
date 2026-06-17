# -*- coding: utf-8 -*-
"""
metrics.py — Performance Metrics Tracker
=========================================
Collects, calculates, and reports performance metrics
across the full Hybrid QKD-PQC pipeline.

Metrics tracked per run:
    Latency     : per-operation and end-to-end (ms)
    Throughput  : key bits per second (Kbps / Mbps)
    Energy      : per-operation and daily (mJ / J)
    Detection   : attack detection rate (%)
    Key rate    : QKD key bits per qubit sent
    Overhead    : hybrid vs classical latency increase (%)

Why this module exists:
    logger.py   → saves raw results to disk (JSON/TXT)
    agent.py    → classifies network state
    metrics.py  → calculates derived performance numbers
                  that the supervisor flagged as missing:
                  throughput (Mbps), energy (J/bit),
                  overhead %, and scalability stats

Usage:
    from src.adaptive.metrics import PerformanceMetrics

    pm = PerformanceMetrics()
    pm.record_latency("bb84",  18.5)
    pm.record_latency("kyber",  0.8)
    pm.record_detection(detected=True)

    report = pm.summary()
    pm.print_report()

Author  : FYP Team
Module  : src/adaptive/metrics.py
Project : Hybrid QKD-PQC Security Framework for IoT/RAN
"""

import time
import numpy as np
from typing import Optional
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config import HYBRID_KEY_LENGTH, IOT_POWER_MW

from src.metrics.energy import EnergyEstimator

# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────

KEY_BITS               = HYBRID_KEY_LENGTH * 8   # 256 bits
DEFAULT_POWER_MW       = IOT_POWER_MW                # pulled from config.py
TARGET_LATENCY_MS      = 100.0                   # <100ms real-time IoT
TARGET_THROUGHPUT_MBPS = 1.0                     # >1 Mbps
TARGET_OVERHEAD_PCT    = 50.0                    # <50% vs classical
TARGET_DETECTION_RATE  = 95.0                    # >95% attack detection


# ─────────────────────────────────────────────
#  Performance Metrics Tracker
# ─────────────────────────────────────────────

class PerformanceMetrics:
    """
    Tracks and calculates all performance metrics for
    the Hybrid QKD-PQC system across multiple operations.

    Args:
        power_mw : IoT device power for energy estimates (default 25mW)
    """

    def __init__(self, power_mw: float = DEFAULT_POWER_MW):
        self.power_mw     = power_mw
        self.estimator    = EnergyEstimator(power_mw=power_mw)
        self._latencies   = {}    # {operation: [ms, ms, ...]}
        self._detections  = []    # [True, False, ...]
        self._key_rates   = []    # bits/qubit
        self._throughputs = []    # Mbps per trial
        self._trial_count = 0

    # ── Recording ────────────────────────────

    def record_latency(self, operation: str, latency_ms: float):
        """Record one latency measurement for a named operation."""
        if operation not in self._latencies:
            self._latencies[operation] = []
        self._latencies[operation].append(latency_ms)

    def record_trial(
        self,
        bb84_ms    : float,
        kyber_ms   : float,
        hkdf_ms    : float,
        encrypt_ms : float,
        detected   : Optional[bool] = None,
        key_rate   : Optional[float] = None
    ):
        """
        Record a full pipeline trial in one call.

        Args:
            bb84_ms    : BB84 QKD latency (ms)
            kyber_ms   : Kyber KEM latency (ms)
            hkdf_ms    : HKDF combination latency (ms)
            encrypt_ms : AES dual encryption latency (ms)
            detected   : eavesdropper detected? (None = not tested)
            key_rate   : QKD key rate bits/qubit (optional)
        """
        self.record_latency("bb84",    bb84_ms)
        self.record_latency("kyber",   kyber_ms)
        self.record_latency("hkdf",    hkdf_ms)
        self.record_latency("encrypt", encrypt_ms)

        pipeline_ms = bb84_ms + kyber_ms + hkdf_ms + encrypt_ms
        self.record_latency("pipeline", pipeline_ms)

        if pipeline_ms > 0:
            throughput_mbps = (KEY_BITS / (pipeline_ms / 1000)) / 1e6
            self._throughputs.append(throughput_mbps)

        if detected is not None:
            self._detections.append(detected)

        if key_rate is not None:
            self._key_rates.append(key_rate)

        self._trial_count += 1

    def record_detection(self, detected: bool):
        """Record one attack detection outcome."""
        self._detections.append(detected)

    def record_key_rate(self, bits_per_qubit: float):
        """Record QKD key rate efficiency."""
        self._key_rates.append(bits_per_qubit)

    # ── Calculations ─────────────────────────

    def latency_stats(self, operation: str) -> dict:
        """Latency statistics for one operation."""
        vals = self._latencies.get(operation, [])
        if not vals:
            return {"operation": operation, "n": 0, "avg_ms": 0.0}
        return {
            "operation"   : operation,
            "n"           : len(vals),
            "avg_ms"      : round(float(np.mean(vals)),            3),
            "min_ms"      : round(float(np.min(vals)),             3),
            "max_ms"      : round(float(np.max(vals)),             3),
            "std_ms"      : round(float(np.std(vals)),             3),
            "p95_ms"      : round(float(np.percentile(vals, 95)), 3),
            "meets_target": float(np.mean(vals)) < TARGET_LATENCY_MS,
        }

    def throughput_stats(self) -> dict:
        """Throughput statistics in Mbps."""
        if not self._throughputs:
            pipeline = self._latencies.get("pipeline", [])
            if pipeline:
                self._throughputs = [
                    (KEY_BITS / (ms / 1000)) / 1e6
                    for ms in pipeline
                ]
        if not self._throughputs:
            return {"avg_mbps": 0.0, "meets_target": False}

        avg = float(np.mean(self._throughputs))
        return {
            "n"           : len(self._throughputs),
            "avg_mbps"    : round(avg, 4),
            "min_mbps"    : round(float(np.min(self._throughputs)),  4),
            "max_mbps"    : round(float(np.max(self._throughputs)),  4),
            "std_mbps"    : round(float(np.std(self._throughputs)),  4),
            "target_mbps" : TARGET_THROUGHPUT_MBPS,
            "meets_target": avg >= TARGET_THROUGHPUT_MBPS,
            "status"      : "PASS ✓" if avg >= TARGET_THROUGHPUT_MBPS else "FAIL ✗",
        }

    def energy_stats(self) -> dict:
        """Energy statistics per operation and full pipeline."""
        result = {}
        for op, vals in self._latencies.items():
            avg_ms = float(np.mean(vals))
            result[op] = {
                "avg_latency_ms": round(avg_ms, 3),
                "energy_mj"     : self.estimator.from_latency_ms(avg_ms),
                "energy_j"      : self.estimator.from_latency_ms_j(avg_ms),
                "per_bit_uj"    : self.estimator.per_bit(avg_ms, KEY_BITS),
                "daily_j_2880"  : self.estimator.daily_energy_j(avg_ms, 2880),
            }

        pipeline_vals = self._latencies.get("pipeline", [])
        if pipeline_vals:
            avg_pipeline = float(np.mean(pipeline_vals))
            iot_check    = self.estimator.meets_iot_target(avg_pipeline)
            result["_totals"] = {
                "avg_pipeline_ms" : round(avg_pipeline, 3),
                "total_energy_mj" : iot_check["per_op_mj"],
                "total_energy_j"  : iot_check["per_op_j"],
                "daily_energy_j"  : iot_check["daily_j"],
                "iot_target_j"    : iot_check["target_j"],
                "meets_iot_target": iot_check["passes"],
                "status"          : iot_check["status"],
                "power_mw"        : self.power_mw,
            }
        return result

    def detection_stats(self) -> dict:
        """Attack detection rate statistics."""
        if not self._detections:
            return {"n": 0, "detection_rate_pct": 0.0, "meets_target": False}

        n        = len(self._detections)
        detected = sum(self._detections)
        rate_pct = round((detected / n) * 100, 2)
        return {
            "n"                  : n,
            "detected"           : detected,
            "missed"             : n - detected,
            "detection_rate_pct" : rate_pct,
            "target_pct"         : TARGET_DETECTION_RATE,
            "meets_target"       : rate_pct >= TARGET_DETECTION_RATE,
            "status"             : "PASS ✓" if rate_pct >= TARGET_DETECTION_RATE else "FAIL ✗",
        }

    def overhead_vs_classical(self, classical_latency_ms: float) -> dict:
        """
        Latency overhead of hybrid system vs classical baseline.
        Problem statement target: <50% overhead.
        """
        pipeline = self._latencies.get("pipeline", [])
        if not pipeline:
            return {"overhead_pct": None, "meets_target": False}

        hybrid_avg   = float(np.mean(pipeline))
        overhead_pct = round(
            ((hybrid_avg - classical_latency_ms) / classical_latency_ms) * 100, 2
        )
        return {
            "classical_ms" : classical_latency_ms,
            "hybrid_avg_ms": round(hybrid_avg, 3),
            "overhead_ms"  : round(hybrid_avg - classical_latency_ms, 3),
            "overhead_pct" : overhead_pct,
            "target_pct"   : TARGET_OVERHEAD_PCT,
            "meets_target" : overhead_pct < TARGET_OVERHEAD_PCT,
            "status"       : "PASS ✓" if overhead_pct < TARGET_OVERHEAD_PCT else "FAIL ✗",
        }

    def key_rate_stats(self) -> dict:
        """QKD key rate statistics (bits per qubit sent)."""
        if not self._key_rates:
            return {"n": 0, "avg_bits_per_qubit": 0.0}
        return {
            "n"                  : len(self._key_rates),
            "avg_bits_per_qubit" : round(float(np.mean(self._key_rates)), 4),
            "min_bits_per_qubit" : round(float(np.min(self._key_rates)),  4),
            "max_bits_per_qubit" : round(float(np.max(self._key_rates)),  4),
        }

    def summary(self, classical_latency_ms: float = 0.5) -> dict:
        """Full performance summary across all metrics."""
        return {
            "trial_count": self._trial_count,
            "latency"    : {op: self.latency_stats(op) for op in self._latencies},
            "throughput" : self.throughput_stats(),
            "energy"     : self.energy_stats(),
            "detection"  : self.detection_stats(),
            "overhead"   : self.overhead_vs_classical(classical_latency_ms),
            "key_rate"   : self.key_rate_stats(),
            "targets"    : {
                "latency_ms"     : TARGET_LATENCY_MS,
                "throughput_mbps": TARGET_THROUGHPUT_MBPS,
                "overhead_pct"   : TARGET_OVERHEAD_PCT,
                "detection_pct"  : TARGET_DETECTION_RATE,
            },
        }

    def print_report(self, classical_latency_ms: float = 0.5):
        """Print a formatted performance report to console."""
        s = self.summary(classical_latency_ms)

        print("=" * 58)
        print("  Performance Metrics Report")
        print("=" * 58)

        # Latency
        print("\n  Latency per operation:")
        print(f"  {'Operation':<14} {'Avg':>9} {'Min':>9} {'Max':>9} {'P95':>9}")
        print("  " + "-" * 44)
        for op, stats in s["latency"].items():
            if stats.get("n", 0) == 0:
                continue
            tick = "✓" if stats.get("meets_target") else "✗"
            ci_95 = 1.96 * stats["std_ms"] / np.sqrt(stats["n"]) if stats["n"] > 1 else 0.0
            print(f"  {op:<14} {stats['avg_ms']:>8.2f}ms "
                  f"{stats['min_ms']:>8.2f}ms "
                  f"{stats['max_ms']:>8.2f}ms "
                  f"{stats['p95_ms']:>8.2f}ms  95%CI±{ci_95:.2f}ms {tick}")

        # Throughput
        tp = s["throughput"]
        if tp.get("n", 0) > 0:
            print(f"\n  Throughput:")
            print(f"  Avg     : {tp['avg_mbps']} Mbps")
            print(f"  Target  : > {tp['target_mbps']} Mbps")
            print(f"  Status  : {tp['status']}")

        # Energy
        en = s["energy"].get("_totals", {})
        if en:
            print(f"\n  Energy ({self.power_mw}mW IoT device):")
            print(f"  Per op  : {en['total_energy_j']} J")
            print(f"  Daily   : {en['daily_energy_j']} J  (2880 ops/day)")
            print(f"  Target  : < {en['iot_target_j']} J/op")
            print(f"  Status  : {en['status']}")

        # Detection
        det = s["detection"]
        if det.get("n", 0) > 0:
            print(f"\n  Attack detection:")
            print(f"  Rate    : {det['detection_rate_pct']}%  "
                  f"({det['detected']}/{det['n']})")
            print(f"  Target  : > {det['target_pct']}%")
            print(f"  Status  : {det['status']}")

        # Overhead
        ov = s["overhead"]
        if ov.get("overhead_pct") is not None:
            print(f"\n  Overhead vs classical:")
            print(f"  Classical : {ov['classical_ms']} ms")
            print(f"  Hybrid    : {ov['hybrid_avg_ms']} ms")
            print(f"  Overhead  : {ov['overhead_pct']}%")
            print(f"  Target    : < {ov['target_pct']}%")
            print(f"  Status    : {ov['status']}")

        print("=" * 58)

    def reset(self):
        """Clear all recorded data."""
        self._latencies   = {}
        self._detections  = []
        self._key_rates   = []
        self._throughputs = []
        self._trial_count = 0


# ─────────────────────────────────────────────
#  TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 58)
    print("  src/adaptive/metrics.py — Module Self-Test")
    print("=" * 58)

    pm = PerformanceMetrics(power_mw=25.0)

    np.random.seed(42)
    print("\n  Simulating 20 pipeline trials...")

    for i in range(20):
        pm.record_trial(
            bb84_ms    = np.random.uniform(15.0, 22.0),
            kyber_ms   = np.random.uniform(0.5,   1.2),
            hkdf_ms    = np.random.uniform(0.03,  0.08),
            encrypt_ms = np.random.uniform(0.2,   0.5),
            detected   = (i < 18),
            key_rate   = np.random.uniform(0.10,  0.14),
        )

    pm.print_report(classical_latency_ms=0.5)

    print("\n  Target verification:")
    s = pm.summary(classical_latency_ms=0.5)
    checks = [
        ("Latency < 100ms",    s["latency"]["pipeline"]["meets_target"]),
        ("Throughput > 1Mbps", s["throughput"]["meets_target"]),
        ("Energy < 10J",       s["energy"]["_totals"]["meets_iot_target"]),
        ("Detection > 95%",    s["detection"]["meets_target"]),
        ("Overhead < 50%",     s["overhead"]["meets_target"]),
    ]
    for label, passed in checks:
        print(f"  [{'✓' if passed else '✗'}] {label}")
