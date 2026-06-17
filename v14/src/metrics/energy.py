# -*- coding: utf-8 -*-
"""
energy.py — IoT Energy Estimation Module
=========================================
Reusable energy measurement and estimation utilities
for the Hybrid QKD-PQC framework.

Method:
    Energy (Joules) = Power (Watts) × Time (seconds)

    For IoT devices (e.g. ARM Cortex-M based):
        Active CPU power ≈ 5–350 mW depending on device class.
        We use 25 mW as the conservative IoT baseline
        (ARM Cortex-M4) unless a profile is specified.

    Energy per bit (J/bit) is reported alongside total energy
    to allow fair comparison between key sizes.

    This module is importable by tests, benchmarks, and main.py.
    The benchmark script (benchmarks/energy_benchmark.py) handles
    multi-trial measurement and plotting; this module handles the
    core calculations and per-operation tracking.

IoT Device Profiles (from hardware datasheets):
    ultra_low : ARM Cortex-M0+      5 mW   (e.g. Nordic nRF51)
    low_end   : ARM Cortex-M4      25 mW   (e.g. STM32L4)
    mid_range : ESP32               80 mW   (e.g. ESP32-D0WD)
    high_end  : Raspberry Pi Zero  350 mW   (e.g. RPi Zero 2W)

Usage:
    from src.metrics.energy import EnergyEstimator, OperationTracker

    est    = EnergyEstimator(power_mw=25.0)
    energy = est.from_latency_ms(18.5)      # → mJ
    bits   = est.per_bit(18.5, key_bits=256)  # → J/bit

    tracker = OperationTracker()
    tracker.record("bb84",  latency_ms=18.5, key_bits=256)
    tracker.record("kyber", latency_ms=0.8,  key_bits=256)
    print(tracker.summary())

Author  : FYP Team
Module  : src/metrics/energy.py
Project : Hybrid QKD-PQC Security Framework for IoT/RAN
"""

import time
import statistics
from typing import Optional

# ─────────────────────────────────────────────
#  IoT Device Power Profiles
# ─────────────────────────────────────────────

IOT_PROFILES = {
    "ultra_low" : {"name": "ARM Cortex-M0+ (nRF51)",  "power_mw":   5.0},
    "low_end"   : {"name": "ARM Cortex-M4 (STM32L4)", "power_mw":  25.0},
    "mid_range" : {"name": "ESP32-D0WD",               "power_mw":  80.0},
    "high_end"  : {"name": "Raspberry Pi Zero 2W",     "power_mw": 350.0},
}

# Default profile used throughout the project
DEFAULT_PROFILE   = "low_end"
DEFAULT_POWER_MW  = IOT_PROFILES[DEFAULT_PROFILE]["power_mw"]  # 25 mW

# IoT target — pulled from config.py
import sys as _sys, os as _os
_sys.path.append(_os.path.join(_os.path.dirname(__file__), '..', '..'))
try:
    from config import IOT_POWER_MW as _IOT_POWER_MW, ENERGY_TARGET_J as _ENERGY_TARGET_J
    DEFAULT_POWER_MW         = _IOT_POWER_MW
    TARGET_ENERGY_J_PER_KEY  = _ENERGY_TARGET_J
except ImportError:
    TARGET_ENERGY_J_PER_KEY  = 10.0   # fallback


# ─────────────────────────────────────────────
#  Core Energy Estimator
# ─────────────────────────────────────────────

class EnergyEstimator:
    """
    Estimates energy consumption from execution latency.

    All energy values are in millijoules (mJ) unless
    the method name says otherwise (_j suffix = Joules).

    Args:
        power_mw : device active power in milliwatts (default 25 mW)
        profile  : optional profile key from IOT_PROFILES
                   (overrides power_mw if provided)
    """

    def __init__(
        self,
        power_mw : float = DEFAULT_POWER_MW,
        profile  : Optional[str] = None
    ):
        if profile and profile in IOT_PROFILES:
            self.power_mw = IOT_PROFILES[profile]["power_mw"]
            self.profile  = profile
        else:
            self.power_mw = power_mw
            self.profile  = "custom"

        self.power_w = self.power_mw / 1000.0

    # ── Core calculations ─────────────────────

    def from_latency_ms(self, latency_ms: float) -> float:
        """
        Estimate energy from latency.

        Args:
            latency_ms : operation duration in milliseconds

        Returns:
            Energy in millijoules (mJ)
        """
        time_s    = latency_ms / 1000.0
        energy_j  = self.power_w * time_s
        return round(energy_j * 1000, 6)   # mJ

    def from_latency_ms_j(self, latency_ms: float) -> float:
        """Same as from_latency_ms but returns Joules."""
        return round(self.from_latency_ms(latency_ms) / 1000.0, 9)

    def per_bit(self, latency_ms: float, key_bits: int) -> float:
        """
        Energy per bit of key generated (J/bit).

        Useful for comparing efficiency across different
        key sizes and algorithms.

        Args:
            latency_ms : operation duration in milliseconds
            key_bits   : number of key bits produced

        Returns:
            Energy per bit in microjoules (µJ/bit)
        """
        if key_bits <= 0:
            return 0.0
        energy_mj  = self.from_latency_ms(latency_ms)
        energy_uj  = energy_mj * 1000          # mJ → µJ
        return round(energy_uj / key_bits, 6)   # µJ/bit

    # ── Aggregate calculations ─────────────────

    def daily_energy_mj(
        self,
        latency_ms         : float,
        operations_per_day : int
    ) -> float:
        """
        Total daily energy for N operations.

        Args:
            latency_ms         : per-operation latency (ms)
            operations_per_day : number of key exchanges per day

        Returns:
            Daily energy in millijoules (mJ)
        """
        return round(self.from_latency_ms(latency_ms) * operations_per_day, 4)

    def daily_energy_j(
        self,
        latency_ms         : float,
        operations_per_day : int
    ) -> float:
        """Same as daily_energy_mj but returns Joules."""
        return round(self.daily_energy_mj(latency_ms, operations_per_day) / 1000.0, 6)

    def meets_iot_target(
        self,
        latency_ms         : float,
        operations_per_day : int = 2880,    # refresh every 30s
        target_j           : float = TARGET_ENERGY_J_PER_KEY
    ) -> dict:
        """
        Check whether energy meets the IoT target from
        the problem statement (< 10 J per key exchange).

        Args:
            latency_ms         : per-operation latency (ms)
            operations_per_day : key exchanges per day (default 2880)
            target_j           : energy target in Joules (default 10 J)

        Returns:
            dict with energy values and pass/fail status
        """
        per_op_mj  = self.from_latency_ms(latency_ms)
        per_op_j   = per_op_mj / 1000.0
        daily_j    = self.daily_energy_j(latency_ms, operations_per_day)
        passes     = per_op_j < target_j

        return {
            "per_op_mj"         : per_op_mj,
            "per_op_j"          : round(per_op_j, 9),
            "daily_j"           : daily_j,
            "operations_per_day": operations_per_day,
            "target_j"          : target_j,
            "passes"            : passes,
            "status"            : "PASS ✓" if passes else "FAIL ✗",
            "power_mw"          : self.power_mw,
            "profile"           : self.profile,
        }

    # ── Profile comparison ────────────────────

    @staticmethod
    def compare_profiles(latency_ms: float, key_bits: int = 256) -> dict:
        """
        Estimate energy across all IoT device profiles
        for a given operation latency.

        Args:
            latency_ms : operation latency in milliseconds
            key_bits   : key size produced by operation

        Returns:
            dict keyed by profile name with energy stats
        """
        results = {}
        for key, profile in IOT_PROFILES.items():
            est = EnergyEstimator(power_mw=profile["power_mw"])
            results[key] = {
                "device"    : profile["name"],
                "power_mw"  : profile["power_mw"],
                "energy_mj" : est.from_latency_ms(latency_ms),
                "energy_j"  : est.from_latency_ms_j(latency_ms),
                "per_bit_uj": est.per_bit(latency_ms, key_bits),
                "daily_j"   : est.daily_energy_j(latency_ms, 2880),
            }
        return results


# ─────────────────────────────────────────────
#  Operation Tracker
# ─────────────────────────────────────────────

class OperationTracker:
    """
    Tracks energy and latency across multiple named operations.

    Use this to accumulate measurements during a full pipeline
    run, then call summary() for a complete breakdown.

    Usage:
        tracker = OperationTracker(power_mw=25.0)
        tracker.record("bb84",  latency_ms=18.5, key_bits=256)
        tracker.record("kyber", latency_ms=0.8,  key_bits=256)
        tracker.record("hkdf",  latency_ms=0.05, key_bits=256)
        print(tracker.summary())
    """

    def __init__(self, power_mw: float = DEFAULT_POWER_MW):
        self.estimator = EnergyEstimator(power_mw=power_mw)
        self._records  = []

    def record(
        self,
        operation  : str,
        latency_ms : float,
        key_bits   : int = 256
    ) -> dict:
        """
        Record one operation's energy and latency.

        Args:
            operation  : operation name (e.g. "bb84", "kyber")
            latency_ms : measured latency in milliseconds
            key_bits   : key bits produced (for J/bit calc)

        Returns:
            The recorded entry dict
        """
        entry = {
            "operation"  : operation,
            "latency_ms" : round(latency_ms, 4),
            "energy_mj"  : self.estimator.from_latency_ms(latency_ms),
            "energy_j"   : self.estimator.from_latency_ms_j(latency_ms),
            "per_bit_uj" : self.estimator.per_bit(latency_ms, key_bits),
            "key_bits"   : key_bits,
        }
        self._records.append(entry)
        return entry

    def total_energy_mj(self) -> float:
        """Sum of all recorded operation energies in mJ."""
        return round(sum(r["energy_mj"] for r in self._records), 6)

    def total_latency_ms(self) -> float:
        """Sum of all recorded operation latencies in ms."""
        return round(sum(r["latency_ms"] for r in self._records), 4)

    def summary(self) -> dict:
        """
        Return full energy and latency summary.

        Returns:
            dict with per-operation breakdown and totals
        """
        total_mj = self.total_energy_mj()
        total_j  = round(total_mj / 1000.0, 9)
        total_ms = self.total_latency_ms()

        return {
            "operations"      : self._records,
            "total_latency_ms": total_ms,
            "total_energy_mj" : total_mj,
            "total_energy_j"  : total_j,
            "power_mw"        : self.estimator.power_mw,
            "iot_target_j"    : TARGET_ENERGY_J_PER_KEY,
            "meets_target"    : total_j < TARGET_ENERGY_J_PER_KEY,
            "status"          : (
                "PASS ✓" if total_j < TARGET_ENERGY_J_PER_KEY else "FAIL ✗"
            ),
        }

    def reset(self):
        """Clear all recorded operations."""
        self._records = []


# ─────────────────────────────────────────────
#  Context Manager for Timed Energy Measurement
# ─────────────────────────────────────────────

class MeasureEnergy:
    """
    Context manager that measures latency and estimates energy
    for any block of code.

    Usage:
        est = EnergyEstimator(power_mw=25.0)

        with MeasureEnergy(est, key_bits=256) as m:
            result = bb84.run()

        print(m.latency_ms)   # e.g. 18.4
        print(m.energy_mj)    # e.g. 0.0046
        print(m.per_bit_uj)   # e.g. 0.018
    """

    def __init__(
        self,
        estimator : Optional[EnergyEstimator] = None,
        key_bits  : int = 256
    ):
        self.estimator  = estimator or EnergyEstimator()
        self.key_bits   = key_bits
        self.latency_ms = 0.0
        self.energy_mj  = 0.0
        self.energy_j   = 0.0
        self.per_bit_uj = 0.0
        self._start     = None

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        elapsed_s       = time.perf_counter() - self._start
        self.latency_ms = round(elapsed_s * 1000, 4)
        self.energy_mj  = self.estimator.from_latency_ms(self.latency_ms)
        self.energy_j   = self.estimator.from_latency_ms_j(self.latency_ms)
        self.per_bit_uj = self.estimator.per_bit(self.latency_ms, self.key_bits)

    def as_dict(self) -> dict:
        return {
            "latency_ms" : self.latency_ms,
            "energy_mj"  : self.energy_mj,
            "energy_j"   : self.energy_j,
            "per_bit_uj" : self.per_bit_uj,
            "key_bits"   : self.key_bits,
        }


# ─────────────────────────────────────────────
#  TEST — Run when executed directly
# ─────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 58)
    print("  src/metrics/energy.py — Module Self-Test")
    print("=" * 58)

    est = EnergyEstimator(power_mw=25.0)

    # ── Test 1: Basic estimates ───────────────
    print("\n  Test 1: Basic energy estimates (25mW IoT device)")
    print("-" * 58)
    test_ops = [
        ("BB84 QKD",      18.5,  256),
        ("Kyber768 KEM",   0.8,  256),
        ("HKDF combiner",  0.05, 256),
        ("AES-256 dual",   0.3,  256),
        ("Full pipeline", 20.0,  256),
    ]
    print(f"  {'Operation':<22} {'Latency':>9} {'Energy':>12} {'J/bit':>12}")
    print("-" * 58)
    for name, lat, bits in test_ops:
        mj  = est.from_latency_ms(lat)
        upb = est.per_bit(lat, bits)
        print(f"  {name:<22} {lat:>8.2f}ms {mj:>10.6f}mJ {upb:>10.6f}µJ/bit")

    # ── Test 2: IoT target check ──────────────
    print("\n  Test 2: IoT energy target check (< 10 J/key)")
    print("-" * 58)
    result = est.meets_iot_target(latency_ms=20.0, operations_per_day=2880)
    print(f"  Per operation : {result['per_op_j']:.9f} J")
    print(f"  Daily total   : {result['daily_j']:.6f} J")
    print(f"  Target        : < {result['target_j']} J/op")
    print(f"  Status        : {result['status']}")

    # ── Test 3: Profile comparison ────────────
    print("\n  Test 3: Energy across device profiles (20ms pipeline)")
    print("-" * 58)
    profiles = EnergyEstimator.compare_profiles(latency_ms=20.0, key_bits=256)
    print(f"  {'Device':<30} {'Power':>8} {'Energy':>12} {'Daily':>12}")
    print("-" * 58)
    for key, data in profiles.items():
        print(f"  {data['device']:<30} {data['power_mw']:>6}mW "
              f"{data['energy_mj']:>10.6f}mJ "
              f"{data['daily_j']:>10.4f}J/day")

    # ── Test 4: Operation tracker ─────────────
    print("\n  Test 4: OperationTracker — full pipeline")
    print("-" * 58)
    tracker = OperationTracker(power_mw=25.0)
    tracker.record("bb84",     latency_ms=18.5, key_bits=256)
    tracker.record("kyber768", latency_ms=0.8,  key_bits=256)
    tracker.record("hkdf",     latency_ms=0.05, key_bits=256)
    tracker.record("aes_dual", latency_ms=0.3,  key_bits=256)
    summary = tracker.summary()
    print(f"  Total latency : {summary['total_latency_ms']} ms")
    print(f"  Total energy  : {summary['total_energy_mj']} mJ")
    print(f"  Total energy  : {summary['total_energy_j']} J")
    print(f"  IoT target    : < {summary['iot_target_j']} J")
    print(f"  Status        : {summary['status']}")

    # ── Test 5: Context manager ───────────────
    print("\n  Test 5: MeasureEnergy context manager")
    print("-" * 58)
    import hashlib

    with MeasureEnergy(est, key_bits=256) as m:
        # Simulate a small computation
        for _ in range(10000):
            hashlib.sha256(b"test").digest()

    print(f"  Latency  : {m.latency_ms} ms")
    print(f"  Energy   : {m.energy_mj} mJ")
    print(f"  Per bit  : {m.per_bit_uj} µJ/bit")

    print("\n  All tests passed ✓")
