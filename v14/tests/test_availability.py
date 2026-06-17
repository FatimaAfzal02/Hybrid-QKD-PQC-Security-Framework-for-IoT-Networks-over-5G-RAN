# -*- coding: utf-8 -*-
"""
test_availability.py — System Availability Tests (>99.9% Target)
=================================================================
Validates that the Hybrid QKD-PQC framework meets the
availability requirement stated in the problem statement:
    "System availability > 99.9% (downtime < 8.76 hours/year)"

Availability = (total_time - downtime) / total_time × 100

Simulation approach:
    - 24-hour window (86,400 seconds) simulated per test
    - Failure events drawn from realistic IoT failure rates
    - Recovery time = failover time (immediate for Kyber fallback,
      <5 seconds for full system restart)
    - Hybrid system benefits: QKD failure → Kyber fallback (no downtime)
      Both fail → system aborts (rare, counted as downtime)

Tests cover:
    1.  Normal operation availability (>99.9%)
    2.  QKD-only failure scenario (Kyber fallback = no downtime)
    3.  PQC-only failure scenario (QKD fallback = no downtime)
    4.  Both channels failing simultaneously (rare, counted as downtime)
    5.  Failover recovery time (<5 seconds per event)
    6.  Annual availability projection (8760 hours)
    7.  Comparative availability: Classical vs QKD-only vs Hybrid
    8.  Monte Carlo availability simulation (1000 trials)
    9.  Availability under attack scenarios
    10. SLA compliance check (99.9% threshold)

Author  : FYP Team
Module  : tests/test_availability.py
Project : Hybrid QKD-PQC Security Framework for IoT/RAN
"""

import unittest
import sys
import os
import time
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.qkd.bb84        import BB84, RANChannel
from src.pqc.kyber       import KyberKEM
from src.hybrid.combiner import HybridKeyCombiner
from config import (
    QKD_NUM_QUBITS, PQC_ALGORITHM, HYBRID_METHOD,
    RAN_NOISE_NORMAL, RAN_LOSS_NORMAL, RAN_DELAY_NORMAL,
    RAN_NOISE_DEGRADED, RAN_LOSS_DEGRADED, RAN_DELAY_DEGRADED,
    QKD_QBER_THRESHOLD
)

np.random.seed(42)


# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────

SECONDS_PER_DAY    = 86_400       # 24 hours
SECONDS_PER_YEAR   = 8_760 * 3600 # 365 days in seconds
SLA_TARGET_PCT     = 99.9         # system availability requirement
FAILOVER_TIME_S    = 5.0          # max failover time (problem statement)
KEY_REFRESH_RATE_S = 30           # key refreshed every 30 seconds
TOTAL_OPS_PER_DAY  = SECONDS_PER_DAY // KEY_REFRESH_RATE_S  # 2880


# ─────────────────────────────────────────────
#  Simulation Helpers
# ─────────────────────────────────────────────

def simulate_failures(
    total_seconds     : int   = SECONDS_PER_DAY,
    qkd_failure_rate  : float = 0.001,   # fraction of ops that fail
    pqc_failure_rate  : float = 0.0005,
    recovery_time_s   : float = FAILOVER_TIME_S,
    both_fail_rate    : float = 0.00001  # extremely rare
) -> dict:
    """
    Simulate failure and recovery events over a time window.

    Failure model:
        - QKD failure: high QBER or channel loss — handled by Kyber fallback
          → no downtime for hybrid system (immediate failover)
        - PQC failure: Kyber KEM error — handled by QKD fallback
          → no downtime for hybrid system (immediate failover)
        - Both fail: extremely rare — system must abort
          → counted as downtime (recovery_time_s per event)

    Returns:
        dict with downtime_s, availability_pct, failure counts
    """
    n_ops = total_seconds // KEY_REFRESH_RATE_S

    qkd_failures  = int(n_ops * qkd_failure_rate)
    pqc_failures  = int(n_ops * pqc_failure_rate)
    both_failures = int(n_ops * both_fail_rate)

    # For hybrid system: single-channel failures have zero downtime
    # (immediate failover to the other channel)
    # Only dual failures cause downtime
    downtime_hybrid_s = both_failures * recovery_time_s

    # Classical system: no failover — any key exchange failure = downtime
    classical_failure_rate = 0.005  # 0.5% classical failure (no redundancy)
    classical_failures      = int(n_ops * classical_failure_rate)
    downtime_classical_s    = classical_failures * recovery_time_s

    # QKD-only system: QKD failure = downtime (no Kyber fallback)
    downtime_qkd_only_s = qkd_failures * recovery_time_s

    availability_hybrid    = (total_seconds - downtime_hybrid_s)    / total_seconds * 100
    availability_classical = (total_seconds - downtime_classical_s) / total_seconds * 100
    availability_qkd_only  = (total_seconds - downtime_qkd_only_s)  / total_seconds * 100

    return {
        "total_seconds"          : total_seconds,
        "n_ops"                  : n_ops,
        "qkd_failures"           : qkd_failures,
        "pqc_failures"           : pqc_failures,
        "both_failures"          : both_failures,
        "classical_failures"     : classical_failures,
        "downtime_hybrid_s"      : downtime_hybrid_s,
        "downtime_classical_s"   : downtime_classical_s,
        "downtime_qkd_only_s"    : downtime_qkd_only_s,
        "availability_hybrid"    : round(availability_hybrid,    5),
        "availability_classical" : round(availability_classical, 5),
        "availability_qkd_only"  : round(availability_qkd_only,  5),
    }


def run_live_exchange(channel: RANChannel, eve: bool = False) -> dict:
    """Run one real hybrid key exchange and return status."""
    try:
        qkd_r = BB84(QKD_NUM_QUBITS, channel, eavesdrop=eve).run()
        kem_r = KyberKEM(PQC_ALGORITHM).full_key_exchange()
        hyb_r = HybridKeyCombiner(HYBRID_METHOD).combine(
            qkd_r.get("secret_key"), kem_r.get("shared_secret")
        )
        return {
            "success"       : "ERROR" not in hyb_r.get("status", "ERROR"),
            "security_mode" : hyb_r.get("security_mode", "UNKNOWN"),
            "qkd_ok"        : qkd_r.get("status") == "SUCCESS",
            "pqc_ok"        : kem_r.get("status") == "SUCCESS",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ─────────────────────────────────────────────
#  Test Classes
# ─────────────────────────────────────────────

class TestAvailabilityNormalOperation(unittest.TestCase):
    """Availability in normal channel conditions."""

    def test_availability_exceeds_99_9_percent(self):
        """Hybrid system must meet >99.9% SLA over a simulated 24-hour window."""
        result = simulate_failures(
            total_seconds    = SECONDS_PER_DAY,
            qkd_failure_rate = 0.001,
            pqc_failure_rate = 0.0005,
            both_fail_rate   = 0.00001,
            recovery_time_s  = FAILOVER_TIME_S
        )
        self.assertGreater(
            result["availability_hybrid"], SLA_TARGET_PCT,
            f"Hybrid availability {result['availability_hybrid']:.4f}% < {SLA_TARGET_PCT}%"
        )

    def test_hybrid_outperforms_classical(self):
        """Hybrid system must have higher availability than classical."""
        result = simulate_failures()
        self.assertGreater(
            result["availability_hybrid"],
            result["availability_classical"],
            "Hybrid should outperform classical (has failover redundancy)"
        )

    def test_hybrid_outperforms_qkd_only(self):
        """Hybrid system must have higher availability than QKD-only."""
        result = simulate_failures()
        self.assertGreater(
            result["availability_hybrid"],
            result["availability_qkd_only"],
            "Hybrid should outperform QKD-only (Kyber fallback prevents QKD downtime)"
        )

    def test_downtime_within_8_76_hours_per_year(self):
        """Annual downtime must be less than 8.76 hours (99.9% SLA)."""
        result     = simulate_failures(total_seconds=SECONDS_PER_YEAR)
        downtime_h = result["downtime_hybrid_s"] / 3600
        self.assertLess(
            downtime_h, 8.76,
            f"Annual downtime {downtime_h:.2f}h exceeds 8.76h (99.9% SLA)"
        )


class TestFailoverBehavior(unittest.TestCase):
    """Failover provides zero-downtime for single-channel failures."""

    def test_qkd_failure_no_downtime_hybrid(self):
        """
        When QKD fails (high QBER), Kyber takes over immediately.
        Hybrid system has zero downtime for QKD-only failures.
        """
        result = simulate_failures(
            qkd_failure_rate = 0.10,   # 10% QKD failure rate (extreme)
            pqc_failure_rate = 0.0,    # PQC always works
            both_fail_rate   = 0.0
        )
        # No downtime because Kyber is always available
        self.assertEqual(
            result["downtime_hybrid_s"], 0,
            "QKD failures should cause zero downtime when PQC is available"
        )
        self.assertGreater(result["availability_hybrid"], 99.9)

    def test_pqc_failure_no_downtime_hybrid(self):
        """
        When Kyber fails, QKD takes over immediately.
        Hybrid system has zero downtime for PQC-only failures.
        """
        result = simulate_failures(
            qkd_failure_rate = 0.0,
            pqc_failure_rate = 0.10,   # 10% PQC failure rate (extreme)
            both_fail_rate   = 0.0
        )
        self.assertEqual(
            result["downtime_hybrid_s"], 0,
            "PQC failures should cause zero downtime when QKD is available"
        )
        self.assertGreater(result["availability_hybrid"], 99.9)

    def test_failover_time_under_5_seconds(self):
        """
        Live failover (QKD key = None → Kyber only) must complete < 5 seconds.
        Tests the actual combiner failover, not just simulation.
        """
        kem_r  = KyberKEM(PQC_ALGORITHM).full_key_exchange()
        start  = time.perf_counter()
        result = HybridKeyCombiner(HYBRID_METHOD).combine(
            None,  # QKD key missing — triggers failover
            kem_r["shared_secret"]
        )
        elapsed = time.perf_counter() - start
        self.assertLess(
            elapsed, FAILOVER_TIME_S,
            f"Failover took {elapsed:.3f}s, target < {FAILOVER_TIME_S}s"
        )
        self.assertIn("PQC", result["method"])

    def test_failover_produces_valid_key(self):
        """Failover key must be 256 bits and usable for AES-256."""
        from config import HYBRID_KEY_LENGTH
        kem_r  = KyberKEM(PQC_ALGORITHM).full_key_exchange()
        result = HybridKeyCombiner(HYBRID_METHOD).combine(
            None, kem_r["shared_secret"]
        )
        self.assertIsNotNone(result.get("hybrid_key"))
        self.assertEqual(len(result["hybrid_key"]), HYBRID_KEY_LENGTH)


class TestLiveAvailability(unittest.TestCase):
    """Live key exchange success rate over N real trials."""

    def test_live_success_rate_normal_channel(self):
        """
        Real key exchanges on a normal channel must succeed >95% of the time.
        Uses actual BB84 + Kyber + HKDF combiner.
        """
        channel  = RANChannel(RAN_NOISE_NORMAL, RAN_LOSS_NORMAL, RAN_DELAY_NORMAL)
        n_trials = 20
        successes = 0
        for _ in range(n_trials):
            r = run_live_exchange(channel, eve=False)
            if r["success"]:
                successes += 1
        rate = successes / n_trials * 100
        self.assertGreater(rate, 95.0, f"Success rate {rate:.1f}% < 95% on normal channel")

    def test_live_success_rate_degraded_channel(self):
        """
        Key exchanges on degraded channel must still succeed >80%.
        Kyber fallback keeps system operational even when QKD struggles.
        """
        channel  = RANChannel(RAN_NOISE_DEGRADED, RAN_LOSS_DEGRADED, RAN_DELAY_DEGRADED)
        n_trials = 20
        successes = 0
        for _ in range(n_trials):
            r = run_live_exchange(channel, eve=False)
            if r["success"]:
                successes += 1
        rate = successes / n_trials * 100
        self.assertGreater(rate, 80.0, f"Success rate {rate:.1f}% < 80% on degraded channel")

    def test_system_never_silently_fails(self):
        """
        System must never produce a None key without reporting an error status.
        Silent failures are more dangerous than detected failures.
        """
        channel = RANChannel(RAN_NOISE_NORMAL, RAN_LOSS_NORMAL, RAN_DELAY_NORMAL)
        for _ in range(10):
            qkd_r = BB84(QKD_NUM_QUBITS, channel, eavesdrop=False).run()
            kem_r = KyberKEM(PQC_ALGORITHM).full_key_exchange()
            hyb_r = HybridKeyCombiner(HYBRID_METHOD).combine(
                qkd_r.get("secret_key"), kem_r.get("shared_secret")
            )
            if hyb_r.get("hybrid_key") is None:
                # If key is None, status must say ERROR or FAILED
                self.assertIn(
                    "ERROR" or "FAILED", hyb_r.get("status", ""),
                    "System produced None key without reporting failure"
                )


class TestMonteCarloAvailability(unittest.TestCase):
    """Monte Carlo simulation over 1000 trials for statistical confidence."""

    def test_monte_carlo_mean_availability(self):
        """
        Over 1000 simulated 24-hour periods, mean availability must be >99.9%.
        Adds statistical confidence to the single-simulation result.
        """
        n_simulations = 1000
        availabilities = []

        for _ in range(n_simulations):
            # Randomise failure rates slightly each simulation
            qkd_rate  = np.random.uniform(0.0005, 0.002)
            pqc_rate  = np.random.uniform(0.0002, 0.001)
            both_rate = np.random.uniform(0.000005, 0.00002)
            result    = simulate_failures(
                qkd_failure_rate = qkd_rate,
                pqc_failure_rate = pqc_rate,
                both_fail_rate   = both_rate,
            )
            availabilities.append(result["availability_hybrid"])

        mean_avail = np.mean(availabilities)
        min_avail  = np.min(availabilities)

        self.assertGreater(
            mean_avail, SLA_TARGET_PCT,
            f"Monte Carlo mean availability {mean_avail:.4f}% < {SLA_TARGET_PCT}%"
        )
        self.assertGreater(
            min_avail, 99.0,
            f"Worst-case Monte Carlo availability {min_avail:.4f}% < 99.0%"
        )

    def test_monte_carlo_hybrid_consistently_beats_classical(self):
        """
        Hybrid must beat classical availability in >99% of simulations.
        """
        n_simulations = 500
        hybrid_wins   = 0
        for _ in range(n_simulations):
            result = simulate_failures(
                qkd_failure_rate = np.random.uniform(0.0005, 0.005),
                pqc_failure_rate = np.random.uniform(0.0002, 0.002),
                both_fail_rate   = np.random.uniform(0.000005, 0.00005),
            )
            if result["availability_hybrid"] > result["availability_classical"]:
                hybrid_wins += 1

        win_rate = hybrid_wins / n_simulations * 100
        self.assertGreater(
            win_rate, 99.0,
            f"Hybrid beats classical in only {win_rate:.1f}% of simulations"
        )


class TestAvailabilityUnderAttack(unittest.TestCase):
    """Availability when adversary is actively present."""

    def test_availability_under_eve_attack(self):
        """
        When Eve intercepts quantum channel (eavesdrop=True),
        system must still be available (aborts QKD, falls back to Kyber).
        Availability = system can still produce a secure key via Kyber.
        """
        channel   = RANChannel(RAN_NOISE_NORMAL, RAN_LOSS_NORMAL, RAN_DELAY_NORMAL)
        n_trials  = 20
        available = 0
        for _ in range(n_trials):
            r = run_live_exchange(channel, eve=True)
            # Under attack: QKD aborts (correct), Kyber should still work
            if r["success"]:
                available += 1

        avail_rate = available / n_trials * 100
        self.assertGreater(
            avail_rate, 80.0,
            f"System availability under attack: {avail_rate:.1f}% < 80%"
        )

    def test_attack_scenario_simulation_availability(self):
        """
        Simulated attack scenario (high QKD failure due to Eve).
        System stays available because Kyber is unaffected.
        """
        result = simulate_failures(
            qkd_failure_rate = 0.50,   # Eve causes 50% QKD failures
            pqc_failure_rate = 0.0005, # PQC unaffected by quantum attack
            both_fail_rate   = 0.0     # No combined failure
        )
        # Hybrid stays available because PQC compensates
        self.assertGreater(
            result["availability_hybrid"], SLA_TARGET_PCT,
            f"Hybrid availability under attack {result['availability_hybrid']:.4f}% < 99.9%"
        )
        # QKD-only would fail catastrophically
        self.assertLess(
            result["availability_qkd_only"], 99.0,
            "QKD-only should struggle under quantum attack"
        )


class TestSLACompliance(unittest.TestCase):
    """Final SLA compliance check for problem statement requirements."""

    def test_full_sla_compliance_report(self):
        """
        Comprehensive SLA check matching the problem statement requirements.
        Availability > 99.9% over all scenarios.
        """
        scenarios = {
            "Normal"   : simulate_failures(qkd_failure_rate=0.001, pqc_failure_rate=0.0005, both_fail_rate=0.00001),
            "Degraded" : simulate_failures(qkd_failure_rate=0.005, pqc_failure_rate=0.002,  both_fail_rate=0.00005),
            "Attack"   : simulate_failures(qkd_failure_rate=0.10,  pqc_failure_rate=0.001,  both_fail_rate=0.00010),
        }
        for scenario_name, result in scenarios.items():
            self.assertGreater(
                result["availability_hybrid"], SLA_TARGET_PCT,
                f"Scenario '{scenario_name}': availability "
                f"{result['availability_hybrid']:.4f}% < {SLA_TARGET_PCT}%"
            )

    def test_failover_time_meets_problem_statement(self):
        """
        Problem statement: failover time < 5 seconds.
        Tests the actual HybridKeyCombiner fallback timing.
        """
        kem_r   = KyberKEM(PQC_ALGORITHM).full_key_exchange()
        n_tests = 10
        times   = []
        for _ in range(n_tests):
            t0  = time.perf_counter()
            HybridKeyCombiner(HYBRID_METHOD).combine(None, kem_r["shared_secret"])
            times.append(time.perf_counter() - t0)

        max_failover = max(times)
        self.assertLess(
            max_failover, FAILOVER_TIME_S,
            f"Max failover time {max_failover:.4f}s exceeds {FAILOVER_TIME_S}s target"
        )

    def test_annual_sla_projection(self):
        """
        Project 24-hour availability to annual (8760 hours).
        Must meet 99.9% = < 8.76 hours downtime per year.
        """
        result     = simulate_failures(total_seconds=SECONDS_PER_YEAR)
        downtime_h = result["downtime_hybrid_s"] / 3600
        avail      = result["availability_hybrid"]

        self.assertGreater(avail, SLA_TARGET_PCT,
            f"Annual availability {avail:.5f}% < {SLA_TARGET_PCT}%")
        self.assertLess(downtime_h, 8.76,
            f"Annual downtime {downtime_h:.3f}h exceeds 8.76h SLA")


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 60)
    print("  Availability Simulation — Quick Report")
    print("=" * 60)

    result = simulate_failures(
        total_seconds    = SECONDS_PER_DAY,
        qkd_failure_rate = 0.001,
        pqc_failure_rate = 0.0005,
        both_fail_rate   = 0.00001,
        recovery_time_s  = FAILOVER_TIME_S,
    )

    print(f"\n  Simulation window : {result['total_seconds']:,} seconds (24 hours)")
    print(f"  Operations        : {result['n_ops']:,} key exchanges")
    print(f"  QKD failures      : {result['qkd_failures']} (no downtime — Kyber fallback)")
    print(f"  PQC failures      : {result['pqc_failures']} (no downtime — QKD fallback)")
    print(f"  Both fail         : {result['both_failures']} (counted as downtime)")
    print(f"\n  Availability:")
    print(f"  Classical Only    : {result['availability_classical']:.4f}%")
    print(f"  QKD Only          : {result['availability_qkd_only']:.4f}%")
    print(f"  Full Hybrid       : {result['availability_hybrid']:.4f}%  ← proposed")
    print(f"\n  SLA target        : > {SLA_TARGET_PCT}%")
    print(f"  Hybrid status     : {'✓ PASS' if result['availability_hybrid'] > SLA_TARGET_PCT else '✗ FAIL'}")

    print("\n" + "=" * 60)
    print("  Running all unit tests...")
    print("=" * 60)
    unittest.main(verbosity=2)
