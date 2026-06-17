# -*- coding: utf-8 -*-
"""
test_scalability.py — Scalability Tests (100+ Devices)
========================================================
Validates that the Hybrid QKD-PQC framework scales to
100+ concurrent IoT devices as required by the problem
statement ("city-wide deployment" objective).

Tests cover:
    1.  Network creation at scale (100, 200, 500 devices)
    2.  Hybrid key exchange across all devices
    3.  Per-device latency within IoT target (<100ms)
    4.  Success rate across all devices (>95%)
    5.  Network stats aggregation at scale
    6.  Mixed scenario scalability (normal/degraded/attack)
    7.  Failover under load (QKD fails → Kyber fallback)
    8.  Energy budget across 100 devices
    9.  QBER consistency across large network
    10. Device ID uniqueness at scale

Author  : FYP Team
Module  : tests/test_scalability.py
Project : Hybrid QKD-PQC Security Framework for IoT/RAN
"""

import unittest
import sys
import os
import time
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.qkd.bb84        import BB84
from src.pqc.kyber       import KyberKEM
from src.hybrid.combiner import HybridKeyCombiner
from src.iot.network     import IoTNetwork, RANChannel
from config import (
    QKD_NUM_QUBITS, PQC_ALGORITHM, HYBRID_METHOD,
    RAN_NOISE_NORMAL, RAN_LOSS_NORMAL
)


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def run_hybrid_exchange_for_device(device: dict) -> dict:
    """
    Run a full hybrid key exchange for one IoT device.
    Returns a result dict with status, latency, and key info.
    """
    start   = time.perf_counter()
    channel = device["channel"]

    qkd_result = BB84(
        num_qubits = QKD_NUM_QUBITS,
        channel    = channel,
        eavesdrop  = False
    ).run()

    kem_result = KyberKEM(PQC_ALGORITHM).full_key_exchange()

    combiner  = HybridKeyCombiner(HYBRID_METHOD)
    qkd_key   = qkd_result.get("secret_key")
    kyber_key = kem_result.get("shared_secret")
    hybrid    = combiner.combine(qkd_key, kyber_key)

    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

    return {
        "device_id"     : device["id"],
        "device_type"   : device["type"],
        "status"        : hybrid.get("status", "ERROR"),
        "security_mode" : hybrid.get("security_mode", "UNKNOWN"),
        "hybrid_key_ok" : hybrid.get("hybrid_key") is not None,
        "latency_ms"    : elapsed_ms,
        "qber"          : qkd_result.get("qber", 0.0),
        "qkd_status"    : qkd_result.get("status", "UNKNOWN"),
    }


def run_network_exchange(num_devices: int, scenario: str = "normal") -> list:
    """Run hybrid key exchange for all devices in an IoTNetwork."""
    network = IoTNetwork(num_devices=num_devices, scenario=scenario)
    return [run_hybrid_exchange_for_device(d) for d in network.devices]


# ─────────────────────────────────────────────
#  Test Classes
# ─────────────────────────────────────────────

class TestNetworkCreationAtScale(unittest.TestCase):
    """Tests that IoTNetwork creates correctly at various scales."""

    def test_network_100_devices_created(self):
        """Network should create exactly 100 devices."""
        network = IoTNetwork(num_devices=100, scenario="normal")
        self.assertEqual(len(network.devices), 100)

    def test_network_200_devices_created(self):
        """Network should scale to 200 devices."""
        network = IoTNetwork(num_devices=200, scenario="normal")
        self.assertEqual(len(network.devices), 200)

    def test_network_500_devices_created(self):
        """Network should scale to 500 devices."""
        network = IoTNetwork(num_devices=500, scenario="normal")
        self.assertEqual(len(network.devices), 500)

    def test_each_device_has_required_fields(self):
        """Every device must have id, type, channel, priority, distance."""
        network = IoTNetwork(num_devices=100)
        for device in network.devices:
            self.assertIn("id",       device)
            self.assertIn("type",     device)
            self.assertIn("channel",  device)
            self.assertIn("priority", device)
            self.assertIn("distance", device)

    def test_device_ids_are_unique(self):
        """All device IDs must be unique across the network."""
        network = IoTNetwork(num_devices=100)
        ids = [d["id"] for d in network.devices]
        self.assertEqual(len(ids), len(set(ids)))

    def test_all_device_types_represented(self):
        """All 5 device types should appear in a 100-device network."""
        network  = IoTNetwork(num_devices=100)
        types    = {d["type"] for d in network.devices}
        expected = {"traffic_sensor", "smart_meter", "camera",
                    "mobile_sensor", "hospital_device"}
        self.assertEqual(types, expected)

    def test_network_stats_correct_device_count(self):
        """get_network_stats should report correct device count."""
        network = IoTNetwork(num_devices=100)
        stats   = network.get_network_stats()
        self.assertEqual(stats["num_devices"], 100)


class TestHybridKeyExchangeAtScale(unittest.TestCase):
    """Tests hybrid key exchange across 100 devices."""

    @classmethod
    def setUpClass(cls):
        """Run one 100-device exchange — shared across all tests in class."""
        cls.results     = run_network_exchange(100, scenario="normal")
        cls.num_devices = 100

    def test_all_devices_produce_result(self):
        """Every device should return a result dict."""
        self.assertEqual(len(self.results), self.num_devices)

    def test_all_devices_have_hybrid_key(self):
        """Every device should successfully produce a hybrid key."""
        no_key = [r for r in self.results if not r["hybrid_key_ok"]]
        self.assertEqual(len(no_key), 0,
            f"{len(no_key)} devices failed to produce a hybrid key")

    def test_success_rate_above_95_percent(self):
        """At least 95% of devices must complete successfully."""
        successful = sum(
            1 for r in self.results
            if r["hybrid_key_ok"] and "ERROR" not in r["status"]
        )
        rate = (successful / self.num_devices) * 100
        self.assertGreaterEqual(rate, 95.0,
            f"Success rate {rate:.1f}% is below 95% target")

    def test_per_device_latency_under_100ms(self):
        """Each device's key exchange must complete within 100ms."""
        slow = [r for r in self.results if r["latency_ms"] > 100.0]
        self.assertEqual(len(slow), 0,
            f"{len(slow)} devices exceeded 100ms: "
            f"{[r['latency_ms'] for r in slow]}")

    def test_average_latency_under_50ms(self):
        """Average latency across all devices should be under 50ms."""
        avg = np.mean([r["latency_ms"] for r in self.results])
        self.assertLess(avg, 50.0,
            f"Average latency {avg:.2f}ms exceeds 50ms")

    def test_all_results_have_latency(self):
        """Every result must include a positive latency_ms field."""
        for r in self.results:
            self.assertIn("latency_ms", r)
            self.assertGreater(r["latency_ms"], 0)

    def test_security_mode_is_valid(self):
        """Security mode must be one of the valid hybrid modes."""
        valid_modes = {
            "FULL_HYBRID",
            "DEGRADED — QKD unavailable",
            "DEGRADED — PQC unavailable",
        }
        for r in self.results:
            self.assertIn(r["security_mode"], valid_modes,
                f"Device {r['device_id']} has unexpected mode: "
                f"{r['security_mode']}")


class TestScalabilityScenarios(unittest.TestCase):
    """Tests scalability under different network conditions."""

    def test_100_devices_normal_scenario(self):
        """100-device normal scenario — all should succeed."""
        results = run_network_exchange(100, scenario="normal")
        success = sum(1 for r in results if r["hybrid_key_ok"])
        self.assertGreaterEqual(success, 95)

    def test_100_devices_degraded_scenario(self):
        """100-device degraded scenario — majority succeed via failover."""
        results = run_network_exchange(100, scenario="degraded")
        success = sum(1 for r in results if r["hybrid_key_ok"])
        rate    = (success / 100) * 100
        self.assertGreaterEqual(rate, 80.0,
            f"Degraded scenario success rate {rate:.1f}% too low")

    def test_100_devices_attack_scenario(self):
        """Attack scenario — Kyber fallback keeps key production alive."""
        results = run_network_exchange(100, scenario="attack")
        has_key = sum(1 for r in results if r["hybrid_key_ok"])
        rate    = (has_key / 100) * 100
        self.assertGreaterEqual(rate, 70.0,
            f"Attack scenario key rate {rate:.1f}% too low")

    def test_200_devices_within_time_budget(self):
        """200-device exchange should finish within 60 seconds total."""
        start   = time.perf_counter()
        results = run_network_exchange(200, scenario="normal")
        elapsed = time.perf_counter() - start
        self.assertLess(elapsed, 60.0,
            f"200-device exchange took {elapsed:.1f}s (limit: 60s)")
        self.assertEqual(len(results), 200)


class TestFailoverAtScale(unittest.TestCase):
    """Tests graceful degradation and failover across many devices."""

    def test_kyber_fallback_when_qkd_fails(self):
        """When QKD key is None, Kyber fallback always produces a key."""
        combiner  = HybridKeyCombiner(HYBRID_METHOD)
        kem       = KyberKEM(PQC_ALGORITHM)
        successes = 0

        for _ in range(50):
            kem_r  = kem.full_key_exchange()
            result = combiner.combine(None, kem_r["shared_secret"])
            if result.get("hybrid_key") is not None:
                successes += 1

        self.assertEqual(successes, 50,
            "Kyber fallback must always produce a key when QKD fails")

    def test_fallback_mode_label_is_degraded(self):
        """Fallback mode label must clearly say DEGRADED."""
        combiner = HybridKeyCombiner(HYBRID_METHOD)
        kem_r    = KyberKEM(PQC_ALGORITHM).full_key_exchange()
        result   = combiner.combine(None, kem_r["shared_secret"])
        self.assertIn("DEGRADED", result["security_mode"])
        self.assertIn("PQC_ONLY", result["method"])

    def test_both_failed_returns_no_key(self):
        """When both QKD and Kyber fail, no key should be produced."""
        combiner = HybridKeyCombiner(HYBRID_METHOD)
        result   = combiner.combine(None, None)
        self.assertIsNone(result["hybrid_key"])
        self.assertEqual(result["security_mode"], "FAILED")

    def test_failover_across_100_devices_with_20pct_qkd_failures(self):
        """
        100 devices with 20% QKD failures — all still get a key
        via Kyber fallback. Proves defense-in-depth at scale.
        """
        combiner = HybridKeyCombiner(HYBRID_METHOD)
        kem      = KyberKEM(PQC_ALGORITHM)
        channel  = RANChannel(RAN_NOISE_NORMAL, RAN_LOSS_NORMAL)
        results  = []

        for i in range(100):
            kem_r     = kem.full_key_exchange()
            kyber_key = kem_r["shared_secret"]

            # 20% QKD failures
            if i % 5 == 0:
                qkd_key = None
            else:
                qkd_r   = BB84(QKD_NUM_QUBITS, channel).run()
                qkd_key = qkd_r.get("secret_key")

            hybrid = combiner.combine(qkd_key, kyber_key)
            results.append(hybrid.get("hybrid_key") is not None)

        self.assertTrue(all(results),
            "All 100 devices must get a key even with 20% QKD failures")


class TestNetworkStatsAtScale(unittest.TestCase):
    """Tests network statistics aggregation at scale."""

    def test_stats_100_devices_has_all_fields(self):
        """Stats should include all required fields for 100 devices."""
        network = IoTNetwork(num_devices=100, scenario="normal")
        stats   = network.get_network_stats()
        for field in ["num_devices", "avg_snr_db", "avg_ber",
                      "avg_delay_ms", "avg_loss",
                      "min_snr_db", "max_snr_db"]:
            self.assertIn(field, stats)

    def test_snr_values_are_physically_realistic(self):
        """Urban 5G SNR should be in realistic range (-10 to +60 dB)."""
        network = IoTNetwork(num_devices=100, scenario="normal")
        stats   = network.get_network_stats()
        self.assertGreater(stats["avg_snr_db"], -20.0)
        self.assertLess(stats["avg_snr_db"],     60.0)

    def test_degraded_has_higher_ber_than_normal(self):
        """
        Degraded scenario must have higher average BER than normal —
        this is guaranteed by the higher interference level in
        IoTNetwork (0.08 vs 0.02). SNR depends on random distances
        so cannot be compared directly between runs.
        """
        np.random.seed(42)
        normal   = IoTNetwork(num_devices=200, scenario="normal").get_network_stats()
        np.random.seed(42)
        degraded = IoTNetwork(num_devices=200, scenario="degraded").get_network_stats()

        self.assertGreater(degraded["avg_ber"], normal["avg_ber"],
            f"Degraded avg BER ({degraded['avg_ber']:.6f}) should be higher "
            f"than normal avg BER ({normal['avg_ber']:.6f}).")

    def test_stats_500_device_network(self):
        """Stats should aggregate correctly at 500 devices."""
        network = IoTNetwork(num_devices=500, scenario="normal")
        stats   = network.get_network_stats()
        self.assertEqual(stats["num_devices"], 500)
        self.assertIsInstance(stats["avg_snr_db"],   float)
        self.assertIsInstance(stats["avg_delay_ms"], float)


class TestQBERAtScale(unittest.TestCase):
    """Tests QBER behaviour across many devices."""

    def test_qber_below_threshold_in_normal_scenario(self):
        """Normal scenario QBER should stay below 11% on average."""
        results = run_network_exchange(50, scenario="normal")
        qbers   = [r["qber"] for r in results if r["qkd_status"] == "SUCCESS"]
        if qbers:
            avg_qber = np.mean(qbers)
            self.assertLess(avg_qber, 11.0,
                f"Average QBER {avg_qber:.2f}% exceeds safe threshold")

    def test_qber_values_non_negative(self):
        """QBER must never be negative."""
        results = run_network_exchange(50, scenario="normal")
        for r in results:
            self.assertGreaterEqual(r["qber"], 0.0,
                f"Device {r['device_id']} has negative QBER: {r['qber']}")


class TestEnergyAtScale(unittest.TestCase):
    """Tests energy budget across 100-device network."""

    def test_per_device_energy_within_10j_budget(self):
        """Each device's key exchange must use < 10 J (IoT target)."""
        from src.metrics.energy import EnergyEstimator
        estimator = EnergyEstimator(power_mw=25.0)
        results   = run_network_exchange(100, scenario="normal")

        for r in results:
            energy_j = estimator.from_latency_ms_j(r["latency_ms"])
            self.assertLess(energy_j, 10.0,
                f"Device {r['device_id']} uses {energy_j:.6f}J, "
                f"exceeds 10J target")

    def test_average_energy_well_under_1j(self):
        """Average energy per device should be well under 1 J."""
        from src.metrics.energy import EnergyEstimator
        estimator  = EnergyEstimator(power_mw=25.0)
        results    = run_network_exchange(100, scenario="normal")
        avg_lat    = np.mean([r["latency_ms"] for r in results])
        avg_energy = estimator.from_latency_ms_j(avg_lat)
        self.assertLess(avg_energy, 1.0,
            f"Average energy {avg_energy:.6f}J exceeds 1J")


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
