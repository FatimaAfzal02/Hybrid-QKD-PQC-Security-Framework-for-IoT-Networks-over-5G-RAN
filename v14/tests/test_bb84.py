# -*- coding: utf-8 -*-
"""
test_bb84.py — Unit Tests for BB84 QKD Protocol
================================================
Author  : FYP Team
Module  : tests/test_bb84.py
"""

import unittest
import sys, os
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.qkd.bb84 import BB84, RANChannel
from src.qkd.qber import QBERCalculator
from config import QKD_QBER_THRESHOLD


class TestRANChannel(unittest.TestCase):

    def setUp(self):
        self.channel = RANChannel(noise_level=0.05,
                                  packet_loss=0.02,
                                  delay_ms=10)
        self.bits    = list(np.random.randint(0, 2, 1000))

    def test_transmit_returns_correct_length(self):
        received = self.channel.transmit(self.bits)
        self.assertEqual(len(received), len(self.bits))

    def test_noise_introduces_errors(self):
        received = self.channel.transmit(self.bits)
        non_none = [r for r in received if r is not None]
        errors   = sum(1 for o, r in zip(self.bits, received)
                       if r is not None and o != r)
        self.assertGreater(errors, 0)

    def test_zero_noise_no_errors(self):
        ch       = RANChannel(noise_level=0.0, packet_loss=0.0)
        received = ch.transmit(self.bits)
        errors   = sum(1 for o, r in zip(self.bits, received)
                       if r is not None and o != r)
        self.assertEqual(errors, 0)

    def test_packet_loss_produces_nones(self):
        ch       = RANChannel(noise_level=0.0, packet_loss=0.5)
        received = ch.transmit(self.bits)
        nones    = received.count(None)
        self.assertGreater(nones, 0)

    def test_transmit_none_for_lost_bits(self):
        """Lost bits should be represented as None in received list."""
        ch = RANChannel(noise_level=0.0, packet_loss=1.0)
        received = ch.transmit(self.bits)
        self.assertTrue(all(r is None for r in received))


class TestBB84(unittest.TestCase):

    def setUp(self):
        self.channel = RANChannel(noise_level=0.03,
                                  packet_loss=0.01,
                                  delay_ms=8)

    def test_run_returns_required_keys(self):
        result = BB84(500, self.channel).run()
        required = ["qber", "secret_key_length",
                    "eavesdropper_detected", "status"]
        for key in required:
            self.assertIn(key, result)

    def test_no_eve_low_qber(self):
        """Without Eve, QBER should stay below threshold."""
        qbers = [BB84(1000, self.channel, False).run()["qber"]
                 for _ in range(10)]
        avg_qber = np.mean(qbers)
        self.assertLess(avg_qber, QKD_QBER_THRESHOLD * 100)

    def test_eve_raises_qber(self):
        """With Eve, QBER should exceed threshold on average."""
        qbers = [BB84(1000, self.channel, True).run()["qber"]
                 for _ in range(10)]
        avg_qber = np.mean(qbers)
        self.assertGreater(avg_qber, QKD_QBER_THRESHOLD * 100)

    def test_eve_detected(self):
        """Eve should be detected in most trials."""
        detected = sum(
            1 for _ in range(20)
            if BB84(1000, self.channel, True).run()["eavesdropper_detected"]
        )
        self.assertGreater(detected, 15)  # >75% detection rate

    def test_no_eve_not_detected(self):
        """Without Eve, false alarm rate should be low."""
        false_alarms = sum(
            1 for _ in range(20)
            if BB84(1000, self.channel, False).run()["eavesdropper_detected"]
        )
        self.assertLess(false_alarms, 5)  # <25% false alarm rate

    def test_secret_key_is_256_bits(self):
        """Final key should always be 256 bits after privacy amplification."""
        result = BB84(2000, self.channel, False).run()
        if result["status"] == "SUCCESS":
            self.assertEqual(result["secret_key_length"], 256)

    def test_secret_key_is_bytes(self):
        """Secret key should be bytes object."""
        result = BB84(2000, self.channel, False).run()
        if result["status"] == "SUCCESS":
            self.assertIsInstance(result["secret_key"], bytes)

    def test_more_qubits_more_sifted_bits(self):
        """More qubits should produce more sifted bits."""
        r_small = BB84(500,  self.channel, False).run()
        r_large = BB84(2000, self.channel, False).run()
        self.assertGreater(r_large["sifted_key_length"],
                           r_small["sifted_key_length"])

    def test_status_success_or_aborted(self):
        """Status should be SUCCESS or ABORTED only."""
        for _ in range(5):
            r = BB84(500, self.channel, False).run()
            self.assertIn(r["status"], ["SUCCESS",
                                         "ABORTED — eavesdropper detected",
                                         "No sifted bits — channel too noisy"])


class TestQBERCalculator(unittest.TestCase):

    def setUp(self):
        self.calc  = QBERCalculator()
        self.alice = list(np.random.randint(0, 2, 500))

    def test_perfect_match_zero_qber(self):
        result = self.calc.calculate(self.alice, self.alice.copy())
        self.assertEqual(result["qber"], 0.0)

    def test_all_errors_high_qber(self):
        bob    = [1 - b for b in self.alice]
        result = self.calc.calculate(self.alice, bob)
        self.assertGreater(result["qber"], 50.0)

    def test_security_level_secure(self):
        result = self.calc.calculate(self.alice, self.alice.copy())
        self.assertEqual(result["security_level"], "SECURE")

    def test_security_level_compromised(self):
        bob    = [1 - b for b in self.alice]
        result = self.calc.calculate(self.alice, bob)
        self.assertEqual(result["security_level"], "COMPROMISED")

    def test_empty_input(self):
        result = self.calc.calculate([], [])
        self.assertEqual(result["security_level"], "COMPROMISED")

    def test_history_tracked(self):
        for _ in range(5):
            self.calc.calculate(self.alice, self.alice.copy())
        history = self.calc.analyse_history()
        self.assertEqual(history["sessions"], 5)

    def test_theoretical_eve_qber(self):
        self.assertAlmostEqual(
            QBERCalculator.theoretical_eve_qber(), 0.25
        )

    def test_is_secure(self):
        self.assertTrue(QBERCalculator.is_secure(5.0))
        self.assertFalse(QBERCalculator.is_secure(15.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
