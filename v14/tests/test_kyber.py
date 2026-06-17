# -*- coding: utf-8 -*-
"""
test_kyber.py — Unit Tests for Kyber ML-KEM
============================================
Author  : FYP Team
Module  : tests/test_kyber.py
"""

import unittest
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.pqc.kyber import KyberKEM, KyberSimulation
from src.pqc.kem   import KeyEncapsulationMechanism, KEMKeyPair, KEMResult


class TestKyberSimulation(unittest.TestCase):

    def setUp(self):
        self.sim = KyberSimulation("Kyber768")

    def test_keypair_correct_sizes(self):
        pk, sk = self.sim.generate_keypair()
        self.assertEqual(len(pk), 1184)  # Kyber768 pk size
        self.assertEqual(len(sk), 2400)  # Kyber768 sk size

    def test_encapsulate_returns_bytes(self):
        pk, _ = self.sim.generate_keypair()
        ct, ss = self.sim.encapsulate(pk)
        self.assertIsInstance(ct, bytes)
        self.assertIsInstance(ss, bytes)

    def test_shared_secret_is_32_bytes(self):
        pk, _ = self.sim.generate_keypair()
        _, ss = self.sim.encapsulate(pk)
        self.assertEqual(len(ss), 32)  # 256 bits

    def test_ciphertext_correct_size(self):
        pk, _ = self.sim.generate_keypair()
        ct, _ = self.sim.encapsulate(pk)
        self.assertEqual(len(ct), 1088)  # Kyber768 ct size

    def test_kyber512_sizes(self):
        sim    = KyberSimulation("Kyber512")
        pk, sk = sim.generate_keypair()
        self.assertEqual(len(pk), 800)
        self.assertEqual(len(sk), 1632)

    def test_kyber1024_sizes(self):
        sim    = KyberSimulation("Kyber1024")
        pk, sk = sim.generate_keypair()
        self.assertEqual(len(pk), 1568)
        self.assertEqual(len(sk), 3168)


class TestKyberKEM(unittest.TestCase):

    def setUp(self):
        self.kem = KyberKEM("Kyber768")

    def test_full_exchange_returns_required_keys(self):
        result = self.kem.full_key_exchange()
        required = ["algorithm", "public_key_size", "shared_secret",
                    "total_time_ms", "status"]
        for key in required:
            self.assertIn(key, result)

    def test_status_success(self):
        result = self.kem.full_key_exchange()
        self.assertEqual(result["status"], "SUCCESS")

    def test_shared_secret_256_bits(self):
        result = self.kem.full_key_exchange()
        self.assertEqual(result["shared_secret_size"] * 8, 256)

    def test_shared_secret_is_bytes(self):
        result = self.kem.full_key_exchange()
        self.assertIsInstance(result["shared_secret"], bytes)

    def test_algorithm_recorded(self):
        result = self.kem.full_key_exchange()
        self.assertEqual(result["algorithm"], "Kyber768")

    def test_security_level_3(self):
        result = self.kem.full_key_exchange()
        self.assertEqual(result["security_level"], 3)

    def test_timing_is_positive(self):
        result = self.kem.full_key_exchange()
        self.assertGreater(result["total_time_ms"], 0)

    def test_timing_under_100ms(self):
        """Kyber should be very fast — well under 100ms."""
        result = self.kem.full_key_exchange()
        self.assertLess(result["total_time_ms"], 100)

    def test_different_runs_different_secrets(self):
        """Each key exchange should produce a unique shared secret."""
        r1 = KyberKEM("Kyber768").full_key_exchange()
        r2 = KyberKEM("Kyber768").full_key_exchange()
        self.assertNotEqual(r1["shared_secret"], r2["shared_secret"])

    def test_all_three_security_levels(self):
        for algo, level in [("Kyber512", 1),
                             ("Kyber768", 3),
                             ("Kyber1024", 5)]:
            kem    = KyberKEM(algo)
            result = kem.full_key_exchange()
            self.assertEqual(result["status"], "SUCCESS")


class TestKEMInterface(unittest.TestCase):

    def setUp(self):
        self.kem = KeyEncapsulationMechanism("Kyber768")

    def test_generate_keypair_returns_keypair(self):
        kp = self.kem.generate_keypair()
        self.assertIsInstance(kp, KEMKeyPair)

    def test_encapsulate_returns_kem_result(self):
        kp  = self.kem.generate_keypair()
        res = self.kem.encapsulate(kp.public_key)
        self.assertIsInstance(res, KEMResult)

    def test_shared_secret_is_32_bytes(self):
        kp  = self.kem.generate_keypair()
        res = self.kem.encapsulate(kp.public_key)
        self.assertEqual(len(res.shared_secret), 32)

    def test_full_exchange_success(self):
        result = self.kem.full_exchange()
        self.assertEqual(result["status"], "SUCCESS")

    def test_full_exchange_256_bit_secret(self):
        result = self.kem.full_exchange()
        self.assertEqual(result["secret_bits"], 256)

    def test_kem_result_shared_secret_bits(self):
        kp  = self.kem.generate_keypair()
        res = self.kem.encapsulate(kp.public_key)
        self.assertEqual(res.shared_secret_bits, 256)


if __name__ == "__main__":
    unittest.main(verbosity=2)
