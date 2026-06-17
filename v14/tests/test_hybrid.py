# -*- coding: utf-8 -*-
"""
test_hybrid.py — Unit Tests for Hybrid Key Combination & Encryption
====================================================================
Author  : FYP Team
Module  : tests/test_hybrid.py
"""

import unittest
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.qkd.bb84            import BB84, RANChannel
from src.pqc.kyber           import KyberKEM
from src.hybrid.combiner     import HybridKeyCombiner, HKDFCombiner, XORCombiner
from src.hybrid.dual_encrypt import (
    AES256Encryptor, DualAES256Encryptor,
    IoTPacketEncryptor, derive_aes_key
)
from config import HYBRID_KEY_LENGTH


# ── Shared fixtures ───────────────────────────
def get_keys():
    channel    = RANChannel(0.03, 0.01, 8)
    qkd_result = BB84(2000, channel, eavesdrop=False).run()
    kem_result = KyberKEM("Kyber768").full_key_exchange()
    return qkd_result, kem_result


class TestHKDFCombiner(unittest.TestCase):

    def setUp(self):
        self.qkd_r, self.kem_r = get_keys()
        self.combiner = HKDFCombiner()

    def test_output_is_32_bytes(self):
        result = self.combiner.combine(
            self.qkd_r["secret_key"],
            self.kem_r["shared_secret"]
        )
        self.assertEqual(len(result["hybrid_key"]), 32)

    def test_status_success(self):
        result = self.combiner.combine(
            self.qkd_r["secret_key"],
            self.kem_r["shared_secret"]
        )
        self.assertEqual(result["status"], "SUCCESS")

    def test_method_is_hkdf(self):
        result = self.combiner.combine(
            self.qkd_r["secret_key"],
            self.kem_r["shared_secret"]
        )
        self.assertEqual(result["method"], "HKDF")

    def test_different_inputs_different_output(self):
        r1 = self.combiner.combine(
            self.qkd_r["secret_key"],
            self.kem_r["shared_secret"]
        )
        qkd2, kem2 = get_keys()
        r2 = self.combiner.combine(
            qkd2["secret_key"],
            kem2["shared_secret"]
        )
        self.assertNotEqual(r1["hybrid_key"], r2["hybrid_key"])

    def test_timing_recorded(self):
        result = self.combiner.combine(
            self.qkd_r["secret_key"],
            self.kem_r["shared_secret"]
        )
        self.assertIn("timing_ms", result)
        self.assertGreater(result["timing_ms"], 0)


class TestXORCombiner(unittest.TestCase):

    def setUp(self):
        self.qkd_r, self.kem_r = get_keys()
        self.combiner = XORCombiner()

    def test_output_is_32_bytes(self):
        result = self.combiner.combine(
            self.qkd_r["secret_key"],
            self.kem_r["shared_secret"]
        )
        self.assertEqual(len(result["hybrid_key"]), 32)

    def test_method_is_xor(self):
        result = self.combiner.combine(
            self.qkd_r["secret_key"],
            self.kem_r["shared_secret"]
        )
        self.assertEqual(result["method"], "XOR")


class TestHybridKeyCombiner(unittest.TestCase):

    def setUp(self):
        self.qkd_r, self.kem_r = get_keys()
        self.combiner = HybridKeyCombiner("HKDF")

    def test_full_hybrid_mode(self):
        result = self.combiner.combine(
            self.qkd_r["secret_key"],
            self.kem_r["shared_secret"]
        )
        self.assertEqual(result["security_mode"], "FULL_HYBRID")

    def test_qkd_failed_fallback(self):
        result = self.combiner.combine(None, self.kem_r["shared_secret"])
        self.assertIn("DEGRADED", result["security_mode"])
        self.assertIsNotNone(result["hybrid_key"])

    def test_kyber_failed_fallback(self):
        result = self.combiner.combine(self.qkd_r["secret_key"], None)
        self.assertIn("DEGRADED", result["security_mode"])
        self.assertIsNotNone(result["hybrid_key"])

    def test_both_failed(self):
        result = self.combiner.combine(None, None)
        self.assertEqual(result["security_mode"], "FAILED")
        self.assertIsNone(result["hybrid_key"])

    def test_hybrid_key_is_correct_length(self):
        result = self.combiner.combine(
            self.qkd_r["secret_key"],
            self.kem_r["shared_secret"]
        )
        self.assertEqual(result["key_length"], HYBRID_KEY_LENGTH)


class TestAES256Encryptor(unittest.TestCase):

    def setUp(self):
        self.key = os.urandom(32)
        self.enc = AES256Encryptor(self.key)

    def test_encrypt_decrypt_roundtrip(self):
        msg = "Test IoT message 123"
        ct, iv = self.enc.encrypt(msg)
        pt = self.enc.decrypt(ct, iv)
        self.assertEqual(msg, pt)

    def test_encrypted_is_bytes(self):
        ct, _ = self.enc.encrypt("hello")
        self.assertIsInstance(ct, bytes)

    def test_different_messages_different_ciphertext(self):
        ct1, _ = self.enc.encrypt("message one")
        ct2, _ = self.enc.encrypt("message two")
        self.assertNotEqual(ct1, ct2)

    def test_same_message_different_iv(self):
        """Same message should produce different ciphertext each time (random IV)."""
        ct1, iv1 = self.enc.encrypt("same message")
        ct2, iv2 = self.enc.encrypt("same message")
        self.assertNotEqual(iv1, iv2)


class TestDualAES256(unittest.TestCase):

    def setUp(self):
        self.qkd_r, self.kem_r = get_keys()
        self.enc = DualAES256Encryptor(
            self.qkd_r["secret_key"],
            self.kem_r["shared_secret"]
        )

    def test_dual_encrypt_decrypt(self):
        msg = "HOSPITAL:003 | BP:120/80 | CRITICAL:NO"
        ct, iv1, iv2 = self.enc.encrypt(msg)
        pt = self.enc.decrypt(ct, iv1, iv2)
        self.assertEqual(msg, pt)

    def test_ciphertext_is_bytes(self):
        ct, _, _ = self.enc.encrypt("test")
        self.assertIsInstance(ct, bytes)


class TestIoTPacketEncryptor(unittest.TestCase):

    def setUp(self):
        self.qkd_r, self.kem_r = get_keys()
        combiner     = HybridKeyCombiner("HKDF")
        hybrid       = combiner.combine(
            self.qkd_r["secret_key"],
            self.kem_r["shared_secret"]
        )
        self.enc_single = IoTPacketEncryptor(
            hybrid["hybrid_key"], mode="SINGLE"
        )
        self.enc_dual = IoTPacketEncryptor(
            hybrid["hybrid_key"],
            qkd_key   = self.qkd_r["secret_key"],
            kyber_key = self.kem_r["shared_secret"],
            mode      = "DUAL"
        )

    def _kwargs(self, device_type):
        templates = {
            "traffic_sensor": dict(id="001", lat="31.52",
                                   lon="74.35", val=42, ts=123),
            "smart_meter"   : dict(id="089", val="423", ts=123),
            "camera"        : dict(id="014", alert="MOTION",
                                   conf=94, ts=123),
            "hospital"      : dict(id="003", bp="120/80",
                                   hr=72, crit="NO", ts=123),
        }
        return templates[device_type]

    def test_single_encrypt_decrypt_all_devices(self):
        for dtype in ["traffic_sensor","smart_meter","camera","hospital"]:
            r  = self.enc_single.encrypt_packet(dtype, **self._kwargs(dtype))
            pt = self.enc_single.decrypt_packet(r)
            self.assertEqual(pt, r["original"],
                             f"Failed for {dtype}")

    def test_dual_encrypt_decrypt_all_devices(self):
        for dtype in ["traffic_sensor","smart_meter","camera","hospital"]:
            r  = self.enc_dual.encrypt_packet(dtype, **self._kwargs(dtype))
            pt = self.enc_dual.decrypt_packet(r)
            self.assertEqual(pt, r["original"],
                             f"Failed for {dtype}")

    def test_result_has_required_fields(self):
        r = self.enc_single.encrypt_packet(
            "hospital", **self._kwargs("hospital")
        )
        for key in ["device_type", "plaintext_len",
                    "ciphertext_len", "timing_ms"]:
            self.assertIn(key, r)

    def test_timing_is_positive(self):
        r = self.enc_dual.encrypt_packet(
            "hospital", **self._kwargs("hospital")
        )
        self.assertGreater(r["timing_ms"], 0)


class TestDeriveAESKey(unittest.TestCase):

    def test_derive_from_bytes(self):
        key = derive_aes_key(os.urandom(32))
        self.assertEqual(len(key), 32)

    def test_derive_from_bit_list(self):
        bits = [1, 0, 1, 1, 0, 0, 1, 0] * 32
        key  = derive_aes_key(bits)
        self.assertEqual(len(key), 32)

    def test_same_input_same_key(self):
        raw  = b"test key material"
        key1 = derive_aes_key(raw)
        key2 = derive_aes_key(raw)
        self.assertEqual(key1, key2)

    def test_different_input_different_key(self):
        key1 = derive_aes_key(b"input one")
        key2 = derive_aes_key(b"input two")
        self.assertNotEqual(key1, key2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
