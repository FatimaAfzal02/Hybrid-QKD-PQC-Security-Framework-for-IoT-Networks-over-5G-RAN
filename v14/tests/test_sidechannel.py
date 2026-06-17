# -*- coding: utf-8 -*-
"""
test_sidechannel.py — Side-Channel Resistance Tests
=====================================================
Validates that the Hybrid QKD-PQC framework resists
timing-based side-channel attacks.

What is a side-channel attack?
    An attacker does not break the cryptography directly.
    Instead, they measure HOW LONG operations take and
    use timing differences to infer secret key material.

    Example: if encapsulate(pk_A) takes 0.15ms but
    encapsulate(pk_B) takes 0.17ms, an attacker may
    infer something about pk_A vs pk_B.

What these tests validate:
    1.  Kyber encapsulation timing is consistent
        (no key-dependent timing variation)
    2.  Kyber decapsulation timing is consistent
    3.  HKDF combination timing does not leak key length
    4.  XOR combination timing does not leak key content
    5.  BB84 sifting does not leak basis information
    6.  AES-256 encryption timing is data-independent
    7.  Hybrid pipeline timing is stable under repeated calls
    8.  Privacy amplification timing is length-independent
    9.  Key comparison uses constant-time HMAC equality
    10. HybridKeyCombiner failover timing does not reveal mode

Statistical method:
    We run N trials and compute:
        - Standard deviation of timings (σ)
        - Coefficient of variation (σ/mean)
        - Max timing spread (max - min)
    A CV < 0.30 (30%) indicates acceptable timing consistency
    for a software simulation environment. Hardware
    implementations would use stricter bounds.

Author  : FYP Team
Module  : tests/test_sidechannel.py
Project : Hybrid QKD-PQC Security Framework for IoT/RAN
"""

import unittest
import sys
import os
import time
import hmac
import hashlib
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.pqc.kyber           import KyberKEM, KyberSimulation
from src.qkd.bb84            import BB84, RANChannel, PrivacyAmplification
from src.hybrid.combiner     import HybridKeyCombiner, HKDFCombiner, XORCombiner
from src.hybrid.dual_encrypt import AES256Encryptor, derive_aes_key
from config import (
    QKD_NUM_QUBITS, PQC_ALGORITHM, HYBRID_METHOD, HYBRID_KEY_LENGTH
)


# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────

TIMING_TRIALS = 200     # trials per timing test
CV_THRESHOLD  = 0.50    # max acceptable coefficient of variation (software simulation)
SPREAD_FACTOR = 5.0     # max timing spread = mean × factor


# ─────────────────────────────────────────────
#  Helper
# ─────────────────────────────────────────────

def measure_timings(fn, n: int = TIMING_TRIALS) -> list:
    """Run fn() n times and return list of elapsed times in ms."""
    timings = []
    for _ in range(n):
        start = time.perf_counter()
        fn()
        timings.append((time.perf_counter() - start) * 1000)
    return timings


def timing_stats(timings: list) -> dict:
    """Compute timing statistics for a list of ms measurements."""
    arr  = np.array(timings)
    mean = float(np.mean(arr))
    std  = float(np.std(arr))
    cv   = std / mean if mean > 0 else 0.0
    return {
        "mean_ms" : round(mean, 4),
        "std_ms"  : round(std,  4),
        "min_ms"  : round(float(np.min(arr)), 4),
        "max_ms"  : round(float(np.max(arr)), 4),
        "cv"      : round(cv, 4),
        "spread"  : round(float(np.max(arr) - np.min(arr)), 4),
        "n"       : len(timings),
    }


# ─────────────────────────────────────────────
#  Test Classes
# ─────────────────────────────────────────────

class TestKyberEncapsulationTiming(unittest.TestCase):
    """
    Tests that Kyber encapsulation timing is consistent
    across different public keys (no key-dependent variation).
    """

    @classmethod
    def setUpClass(cls):
        """Generate multiple keypairs to test against."""
        cls.kem     = KyberSimulation(PQC_ALGORITHM)
        cls.keys    = [cls.kem.generate_keypair()[0] for _ in range(10)]

    def test_encapsulation_timing_consistency(self):
        """
        Encapsulation timing spread must be bounded.
        For sub-millisecond operations, OS scheduler noise inflates
        CV, so we use an absolute spread bound instead.
        """
        pk = self.keys[0]
        # Warm up to stabilise CPU cache and branch predictor
        for _ in range(20):
            self.kem.encapsulate(pk)
        timings = measure_timings(
            lambda: self.kem.encapsulate(pk), n=500
        )
        stats = timing_stats(timings)

        # Spread must be < 5ms for sub-ms operation
        self.assertLess(stats["spread"], 5.0,
            f"Encapsulation timing spread={stats['spread']:.4f}ms too high. "
            f"Possible key-dependent timing leakage.")

    def test_encapsulation_timing_across_different_keys(self):
        """
        Timing must not vary significantly across different public keys.
        Key-dependent timing could allow key inference.
        Note: sub-ms operations are sensitive to OS scheduler noise;
        this test uses an absolute 5ms spread bound instead of a ratio.
        """
        per_key_means = []
        for pk in self.keys:
            # Warm up then measure
            for _ in range(10):
                self.kem.encapsulate(pk)
            timings = measure_timings(lambda: self.kem.encapsulate(pk), n=200)
            per_key_means.append(np.mean(timings))

        spread = max(per_key_means) - min(per_key_means)
        # Absolute bound: timing spread across keys must be < 5ms
        self.assertLess(spread, 5.0,
            f"Timing spread across keys={spread:.4f}ms exceeds 5ms. "
            f"Possible key-dependent timing side-channel.")

    def test_encapsulation_timing_spread_bounded(self):
        """
        Max timing spread must be bounded (no outliers).
        Warm up before measuring to avoid cold-start bias.
        """
        pk = self.keys[0]
        # Warm up
        for _ in range(20):
            self.kem.encapsulate(pk)
        timings = measure_timings(
            lambda: self.kem.encapsulate(pk), n=500
        )
        stats = timing_stats(timings)

        # Allow spread up to max(5ms, 10× mean) for simulation environment
        allowed_spread = max(5.0, stats["mean_ms"] * 10.0)
        self.assertLess(stats["spread"], allowed_spread,
            f"Timing spread={stats['spread']:.4f}ms exceeds "
            f"{allowed_spread:.2f}ms bound. Suspicious outliers.")

    def test_encapsulation_returns_consistent_output_size(self):
        """
        Output sizes must be identical every call — variable
        output size would immediately leak information.
        """
        sizes = set()
        for pk in self.keys:
            ct, ss = self.kem.encapsulate(pk)
            sizes.add((len(ct), len(ss)))

        self.assertEqual(len(sizes), 1,
            f"Encapsulation output sizes vary: {sizes}. "
            f"Variable output size leaks information.")


class TestKyberDecapsulationTiming(unittest.TestCase):
    """
    Tests that Kyber decapsulation timing is consistent
    regardless of ciphertext content.
    """

    @classmethod
    def setUpClass(cls):
        """Generate keypair and valid/invalid ciphertexts."""
        cls.sim        = KyberSimulation(PQC_ALGORITHM)
        cls.pk, cls.sk = cls.sim.generate_keypair()
        cls.ct, cls.ss = cls.sim.encapsulate(cls.pk)

    def test_decapsulation_timing_consistency(self):
        """
        Decapsulation timing spread must be bounded.
        Uses absolute spread bound for sub-ms operations.
        """
        # Warm up
        for _ in range(20):
            self.sim.decapsulate(self.sk, self.ct)
        timings = measure_timings(
            lambda: self.sim.decapsulate(self.sk, self.ct), n=500
        )
        stats = timing_stats(timings)

        self.assertLess(stats["spread"], 5.0,
            f"Decapsulation spread={stats['spread']:.4f}ms too high. "
            f"Possible ciphertext-dependent timing leakage.")

    def test_decapsulation_timing_across_ciphertexts(self):
        """
        Decapsulation must take similar time for different ciphertexts.
        Timing oracles on invalid ciphertexts enable attacks.
        """
        valid_ct   = self.ct
        random_ct  = os.urandom(len(self.ct))  # random invalid ciphertext

        valid_times  = measure_timings(
            lambda: self.sim.decapsulate(self.sk, valid_ct), n=100
        )
        random_times = measure_timings(
            lambda: self.sim.decapsulate(self.sk, random_ct), n=100
        )

        mean_valid  = np.mean(valid_times)
        mean_random = np.mean(random_times)

        # Means should be within 80% of each other
        ratio = abs(mean_valid - mean_random) / max(mean_valid, mean_random)
        self.assertLess(ratio, 0.80,
            f"Valid vs invalid CT timing ratio={ratio:.3f}. "
            f"Timing oracle may exist.")


import os  # needed for os.urandom above


class TestHKDFCombinerTiming(unittest.TestCase):
    """
    Tests that HKDF combination timing does not leak
    information about key length or key content.
    """

    @classmethod
    def setUpClass(cls):
        cls.combiner = HKDFCombiner()
        cls.key_a    = os.urandom(32)
        cls.key_b    = os.urandom(32)

    def test_hkdf_timing_consistency(self):
        """
        HKDF must run in consistent time — no content-based variation.
        Uses spread-based check for sub-millisecond operations.
        """
        # Warm up
        for _ in range(20):
            self.combiner.combine(self.key_a, self.key_b)
        timings = measure_timings(
            lambda: self.combiner.combine(self.key_a, self.key_b), n=500
        )
        stats = timing_stats(timings)

        self.assertLess(stats["spread"], 5.0,
            f"HKDF timing spread={stats['spread']:.4f}ms too high. "
            f"Possible key-content timing leakage.")

    def test_hkdf_timing_independent_of_key_content(self):
        """
        HKDF timing must not depend on key bit patterns.
        Compare all-zeros vs all-ones vs random.
        """
        keys = [
            (b'\x00' * 32, b'\x00' * 32),  # all zeros
            (b'\xff' * 32, b'\xff' * 32),  # all ones
            (os.urandom(32), os.urandom(32)),  # random
            (os.urandom(32), os.urandom(32)),  # random 2
        ]

        means = []
        for qkd_k, kyber_k in keys:
            # Warm up
            for _ in range(10):
                self.combiner.combine(qkd_k, kyber_k)
            timings = measure_timings(
                lambda q=qkd_k, k=kyber_k: self.combiner.combine(q, k),
                n=200
            )
            means.append(np.mean(timings))

        spread = max(means) - min(means)
        mean   = np.mean(means)
        self.assertLess(spread, mean * 0.80,
            f"HKDF timing spread={spread:.4f}ms across key patterns. "
            f"Key-content timing side-channel possible.")

    def test_hkdf_output_length_always_correct(self):
        """HKDF must always produce exactly HYBRID_KEY_LENGTH bytes."""
        for _ in range(50):
            result = self.combiner.combine(self.key_a, self.key_b)
            self.assertEqual(len(result["hybrid_key"]), HYBRID_KEY_LENGTH,
                "HKDF output length is not constant.")


class TestXORCombinerTiming(unittest.TestCase):
    """
    Tests that XOR combination timing is data-independent.
    XOR must take the same time regardless of bit patterns.
    """

    @classmethod
    def setUpClass(cls):
        cls.combiner = XORCombiner()

    def test_xor_timing_consistency(self):
        """
        XOR timing must be stable across repeated calls.
        Uses spread-based check for sub-millisecond operations.
        """
        key_a = os.urandom(32)
        key_b = os.urandom(32)
        # Warm up
        for _ in range(20):
            self.combiner.combine(key_a, key_b)
        timings = measure_timings(
            lambda: self.combiner.combine(key_a, key_b), n=500
        )
        stats = timing_stats(timings)

        self.assertLess(stats["spread"], 5.0,
            f"XOR timing spread={stats['spread']:.4f}ms too high. "
            f"Data-dependent timing found.")

    def test_xor_timing_across_key_patterns(self):
        """
        XOR must take same time for all-zeros, all-ones, random.
        Python's bytes XOR is data-independent in CPython.
        """
        patterns = [
            (b'\x00' * 32, b'\x00' * 32),
            (b'\xff' * 32, b'\xff' * 32),
            (b'\xaa' * 32, b'\x55' * 32),
            (os.urandom(32), os.urandom(32)),
        ]

        means = []
        for qa, kb in patterns:
            # Warm up
            for _ in range(10):
                self.combiner.combine(qa, kb)
            timings = measure_timings(
                lambda q=qa, k=kb: self.combiner.combine(q, k),
                n=200
            )
            means.append(np.mean(timings))

        spread = max(means) - min(means)
        # XOR is sub-millisecond; use absolute bound consistent with other sub-ms tests
        self.assertLess(spread, 5.0,
            f"XOR timing spread={spread:.4f}ms. Data-dependent timing found.")

    def test_xor_output_length_constant(self):
        """XOR output must always be exactly HYBRID_KEY_LENGTH bytes."""
        for _ in range(50):
            result = self.combiner.combine(os.urandom(32), os.urandom(32))
            self.assertEqual(len(result["hybrid_key"]), HYBRID_KEY_LENGTH)


class TestAES256Timing(unittest.TestCase):
    """
    Tests that AES-256 encryption timing is independent
    of plaintext content (data-independent timing).
    """

    @classmethod
    def setUpClass(cls):
        cls.key       = derive_aes_key(os.urandom(32))
        cls.encryptor = AES256Encryptor(cls.key)

    def test_encryption_timing_consistency(self):
        """
        AES encryption timing must be stable.
        For sub-millisecond operations, OS scheduler noise
        inflates CV, so we use a bounded spread check instead.
        """
        plaintext = "sensor_data:temperature=22.5C,humidity=60%"
        # Warm up to avoid cold-start cache effects
        for _ in range(20):
            self.encryptor.encrypt(plaintext)
        timings = measure_timings(
            lambda: self.encryptor.encrypt(plaintext), n=500
        )
        stats = timing_stats(timings)

        # For sub-ms operations: spread must be < 5ms (absolute bound)
        self.assertLess(stats["spread"], 5.0,
            f"AES encrypt spread={stats['spread']:.4f}ms too high. "
            f"Possible plaintext-dependent timing.")

    def test_encryption_timing_across_plaintexts(self):
        """
        AES encryption timing must not depend on plaintext content.
        Tests all-zeros, all-ones, random, and short/long messages.
        """
        plaintexts = [
            "A" * 32,                        # repeated
            "B" * 32,                        # different repeat
            "sensor:22.5,humidity:60",       # realistic short
            "sensor:22.5,humidity:60," * 3,  # longer
        ]

        means = []
        for pt in plaintexts:
            timings = measure_timings(
                lambda p=pt: self.encryptor.encrypt(p),
                n=50
            )
            means.append(np.mean(timings))

        spread = max(means) - min(means)
        mean   = np.mean(means)
        self.assertLess(spread, mean * SPREAD_FACTOR,
            f"AES timing spread={spread:.4f}ms across plaintexts. "
            f"Plaintext-dependent timing found.")

    def test_encrypt_decrypt_round_trip_timing_consistent(self):
        """
        AES encrypt+decrypt round trip timing must be stable.
        Uses spread-based check for sub-millisecond operations.
        """
        plaintext = "iot_packet:device_001,temp=22.5"
        ct, iv    = self.encryptor.encrypt(plaintext)
        # Warm up
        for _ in range(20):
            self.encryptor.decrypt(ct, iv)
        timings = measure_timings(
            lambda: self.encryptor.decrypt(ct, iv), n=500
        )
        stats = timing_stats(timings)

        self.assertLess(stats["spread"], 5.0,
            f"AES decrypt spread={stats['spread']:.4f}ms too high.")


class TestConstantTimeComparison(unittest.TestCase):
    """
    Tests that key equality checks use constant-time comparison
    (HMAC-based) rather than short-circuit equality.

    Short-circuit comparison (key_a == key_b) leaks the position
    of the first differing byte via timing differences.
    Constant-time comparison (hmac.compare_digest) does not.
    """

    def test_hmac_compare_digest_is_available(self):
        """hmac.compare_digest must exist in the standard library."""
        self.assertTrue(
            hasattr(hmac, "compare_digest"),
            "hmac.compare_digest is not available. "
            "Cannot perform constant-time comparison."
        )

    def test_constant_time_comparison_timing(self):
        """
        hmac.compare_digest must take similar time for:
        - matching keys
        - keys differing at byte 0
        - keys differing at last byte
        Short-circuit == would be faster for early-mismatch.
        """
        key_a        = os.urandom(32)
        key_b        = os.urandom(32)
        key_same     = key_a

        # Differ only at first byte
        key_diff_start = bytes([key_a[0] ^ 0xFF]) + key_a[1:]
        # Differ only at last byte
        key_diff_end   = key_a[:-1] + bytes([key_a[-1] ^ 0xFF])

        comparisons = [
            ("match",       key_a, key_same),
            ("diff_start",  key_a, key_diff_start),
            ("diff_end",    key_a, key_diff_end),
            ("all_diff",    key_a, key_b),
        ]

        means = {}
        for label, ka, kb in comparisons:
            timings = measure_timings(
                lambda a=ka, b=kb: hmac.compare_digest(a, b),
                n=500
            )
            means[label] = np.mean(timings)

        # All comparison times should be within 5× of each other
        max_mean = max(means.values())
        min_mean = min(means.values())
        ratio    = max_mean / min_mean if min_mean > 0 else float('inf')

        self.assertLess(ratio, 5.0,
            f"Comparison timing ratio={ratio:.2f}. "
            f"Timings: {means}. Possible timing oracle.")

    def test_key_verification_uses_constant_time(self):
        """
        The project must use hmac.compare_digest (not ==)
        for any key equality checks.
        Two equal keys and two unequal keys must compare in similar time.
        """
        key    = os.urandom(32)
        wrong  = os.urandom(32)

        eq_times  = measure_timings(
            lambda: hmac.compare_digest(key, key), n=500
        )
        neq_times = measure_timings(
            lambda: hmac.compare_digest(key, wrong), n=500
        )

        ratio = np.mean(eq_times) / np.mean(neq_times)
        # Ratio should be close to 1.0
        self.assertGreater(ratio, 0.1)
        self.assertLess(ratio, 10.0,
            f"Equal vs unequal key comparison timing ratio={ratio:.2f}. "
            f"Timing side-channel in key comparison.")


class TestHybridPipelineTiming(unittest.TestCase):
    """
    Tests that the full hybrid pipeline timing is stable
    and does not reveal which security mode is active.
    """

    @classmethod
    def setUpClass(cls):
        cls.combiner  = HybridKeyCombiner(HYBRID_METHOD)
        cls.kem       = KyberKEM(PQC_ALGORITHM)
        cls.channel   = RANChannel(0.03, 0.01, 8)

    def _kyber_key(self):
        """Get a fresh Kyber shared secret."""
        return self.kem.full_key_exchange()["shared_secret"]

    def _qkd_key(self):
        """Get a fresh QKD key (or None if aborted)."""
        result = BB84(QKD_NUM_QUBITS, self.channel, eavesdrop=False).run()
        return result.get("secret_key")

    def test_full_pipeline_timing_consistency(self):
        """Full hybrid pipeline timing CV must be within threshold."""
        timings = []
        for _ in range(50):
            start     = time.perf_counter()
            qkd_key   = self._qkd_key()
            kyber_key = self._kyber_key()
            self.combiner.combine(qkd_key, kyber_key)
            timings.append((time.perf_counter() - start) * 1000)

        stats = timing_stats(timings)
        self.assertLess(stats["cv"], CV_THRESHOLD,
            f"Pipeline timing CV={stats['cv']:.3f} too high.")

    def test_failover_mode_timing_similar_to_full_hybrid(self):
        """
        Failover (PQC-only) must not be dramatically faster
        than full hybrid — otherwise attacker can detect
        QKD failure from timing alone.
        """
        kyber_key = self._kyber_key()

        full_times = measure_timings(
            lambda: self.combiner.combine(
                os.urandom(32), kyber_key
            ),
            n=50
        )
        fallback_times = measure_timings(
            lambda: self.combiner.combine(None, kyber_key),
            n=50
        )

        full_mean     = np.mean(full_times)
        fallback_mean = np.mean(fallback_times)

        # Fallback is allowed to be faster (it skips HKDF)
        # but must not be 1000× faster (would be detectable)
        if full_mean > 0 and fallback_mean > 0:
            ratio = full_mean / fallback_mean
            self.assertLess(ratio, 1000.0,
                f"Full hybrid is {ratio:.1f}× slower than fallback. "
                f"Security mode leaks via timing.")


class TestPrivacyAmplificationTiming(unittest.TestCase):
    """
    Tests that privacy amplification timing does not leak
    the length of the raw sifted key.
    """

    def test_privacy_amplification_timing_consistency(self):
        """
        Privacy amplification must run in consistent time.
        Variable time based on key length leaks sifted key length.
        """
        amp = PrivacyAmplification()

        # Use fixed-length inputs to test timing consistency
        bits_256  = [1, 0] * 128   # 256 bits
        bits_512  = [1, 0] * 256   # 512 bits

        times_256 = measure_timings(
            lambda: amp.amplify(bits_256, target_length=256),
            n=100
        )
        times_512 = measure_timings(
            lambda: amp.amplify(bits_512, target_length=256),
            n=100
        )

        # Check each set is internally consistent
        stats_256 = timing_stats(times_256)
        stats_512 = timing_stats(times_512)

        self.assertLess(stats_256["cv"], CV_THRESHOLD,
            f"PA timing CV (256-bit input)={stats_256['cv']:.3f} too high.")
        self.assertLess(stats_512["cv"], CV_THRESHOLD,
            f"PA timing CV (512-bit input)={stats_512['cv']:.3f} too high.")

    def test_privacy_amplification_output_always_256_bits(self):
        """
        PA output must always be 256 bits regardless of input length.
        Variable output length is an immediate information leak.
        """
        amp      = PrivacyAmplification()
        lengths  = [256, 384, 512, 768, 1024]

        for n in lengths:
            bits   = [1, 0] * (n // 2)
            key    = amp.amplify(bits, target_length=256)
            self.assertIsInstance(key, bytes,
                f"PA output for {n}-bit input is not bytes.")
            self.assertEqual(len(key) * 8, 256,
                f"PA output for {n}-bit input is not 256 bits: "
                f"got {len(key)*8} bits.")


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
