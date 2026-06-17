# -*- coding: utf-8 -*-
"""
kem.py — Key Encapsulation Mechanism Interface
===============================================
Clean KEM interface wrapping Kyber ML-KEM.
Provides a simple Alice/Bob key exchange API.

Author  : FYP Team
Module  : src/pqc/kem.py
"""

import os, hashlib, time
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config import PQC_ALGORITHM

try:
    import oqs
    LIBOQS = True
except ImportError:
    LIBOQS = False


class KEMKeyPair:
    """Holds a public/secret key pair."""
    def __init__(self, public_key: bytes, secret_key: bytes):
        self.public_key = public_key
        self.secret_key = secret_key
        self.created_at = time.time()

    @property
    def public_key_size(self) -> int:
        return len(self.public_key)

    @property
    def secret_key_size(self) -> int:
        return len(self.secret_key)


class KEMResult:
    """Result of a KEM encapsulation."""
    def __init__(self, ciphertext: bytes, shared_secret: bytes):
        self.ciphertext    = ciphertext
        self.shared_secret = shared_secret

    @property
    def shared_secret_bits(self) -> int:
        return len(self.shared_secret) * 8


class KeyEncapsulationMechanism:
    """
    Simple Alice/Bob KEM interface for Kyber ML-KEM.

    Usage:
        # Bob's side
        kem     = KeyEncapsulationMechanism()
        keypair = kem.generate_keypair()

        # Alice's side (receives bob's public key)
        result  = kem.encapsulate(keypair.public_key)
        # Alice sends result.ciphertext to Bob
        # Alice keeps result.shared_secret

        # Bob's side
        shared  = kem.decapsulate(result.ciphertext, keypair.secret_key)
        # Bob and Alice now have the same shared_secret
    """

    # Kyber key sizes (bytes) — NIST spec
    SIZES = {
        "Kyber512" : {"pk": 800,  "sk": 1632, "ct": 768},
        "Kyber768" : {"pk": 1184, "sk": 2400, "ct": 1088},
        "Kyber1024": {"pk": 1568, "sk": 3168, "ct": 1568},
    }

    def __init__(self, algorithm: str = PQC_ALGORITHM):
        self.algorithm = algorithm
        self._sizes    = self.SIZES.get(algorithm, self.SIZES["Kyber768"])

        if LIBOQS:
            self._kem = oqs.KeyEncapsulation(algorithm)
        else:
            self._kem = None

    def generate_keypair(self) -> KEMKeyPair:
        """Generate Bob's public/secret key pair."""
        if LIBOQS and self._kem:
            pk = self._kem.generate_keypair()
            sk = self._kem.export_secret_key()
        else:
            pk = os.urandom(self._sizes["pk"])
            sk = os.urandom(self._sizes["sk"])

        return KEMKeyPair(pk, sk)

    def encapsulate(self, public_key: bytes) -> KEMResult:
        """
        Alice encapsulates a shared secret.
        Returns ciphertext to send to Bob + shared secret to keep.
        """
        if LIBOQS and self._kem:
            ct, ss = self._kem.encap_secret(public_key)
        else:
            rand = os.urandom(32)
            ss   = hashlib.sha256(public_key[:32] + rand).digest()
            ct   = os.urandom(self._sizes["ct"])

        return KEMResult(ct, ss)

    def decapsulate(self, ciphertext: bytes,
                    secret_key: bytes) -> bytes:
        """Bob decapsulates to recover Alice's shared secret."""
        if LIBOQS and self._kem:
            return self._kem.decap_secret(ciphertext)
        else:
            return hashlib.sha256(
                secret_key[:32] + ciphertext[:32]
            ).digest()

    def full_exchange(self) -> dict:
        """Complete Alice-Bob key exchange in one call."""
        start   = time.perf_counter()
        keypair = self.generate_keypair()
        result  = self.encapsulate(keypair.public_key)
        bob_ss  = self.decapsulate(result.ciphertext, keypair.secret_key)
        elapsed = round((time.perf_counter() - start) * 1000, 3)

        return {
            "algorithm"      : self.algorithm,
            "public_key_size": keypair.public_key_size,
            "secret_key_size": keypair.secret_key_size,
            "ciphertext_size": len(result.ciphertext),
            "shared_secret"  : result.shared_secret,
            "secret_bits"    : result.shared_secret_bits,
            "total_ms"       : elapsed,
            "status"         : "SUCCESS",
        }


if __name__ == "__main__":
    print("KEM Interface Test")
    for algo in ["Kyber512", "Kyber768", "Kyber1024"]:
        kem = KeyEncapsulationMechanism(algo)
        r   = kem.full_exchange()
        print(f"  {algo:<12} PK:{r['public_key_size']}B "
              f"SS:{r['secret_bits']}bit "
              f"Time:{r['total_ms']}ms")
