# -*- coding: utf-8 -*-
"""
dual_encrypt.py — Hybrid AES-256 Encryption Layer
===================================================
Encrypts IoT data using the hybrid key from
QKD + Kyber combination.

Encryption modes:
    SINGLE : one AES-256 pass with hybrid key
    DUAL   : two AES-256 passes (QKD key then Kyber key)
             attacker must break BOTH keys to decrypt

Also handles:
    - Key derivation from raw bits
    - IV generation and management
    - Encryption timing measurement
    - IoT packet formatting

Author  : FYP Team
Module  : src/hybrid/dual_encrypt.py
Project : Hybrid QKD-PQC Security Framework for IoT/RAN
"""

import os
import hashlib
import time
import numpy as np
from typing import Optional, Tuple
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config import HYBRID_KEY_LENGTH


# ─────────────────────────────────────────────
#  Key Derivation
# ─────────────────────────────────────────────

def derive_aes_key(raw_key) -> bytes:
    """
    Derive a 256-bit AES key from raw input.

    Accepts:
        - bytes  : direct key material
        - list   : bit list from BB84

    Uses SHA-256 to ensure exactly 32 bytes output.
    """
    if isinstance(raw_key, list):
        # Convert bit list to bytes
        bits   = raw_key + [0] * ((8 - len(raw_key) % 8) % 8)
        ba     = bytearray()
        for i in range(0, len(bits), 8):
            byte = 0
            for bit in bits[i:i+8]:
                byte = (byte << 1) | int(bit)
            ba.append(byte)
        raw_key = bytes(ba)

    return hashlib.sha256(raw_key).digest()  # 32 bytes = 256 bits


# ─────────────────────────────────────────────
#  Single Layer AES-256
# ─────────────────────────────────────────────

class AES256Encryptor:
    """
    Single-layer AES-256 encryption.

    Uses the hybrid key (from QKD + Kyber combination)
    to encrypt IoT data with AES-256-CBC.

    Args:
        key : 32-byte AES key (from hybrid combiner)
    """

    def __init__(self, key: bytes):
        if len(key) != 32:
            key = hashlib.sha256(key).digest()
        self.key  = key
        self.mode = "SINGLE"

    def encrypt(self, plaintext: str) -> Tuple[bytes, bytes]:
        """
        Encrypt plaintext string.

        Returns:
            ciphertext : encrypted bytes
            iv         : initialization vector (needed for decryption)
        """
        iv     = os.urandom(16)
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        ct     = cipher.encrypt(pad(plaintext.encode('utf-8'), AES.block_size))
        return ct, iv

    def decrypt(self, ciphertext: bytes, iv: bytes) -> str:
        """Decrypt ciphertext back to plaintext."""
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        pt     = unpad(cipher.decrypt(ciphertext), AES.block_size)
        return pt.decode('utf-8')


# ─────────────────────────────────────────────
#  Dual Layer AES-256
# ─────────────────────────────────────────────

class DualAES256Encryptor:
    """
    Dual-layer AES-256 encryption.

    Encrypts data twice:
        Layer 1: AES-256 with QKD-derived key
        Layer 2: AES-256 with Kyber-derived key

    Security:
        Attacker must break BOTH QKD key AND Kyber key
        to decrypt. Provides strongest layered protection.

    Overhead:
        ~2x encryption time vs single layer.
        Still fast enough for IoT (< 5ms typical).

    Args:
        qkd_key   : key material from BB84
        kyber_key : shared secret from Kyber KEM
    """

    def __init__(self, qkd_key, kyber_key: bytes):
        self.qkd_key   = derive_aes_key(qkd_key)
        self.kyber_key = derive_aes_key(kyber_key)
        self.mode      = "DUAL"

    def encrypt(self, plaintext: str) -> Tuple[bytes, bytes, bytes]:
        """
        Dual-layer encryption.

        Returns:
            ciphertext : doubly encrypted bytes
            iv1        : IV for layer 1 (QKD)
            iv2        : IV for layer 2 (Kyber)
        """
        # Layer 1: encrypt with QKD key
        iv1     = os.urandom(16)
        cipher1 = AES.new(self.qkd_key, AES.MODE_CBC, iv1)
        layer1  = cipher1.encrypt(
            pad(plaintext.encode('utf-8'), AES.block_size)
        )

        # Layer 2: encrypt layer1 output with Kyber key
        iv2     = os.urandom(16)
        cipher2 = AES.new(self.kyber_key, AES.MODE_CBC, iv2)
        layer2  = cipher2.encrypt(pad(layer1, AES.block_size))

        return layer2, iv1, iv2

    def decrypt(
        self,
        ciphertext: bytes,
        iv1: bytes,
        iv2: bytes
    ) -> str:
        """Dual-layer decryption."""
        # Reverse layer 2 first
        cipher2 = AES.new(self.kyber_key, AES.MODE_CBC, iv2)
        layer1  = unpad(cipher2.decrypt(ciphertext), AES.block_size)

        # Then reverse layer 1
        cipher1 = AES.new(self.qkd_key, AES.MODE_CBC, iv1)
        pt      = unpad(cipher1.decrypt(layer1), AES.block_size)

        return pt.decode('utf-8')


# ─────────────────────────────────────────────
#  IoT Packet Encryptor
# ─────────────────────────────────────────────

class IoTPacketEncryptor:
    """
    High-level IoT packet encryption manager.

    Handles realistic IoT data packets:
        - Traffic sensor readings
        - Smart meter data
        - Camera alerts
        - Hospital device data

    Automatically selects encryption mode based
    on device priority and security mode from AI agent.
    """

    # IoT message templates
    MESSAGE_TEMPLATES = {
        "traffic_sensor" : "SENSOR:{id} | LOC:{lat},{lon} | COUNT:{val} | TS:{ts}",
        "smart_meter"    : "METER:{id}  | READ:{val}kWh | STATUS:OK | TS:{ts}",
        "camera"         : "CAM:{id}    | ALERT:{alert} | CONF:{conf}% | TS:{ts}",
        "hospital"       : "HOSP:{id}   | BP:{bp} HR:{hr} | CRIT:{crit} | TS:{ts}",
        "mobile_sensor"  : "MOB:{id}    | GPS:{lat},{lon} | TEMP:{val}C | TS:{ts}",
    }

    def __init__(
        self,
        hybrid_key   : bytes,
        qkd_key      = None,
        kyber_key    : Optional[bytes] = None,
        mode         : str = "SINGLE"
    ):
        self.hybrid_key = hybrid_key
        self.qkd_key    = qkd_key
        self.kyber_key  = kyber_key
        self.mode       = mode

        # Initialise appropriate encryptor
        if mode == "DUAL" and qkd_key is not None and kyber_key is not None:
            self.encryptor = DualAES256Encryptor(qkd_key, kyber_key)
        else:
            self.encryptor = AES256Encryptor(hybrid_key)

    def encrypt_packet(self, device_type: str, **kwargs) -> dict:
        """
        Encrypt an IoT data packet.

        Args:
            device_type : type of IoT device
            **kwargs    : packet data fields

        Returns:
            Encrypted packet with metadata and timing.
        """
        start = time.perf_counter()

        # Generate message
        template = self.MESSAGE_TEMPLATES.get(
            device_type,
            "DEVICE:{id} | DATA:{val} | TS:{ts}"
        )
        message = template.format(**kwargs)

        # Encrypt
        if self.mode == "DUAL":
            ct, iv1, iv2 = self.encryptor.encrypt(message)
            packet = {
                "ciphertext" : ct,
                "iv1"        : iv1,
                "iv2"        : iv2,
                "mode"       : "DUAL_AES256"
            }
        else:
            ct, iv = self.encryptor.encrypt(message)
            packet = {
                "ciphertext" : ct,
                "iv"         : iv,
                "mode"       : "SINGLE_AES256"
            }

        timing_ms = round((time.perf_counter() - start) * 1000, 4)

        return {
            "device_type"    : device_type,
            "plaintext_len"  : len(message),
            "ciphertext_len" : len(ct),
            "encryption_mode": self.mode,
            "timing_ms"      : timing_ms,
            "packet"         : packet,
            "original"       : message   # remove in production
        }

    def decrypt_packet(self, encrypted_result: dict) -> str:
        """Decrypt an encrypted packet."""
        packet = encrypted_result["packet"]

        if self.mode == "DUAL":
            return self.encryptor.decrypt(
                packet["ciphertext"],
                packet["iv1"],
                packet["iv2"]
            )
        else:
            return self.encryptor.decrypt(
                packet["ciphertext"],
                packet["iv"]
            )


# ─────────────────────────────────────────────
#  TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":

    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from qkd.bb84 import BB84, RANChannel
    from pqc.kyber import KyberKEM
    from hybrid.combiner import HybridKeyCombiner

    print("=" * 58)
    print("  Dual Encryption Layer — Integration Test")
    print("=" * 58)

    # ── Get keys from QKD and Kyber ───────────
    channel    = RANChannel(0.03, 0.01, 8)
    qkd_result = BB84(2000, channel).run()
    kem_result = KyberKEM("Kyber768").full_key_exchange()

    combiner   = HybridKeyCombiner("HKDF")
    hybrid     = combiner.combine(
        qkd_result["secret_key"],
        kem_result["shared_secret"]
    )

    print(f"\n  QKD key    : {qkd_result['secret_key_length']} bits")
    print(f"  Kyber key  : {kem_result['shared_secret_size']*8} bits")
    print(f"  Hybrid key : {hybrid['key_length']*8} bits")

    # ── Test single layer ─────────────────────
    print(f"\n  Single-layer AES-256 encryption")
    print("-" * 58)

    enc    = IoTPacketEncryptor(hybrid["hybrid_key"], mode="SINGLE")
    result = enc.encrypt_packet(
        "traffic_sensor",
        id="001", lat="31.52", lon="74.35",
        val=42, ts=1711360800
    )

    decrypted = enc.decrypt_packet(result)
    ok = decrypted == result["original"]

    print(f"  Original  : {result['original']}")
    print(f"  Encrypted : {result['packet']['ciphertext'].hex()[:40]}...")
    print(f"  Decrypted : {decrypted}")
    print(f"  Match     : {'✓ PASS' if ok else '✗ FAIL'}")
    print(f"  Time      : {result['timing_ms']} ms")

    # ── Test dual layer ───────────────────────
    print(f"\n  Dual-layer AES-256 encryption (QKD + Kyber)")
    print("-" * 58)

    enc_dual = IoTPacketEncryptor(
        hybrid["hybrid_key"],
        qkd_key   = qkd_result["secret_key"],
        kyber_key = kem_result["shared_secret"],
        mode      = "DUAL"
    )

    iot_packets = [
        ("traffic_sensor", dict(id="001", lat="31.52", lon="74.35", val=42,  ts=1711360800)),
        ("smart_meter",    dict(id="089", val="423.7", ts=1711360800)),
        ("camera",         dict(id="014", alert="MOTION", conf=94,  ts=1711360800)),
        ("hospital",       dict(id="003", bp="120/80", hr=72, crit="NO", ts=1711360800)),
    ]

    all_pass  = True
    total_time = 0

    for device_type, kwargs in iot_packets:
        result    = enc_dual.encrypt_packet(device_type, **kwargs)
        decrypted = enc_dual.decrypt_packet(result)
        ok        = decrypted == result["original"]
        if not ok:
            all_pass = False
        total_time += result["timing_ms"]

        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  [{status}] {device_type:<18} "
              f"| {result['timing_ms']:.4f}ms "
              f"| {result['plaintext_len']}→{result['ciphertext_len']} bytes")

    print(f"\n  All packets: {'✓ PASS' if all_pass else '✗ FAIL'}")
    print(f"  Total time : {round(total_time, 4)} ms")
    print(f"  Avg/packet : {round(total_time/len(iot_packets), 4)} ms")
    print(f"  Mode       : DUAL AES-256 (QKD key + Kyber key)")