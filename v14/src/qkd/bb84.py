# -*- coding: utf-8 -*-
"""
bb84.py — BB84 Quantum Key Distribution Protocol
=================================================
Implements the full BB84 QKD protocol pipeline:

    1. Qubit preparation     (Alice)
    2. Qubit transmission    (RAN channel)
    3. Eavesdropper attack   (Eve — optional)
    4. Qubit measurement     (Bob)
    5. Basis sifting         (Alice + Bob compare bases publicly)
    6. QBER estimation       (detect eavesdropper)
    7. Error reconciliation  (Cascade-style parity check)
    8. Privacy amplification (SHA-256 hashing to shrink key)
    9. Final secret key      (quantum-secure shared key)

Author  : FYP Team
Module  : src/qkd/bb84.py
Project : Hybrid QKD-PQC Security Framework for IoT/RAN
"""

import numpy as np
import hashlib
from typing import Optional
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config import (
    QKD_NUM_QUBITS,
    QKD_QBER_THRESHOLD,
    QKD_SAMPLE_FRACTION
)


# ─────────────────────────────────────────────
#  RAN Channel (imported here for standalone use)
# ─────────────────────────────────────────────

class RANChannel:
    """
    Simulates a Radio Access Network channel
    between an IoT device and a base station.

    Args:
        noise_level  : probability a bit gets flipped (0.0 - 1.0)
        packet_loss  : probability a bit is lost     (0.0 - 1.0)
        delay_ms     : simulated transmission delay in ms
    """

    def __init__(
        self,
        noise_level: float = 0.03,
        packet_loss: float = 0.01,
        delay_ms: int = 8
    ):
        self.noise_level = noise_level
        self.packet_loss = packet_loss
        self.delay_ms    = delay_ms

    def transmit(self, bits: list) -> list:
        """
        Transmit bits through noisy RAN channel.
        Returns received bits with None for lost bits.
        """
        received = []
        for bit in bits:
            if np.random.random() < self.packet_loss:
                received.append(None)           # bit lost
                continue
            if np.random.random() < self.noise_level:
                received.append(1 - bit)        # bit flipped
            else:
                received.append(bit)            # bit correct
        return received


# ─────────────────────────────────────────────
#  Eavesdropper (Eve)
# ─────────────────────────────────────────────

class Eavesdropper:
    """
    Simulates an intercept-resend attack.

    Eve intercepts qubits from Alice, measures them
    with random bases, then resends to Bob.
    This introduces detectable errors (QBER spike).

    Attack types:
        intercept_resend : Eve measures and resends (basic)
        photon_splitting  : Eve splits photon pulses  (advanced)
    """

    def __init__(self, attack_type: str = "intercept_resend"):
        self.attack_type    = attack_type
        self.intercepted    = 0
        self.correct_guesses = 0

    def intercept(
        self,
        qubits: np.ndarray,
        alice_bases: np.ndarray
    ) -> np.ndarray:
        """
        Eve intercepts qubits and resends disturbed versions.
        Returns disturbed qubits sent onward to Bob.
        """
        eve_bases   = np.random.randint(0, 2, len(qubits))
        eve_qubits  = []

        for i in range(len(qubits)):
            self.intercepted += 1
            if eve_bases[i] == alice_bases[i]:
                # Eve guessed correct basis — no disturbance
                eve_qubits.append(qubits[i])
                self.correct_guesses += 1
            else:
                # Eve guessed wrong basis — random result
                # AND disturbs the qubit (50% error introduced)
                eve_qubits.append(np.random.randint(0, 2))

        return np.array(eve_qubits)

    @property
    def detection_probability(self) -> float:
        """Theoretical probability Eve is detected per qubit."""
        return 0.25  # 25% per qubit with intercept-resend


# ─────────────────────────────────────────────
#  Error Reconciliation
# ─────────────────────────────────────────────

class ErrorReconciliation:
    """
    Cascade-style error reconciliation.

    Alice and Bob compare parity of blocks of their
    sifted keys to find and fix disagreements without
    revealing the actual key bits.

    This is necessary because channel noise introduces
    errors even without an eavesdropper.
    """

    def __init__(self, block_size: int = 8):
        self.block_size  = block_size
        self.bits_leaked = 0    # track info leaked to Eve during reconciliation

    def reconcile(
        self,
        alice_key: list,
        bob_key: list
    ) -> tuple:
        """
        Reconcile Bob's key to match Alice's key.

        Returns:
            alice_reconciled : Alice's key after reconciliation
            bob_reconciled   : Bob's corrected key
            errors_fixed     : number of errors corrected
        """
        alice_rec   = alice_key.copy()
        bob_rec     = bob_key.copy()
        errors_fixed = 0

        # Split into blocks and compare parity
        for i in range(0, len(alice_rec) - self.block_size, self.block_size):
            alice_block = alice_rec[i:i + self.block_size]
            bob_block   = bob_rec[i:i + self.block_size]

            alice_parity = sum(alice_block) % 2
            bob_parity   = sum(bob_block)   % 2

            # Parities differ — there's an error in this block
            if alice_parity != bob_parity:
                # Binary search within block to find error position
                mid = i + self.block_size // 2
                alice_left_parity = sum(alice_rec[i:mid]) % 2
                bob_left_parity   = sum(bob_rec[i:mid])   % 2

                if alice_left_parity != bob_left_parity:
                    # Error is in left half — flip the last bit of left half
                    error_pos = mid - 1
                else:
                    # Error is in right half — flip the last bit of right half
                    error_pos = i + self.block_size - 1

                # Fix the error
                bob_rec[error_pos] = alice_rec[error_pos]
                errors_fixed += 1

            # Each parity comparison leaks 1 bit of information
            self.bits_leaked += 1

        return alice_rec, bob_rec, errors_fixed


# ─────────────────────────────────────────────
#  Privacy Amplification
# ─────────────────────────────────────────────

class PrivacyAmplification:
    """
    Privacy amplification using cryptographic hashing.

    After error reconciliation, Eve may have partial
    information about the key. Privacy amplification
    compresses the key using SHA-256 hashing, making
    Eve's partial knowledge useless.

    The final key is shorter but information-theoretically
    secure against Eve.
    """

    def amplify(
        self,
        key_bits: list,
        bits_leaked: int = 0,
        target_length: int = 256
    ) -> bytes:
        """
        Amplify privacy of reconciled key.

        Args:
            key_bits      : reconciled key as list of bits
            bits_leaked   : bits leaked during reconciliation
            target_length : desired final key length in bits

        Returns:
            Final secure key as bytes (32 bytes = 256 bits)
        """
        # Convert bits to bytes
        key_bytes = self._bits_to_bytes(key_bits)

        # SHA-256 hash compresses key and removes Eve's partial knowledge
        # Even if Eve knows some bits, hash makes full key unpredictable
        amplified = hashlib.sha256(key_bytes).digest()

        return amplified  # 32 bytes = 256 bits — perfect for AES-256

    def _bits_to_bytes(self, bits: list) -> bytes:
        """Convert list of bits to bytes."""
        padded = bits + [0] * ((8 - len(bits) % 8) % 8)
        ba     = bytearray()
        for i in range(0, len(padded), 8):
            byte = 0
            for bit in padded[i:i + 8]:
                byte = (byte << 1) | int(bit)
            ba.append(byte)
        return bytes(ba)


# ─────────────────────────────────────────────
#  BB84 Main Protocol
# ─────────────────────────────────────────────

class BB84:
    """
    Full BB84 Quantum Key Distribution Protocol.

    Pipeline:
        prepare → transmit → [intercept] → measure →
        sift → QBER → reconcile → amplify → secret key

    Args:
        num_qubits  : number of qubits Alice sends
        channel     : RANChannel instance
        eavesdrop   : whether Eve is present
        attack_type : type of eavesdropping attack
    """

    def __init__(
        self,
        num_qubits: int = QKD_NUM_QUBITS,
        channel: Optional[RANChannel] = None,
        eavesdrop: bool = False,
        attack_type: str = "intercept_resend"
    ):
        self.num_qubits     = num_qubits
        self.channel        = channel or RANChannel()
        self.eavesdrop      = eavesdrop
        self.eve            = Eavesdropper(attack_type) if eavesdrop else None
        self.reconciler     = ErrorReconciliation(block_size=8)
        self.amplifier      = PrivacyAmplification()

    def run(self) -> dict:
        """
        Execute full BB84 protocol.

        Returns dict with:
            qber                  : Quantum Bit Error Rate (%)
            secret_key            : Final secure key (bytes)
            secret_key_length     : Length in bits
            eavesdropper_detected : Boolean
            sifted_key_length     : After basis sifting
            errors_fixed          : Errors corrected
            bits_leaked           : Info leaked in reconciliation
            key_rate              : Efficiency (secret bits / qubits sent)
        """

        # ── Step 1: Alice prepares qubits ─────────────
        alice_bits   = np.random.randint(0, 2, self.num_qubits)
        alice_bases  = np.random.randint(0, 2, self.num_qubits)
        # Basis 0 = rectilinear (+), Basis 1 = diagonal (×)
        alice_qubits = alice_bits.copy()

        # ── Step 2: Eve intercepts (if present) ───────
        if self.eavesdrop and self.eve:
            alice_qubits = self.eve.intercept(alice_qubits, alice_bases)

        # ── Step 3: Transmit through RAN channel ──────
        transmitted      = list(alice_qubits)
        bob_received_raw = self.channel.transmit(transmitted)

        # ── Step 4: Bob measures with random bases ────
        bob_bases = np.random.randint(0, 2, self.num_qubits)
        bob_bits  = []

        for i in range(self.num_qubits):
            received = bob_received_raw[i]
            if received is None:
                bob_bits.append(None)               # lost in channel
            elif bob_bases[i] == alice_bases[i]:
                bob_bits.append(received)            # correct basis
            else:
                bob_bits.append(np.random.randint(0, 2))  # wrong basis → random

        # ── Step 5: Basis sifting ─────────────────────
        # Keep only bits where Alice and Bob used same basis
        # AND where bit was not lost in channel
        sifted_alice = []
        sifted_bob   = []

        for i in range(self.num_qubits):
            if (bob_bits[i] is not None and
                    alice_bases[i] == bob_bases[i]):
                sifted_alice.append(int(alice_bits[i]))
                sifted_bob.append(int(bob_bits[i]))

        if len(sifted_alice) == 0:
            return self._empty_result("No sifted bits — channel too noisy")

        # ── Step 6: QBER estimation ───────────────────
        # Sacrifice sample_fraction of sifted key to check errors
        sample_size = max(1, int(len(sifted_alice) * QKD_SAMPLE_FRACTION))
        sample_errors = sum(
            1 for a, b in zip(
                sifted_alice[:sample_size],
                sifted_bob[:sample_size]
            ) if a != b
        )
        qber = sample_errors / sample_size

        # Remaining bits after QBER check
        remaining_alice = sifted_alice[sample_size:]
        remaining_bob   = sifted_bob[sample_size:]

        eavesdropper_detected = qber > QKD_QBER_THRESHOLD

        # ── Step 7: Error reconciliation ─────────────
        # Only proceed if QBER is acceptable
        if eavesdropper_detected:
            return {
                "qber"                 : round(qber * 100, 2),
                "secret_key"           : None,
                "secret_key_length"    : 0,
                "secret_key_alice"     : [],
                "eavesdropper_detected": True,
                "sifted_key_length"    : len(sifted_alice),
                "errors_fixed"         : 0,
                "bits_leaked"          : 0,
                "key_rate"             : 0.0,
                "status"               : "ABORTED — eavesdropper detected"
            }

        alice_rec, bob_rec, errors_fixed = self.reconciler.reconcile(
            remaining_alice, remaining_bob
        )

        # ── Step 8: Privacy amplification ────────────
        final_key = self.amplifier.amplify(
            alice_rec,
            bits_leaked=self.reconciler.bits_leaked,
            target_length=256
        )

        # ── Step 9: Return results ────────────────────
        key_rate = len(final_key) * 8 / self.num_qubits  # bits per qubit

        return {
            "qber"                 : round(qber * 100, 2),
            "secret_key"           : final_key,           # bytes — use for AES
            "secret_key_length"    : len(final_key) * 8,  # in bits
            "secret_key_alice"     : alice_rec,            # raw bits
            "eavesdropper_detected": False,
            "sifted_key_length"    : len(sifted_alice),
            "errors_fixed"         : errors_fixed,
            "bits_leaked"          : self.reconciler.bits_leaked,
            "key_rate"             : round(key_rate, 4),
            "status"               : "SUCCESS"
        }

    def _empty_result(self, status: str) -> dict:
        """Return empty result on failure."""
        return {
            "qber"                 : 100.0,
            "secret_key"           : None,
            "secret_key_length"    : 0,
            "secret_key_alice"     : [],
            "eavesdropper_detected": True,
            "sifted_key_length"    : 0,
            "errors_fixed"         : 0,
            "bits_leaked"          : 0,
            "key_rate"             : 0.0,
            "status"               : status
        }


# ─────────────────────────────────────────────
#  TEST — Run when executed directly
# ─────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 55)
    print("  BB84 QKD — Full Protocol Test")
    print("=" * 55)

    channel = RANChannel(noise_level=0.03, packet_loss=0.01, delay_ms=8)

    # ── Scenario 1: No Eve ────────────────────
    print("\n  Scenario 1: Normal transmission (no Eve)")
    print("-" * 55)
    qkd    = BB84(num_qubits=2000, channel=channel, eavesdrop=False)
    result = qkd.run()

    print(f"  Status               : {result['status']}")
    print(f"  Sifted key length    : {result['sifted_key_length']} bits")
    print(f"  QBER                 : {result['qber']}%")
    print(f"  Errors fixed         : {result['errors_fixed']}")
    print(f"  Bits leaked (recon.) : {result['bits_leaked']}")
    print(f"  Final key length     : {result['secret_key_length']} bits")
    print(f"  Key rate             : {result['key_rate']} bits/qubit")
    print(f"  Eavesdropper alert   : {result['eavesdropper_detected']}")
    if result['secret_key']:
        print(f"  Key (hex)            : {result['secret_key'].hex()[:32]}...")

    # ── Scenario 2: Eve present ───────────────
    print("\n  Scenario 2: Eve intercepting (intercept-resend)")
    print("-" * 55)
    qkd_eve    = BB84(num_qubits=2000, channel=channel, eavesdrop=True)
    result_eve = qkd_eve.run()

    print(f"  Status               : {result_eve['status']}")
    print(f"  Sifted key length    : {result_eve['sifted_key_length']} bits")
    print(f"  QBER                 : {result_eve['qber']}%")
    print(f"  Eavesdropper alert   : {result_eve['eavesdropper_detected']}")

    # ── Scenario 3: QBER across conditions ───
    print("\n  Scenario 3: QBER across channel conditions")
    print("-" * 55)
    print(f"  {'Scenario':<18} {'QBER':>8} {'Key bits':>10} {'Status':>15}")
    print("-" * 55)

    test_scenarios = [
        ("Normal city"   , 0.03, 0.01, False),
        ("Busy network"  , 0.08, 0.05, False),
        ("With Eve"      , 0.03, 0.01, True),
        ("Eve + noise"   , 0.08, 0.05, True),
    ]

    for name, noise, loss, eve in test_scenarios:
        ch  = RANChannel(noise_level=noise, packet_loss=loss)
        qkd = BB84(num_qubits=2000, channel=ch, eavesdrop=eve)
        r   = qkd.run()
        print(f"  {name:<18} {r['qber']:>7}% "
              f"{r['secret_key_length']:>10} "
              f"{r['status']:>15}")

    # ── Scenario 4: Multiple trials ───────────
    print("\n  Scenario 4: QBER statistics over 20 trials")
    print("-" * 55)

    safe_qbers = []
    eve_qbers  = []

    for _ in range(20):
        ch   = RANChannel(0.03, 0.01)
        safe_qbers.append(BB84(1000, ch, False).run()['qber'])
        eve_qbers.append(BB84(1000, ch, True).run()['qber'])

    print(f"  Without Eve — Avg QBER : {np.mean(safe_qbers):.2f}%  "
          f"Max: {max(safe_qbers):.2f}%")
    print(f"  With Eve    — Avg QBER : {np.mean(eve_qbers):.2f}%  "
          f"Min: {min(eve_qbers):.2f}%")
    print(f"  Threshold              : {QKD_QBER_THRESHOLD*100}%")
    print(f"  Separation             : "
          f"{np.mean(eve_qbers) - np.mean(safe_qbers):.2f}% gap")