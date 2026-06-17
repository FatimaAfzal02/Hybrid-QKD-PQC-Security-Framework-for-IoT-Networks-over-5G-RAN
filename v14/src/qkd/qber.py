# -*- coding: utf-8 -*-
"""
qber.py — Quantum Bit Error Rate Calculator
============================================
Standalone QBER calculation and analysis.

Author  : FYP Team
Module  : src/qkd/qber.py
"""

import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config import QKD_QBER_THRESHOLD


class QBERCalculator:
    """
    Calculates and analyses Quantum Bit Error Rate.
    QBER = errors / total bits compared
    Threshold: 11% — above this Eve is detected.
    """

    def __init__(self, threshold: float = QKD_QBER_THRESHOLD):
        self.threshold = threshold
        self.history   = []

    def calculate(self, alice_bits: list, bob_bits: list,
                  sample_fraction: float = 0.25) -> dict:
        """Calculate QBER from sifted key samples."""
        if len(alice_bits) == 0:
            return {"qber": 100.0, "errors": 0, "sample_size": 0,
                    "eavesdropper_detected": True,
                    "security_level": "COMPROMISED", "remaining_bits": 0}

        sample_size = max(1, int(len(alice_bits) * sample_fraction))
        errors = sum(1 for a, b in zip(
            alice_bits[:sample_size], bob_bits[:sample_size]) if a != b)
        qber   = errors / sample_size

        result = {
            "qber"                 : round(qber * 100, 2),
            "errors"               : errors,
            "sample_size"          : sample_size,
            "eavesdropper_detected": qber > self.threshold,
            "security_level"       : self._level(qber),
            "remaining_bits"       : len(alice_bits) - sample_size,
        }
        self.history.append(result["qber"])
        return result

    def _level(self, qber: float) -> str:
        if qber < 0.05:       return "SECURE"
        elif qber < self.threshold: return "WARNING"
        else:                 return "COMPROMISED"

    def analyse_history(self) -> dict:
        if not self.history: return {}
        return {
            "sessions": len(self.history),
            "avg_qber": round(np.mean(self.history), 2),
            "max_qber": round(max(self.history), 2),
            "min_qber": round(min(self.history), 2),
            "alerts"  : sum(1 for q in self.history
                            if q > self.threshold * 100),
        }

    @staticmethod
    def theoretical_eve_qber() -> float:
        """Theoretical QBER from intercept-resend: 25%."""
        return 0.25

    @staticmethod
    def is_secure(qber_percent: float) -> bool:
        return qber_percent < QKD_QBER_THRESHOLD * 100


if __name__ == "__main__":
    calc  = QBERCalculator()
    alice = list(np.random.randint(0, 2, 500))
    bob   = alice.copy()
    for i in range(0, len(bob), 33):
        bob[i] = 1 - bob[i]
    r = calc.calculate(alice, bob)
    print(f"QBER: {r['qber']}% | Security: {r['security_level']}")
