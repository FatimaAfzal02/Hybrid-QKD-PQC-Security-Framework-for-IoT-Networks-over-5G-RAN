# -*- coding: utf-8 -*-
"""
eavesdropper.py — Eavesdropper Attack Simulation
=================================================
Standalone eavesdropper module for BB84 QKD.

Attack types:
    intercept_resend : Eve measures and resends (basic, detectable)
    photon_splitting : Eve splits photon pulses  (advanced)

Author  : FYP Team
Module  : src/qkd/eavesdropper.py
"""

import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


class Eavesdropper:
    """
    Simulates quantum eavesdropping attacks on BB84.

    Intercept-resend:
        Eve measures each qubit with a random basis.
        When she guesses wrong (50% chance) she disturbs
        the qubit — Bob sees higher QBER.
        Detection probability: 25% per qubit.

    Photon splitting (simulated):
        Eve splits multi-photon pulses, stores one photon,
        forwards others. Lower detection probability.
        Simulated by reducing disturbance rate.
    """

    ATTACK_TYPES = ["intercept_resend", "photon_splitting"]

    def __init__(self, attack_type: str = "intercept_resend"):
        if attack_type not in self.ATTACK_TYPES:
            raise ValueError(f"Unknown attack: {attack_type}. "
                             f"Choose from {self.ATTACK_TYPES}")
        self.attack_type     = attack_type
        self.intercepted     = 0
        self.correct_guesses = 0
        self.errors_caused   = 0

    def intercept(self, qubits: np.ndarray,
                  alice_bases: np.ndarray) -> np.ndarray:
        """
        Intercept qubits and return disturbed versions.

        Args:
            qubits      : Alice's qubits
            alice_bases : Alice's encoding bases

        Returns:
            Disturbed qubits forwarded to Bob
        """
        if self.attack_type == "intercept_resend":
            return self._intercept_resend(qubits, alice_bases)
        else:
            return self._photon_splitting(qubits, alice_bases)

    def _intercept_resend(self, qubits: np.ndarray,
                          alice_bases: np.ndarray) -> np.ndarray:
        """Basic intercept-resend attack."""
        eve_bases  = np.random.randint(0, 2, len(qubits))
        result     = []

        for i in range(len(qubits)):
            self.intercepted += 1
            if eve_bases[i] == alice_bases[i]:
                # Correct basis — no disturbance
                result.append(qubits[i])
                self.correct_guesses += 1
            else:
                # Wrong basis — random result, qubit disturbed
                new_bit = np.random.randint(0, 2)
                result.append(new_bit)
                if new_bit != qubits[i]:
                    self.errors_caused += 1

        return np.array(result)

    def _photon_splitting(self, qubits: np.ndarray,
                          alice_bases: np.ndarray) -> np.ndarray:
        """
        Simulated photon number splitting attack.
        Eve only intercepts multi-photon pulses (30% of traffic).
        Causes fewer detectable errors than intercept-resend.
        """
        result = []
        for i in range(len(qubits)):
            self.intercepted += 1
            # Only intercept 30% of pulses (multi-photon)
            if np.random.random() < 0.30:
                eve_basis = np.random.randint(0, 2)
                if eve_basis == alice_bases[i]:
                    result.append(qubits[i])
                    self.correct_guesses += 1
                else:
                    new_bit = np.random.randint(0, 2)
                    result.append(new_bit)
                    if new_bit != qubits[i]:
                        self.errors_caused += 1
            else:
                # Pass through unchanged
                result.append(qubits[i])

        return np.array(result)

    @property
    def detection_probability(self) -> float:
        """Theoretical detection probability per qubit."""
        return 0.25 if self.attack_type == "intercept_resend" else 0.075

    @property
    def info_gained(self) -> float:
        """Fraction of key bits Eve correctly learned."""
        if self.intercepted == 0:
            return 0.0
        return round(self.correct_guesses / self.intercepted, 3)

    def stats(self) -> dict:
        return {
            "attack_type"         : self.attack_type,
            "intercepted"         : self.intercepted,
            "correct_guesses"     : self.correct_guesses,
            "errors_caused"       : self.errors_caused,
            "info_gained"         : self.info_gained,
            "detection_probability": self.detection_probability,
        }


if __name__ == "__main__":
    print("Eavesdropper simulation test")
    qubits = np.random.randint(0, 2, 1000)
    bases  = np.random.randint(0, 2, 1000)

    for attack in ["intercept_resend", "photon_splitting"]:
        eve  = Eavesdropper(attack)
        dist = eve.intercept(qubits, bases)
        s    = eve.stats()
        print(f"\n  {attack}")
        print(f"  Info gained : {s['info_gained']*100:.1f}%")
        print(f"  Errors caused: {s['errors_caused']}")
        print(f"  Detection prob: {s['detection_probability']*100:.1f}%")
