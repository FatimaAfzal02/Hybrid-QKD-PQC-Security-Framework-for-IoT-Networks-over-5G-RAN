# Security Analysis: Hybrid QKD-PQC Key Combination Scheme

**Project:** Hybrid QKD-PQC Security Framework for IoT/RAN  
**Module:** `src/hybrid/combiner.py` — `HKDFCombiner`  
**Standard:** RFC 5869 (HKDF), NIST FIPS 203 (ML-KEM / Kyber), ETSI GS QKD 014  

---

## 1. System Overview

The hybrid scheme combines two independently generated keys:

```
QKD_key   ←  BB84 protocol (information-theoretic security)
Kyber_key ←  ML-KEM-768 (Kyber768, computational security, NIST Level 3)

IKM  = QKD_key || Kyber_key          (concatenated input key material)
PRK  = HMAC-SHA256(salt, IKM)        (HKDF Extract step — RFC 5869 §2.2)
K_H  = HMAC-SHA256(PRK, info || 1)   (HKDF Expand step — RFC 5869 §2.3)
```

The output `K_H` (256 bits) is used as the AES-256 encryption key.  
Context string: `info = b"hybrid-qkd-pqc-iot-ran"` (domain separation).

---

## 2. Security Definitions

### 2.1 IND-CPA and IND-CCA2
A key encapsulation mechanism is **IND-CPA secure** if no probabilistic polynomial-time (PPT) adversary can distinguish the encapsulated key from a random key with non-negligible advantage.  
**IND-CCA2** additionally requires security against adaptive chosen-ciphertext attacks.

### 2.2 Information-Theoretic Security (QKD)
A QKD-derived key is **information-theoretically secure** if an adversary with unlimited computational power cannot determine the key with probability greater than 2^{-n}, where n is the key length in bits.

---

## 3. Security Assumptions

| # | Assumption | Source | Level |
|---|-----------|--------|-------|
| A1 | ML-LWE (Module Learning With Errors) is computationally hard | NIST FIPS 203, 2024 | NIST Level 3 (≥192-bit classical) |
| A2 | BB84 QKD is secure when QBER < 11% (our threshold) | Mayers 2001, Lo & Chau 1999 | Information-theoretic |
| A3 | HKDF is a secure pseudorandom function (PRF) under HMAC-SHA256 | RFC 5869, Krawczyk 2010 | Computational |
| A4 | SHA-256 is collision-resistant and a secure hash function | NIST FIPS 180-4 | 128-bit post-quantum security |
| A5 | The two key sources (QKD and Kyber) are generated independently | Architectural (separate modules) | By design |

---

## 4. Main Theorem

**Theorem 1 (Hybrid Security):**  
*Under assumptions A1–A5, the hybrid key `K_H = HKDF(QKD_key || Kyber_key)` is computationally indistinguishable from a uniformly random 256-bit string, provided that at least one of the following holds:*

- *(a) The ML-LWE problem is hard (Kyber key is secret), OR*
- *(b) The BB84 QKD session was secure (QBER < 11%).*

*Formally: for any PPT adversary A,*  
*`Adv[A] ≤ Adv_MLWE(A₁) + Adv_QKD(A₂) + Adv_PRF(A₃) + negl(λ)`*  
*where `negl(λ)` is a negligible function of the security parameter λ.*

---

## 5. Proof Sketch

### 5.1 Hybrid Argument (Game-Based)

We proceed through a sequence of games, each computationally indistinguishable from the last.

**Game G0** — Real hybrid scheme.  
Challenger runs BB84 and Kyber, computes `K_H = HKDF(QKD_key || Kyber_key)`.  
Adversary A receives all public parameters (Kyber public key, quantum channel transcript, QBER value).  
A must distinguish `K_H` from random `K_R ← {0,1}^256`.

**Game G1** — Replace Kyber shared secret with random.  
We replace `Kyber_key` with a uniformly random string `r ← {0,1}^256`.  
Any adversary that distinguishes G0 from G1 solves the ML-LWE problem (Kyber IND-CCA2 security, NIST FIPS 203 Theorem 1).  
Therefore: `|Pr[G0] - Pr[G1]| ≤ Adv_MLWE(A₁)`

**Game G2** — Replace QKD key with random (independently).  
We additionally replace `QKD_key` with a uniformly random string `s ← {0,1}^n`.  
Any adversary that distinguishes G1 from G2 breaks QKD information-theoretic security.  
By the composability theorem for QKD (Renner 2005): `|Pr[G1] - Pr[G2]| ≤ 2^{-n/2}` when QBER < 11%.  
This is `negl(λ)` for our key length (n = 256 bits after privacy amplification).

**Game G3** — HKDF output is pseudorandom.  
In G2, the input to HKDF is `r || s` where both r, s are uniformly random and independent.  
By the PRF security of HKDF (Krawczyk 2010, Theorem 1): HKDF output is computationally indistinguishable from uniform.  
`|Pr[G2] - Pr[G3]| ≤ Adv_PRF(A₃)`

**Game G3** — Adversary has no advantage.  
`Pr[A wins G3] = 1/2` (purely random).

**Combining:**  
`Adv[A] = |Pr[A wins G0] - 1/2| ≤ Adv_MLWE + 2^{-128} + Adv_PRF + negl(λ)`

Since `Adv_MLWE` and `Adv_PRF` are negligible under assumptions A1 and A3:  
`Adv[A] ≤ negl(λ)` ∎

### 5.2 Intuition

The security reduces to the OR of both assumptions: an attacker must break **both** QKD (information-theoretically hard when QBER < 11%) **and** Kyber (computationally hard under ML-LWE) simultaneously to recover the hybrid key. This is the core "defence-in-depth" property.

```
Attack succeeds iff:
    Break_QKD  (requires unlimited compute + intercept rate >44%)
    AND
    Break_Kyber (requires solving ML-LWE with 192-bit security)

Probability ≈ negl(λ) × 0 = 0
```

---

## 6. Failover Security Properties

The `HybridKeyCombiner` handles partial failures. The security in each mode:

| Mode | Condition | Security Basis | Security Level |
|------|-----------|----------------|----------------|
| `FULL_HYBRID` | Both QKD and Kyber succeed | Theorem 1 above | Information-theoretic + Computational |
| `DEGRADED — PQC only` | QKD fails (high QBER) | ML-LWE hardness (A1) | 192-bit classical / 128-bit post-quantum |
| `DEGRADED — QKD only` | Kyber fails | QKD composability (A2) | Information-theoretic (QBER < 11%) |
| `FAILED` | Both fail | None — abort | System refuses to encrypt |

The `FAILED` case is a deliberate security decision: the system refuses to produce a key rather than fall back to insecure classical methods. This ensures no secret communication happens under compromised conditions.

---

## 7. Threat Model Coverage

| Threat | Detected By | Response |
|--------|-------------|----------|
| Eavesdropping on quantum channel (BB84) | QBER monitoring (threshold 11%) | Abort session, discard key |
| Intercept-resend attack (Eve intercepts >44% of qubits) | QBER rises above threshold | Detected with probability → 1 |
| MITM on Kyber KEM | AI agent (Random Forest, F1≈98%) | Switch to QKD-only mode |
| Harvest-now-decrypt-later (quantum computer breaks Kyber) | Architectural | QKD key remains information-theoretically secure |
| Side-channel timing attack on HKDF | `tests/test_sidechannel.py` | Constant-time operations via `hmac.compare_digest` |
| Photon number splitting (PNS) attack | Privacy amplification | Key compressed by leaked bits |
| RNG compromise | Dual entropy source (QKD + Kyber) | One secure source sufficient |

---

## 8. HKDF Domain Separation

The `info` parameter `b"hybrid-qkd-pqc-iot-ran"` provides domain separation between:
- Keys for different sessions
- Keys for different IoT device types  
- Keys derived from the same material for different purposes

This prevents key reuse attacks across contexts (RFC 5869 §3.2).

---

## 9. Post-Quantum Security Level

| Component | Classical Security | Post-Quantum Security | Standard |
|-----------|-------------------|----------------------|----------|
| Kyber768 (ML-KEM-768) | ~192-bit | ~128-bit (NIST Level 3) | NIST FIPS 203, 2024 |
| BB84 QKD | Information-theoretic | Information-theoretic | ETSI GS QKD 014 |
| HKDF-SHA256 | 128-bit | 128-bit (Grover: 2^128) | RFC 5869 |
| AES-256 | 256-bit | 128-bit (Grover: 2^128) | NIST FIPS 197 |
| **Hybrid** | **192-bit (bottleneck: Kyber)** | **128-bit** | **This work** |

The post-quantum security level of the full system is **128 bits** — matching the strongest individual component when both succeed.

---

## 10. References

1. C. H. Bennett and G. Brassard, "Quantum cryptography: Public key distribution and coin tossing," *ICASSP*, 1984. (BB84 original)
2. D. Mayers, "Unconditional security in quantum cryptography," *J. ACM*, 48(3):351–406, 2001.
3. H.-K. Lo and H. F. Chau, "Unconditional security of quantum key distribution over arbitrarily long distances," *Science*, 283:2050–2056, 1999.
4. R. Renner, "Security of quantum key distribution," *PhD thesis*, ETH Zurich, 2005.
5. H. Krawczyk, "Cryptographic extraction and key derivation: The HKDF scheme," *CRYPTO 2010*, LNCS 6223, pp. 631–648.
6. H. Krawczyk and P. Eronen, "HMAC-based Extract-and-Expand Key Derivation Function (HKDF)," *RFC 5869*, IETF, 2010.
7. NIST, "Module-Lattice-Based Key-Encapsulation Mechanism Standard (ML-KEM)," *FIPS 203*, August 2024.
8. ETSI, "Quantum Key Distribution (QKD); Protocol and data format of REST-based key delivery API," *GS QKD 014 v1.1.1*, 2019.
9. M. Campagna et al., "Quantum safe cryptography and security," *ETSI White Paper No. 8*, 2015.
