# -*- coding: utf-8 -*-
"""
qber_analysis.py — QBER Analysis Script
Run this directly if you don't want to use Jupyter.
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib.pyplot as plt
from src.qkd.bb84 import BB84, RANChannel
from config import *
import warnings
warnings.filterwarnings('ignore')

print("=" * 55)
print("  QBER Analysis")
print("=" * 55)

# ── 1. QBER vs Noise Level ────────────────────
print("\n  [1/4] QBER vs noise level...")
noise_levels = np.linspace(0.01, 0.35, 35)
qber_safe, qber_eve = [], []

for n in noise_levels:
    ch = RANChannel(noise_level=n, packet_loss=0.01)
    qber_safe.append(BB84(1000, ch, eavesdrop=False).run()['qber'])
    qber_eve.append(BB84(1000, ch, eavesdrop=True).run()['qber'])

plt.figure(figsize=(10, 4))
plt.plot(noise_levels*100, qber_safe, color='#1D9E75', linewidth=2, label='No Eve')
plt.plot(noise_levels*100, qber_eve,  color='#D85A30', linewidth=2, label='Eve present')
plt.axhline(y=11, color='#7F77DD', linestyle='--', label='Threshold (11%)')
plt.fill_between(noise_levels*100, qber_safe, alpha=0.12, color='#1D9E75')
plt.fill_between(noise_levels*100, qber_eve,  alpha=0.12, color='#D85A30')
plt.xlabel('Noise level (%)')
plt.ylabel('QBER (%)')
plt.title('QBER vs Channel Noise Level')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
os.makedirs('results/plots', exist_ok=True)
plt.savefig('results/plots/qber_vs_noise.png', dpi=150)
plt.show()

# ── 2. QBER Distribution ─────────────────────
print("  [2/4] QBER distribution over 100 trials...")
trials = 100
ch = RANChannel(0.03, 0.01)
safe_qbers = [BB84(1000, ch, False).run()['qber'] for _ in range(trials)]
eve_qbers  = [BB84(1000, ch, True).run()['qber']  for _ in range(trials)]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(safe_qbers, color='#1D9E75', linewidth=1.5, label='No Eve')
axes[0].plot(eve_qbers,  color='#D85A30', linewidth=1.5, label='Eve')
axes[0].axhline(y=11, color='#7F77DD', linestyle='--', label='Threshold')
axes[0].set_title('QBER over 100 trials')
axes[0].set_xlabel('Trial')
axes[0].set_ylabel('QBER (%)')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].hist(safe_qbers, bins=20, color='#1D9E75', alpha=0.7, label='No Eve')
axes[1].hist(eve_qbers,  bins=20, color='#D85A30', alpha=0.7, label='Eve')
axes[1].axvline(x=11, color='#7F77DD', linestyle='--', label='Threshold')
axes[1].set_title('QBER distribution')
axes[1].set_xlabel('QBER (%)')
axes[1].set_ylabel('Frequency')
axes[1].legend()
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig('results/plots/qber_distribution.png', dpi=150)
plt.show()

# ── 3. QBER vs Qubit Count ───────────────────
print("  [3/4] QBER vs qubit count...")
qubit_counts = [100, 200, 500, 1000, 2000, 5000]
means, stds, key_rates = [], [], []

for n in qubit_counts:
    qbers = [BB84(n, ch, False).run()['qber'] for _ in range(20)]
    r     = BB84(n, ch, False).run()
    means.append(np.mean(qbers))
    stds.append(np.std(qbers))
    key_rates.append(r['key_rate'])

plt.figure(figsize=(9, 4))
plt.errorbar(qubit_counts, means, yerr=stds, fmt='o-',
             color='#7F77DD', linewidth=2, capsize=5)
plt.axhline(y=11, color='#D85A30', linestyle='--', label='Threshold (11%)')
plt.xscale('log')
plt.xlabel('Number of qubits')
plt.ylabel('QBER (%)')
plt.title('QBER stability vs qubit count')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('results/plots/qber_vs_qubits.png', dpi=150)
plt.show()

# ── 4. Key Length Analysis ───────────────────
print("  [4/4] Key length analysis...")
key_lengths = [BB84(n, ch, False).run()['secret_key_length'] for n in qubit_counts]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(qubit_counts, key_lengths, 'o-', color='#1D9E75', linewidth=2)
axes[0].set_xscale('log')
axes[0].set_title('Secret key length vs qubits sent')
axes[0].set_xlabel('Qubits sent')
axes[0].set_ylabel('Key length (bits)')
axes[0].grid(alpha=0.3)

axes[1].plot(qubit_counts, key_rates, 'o-', color='#D85A30', linewidth=2)
axes[1].set_xscale('log')
axes[1].set_title('Key rate (bits per qubit)')
axes[1].set_xlabel('Qubits sent')
axes[1].set_ylabel('Key rate')
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig('results/plots/key_length_analysis.png', dpi=150)
plt.show()

# ── Summary ───────────────────────────────────
print("\n  QBER Analysis Summary")
print("=" * 45)
print(f"  Normal QBER  : ~{np.mean(safe_qbers):.2f}% (below 11% threshold)")
print(f"  Attack QBER  : ~{np.mean(eve_qbers):.2f}% (above threshold)")
print(f"  Gap          : {np.mean(eve_qbers)-np.mean(safe_qbers):.2f}%")
print(f"  Recommended  : {QKD_NUM_QUBITS} qubits")
print(f"  Key rate     : ~{key_rates[-2]:.3f} bits/qubit at {qubit_counts[-2]} qubits")
print(f"\n  Plots saved to results/plots/")
