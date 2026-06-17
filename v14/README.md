# Hybrid QKD-PQC Security Framework for IoT Networks over RAN

## Final Year Project — Electrical Engineering

### Overview
A simulation-based hybrid security framework combining:
- **BB84 Quantum Key Distribution (QKD)** — eavesdropping detection
- **ML-KEM / Kyber Post-Quantum Cryptography (PQC)** — quantum-safe encryption
- **AI Anomaly Detection** — adaptive security management
- **AES-256 Encryption** — data confidentiality

### Project Structure
```
hybrid-qkd-pqc/
├── src/
│   ├── qkd/          # BB84 protocol, QBER, eavesdropper simulation
│   ├── pqc/          # Kyber ML-KEM implementation
│   ├── hybrid/       # Key combination schemes
│   ├── adaptive/     # AI-driven mode selection
│   └── iot/          # IoT device and RAN simulation
├── tests/            # Unit tests
├── benchmarks/       # Performance measurements
├── notebooks/        # Jupyter analysis notebooks
├── data/             # Raw and processed datasets
├── results/          # Plots and logs
├── config.py         # Global configuration
├── main.py           # Entry point
└── requirements.txt  # Dependencies
```

### Installation
```bash
conda create -n fyp_qkd python=3.10
conda activate fyp_qkd
pip install -r requirements.txt
```

### Run
```bash
python main.py
```

### Research Question
How can a hybrid QKD-PQC framework provide defense-in-depth security
for IoT networks over RAN, achieving quantum-safe encryption and
active eavesdropping detection while maintaining practical deployability?

### Team
- Supervisor: [Name]
- University: [Name]
- Semester: 7

---

## Recent Additions (v2 — Supervisor Feedback)

### New Modules
| File | Description |
|------|-------------|
| `src/metrics/energy.py` | `EnergyEstimator`, `OperationTracker`, `MeasureEnergy` — IoT energy estimation from latency using real device power profiles (ARM Cortex-M0 through RPi Zero) |
| `src/adaptive/metrics.py` | `PerformanceMetrics` — tracks latency, throughput (Mbps), energy, detection rate, and overhead vs classical baseline across the full pipeline |

### New Tests
| File | What it covers |
|------|---------------|
| `tests/test_scalability.py` | 100 / 200 / 500 device creation, hybrid key exchange at scale, success rate >95%, per-device latency <100ms, failover with 20% QKD failures, energy budget, QBER consistency |
| `tests/test_sidechannel.py` | Kyber encap/decap timing consistency, HKDF/XOR key-content independence, AES-256 plaintext-independent timing, constant-time key comparison, privacy amplification output length |

### New Benchmark
| File | What it measures |
|------|----------------|
| `benchmarks/throughput_benchmark.py` | Data throughput (Mbps) per component, payload size scaling (32B–2048B), aggregate throughput under 10/50/100 device load |

### Improvements
- `src/adaptive/agent.py` — key refresh interval is now **truly adaptive** via `AdaptiveRefreshCalculator` (QBER + AI confidence + signal strength factors) instead of fixed values
- `main.py` — now tracks and prints energy per operation using `OperationTracker`
- `config.py` — added `IOT_POWER_MW` and `ENERGY_TARGET_J` constants
- `requirements.txt` — `qiskit` and `liboqs-python` marked as optional (not required to run)

### Run Tests
```bash
python -m pytest tests/ -v          # all 122 tests
python -m pytest tests/test_scalability.py -v
python -m pytest tests/test_sidechannel.py -v
```

### Run Benchmarks
```bash
python benchmarks/throughput_benchmark.py
python benchmarks/energy_benchmark.py
python benchmarks/latency_benchmark.py
```
