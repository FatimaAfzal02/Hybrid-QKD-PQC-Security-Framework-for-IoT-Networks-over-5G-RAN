# ─────────────────────────────────────────────
#  config.py — Global project configuration
#  All parameters in one place — change here,
#  affects the entire system automatically
# ─────────────────────────────────────────────

# ── QKD Parameters ────────────────────────────
QKD_NUM_QUBITS      = 2000      # Number of qubits per BB84 session
QKD_QBER_THRESHOLD  = 0.11      # 11% — eavesdropper detection threshold
QKD_SAMPLE_FRACTION = 0.25      # 25% of sifted key used for QBER check

# ── RAN Channel Parameters ────────────────────
RAN_NOISE_NORMAL    = 0.03      # 3% noise — normal city conditions
RAN_NOISE_DEGRADED  = 0.08      # 8% noise — busy network
RAN_NOISE_ATTACK    = 0.20      # 20% noise — under attack
RAN_LOSS_NORMAL     = 0.01      # 1% packet loss — normal
RAN_LOSS_DEGRADED   = 0.05      # 5% packet loss — degraded
RAN_LOSS_ATTACK     = 0.10      # 10% packet loss — attack
RAN_DELAY_NORMAL    = 8         # ms — normal delay
RAN_DELAY_DEGRADED  = 25        # ms — degraded delay
RAN_DELAY_ATTACK    = 50        # ms — attack delay

# ── PQC Parameters ────────────────────────────
PQC_ALGORITHM       = "Kyber768"  # NIST ML-KEM standard
PQC_SECURITY_LEVEL  = 3          # 1=Kyber512, 3=Kyber768, 5=Kyber1024

# ── Hybrid Key Combination ────────────────────
HYBRID_KEY_LENGTH   = 32        # 256 bits = 32 bytes for AES-256
HYBRID_METHOD       = "HKDF"    # Options: HKDF, XOR, DUAL

# ── AI Model Parameters ───────────────────────
AI_N_ESTIMATORS     = 100       # Random Forest trees
AI_MAX_DEPTH        = 10        # Max tree depth
AI_TEST_SIZE        = 0.2       # 20% test split
AI_RANDOM_STATE     = 42        # Reproducibility seed
AI_CLASSES          = {0: "Normal", 1: "Degraded", 2: "Under Attack"}

# ── Evaluation Parameters ─────────────────────
EVAL_NUM_TRIALS     = 50        # Trials per scenario
EVAL_TARGET_F1      = 0.95      # Target F1 score
EVAL_TARGET_DETECT  = 95.0      # Target attack detection rate %

# ── IoT Device Parameters ─────────────────────
IOT_NUM_DEVICES     = 100       # Simulated IoT devices in network
# IOT_KEY_REFRESH is DEPRECATED — kept for reference only.
# AdaptiveRefreshCalculator in src/adaptive/agent.py supersedes this with
# dynamic intervals computed from live QBER + model confidence scores.
# Do not use this constant in new code.
# IOT_KEY_REFRESH = 30

# ── Paths ─────────────────────────────────────
DATA_RAW_PATH       = "data/raw/"
DATA_PROCESSED_PATH = "data/processed/"
RESULTS_PLOTS_PATH  = "results/plots/"
RESULTS_LOGS_PATH   = "results/logs/"

# ── Energy / Power Parameters ─────────────────
IOT_POWER_MW        = 25.0      # ARM Cortex-M4 baseline active power (mW)
ENERGY_TARGET_J     = 10.0      # < 10 J per key exchange (problem statement)
