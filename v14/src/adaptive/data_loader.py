# -*- coding: utf-8 -*-
"""
data_loader.py — QKD Dataset Loader and Preprocessor
======================================================
Loads and preprocesses the QKD Attack Detection Dataset
for training the AI anomaly detection model.

Dataset: QKD Attack Detection Dataset (Kaggle)
Source : kaggle.com/datasets/sattwiksarkar999/qkd-attack-dataset

Original columns:
    Sifted_Key_Length      → key_length
    QBER                   → qber
    Measurement_entropy    → measurement_entropy (new feature)
    Signal_detection_rate  → signal_strength proxy
    Decoy_detection_rate   → decoy_rate (new feature)
    Avg_Photon_time        → delay proxy
    Whole_key_time         → whole_key_time (new feature)
    Arrival_var            → bit_error_rate proxy
    Arrival_dev            → packet_loss proxy
    Label                  → attack class

Label mapping:
    normal                          → 0 (Normal)
    pns_attack                      → 1 (Under Attack)
    mitm_attack                     → 2 (Under Attack)
    trojan_horse_attack             → 3 (Under Attack)
    wavelength_dependent_trojan_... → 4 (Under Attack)

    Simplified mapping (3 classes):
    normal      → 0 (Normal)
    any_attack  → 2 (Under Attack)

Author  : FYP Team
Module  : src/adaptive/data_loader.py
Project : Hybrid QKD-PQC Security Framework for IoT/RAN
"""

import numpy as np
import pandas as pd
import os
import sys
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config import AI_TEST_SIZE, AI_RANDOM_STATE


# ─────────────────────────────────────────────
#  Label Mapping
# ─────────────────────────────────────────────

# Full 5-class mapping (all attack types separate)
LABEL_MAP_FULL = {
    "normal"                           : 0,
    "pns_attack"                       : 1,
    "mitm_attack"                      : 2,
    "trojan_horse_attack"              : 3,
    "wavelength_dependent_trojan_attack": 4,
}

# Simplified 3-class mapping (matches your system)
LABEL_MAP_SIMPLE = {
    "normal"                           : 0,  # Normal
    "pns_attack"                       : 2,  # Under Attack
    "mitm_attack"                      : 2,  # Under Attack
    "trojan_horse_attack"              : 2,  # Under Attack
    "wavelength_dependent_trojan_attack": 2,  # Under Attack
}

# Class names for reports
CLASS_NAMES_SIMPLE = {0: "Normal", 1: "Degraded", 2: "Under Attack"}
CLASS_NAMES_FULL   = {
    0: "Normal",
    1: "PNS Attack",
    2: "MITM Attack",
    3: "Trojan Horse",
    4: "Wavelength Trojan"
}


# ─────────────────────────────────────────────
#  QKD Dataset Loader
# ─────────────────────────────────────────────

class QKDDataLoader:
    """
    Loads and preprocesses the QKD Attack Detection Dataset.

    Handles:
        - CSV loading
        - Missing value imputation
        - Feature engineering
        - Label encoding
        - Train/test splitting
        - Feature scaling

    Args:
        csv_path      : path to the CSV file
        mode          : "simple" (3 classes) or "full" (5 classes)
        use_all_features : use all 9 features or 6 core features
    """

    # Core 6 features (matching your existing model)
    CORE_FEATURES = [
        "Arrival_var",           # → bit_error_rate
        "Arrival_dev",           # → packet_loss
        "Avg_Photon_time",       # → delay
        "QBER",                  # → qber
        "Sifted_Key_Length",     # → key_length
        "Signal_detection_rate", # → signal_strength
    ]

    # All 9 features (extended model)
    ALL_FEATURES = [
        "Arrival_var",
        "Arrival_dev",
        "Avg_Photon_time",
        "QBER",
        "Sifted_Key_Length",
        "Signal_detection_rate",
        "Measurement_entropy",
        "Decoy_detection_rate",
        "Whole_key_time",
    ]

    # Friendly names for display
    FEATURE_NAMES_CORE = [
        "bit_error_rate",
        "packet_loss",
        "delay_ms",
        "qber",
        "key_length",
        "signal_strength",
    ]

    FEATURE_NAMES_ALL = [
        "bit_error_rate",
        "packet_loss",
        "delay_ms",
        "qber",
        "key_length",
        "signal_strength",
        "measurement_entropy",
        "decoy_detection_rate",
        "whole_key_time",
    ]

    def __init__(
        self,
        csv_path         : str,
        mode             : str  = "simple",
        use_all_features : bool = True
    ):
        self.csv_path         = csv_path
        self.mode             = mode
        self.use_all_features = use_all_features
        self.scaler           = StandardScaler()
        self.df               = None
        self._stats           = {}

        # Select features and names
        if use_all_features:
            self.features      = self.ALL_FEATURES
            self.feature_names = self.FEATURE_NAMES_ALL
        else:
            self.features      = self.CORE_FEATURES
            self.feature_names = self.FEATURE_NAMES_CORE

        # Select label map
        self.label_map  = (
            LABEL_MAP_SIMPLE if mode == "simple"
            else LABEL_MAP_FULL
        )
        self.class_names = (
            CLASS_NAMES_SIMPLE if mode == "simple"
            else CLASS_NAMES_FULL
        )

    def load(self) -> pd.DataFrame:
        """
        Load CSV and perform initial cleaning.

        Returns cleaned DataFrame.
        """
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(
                f"Dataset not found at: {self.csv_path}\n"
                f"Download from: kaggle.com/datasets/"
                f"sattwiksarkar999/qkd-attack-dataset\n"
                f"Place in: data/raw/"
            )

        # Load CSV
        df = pd.read_csv(self.csv_path)
        print(f"  Loaded: {len(df)} rows, {len(df.columns)} columns")

        # ── Check columns ─────────────────────
        missing_cols = [
            c for c in self.features + ["Label"]
            if c not in df.columns
        ]
        if missing_cols:
            print(f"  Available columns: {list(df.columns)}")
            raise ValueError(f"Missing columns: {missing_cols}")

        # ── Handle missing values ─────────────
        before = len(df)
        df     = df.dropna(subset=self.features + ["Label"])
        after  = len(df)
        if before != after:
            print(f"  Dropped {before - after} rows with missing values")

        # ── Normalise label strings ───────────
        df["Label"] = df["Label"].str.strip().str.lower()

        # ── Remove unknown labels ─────────────
        known_labels = set(self.label_map.keys())
        df = df[df["Label"].isin(known_labels)]
        print(f"  Clean dataset: {len(df)} rows")

        self.df = df
        return df

    def get_label_distribution(self) -> dict:
        """Show class distribution in dataset."""
        if self.df is None:
            self.load()

        dist = self.df["Label"].value_counts().to_dict()
        print(f"\n  Label distribution:")
        for label, count in dist.items():
            pct = count / len(self.df) * 100
            bar = "█" * int(pct / 3)
            print(f"  {label:<40} {bar} {count} ({pct:.1f}%)")
        return dist

    def preprocess(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess dataset into features and labels.

        Returns:
            X : feature matrix (n_samples, n_features)
            y : label array   (n_samples,)
        """
        if self.df is None:
            self.load()

        df = self.df.copy()

        # ── Extract features ──────────────────
        X = df[self.features].values.astype(np.float64)

        # ── Encode labels ─────────────────────
        y = np.array([self.label_map[label] for label in df["Label"]])

        # ── Feature engineering ───────────────
        # Clip extreme outliers (keep 99th percentile)
        for i in range(X.shape[1]):
            p99 = np.percentile(X[:, i], 99)
            p01 = np.percentile(X[:, i], 1)
            X[:, i] = np.clip(X[:, i], p01, p99)

        # ── Store stats ───────────────────────
        self._stats = {
            "n_samples"     : len(X),
            "n_features"    : X.shape[1],
            "n_classes"     : len(set(y)),
            "class_dist"    : {
                self.class_names.get(c, str(c)): int(np.sum(y == c))
                for c in np.unique(y)
            },
            "feature_means" : dict(zip(
                self.feature_names, X.mean(axis=0).round(4).tolist()
            )),
            "feature_stds"  : dict(zip(
                self.feature_names, X.std(axis=0).round(4).tolist()
            )),
        }

        return X, y

    def get_train_test_split(
        self,
        scale: bool = True
    ) -> Tuple:
        """
        Get train/test split with optional scaling.

        Returns:
            X_train, X_test, y_train, y_test
        """
        X, y = self.preprocess()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size    = AI_TEST_SIZE,
            random_state = AI_RANDOM_STATE,
            stratify     = y
        )

        if scale:
            X_train = self.scaler.fit_transform(X_train)
            X_test  = self.scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

    def get_single_sample(self, idx: int = 0) -> Tuple:
        """
        Get a single sample for live prediction testing.

        Returns:
            features : feature vector
            label    : true label
        """
        X, y = self.preprocess()
        return X[idx], y[idx]

    @property
    def stats(self) -> dict:
        return self._stats

    @property
    def n_features(self) -> int:
        return len(self.features)


# ─────────────────────────────────────────────
#  TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 58)
    print("  QKD Dataset Loader — Test")
    print("=" * 58)

    # ── Find dataset ──────────────────────────
    # Look in common locations
    possible_paths = [
        "data/raw/qkd_attack_dataset.csv",
        "data/raw/dataset.csv",
        "../../data/raw/qkd_attack_dataset.csv",
        "D:/final year project/hybrid-qkd-pqc/hybrid-qkd-pqc/data/raw/qkd_attack_dataset.csv",
    ]

    csv_path = None
    for path in possible_paths:
        if os.path.exists(path):
            csv_path = path
            break

    if csv_path is None:
        print("\n  Dataset not found in default locations.")
        print("  Please enter the full path to your CSV file:")
        csv_path = input("  Path: ").strip().strip('"')

    print(f"\n  Loading from: {csv_path}")

    # ── Load with all features ────────────────
    print("\n  Mode: Simple (3 classes) + All 9 features")
    print("-" * 58)

    loader = QKDDataLoader(
        csv_path         = csv_path,
        mode             = "simple",
        use_all_features = True
    )

    df = loader.load()
    loader.get_label_distribution()

    X_train, X_test, y_train, y_test = loader.get_train_test_split(scale=True)

    stats = loader.stats
    print(f"\n  Dataset statistics:")
    print(f"  Total samples  : {stats['n_samples']}")
    print(f"  Features       : {stats['n_features']}")
    print(f"  Classes        : {stats['n_classes']}")
    print(f"  Class dist     : {stats['class_dist']}")
    print(f"\n  Split:")
    print(f"  Train samples  : {len(X_train)}")
    print(f"  Test samples   : {len(X_test)}")
    print(f"  Feature shape  : {X_train.shape}")

    print(f"\n  Feature statistics (before scaling):")
    print(f"  {'Feature':<25} {'Mean':>10} {'Std':>10}")
    print("-" * 48)
    for fname, mean, std in zip(
        loader.feature_names,
        stats["feature_means"].values(),
        stats["feature_stds"].values()
    ):
        print(f"  {fname:<25} {mean:>10.4f} {std:>10.4f}")

    # ── Quick model test ──────────────────────
    print(f"\n  Quick model test with real data...")
    print("-" * 58)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score, classification_report

    model  = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1     = f1_score(y_test, y_pred, average="weighted")

    print(f"  F1 Score : {f1*100:.2f}%")
    print()
    print(classification_report(
        y_test, y_pred,
        target_names=[
            loader.class_names[c]
            for c in sorted(loader.class_names.keys())
            if c in np.unique(y_test)
        ]
    ))

    print(f"\n  Dataset loaded and model trained successfully.")
    print(f"  Ready to integrate with agent.py")
