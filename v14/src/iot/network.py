# -*- coding: utf-8 -*-
"""
network.py — RAN Network Simulation
=====================================
Realistic Radio Access Network simulation for
urban IoT environments in a smart city.

Improvements over basic channel model:
    - SNR (Signal-to-Noise Ratio) modelling
    - Multipath fading (Rayleigh fading)
    - Distance-based path loss (Free Space + Urban)
    - Doppler effect for moving IoT devices
    - Interference from neighbouring cells
    - Realistic 5G NR channel parameters
    - Multiple IoT device simulation

Channel Models:
    AWGN     : Additive White Gaussian Noise (baseline)
    Rayleigh : Multipath fading (urban environment)
    Urban    : Full urban IoT model with all effects

Author  : FYP Team
Module  : src/iot/network.py
Project : Hybrid QKD-PQC Security Framework for IoT/RAN
"""

import numpy as np
import time
from typing import Optional
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config import (
    RAN_NOISE_NORMAL, RAN_NOISE_DEGRADED, RAN_NOISE_ATTACK,
    RAN_LOSS_NORMAL,  RAN_LOSS_DEGRADED,  RAN_LOSS_ATTACK,
    RAN_DELAY_NORMAL, RAN_DELAY_DEGRADED, RAN_DELAY_ATTACK
)


# ─────────────────────────────────────────────
#  Basic RAN Channel (AWGN)
# ─────────────────────────────────────────────

class RANChannel:
    """
    Basic RAN channel with AWGN noise model.

    Models:
        - Random bit flips (noise)
        - Packet loss
        - Transmission delay

    Args:
        noise_level : bit flip probability (0.0 - 1.0)
        packet_loss : bit loss probability  (0.0 - 1.0)
        delay_ms    : transmission delay in ms
    """

    def __init__(
        self,
        noise_level : float = RAN_NOISE_NORMAL,
        packet_loss : float = RAN_LOSS_NORMAL,
        delay_ms    : int   = RAN_DELAY_NORMAL
    ):
        self.noise_level = noise_level
        self.packet_loss = packet_loss
        self.delay_ms    = delay_ms
        self.channel_type = "AWGN"

    def transmit(self, bits: list) -> list:
        """Transmit bits through noisy channel."""
        received = []
        for bit in bits:
            if np.random.random() < self.packet_loss:
                received.append(None)
                continue
            if np.random.random() < self.noise_level:
                received.append(1 - bit)
            else:
                received.append(bit)
        return received

    def get_stats(self, original: list, received: list) -> dict:
        """Compare sent vs received bits."""
        total  = len(original)
        lost   = received.count(None)
        errors = sum(
            1 for o, r in zip(original, received)
            if r is not None and o != r
        )
        rate = errors / (total - lost) if (total - lost) > 0 else 0
        return {
            "total"       : total,
            "lost"        : lost,
            "errors"      : errors,
            "error_rate"  : round(rate * 100, 2),
            "loss_rate"   : round(lost / total * 100, 2),
            "delay_ms"    : self.delay_ms,
            "channel_type": self.channel_type
        }


# ─────────────────────────────────────────────
#  Rayleigh Fading Channel
# ─────────────────────────────────────────────

class RayleighChannel(RANChannel):
    """
    Rayleigh fading channel for urban multipath environments.

    In urban IoT environments signals bounce off buildings,
    vehicles, and other obstacles — creating multiple signal
    paths that interfere with each other. This is modelled
    using Rayleigh fading.

    Additional parameters:
        doppler_hz   : Doppler frequency shift (moving devices)
        coherence_ms : channel coherence time
    """

    def __init__(
        self,
        noise_level  : float = RAN_NOISE_NORMAL,
        packet_loss  : float = RAN_LOSS_NORMAL,
        delay_ms     : int   = RAN_DELAY_NORMAL,
        doppler_hz   : float = 10.0,
        coherence_ms : float = 50.0
    ):
        super().__init__(noise_level, packet_loss, delay_ms)
        self.doppler_hz   = doppler_hz
        self.coherence_ms = coherence_ms
        self.channel_type = "Rayleigh"

    def _rayleigh_fading_factor(self) -> float:
        """
        Generate Rayleigh fading coefficient.
        Models random signal amplitude variation.
        """
        # Rayleigh distribution — envelope of two Gaussian signals
        real = np.random.normal(0, 1)
        imag = np.random.normal(0, 1)
        amplitude = np.sqrt(real**2 + imag**2) / np.sqrt(2)
        return amplitude

    def transmit(self, bits: list) -> list:
        """Transmit through Rayleigh fading channel."""
        received = []
        for bit in bits:
            # Packet loss
            if np.random.random() < self.packet_loss:
                received.append(None)
                continue

            # Rayleigh fading — signal amplitude varies
            fading = self._rayleigh_fading_factor()

            # Effective noise increases when signal fades
            effective_noise = self.noise_level / (fading + 1e-6)
            effective_noise = min(effective_noise, 0.5)

            if np.random.random() < effective_noise:
                received.append(1 - bit)
            else:
                received.append(bit)

        return received


# ─────────────────────────────────────────────
#  Urban IoT Channel (Full Model)
# ─────────────────────────────────────────────

class UrbanIoTChannel(RayleighChannel):
    """
    Full urban IoT channel model for smart city RAN.

    Combines:
        - AWGN noise
        - Rayleigh multipath fading
        - Distance-based path loss
        - Interference from neighbouring base stations
        - 5G NR realistic parameters

    Path loss model (Urban Macro — 3GPP TR 38.901):
        PL = 28.0 + 22*log10(d) + 20*log10(fc)

    Args:
        distance_m   : distance between IoT device and base station (m)
        frequency_ghz: carrier frequency in GHz (5G NR: 3.5 GHz typical)
        tx_power_dbm : transmit power in dBm
        interference : interference level from other cells (0.0-1.0)
    """

    # 5G NR typical parameters
    FREQUENCY_GHZ   = 3.5      # 5G NR mid-band
    TX_POWER_DBM    = 23.0     # typical IoT device tx power
    NOISE_FIGURE_DB = 7.0      # receiver noise figure
    BANDWIDTH_MHZ   = 20.0     # channel bandwidth

    def __init__(
        self,
        distance_m    : float = 100.0,
        frequency_ghz : float = 3.5,
        tx_power_dbm  : float = 23.0,
        interference  : float = 0.05,
        doppler_hz    : float = 10.0,
        scenario      : str   = "normal"
    ):
        self.distance_m    = distance_m
        self.frequency_ghz = frequency_ghz
        self.tx_power_dbm  = tx_power_dbm
        self.interference  = interference
        self.scenario      = scenario
        self.channel_type  = "Urban5G"

        # Calculate channel parameters from physics
        noise_level, packet_loss, delay_ms = self._calculate_params()

        super().__init__(
            noise_level  = noise_level,
            packet_loss  = packet_loss,
            delay_ms     = delay_ms,
            doppler_hz   = doppler_hz
        )

    def _calculate_path_loss(self) -> float:
        """
        Calculate urban path loss using 3GPP model.
        Returns path loss in dB.
        """
        d  = max(self.distance_m, 1.0)
        fc = self.frequency_ghz

        # 3GPP TR 38.901 Urban Macro path loss
        path_loss_db = (
            28.0 +
            22.0 * np.log10(d) +
            20.0 * np.log10(fc)
        )
        return path_loss_db

    def _calculate_snr(self) -> float:
        """Calculate received SNR in dB."""
        path_loss_db  = self._calculate_path_loss()
        received_power = self.tx_power_dbm - path_loss_db

        # Thermal noise power
        thermal_noise = -174 + 10 * np.log10(
            self.BANDWIDTH_MHZ * 1e6
        ) + self.NOISE_FIGURE_DB

        snr_db = received_power - thermal_noise
        return snr_db

    def _snr_to_ber(self, snr_db: float) -> float:
        """
        Convert SNR to Bit Error Rate using QPSK model.
        BER = 0.5 * erfc(sqrt(SNR_linear))
        """
        snr_linear = 10 ** (snr_db / 10)
        # Approximation of erfc for QPSK
        ber = 0.5 * np.exp(-snr_linear)
        return min(max(ber, 0.001), 0.5)

    def _calculate_params(self) -> tuple:
        """Calculate channel parameters from physical model."""
        snr_db      = self._calculate_snr()
        base_ber    = self._snr_to_ber(snr_db)

        # Add interference
        effective_ber = base_ber + self.interference * 0.05

        # Packet loss from distance and interference
        packet_loss = min(
            self.distance_m / 5000 + self.interference * 0.1,
            0.3
        )

        # Delay from distance (speed of light) + processing
        prop_delay = (self.distance_m / 3e8) * 1000  # ms
        proc_delay = 5.0 + self.interference * 20     # ms
        delay_ms   = int(prop_delay + proc_delay)

        return effective_ber, packet_loss, max(delay_ms, 1)

    def get_channel_quality(self) -> dict:
        """Return full channel quality report."""
        snr_db = self._calculate_snr()
        return {
            "scenario"        : self.scenario,
            "distance_m"      : self.distance_m,
            "frequency_ghz"   : self.frequency_ghz,
            "path_loss_db"    : round(self._calculate_path_loss(), 2),
            "snr_db"          : round(snr_db, 2),
            "noise_level"     : round(self.noise_level, 4),
            "packet_loss"     : round(self.packet_loss, 4),
            "delay_ms"        : self.delay_ms,
            "doppler_hz"      : self.doppler_hz,
            "interference"    : self.interference,
            "channel_type"    : self.channel_type
        }


# ─────────────────────────────────────────────
#  IoT Network — Multiple Devices
# ─────────────────────────────────────────────

class IoTNetwork:
    """
    Simulates a smart city IoT network with multiple
    devices communicating over RAN to a base station.

    Device types:
        traffic_sensor  : fixed, low mobility
        smart_meter     : fixed, indoor
        camera          : fixed, outdoor
        mobile_sensor   : moving (higher Doppler)
        hospital_device : critical, highest priority
    """

    DEVICE_PROFILES = {
        "traffic_sensor" : {"distance_m": 50,  "doppler_hz": 0,   "priority": 2},
        "smart_meter"    : {"distance_m": 30,  "doppler_hz": 0,   "priority": 1},
        "camera"         : {"distance_m": 80,  "doppler_hz": 0,   "priority": 2},
        "mobile_sensor"  : {"distance_m": 150, "doppler_hz": 50,  "priority": 2},
        "hospital_device": {"distance_m": 200, "doppler_hz": 0,   "priority": 3},
    }

    def __init__(self, num_devices: int = 10, scenario: str = "normal"):
        self.num_devices = num_devices
        self.scenario    = scenario
        self.devices     = self._create_devices()

    def _create_devices(self) -> list:
        """Create IoT devices with realistic profiles."""
        device_types = list(self.DEVICE_PROFILES.keys())
        devices      = []

        interference = {
            "normal" : 0.02,
            "degraded": 0.08,
            "attack" : 0.15
        }.get(self.scenario, 0.02)

        for i in range(self.num_devices):
            device_type = device_types[i % len(device_types)]
            profile     = self.DEVICE_PROFILES[device_type]

            # Add some randomness to distance
            distance = profile["distance_m"] * np.random.uniform(0.8, 1.2)

            channel = UrbanIoTChannel(
                distance_m   = distance,
                interference = interference,
                doppler_hz   = profile["doppler_hz"],
                scenario     = self.scenario
            )

            devices.append({
                "id"      : f"{device_type}_{i:03d}",
                "type"    : device_type,
                "priority": profile["priority"],
                "channel" : channel,
                "distance": round(distance, 1)
            })

        return devices

    def get_network_stats(self) -> dict:
        """Get aggregated network statistics."""
        all_snr      = []
        all_ber      = []
        all_delay    = []
        all_loss     = []

        for device in self.devices:
            ch = device["channel"]
            q  = ch.get_channel_quality()
            all_snr.append(q["snr_db"])
            all_ber.append(q["noise_level"])
            all_delay.append(q["delay_ms"])
            all_loss.append(q["packet_loss"])

        return {
            "num_devices"   : self.num_devices,
            "scenario"      : self.scenario,
            "avg_snr_db"    : round(np.mean(all_snr), 2),
            "avg_ber"       : round(np.mean(all_ber), 4),
            "avg_delay_ms"  : round(np.mean(all_delay), 2),
            "avg_loss"      : round(np.mean(all_loss), 4),
            "min_snr_db"    : round(min(all_snr), 2),
            "max_snr_db"    : round(max(all_snr), 2),
        }


# ─────────────────────────────────────────────
#  TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 58)
    print("  Urban IoT RAN Network Simulation")
    print("=" * 58)

    # ── Test 1: Channel quality at distances ──
    print("\n  Channel quality vs distance (5G NR, 3.5GHz)")
    print("-" * 58)
    print(f"  {'Distance':>10} {'Path Loss':>12} {'SNR':>8} "
          f"{'BER':>8} {'Delay':>8}")
    print("-" * 58)

    for d in [10, 50, 100, 200, 500, 1000]:
        ch = UrbanIoTChannel(distance_m=d, interference=0.02)
        q  = ch.get_channel_quality()
        print(f"  {d:>8}m  {q['path_loss_db']:>10}dB "
              f"{q['snr_db']:>7}dB "
              f"{q['noise_level']:>8.4f} "
              f"{q['delay_ms']:>6}ms")

    # ── Test 2: Three scenarios ───────────────
    print("\n  Three channel scenarios")
    print("-" * 58)

    scenarios = {
        "Normal"      : {"interference": 0.02, "distance_m": 100},
        "Degraded"    : {"interference": 0.10, "distance_m": 300},
        "Under Attack": {"interference": 0.20, "distance_m": 500},
    }

    bits = list(np.random.randint(0, 2, 1000))

    for name, params in scenarios.items():
        ch   = UrbanIoTChannel(**params)
        recv = ch.transmit(bits)
        s    = ch.get_stats(bits, recv)
        q    = ch.get_channel_quality()
        print(f"\n  {name}")
        print(f"  SNR: {q['snr_db']}dB | "
              f"BER: {s['error_rate']}% | "
              f"Loss: {s['loss_rate']}% | "
              f"Delay: {s['delay_ms']}ms")

    # ── Test 3: IoT Network ───────────────────
    print("\n\n  Smart city IoT network simulation")
    print("-" * 58)

    for scenario in ["normal", "degraded", "attack"]:
        net   = IoTNetwork(num_devices=10, scenario=scenario)
        stats = net.get_network_stats()
        print(f"\n  Scenario     : {scenario.upper()}")
        print(f"  Devices      : {stats['num_devices']}")
        print(f"  Avg SNR      : {stats['avg_snr_db']} dB")
        print(f"  Avg BER      : {stats['avg_ber']:.4f}")
        print(f"  Avg Delay    : {stats['avg_delay_ms']} ms")
        print(f"  Avg Loss     : {stats['avg_loss']:.4f}")
        print(f"  SNR range    : {stats['min_snr_db']} — "
              f"{stats['max_snr_db']} dB")