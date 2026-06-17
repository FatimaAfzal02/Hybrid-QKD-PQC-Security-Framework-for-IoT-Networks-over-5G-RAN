# -*- coding: utf-8 -*-
"""
logger.py — Results Logger
===========================
Saves all system results to results/logs/
in both JSON and human-readable TXT format.

Author  : FYP Team
Module  : src/adaptive/logger.py
Project : Hybrid QKD-PQC Security Framework for IoT/RAN
"""

import json
import os
import time
from datetime import datetime
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config import RESULTS_LOGS_PATH


class SystemLogger:
    """
    Logs all module results to files in results/logs/

    Creates two files per run:
        run_YYYYMMDD_HHMMSS.json — machine readable
        run_YYYYMMDD_HHMMSS.txt  — human readable report
    """

    def __init__(self):
        os.makedirs(RESULTS_LOGS_PATH, exist_ok=True)
        self.timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_data   = {
            "run_timestamp"  : self.timestamp,
            "run_datetime"   : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "modules"        : {}
        }
        self.txt_lines  = []

    def log(self, module_name: str, data: dict):
        """Log a module's results."""
        self.log_data["modules"][module_name] = data

        # Make JSON serializable
        clean = self._make_serializable(data)
        self.log_data["modules"][module_name] = clean

    def _make_serializable(self, obj):
        """Convert non-serializable types to serializable."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(i) for i in obj]
        elif isinstance(obj, bytes):
            return obj.hex()
        elif hasattr(obj, 'item'):
            return obj.item()
        elif isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj
        else:
            return str(obj)

    def save(self, summary: dict = None, total_time: float = 0):
        """Save logs to files."""
        if summary:
            self.log_data["evaluation_summary"] = summary
        self.log_data["total_runtime_s"] = round(total_time, 2)

        # ── Save JSON ─────────────────────────
        json_path = os.path.join(
            RESULTS_LOGS_PATH, f"run_{self.timestamp}.json"
        )
        with open(json_path, "w") as f:
            json.dump(self.log_data, f, indent=2)

        # ── Save TXT report ───────────────────
        txt_path = os.path.join(
            RESULTS_LOGS_PATH, f"run_{self.timestamp}.txt"
        )
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(self._build_report(summary, total_time))

        print(f"\n  Logs saved:")
        print(f"  JSON : {json_path}")
        print(f"  TXT  : {txt_path}")

        return json_path, txt_path

    def _build_report(self, summary, total_time) -> str:
        lines = []
        lines.append("=" * 65)
        lines.append("  HYBRID QKD-PQC SYSTEM — RUN REPORT")
        lines.append(f"  Date  : {self.log_data['run_datetime']}")
        lines.append("=" * 65)

        # Module results
        for module, data in self.log_data["modules"].items():
            lines.append(f"\n  [{module}]")
            for k, v in data.items():
                if not isinstance(v, (dict, list)):
                    lines.append(f"    {k:<25}: {v}")

        # Evaluation summary
        if summary:
            lines.append(f"\n  [EVALUATION SUMMARY]")
            lines.append(f"  {'System':<18} {'Detection':>12} "
                         f"{'Key Rate':>10} {'Latency':>10}")
            lines.append("  " + "-" * 52)
            for system, s in summary.items():
                marker = " ◄" if system == "Full Hybrid" else ""
                lines.append(
                    f"  {system:<18} "
                    f"{s['detection_rate']:>11}% "
                    f"{s['avg_key_rate']:>10.0f} "
                    f"{s['avg_latency']:>9.1f}ms{marker.replace(chr(9668), '<--')}"
                )

        lines.append(f"\n  Total runtime : {round(total_time, 2)}s")
        lines.append("=" * 65)
        return "\n".join(lines)
