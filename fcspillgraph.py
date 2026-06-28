#!/usr/bin/env python3
"""
fcspillgraph — measure register spill counts in DSP code emitted by
multiple FAUST configurations.

Each (file, faust_params) cell runs:
    faust <params> file.dsp -o tmp.cpp
    fcspilltool tmp.cpp        →  compute_spills, compute_stack, all_spills

and the resulting matrix is printed and optionally plotted.

Usage:
    fcspillgraph.py <pattern> <faust_params1> [<faust_params2> ...] [OPTIONS]

Examples:
    fcspillgraph.py "tests/*.dsp" "-lang cpp" "-lang cpp -vec"
    fcspillgraph.py "*.dsp" "-lang cpp" "-lang cpp -vec" "-lang cpp -vec -fm 0"
    fcspillgraph.py "*.dsp" "-lang cpp" "-lang cpp -vec" --metric compute_stack
"""

import argparse
import glob
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


METRICS = ("compute_spills", "compute_cost", "compute_stack",
           "all_spills", "all_stack", "total")


class FaustSpillMeter:
    def __init__(self):
        # results[filename][config_idx] = {'status': str, 'metrics': dict|None}
        self.results: Dict[str, Dict[int, dict]] = {}
        self.file_list: List[str] = []
        # config_stats[config_idx] = {'total': int, 'count': int}
        self.config_stats: Dict[int, dict] = {}
        self.temp_cpp = "temp_spill.cpp"

    # ------------------------------------------------------------------
    # CLI
    # ------------------------------------------------------------------

    def parse_args(self):
        parser = argparse.ArgumentParser(
            prog="fcspillgraph",
            description="Measure register spill counts in DSP code emitted "
                        "by multiple FAUST configurations.",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  %(prog)s "tests/impulse-tests/dsp/*.dsp" "-lang cpp"
  %(prog)s "*.dsp" "-lang cpp" "-lang cpp -vec"
  %(prog)s "*.dsp" "-lang cpp" "-lang cpp -vec" --metric compute_stack
            """,
        )
        parser.add_argument("file_pattern",
                            help="Pattern of .dsp files to analyse")
        parser.add_argument("faust_configs", nargs="+",
                            help="One or more FAUST parameter sets")
        parser.add_argument("--metric", default="compute_spills",
                            choices=METRICS,
                            help="Which spill metric to plot/aggregate "
                                 "(default: compute_spills)")
        parser.add_argument("--class", dest="dsp_class", default="mydsp",
                            help="DSP class name to filter compute() on "
                                 "(default: mydsp)")
        parser.add_argument("--no-graph", action="store_true",
                            help="Disable graph generation")
        parser.add_argument("--graph-output",
                            help="Graph file name (default: spills_YYYYMMDD_HHMMSS.png)")
        return parser.parse_args()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def find_files(pattern: str) -> List[str]:
        files = glob.glob(pattern)
        return sorted([f for f in files if f.endswith(".dsp")])

    @staticmethod
    def run_command(cmd: List[str], timeout: int = 60) -> Tuple[int, str]:
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout
            )
            return result.returncode, result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return -1, "TIMEOUT"
        except Exception as e:
            return -1, f"ERROR: {e}"

    @staticmethod
    def parse_spill_output(output: str) -> Dict[str, int]:
        """Parse fcspilltool's key:value output."""
        metrics: Dict[str, int] = {}
        for line in output.splitlines():
            if ":" not in line:
                continue
            key, val = line.split(":", 1)
            key, val = key.strip(), val.strip()
            try:
                metrics[key] = int(val)
            except ValueError:
                pass
        return metrics

    # ------------------------------------------------------------------
    # Single-cell measurement
    # ------------------------------------------------------------------

    def measure_file(self, dsp_file: str, config_idx: int,
                     faust_params: str, dsp_class: str
                     ) -> Tuple[str, Optional[Dict[str, int]]]:
        print(f"  → Configuration [{config_idx + 1}]: {faust_params}")

        # Step 1: FAUST compilation
        faust_cmd = ["faust"] + faust_params.split() + [dsp_file,
                                                        "-o", self.temp_cpp]
        ret, out = self.run_command(faust_cmd)
        if ret != 0:
            print(f"    ✗ FAUST compilation error")
            return "FAUST_ERR", None

        # Step 2: fcspilltool
        spill_cmd = ["fcspilltool", "--class", dsp_class, self.temp_cpp]
        ret, out = self.run_command(spill_cmd)
        if ret != 0:
            print(f"    ✗ fcspilltool error")
            try:
                os.remove(self.temp_cpp)
            except OSError:
                pass
            return "COMPILE_ERR", None

        metrics = self.parse_spill_output(out)
        if "compute_spills" not in metrics:
            print(f"    ✗ Cannot parse fcspilltool output")
            return "PARSE_ERR", None

        try:
            os.remove(self.temp_cpp)
        except OSError:
            pass

        print(f"    ✓ compute_spills={metrics['compute_spills']:>4}  "
              f"compute_stack={metrics['compute_stack']:>5}  "
              f"all_spills={metrics.get('all_spills', '?')}")
        return "SUCCESS", metrics

    # ------------------------------------------------------------------
    # Driver
    # ------------------------------------------------------------------

    def run(self, args):
        files = self.find_files(args.file_pattern)
        if not files:
            print(f"No files found for pattern: {args.file_pattern}")
            return

        for i in range(len(args.faust_configs)):
            self.config_stats[i] = {"total": 0, "count": 0}

        print("=== FAUST register-spill measurement ===")
        print(f"File pattern: {args.file_pattern}")
        print(f"DSP class:    {args.dsp_class}")
        print(f"Metric:       {args.metric}")
        print("FAUST parameter sets:")
        for i, cfg in enumerate(args.faust_configs):
            print(f"  [{i+1}] {cfg}")
        print("=" * 40)
        print()

        total_runs = 0
        ok_runs = 0

        for file_idx, dsp_file in enumerate(files):
            basename = Path(dsp_file).stem
            self.file_list.append(basename)
            self.results[basename] = {}

            print(f"[{file_idx + 1}] Measuring: {dsp_file}")

            for ci, params in enumerate(args.faust_configs):
                total_runs += 1
                status, metrics = self.measure_file(
                    dsp_file, ci, params, args.dsp_class
                )
                self.results[basename][ci] = {
                    "status": status, "metrics": metrics,
                }
                if status == "SUCCESS" and metrics is not None:
                    ok_runs += 1
                    self.config_stats[ci]["total"] += metrics.get(args.metric, 0)
                    self.config_stats[ci]["count"] += 1
            print()

        self.display_results(args, len(files), total_runs, ok_runs)

        if not args.no_graph:
            self.generate_graph(args)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def display_results(self, args, total_files: int,
                        total_runs: int, ok_runs: int):
        metric = args.metric
        print("=" * 60)
        print(f"=== RESULTS MATRIX ({metric}) ===")
        print("=" * 60)

        print(f"{'File':<25}", end="")
        for i in range(len(args.faust_configs)):
            print(f" | {'Config' + str(i+1):>12}", end="")
        print()
        print("=" * 25, end="")
        for _ in range(len(args.faust_configs)):
            print(" | " + "=" * 12, end="")
        print()

        for fname in self.file_list:
            print(f"{fname:<25}", end="")
            for ci in range(len(args.faust_configs)):
                cell = self.results[fname].get(ci, {"status": "MISSING", "metrics": None})
                if cell["status"] == "SUCCESS" and cell["metrics"] is not None:
                    val = cell["metrics"].get(metric, 0)
                    print(f" | {val:>12}", end="")
                else:
                    print(f" | {cell['status']:>12}", end="")
            print()
        print()

        print("=== CONFIGURATIONS ===")
        for i, cfg in enumerate(args.faust_configs):
            print(f"Config{i+1}: {cfg}")
        print()

        print(f"=== STATISTICS PER CONFIGURATION ({metric}) ===")
        for i, cfg in enumerate(args.faust_configs):
            s = self.config_stats[i]
            print(f"Config{i+1} ({cfg}):")
            print(f"  - Successful runs : {s['count']}/{total_files}")
            if s["count"] > 0:
                avg = s["total"] / s["count"]
                print(f"  - Total {metric:<14}: {s['total']}")
                print(f"  - Mean  {metric:<14}: {avg:.2f}")
            else:
                print(f"  - No successful runs")
            print()

        print("=== GLOBAL ===")
        print(f"Files                    : {total_files}")
        print(f"Configurations           : {len(args.faust_configs)}")
        print(f"Runs attempted           : {total_runs}")
        print(f"Successful runs          : {ok_runs}")
        if total_runs:
            print(f"Global success rate      : {ok_runs * 100 // total_runs}%")
        print("=" * 60)

    # ------------------------------------------------------------------
    # Graph
    # ------------------------------------------------------------------

    def generate_graph(self, args):
        if not MATPLOTLIB_AVAILABLE:
            print("\n⚠️  matplotlib is not installed. Graph not generated.")
            print("   To install: pip install matplotlib")
            return
        if not self.file_list:
            print("\n⚠️  No data to graph.")
            return

        out = args.graph_output
        if out is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out = f"spills_{ts}.png"

        print(f"\n=== GRAPH GENERATION ===")
        print(f"Output file: {out}")

        _, ax = plt.subplots(figsize=(12, 8))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

        for ci in range(len(args.faust_configs)):
            xs, ys = [], []
            for file_idx, fname in enumerate(self.file_list):
                cell = self.results[fname].get(ci, {})
                if cell.get("status") == "SUCCESS" and cell.get("metrics"):
                    xs.append(file_idx)
                    ys.append(cell["metrics"].get(args.metric, 0))
            if ys:
                ax.plot(xs, ys, marker='o', linestyle='-', linewidth=2,
                        markersize=6, color=colors[ci % len(colors)],
                        label=f"Config{ci + 1}")

        ax.set_xlabel('DSP Files', fontsize=12, fontweight='bold')
        ax.set_ylabel(args.metric, fontsize=12, fontweight='bold')
        ax.set_title(f'Register spill comparison ({args.metric})',
                     fontsize=14, fontweight='bold', pad=40)

        ax.set_xticks(range(len(self.file_list)))
        ax.set_xticklabels(self.file_list, rotation=45, ha='right', fontsize=9)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.4)

        # Command used as subtitle
        script = sys.argv[0] if sys.argv else "fcspillgraph.py"
        cmd_parts = [f'"{script}"', f'"{args.file_pattern}"']
        for cfg in args.faust_configs:
            cmd_parts.append(f'"{cfg}"')
        if args.metric != "compute_spills":
            cmd_parts.append(f"--metric {args.metric}")
        if args.dsp_class != "mydsp":
            cmd_parts.append(f"--class {args.dsp_class}")
        cmd_str = " ".join(cmd_parts)
        if len(cmd_str) > 100:
            cmd_str = cmd_str[:97] + "..."
        ax.text(0.5, 1.02, cmd_str, transform=ax.transAxes,
                ha='center', va='bottom', fontsize=9,
                style='italic', color='dimgray')

        plt.tight_layout()
        plt.savefig(out, dpi=120)
        plt.close()
        print(f"✓ Graph saved to {out}")


def main():
    meter = FaustSpillMeter()
    args = meter.parse_args()
    meter.run(args)


if __name__ == "__main__":
    main()
