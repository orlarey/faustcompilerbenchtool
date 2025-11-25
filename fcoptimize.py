#!/usr/bin/env python3
"""
fcoptimize - Automatic optimization tool to find the best Faust compilation options
Usage: fcoptimize.py <dsp_file> [OPTIONS]
"""

import argparse
import glob
import json
import os
import random
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class FaustOptionSpace:
    """Defines the space of coherent Faust compilation options for scalar modes."""

    def __init__(self, lang: str = 'cpp'):
        self.lang = lang

        # Define option categories with their possible values
        # FOCUSED ON SCALAR MODES ONLY (no vectorization, no parallelism)
        # FOCUSED ON PERFORMANCE (not precision, so only -single)
        self.options = {
            'max_copy_delay': {
                'values': [0, 2, 4, 6, 8, 10, 12, 16, 20] if lang == 'ocpp' else [0, 2, 4, 6, 8, 9, 12, 16, 20],
                'default': 9,
                'flag': '-mcd',
                'description': 'Max copy delay threshold'
            },
            'use_dense_delay': {
                'values': [0, 1],
                'default': 1,
                'flag': '-udd',
                'description': 'Allow use of dense delay instead of short ring buffers'
            },
            'max_copy_loop': {
                'values': [2, 4, 8, 16] if lang == 'ocpp' else [4],
                'default': 4,
                'flag': '-mcl',
                'description': 'Threshold to switch from inline to loop based copy'
            },
            'max_dense_delay': {
                'values': [256, 512, 1024, 2048, 4096] if lang == 'ocpp' else [1024],
                'default': 1024,
                'flag': '-mdd',
                'description': 'Max dense delay threshold (ocpp)'
            },
            'max_cache_delay': {
                'values': [4, 8, 16, 32] if lang == 'ocpp' else [8],
                'default': 8,
                'flag': '-mca',
                'description': 'Max cache delay (ocpp)'
            },
            'min_density': {
                'values': [70, 80, 90, 95] if lang == 'ocpp' else [90],
                'default': 90,
                'flag': '-mdy',
                'description': 'Min density for dense delays (ocpp)'
            },
            'scheduling_strategy': {
                'values': [0, 1, 2, 3],
                'default': 0,
                'flag': '-ss',
                'description': 'Scheduling strategy (0=depth-first, 1=breadth-first, 2=special, 3+=reverse breadth-first)'
            },
            'fixed_sample_rate': {
                'values': [None, 44100],
                'default': None,
                'flag': '-fsr',
                'description': 'Fixed sample rate (allows compile-time optimizations)'
            },
            'compute_mix': {
                'values': [None, '-cm'] if lang == 'cpp' else [None],
                'default': None,
                'exclusive': True,
                'description': 'Mix in output buffers (cpp only)'
            },
            'fast_math': {
                'values': [None, 'def'] if lang == 'cpp' else [None],
                'default': None,
                'flag': '-fm',
                'description': 'Fast math optimizations (cpp only)'
            },
            'math_approximation': {
                'values': [None, '-mapp'],
                'default': None,
                'exclusive': True,
                'description': 'Math function approximations'
            },
            'exp10': {
                'values': [None, '-exp10'],
                'default': None,
                'exclusive': True,
                'description': 'Replace pow(10,x) with exp10(x)'
            },
            'inline_table': {
                'values': [None, '-it'] if lang == 'cpp' else [None],
                'default': None,
                'exclusive': True,
                'description': 'Inline rdtable/rwtable code (cpp only)'
            },
            'fir_iir': {
                'values': [None, '-fir'],
                'default': None,
                'exclusive': True,
                'description': 'Activate reconstruction of FIRs and IIRs internally'
            },
            'factorize_fir_iir': {
                'values': [None, '-ff'],
                'default': None,
                'exclusive': True,
                'description': 'Find common factor in FIRs or IIRs coefficients'
            },
            'max_fir_size': {
                'values': [256, 512, 1024, 2048],
                'default': 1024,
                'flag': '-mfs',
                'description': 'Max size threshold to reconstruct a FIR'
            },
            'fir_loop_size': {
                'values': [2, 4, 8, 16],
                'default': 4,
                'flag': '-fls',
                'description': 'Size threshold for FIR loop vs unrolled'
            },
            'iir_ring_threshold': {
                'values': [2, 4, 8, 16],
                'default': 4,
                'flag': '-irt',
                'description': 'Size threshold for IIR ring buffers vs copying'
            },
            'simplify_select2': {
                'values': [None, '-ssel'],
                'default': None,
                'exclusive': True,
                'description': 'Apply select2 simplifications based on type/interval analysis'
            }
        }

    def generate_random_config(self) -> Dict[str, Any]:
        """Generate a random but coherent configuration."""
        config = {}

        # First pass: select values for all options
        for opt_name, opt_def in self.options.items():
            if opt_def.get('exclusive', False):
                # Choose one value or None
                config[opt_name] = random.choice(opt_def['values'])
            else:
                # For non-exclusive options with flag (numeric values)
                if 'flag' in opt_def:
                    config[opt_name] = random.choice(opt_def['values'])
                else:
                    config[opt_name] = random.choice(opt_def['values'])

        # Second pass: apply constraints
        config = self._apply_constraints(config)

        return config

    def _apply_constraints(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply constraints to ensure coherent configuration.

        For scalar modes, constraints are minimal since we removed
        vectorization and parallelism options.
        """
        # No complex constraints needed for scalar mode
        # All options are independent and compatible
        return config

    def config_to_string(self, config: Dict[str, Any]) -> str:
        """Convert configuration dict to Faust command-line arguments."""
        args = [f'-lang {self.lang}']

        for opt_name, value in sorted(config.items()):
            if value is None:
                continue

            opt_def = self.options.get(opt_name)
            if not opt_def:
                continue

            if 'flag' in opt_def:
                # Options with flags and values (e.g., -vs 32)
                if value == 'def':  # Special case for -fm def
                    args.append(f"{opt_def['flag']} {value}")
                else:
                    args.append(f"{opt_def['flag']} {value}")
            elif isinstance(value, str):
                # String flags (e.g., -vec, -omp)
                args.append(value)

        return ' '.join(args)


class FaustOptimizer:
    """Main optimizer class."""

    def __init__(self):
        self.results = []  # List of (config_str, time, config_dict)
        self.temp_cpp = "temp_optimize.cpp"
        self.option_space = None

    def parse_args(self):
        parser = argparse.ArgumentParser(
            prog="fcoptimize",
            description="Find the best Faust compilation options automatically",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  %(prog)s foo.dsp --iterations 100
  %(prog)s foo.dsp --lang ocpp --strategy adaptive --max-trials 200
  %(prog)s foo.dsp --iterations 50 --save-results results.json

The script explores different Faust compilation options to find
the configuration that produces the fastest executable.
            """
        )

        parser.add_argument('dsp_file',
                          help='Faust DSP file to optimize')
        parser.add_argument('--lang', choices=['cpp', 'ocpp'], default='cpp',
                          help='Target language (default: cpp)')
        parser.add_argument('--strategy', choices=['random', 'adaptive'],
                          default='random',
                          help='Search strategy (default: random)')
        parser.add_argument('--max-trials', type=int, default=100,
                          help='Maximum number of configurations to try (default: 100)')
        parser.add_argument('--iterations', type=int, default=1000,
                          help='Benchmark iterations per configuration (default: 1000)')
        parser.add_argument('--top-n', type=int, default=10,
                          help='Show top N best configurations (default: 10)')
        parser.add_argument('--save-results',
                          help='Save detailed results to JSON file')
        parser.add_argument('--graph-output',
                          help='Generate optimization progress graph')
        parser.add_argument('--baseline',
                          help='Baseline configuration to compare against (e.g., "-lang cpp")')
        parser.add_argument('--timeout', type=int, default=60,
                          help='Timeout for each benchmark in seconds (default: 60)')
        parser.add_argument('--sensitivity-analysis', action='store_true',
                          help='Perform sensitivity analysis on best configuration')

        return parser.parse_args()

    def run_command(self, cmd: List[str], timeout: int = 60) -> Tuple[int, str]:
        """Execute a command and return the return code and output."""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return result.returncode, result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return -1, "TIMEOUT"
        except Exception as e:
            return -1, f"ERROR: {str(e)}"

    def extract_benchmark_time(self, output: str) -> Optional[float]:
        """Extract benchmark time from output."""
        matches = re.findall(r'\d+(?:\.\d+)?', output)
        if matches:
            try:
                return float(matches[-1])
            except ValueError:
                return None
        return None

    def benchmark_config(self, dsp_file: str, config_str: str,
                        iterations: int, timeout: int) -> Optional[float]:
        """Benchmark a specific configuration."""

        # Step 1: FAUST compilation
        faust_cmd = f"faust {config_str} {dsp_file} -o {self.temp_cpp}".split()
        ret_code, output = self.run_command(faust_cmd, timeout)

        if ret_code != 0:
            return None

        # Step 2: Compilation with fcbenchtool
        bench_binary = "temp_optimize.bench"
        fcbench_cmd = ["fcbenchtool", self.temp_cpp, ".bench"]
        ret_code, output = self.run_command(fcbench_cmd, timeout)

        if ret_code != 0:
            return None

        if not os.path.exists(bench_binary):
            return None

        # Step 3: Benchmark execution
        if iterations != 1000:
            bench_cmd = [f"./{bench_binary}", str(iterations)]
        else:
            bench_cmd = [f"./{bench_binary}"]

        ret_code, output = self.run_command(bench_cmd, timeout)

        # Cleanup
        try:
            if os.path.exists(bench_binary):
                os.remove(bench_binary)
            if os.path.exists(self.temp_cpp):
                os.remove(self.temp_cpp)
        except:
            pass

        if ret_code != 0:
            return None

        return self.extract_benchmark_time(output)

    def optimize_random(self, args):
        """Random search strategy."""
        print(f"\n=== RANDOM SEARCH OPTIMIZATION ===")
        print(f"DSP file: {args.dsp_file}")
        print(f"Language: {args.lang}")
        print(f"Max trials: {args.max_trials}")
        print(f"Benchmark iterations: {args.iterations}")
        print("=" * 50)
        print()

        self.option_space = FaustOptionSpace(args.lang)

        # Baseline if provided
        baseline_time = None
        if args.baseline:
            print(f"Testing baseline configuration: {args.baseline}")
            baseline_time = self.benchmark_config(
                args.dsp_file, args.baseline, args.iterations, args.timeout
            )
            if baseline_time:
                print(f"  Baseline: {baseline_time:.3f}ms")
                self.results.append((args.baseline, baseline_time, {}))
            else:
                print(f"  Baseline: FAILED")
            print()

        # Random exploration
        successful = 0
        failed = 0
        best_time = float('inf')
        best_config = None

        for trial in range(args.max_trials):
            # Generate random configuration
            config_dict = self.option_space.generate_random_config()
            config_str = self.option_space.config_to_string(config_dict)

            # Skip if already tested
            if any(c[0] == config_str for c in self.results):
                continue

            print(f"[{trial + 1}/{args.max_trials}] Testing: {config_str}")

            bench_time = self.benchmark_config(
                args.dsp_file, config_str, args.iterations, args.timeout
            )

            if bench_time is not None:
                successful += 1
                self.results.append((config_str, bench_time, config_dict))

                improvement = ""
                if baseline_time:
                    speedup = (baseline_time / bench_time - 1) * 100
                    improvement = f" ({speedup:+.1f}% vs baseline)"

                if bench_time < best_time:
                    best_time = bench_time
                    best_config = config_str
                    print(f"  Result: {bench_time:.3f}ms ✓ NEW BEST!{improvement}")
                else:
                    print(f"  Result: {bench_time:.3f}ms{improvement}")
            else:
                failed += 1
                print(f"  Result: FAILED")

            # Show progress every 10 trials
            if (trial + 1) % 10 == 0:
                print(f"\n--- Progress: {successful} successful, {failed} failed ---")
                if best_config:
                    print(f"Current best: {best_time:.3f}ms")
                print()

        return baseline_time

    def optimize_adaptive(self, args):
        """Adaptive search strategy - focuses on promising regions."""
        print(f"\n=== ADAPTIVE SEARCH OPTIMIZATION ===")
        print(f"DSP file: {args.dsp_file}")
        print(f"Language: {args.lang}")
        print(f"Max trials: {args.max_trials}")
        print(f"Benchmark iterations: {args.iterations}")
        print("=" * 50)
        print()

        self.option_space = FaustOptionSpace(args.lang)

        # Baseline if provided
        baseline_time = None
        if args.baseline:
            print(f"Testing baseline configuration: {args.baseline}")
            baseline_time = self.benchmark_config(
                args.dsp_file, args.baseline, args.iterations, args.timeout
            )
            if baseline_time:
                print(f"  Baseline: {baseline_time:.3f}ms")
                self.results.append((args.baseline, baseline_time, {}))
            else:
                print(f"  Baseline: FAILED")
            print()

        # Phase 1: Random exploration (first 30%)
        exploration_trials = max(10, args.max_trials // 3)
        print(f"Phase 1: Random exploration ({exploration_trials} trials)")
        print("-" * 50)

        for trial in range(exploration_trials):
            config_dict = self.option_space.generate_random_config()
            config_str = self.option_space.config_to_string(config_dict)

            if any(c[0] == config_str for c in self.results):
                continue

            print(f"[{trial + 1}/{exploration_trials}] Testing: {config_str}")

            bench_time = self.benchmark_config(
                args.dsp_file, config_str, args.iterations, args.timeout
            )

            if bench_time is not None:
                self.results.append((config_str, bench_time, config_dict))
                print(f"  Result: {bench_time:.3f}ms")
            else:
                print(f"  Result: FAILED")

        # Phase 2: Adaptive refinement (remaining 70%)
        print(f"\nPhase 2: Adaptive refinement")
        print("-" * 50)

        refinement_trials = args.max_trials - exploration_trials

        for trial in range(refinement_trials):
            # Find top 20% configurations
            successful_results = [(c, t, d) for c, t, d in self.results if d]
            if not successful_results:
                # Fallback to random if no successful results yet
                config_dict = self.option_space.generate_random_config()
            else:
                successful_results.sort(key=lambda x: x[1])
                top_configs = successful_results[:max(1, len(successful_results) // 5)]

                # Select one of the top configs randomly
                base_config_str, base_time, base_dict = random.choice(top_configs)

                # Mutate it slightly
                config_dict = base_dict.copy()

                # Change 1-3 random options
                num_changes = random.randint(1, 3)
                options_to_change = random.sample(list(self.option_space.options.keys()),
                                                 min(num_changes, len(self.option_space.options)))

                for opt_name in options_to_change:
                    opt_def = self.option_space.options[opt_name]
                    config_dict[opt_name] = random.choice(opt_def['values'])

                config_dict = self.option_space._apply_constraints(config_dict)

            config_str = self.option_space.config_to_string(config_dict)

            if any(c[0] == config_str for c in self.results):
                continue

            total_trial = exploration_trials + trial + 1
            print(f"[{total_trial}/{args.max_trials}] Testing: {config_str}")

            bench_time = self.benchmark_config(
                args.dsp_file, config_str, args.iterations, args.timeout
            )

            if bench_time is not None:
                self.results.append((config_str, bench_time, config_dict))
                print(f"  Result: {bench_time:.3f}ms")
            else:
                print(f"  Result: FAILED")

        return baseline_time

    def display_results(self, args, baseline_time: Optional[float]):
        """Display final results."""
        print("\n" + "=" * 70)
        print("=== OPTIMIZATION RESULTS ===")
        print("=" * 70)

        # Filter successful results
        successful = [(c, t, d) for c, t, d in self.results if t is not None]

        if not successful:
            print("No successful configurations found!")
            return

        # Sort by time
        successful.sort(key=lambda x: x[1])

        print(f"\nTotal configurations tested: {len(self.results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(self.results) - len(successful)}")

        # Top N results
        top_n = min(args.top_n, len(successful))
        print(f"\n=== TOP {top_n} CONFIGURATIONS ===")
        print()

        for rank, (config_str, bench_time, config_dict) in enumerate(successful[:top_n], 1):
            speedup = ""
            if baseline_time and baseline_time > 0:
                improvement = (baseline_time / bench_time - 1) * 100
                speedup = f" ({improvement:+.1f}% vs baseline)"

            print(f"#{rank}: {bench_time:.3f}ms{speedup}")
            print(f"    {config_str}")
            print()

        # Best configuration details
        best_config, best_time, best_dict = successful[0]
        print("=" * 70)
        print("BEST CONFIGURATION:")
        print(f"  Time: {best_time:.3f}ms")
        if baseline_time:
            improvement = (baseline_time / best_time - 1) * 100
            print(f"  Speedup vs baseline: {improvement:.1f}%")
        print(f"  Command: faust {best_config} <file.dsp> -o <file.cpp>")
        print("=" * 70)

    def compute_parameter_importance(self, sensitivity_results: list) -> list:
        """Compute relative importance of each parameter.

        Args:
            sensitivity_results: List of sensitivity analysis results

        Returns:
            List of dicts with parameter, importance_score, and category
        """
        # Calculate total impact across all parameters
        total_impact = sum(r['max_impact'] for r in sensitivity_results)

        if total_impact == 0:
            # All parameters have zero impact, assign equal importance
            importance = []
            for result in sensitivity_results:
                importance.append({
                    'parameter': result['option'],
                    'importance_score': 1.0 / len(sensitivity_results) if sensitivity_results else 0,
                    'category': 'LOW'
                })
            return importance

        # Compute relative importance
        importance = []
        for result in sensitivity_results:
            relative_importance = result['max_impact'] / total_impact
            category = self.categorize_importance(relative_importance)

            importance.append({
                'parameter': result['option'],
                'importance_score': relative_importance,
                'category': category
            })

        return importance

    def categorize_importance(self, score: float) -> str:
        """Categorize parameter importance.

        Args:
            score: Relative importance score (0-1)

        Returns:
            Category string: CRITICAL, HIGH, MODERATE, or LOW
        """
        if score > 0.20:
            return "CRITICAL"
        elif score > 0.10:
            return "HIGH"
        elif score > 0.05:
            return "MODERATE"
        else:
            return "LOW"

    def save_results(self, filename: str):
        """Save results to JSON file."""
        data = {
            'timestamp': datetime.now().isoformat(),
            'results': [
                {
                    'config': config_str,
                    'time_ms': bench_time,
                    'config_dict': config_dict
                }
                for config_str, bench_time, config_dict in self.results
                if bench_time is not None
            ]
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\nResults saved to: {filename}")

    def perform_sensitivity_analysis(self, best_config_dict: dict, best_time: float,
                                     args, dsp_basename: str, timestamp: str) -> None:
        """Perform iterative sensitivity analysis on the best configuration.

        This identifies which options have the most impact on performance by testing
        variations around the optimal configuration (one-at-a-time method).
        If a better configuration is found, the analysis iterates until convergence
        to a local optimum.
        """
        # Significance threshold: only improvements > 0.5% are considered significant
        MIN_IMPROVEMENT_THRESHOLD = 0.005  # 0.5%

        print("\n" + "=" * 70)
        print("=== SENSITIVITY ANALYSIS WITH LOCAL OPTIMIZATION ===")
        print("=" * 70)
        print(f"Significance threshold: {MIN_IMPROVEMENT_THRESHOLD * 100:.1f}%")
        print("(Only improvements exceeding this threshold will trigger a new iteration)")
        print("=" * 70)

        current_config = best_config_dict.copy()
        current_time = best_time
        iteration = 0
        max_iterations = 10  # Prevent infinite loops

        all_iterations_results = []

        while iteration < max_iterations:
            iteration += 1
            print(f"\n{'=' * 70}")
            print(f"ITERATION {iteration}: Analyzing around configuration ({current_time:.3f}ms)")
            print(f"{'=' * 70}\n")

            sensitivity_results = []
            found_better = False
            best_improvement = None

            # For each option in the option space
            for option_name, option_def in self.option_space.options.items():
                print(f"Analyzing option: {option_name}")

                current_value = current_config.get(option_name)
                values = option_def['values']

                # Skip if only one value available
                if len(values) <= 1:
                    print(f"  → Skipped (only one value available)")
                    continue

                impacts = []

                # Test each alternative value
                for test_value in values:
                    if test_value == current_value:
                        continue  # Skip current value

                    # Create modified config
                    modified_config = current_config.copy()
                    modified_config[option_name] = test_value

                    # Convert to string
                    config_str = self.option_space.config_to_string(modified_config)

                    # Benchmark
                    test_time = self.benchmark_config(
                        args.dsp_file, config_str, args.iterations, args.timeout
                    )

                    if test_time is not None:
                        # Calculate impact (percentage change)
                        impact = ((test_time - current_time) / current_time) * 100
                        impacts.append((test_value, test_time, impact))

                        sign = "+" if impact > 0 else ""
                        improvement_marker = ""

                        # Check if this is significantly better (exceeds threshold)
                        improvement_ratio = (current_time - test_time) / current_time
                        if improvement_ratio > MIN_IMPROVEMENT_THRESHOLD:
                            improvement_marker = " ⚠️  BETTER!"
                            if best_improvement is None or test_time < best_improvement[1]:
                                best_improvement = (option_name, test_time, test_value, modified_config)
                                found_better = True
                        elif test_time < current_time:
                            # Better but below significance threshold
                            improvement_marker = " (below threshold)"

                        print(f"  → {option_name}={test_value}: {test_time:.3f}ms ({sign}{impact:.1f}%){improvement_marker}")
                    else:
                        print(f"  → {option_name}={test_value}: FAILED")

                # Calculate maximum absolute impact for this option
                if impacts:
                    max_impact = max(abs(imp) for _, _, imp in impacts)
                    avg_impact = sum(abs(imp) for _, _, imp in impacts) / len(impacts)

                    sensitivity_results.append({
                        'option': option_name,
                        'current_value': current_value,
                        'max_impact': max_impact,
                        'avg_impact': avg_impact,
                        'variations': impacts
                    })

                print()

            # Sort by maximum impact (most sensitive first)
            sensitivity_results.sort(key=lambda x: x['max_impact'], reverse=True)

            # Store this iteration's results
            all_iterations_results.append({
                'iteration': iteration,
                'starting_time': current_time,
                'starting_config': current_config.copy(),
                'sensitivity_results': sensitivity_results
            })

            # Check if we found a better configuration
            if found_better:
                opt_name, new_time, new_value, new_config = best_improvement
                improvement = ((current_time - new_time) / current_time) * 100

                print(f"\n{'!' * 70}")
                print(f"IMPROVEMENT FOUND!")
                print(f"  Option: {opt_name} = {new_value}")
                print(f"  Previous: {current_time:.3f}ms")
                print(f"  New:      {new_time:.3f}ms")
                print(f"  Gain:     {improvement:.1f}%")
                print(f"{'!' * 70}")

                # Update for next iteration
                current_config = new_config
                current_time = new_time
            else:
                print(f"\n{'=' * 70}")
                print(f"CONVERGENCE REACHED!")
                print(f"No significant improvement found in iteration {iteration}.")
                print(f"(All improvements were below the {MIN_IMPROVEMENT_THRESHOLD * 100:.1f}% threshold)")
                print(f"Local optimum: {current_time:.3f}ms")
                print(f"{'=' * 70}")
                break

        if iteration >= max_iterations:
            print(f"\n⚠️  Maximum iterations ({max_iterations}) reached.")

        # Use the last iteration's sensitivity results for display and saving
        final_sensitivity = all_iterations_results[-1]['sensitivity_results']

        # Display summary of final (converged) configuration
        print("\n" + "=" * 70)
        print("=== FINAL SENSITIVITY RANKING ===")
        print("=" * 70)
        print(f"\nDSP file: {args.dsp_file}")
        print(f"Final optimized time: {current_time:.3f}ms")
        if current_time < best_time:
            improvement = ((best_time - current_time) / best_time) * 100
            print(f"Total improvement from sensitivity analysis: {improvement:.1f}%")
        print()
        print(f"{'Rank':<6} {'Option':<25} {'Current':<15} {'Max Impact':<12} {'Avg Impact':<12}")
        print("-" * 70)

        for rank, result in enumerate(final_sensitivity, 1):
            print(f"{rank:<6} {result['option']:<25} {str(result['current_value']):<15} "
                  f"{result['max_impact']:>10.1f}% {result['avg_impact']:>10.1f}%")

        # Compute parameter importance
        print("\n" + "=" * 70)
        print("=== PARAMETER IMPORTANCE ANALYSIS ===")
        print("=" * 70)

        importance_data = self.compute_parameter_importance(final_sensitivity)

        # Display by category
        categories = {
            'CRITICAL': [],
            'HIGH': [],
            'MODERATE': [],
            'LOW': []
        }

        for item in importance_data:
            categories[item['category']].append(item)

        if categories['CRITICAL']:
            print("\n🔴 CRITICAL IMPACT (>20% of total impact):")
            for item in categories['CRITICAL']:
                print(f"   • {item['parameter']:<25s}: {item['importance_score']*100:5.1f}%")

        if categories['HIGH']:
            print("\n🟡 HIGH IMPACT (10-20%):")
            for item in categories['HIGH']:
                print(f"   • {item['parameter']:<25s}: {item['importance_score']*100:5.1f}%")

        if categories['MODERATE']:
            print("\n🟢 MODERATE IMPACT (5-10%):")
            for item in categories['MODERATE']:
                print(f"   • {item['parameter']:<25s}: {item['importance_score']*100:5.1f}%")

        if categories['LOW']:
            print("\n⚪ LOW IMPACT (<5%):")
            for item in categories['LOW']:
                print(f"   • {item['parameter']:<25s}: {item['importance_score']*100:5.1f}%")

        print("\n" + "=" * 70)
        print("RECOMMENDATIONS:")
        print("  ✓ Focus optimization on CRITICAL and HIGH impact parameters")
        print("  ✓ MODERATE parameters can use default or simple heuristics")
        print("  ✓ LOW impact parameters should be fixed to safe defaults")
        if iteration > 1:
            print(f"  ✓ Converged after {iteration} iterations of local optimization")
        print("=" * 70)

        # Save sensitivity results to JSON (for machine processing)
        sensitivity_json_file = f"{dsp_basename}_sensitivity_{args.lang}_{timestamp}.json"
        sensitivity_data = {
            'timestamp': datetime.now().isoformat(),
            'dsp_file': args.dsp_file,
            'initial_time_ms': best_time,
            'initial_config': best_config_dict,
            'final_time_ms': current_time,
            'final_config': current_config,
            'total_iterations': iteration,
            'all_iterations': all_iterations_results,
            'final_sensitivity_ranking': final_sensitivity,
            'parameter_importance': importance_data
        }

        with open(sensitivity_json_file, 'w') as f:
            json.dump(sensitivity_data, f, indent=2)

        print(f"\nSensitivity results saved to: {sensitivity_json_file}")

        # Save sensitivity results to text file (human-readable)
        sensitivity_txt_file = f"{dsp_basename}_sensitivity_{args.lang}_{timestamp}.txt"
        self.save_sensitivity_report(
            sensitivity_txt_file, args.dsp_file, best_time, current_time,
            iteration, final_sensitivity, importance_data, current_config
        )
        print(f"Human-readable report saved to: {sensitivity_txt_file}")

        # Generate sensitivity graph if matplotlib available
        if MATPLOTLIB_AVAILABLE and final_sensitivity:
            self.generate_sensitivity_graph(
                final_sensitivity,
                importance_data,
                f"{dsp_basename}_sensitivity_{args.lang}_{timestamp}.png"
            )

        # Display final optimized command
        if current_time < best_time:
            print("\n" + "=" * 70)
            print("OPTIMIZED CONFIGURATION:")
            print(f"  Time: {current_time:.3f}ms")
            improvement = ((best_time - current_time) / best_time) * 100
            print(f"  Improvement: {improvement:.1f}% better than initial")
            config_str = self.option_space.config_to_string(current_config)
            print(f"  Command: faust {config_str} <file.dsp> -o <file.cpp>")
            print("=" * 70)

    def save_sensitivity_report(self, filename: str, dsp_file: str, initial_time: float,
                                 final_time: float, iterations: int, sensitivity_results: list,
                                 importance_data: list, final_config: dict) -> None:
        """Save sensitivity analysis report to human-readable text file."""
        with open(filename, 'w', encoding='utf-8') as f:
            # Header
            f.write("=" * 70 + "\n")
            f.write("SENSITIVITY ANALYSIS REPORT\n")
            f.write("=" * 70 + "\n\n")

            # Basic information
            f.write(f"DSP file: {dsp_file}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Initial time: {initial_time:.3f}ms\n")
            f.write(f"Final optimized time: {final_time:.3f}ms\n")

            if final_time < initial_time:
                improvement = ((initial_time - final_time) / initial_time) * 100
                f.write(f"Total improvement from sensitivity analysis: {improvement:.1f}%\n")

            if iterations > 1:
                f.write(f"Converged after {iterations} iterations of local optimization\n")

            f.write("\n")

            # Sensitivity ranking table
            f.write("=" * 70 + "\n")
            f.write("SENSITIVITY RANKING\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"{'Rank':<6} {'Option':<25} {'Current':<15} {'Max Impact':<12} {'Avg Impact':<12}\n")
            f.write("-" * 70 + "\n")

            for rank, result in enumerate(sensitivity_results, 1):
                f.write(f"{rank:<6} {result['option']:<25} {str(result['current_value']):<15} "
                       f"{result['max_impact']:>10.1f}% {result['avg_impact']:>10.1f}%\n")

            f.write("\n")

            # Parameter importance analysis
            f.write("=" * 70 + "\n")
            f.write("PARAMETER IMPORTANCE ANALYSIS\n")
            f.write("=" * 70 + "\n\n")

            # Display by category
            categories = {
                'CRITICAL': [],
                'HIGH': [],
                'MODERATE': [],
                'LOW': []
            }

            for item in importance_data:
                categories[item['category']].append(item)

            if categories['CRITICAL']:
                f.write("🔴 CRITICAL IMPACT (>20% of total impact):\n")
                for item in categories['CRITICAL']:
                    f.write(f"   • {item['parameter']:<25s}: {item['importance_score']*100:5.1f}%\n")
                f.write("\n")

            if categories['HIGH']:
                f.write("🟡 HIGH IMPACT (10-20%):\n")
                for item in categories['HIGH']:
                    f.write(f"   • {item['parameter']:<25s}: {item['importance_score']*100:5.1f}%\n")
                f.write("\n")

            if categories['MODERATE']:
                f.write("🟢 MODERATE IMPACT (5-10%):\n")
                for item in categories['MODERATE']:
                    f.write(f"   • {item['parameter']:<25s}: {item['importance_score']*100:5.1f}%\n")
                f.write("\n")

            if categories['LOW']:
                f.write("⚪ LOW IMPACT (<5%):\n")
                for item in categories['LOW']:
                    f.write(f"   • {item['parameter']:<25s}: {item['importance_score']*100:5.1f}%\n")
                f.write("\n")

            # Recommendations
            f.write("=" * 70 + "\n")
            f.write("RECOMMENDATIONS\n")
            f.write("=" * 70 + "\n\n")

            f.write("  ✓ Focus optimization on CRITICAL and HIGH impact parameters\n")
            f.write("  ✓ MODERATE parameters can use default or simple heuristics\n")
            f.write("  ✓ LOW impact parameters should be fixed to safe defaults\n")

            if iterations > 1:
                f.write(f"  ✓ Converged after {iterations} iterations of local optimization\n")

            f.write("\n")

            # Final optimized configuration
            if final_time < initial_time:
                f.write("=" * 70 + "\n")
                f.write("OPTIMIZED CONFIGURATION\n")
                f.write("=" * 70 + "\n\n")

                f.write(f"  Time: {final_time:.3f}ms\n")
                improvement = ((initial_time - final_time) / initial_time) * 100
                f.write(f"  Improvement: {improvement:.1f}% better than initial\n")

                config_str = self.option_space.config_to_string(final_config)
                f.write(f"  Command: faust {config_str} <file.dsp> -o <file.cpp>\n")

                f.write("\n" + "=" * 70 + "\n")

    def generate_sensitivity_graph(self, sensitivity_results: list, importance_data: list, filename: str) -> None:
        """Generate a bar chart showing option sensitivity with importance categories."""
        if not sensitivity_results:
            return

        fig, ax = plt.subplots(figsize=(14, max(6, len(sensitivity_results) * 0.4)))

        # Prepare data
        options = [r['option'] for r in sensitivity_results]
        max_impacts = [r['max_impact'] for r in sensitivity_results]

        # Map categories to colors
        category_colors = {
            'CRITICAL': '#d62728',  # Red
            'HIGH': '#ff7f0e',      # Orange
            'MODERATE': '#2ca02c',  # Green
            'LOW': '#7f7f7f'        # Gray
        }

        # Create importance lookup
        importance_map = {item['parameter']: item['category'] for item in importance_data}

        # Get colors for each option
        colors = [category_colors.get(importance_map.get(opt, 'LOW'), '#7f7f7f') for opt in options]

        y_pos = range(len(options))

        # Create horizontal bar chart with colored bars
        bars = ax.barh(y_pos, max_impacts, 0.6, color=colors, alpha=0.8)

        # Add category markers on the left
        for i, opt in enumerate(options):
            category = importance_map.get(opt, 'LOW')
            emoji = {'CRITICAL': '🔴', 'HIGH': '🟡', 'MODERATE': '🟢', 'LOW': '⚪'}.get(category, '')
            ax.text(-0.5, i, emoji, fontsize=12, ha='right', va='center')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(options)
        ax.invert_yaxis()  # Most sensitive at top
        ax.set_xlabel('Max Performance Impact (%)', fontsize=12, fontweight='bold')
        ax.set_title('Parameter Sensitivity & Importance Analysis',
                    fontsize=14, fontweight='bold')

        # Create custom legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#d62728', alpha=0.8, label='CRITICAL (>20%)'),
            Patch(facecolor='#ff7f0e', alpha=0.8, label='HIGH (10-20%)'),
            Patch(facecolor='#2ca02c', alpha=0.8, label='MODERATE (5-10%)'),
            Patch(facecolor='#7f7f7f', alpha=0.8, label='LOW (<5%)')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

        ax.grid(True, alpha=0.3, axis='x')

        # Add value labels on bars
        for bar in bars:
            width = bar.get_width()
            if width > 0:
                ax.text(width, bar.get_y() + bar.get_height()/2,
                       f' {width:.1f}%', ha='left', va='center', fontsize=9, fontweight='bold')

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Sensitivity graph saved to: {filename}")
        plt.close()

    def generate_graph(self, filename: str, baseline_time: Optional[float]):
        """Generate optimization progress graph."""
        if not MATPLOTLIB_AVAILABLE:
            print("\nmatplotlib not available, skipping graph generation")
            return

        successful = [(i, t) for i, (c, t, d) in enumerate(self.results) if t is not None]

        if not successful:
            print("\nNo data to plot")
            return

        indices = [i for i, _ in successful]
        times = [t for _, t in successful]

        # Calculate running minimum
        running_min = []
        current_min = float('inf')
        for t in times:
            current_min = min(current_min, t)
            running_min.append(current_min)

        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot all results
        ax.scatter(indices, times, alpha=0.5, s=30, label='Individual tests')

        # Plot running minimum
        ax.plot(indices, running_min, 'r-', linewidth=2, label='Best so far')

        # Baseline if available
        if baseline_time:
            ax.axhline(y=baseline_time, color='g', linestyle='--',
                      linewidth=2, label=f'Baseline ({baseline_time:.3f}ms)')

        ax.set_xlabel('Trial Number', fontsize=12)
        ax.set_ylabel('Execution Time (ms)', fontsize=12)
        ax.set_title('Faust Compiler Optimization Progress', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\nGraph saved to: {filename}")
        plt.close()

    def run(self):
        """Main optimization loop."""
        args = self.parse_args()

        # Verify DSP file exists
        if not os.path.exists(args.dsp_file):
            print(f"Error: DSP file not found: {args.dsp_file}")
            return 1

        # Run optimization
        if args.strategy == 'random':
            baseline_time = self.optimize_random(args)
        elif args.strategy == 'adaptive':
            baseline_time = self.optimize_adaptive(args)

        # Display results
        self.display_results(args, baseline_time)

        # Automatic save with timestamp
        dsp_basename = Path(args.dsp_file).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        auto_json_file = f"{dsp_basename}_opt_{args.lang}_{args.strategy}_{timestamp}.json"

        # Save results (use custom filename if provided, otherwise automatic)
        json_file = args.save_results if args.save_results else auto_json_file
        self.save_results(json_file)

        # Automatic graph generation with timestamp
        auto_graph_file = f"{dsp_basename}_opt_{args.lang}_{args.strategy}_{timestamp}.png"

        # Generate graph (use custom filename if provided, otherwise automatic)
        graph_file = args.graph_output if args.graph_output else auto_graph_file
        if MATPLOTLIB_AVAILABLE:
            self.generate_graph(graph_file, baseline_time)

        # Sensitivity analysis (optional)
        if args.sensitivity_analysis:
            # Get best configuration
            successful = [(c, t, d) for c, t, d in self.results if t is not None]
            if successful:
                successful.sort(key=lambda x: x[1])
                _, best_time, best_config_dict = successful[0]

                self.perform_sensitivity_analysis(
                    best_config_dict, best_time, args, dsp_basename, timestamp
                )

        # Cleanup
        for f in glob.glob("temp_optimize.*"):
            try:
                os.remove(f)
            except:
                pass

        return 0


def main():
    optimizer = FaustOptimizer()
    return optimizer.run()


if __name__ == "__main__":
    sys.exit(main())
