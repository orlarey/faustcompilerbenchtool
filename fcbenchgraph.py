#!/usr/bin/env python3
"""
fcbenchgraph - Script to benchmark FAUST files with multiple parameter sets
Usage: fcbenchgraph.py <pattern> <faust_params1> [faust_params2] ... [OPTIONS]
"""

import argparse
import glob
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

class FaustBenchmarker:
    def __init__(self):
        self.results = {}  # {filename: {config_idx: time_or_error}}
        self.file_list = []
        self.config_stats = {}  # {config_idx: {'total': float, 'count': int}}
        self.temp_cpp = "temp_benchmark.cpp"
        
    def parse_args(self):
        parser = argparse.ArgumentParser(
            prog="fcbenchgraph",
            description="Benchmark FAUST files with multiple configurations",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  %(prog)s "tests/impulse-tests/dsp/*.dsp" "-lang cpp"
  %(prog)s "*.dsp" "-lang cpp" "-lang cpp -vec" "-lang cpp -double"
  %(prog)s "*.dsp" "-lang cpp" "-lang rust" --iterations=500

The script tests each DSP file with all FAUST parameter sets
and displays a comparative results matrix with graph generation.
            """
        )
        
        parser.add_argument('file_pattern', 
                          help='Pattern of .dsp files to benchmark')
        parser.add_argument('faust_configs', nargs='+',
                          help='One or more FAUST parameter sets')
        parser.add_argument('--iterations', type=int, default=1000,
                          help='Number of iterations for benchmark (default: 1000)')
        parser.add_argument('--extension', default='.bench',
                          help='Extension for generated binaries (default: .bench)')
        parser.add_argument('--no-graph', action='store_true',
                          help='Disable graph generation')
        parser.add_argument('--graph-output', 
                          help='Graph file name (default: benchmark_YYYYMMDD_HHMMSS.png)')
        
        return parser.parse_args()

    def find_files(self, pattern: str) -> List[str]:
        """Find all files matching the pattern."""
        files = glob.glob(pattern)
        return sorted([f for f in files if f.endswith('.dsp')])

    def extract_benchmark_time(self, output: str) -> Optional[float]:
        """Extract benchmark time from output."""
        # Look for a decimal number (potentially with decimals)
        matches = re.findall(r'\d+(?:\.\d+)?', output)
        if matches:
            try:
                return float(matches[-1])  # Take the last number found
            except ValueError:
                return None
        return None

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

    def benchmark_file(self, dsp_file: str, config_idx: int, faust_params: str, 
                      iterations: int, extension: str) -> Tuple[str, Optional[float]]:
        """Benchmark a file with a given configuration."""
        # Variables used
        config_extension = f".config{config_idx + 1}"
        bench_binary = f"temp_benchmark{config_extension}"
        
        # Preventive cleanup
        for f in glob.glob("temp_benchmark.config*"):
            try:
                os.remove(f)
            except:
                pass
                
        print(f"  → Configuration [{config_idx + 1}]: {faust_params}")
        
        # Step 1: FAUST compilation
        faust_cmd = f"faust {faust_params} {dsp_file} -o {self.temp_cpp}".split()
        ret_code, output = self.run_command(faust_cmd)
        
        if ret_code != 0:
            print(f"    ✗ FAUST compilation error")
            return "FAUST_ERR", None
            
        # Step 2: Compilation with fcbenchtool
        fcbench_cmd = ["fcbenchtool", self.temp_cpp, config_extension]
        ret_code, output = self.run_command(fcbench_cmd)
        
        if ret_code != 0:
            print(f"    ✗ fcbenchtool error")
            return "COMPILE_ERR", None
            
        if not os.path.exists(bench_binary):
            print(f"    ✗ Binary not generated")
            return "NO_BIN", None
            
        # Step 3: Benchmark execution
        print(f"    → Running benchmark ({iterations} iterations)...")
        if iterations != 1000:
            bench_cmd = [f"./{bench_binary}", str(iterations)]
        else:
            bench_cmd = [f"./{bench_binary}"]
            
        ret_code, output = self.run_command(bench_cmd)
        
        if ret_code != 0:
            print(f"    ✗ Execution error")
            return "FAILED", None
            
        # Time extraction
        bench_time = self.extract_benchmark_time(output)
        
        if bench_time is None:
            print(f"    ✗ Cannot extract time")
            return "ERROR", None
            
        print(f"    ✓ {bench_time}ms")
        
        # Cleanup
        try:
            os.remove(bench_binary)
            os.remove(self.temp_cpp)
        except:
            pass
            
        return "SUCCESS", bench_time

    def run_benchmarks(self, args):
        """Run all benchmarks."""
        # Find files
        files = self.find_files(args.file_pattern)
        if not files:
            print(f"No files found for pattern: {args.file_pattern}")
            return
            
        # Statistics initialization
        for i in range(len(args.faust_configs)):
            self.config_stats[i] = {'total': 0.0, 'count': 0}
            
        print("=== FAUST files benchmark ===")
        print(f"File pattern: {args.file_pattern}")
        print("FAUST parameter sets:")
        for i, config in enumerate(args.faust_configs):
            print(f"  [{i+1}] {config}")
        print(f"Iterations: {args.iterations}")
        print(f"Binary extension: {args.extension}")
        print("=====================================")
        print()
        
        total_benchmarks = 0
        successful_benchmarks = 0
        
        # Benchmark each file with each configuration
        for file_idx, dsp_file in enumerate(files):
            basename = Path(dsp_file).stem
            self.file_list.append(basename)
            self.results[basename] = {}
            
            print(f"[{file_idx + 1}] Benchmarking: {dsp_file}")
            
            for config_idx, faust_params in enumerate(args.faust_configs):
                total_benchmarks += 1
                
                status, bench_time = self.benchmark_file(
                    dsp_file, config_idx, faust_params, 
                    args.iterations, args.extension
                )
                
                self.results[basename][config_idx] = {
                    'status': status,
                    'time': bench_time
                }
                
                if status == "SUCCESS" and bench_time is not None:
                    successful_benchmarks += 1
                    self.config_stats[config_idx]['total'] += bench_time
                    self.config_stats[config_idx]['count'] += 1
                    
            print()
            
        # Final cleanup
        for f in glob.glob("temp_benchmark.*"):
            try:
                os.remove(f)
            except:
                pass
                
        # Display results
        self.display_results(args.faust_configs, len(files), total_benchmarks, successful_benchmarks)
        
        # Graph generation
        if not args.no_graph:
            self.generate_graph(args.faust_configs, args.graph_output, args)

    def display_results(self, faust_configs: List[str], total_files: int, 
                       total_benchmarks: int, successful_benchmarks: int):
        """Display the results matrix and statistics."""
        
        print("=========================================")
        print("=== RESULTS MATRIX (time in ms) ===")
        print("=========================================")
        
        # Table header
        print(f"{'File':<25}", end="")
        for i in range(len(faust_configs)):
            print(f" | {'Config' + str(i+1):>12}", end="")
        print()
        
        print("=" * 25, end="")
        for i in range(len(faust_configs)):
            print(" | " + "=" * 12, end="")
        print()
        
        # Table data
        for filename in self.file_list:
            print(f"{filename:<25}", end="")
            for config_idx in range(len(faust_configs)):
                result = self.results[filename].get(config_idx, {'status': 'MISSING', 'time': None})
                
                if result['status'] == 'SUCCESS' and result['time'] is not None:
                    print(f" | {result['time']:>10.3f}ms", end="")
                else:
                    print(f" | {result['status']:>12}", end="")
            print()
        
        print()
        
        # Configurations
        print("=== CONFIGURATIONS ===")
        for i, config in enumerate(faust_configs):
            print(f"Config{i+1}: {config}")
        print()
        
        # Statistics per configuration
        print("=== STATISTICS PER CONFIGURATION ===")
        for i, config in enumerate(faust_configs):
            stats = self.config_stats[i]
            print(f"Config{i+1} ({config}):")
            print(f"  - Successful benchmarks: {stats['count']}/{total_files}")
            if stats['count'] > 0:
                avg = stats['total'] / stats['count']
                print(f"  - Total time: {stats['total']:.3f}ms")
                print(f"  - Average time: {avg:.3f}ms")
            else:
                print(f"  - No successful benchmarks")
            print()
        
        # Global statistics
        print("=== GLOBAL STATISTICS ===")
        print(f"Total files: {total_files}")
        print(f"Total configurations: {len(faust_configs)}")
        print(f"Total benchmarks attempted: {total_benchmarks}")
        print(f"Successful benchmarks: {successful_benchmarks}")
        if total_benchmarks > 0:
            success_rate = (successful_benchmarks * 100) // total_benchmarks
            print(f"Global success rate: {success_rate}%")
        print("=========================================")

    def generate_graph(self, faust_configs: List[str], graph_output: Optional[str] = None, 
                      original_args: Optional[argparse.Namespace] = None):
        """Generate a graph of benchmark results."""
        
        if not MATPLOTLIB_AVAILABLE:
            print("\n⚠️  matplotlib is not installed. Graph not generated.")
            print("   To install: pip install matplotlib")
            return
            
        if not self.file_list or not self.results:
            print("\n⚠️  No data to graph.")
            return
            
        # Automatic filename if not specified
        if graph_output is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            graph_output = f"benchmark_{timestamp}.png"
            
        print(f"\n=== GRAPH GENERATION ===")
        print(f"Output file: {graph_output}")
        
        # Data preparation
        _, ax = plt.subplots(figsize=(12, 10))  # A bit taller for the command
        
        # Colors for different configurations
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        # For each configuration, plot a curve
        for config_idx in range(len(faust_configs)):
            x_positions = []
            times = []
            
            # Go through ALL files to maintain X position alignment
            for file_idx, filename in enumerate(self.file_list):
                result = self.results[filename].get(config_idx, {'status': 'MISSING', 'time': None})
                if result['status'] == 'SUCCESS' and result['time'] is not None:
                    x_positions.append(file_idx)  # X position = index in file_list
                    times.append(result['time'])
            
            if times:  # Only if we have data
                color = colors[config_idx % len(colors)]
                label = f"Config{config_idx + 1}"
                
                # Plot points at correct X positions
                ax.plot(x_positions, times, 
                       marker='o', linestyle='-', linewidth=2, markersize=6,
                       color=color, label=label)
        
        # Add a line for failures in the legend if any
        has_failures = any(
            self.results[filename].get(config_idx, {}).get('status') != 'SUCCESS'
            for filename in self.file_list
            for config_idx in range(len(faust_configs))
        )
        
        if has_failures:
            # Ghost line to indicate that missing points are failures
            ax.plot([], [], marker='None', linestyle='None', 
                   label='Note: Missing points = compilation failures', 
                   color='none')
        
        # Axis configuration
        ax.set_xlabel('DSP Files', fontsize=12, fontweight='bold')
        ax.set_ylabel('Benchmark time (ms)', fontsize=12, fontweight='bold')
        ax.set_title('Performance comparison by FAUST configuration', 
                    fontsize=14, fontweight='bold', pad=40)  # More space for the command
        
        # Add the command used as subtitle
        if original_args:
            # Command reconstruction
            script_name = sys.argv[0] if sys.argv else "fcbenchgraph.py"
            command_parts = [f'"{script_name}"', f'"{original_args.file_pattern}"']
            
            # Add FAUST configurations
            for config in original_args.faust_configs:
                command_parts.append(f'"{config}"')
            
            # Add non-default options
            if original_args.iterations != 1000:
                command_parts.append(f'--iterations={original_args.iterations}')
            if original_args.extension != '.bench':
                command_parts.append(f'--extension={original_args.extension}')
            if original_args.graph_output:
                command_parts.append(f'--graph-output="{original_args.graph_output}"')
                
            command_str = ' '.join(command_parts)
            
            # Truncate command if too long for display
            if len(command_str) > 100:
                command_str = command_str[:97] + "..."
                
            ax.text(0.5, 0.98, f"Command: {command_str}", 
                   transform=ax.transAxes, fontsize=9, style='italic',
                   ha='center', va='top', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.7))
        
        # Add generation date/time
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ax.text(0.99, 0.01, f"Generated on: {timestamp}", 
               transform=ax.transAxes, fontsize=8, style='italic',
               ha='right', va='bottom', alpha=0.7)
        
        # X-axis labels (file names)
        # To avoid overlap, take a sample or rotate
        if len(self.file_list) <= 20:
            # Display all names if not too many
            ax.set_xticks(range(len(self.file_list)))
            ax.set_xticklabels(self.file_list, rotation=45, ha='right')
        else:
            # Display a subset if too many
            step = len(self.file_list) // 15  # About 15 labels max
            indices = list(range(0, len(self.file_list), step))
            ax.set_xticks(indices)
            ax.set_xticklabels([self.file_list[i] for i in indices], rotation=45, ha='right')
        
        # Grid for easier reading
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Legend with complete configurations
        legend_labels = []
        for i, config in enumerate(faust_configs):
            # Truncate configs too long for legend
            config_short = config if len(config) <= 50 else config[:47] + "..."
            legend_labels.append(f"Config{i+1}: {config_short}")
        
        # Get current handles and labels, then replace them
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles, legend_labels, 
                 bbox_to_anchor=(1.05, 1), loc='upper left',
                 fontsize=10, framealpha=0.9)
        
        # Automatic layout adjustment
        plt.tight_layout()
        
        # Save
        try:
            plt.savefig(graph_output, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"✓ Graph saved: {graph_output}")
            
            # Graph statistics
            total_successful_benchmarks = sum(
                1 for filename in self.file_list 
                for config_idx in range(len(faust_configs))
                if (self.results[filename].get(config_idx, {}).get('status') == 'SUCCESS')
            )
            print(f"  - Data points displayed: {total_successful_benchmarks}")
            print(f"  - DSP files: {len(self.file_list)}")
            print(f"  - Configurations: {len(faust_configs)}")
            
        except Exception as e:
            print(f"✗ Error saving graph: {e}")
        
        # Close figure to free memory
        plt.close()
        
        print("================================")

def main():
    benchmarker = FaustBenchmarker()
    args = benchmarker.parse_args()
    benchmarker.run_benchmarks(args)

if __name__ == "__main__":
    main()