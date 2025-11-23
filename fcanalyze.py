#!/usr/bin/env python3
"""
fcanalyze - Script to analyze FAUST files with multiple parameter sets using fcanalyzetool
Usage: fcanalyze.py <pattern> <faust_params1> [faust_params2] ... [OPTIONS]
"""

import argparse
import glob
import os
import subprocess
from pathlib import Path
from typing import List, Tuple

class FaustAnalyzer:
    def __init__(self):
        self.results = {}  # {filename: {config_idx: analysis_result}}
        self.file_list = []
        self.config_stats = {}  # {config_idx: {'issues': int, 'warnings': int, 'errors': int, 'count': int}}
        self.temp_cpp = "temp_analyze.cpp"
        
    def parse_args(self):
        parser = argparse.ArgumentParser(
            prog="fcanalyze",
            description="Analyze FAUST files with multiple configurations using fcanalyzetool",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  %(prog)s "tests/impulse-tests/dsp/*.dsp" "-lang cpp"
  %(prog)s "*.dsp" "-lang cpp" "-lang cpp -vec" "-lang cpp -double"
  %(prog)s "*.dsp" "-lang cpp" "-lang rust"
  %(prog)s "*.dsp" "-lang cpp" --show-warnings

The script analyzes each DSP file with all FAUST parameter sets
and displays a comparative analysis results matrix.
            """
        )

        parser.add_argument('file_pattern',
                          help='Pattern of .dsp files to analyze')
        parser.add_argument('faust_configs', nargs='+',
                          help='One or more FAUST parameter sets')
        parser.add_argument('-w', '--show-warnings',
                          action='store_true',
                          help='Display detailed warnings and issues found during analysis')

        return parser.parse_args()

    def find_files(self, pattern: str) -> List[str]:
        """Find all files matching the pattern."""
        files = glob.glob(pattern)
        return sorted([f for f in files if f.endswith('.dsp')])

    def parse_analysis_output(self, output: str) -> Tuple[int, int, List[str]]:
        """Parse fcanalyzetool output to extract warnings and errors."""
        warnings = 0
        errors = 0
        issues = []
        
        lines = output.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Count warnings
            if 'warning:' in line.lower():
                warnings += 1
                issues.append(f"WARNING: {line}")
            # Count errors
            elif 'error:' in line.lower():
                errors += 1
                issues.append(f"ERROR: {line}")
            # Capture other analysis issues
            elif any(keyword in line.lower() for keyword in ['potential', 'suspicious', 'dead code', 'unused']):
                issues.append(f"ISSUE: {line}")
        
        return warnings, errors, issues

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

    def analyze_file(self, dsp_file: str, config_idx: int, faust_params: str, show_warnings: bool = False) -> Tuple[str, dict]:
        """Analyze a file with a given configuration."""

        print(f"  → Configuration [{config_idx + 1}]: {faust_params}")
        
        # Step 1: FAUST compilation
        faust_cmd = f"faust {faust_params} {dsp_file} -o {self.temp_cpp}".split()
        ret_code, output = self.run_command(faust_cmd)
        
        if ret_code != 0:
            print(f"    ✗ FAUST compilation error")
            return "FAUST_ERR", {
                'status': 'FAUST_ERR',
                'warnings': 0,
                'errors': 1,
                'issues': [f"FAUST compilation failed: {output[:200]}..."]
            }
            
        # Step 2: Analysis with fcanalyzetool
        fcanalyze_cmd = ["fcanalyzetool", self.temp_cpp]
        ret_code, output = self.run_command(fcanalyze_cmd)
        
        # Parse analysis output
        warnings, errors, issues = self.parse_analysis_output(output)
        total_issues = len(issues)
        
        # Check if compilation failed (fcanalyzetool returns non-zero and has "error" in output)
        compilation_failed = ret_code != 0 and "error" in output.lower()
        
        # Determine status
        if compilation_failed:
            status = "COMPILE_ERR"
            # Count compilation errors if not already counted
            if errors == 0:
                errors = output.lower().count('error:')
        elif warnings > 0 or errors > 0 or total_issues > 0:
            status = "ISSUES_FOUND"
        else:
            status = "CLEAN"
            
        print(f"    → Analysis: {warnings} warnings, {errors} errors, {total_issues} total issues")

        # Display warnings if requested
        if show_warnings and len(issues) > 0:
            print(f"    → Details:")
            for issue in issues:
                print(f"       {issue}")

        # Cleanup
        try:
            if os.path.exists(self.temp_cpp):
                os.remove(self.temp_cpp)
            # Clean up temp files created by fcanalyzetool
            temp_files = glob.glob("temp_analyze.tmp.cpp")
            for f in temp_files:
                os.remove(f)
        except:
            pass
            
        result = {
            'status': status,
            'warnings': warnings,
            'errors': errors,
            'total_issues': total_issues,
            'issues': issues  # Keep all issues
        }
        
        return status, result

    def run_analysis(self, args):
        """Run all analyses."""
        # Find files
        files = self.find_files(args.file_pattern)
        if not files:
            print(f"No files found for pattern: {args.file_pattern}")
            return

        # Statistics initialization
        for i in range(len(args.faust_configs)):
            self.config_stats[i] = {'issues': 0, 'warnings': 0, 'errors': 0, 'count': 0}

        print("=== FAUST files analysis ===")
        print(f"File pattern: {args.file_pattern}")
        print("FAUST parameter sets:")
        for i, config in enumerate(args.faust_configs):
            print(f"  [{i+1}] {config}")
        if args.show_warnings:
            print("Show warnings: ENABLED")
        print("=====================================")
        print()

        total_analyses = 0
        successful_analyses = 0
        
        # Analyze each file with each configuration
        for file_idx, dsp_file in enumerate(files):
            basename = Path(dsp_file).stem
            self.file_list.append(basename)
            self.results[basename] = {}
            
            print(f"[{file_idx + 1}] Analyzing: {dsp_file}")
            
            for config_idx, faust_params in enumerate(args.faust_configs):
                total_analyses += 1

                status, result = self.analyze_file(
                    dsp_file, config_idx, faust_params, args.show_warnings
                )
                
                self.results[basename][config_idx] = result
                
                if status not in ["FAUST_ERR", "COMPILE_ERR"]:
                    successful_analyses += 1
                    
                stats = self.config_stats[config_idx]
                stats['count'] += 1
                stats['warnings'] += result['warnings']
                stats['errors'] += result['errors']
                if 'total_issues' in result:
                    stats['issues'] += result['total_issues']
                    
            print()
            
        # Display results
        self.display_results(args.faust_configs, len(files), total_analyses, successful_analyses)

    def display_results(self, faust_configs: List[str], total_files: int, 
                       total_analyses: int, successful_analyses: int):
        """Display the results matrix and statistics."""
        
        print("=========================================")
        print("=== ANALYSIS RESULTS MATRIX ===")
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
                result = self.results[filename].get(config_idx, {'status': 'MISSING'})
                
                status = result['status']
                if status == 'CLEAN':
                    print(f" | {'✓ CLEAN':>12}", end="")
                elif status == 'ISSUES_FOUND':
                    w = result.get('warnings', 0)
                    e = result.get('errors', 0)
                    print(f" | {w}W/{e}E", end="")
                    print(" " * (12 - len(f"{w}W/{e}E")), end="")
                elif status == 'COMPILE_ERR':
                    e = result.get('errors', 0)
                    print(f" | COMPILE_ERR({e})", end="")
                else:
                    print(f" | {status:>12}", end="")
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
            print(f"  - Successful analyses: {stats['count']}/{total_files}")
            if stats['count'] > 0:
                print(f"  - Total warnings: {stats['warnings']}")
                print(f"  - Total errors: {stats['errors']}")
                print(f"  - Total issues: {stats['issues']}")
                avg_issues = stats['issues'] / stats['count']
                print(f"  - Average issues per file: {avg_issues:.1f}")
            else:
                print(f"  - No successful analyses")
            print()
        
        # Global statistics
        print("=== GLOBAL STATISTICS ===")
        print(f"Total files: {total_files}")
        print(f"Total configurations: {len(faust_configs)}")
        print(f"Total analyses attempted: {total_analyses}")
        print(f"Successful analyses: {successful_analyses}")
        if total_analyses > 0:
            success_rate = (successful_analyses * 100) // total_analyses
            print(f"Global success rate: {success_rate}%")
        
        # Summary of most problematic files
        print("\n=== MOST PROBLEMATIC FILES ===")
        file_issues = {}
        for filename in self.file_list:
            total_issues = 0
            total_errors = 0
            for config_idx in range(len(faust_configs)):
                result = self.results[filename].get(config_idx, {})
                status = result.get('status', '')
                if status in ['ISSUES_FOUND', 'COMPILE_ERR', 'FAUST_ERR']:
                    total_issues += result.get('total_issues', 0)
                    total_errors += result.get('errors', 0)
            
            # Include files with any issues or errors
            if total_issues > 0 or total_errors > 0:
                # Use errors as primary sort key, then issues
                file_issues[filename] = (total_errors, total_issues)
        
        if file_issues:
            sorted_files = sorted(file_issues.items(), key=lambda x: (x[1][0], x[1][1]), reverse=True)
            for filename, (errors, issues) in sorted_files[:5]:  # Top 5
                if errors > 0 and issues > 0:
                    print(f"  {filename}: {errors} errors, {issues} issues")
                elif errors > 0:
                    print(f"  {filename}: {errors} errors")
                else:
                    print(f"  {filename}: {issues} issues")
        else:
            print("  No issues found across all files!")
            
        print("=========================================")

def main():
    analyzer = FaustAnalyzer()
    args = analyzer.parse_args()
    analyzer.run_analysis(args)

if __name__ == "__main__":
    main()