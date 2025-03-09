#!/usr/bin/env python3
import re
import sys
import os
import subprocess
from collections import defaultdict

def run_benchmark():
    """Run the benchmark and capture the output"""
    result = subprocess.run(["cargo", "bench", "--bench", "conv_benchmark"], 
                           capture_output=True, text=True)
    return result.stdout

def parse_benchmark_results(output):
    """Parse the benchmark results from the output"""
    results = defaultdict(dict)
    
    # Extract implementation and resolution from the benchmark name
    impl_pattern = r"metal_convolution/(\w+)/(\d+)x(\d+)"
    # Extract the middle time value (mean)
    time_pattern = r"time:\s+\[[\d\.]+ [µm]s ([\d\.]+) [µm]s [\d\.]+ [µm]s\]"
    
    lines = output.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i]
        impl_match = re.search(impl_pattern, line)
        if impl_match:
            impl, width, height = impl_match.groups()
            resolution = f"{width}x{height}"
            
            # Look for the time in the next line
            if i + 1 < len(lines):
                time_line = lines[i + 1]
                time_match = re.search(time_pattern, time_line)
                if time_match:
                    mean_time = float(time_match.group(1))
                    
                    # Convert to microseconds if in milliseconds
                    if 'ms' in time_line:
                        mean_time *= 1000
                    
                    results[resolution][impl] = mean_time
        i += 1
    
    return results

def analyze_results(results):
    """Analyze the results and calculate comparisons"""
    analysis = []
    
    # Sort resolutions by total pixels (width * height)
    def resolution_key(res):
        w, h = map(int, res.split('x'))
        return w * h
    
    for resolution in sorted(results.keys(), key=resolution_key):
        resolution_results = results[resolution]
        
        if 'mps' in resolution_results:
            mps_time = resolution_results['mps']
            
            for impl in ['basic', 'optimized']:
                if impl in resolution_results:
                    impl_time = resolution_results[impl]
                    speedup = impl_time / mps_time
                    analysis.append({
                        'resolution': resolution,
                        'implementation': impl,
                        'time_us': impl_time,
                        'mps_time_us': mps_time,
                        'speedup': speedup
                    })
    
    return analysis

def print_results_table(analysis):
    """Print the results in a formatted table"""
    print("\n=== Metal Convolution Performance Comparison ===\n")
    print("Resolution | Implementation | Time (µs) | MPS Time (µs) | Speedup vs MPS")
    print("-----------+----------------+-----------+--------------+---------------")
    
    for result in analysis:
        print(f"{result['resolution']:^10} | {result['implementation']:^14} | {result['time_us']:>9.2f} | {result['mps_time_us']:>12.2f} | {result['speedup']:>13.2f}x")
    
    print("\n")

def main():
    # Check if benchmark output file exists
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        with open(sys.argv[1], 'r') as f:
            output = f.read()
    else:
        print("Running benchmarks...")
        output = run_benchmark()
    
    results = parse_benchmark_results(output)
    analysis = analyze_results(results)
    print_results_table(analysis)

if __name__ == "__main__":
    main() 