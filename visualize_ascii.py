#!/usr/bin/env python3
import sys
from analyze_benchmarks import parse_benchmark_results

def visualize_results_ascii(results):
    """Create ASCII visualizations of the benchmark results"""
    # Sort resolutions by total pixels
    def resolution_key(res):
        w, h = map(int, res.split('x'))
        return w * h
    
    resolutions = sorted(results.keys(), key=resolution_key)
    
    # Extract data for plotting
    basic_times = []
    optimized_times = []
    mps_times = []
    
    for resolution in resolutions:
        resolution_results = results[resolution]
        if 'basic' in resolution_results:
            basic_times.append(resolution_results['basic'] / 1000)  # Convert to ms
        else:
            basic_times.append(0)
            
        if 'optimized' in resolution_results:
            optimized_times.append(resolution_results['optimized'] / 1000)  # Convert to ms
        else:
            optimized_times.append(0)
            
        if 'mps' in resolution_results:
            mps_times.append(resolution_results['mps'] / 1000)  # Convert to ms
        else:
            mps_times.append(0)
    
    # Find the maximum time for scaling
    max_time = max(max(basic_times), max(optimized_times), max(mps_times))
    
    # Create ASCII bar chart
    print("\n=== Metal Convolution Performance by Resolution (Time in ms) ===\n")
    
    # Print header
    print("Resolution".ljust(12) + "| " + "Basic".ljust(10) + "| " + "Optimized".ljust(10) + "| " + "MPS".ljust(10))
    print("-" * 46)
    
    # Print bars for each resolution
    for i, resolution in enumerate(resolutions):
        print(f"{resolution}".ljust(12) + "| ", end="")
        
        # Basic implementation bar
        basic_bar = create_ascii_bar(basic_times[i], max_time, 10)
        print(f"{basic_bar} {basic_times[i]:.1f}ms".ljust(20) + "| ", end="")
        
        # Optimized implementation bar
        optimized_bar = create_ascii_bar(optimized_times[i], max_time, 10)
        print(f"{optimized_bar} {optimized_times[i]:.1f}ms".ljust(20) + "| ", end="")
        
        # MPS implementation bar
        mps_bar = create_ascii_bar(mps_times[i], max_time, 10)
        print(f"{mps_bar} {mps_times[i]:.1f}ms")
    
    print("\n")
    
    # Create ASCII speedup chart
    print("\n=== MPS Speedup Factor by Resolution ===\n")
    
    # Print header
    print("Resolution".ljust(12) + "| " + "Basic vs MPS".ljust(20) + "| " + "Optimized vs MPS".ljust(20))
    print("-" * 56)
    
    # Print bars for each resolution
    for i, resolution in enumerate(resolutions):
        print(f"{resolution}".ljust(12) + "| ", end="")
        
        # Calculate speedup factors
        basic_speedup = basic_times[i] / mps_times[i] if mps_times[i] > 0 else 0
        optimized_speedup = optimized_times[i] / mps_times[i] if mps_times[i] > 0 else 0
        
        # Basic speedup bar
        basic_bar = create_ascii_bar(basic_speedup, 15, 15)
        print(f"{basic_bar} {basic_speedup:.1f}x".ljust(30) + "| ", end="")
        
        # Optimized speedup bar
        optimized_bar = create_ascii_bar(optimized_speedup, 15, 15)
        print(f"{optimized_bar} {optimized_speedup:.1f}x")
    
    print("\n")

def create_ascii_bar(value, max_value, max_length):
    """Create an ASCII bar representation of a value"""
    if max_value == 0:
        return ""
    
    # Calculate the length of the bar
    bar_length = int((value / max_value) * max_length)
    
    # Create the bar
    bar = "█" * bar_length
    
    return bar

def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_ascii.py <benchmark_results_file>")
        sys.exit(1)
    
    benchmark_file = sys.argv[1]
    with open(benchmark_file, 'r') as f:
        output = f.read()
    
    results = parse_benchmark_results(output)
    visualize_results_ascii(results)

if __name__ == "__main__":
    main() 