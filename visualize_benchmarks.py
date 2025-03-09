#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import sys
from analyze_benchmarks import parse_benchmark_results

def visualize_results(results):
    """Create visualizations of the benchmark results"""
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
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(resolutions))
    width = 0.25
    
    ax.bar(x - width, basic_times, width, label='Basic')
    ax.bar(x, optimized_times, width, label='Optimized')
    ax.bar(x + width, mps_times, width, label='MPS')
    
    ax.set_xlabel('Resolution')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Metal Convolution Performance by Resolution')
    ax.set_xticks(x)
    ax.set_xticklabels(resolutions)
    ax.legend()
    
    # Add values on top of bars
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{height:.1f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
    
    add_labels(ax.patches)
    
    # Create speedup chart
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    
    basic_speedup = []
    optimized_speedup = []
    
    for i, resolution in enumerate(resolutions):
        if mps_times[i] > 0:
            basic_speedup.append(basic_times[i] / mps_times[i])
            optimized_speedup.append(optimized_times[i] / mps_times[i])
        else:
            basic_speedup.append(0)
            optimized_speedup.append(0)
    
    ax2.bar(x - width/2, basic_speedup, width, label='Basic vs MPS')
    ax2.bar(x + width/2, optimized_speedup, width, label='Optimized vs MPS')
    
    ax2.set_xlabel('Resolution')
    ax2.set_ylabel('Speedup Factor (higher is better for MPS)')
    ax2.set_title('MPS Speedup Factor by Resolution')
    ax2.set_xticks(x)
    ax2.set_xticklabels(resolutions)
    ax2.legend()
    
    # Add values on top of bars
    for i, rect in enumerate(ax2.patches):
        height = rect.get_height()
        if height > 0:
            ax2.annotate(f'{height:.1f}x',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    # Save the figures
    fig.tight_layout()
    fig2.tight_layout()
    fig.savefig('benchmark_times.png')
    fig2.savefig('benchmark_speedup.png')
    
    print("Visualizations saved as 'benchmark_times.png' and 'benchmark_speedup.png'")

def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_benchmarks.py <benchmark_results_file>")
        sys.exit(1)
    
    benchmark_file = sys.argv[1]
    with open(benchmark_file, 'r') as f:
        output = f.read()
    
    results = parse_benchmark_results(output)
    visualize_results(results)

if __name__ == "__main__":
    main() 