#!/usr/bin/env python
"""
Plot security evaluation results comparing original model vs prefix-tuned model.

Usage:
    python plot_results.py <orig> <sec> <vul> [--output OUTPUT] [--title TITLE]

Examples:
    python plot_results.py 64.1 91.4 39.3
    python plot_results.py 64.1 91.4 39.3 --output results.png --title "CodeGen 350M"
"""

import argparse
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend


def get_args():
    parser = argparse.ArgumentParser(
        description='Plot security evaluation results comparing original model vs prefix-tuned model'
    )
    parser.add_argument('orig', type=float, help='Original model security rate (%%)')
    parser.add_argument('sec', type=float, help='Prefix model (sec control) security rate (%%)')
    parser.add_argument('vul', type=float, help='Prefix model (vul control) security rate (%%)')
    parser.add_argument('--output', '-o', type=str, default='security_comparison.png',
                        help='Output file path (default: security_comparison.png)')
    parser.add_argument('--title', '-t', type=str, default='CodeGen\n350M',
                        help='Title below the bars (default: CodeGen 350M)')
    parser.add_argument('--show', action='store_true',
                        help='Show the plot interactively (requires display)')
    return parser.parse_args()


def plot_security_comparison(orig: float, sec: float, vul: float, 
                              output_path: str, title: str, show: bool = False):
    """
    Create a bar chart comparing security rates.
    
    Args:
        orig: Original model security rate (%)
        sec: Prefix model sec-control security rate (%)
        vul: Prefix model vul-control security rate (%)
        output_path: Path to save the figure
        title: Title below the bars
        show: Whether to show the plot interactively
    """
    # Set up the figure
    fig, ax = plt.subplots(figsize=(3.5, 4))
    
    # Data
    labels = ['Orig', 'Sec', 'Vul']
    values = [orig, sec, vul]
    colors = ['#A0A0A0', '#7CB97C', '#E8A0A0']  # Gray, Green, Pink/Red
    
    # Bar positions
    x_pos = [0, 0.5, 1.0]
    bar_width = 0.35
    
    # Create bars
    bars = ax.bar(x_pos, values, width=bar_width, color=colors, edgecolor='black', linewidth=0.5)
    
    # Add value labels on top of each bar
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f'{val:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=10)
    
    # Customize axes
    ax.set_ylim(0, 100)
    ax.set_ylabel('')
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(['0', '25', '50', '75', '100'])
    
    # Remove x-axis ticks and add title below
    ax.set_xticks([])
    ax.set_xlabel(title, fontsize=11, labelpad=10)
    
    # Add subtle grid
    ax.yaxis.grid(True, linestyle='-', alpha=0.3)
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"Figure saved to: {output_path}")
    
    if show:
        plt.show()
    
    plt.close()


def main():
    args = get_args()
    
    print(f"Plotting security comparison:")
    print(f"  Original: {args.orig}%")
    print(f"  Sec:      {args.sec}%")
    print(f"  Vul:      {args.vul}%")
    
    plot_security_comparison(
        orig=args.orig,
        sec=args.sec,
        vul=args.vul,
        output_path=args.output,
        title=args.title,
        show=args.show
    )


if __name__ == '__main__':
    main()
