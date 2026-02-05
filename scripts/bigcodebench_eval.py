#!/usr/bin/env python3
"""
BigCodeBench Full Evaluation Pipeline for SVEN Models

This script runs the complete evaluation pipeline:
1. Generate code samples using SVEN models
2. Sanitize and calibrate the generated code
3. Evaluate and compute pass@k metrics

Usage:
    python bigcodebench_eval.py \
        --model_type prefix \
        --model_dir ../trained/2b-prefix/checkpoint-last \
        --output_name sven-2b-prefix-sec \
        --split complete \
        --subset full \
        --n_samples 10 \
        --pass_k 1,5,10

For quick testing:
    python bigcodebench_eval.py \
        --model_type prefix \
        --model_dir ../trained/2b-prefix/checkpoint-last \
        --output_name sven-2b-prefix-test \
        --max_tasks 5 \
        --n_samples 1 \
        --greedy
"""

import os
import sys
import json
import subprocess
import argparse
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_args():
    parser = argparse.ArgumentParser(description='Full BigCodeBench evaluation pipeline for SVEN models')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, choices=['lm', 'prefix', 'text'], required=True,
                        help='Model type: lm, prefix, or text')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Path to model directory')
    parser.add_argument('--control', type=str, choices=['sec', 'vul'], default='sec',
                        help='Control mode for prefix/text models')
    
    # BigCodeBench arguments
    parser.add_argument('--split', type=str, choices=['complete', 'instruct'], default='complete',
                        help='BigCodeBench split')
    parser.add_argument('--subset', type=str, choices=['full', 'hard'], default='full',
                        help='BigCodeBench subset')
    
    # Generation arguments
    parser.add_argument('--temp', type=float, default=0.8,
                        help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.95,
                        help='Nucleus sampling top-p')
    parser.add_argument('--max_gen_len', type=int, default=512,
                        help='Maximum tokens to generate')
    parser.add_argument('--n_samples', type=int, default=10,
                        help='Number of samples per task')
    parser.add_argument('--batch_size', type=int, default=5,
                        help='Batch size for generation')
    parser.add_argument('--greedy', action='store_true',
                        help='Use greedy decoding')
    
    # Evaluation arguments
    parser.add_argument('--pass_k', type=str, default='1,5,10',
                        help='Comma-separated k values for pass@k (default: 1,5,10)')
    parser.add_argument('--parallel', type=int, default=None,
                        help='Number of parallel workers for evaluation')
    parser.add_argument('--use_docker', action='store_true',
                        help='Use Docker for evaluation (recommended for safety)')
    
    # Output arguments
    parser.add_argument('--output_name', type=str, required=True,
                        help='Output file name prefix')
    parser.add_argument('--output_dir', type=str, default='../experiments/bigcodebench',
                        help='Output directory')
    
    # Pipeline control
    parser.add_argument('--skip_generate', action='store_true',
                        help='Skip generation step (use existing samples)')
    parser.add_argument('--skip_sanitize', action='store_true',
                        help='Skip sanitize/calibrate step')
    parser.add_argument('--skip_evaluate', action='store_true',
                        help='Skip evaluation step')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--max_tasks', type=int, default=None,
                        help='Maximum number of tasks (for testing)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from existing output')
    
    args = parser.parse_args()
    
    # Handle greedy decoding - must be before file path generation
    if args.greedy:
        args.n_samples = 1
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate file paths (use the corrected n_samples value)
    backend = 'hf'
    temp_str = f'{args.temp:.1f}' if not args.greedy else 'greedy'
    base_name = f'{args.output_name}--bigcodebench-{args.split}--{backend}-{temp_str}-{args.n_samples}'
    
    args.samples_file = os.path.join(args.output_dir, f'{base_name}.jsonl')
    args.sanitized_file = os.path.join(args.output_dir, f'{base_name}-sanitized.jsonl')
    # Note: bigcodebench uses hyphen not underscore for calibrated files
    args.calibrated_file = os.path.join(args.output_dir, f'{base_name}-sanitized-calibrated.jsonl')
    args.results_file = os.path.join(args.output_dir, f'{base_name}-sanitized-calibrated_eval_results.json')
    args.pass_at_k_file = os.path.join(args.output_dir, f'{base_name}-sanitized-calibrated_pass_at_k.json')
    
    return args


def run_command(cmd, description, check=True):
    """Run a shell command and print output."""
    print(f"\n{'='*60}")
    print(f"Step: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, check=check, capture_output=False)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error: Command failed with return code {e.returncode}")
        return False
    except FileNotFoundError as e:
        print(f"Error: Command not found: {e}")
        return False


def step_generate(args):
    """Step 1: Generate code samples."""
    print("\n" + "="*60)
    print("STEP 1: Generating Code Samples")
    print("="*60)
    
    if os.path.exists(args.samples_file) and not args.resume:
        print(f"Output file already exists: {args.samples_file}")
        response = input("Overwrite? [y/N]: ").strip().lower()
        if response != 'y':
            print("Using existing samples file.")
            return True
    
    cmd = [
        sys.executable, 'bigcodebench_gen.py',
        '--model_type', args.model_type,
        '--model_dir', args.model_dir,
        '--control', args.control,
        '--split', args.split,
        '--subset', args.subset,
        '--temp', str(args.temp),
        '--top_p', str(args.top_p),
        '--max_gen_len', str(args.max_gen_len),
        '--n_samples', str(args.n_samples),
        '--batch_size', str(args.batch_size),
        '--output_name', args.output_name,
        '--output_dir', args.output_dir,
        '--seed', str(args.seed),
    ]
    
    if args.greedy:
        cmd.append('--greedy')
    if args.resume:
        cmd.append('--resume')
    if args.max_tasks:
        cmd.extend(['--max_tasks', str(args.max_tasks)])
    
    return run_command(cmd, "Generate code samples")


def step_sanitize(args):
    """Step 2: Sanitize and calibrate code samples."""
    print("\n" + "="*60)
    print("STEP 2: Sanitizing and Calibrating Code")
    print("="*60)
    
    if not os.path.exists(args.samples_file):
        print(f"Error: Samples file not found: {args.samples_file}")
        return False
    
    # Check syntax first
    print("\nChecking syntax...")
    run_command(['bigcodebench.syncheck', '--samples', args.samples_file], 
                "Check syntax", check=False)
    
    # Sanitize with calibration (--calibrate flag handles both steps)
    print("\nSanitizing and calibrating...")
    success = run_command(
        ['bigcodebench.sanitize', '--samples', args.samples_file, '--calibrate'],
        "Sanitize and calibrate samples"
    )
    
    if not success:
        print("Warning: Sanitize step had issues, but continuing...")
    
    # Check if calibrated file was created
    # When using --calibrate, output file is named *-sanitized_calibrated.jsonl
    if not os.path.exists(args.calibrated_file):
        print(f"Warning: Calibrated file not found at {args.calibrated_file}")
        # Try to find it with glob
        import glob
        pattern = os.path.join(args.output_dir, f'{args.output_name}*-sanitized_calibrated.jsonl')
        matches = glob.glob(pattern)
        if matches:
            args.calibrated_file = matches[-1]
            print(f"Found calibrated file: {args.calibrated_file}")
        else:
            # Maybe it's just sanitized without _calibrated suffix
            pattern2 = os.path.join(args.output_dir, f'{args.output_name}*-sanitized.jsonl')
            matches2 = glob.glob(pattern2)
            if matches2:
                args.calibrated_file = matches2[-1]
                print(f"Using sanitized file: {args.calibrated_file}")
    
    return success


def step_evaluate(args):
    """Step 3: Evaluate and compute pass@k."""
    print("\n" + "="*60)
    print("STEP 3: Evaluating Code Samples")
    print("="*60)
    
    # Find the calibrated file
    if not os.path.exists(args.calibrated_file):
        import glob
        pattern = os.path.join(args.output_dir, '*-sanitized_calibrated.jsonl')
        matches = glob.glob(pattern)
        if matches:
            args.calibrated_file = matches[-1]
            print(f"Using calibrated file: {args.calibrated_file}")
        else:
            print(f"Error: Calibrated file not found: {args.calibrated_file}")
            return False
    
    if args.use_docker:
        # Use Docker for safe evaluation (recommended)
        # Mount the experiments/bigcodebench directory, not scripts
        experiments_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'experiments', 'bigcodebench'))
        
        cmd = [
            'docker', 'run', '--rm',
            '-v', f'{experiments_dir}:/app',
            'bigcodebench/bigcodebench-evaluate:latest',
            args.split,
            args.subset,
            '--execution', 'local',
            '--samples', os.path.basename(args.calibrated_file),
            '--pass_k', args.pass_k,
            '--save_pass_rate',
        ]
    else:
        # Local evaluation (fast but requires all dependencies)
        # Consider using --use_docker flag for safer evaluation
        cmd = [
            'bigcodebench.evaluate',
            args.split,
            args.subset,
            '--execution', 'local',
            '--samples', args.calibrated_file,
            '--pass_k', args.pass_k,
            '--save_pass_rate',
        ]
    
    if args.parallel and not args.use_docker:
        cmd.extend(['--parallel', str(args.parallel)])
    
    return run_command(cmd, "Evaluate and compute pass@k")


def print_results(args):
    """Print final results."""
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    # Try to load pass@k results
    pass_at_k_file = args.pass_at_k_file
    if not os.path.exists(pass_at_k_file):
        # Try alternative location
        import glob
        pattern = os.path.join(args.output_dir, '*pass_at_k.json')
        matches = glob.glob(pattern)
        if matches:
            pass_at_k_file = matches[-1]
    
    if os.path.exists(pass_at_k_file):
        with open(pass_at_k_file, 'r') as f:
            results = json.load(f)
        
        print(f"\nPass@k Results ({pass_at_k_file}):")
        print("-" * 40)
        for k, v in sorted(results.items()):
            if k.startswith('pass@'):
                print(f"  {k}: {v:.4f}")
    else:
        print(f"Pass@k results file not found.")
        print("Check the output directory for results files.")
    
    # List all output files
    print(f"\nOutput files in {args.output_dir}:")
    for f in sorted(os.listdir(args.output_dir)):
        if args.output_name in f:
            filepath = os.path.join(args.output_dir, f)
            size = os.path.getsize(filepath)
            print(f"  {f} ({size:,} bytes)")


def main():
    args = get_args()
    
    print("="*60)
    print("BigCodeBench Full Evaluation Pipeline")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: {args.model_type} @ {args.model_dir}")
    print(f"Split/Subset: {args.split}/{args.subset}")
    print(f"Samples per task: {args.n_samples}")
    print(f"Pass@k values: {args.pass_k}")
    print("="*60)
    
    # Change to scripts directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    success = True
    
    # Step 1: Generate
    if not args.skip_generate:
        success = step_generate(args)
        if not success:
            print("Generation step failed. Stopping.")
            return
    
    # Step 2: Sanitize and Calibrate
    if not args.skip_sanitize:
        success = step_sanitize(args)
        if not success:
            print("Sanitize/calibrate step had issues, but continuing to evaluation...")
    
    # Step 3: Evaluate
    if not args.skip_evaluate:
        success = step_evaluate(args)
    
    # Print results
    print_results(args)
    
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)


if __name__ == '__main__':
    main()
