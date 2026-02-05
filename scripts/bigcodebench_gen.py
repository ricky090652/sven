#!/usr/bin/env python3
"""
BigCodeBench Code Generation Script for SVEN Models

This script generates code samples using SVEN models (LM, prefix, or text-prompt)
and saves them in the format required by BigCodeBench for evaluation.

Usage:
    python bigcodebench_gen.py \
        --model_type prefix \
        --model_dir ../trained/2b-prefix/checkpoint-last \
        --output_name sven-2b-prefix \
        --split complete \
        --subset full \
        --n_samples 10 \
        --temp 0.8

Output:
    The script generates a JSONL file with the following format:
    {"task_id": "BigCodeBench/0", "solution": "..."}
    
    This file can be used with BigCodeBench for evaluation:
    bigcodebench.sanitize --samples <output_file>
    bigcodebench.evaluate --samples <sanitized_file> --split complete
"""

import os
import sys
import json
import torch
import argparse
from tqdm import tqdm
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sven.utils import set_seed
from sven.model import load_model, XGLMForCausalLM, GPT2LMHeadCustomModel
from sven.constant import PROMPTS, MODEL_DIRS


def get_args():
    parser = argparse.ArgumentParser(description='Generate code samples for BigCodeBench evaluation')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, choices=['lm', 'prefix', 'text'], required=True,
                        help='Model type: lm (language model), prefix (prefix-tuned), text (text-prompted)')
    parser.add_argument('--model_dir', type=str, default=None,
                        help='Path to model directory or model name (e.g., 2b, 350m, trained/2b-prefix/checkpoint-last)')
    parser.add_argument('--control', type=str, choices=['sec', 'vul'], default='sec',
                        help='Control mode for prefix/text models: sec (secure) or vul (vulnerable)')
    
    # BigCodeBench arguments
    parser.add_argument('--split', type=str, choices=['complete', 'instruct'], default='complete',
                        help='BigCodeBench split: complete (docstring-based) or instruct (NL-based)')
    parser.add_argument('--subset', type=str, choices=['full', 'hard'], default='full',
                        help='BigCodeBench subset: full (1140 tasks) or hard (harder subset)')
    
    # Generation arguments
    parser.add_argument('--temp', type=float, default=0.4,
                        help='Sampling temperature (default: 0.4)')
    parser.add_argument('--top_p', type=float, default=0.95,
                        help='Nucleus sampling top-p (default: 0.95)')
    parser.add_argument('--max_gen_len', type=int, default=1024,
                        help='Maximum number of tokens to generate (default: 1024)')
    parser.add_argument('--n_samples', type=int, default=10,
                        help='Number of samples per task (default: 10, use 1 for greedy)')
    parser.add_argument('--batch_size', type=int, default=5,
                        help='Number of samples to generate per batch (default: 5)')
    parser.add_argument('--greedy', action='store_true',
                        help='Use greedy decoding (overrides temp and n_samples)')
    
    # Output arguments
    parser.add_argument('--output_name', type=str, required=True,
                        help='Output file name prefix (e.g., sven-2b-prefix)')
    parser.add_argument('--output_dir', type=str, default='../experiments/bigcodebench',
                        help='Output directory (default: ../experiments/bigcodebench)')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from existing output file')
    parser.add_argument('--max_tasks', type=int, default=None,
                        help='Maximum number of tasks to process (for testing)')
    
    args = parser.parse_args()
    
    # Handle greedy decoding
    if args.greedy:
        args.temp = 0.0
        args.n_samples = 1
        args.batch_size = 1
    
    # Ensure batch_size doesn't exceed n_samples
    args.batch_size = min(args.batch_size, args.n_samples)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate output filename
    backend = 'hf'
    temp_str = f'{args.temp:.1f}' if not args.greedy else 'greedy'
    args.output_file = os.path.join(
        args.output_dir,
        f'{args.output_name}--bigcodebench-{args.split}--{backend}-{temp_str}-{args.n_samples}.jsonl'
    )
    
    return args


def load_bigcodebench_problems(split='complete', subset='full'):
    """Load BigCodeBench problems using the bigcodebench package."""
    try:
        from bigcodebench.data import get_bigcodebench
        problems = get_bigcodebench(subset=subset)
        print(f"Loaded {len(problems)} BigCodeBench problems (subset={subset})")
        return problems
    except ImportError:
        print("Error: bigcodebench package not installed.")
        print("Please install it with: pip install bigcodebench")
        print("Or: pip install 'git+https://github.com/bigcode-project/bigcodebench.git'")
        sys.exit(1)


def load_existing_results(output_file):
    """Load existing results for resuming."""
    completed_tasks = set()
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    completed_tasks.add(data['task_id'])
                except:
                    pass
    return completed_tasks



def generate_samples(model, tokenizer, prompt, args, device):
    """Generate code samples for a given prompt."""
    samples = []
    
    # Prepare input
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    
    # Remove token_type_ids for certain models
    if isinstance(model, XGLMForCausalLM):
        if 'token_type_ids' in inputs:
            del inputs['token_type_ids']
    
    # Prepare generation kwargs
    gen_kwargs = {
        'do_sample': not args.greedy,
        'max_new_tokens': args.max_gen_len,
        'pad_token_id': tokenizer.eos_token_id,
        'eos_token_id': tokenizer.eos_token_id,
        'use_cache': True,
    }
    
    if not args.greedy:
        gen_kwargs['temperature'] = args.temp
        gen_kwargs['top_p'] = args.top_p
    
    # Add control_id for prefix models
    if args.model_type == 'prefix':
        gen_kwargs['control_id'] = 0 if args.control == 'sec' else 1
    
    # Generate in batches
    num_batches = (args.n_samples + args.batch_size - 1) // args.batch_size
    
    for batch_idx in range(num_batches):
        current_batch_size = min(args.batch_size, args.n_samples - batch_idx * args.batch_size)
        gen_kwargs['num_return_sequences'] = current_batch_size
        
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
        
        for output in outputs:
            # Extract only the generated part (remove prompt)
            generated_ids = output[inputs['input_ids'].shape[1]:]
            
            # Decode
            if tokenizer.eos_token_id in generated_ids.tolist():
                eos_idx = generated_ids.tolist().index(tokenizer.eos_token_id)
                generated_ids = generated_ids[:eos_idx]
            
            completion = tokenizer.decode(generated_ids, skip_special_tokens=True)
            # Note: Don't trim here - let bigcodebench.sanitize handle stop tokens
            # BigCodeBench problems may need helper functions/classes
            
            # Combine prompt and completion for the solution
            solution = prompt + completion
            samples.append(solution)
    
    return samples


def main():
    args = get_args()
    
    # Setup device (must be before set_seed which uses args.n_gpu)
    args.n_gpu = torch.cuda.device_count()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    set_seed(args)
    
    print("=" * 60)
    print("BigCodeBench Code Generation for SVEN Models")
    print("=" * 60)
    print(f"Model type: {args.model_type}")
    print(f"Model dir: {args.model_dir}")
    print(f"Split: {args.split}")
    print(f"Subset: {args.subset}")
    print(f"Temperature: {args.temp}")
    print(f"Samples per task: {args.n_samples}")
    print(f"Output file: {args.output_file}")
    print("=" * 60)
    
    # Load BigCodeBench problems
    problems = load_bigcodebench_problems(args.split, args.subset)
    
    # Handle resume
    completed_tasks = set()
    if args.resume:
        completed_tasks = load_existing_results(args.output_file)
        print(f"Resuming: {len(completed_tasks)} tasks already completed")
    
    # Determine model directory
    if args.model_type in ('lm', 'text'):
        model_dir = '2b' if args.model_dir is None else args.model_dir
        if model_dir in MODEL_DIRS:
            model_dir = MODEL_DIRS[model_dir]
    else:
        assert args.model_dir is not None, "model_dir is required for prefix models"
        model_dir = args.model_dir
    
    print(f"Loading model from: {model_dir}")
    
    # Load model
    model_type_for_load = 'prefix' if args.model_type == 'prefix' else 'lm'
    tokenizer, model, device = load_model(model_type_for_load, model_dir, False, args)
    model.eval()
    
    print(f"Model loaded on device: {device}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Open output file
    mode = 'a' if args.resume else 'w'
    output_f = open(args.output_file, mode)
    
    # Process tasks
    task_ids = sorted(problems.keys())
    if args.max_tasks:
        task_ids = task_ids[:args.max_tasks]
    
    total_samples = 0
    
    try:
        for task_id in tqdm(task_ids, desc="Generating code"):
            # Skip if already completed
            if task_id in completed_tasks:
                continue
            
            problem = problems[task_id]
            
            # Get prompt based on split
            if args.split == 'complete':
                prompt = problem['complete_prompt']
            else:
                prompt = problem['instruct_prompt']
            
            # Add text prompt for text-prompt mode
            if args.model_type == 'text':
                prefix = PROMPTS[0] if args.control == 'sec' else PROMPTS[1]
                prompt = prefix + prompt
            
            # Handle santacoder (strip prompt)
            if isinstance(model, GPT2LMHeadCustomModel):
                prompt = prompt.strip()
            
            # Generate samples
            samples = generate_samples(model, tokenizer, prompt, args, device)
            
            # Write results
            for solution in samples:
                result = {
                    'task_id': task_id,
                    'solution': solution,
                }
                output_f.write(json.dumps(result) + '\n')
                total_samples += 1
            
            output_f.flush()
            
    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving progress...")
    finally:
        output_f.close()
    
    print("=" * 60)
    print(f"Generation complete!")
    print(f"Total samples generated: {total_samples}")
    print(f"Output saved to: {args.output_file}")
    print("=" * 60)
    print("\nNext steps:")
    print(f"  1. Sanitize: bigcodebench.sanitize --samples {args.output_file}")
    print(f"  2. Calibrate: bigcodebench.calibrate --samples {args.output_file.replace('.jsonl', '-sanitized.jsonl')}")
    print(f"  3. Evaluate: bigcodebench.evaluate --execution local --split {args.split} --subset {args.subset} --samples <sanitized_calibrated_file> --pass_k 1,5,10")


if __name__ == '__main__':
    main()
