#!/bin/bash
# BigCodeBench Evaluation Examples for SVEN Models
# This file contains example commands for different evaluation scenarios

# ==========================================
# QUICK TEST (5 tasks, greedy decoding)
# ==========================================

# Test prefix model (secure)
python bigcodebench_eval.py \
    --model_type prefix \
    --model_dir ../trained/2b-prefix/checkpoint-last \
    --output_name test-2b-prefix-sec \
    --control sec \
    --max_tasks 5 \
    --greedy

# ==========================================
# FULL EVALUATION - PREFIX MODELS
# ==========================================

# Evaluate 2B prefix model (secure mode)
python bigcodebench_eval.py \
    --model_type prefix \
    --model_dir ../trained/2b-prefix/checkpoint-last \
    --output_name sven-2b-prefix-sec \
    --control sec \
    --split complete \
    --subset full \
    --n_samples 10 \
    --temp 0.8 \
    --pass_k 1,5,10

# Evaluate 2B prefix model (vulnerable mode)
python bigcodebench_eval.py \
    --model_type prefix \
    --model_dir ../trained/2b-prefix/checkpoint-last \
    --output_name sven-2b-prefix-vul \
    --control vul \
    --split complete \
    --subset full \
    --n_samples 10 \
    --temp 0.8 \
    --pass_k 1,5,10

# Evaluate 350M prefix model (secure mode)
python bigcodebench_eval.py \
    --model_type prefix \
    --model_dir ../trained/350m-prefix/checkpoint-last \
    --output_name sven-350m-prefix-sec \
    --control sec \
    --split complete \
    --subset full \
    --n_samples 10 \
    --temp 0.8 \
    --pass_k 1,5,10

# ==========================================
# FULL EVALUATION - BASE LM MODELS
# ==========================================

# Evaluate CodeGen-2B (base model for comparison)
python bigcodebench_eval.py \
    --model_type lm \
    --model_dir 2b \
    --output_name codegen-2b-base \
    --split complete \
    --subset full \
    --n_samples 10 \
    --temp 0.8 \
    --pass_k 1,5,10

# Evaluate CodeGen-350M
python bigcodebench_eval.py \
    --model_type lm \
    --model_dir 350m \
    --output_name codegen-350m-base \
    --split complete \
    --subset full \
    --n_samples 10 \
    --temp 0.8 \
    --pass_k 1,5,10

# ==========================================
# TEXT-PROMPT MODELS
# ==========================================

# Evaluate with text prompt (secure)
python bigcodebench_eval.py \
    --model_type text \
    --model_dir 2b \
    --output_name codegen-2b-text-sec \
    --control sec \
    --split complete \
    --subset full \
    --n_samples 10 \
    --temp 0.8 \
    --pass_k 1,5,10

# ==========================================
# INSTRUCT SPLIT
# ==========================================

# Evaluate on instruct split (natural language instructions)
python bigcodebench_eval.py \
    --model_type prefix \
    --model_dir ../trained/2b-prefix/checkpoint-last \
    --output_name sven-2b-prefix-sec-instruct \
    --control sec \
    --split instruct \
    --subset full \
    --n_samples 10 \
    --temp 0.8 \
    --pass_k 1,5,10

# ==========================================
# HARD SUBSET ONLY
# ==========================================

# Evaluate only on hard subset
python bigcodebench_eval.py \
    --model_type prefix \
    --model_dir ../trained/2b-prefix/checkpoint-last \
    --output_name sven-2b-prefix-sec-hard \
    --control sec \
    --split complete \
    --subset hard \
    --n_samples 10 \
    --temp 0.8 \
    --pass_k 1,5,10

# ==========================================
# GREEDY DECODING (Pass@1 only)
# ==========================================

# Greedy decoding for deterministic results
python bigcodebench_eval.py \
    --model_type prefix \
    --model_dir ../trained/2b-prefix/checkpoint-last \
    --output_name sven-2b-prefix-sec-greedy \
    --control sec \
    --split complete \
    --subset full \
    --greedy \
    --pass_k 1

# ==========================================
# STEP-BY-STEP (Manual Pipeline)
# ==========================================

# Step 1: Generate only
python bigcodebench_gen.py \
    --model_type prefix \
    --model_dir ../trained/2b-prefix/checkpoint-last \
    --output_name manual-test \
    --control sec \
    --n_samples 10

# Step 2: Sanitize and calibrate manually
cd ../experiments/bigcodebench
bigcodebench.sanitize --samples manual-test--bigcodebench-complete--hf-0.8-10.jsonl
bigcodebench.calibrate --samples manual-test--bigcodebench-complete--hf-0.8-10-sanitized.jsonl

# Step 3: Evaluate
bigcodebench.evaluate \
    --execution local \
    --split complete \
    --subset full \
    --samples manual-test--bigcodebench-complete--hf-0.8-10-sanitized_calibrated.jsonl \
    --pass_k 1,5,10 \
    --save_pass_rate

# ==========================================
# RESUME INTERRUPTED RUN
# ==========================================

# Resume if generation was interrupted
python bigcodebench_eval.py \
    --model_type prefix \
    --model_dir ../trained/2b-prefix/checkpoint-last \
    --output_name sven-2b-prefix-sec \
    --control sec \
    --n_samples 10 \
    --resume

# ==========================================
# USING DOCKER FOR SAFE EVALUATION
# ==========================================

# Use Docker for evaluation (recommended)
python bigcodebench_eval.py \
    --model_type prefix \
    --model_dir ../trained/2b-prefix/checkpoint-last \
    --output_name sven-2b-prefix-sec-docker \
    --control sec \
    --n_samples 10 \
    --use_docker
