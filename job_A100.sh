#!/bin/bash

#SBATCH --job-name=COCONUT_exps                     # Request 1 compute node per job instance
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=36GB                     # Request 2GB of RAM per job instance
#SBATCH --time=06:00:00                # Request 10 mins per job instance
#SBATCH --output=/scratch/vt2369/COCONUT/coconut/coconut_exp_%A.out  # The output will be saved here. %A will be replaced by the slurm job ID, and %a will be replaced by the SLURM_ARRAY_TASK_ID
#SBATCH --mail-user=vt2369@nyu.edu    # Email address
#SBATCH --mail-type=BEGIN,END         # Send an email when all the instances of this job are completed
module purge                          # unload all currently loaded modules in the environment

echo "Running: CoT eval for GPT2"
/scratch/vt2369/COCONUT/coconut/spin_env_rw.sh torchrun --nnodes 1 --nproc_per_node 2 run.py args/gsm_gpt2_cot.yaml 
echo "Running: COCONUT eval for GPT2"
/scratch/vt2369/COCONUT/coconut/spin_env_rw.sh torchrun --nnodes 1 --nproc_per_node 2 run.py args/gsm_gpt2_coconut_eval.yaml

echo "Running: CoT eval for DeepSeek-Distilled-Qwen1.5B, Dataset: GSM8k"
/scratch/vt2369/COCONUT/coconut/spin_env_rw.sh torchrun --nnodes 1 --nproc_per_node 2 run.py args/gsm_deepseek_cot_gsm8k.yaml
echo "Running: Cot eval for DeepSeek-Distilled-Qwen1.5B, Dataset: GSM8k-hard"
/scratch/vt2369/COCONUT/coconut/spin_env_rw.sh torchrun --nnodes 1 --nproc_per_node 2 run.py args/gsm_deepseek_cot_gsm8khard.yaml
