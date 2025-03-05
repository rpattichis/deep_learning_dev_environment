#!/bin/bash -l
# # run the following once to create a logs dir
# # mkdir --parents "$HOME/logs"
# #SBATCH --partition=defq                 # a6000 -> gives you access to this node only (3 / 4 cards accessible) -> normal-a6000 for qos              
#SBATCH --job-name=deepspeed-train-ex          # Job name
# # SBATCH --exclude=gpu01                    # Nodes to exclude (for multiple separate with commas)
#SBATCH --ntasks-per-node=1                 
#SBATCH --cpus-per-task=1                   # CPU cores (1-32)
#SBATCH --mem=128G                          # Total RAM (1-500) NOTE: change this 
#SBATCH --gres=gpu:4                        # Number of GPUs to use (if no GPU required, use 0)
# # SBATCH --output=./logs/job.%A_%a.out       # Stdout & Stderr (%j=jobId)
#SBATCH --output=/trinity/home/rpattichis/logs/job.%j.out       # Stdout & Stderr (%j=jobId)
#SBATCH --time=1-00:00:00                   # Time limit days-hrs:min:sec
#SBATCH --qos=normal                        # normal=24hrs (high), long=7days (medium), preemptive (low)
# # SBATCH --array=0-14

export MAIN_PATH=/trinity/home/rpattichis
export TRITON_CACHE_DIR=$MAIN_PATH/data/cygr-project/.triton_cache
# export CUDA_LAUNCH_BLOCKING=1
# export DS_SKIP_CUDA_CHECK=1 # NOTE: I should figure out the solution to this package mismatch
# export TOKENIZERS_PARALLELISM=false
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export CUDA_HOME="/usr/local/cuda"
#export LIBRARY_PATH="/usr/local/cuda/lib64:$LIBRARY_PATH"
#export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

# activate conda environment
conda activate testing

accelerate launch $MAIN_PATH/data/cygr-project/train_and_save_ex.py \
--seed 100 \
--model_name_or_path "ilsp/Meltemi-7B-Instruct-v1.5" \
--cache_dir "$MAIN_PATH/data/cygr-project/cache/" \
--datapath "$MAIN_PATH/data/cygr-project/" \
--datasets 'train.csv,val.csv,test.csv' \
--format_chatbot_turns True \
--chat_template_format "meltemi" \
--use_packing True \
--add_special_tokens False \
--append_concat_token False \
--max_seq_len 512 \
--num_train_epochs 1 \
--logging_steps 5 \
--log_level "info" \
--logging_strategy "epoch" \
--save_strategy "epoch" \
--eval_strategy "epoch" \
--bf16 True \
--learning_rate 1e-4 \
--lr_scheduler_type "cosine" \
--weight_decay 1e-4 \
--warmup_ratio 0.0 \
--max_grad_norm 1.0 \
--output_dir "$MAIN_PATH/data/cygr-project/model-outputs/meltemi-small" \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 4 \
--use_reentrant False \
--use_flash_attn True \
--use_peft_lora True \
--lora_r 8 \
--lora_alpha 16 \
--lora_dropout 0.1 \
--lora_target_modules "all-linear" \
--use_4bit_quantization False \
--save_model True