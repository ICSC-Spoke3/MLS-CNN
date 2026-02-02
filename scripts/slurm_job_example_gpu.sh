#!/bin/bash
#SBATCH --time=00-24:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1

#SBATCH --job-name=cnn_test

########## Param. file #########

param_file="inputs/input_cnn_example.toml"

########## Output dir. #########

output_dir="results/cnn_example"

# Python main script.
exe_python="main.py"

mkdir -p $output_dir
mkdir -p $output_dir/logs

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Hyperparameter tuning.
python $exe_python tune -n 1 -f $param_file -o $output_dir &> ${output_dir}/logs/tune.log

# Model training with the best hyperparameter combination.
python $exe_python train -n 1 -f $param_file -o $output_dir &> ${output_dir}/logs/train.log
