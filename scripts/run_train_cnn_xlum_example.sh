#!/usr/bin/env sh

# Number of threads used by pytorch in each task.
n_threads=1

# Input parameter file.
param_file="inputs/input_cnn_xlum_example.toml"

# Output directory.
output_dir="/home/isaezcasares/data/mls/nn_inference_tests/cnn_xlum_test"

# Make output directory.
mkdir -p $output_dir

python main.py train -n $n_threads -f $param_file -o $output_dir &> ${output_dir}/train_example.log &
