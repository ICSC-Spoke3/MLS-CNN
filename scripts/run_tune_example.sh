#!/usr/bin/env sh

# Number of independent tasks.
# Each task runs an whole tuning job, but they all take into account the results from the others.
n_tasks=4

# Number of threads used by pytorch in each task.
n_threads=1

# Input parameter file.
param_file="inputs/input_parameters_example.toml"

# Output directory.
output_dir="/home/isaezcasares/data/mls/nn_inference_tests/power_spectrum_test"

# Make output directory.
mkdir -p $output_dir

for i in {1..$n_tasks}
do
    python main.py tune -n $n_threads -f $param_file -o $output_dir &> ${output_dir}/tune_example_${i}.log &

done
