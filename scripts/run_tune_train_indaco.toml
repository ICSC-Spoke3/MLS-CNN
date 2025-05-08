#!/bin/bash
#SBATCH -A MLS
#SBATCH -J tu_ps_n128_m10_z_0
#SBATCH --partition=a100-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01-00:00:00
#SBATCH --output=/exa/projects/MLS/inigo.saez/logs/output/%x.%j.out
#SBATCH --error=/exa/projects/MLS/inigo.saez/logs/error/%x.%j.err

set -x

# Number of threads.
n_threads=1

########## Param. file: PS #########

param_file="/home/users/inigo.saez/codes/MLS-CNN/inputs/input_indaco_ps_n128_mass_1e14_1e15_10_z_0.toml"

########## Output dir: PS #########

output_dir="/exa/projects/MLS/inigo.saez/trained_models/ps_n128_mass_1e14_1e15_10_z_0"

# Python main script.
exe_python="/home/users/inigo.saez/codes/MLS-CNN/main.py"

mkdir -p $output_dir
cd $output_dir
mkdir -p logs

python $exe_python tune -n $n_threads -f $param_file -o $output_dir &> ${output_dir}/logs/tune.log

python $exe_python train -f $param_file &> logs/train.log
