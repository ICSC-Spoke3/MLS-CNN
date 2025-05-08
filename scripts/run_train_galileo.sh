#!/bin/bash
#SBATCH -J tr_nc_x5_z_0_xBA_s8
#SBATCH --partition=astro
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=01-00:00:00
#SBATCH --output=/home/inigosaez/mls/logs/output/%x.%j.out
#SBATCH --error=/home/inigosaez/mls/logs/error/%x.%j.err

set -x

########## Param. file: NC #########

#param_file="/home/inigosaez/mls/nn_inference/inputs/input_galileo_nc_xlum_1.3e-1_1e0_5_z_0.toml"
param_file="/home/inigosaez/mls/nn_inference/inputs/input_galileo_nc_xlum_1.3e-1_1e0_5_z_0_xlumBA4_sobol8.toml"

########## Output dir: NC #########

#output_dir="/home/inigosaez/mls/trained_models/number_counts_xlum_1.3e-1_1e0_5_z_0"
output_dir="/home/inigosaez/mls/trained_models/number_counts_xlum_1.3e-1_1e0_5_z_0_xlumBA4_sobol8"

exe_python="/home/inigosaez/mls/nn_inference/main.py"

mkdir -p $output_dir
cd $output_dir
mkdir -p logs

python $exe_python train -f $param_file &> logs/train.log

