#!/bin/bash
#SBATCH --time=00-24:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1

##SBATCH --job-name=cnn_n64_x1_z_0.1_9
#SBATCH --job-name=nc_ps_cnn_n64_x1_z_0.1_10
#SBATCH --err=/leonardo_work/IscrC_GraphMLS/isaezcas/logs/error/%x.%j.err
#SBATCH --out=/leonardo_work/IscrC_GraphMLS/isaezcas/logs/output/%x.%j.out
#SBATCH --account=IscrC_GraphMLS
#SBATCH --partition=boost_usr_prod

set -x

# Number of threads.
n_threads=1

########## Param. file #########

## CNN

#param_file=$WORK_MLS"/codes/MLS-CNN/inputs/input_leonardo_sobol_lcdm_cnn_o_n64_xlum_0.3_z_0.1.toml"
param_file=$WORK_MLS"/codes/MLS-CNN/inputs/input_leonardo_sobol_lcdm_nc_ps_cnn_o_n64_xlum_0.3_z_0.1.toml"

## PS

#param_file=$WORK_MLS"/codes/MLS-CNN/inputs/input_leonardo_sobol_lcdm_nc_ps_n64_xlum_0.3_z_0.1.toml"

## NC

########## Output dir. #########

## CNN

#output_dir=$WORK_MLS"/results/trained_models_leonardo/sobol_lcdm/cnn_o_n64_xlum_0.3_z_0.1"
output_dir=$WORK_MLS"/results/trained_models_leonardo/sobol_lcdm/nc_ps_cnn_o_n64_xlum_0.3_z_0.1"

## PS

#output_dir=$WORK_MLS"/results/trained_models_leonardo/sobol_lcdm/nc_ps_n64_xlum_0.3_z_0.1"

## NC

# Python main script.
exe_python=$WORK_MLS"/codes/MLS-CNN/main.py"

mkdir -p $output_dir
mkdir -p $output_dir/logs

cd $WORK_MLS/codes/MLS-CNN

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

uv run python $exe_python tune -n $n_threads -f $param_file -o $output_dir &> ${output_dir}/logs/tune_10.log
#uv run python $exe_python tune -n $n_threads -f $param_file -o $output_dir &> ${output_dir}/logs/tune_6.log &
#uv run python $exe_python tune -n $n_threads -f $param_file -o $output_dir &> ${output_dir}/logs/tune_7.log &
#uv run python $exe_python tune -n $n_threads -f $param_file -o $output_dir &> ${output_dir}/logs/tune_8.log &

#wait

#uv run python $exe_python train -n $n_threads -f $param_file -o $output_dir &> ${output_dir}/logs/train.log
