#!/bin/bash
#SBATCH -A MLS
#SBATCH -J nc_cnn_n64_x1mm_z_0.1_12
#SBATCH --partition=a100-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --time=03-00:00:00
#SBATCH --output=/exa/projects/MLS/inigo.saez/logs/output/%x.%j.out
#SBATCH --error=/exa/projects/MLS/inigo.saez/logs/error/%x.%j.err

set -x

# Number of threads.
n_threads=1

########## Param. file #########

## CNN

#param_file="/home/users/inigo.saez/codes/MLS-CNN/inputs/input_indaco_sobol_lcdm_nc_cnn_o_n32_xlum_0.3_z_0.1.toml"
#param_file="/home/users/inigo.saez/codes/MLS-CNN/inputs/input_indaco_sobol_lcdm_nc_cnn_o_n32_xlum_0.3_mm_z_0.1.toml"

#param_file="/home/users/inigo.saez/codes/MLS-CNN/inputs/input_indaco_sobol_lcdm_nc_cnn_o_n64_xlum_0.3_z_0.1.toml"
param_file="/home/users/inigo.saez/codes/MLS-CNN/inputs/input_indaco_sobol_lcdm_nc_cnn_o_n64_xlum_0.3_mm_z_0.1.toml"
#param_file="/home/users/inigo.saez/codes/MLS-CNN/inputs/input_indaco_sobol_lcdm_nc_cnn_o_n64_xlum_0.3_1.08_2_z_0.1.toml"
#param_file="/home/users/inigo.saez/codes/MLS-CNN/inputs/input_indaco_sobol_lcdm_nc_cnn_o_n64_xlum_0.3_3_3_z_0.1.toml"

#param_file="/home/users/inigo.saez/codes/MLS-CNN/inputs/input_indaco_sobol_lcdm_nc_cnn_o_n64_xlum_0.3_z_0.1_f1.toml"
#param_file="/home/users/inigo.saez/codes/MLS-CNN/inputs/input_indaco_sobol_lcdm_nc_cnn_o_n64_xlum_0.3_z_0.1_f3.toml"
#param_file="/home/users/inigo.saez/codes/MLS-CNN/inputs/input_indaco_sobol_lcdm_nc_cnn_o_n64_xlum_0.3_z_0.1_f7.toml"

## PS

#param_file="/home/users/inigo.saez/codes/MLS-CNN/inputs/input_indaco_sobol_lcdm_nc_ps_n32_xlum_0.3_z_0.1.toml"

#param_file="/home/users/inigo.saez/codes/MLS-CNN/inputs/input_indaco_sobol_lcdm_nc_ps_n64_xlum_0.3_z_0.1.toml"
#param_file="/home/users/inigo.saez/codes/MLS-CNN/inputs/input_indaco_sobol_lcdm_nc_ps_n64_xlum_0.3_mm_z_0.1.toml"

#param_file="/home/users/inigo.saez/codes/MLS-CNN/inputs/input_indaco_sobol_lcdm_nc_ps_n64_xlum_0.3_z_0.1_nc_2048_nx_8.toml"
#param_file="/home/users/inigo.saez/codes/MLS-CNN/inputs/input_indaco_sobol_lcdm_nc_ps_n64_xlum_0.3_z_0.1_nc_1024_nx_8.toml"
#param_file="/home/users/inigo.saez/codes/MLS-CNN/inputs/input_indaco_sobol_lcdm_nc_ps_n64_xlum_0.3_z_0.1_nc_4096_nx_4.toml"
#param_file="/home/users/inigo.saez/codes/MLS-CNN/inputs/input_indaco_sobol_lcdm_nc_ps_n64_xlum_0.3_z_0.1_nc_2048_nx_4.toml"
#param_file="/home/users/inigo.saez/codes/MLS-CNN/inputs/input_indaco_sobol_lcdm_nc_ps_n64_xlum_0.3_z_0.1_nc_1024_nx_4.toml"
#param_file="/home/users/inigo.saez/codes/MLS-CNN/inputs/input_indaco_sobol_lcdm_nc_ps_n64_xlum_0.3_z_0.1_nc_4096_nx_2.toml"
#param_file="/home/users/inigo.saez/codes/MLS-CNN/inputs/input_indaco_sobol_lcdm_nc_ps_n64_xlum_0.3_z_0.1_nc_2048_nx_2.toml"
#param_file="/home/users/inigo.saez/codes/MLS-CNN/inputs/input_indaco_sobol_lcdm_nc_ps_n64_xlum_0.3_z_0.1_nc_1024_nx_2.toml"

## NC

#param_file="/home/users/inigo.saez/codes/MLS-CNN/inputs/input_indaco_sobol_lcdm_nc_xlum_0.3_z_0.1.toml"

########## Output dir. #########

## CNN

#output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_lcdm/nc_cnn_o_n32_xlum_0.3_z_0.1"
#output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_lcdm/nc_cnn_o_n32_xlum_0.3_mm_z_0.1"

#output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_lcdm/nc_cnn_o_n64_xlum_0.3_z_0.1"
output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_lcdm/nc_cnn_o_n64_xlum_0.3_mm_z_0.1"
#output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_lcdm/nc_cnn_o_n64_xlum_0.3_1.08_2_z_0.1"
#output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_lcdm/nc_cnn_o_n64_xlum_0.3_3_3_z_0.1"

#output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_lcdm/nc_cnn_o_n64_xlum_0.3_z_0.1_f1"
#output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_lcdm/nc_cnn_o_n64_xlum_0.3_z_0.1_f3"
#output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_lcdm/nc_cnn_o_n64_xlum_0.3_z_0.1_f7"

## PS

#output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_lcdm/nc_ps_n32_xlum_0.3_z_0.1"

#output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_lcdm/nc_ps_n64_xlum_0.3_z_0.1"
#output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_lcdm/nc_ps_n64_xlum_0.3_mm_z_0.1"

#output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_lcdm/nc_ps_n64_xlum_0.3_z_0.1_nc_2048_nx_8"
#output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_lcdm/nc_ps_n64_xlum_0.3_z_0.1_nc_1024_nx_8"
#output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_lcdm/nc_ps_n64_xlum_0.3_z_0.1_nc_4096_nx_4"
#output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_lcdm/nc_ps_n64_xlum_0.3_z_0.1_nc_2048_nx_4"
#output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_lcdm/nc_ps_n64_xlum_0.3_z_0.1_nc_1024_nx_4"
#output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_lcdm/nc_ps_n64_xlum_0.3_z_0.1_nc_4096_nx_2"
#output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_lcdm/nc_ps_n64_xlum_0.3_z_0.1_nc_2048_nx_2"
#output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_lcdm/nc_ps_n64_xlum_0.3_z_0.1_nc_1024_nx_2"

## NC

#output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_lcdm/nc_xlum_0.3_z_0.1"

# Python main script.
exe_python="/home/users/inigo.saez/codes/MLS-CNN/main.py"

mkdir -p $output_dir
cd $output_dir
mkdir -p logs

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python $exe_python tune -n $n_threads -f $param_file -o $output_dir &> ${output_dir}/logs/tune_1.log &
python $exe_python tune -n $n_threads -f $param_file -o $output_dir &> ${output_dir}/logs/tune_2.log &

wait

python $exe_python train -n $n_threads -f $param_file -o $output_dir &> ${output_dir}/logs/train.log
