#!/bin/bash
#SBATCH -A MLS
#SBATCH -J ps_n32_2e13_z_0.1_1
#SBATCH --partition=a100-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=02-00:00:00
#SBATCH --output=/exa/projects/MLS/inigo.saez/logs/output/%x.%j.out
#SBATCH --error=/exa/projects/MLS/inigo.saez/logs/error/%x.%j.err

set -x

# Number of threads.
n_threads=1

########## Param. file: CNN #########

#param_file="/home/users/inigo.saez/codes/MLS-CNN/inputs/input_indaco_sobol_lcdm_cnn_o_n32_mass_2e13_z_0.1.toml"
#param_file="/home/users/inigo.saez/codes/MLS-CNN/inputs/input_indaco_sobol_lcdm_cnn_o_n32_mass_2e13_z_0.1_wo_om_s8.toml"
#param_file="/home/users/inigo.saez/codes/MLS-CNN/inputs/input_indaco_sobol_lcdm_cnn_o_n32_mass_2e13_4e14_4_z_0.1.toml"
#param_file="/home/users/inigo.saez/codes/MLS-CNN/inputs/input_indaco_sobol_lcdm_cnn_o_n64_mass_2e13_z_0.1.toml"
#param_file="/home/users/inigo.saez/codes/MLS-CNN/inputs/input_indaco_sobol_lcdm_cnn_o_n64_mass_2e13_4e14_4_z_0.1.toml"
#param_file="/home/users/inigo.saez/codes/MLS-CNN/inputs/input_indaco_sobol_lcdm_cnn_o_n128_mass_2e13_z_0.1.toml"

########## Param. file: PS #########

param_file="/home/users/inigo.saez/codes/MLS-CNN/inputs/input_indaco_sobol_lcdm_ps_n32_mass_2e13_z_0.1.toml"
#param_file="/home/users/inigo.saez/codes/MLS-CNN/inputs/input_indaco_sobol_lcdm_ps_n64_mass_2e13_z_0.1.toml"
#param_file="/home/users/inigo.saez/codes/MLS-CNN/inputs/input_indaco_sobol_lcdm_ps_n128_mass_2e13_z_0.1.toml"

########## Output dir.: CNN #########

#output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_lcdm/cnn_o_n32_mass_2e13_z_0.1"
#output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_lcdm/cnn_o_n32_mass_2e13_z_0.1_wo_om_s8"
#output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_lcdm/cnn_o_n32_mass_2e13_4e14_4_z_0.1"
#output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_lcdm/cnn_o_n32_mass_2e13_z_0.1_n2800"
#output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_lcdm/cnn_o_n64_mass_2e13_z_0.1"
#output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_lcdm/cnn_o_n64_mass_2e13_z_0.1_n2800_e"
#output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_lcdm/cnn_o_n64_mass_2e13_4e14_4_z_0.1"
#output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_lcdm/cnn_o_n128_mass_2e13_z_0.1"

########## Output dir.: PS #########

#output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_lcdm/ps_n32_mass_2e13_z_0.1"
output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_lcdm/ps_n32_mass_2e13_z_0.1_test_kmin"
#output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_lcdm/ps_n32_mass_2e13_z_0.1_n2800"
#output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_lcdm/ps_n64_mass_2e13_z_0.1"
#output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_lcdm/ps_n64_mass_2e13_z_0.1_n2800"
#output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_lcdm/ps_n128_mass_2e13_z_0.1"

# Python main script.
exe_python="/home/users/inigo.saez/codes/MLS-CNN/main.py"

mkdir -p $output_dir
cd $output_dir
mkdir -p logs

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python $exe_python tune -n $n_threads -f $param_file -o $output_dir &> ${output_dir}/logs/tune_1.log

python $exe_python train -n $n_threads -f $param_file -o $output_dir &> ${output_dir}/logs/train.log
