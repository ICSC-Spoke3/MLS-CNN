#!/bin/bash
#SBATCH -A MLS
#SBATCH -J cnn16_lcdm_m7_z_0.2_gauss_f7_2
#SBATCH --partition=a100-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=01-00:00:00
#SBATCH --output=/exa/projects/MLS/inigo.saez/logs/output/%x.%j.out
#SBATCH --error=/exa/projects/MLS/inigo.saez/logs/error/%x.%j.err

set -x

# Number of threads.
n_threads=1

########## Param. file: CNN #########

## NDIM2

#param_file="/home/users/inigo.saez/codes/MLS-CNN/inputs/input_indaco_sobol_ndim2_cnn_n16_mass_1e14_z_0.25.toml"

#param_file="/home/users/inigo.saez/codes/MLS-CNN/inputs/input_indaco_sobol_ndim2_cnn_n16_mass_1e14_1e15_10_z_0.25.toml"
#param_file="/home/users/inigo.saez/codes/MLS-CNN/inputs/input_indaco_sobol_ndim2_cnn_n16_skew_mass_1e14_1e15_10_z_0.25.toml"

#param_file="/home/users/inigo.saez/codes/MLS-CNN/inputs/input_indaco_sobol_ndim2_cnn_n16_xlum_1.3e-1_1e0_5_z_0.toml"
#param_file="/home/users/inigo.saez/codes/MLS-CNN/inputs/input_indaco_sobol_ndim2_cnn_n16_xlum_1.3e-1_1e0_5_z_0.25.toml"
#param_file="/home/users/inigo.saez/codes/MLS-CNN/inputs/input_indaco_sobol_ndim2_cnn_n16_xlum_1.3e-1_1e0_5_z_0_0.25.toml"
#param_file="/home/users/inigo.saez/codes/MLS-CNN/inputs/input_indaco_sobol_ndim2_cnn_n16_xlum_1.3e-1_1e0_5_z_0_0.25_0.5.toml"

## LCDM

#param_file="/home/users/inigo.saez/codes/MLS-CNN/inputs/input_indaco_sobol_lcdm_cnn_n16_mass_3.6e13_7.7e14_7_z_0.2.toml"
#param_file="/home/users/inigo.saez/codes/MLS-CNN/inputs/input_indaco_sobol_lcdm_cnn_n16_mass_3.6e13_7.7e14_7_z_0.2_f7.toml"

#param_file="/home/users/inigo.saez/codes/MLS-CNN/inputs/input_indaco_sobol_lcdm_cnn_n16_mass_3.6e13_7.7e14_7_z_0.2_nm.toml"
#param_file="/home/users/inigo.saez/codes/MLS-CNN/inputs/input_indaco_sobol_lcdm_cnn_n16_mass_3.6e13_7.7e14_7_z_0.2_f7_nm.toml"

#param_file="/home/users/inigo.saez/codes/MLS-CNN/inputs/input_indaco_sobol_lcdm_cnn_n16_mass_3.6e13_7.7e14_7_z_0.2_gauss.toml"
param_file="/home/users/inigo.saez/codes/MLS-CNN/inputs/input_indaco_sobol_lcdm_cnn_n16_mass_3.6e13_7.7e14_7_z_0.2_gauss_f7.toml"

#param_file="/home/users/inigo.saez/codes/MLS-CNN/inputs/input_indaco_sobol_lcdm_cnn_n32_mass_3.6e13_7.7e14_7_z_0.2.toml"
#param_file="/home/users/inigo.saez/codes/MLS-CNN/inputs/input_indaco_sobol_lcdm_cnn_n32_mass_3.6e13_7.7e14_7_z_0.2_f7.toml"

#param_file="/home/users/inigo.saez/codes/MLS-CNN/inputs/input_indaco_sobol_lcdm_cnn_n32_mass_3.6e13_7.7e14_7_z_0.2_nm.toml"

#param_file="/home/users/inigo.saez/codes/MLS-CNN/inputs/input_indaco_sobol_lcdm_cnn_n32_mass_3.6e13_7.7e14_7_z_0.2_gauss.toml"

#param_file="/home/users/inigo.saez/codes/MLS-CNN/inputs/input_indaco_sobol_lcdm_cnn_n64_mass_3.6e13_7.7e14_7_z_0.2.toml"

#param_file="/home/users/inigo.saez/codes/MLS-CNN/inputs/input_indaco_sobol_lcdm_cnn_n64_mass_3.6e13_7.7e14_7_z_0.2_nm.toml"

#param_file="/home/users/inigo.saez/codes/MLS-CNN/inputs/input_indaco_sobol_lcdm_cnn_n16_xlum_1.3e-1_1e0_5_z_0.2.toml"
#param_file="/home/users/inigo.saez/codes/MLS-CNN/inputs/input_indaco_sobol_lcdm_cnn_n16_xlum_1.3e-1_1e0_5_z_0.2_nm.toml"

#param_file="/home/users/inigo.saez/codes/MLS-CNN/inputs/input_indaco_sobol_lcdm_cnn_n32_xlum_1.3e-1_1e0_5_z_0.2.toml"

## LCDM REFLEX

#param_file="/home/users/inigo.saez/codes/MLS-CNN/inputs/input_indaco_reflex_sobol_lcdm_cnn_n16_xlum_1.3e-1_1e0_5_z_0.2_nm.toml"

#param_file="/home/users/inigo.saez/codes/MLS-CNN/inputs/input_indaco_reflex_sobol_lcdm_fc_cnn_n16_xlum_1.3e-1_1e0_5_z_0.2_nm.toml"

########## Output dir.: CNN #########

# NDIM2

#output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_ndim2/cnn_n16_mass_1e14_z_0.25"

#output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_ndim2/cnn_n16_mass_1e14_1e15_10_z_0.25"
#output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_ndim2/cnn_n16_skew_mass_1e14_1e15_10_z_0.25"

#output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_ndim2/cnn_n16_xlum_1.3e-1_1e0_5_z_0"
#output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_ndim2/cnn_n16_xlum_1.3e-1_1e0_5_z_0.25"
#output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_ndim2/cnn_n16_xlum_1.3e-1_1e0_5_z_0_0.25"
#output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_ndim2/cnn_n16_xlum_1.3e-1_1e0_5_z_0_0.25_0.5"

## LCDM

#output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_lcdm/cnn_n16_mass_3.6e13_7.7e14_7_z_0.2"
#output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_lcdm/cnn_n16_mass_3.6e13_7.7e14_7_z_0.2_f7"

#output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_lcdm/cnn_n16_mass_3.6e13_7.7e14_7_z_0.2_nm"
#output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_lcdm/cnn_n16_mass_3.6e13_7.7e14_7_z_0.2_f7_nm"

#output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_lcdm/cnn_n16_mass_3.6e13_7.7e14_7_z_0.2_gauss"
output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_lcdm/cnn_n16_mass_3.6e13_7.7e14_7_z_0.2_gauss_f7"

#output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_lcdm/cnn_n32_mass_3.6e13_7.7e14_7_z_0.2"
#output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_lcdm/cnn_n32_mass_3.6e13_7.7e14_7_z_0.2_f7"

#output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_lcdm/cnn_n32_mass_3.6e13_7.7e14_7_z_0.2_nm"

#output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_lcdm/cnn_n32_mass_3.6e13_7.7e14_7_z_0.2_gauss"

#output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_lcdm/cnn_n64_mass_3.6e13_7.7e14_7_z_0.2"

#output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_lcdm/cnn_n64_mass_3.6e13_7.7e14_7_z_0.2_nm"

#output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_lcdm/cnn_n16_xlum_1.3e-1_1e0_5_z_0.2"
#output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_lcdm/cnn_n16_xlum_1.3e-1_1e0_5_z_0.2_nm"

#output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_lcdm/cnn_n32_xlum_1.3e-1_1e0_5_z_0.2"

## LCDM REFLEX

#output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco_reflex/sobol_lcdm/cnn_n16_xlum_1.3e-1_1e0_5_z_0.2"

#output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco_reflex/sobol_lcdm_fc/cnn_n16_xlum_1.3e-1_1e0_5_z_0.2"

# Python main script.
exe_python="/home/users/inigo.saez/codes/MLS-CNN/main.py"

mkdir -p $output_dir
cd $output_dir
mkdir -p logs

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python $exe_python tune -n $n_threads -f $param_file -o $output_dir &> ${output_dir}/logs/tune_2.log

python $exe_python train -n $n_threads -f $param_file -o $output_dir &> ${output_dir}/logs/train.log
