#!/bin/bash
#SBATCH -A MLS
#SBATCH -J ps_2d_m1_mm_z_0.25
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

## NDIM2

#param_file="/home/users/inigo.saez/codes/MLS-CNN/inputs/input_indaco_sobol_ndim2_ps_mass_1e14_z_0.25.toml"

param_file="/home/users/inigo.saez/codes/MLS-CNN/inputs/input_indaco_sobol_ndim2_ps_mass_1e14_mark_mean_z_0.25.toml"

#param_file="/home/users/inigo.saez/codes/MLS-CNN/inputs/input_indaco_sobol_ndim2_ps_xlum_1.3e-1_z_0.25.toml"

#param_file="/home/users/inigo.saez/codes/MLS-CNN/inputs/input_indaco_sobol_ndim2_ps_xlum_1.3e-1_mark_mean_z_0.25.toml"

########## Param. file: NC #########

## NDIM2

#param_file="/home/users/inigo.saez/codes/MLS-CNN/inputs/input_indaco_sobol_ndim2_nc_mass_1e14_1e15_10_z_0.toml"
#param_file="/home/users/inigo.saez/codes/MLS-CNN/inputs/input_indaco_sobol_ndim2_nc_mass_1e14_1e15_10_z_0.25.toml"

#param_file="/home/users/inigo.saez/codes/MLS-CNN/inputs/input_indaco_sobol_ndim2_nc_xlum_1.3e-1_1e0_5_z_0.toml"
#param_file="/home/users/inigo.saez/codes/MLS-CNN/inputs/input_indaco_sobol_ndim2_nc_xlum_1.3e-1_1e0_5_z_0.25.toml"
#param_file="/home/users/inigo.saez/codes/MLS-CNN/inputs/input_indaco_sobol_ndim2_nc_xlum_1.3e-1_1e0_5_z_0_0.25.toml"
#param_file="/home/users/inigo.saez/codes/MLS-CNN/inputs/input_indaco_sobol_ndim2_nc_xlum_1.3e-1_1e0_5_z_0_0.25_0.5.toml"

## LCDM

#param_file="/home/users/inigo.saez/codes/MLS-CNN/inputs/input_indaco_sobol_lcdm_nc_mass_1e14_1e15_10_z_0.2.toml"
#param_file="/home/users/inigo.saez/codes/MLS-CNN/inputs/input_indaco_sobol_lcdm_nc_mass_1e14_1e15_10_z_0.2_0.5.toml"

########## Param. file: CNN #########

## NDIM2

#param_file="/home/users/inigo.saez/codes/MLS-CNN/inputs/input_indaco_sobol_ndim2_cnn_n16_mass_1e14_z_0.25.toml"

#param_file="/home/users/inigo.saez/codes/MLS-CNN/inputs/input_indaco_sobol_ndim2_cnn_n16_mass_1e14_1e15_10_z_0.25.toml"

#param_file="/home/users/inigo.saez/codes/MLS-CNN/inputs/input_indaco_sobol_ndim2_cnn_n16_mass_1e14_mark_mean_z_0.25.toml"

#param_file="/home/users/inigo.saez/codes/MLS-CNN/inputs/input_indaco_sobol_ndim2_cnn_n16_xlum_1.3e-1_1e0_5_z_0.toml"
#param_file="/home/users/inigo.saez/codes/MLS-CNN/inputs/input_indaco_sobol_ndim2_cnn_n16_xlum_1.3e-1_1e0_5_z_0.25.toml"
#param_file="/home/users/inigo.saez/codes/MLS-CNN/inputs/input_indaco_sobol_ndim2_cnn_n16_xlum_1.3e-1_1e0_5_z_0_0.25.toml"
#param_file="/home/users/inigo.saez/codes/MLS-CNN/inputs/input_indaco_sobol_ndim2_cnn_n16_xlum_1.3e-1_1e0_5_z_0_0.25_0.5.toml"

#param_file="/home/users/inigo.saez/codes/MLS-CNN/inputs/input_indaco_sobol_ndim2_cnn_n16_xlum_1.3e-1_mark_mean_z_0.25.toml"

## LCDM

#param_file="/home/users/inigo.saez/codes/MLS-CNN/inputs/input_indaco_sobol_lcdm_cnn_n16_mass_1e14_1e15_10_z_0.2.toml"
#param_file="/home/users/inigo.saez/codes/MLS-CNN/inputs/input_indaco_sobol_lcdm_cnn_n16_mass_1e14_1e15_10_z_0.2_0.5.toml"

########## Output dir: PS #########

## NDIM2

#output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_ndim2/ps_mass_1e14_z_0.25"

output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_ndim2/ps_mass_1e14_mark_mean_z_0.25"

#output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_ndim2/ps_xlum_1.3e-1_z_0.25"

#output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_ndim2/ps_xlum_1.3e-1_mark_mean_z_0.25"

########## Output dir.: NC #########

## NDIM2

#output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_ndim2/nc_mass_1e14_1e15_10_z_0"
#output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_ndim2/nc_mass_1e14_1e15_10_z_0.25"

#output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_ndim2/nc_xlum_1.3e-1_1e0_5_z_0"
#output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_ndim2/nc_xlum_1.3e-1_1e0_5_z_0.25"
#output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_ndim2/nc_xlum_1.3e-1_1e0_5_z_0_0.25"
#output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_ndim2/nc_xlum_1.3e-1_1e0_5_z_0_0.25_0.5"

## LCDM

#output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_lcdm/nc_mass_1e14_1e15_10_z_0.2"
#output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_lcdm/nc_mass_1e14_1e15_10_z_0.2_0.5"

########## Output dir.: CNN #########

# NDIM2

#output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_ndim2/cnn_n16_mass_1e14_z_0.25"

#output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_ndim2/cnn_n16_mass_1e14_1e15_10_z_0.25"

#output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_ndim2/cnn_n16_mass_1e14_mark_mean_z_0.25"

#output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_ndim2/cnn_n16_xlum_1.3e-1_1e0_5_z_0"
#output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_ndim2/cnn_n16_xlum_1.3e-1_1e0_5_z_0.25"
#output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_ndim2/cnn_n16_xlum_1.3e-1_1e0_5_z_0_0.25"
#output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_ndim2/cnn_n16_xlum_1.3e-1_1e0_5_z_0_0.25_0.5"

#output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_ndim2/cnn_n16_xlum_1.3e-1_mark_mean_z_0.25"

## LCDM

#output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_lcdm/cnn_n16_mass_1e14_1e15_10_z_0.2"
#output_dir="/exa/projects/MLS/inigo.saez/trained_models_indaco/sobol_lcdm/cnn_n16_mass_1e14_1e15_10_z_0.2_0.5"

# Python main script.
exe_python="/home/users/inigo.saez/codes/MLS-CNN/main.py"

mkdir -p $output_dir
cd $output_dir
mkdir -p logs

python $exe_python tune -n $n_threads -f $param_file -o $output_dir &> ${output_dir}/logs/tune.log

python $exe_python train -n $n_threads -f $param_file -o $output_dir &> ${output_dir}/logs/train.log
