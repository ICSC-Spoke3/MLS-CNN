#!/bin/bash
#SBATCH --time=00-24:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1

#SBATCH --job-name=nc_ps_cnn_n64_x1_z_0.1_tr
##SBATCH --job-name=nc_ps_n32_x1_z_0.1
#SBATCH --err=/leonardo_work/IscrC_GraphMLS/isaezcas/logs/error/%x.%j.err
#SBATCH --out=/leonardo_work/IscrC_GraphMLS/isaezcas/logs/output/%x.%j.out
#SBATCH --account=IscrC_GraphMLS
#SBATCH --partition=boost_usr_prod

set -x

# Number of threads.
n_threads=1

########## Param. file #########

## NC

#param_file=$WORK_MLS"/codes/MLS-CNN/inputs/input_leonardo_sobol_lcdm_nc_xlum_0.3_z_0.1.toml"

## CNN

#param_file=$WORK_MLS"/codes/MLS-CNN/inputs/input_leonardo_sobol_lcdm_cnn_o_n64_xlum_0.3_z_0.1.toml"
param_file=$WORK_MLS"/codes/MLS-CNN/inputs/input_leonardo_sobol_lcdm_nc_ps_cnn_o_n64_xlum_0.3_z_0.1.toml"

#param_file=$WORK_MLS"/codes/MLS-CNN/inputs/input_leonardo_sobol_lcdm_nc_ps_kf0.33_cnn_o_n64_xlum_0.3_z_0.1.toml"
#param_file=$WORK_MLS"/codes/MLS-CNN/inputs/input_leonardo_sobol_lcdm_nc_ps_kf0.35_cnn_o_n64_xlum_0.3_z_0.1.toml"

#param_file=$WORK_MLS"/codes/MLS-CNN/inputs/input_leonardo_sobol_lcdm_nc_cnn_o_n32_xlum_0.3_z_0.1.toml"
#param_file=$WORK_MLS"/codes/MLS-CNN/inputs/input_leonardo_sobol_lcdm_nc_cnn_o_n16_xlum_0.3_z_0.1.toml"
#param_file=$WORK_MLS"/codes/MLS-CNN/inputs/input_leonardo_sobol_lcdm_nc_cnn_o_n8_xlum_0.3_z_0.1.toml"

## PS

#param_file=$WORK_MLS"/codes/MLS-CNN/inputs/input_leonardo_sobol_lcdm_nc_ps_n64_xlum_0.3_z_0.1.toml"
#param_file=$WORK_MLS"/codes/MLS-CNN/inputs/input_leonardo_sobol_lcdm_nc_ps_n32_xlum_0.3_z_0.1.toml"
#param_file=$WORK_MLS"/codes/MLS-CNN/inputs/input_leonardo_sobol_lcdm_nc_ps_n16_xlum_0.3_z_0.1.toml"

#param_file=$WORK_MLS"/codes/MLS-CNN/inputs/input_leonardo_sobol_lcdm_nc_ps_n64_kf0.8_xlum_0.3_z_0.1.toml"
#param_file=$WORK_MLS"/codes/MLS-CNN/inputs/input_leonardo_sobol_lcdm_nc_ps_n64_kf0.6_xlum_0.3_z_0.1.toml"
#param_file=$WORK_MLS"/codes/MLS-CNN/inputs/input_leonardo_sobol_lcdm_nc_ps_n64_kf0.4_xlum_0.3_z_0.1.toml"
#param_file=$WORK_MLS"/codes/MLS-CNN/inputs/input_leonardo_sobol_lcdm_nc_ps_n64_kf0.37_xlum_0.3_z_0.1.toml"
#param_file=$WORK_MLS"/codes/MLS-CNN/inputs/input_leonardo_sobol_lcdm_nc_ps_n64_kf0.36_xlum_0.3_z_0.1.toml"
#param_file=$WORK_MLS"/codes/MLS-CNN/inputs/input_leonardo_sobol_lcdm_nc_ps_n64_kf0.34_xlum_0.3_z_0.1.toml"

#param_file=$WORK_MLS"/codes/MLS-CNN/inputs/input_leonardo_sobol_lcdm_nc_ps_n32_kf0.33_xlum_0.3_z_0.1.toml"
#param_file=$WORK_MLS"/codes/MLS-CNN/inputs/input_leonardo_sobol_lcdm_nc_ps_n16_kf0.33_xlum_0.3_z_0.1.toml"

#param_file=$WORK_MLS"/codes/MLS-CNN/inputs/input_leonardo_sobol_lcdm_nc_ps_n64_kf0.35_xlum_0.3_z_0.1.toml"
#param_file=$WORK_MLS"/codes/MLS-CNN/inputs/input_leonardo_sobol_lcdm_nc_ps_n32_kf0.35_xlum_0.3_z_0.1.toml"
#param_file=$WORK_MLS"/codes/MLS-CNN/inputs/input_leonardo_sobol_lcdm_nc_ps_n16_kf0.35_xlum_0.3_z_0.1.toml"
#param_file=$WORK_MLS"/codes/MLS-CNN/inputs/input_leonardo_sobol_lcdm_nc_ps_n8_kf0.35_xlum_0.3_z_0.1.toml"

#param_file=$WORK_MLS"/codes/MLS-CNN/inputs/input_leonardo_sobol_lcdm_ps_n64_xlum_0.3_z_0.1.toml"
#param_file=$WORK_MLS"/codes/MLS-CNN/inputs/input_leonardo_sobol_lcdm_ps_n64_kf0.8_xlum_0.3_z_0.1.toml"
#param_file=$WORK_MLS"/codes/MLS-CNN/inputs/input_leonardo_sobol_lcdm_ps_n64_kf0.6_xlum_0.3_z_0.1.toml"
#param_file=$WORK_MLS"/codes/MLS-CNN/inputs/input_leonardo_sobol_lcdm_ps_n64_kf0.5_xlum_0.3_z_0.1.toml"
#param_file=$WORK_MLS"/codes/MLS-CNN/inputs/input_leonardo_sobol_lcdm_ps_n64_kf0.4_xlum_0.3_z_0.1.toml"
#param_file=$WORK_MLS"/codes/MLS-CNN/inputs/input_leonardo_sobol_lcdm_ps_n64_kf0.37_xlum_0.3_z_0.1.toml"
#param_file=$WORK_MLS"/codes/MLS-CNN/inputs/input_leonardo_sobol_lcdm_ps_n64_kf0.36_xlum_0.3_z_0.1.toml"
#param_file=$WORK_MLS"/codes/MLS-CNN/inputs/input_leonardo_sobol_lcdm_ps_n64_kf0.35_xlum_0.3_z_0.1.toml"
#param_file=$WORK_MLS"/codes/MLS-CNN/inputs/input_leonardo_sobol_lcdm_ps_n64_kf0.34_xlum_0.3_z_0.1.toml"
#param_file=$WORK_MLS"/codes/MLS-CNN/inputs/input_leonardo_sobol_lcdm_ps_n64_kf0.33_xlum_0.3_z_0.1.toml"
#param_file=$WORK_MLS"/codes/MLS-CNN/inputs/input_leonardo_sobol_lcdm_ps_n64_kf0.31_xlum_0.3_z_0.1.toml"
#param_file=$WORK_MLS"/codes/MLS-CNN/inputs/input_leonardo_sobol_lcdm_ps_n64_kf0.3_xlum_0.3_z_0.1.toml"
#param_file=$WORK_MLS"/codes/MLS-CNN/inputs/input_leonardo_sobol_lcdm_ps_n64_kf0.27_xlum_0.3_z_0.1.toml"
#param_file=$WORK_MLS"/codes/MLS-CNN/inputs/input_leonardo_sobol_lcdm_ps_n64_kf0.2_xlum_0.3_z_0.1.toml"
#param_file=$WORK_MLS"/codes/MLS-CNN/inputs/input_leonardo_sobol_lcdm_ps_n64_kf0.1_xlum_0.3_z_0.1.toml"

########## Output dir. #########

## NC

#output_dir=$WORK_MLS"/results/trained_models_leonardo/sobol_lcdm/nc_n64_xlum_0.3_z_0.1"

## CNN

#output_dir=$WORK_MLS"/results/trained_models_leonardo/sobol_lcdm/cnn_o_n64_xlum_0.3_z_0.1"
#output_dir=$WORK_MLS"/results/trained_models_leonardo/sobol_lcdm/nc_ps_cnn_o_n64_xlum_0.3_z_0.1"

output_dir=$WORK_MLS"/results/trained_models_leonardo/sobol_lcdm/nc_ps_kf0.33_cnn_o_n64_xlum_0.3_z_0.1"
#output_dir=$WORK_MLS"/results/trained_models_leonardo/sobol_lcdm/nc_ps_kf0.35_cnn_o_n64_xlum_0.3_z_0.1"

#output_dir=$WORK_MLS"/results/trained_models_leonardo/sobol_lcdm/nc_cnn_o_n32_xlum_0.3_z_0.1"
#output_dir=$WORK_MLS"/results/trained_models_leonardo/sobol_lcdm/nc_cnn_o_n16_xlum_0.3_z_0.1"
#output_dir=$WORK_MLS"/results/trained_models_leonardo/sobol_lcdm/nc_cnn_o_n8_xlum_0.3_z_0.1"

## PS

#output_dir=$WORK_MLS"/results/trained_models_leonardo/sobol_lcdm/nc_ps_n64_xlum_0.3_z_0.1"
#output_dir=$WORK_MLS"/results/trained_models_leonardo/sobol_lcdm/nc_ps_n32_xlum_0.3_z_0.1"
#output_dir=$WORK_MLS"/results/trained_models_leonardo/sobol_lcdm/nc_ps_n16_xlum_0.3_z_0.1"

#output_dir=$WORK_MLS"/results/trained_models_leonardo/sobol_lcdm/nc_ps_n64_kf0.8_xlum_0.3_z_0.1"
#output_dir=$WORK_MLS"/results/trained_models_leonardo/sobol_lcdm/nc_ps_n64_kf0.6_xlum_0.3_z_0.1"
#output_dir=$WORK_MLS"/results/trained_models_leonardo/sobol_lcdm/nc_ps_n64_kf0.4_xlum_0.3_z_0.1"
#output_dir=$WORK_MLS"/results/trained_models_leonardo/sobol_lcdm/nc_ps_n64_kf0.37_xlum_0.3_z_0.1"
#output_dir=$WORK_MLS"/results/trained_models_leonardo/sobol_lcdm/nc_ps_n64_kf0.36_xlum_0.3_z_0.1"
#output_dir=$WORK_MLS"/results/trained_models_leonardo/sobol_lcdm/nc_ps_n64_kf0.34_xlum_0.3_z_0.1"

#output_dir=$WORK_MLS"/results/trained_models_leonardo/sobol_lcdm/nc_ps_n32_kf0.33_xlum_0.3_z_0.1"
#output_dir=$WORK_MLS"/results/trained_models_leonardo/sobol_lcdm/nc_ps_n16_kf0.33_xlum_0.3_z_0.1"

#output_dir=$WORK_MLS"/results/trained_models_leonardo/sobol_lcdm/nc_ps_n64_kf0.35_xlum_0.3_z_0.1"
#output_dir=$WORK_MLS"/results/trained_models_leonardo/sobol_lcdm/nc_ps_n32_kf0.35_xlum_0.3_z_0.1"
#output_dir=$WORK_MLS"/results/trained_models_leonardo/sobol_lcdm/nc_ps_n16_kf0.35_xlum_0.3_z_0.1"
#output_dir=$WORK_MLS"/results/trained_models_leonardo/sobol_lcdm/nc_ps_n8_kf0.35_xlum_0.3_z_0.1"

#output_dir=$WORK_MLS"/results/trained_models_leonardo/sobol_lcdm/ps_n64_xlum_0.3_z_0.1"
#output_dir=$WORK_MLS"/results/trained_models_leonardo/sobol_lcdm/ps_n64_kf0.8_xlum_0.3_z_0.1"
#output_dir=$WORK_MLS"/results/trained_models_leonardo/sobol_lcdm/ps_n64_kf0.6_xlum_0.3_z_0.1"
#output_dir=$WORK_MLS"/results/trained_models_leonardo/sobol_lcdm/ps_n64_kf0.5_xlum_0.3_z_0.1"
#output_dir=$WORK_MLS"/results/trained_models_leonardo/sobol_lcdm/ps_n64_kf0.4_xlum_0.3_z_0.1"
#output_dir=$WORK_MLS"/results/trained_models_leonardo/sobol_lcdm/ps_n64_kf0.37_xlum_0.3_z_0.1"
#output_dir=$WORK_MLS"/results/trained_models_leonardo/sobol_lcdm/ps_n64_kf0.36_xlum_0.3_z_0.1"
#output_dir=$WORK_MLS"/results/trained_models_leonardo/sobol_lcdm/ps_n64_kf0.35_xlum_0.3_z_0.1"
#output_dir=$WORK_MLS"/results/trained_models_leonardo/sobol_lcdm/ps_n64_kf0.34_xlum_0.3_z_0.1"
#output_dir=$WORK_MLS"/results/trained_models_leonardo/sobol_lcdm/ps_n64_kf0.33_xlum_0.3_z_0.1"
#output_dir=$WORK_MLS"/results/trained_models_leonardo/sobol_lcdm/ps_n64_kf0.31_xlum_0.3_z_0.1"
#output_dir=$WORK_MLS"/results/trained_models_leonardo/sobol_lcdm/ps_n64_kf0.3_xlum_0.3_z_0.1"
#output_dir=$WORK_MLS"/results/trained_models_leonardo/sobol_lcdm/ps_n64_kf0.27_xlum_0.3_z_0.1"
#output_dir=$WORK_MLS"/results/trained_models_leonardo/sobol_lcdm/ps_n64_kf0.2_xlum_0.3_z_0.1"
#output_dir=$WORK_MLS"/results/trained_models_leonardo/sobol_lcdm/ps_n64_kf0.1_xlum_0.3_z_0.1"

# Python main script.
exe_python=$WORK_MLS"/codes/MLS-CNN/main.py"

mkdir -p $output_dir
mkdir -p $output_dir/logs

cd $WORK_MLS/codes/MLS-CNN

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

#uv run python $exe_python tune -n $n_threads -f $param_file -o $output_dir &> ${output_dir}/logs/tune_11.log

#uv run python $exe_python tune -n $n_threads -f $param_file -o $output_dir &> ${output_dir}/logs/tune_1.log &
#uv run python $exe_python tune -n $n_threads -f $param_file -o $output_dir &> ${output_dir}/logs/tune_2.log &
#uv run python $exe_python tune -n $n_threads -f $param_file -o $output_dir &> ${output_dir}/logs/tune_3.log &
#uv run python $exe_python tune -n $n_threads -f $param_file -o $output_dir &> ${output_dir}/logs/tune_4.log &

#wait

uv run python $exe_python train -n $n_threads -f $param_file -o $output_dir &> ${output_dir}/logs/train.log
