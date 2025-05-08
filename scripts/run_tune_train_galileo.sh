#!/bin/bash
#SBATCH -J tu_cnn_x5_z_0
#SBATCH --partition=astro
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02-00:00:00
#SBATCH --output=/home/inigosaez/mls/logs/output/%x.%j.out
#SBATCH --error=/home/inigosaez/mls/logs/error/%x.%j.err

set -x

########## Param. file: PS #########

#param_file="/home/inigosaez/mls/nn_inference/inputs/input_galileo_ps.toml"

########## Param. file: NC #########

#param_file="/home/inigosaez/mls/nn_inference/inputs/input_galileo_nc.toml"

#param_file="/home/inigosaez/mls/nn_inference/inputs/input_galileo_nc_xlum_1.3e-1_1e0_5_z_0.toml"

#param_file="/home/inigosaez/mls/nn_inference/inputs/input_galileo_nc_xlum_1.3e-1_1e0_5_z_0_xlumBA4_sobol8.toml"

########## Param. file: PS x NC #########

#param_file="/home/inigosaez/mls/nn_inference/inputs/input_galileo_ps_nc_m10_z1.toml"
#param_file="/home/inigosaez/mls/nn_inference/inputs/input_galileo_ps_nc_m7_z1.toml"
#param_file="/home/inigosaez/mls/nn_inference/inputs/input_galileo_ps_nc_m5_z1.toml"

########## Param. file: CNN #########

#param_file="/home/inigosaez/mls/nn_inference/inputs/input_galileo_cnn_n32_m1_z1.toml"
#param_file="/home/inigosaez/mls/nn_inference/inputs/input_galileo_cnn_n32_m3_z1.toml"
#param_file="/home/inigosaez/mls/nn_inference/inputs/input_galileo_cnn_n32_m10_z1.toml"

#param_file="/home/inigosaez/mls/nn_inference/inputs/input_galileo_cnn2_n32_m10_z1.toml"
#param_file="/home/inigosaez/mls/nn_inference/inputs/input_galileo_cnn2_n32_m7_z1.toml"
#param_file="/home/inigosaez/mls/nn_inference/inputs/input_galileo_cnn2_n32_m5_z1.toml"

#param_file="/home/inigosaez/mls/nn_inference/inputs/input_galileo_cnn_n32_xlum_1e-2_1e0_10_z_0.toml"
param_file="/home/inigosaez/mls/nn_inference/inputs/input_galileo_cnn_n32_xlum_1.3e-1_1e0_5_z_0.toml"

#param_file="/home/inigosaez/mls/nn_inference/inputs/input_galileo_cnn_n32_xlum_1.3e-1_1e0_5_z_0_xlumBA4_sobol8.toml"

########## Output dir: PS #########

#output_dir="/home/inigosaez/mls/trained_models/power_spectrum_norsd_nmass1_nz1"
#output_dir="/home/inigosaez/mls/trained_models/power_spectrum_norsd_nxlum1_nz1"
#output_dir="/home/inigosaez/mls/trained_models/power_spectrum_norsd_nmass1_nz1_noskew"
#output_dir="/home/inigosaez/mls/trained_models/power_spectrum_norsd_nmass1_nz1_gauss"

########## Output dir: NC #########

#output_dir="/home/inigosaez/mls/trained_models/number_counts_nmass10_nz1"

#output_dir="/home/inigosaez/mls/trained_models/number_counts_xlum_1.3e-1_1e0_5_z_0"

#output_dir="/home/inigosaez/mls/trained_models/number_counts_xlum_1.3e-1_1e0_5_z_0_xlumBA4_sobol8"

########## Output dir: PS x NC #########

#output_dir="/home/inigosaez/mls/trained_models/power_spectrum_norsd_and_number_counts_nmass10_nz1"
#output_dir="/home/inigosaez/mls/trained_models/power_spectrum_norsd_and_number_counts_nmass7_nz1"
#output_dir="/home/inigosaez/mls/trained_models/power_spectrum_norsd_and_number_counts_nmass5_nz1"

########## Output dir: CNN #########

#output_dir="/home/inigosaez/mls/trained_models/density_field_n32_norsd_nmass1_nz1"
#output_dir="/home/inigosaez/mls/trained_models/density_field_n32_norsd_nmass3_nz1"
#output_dir="/home/inigosaez/mls/trained_models/density_field_n32_norsd_nmass10_nz1"

#output_dir="/home/inigosaez/mls/trained_models/density_field_n32_norsd_nmass10_nz1_cnn2"
#output_dir="/home/inigosaez/mls/trained_models/density_field_n32_norsd_nmass7_nz1_cnn2"
#output_dir="/home/inigosaez/mls/trained_models/density_field_n32_norsd_nmass5_nz1_cnn2"

#output_dir="/home/inigosaez/mls/trained_models/density_field_n16_norsd_nmass1_nz1_f0.0010"
#output_dir="/home/inigosaez/mls/trained_models/density_field_n16_norsd_nmass3_nz1_f0.0010"
#output_dir="/home/inigosaez/mls/trained_models/density_field_n16_norsd_nmass10_nz1_f0.0010"

#output_dir="/home/inigosaez/mls/trained_models/density_field_n32_xlum_1e-2_1e0_10_z_0"
output_dir="/home/inigosaez/mls/trained_models/density_field_n32_xlum_1.3e-1_1e0_5_z_0"

#output_dir="/home/inigosaez/mls/trained_models/density_field_n32_xlum_1.3e-1_1e0_5_z_0_xlumBA4_sobol8"

exe_python="/home/inigosaez/mls/nn_inference/main.py"

mkdir -p $output_dir
cd $output_dir
mkdir -p logs

parallel --line-buffer --tmpdir /home/inigosaez/parallel_tmpdir/ -j 32 "python $exe_python tune -f $param_file &> logs/tune_{}.log" ::: {1..32}

python $exe_python train -f $param_file &> logs/train.log
