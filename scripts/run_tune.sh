#!/usr/bin/env sh

n_tasks=10
param_file="inputs/input_ps.toml"
# param_file="inputs/input_df.toml"
# param_file="inputs/input_nc.toml"
# param_file="inputs/input_ps_nc.toml"

for i in {1..$n_tasks}
do
    python main.py tune -f $param_file &> logs/tune_ps_$i.log &
    # python main.py tune -f $param_file &> logs/tune_nc_$i.log &
    # python main.py tune -f $param_file &> logs/tune_ps_nc_$i.log &
done
