import os
import tomllib

# WARNING: Import sklearn BEFORE torch, otherwise there is a weird bug on galileo.
# DO NOT remove this import even if it is not used in this file.
import sklearn

import torch
from rich import print

from input_args import Inputs, get_cli_args
from training import do_train
from tuning import do_tune


def main(cli_args):

    print("\n----- CUDA and GPU availability -----\n")
    print("CUDA available: ", torch.cuda.is_available())
    print("Number of GPUs available: ", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_properties(i).name)
    print("\n-------------------------------------\n")

    with open(cli_args.input_file, mode="rb") as mytoml:
        input_args = tomllib.load(mytoml)

    # Use output directory from the CLI if provided.
    if cli_args.output_dir is not None:
        input_args["output_dir"] = cli_args.output_dir

    # Copy n_threads from CLI to input_args.
    input_args["n_threads"] = cli_args.n_threads

    input_args = Inputs(**input_args)

    if not torch.cuda.is_available():
        torch.set_num_threads(input_args.n_threads)

    print("\n-------------------------------\n")
    print(cli_args)
    print(input_args.model_dump())
    print("\n-------------------------------\n")

    # Make output dir.
    os.makedirs(input_args.output_dir, exist_ok=True)

    if cli_args.mode == "train":

        do_train(input_args)

    elif cli_args.mode == "tune":

        do_tune(input_args)


if __name__ == "__main__":

    cli_args = get_cli_args()
    main(cli_args)
