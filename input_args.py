import argparse
from typing import Literal

import optuna
from pydantic import BaseModel, ConfigDict


def get_cli_args():

    parser = argparse.ArgumentParser(
        description="NN inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="mode", required=True)

    train = subparsers.add_parser(
        "train",
        help="Train NN.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_args_common(train)

    tune = subparsers.add_parser(
        "tune",
        help="Tune NN hyperparameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_args_common(tune)

    cli_args = parser.parse_args()

    return cli_args


def add_args_common(parser):

    parser.add_argument("-f", "--input-file", help="Input parameter file.", type=str)
    parser.add_argument(
        "-o",
        "--output-dir",
        help="The output directory. If provided, it overwrites any output directory specified in the input parameter file.",
        type=str,
    )
    parser.add_argument(
        "-n",
        "--n-threads",
        help="Number of threads used by pytorch.",
        type=int,
        default=1,
    )


class TrainInputs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    train_from_tune: bool
    tune_dir: str
    study_name: str

    loss_skew: bool
    loss_kurt: bool
    gauss_nllloss: bool

    n_epochs: int
    patience_early_stopping: int
    patience_early_stopping_factor: float

    reduce_on_plateau_patience: int
    reduce_on_plateau_factor: float

    batch_size: int

    learning_rate: float
    weight_decay: float

    dropout: float

    batch_norm: bool

    regressor_fc_layers: int
    regressor_fc_units_per_layer: int

    ps_fc_layers: int
    ps_fc_units_per_layer: int

    nc_fc_layers: int
    nc_fc_units_per_layer: int

    density_field_n_channels_base: int


class PowerSpectrumInputs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_dir: str
    mobs_min: list[float]
    mobs_type: Literal["mass", "xlum"]
    redshift: list[float]


class NumberCountsInputs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_dir: str
    mobs_min: list[float]
    mobs_type: Literal["mass", "xlum"]
    redshift: list[float]

    cumulative: bool


class DensityFieldInputs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_dir: str
    mobs_min: list[float]
    mobs_type: Literal["mass", "xlum"]
    redshift: list[float]

    overdensity: bool = False


class ProbesInputs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    probe_list: list[str]

    data_dir_root: str

    power_spectrum: PowerSpectrumInputs
    number_counts: NumberCountsInputs
    density_field: DensityFieldInputs


class HyperParamFloat(BaseModel):
    model_config = ConfigDict(extra="forbid")

    low: float
    high: float
    step: float | None = None
    log: bool = False


class HyperParamInt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    low: int
    high: int
    step: int = 1
    log: bool = False


class HyperParamCategorical(BaseModel):
    model_config = ConfigDict(extra="forbid")

    choices: list


class TuneInputs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    study_name: str

    resume: bool

    trials: int
    n_epochs: int

    pruner: Literal["median", "hyperband"]

    median_n_startup_trials: int
    median_n_warmup_steps: int
    median_n_min_trials: int

    hyperband_min_resource: int
    hyperband_reduction_factor: int

    learning_rate: HyperParamFloat
    weight_decay: HyperParamFloat
    batch_size: HyperParamCategorical

    regressor_fc_layers: HyperParamInt
    regressor_fc_units_per_layer: HyperParamInt

    ps_fc_layers: HyperParamInt
    ps_fc_units_per_layer: HyperParamInt

    nc_fc_layers: HyperParamInt
    nc_fc_units_per_layer: HyperParamInt

    density_field_n_channels_base: HyperParamInt

    dropout: HyperParamFloat

    reduce_on_plateau_factor: HyperParamFloat


class Inputs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_dir: str = "./"

    cosmo_params_file: str
    cosmo_params_names: list[str]

    xlum_params_file: str | None = None
    xlum_sobol_n_models: int

    fraction_total: float
    fraction_validation: float
    fraction_test: float
    split_seed: int

    verbose: bool

    lazy_loading: bool

    train: TrainInputs
    probes: ProbesInputs
    tune: TuneInputs


def suggest_args(
    trial: optuna.trial.Trial | optuna.trial.FrozenTrial, args: Inputs
) -> Inputs:

    # Deepcopy args before changing values with current trial.
    args = args.model_copy(deep=True)

    args.train.learning_rate = trial.suggest_float(
        "learning_rate",
        args.tune.learning_rate.low,
        args.tune.learning_rate.high,
        step=args.tune.learning_rate.step,
        log=args.tune.learning_rate.log,
    )
    args.train.weight_decay = trial.suggest_float(
        "weight_decay",
        args.tune.weight_decay.low,
        args.tune.weight_decay.high,
        step=args.tune.weight_decay.step,
        log=args.tune.weight_decay.log,
    )
    args.train.batch_size = trial.suggest_categorical(
        "batch_size", args.tune.batch_size.choices
    )
    args.train.dropout = trial.suggest_float(
        "dropout",
        args.tune.dropout.low,
        args.tune.dropout.high,
        step=args.tune.dropout.step,
        log=args.tune.dropout.log,
    )
    args.train.reduce_on_plateau_factor = trial.suggest_float(
        "reduce_on_plateau_factor",
        args.tune.reduce_on_plateau_factor.low,
        args.tune.reduce_on_plateau_factor.high,
        step=args.tune.reduce_on_plateau_factor.step,
        log=args.tune.reduce_on_plateau_factor.log,
    )

    for probe in args.probes.probe_list:

        if probe == "density_field":

            args.train.density_field_n_channels_base = trial.suggest_int(
                "density_field_n_channels_base",
                args.tune.density_field_n_channels_base.low,
                args.tune.density_field_n_channels_base.high,
                step=args.tune.density_field_n_channels_base.step,
                log=args.tune.density_field_n_channels_base.log,
            )

        elif probe == "power_spectrum":

            args.train.ps_fc_layers = trial.suggest_int(
                "ps_fc_layers",
                args.tune.ps_fc_layers.low,
                args.tune.ps_fc_layers.high,
                step=args.tune.ps_fc_layers.step,
                log=args.tune.ps_fc_layers.log,
            )
            args.train.ps_fc_units_per_layer = trial.suggest_int(
                "ps_fc_units_per_layer",
                args.tune.ps_fc_units_per_layer.low,
                args.tune.ps_fc_units_per_layer.high,
                step=args.tune.ps_fc_units_per_layer.step,
                log=args.tune.ps_fc_units_per_layer.log,
            )

        elif probe == "number_counts":

            args.train.nc_fc_layers = trial.suggest_int(
                "nc_fc_layers",
                args.tune.nc_fc_layers.low,
                args.tune.nc_fc_layers.high,
                step=args.tune.nc_fc_layers.step,
                log=args.tune.nc_fc_layers.log,
            )
            args.train.nc_fc_units_per_layer = trial.suggest_int(
                "nc_fc_units_per_layer",
                args.tune.nc_fc_units_per_layer.low,
                args.tune.nc_fc_units_per_layer.high,
                step=args.tune.nc_fc_units_per_layer.step,
                log=args.tune.nc_fc_units_per_layer.log,
            )

        else:
            raise ValueError(f"Unsupported probe: {probe}.")

    args.train.regressor_fc_layers = trial.suggest_int(
        "regressor_fc_layers",
        args.tune.regressor_fc_layers.low,
        args.tune.regressor_fc_layers.high,
        step=args.tune.regressor_fc_layers.step,
        log=args.tune.regressor_fc_layers.log,
    )
    args.train.regressor_fc_units_per_layer = trial.suggest_int(
        "regressor_fc_units_per_layer",
        args.tune.regressor_fc_units_per_layer.low,
        args.tune.regressor_fc_units_per_layer.high,
        step=args.tune.regressor_fc_units_per_layer.step,
        log=args.tune.regressor_fc_units_per_layer.log,
    )

    return args
