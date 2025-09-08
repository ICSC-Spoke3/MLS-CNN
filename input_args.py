import argparse
from typing import Literal

import optuna
from pydantic import BaseModel, ConfigDict

_SIM_TYPES = Literal["pinocchio", "abacus", "pinocchio_lcdm", "pinocchio_fiducial"]


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

    compile_model: bool
    compile_mode: str

    train_from_tune: bool
    tune_dir: str
    study_name: str

    loss_skew: bool
    loss_kurt: bool
    gauss_nllloss: bool
    mse_loss: bool = False

    n_epochs: int

    early_stopping: bool
    patience_early_stopping: int

    scheduler: str

    reduce_on_plateau_patience: int
    reduce_on_plateau_factor: float

    cosine_warm_restarts_t_0: int = 20
    cosine_warm_restarts_t_mult: int = 1

    swa: bool
    swa_n_epochs: int
    swa_lr: float

    ema: bool
    ema_decay_rate: float = 0.999

    batch_size: int

    learning_rate: float
    weight_decay: float
    beta_1: float
    beta_2: float
    eps: float

    dropout: float

    regressor_fc_layers: int
    regressor_fc_units_per_layer: int
    regressor_activation: str = "relu"
    regressor_batch_norm: bool = True

    ps_fc_layers: int
    ps_fc_units_per_layer: int
    ps_activation: str = "relu"
    ps_batch_norm: bool = True

    nc_fc_layers: int
    nc_fc_units_per_layer: int
    nc_activation: str = "relu"
    nc_batch_norm: bool = True

    density_field_n_channels_first: int
    density_field_final_nside: int
    density_field_n_conv_per_block: int = 1
    density_field_activation: str = "relu"
    density_field_batch_norm: bool = True
    density_field_pooling: str = "average"


class PowerSpectrumInputs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_dir: str
    mobs_min: list[float]
    mobs_type: Literal["mass", "xlum"]
    redshift: list[float]

    kmax: float | None = None


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
    n_augment_flip: int = 0


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

    compile_model: bool
    compile_mode: str

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
    beta_1: HyperParamFloat
    beta_2: HyperParamFloat
    eps: HyperParamFloat

    batch_size: HyperParamCategorical

    regressor_fc_layers: HyperParamInt
    regressor_fc_units_per_layer: HyperParamInt
    regressor_activation: HyperParamCategorical
    regressor_batch_norm: HyperParamCategorical

    ps_fc_layers: HyperParamInt
    ps_fc_units_per_layer: HyperParamInt
    ps_activation: HyperParamCategorical
    ps_batch_norm: HyperParamCategorical

    nc_fc_layers: HyperParamInt
    nc_fc_units_per_layer: HyperParamInt
    nc_activation: HyperParamCategorical
    nc_batch_norm: HyperParamCategorical

    density_field_n_channels_first: HyperParamInt
    density_field_final_nside: HyperParamCategorical
    density_field_n_conv_per_block: HyperParamInt
    density_field_activation: HyperParamCategorical
    density_field_batch_norm: HyperParamCategorical
    density_field_pooling: HyperParamCategorical

    dropout: HyperParamFloat

    reduce_on_plateau_factor: HyperParamFloat

    cosine_warm_restarts_t_0: HyperParamInt
    cosine_warm_restarts_t_mult: HyperParamInt


class Inputs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_dir: str = "./"
    n_threads: int = 1

    sim_type: _SIM_TYPES

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

    pred_moments: bool = True

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
    args.train.beta_1 = trial.suggest_float(
        "beta_1",
        args.tune.beta_1.low,
        args.tune.beta_1.high,
        step=args.tune.beta_1.step,
        log=args.tune.beta_1.log,
    )
    args.train.beta_2 = trial.suggest_float(
        "beta_2",
        args.tune.beta_2.low,
        args.tune.beta_2.high,
        step=args.tune.beta_2.step,
        log=args.tune.beta_2.log,
    )
    args.train.eps = trial.suggest_float(
        "eps",
        args.tune.eps.low,
        args.tune.eps.high,
        step=args.tune.eps.step,
        log=args.tune.eps.log,
    )
    if len(args.tune.batch_size.choices) > 1:
        args.train.batch_size = trial.suggest_categorical(
            "batch_size", args.tune.batch_size.choices
        )
    else:
        args.train.batch_size = args.tune.batch_size.choices[0]
    args.train.dropout = trial.suggest_float(
        "dropout",
        args.tune.dropout.low,
        args.tune.dropout.high,
        step=args.tune.dropout.step,
        log=args.tune.dropout.log,
    )
    if args.train.scheduler == "reduce_on_plateau":
        args.train.reduce_on_plateau_factor = trial.suggest_float(
            "reduce_on_plateau_factor",
            args.tune.reduce_on_plateau_factor.low,
            args.tune.reduce_on_plateau_factor.high,
            step=args.tune.reduce_on_plateau_factor.step,
            log=args.tune.reduce_on_plateau_factor.log,
        )
    if args.train.scheduler == "cosine_warm_restarts":
        args.train.cosine_warm_restarts_t_0 = trial.suggest_int(
            "cosine_warm_restarts_t_0",
            args.tune.cosine_warm_restarts_t_0.low,
            args.tune.cosine_warm_restarts_t_0.high,
            step=args.tune.cosine_warm_restarts_t_0.step,
            log=args.tune.cosine_warm_restarts_t_0.log,
        )
        args.train.cosine_warm_restarts_t_mult = trial.suggest_int(
            "cosine_warm_restarts_t_mult",
            args.tune.cosine_warm_restarts_t_mult.low,
            args.tune.cosine_warm_restarts_t_mult.high,
            step=args.tune.cosine_warm_restarts_t_mult.step,
            log=args.tune.cosine_warm_restarts_t_mult.log,
        )

    for probe in args.probes.probe_list:

        if probe == "density_field":

            args.train.density_field_n_channels_first = trial.suggest_int(
                "density_field_n_channels_first",
                args.tune.density_field_n_channels_first.low,
                args.tune.density_field_n_channels_first.high,
                step=args.tune.density_field_n_channels_first.step,
                log=args.tune.density_field_n_channels_first.log,
            )
            if len(args.tune.density_field_final_nside.choices) > 1:
                args.train.density_field_final_nside = trial.suggest_categorical(
                    "density_field_final_nside",
                    args.tune.density_field_final_nside.choices,
                )
            else:
                args.train.density_field_final_nside = (
                    args.tune.density_field_final_nside.choices[0]
                )
            args.train.density_field_n_conv_per_block = trial.suggest_int(
                "density_field_n_conv_per_block",
                args.tune.density_field_n_conv_per_block.low,
                args.tune.density_field_n_conv_per_block.high,
                step=args.tune.density_field_n_conv_per_block.step,
                log=args.tune.density_field_n_conv_per_block.log,
            )
            if len(args.tune.density_field_activation.choices) > 1:
                args.train.density_field_activation = trial.suggest_categorical(
                    "density_field_activation",
                    args.tune.density_field_activation.choices,
                )
            else:
                args.train.density_field_activation = (
                    args.tune.density_field_activation.choices[0]
                )
            if len(args.tune.density_field_batch_norm.choices) > 1:
                args.train.density_field_batch_norm = trial.suggest_categorical(
                    "density_field_batch_norm",
                    args.tune.density_field_batch_norm.choices,
                )
            else:
                args.train.density_field_batch_norm = (
                    args.tune.density_field_batch_norm.choices[0]
                )
            if len(args.tune.density_field_pooling.choices) > 1:
                args.train.density_field_pooling = trial.suggest_categorical(
                    "density_field_pooling", args.tune.density_field_pooling.choices
                )
            else:
                args.train.density_field_pooling = (
                    args.tune.density_field_pooling.choices[0]
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
            if len(args.tune.ps_activation.choices) > 1:
                args.train.ps_activation = trial.suggest_categorical(
                    "ps_activation", args.tune.ps_activation.choices
                )
            else:
                args.train.ps_activation = args.tune.ps_activation.choices[0]
            if len(args.tune.ps_batch_norm.choices) > 1:
                args.train.ps_batch_norm = trial.suggest_categorical(
                    "ps_batch_norm", args.tune.ps_batch_norm.choices
                )
            else:
                args.train.ps_batch_norm = args.tune.ps_batch_norm.choices[0]

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
            if len(args.tune.nc_activation.choices) > 1:
                args.train.nc_activation = trial.suggest_categorical(
                    "nc_activation", args.tune.nc_activation.choices
                )
            else:
                args.train.nc_activation = args.tune.nc_activation.choices[0]
            if len(args.tune.nc_batch_norm.choices) > 1:
                args.train.nc_batch_norm = trial.suggest_categorical(
                    "nc_batch_norm", args.tune.nc_batch_norm.choices
                )
            else:
                args.train.nc_batch_norm = args.tune.nc_batch_norm.choices[0]

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
    if len(args.tune.regressor_activation.choices) > 1:
        args.train.regressor_activation = trial.suggest_categorical(
            "regressor_activation", args.tune.regressor_activation.choices
        )
    else:
        args.train.regressor_activation = args.tune.regressor_activation.choices[0]
    if len(args.tune.regressor_batch_norm.choices) > 1:
        args.train.regressor_batch_norm = trial.suggest_categorical(
            "regressor_batch_norm", args.tune.regressor_batch_norm.choices
        )
    else:
        args.train.regressor_batch_norm = args.tune.regressor_batch_norm.choices[0]

    return args
