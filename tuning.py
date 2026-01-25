from typing import Callable
import os

import numpy as np
import optuna
import torch
from optuna.trial import TrialState
from rich import print
from torch.amp import GradScaler
from torch.utils.data import DataLoader, Dataset, random_split

import models as models
from data import (
    AugmentedDensityFieldDataset,
    AugmentedMultiProbeDataset,
    get_dataset_single_probe,
    get_datasets_multiprobe,
)
from input_args import Inputs, suggest_args
from training import train_loop, validation_loop

device = "cuda" if torch.cuda.is_available() else "cpu"


def do_tune(args: Inputs) -> None:

    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(
            f"{args.output_dir}/optuna_journal.log"
        )
    )

    sampler = optuna.samplers.TPESampler(n_startup_trials=10, constant_liar=True)

    if args.tune.pruner == "hyperband":
        hyperband_max_resource = args.tune.n_epochs
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=args.tune.hyperband_min_resource,
            max_resource=hyperband_max_resource,
            reduction_factor=args.tune.hyperband_reduction_factor,
        )

        if args.verbose:
            print("Using hyperband pruner.")
            print(
                f"Number of HyperBand brackets (pruners): {int(np.emath.logn(args.tune.hyperband_reduction_factor, hyperband_max_resource / args.tune.hyperband_min_resource)) + 1}"
            )

    elif args.tune.pruner == "median":

        if args.verbose:
            print("Using median pruner.")

        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=args.tune.median_n_startup_trials,
            n_warmup_steps=args.tune.median_n_warmup_steps,
            n_min_trials=args.tune.median_n_min_trials,
        )
    else:
        raise ValueError(
            "Wrong pruner name. Supported pruners are: 'median' or 'hyperband'."
        )

    study = optuna.create_study(
        study_name=args.tune.study_name,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=args.tune.resume,
    )

    # Get training, validation, and test datasets.
    print(f"-------------------------------")
    # Get full dataset.
    if len(args.probes.probe_list) == 1:
        dataset, _, _ = get_dataset_single_probe(
            args.probes.probe_list[0], args, verbose=True
        )
    else:
        dataset, _, _ = get_datasets_multiprobe(args, verbose=True)
    # Split into training, validation, and test datasets.
    fraction_train = 1 - args.fraction_validation - args.fraction_test
    generator = torch.Generator().manual_seed(args.split_seed)
    dataset_train, dataset_val, _ = random_split(
        dataset,
        [fraction_train, args.fraction_validation, args.fraction_test],
        generator=generator,
    )
    if "density_field" in args.probes.probe_list:
        if args.probes.density_field.n_augment > 0:
            if len(args.probes.probe_list) == 1:
                dataset_train = AugmentedDensityFieldDataset(
                    dataset_train, args.probes.density_field.n_augment
                )
            else:
                dataset_train = AugmentedMultiProbeDataset(
                    dataset_train,
                    args.probes.density_field.n_augment,
                    args.probes.probe_list.index("density_field"),
                )

    print("Train dataset length: ", len(dataset_train))
    print("Val dataset length: ", len(dataset_val))
    print(f"-------------------------------\n")

    objective_func = get_objective_func(args, dataset_train, dataset_val)
    study.optimize(
        objective_func, n_trials=args.tune.trials, catch=(torch.OutOfMemoryError,)
    )

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Number: ", trial.number)
    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


def get_objective_func(
    args: Inputs, dataset_train: Dataset, dataset_val: Dataset
) -> Callable[[optuna.trial.Trial], float]:

    def objective_func(trial: optuna.trial.Trial) -> float:

        return objective(trial, args, dataset_train, dataset_val)

    return objective_func


def objective(
    trial: optuna.trial.Trial,
    args: Inputs,
    dataset_train: Dataset,
    dataset_val: Dataset,
) -> float:

    args = suggest_args(trial, args)

    # Set torch seed to try getting reproducible results.
    # Set it at the same stage as in the training module.
    torch.manual_seed(0)

    # Init. dataloaders.
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=2**args.train.batch_size_two_power,
        drop_last=True,
        shuffle=True,
        num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]) if args.lazy_loading else 0,
        pin_memory=args.lazy_loading,
    )
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=2**args.train.batch_size_two_power,
        drop_last=False,
        shuffle=False,
        num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]) if args.lazy_loading else 0,
        pin_memory=args.lazy_loading,
    )

    # Init. model.
    model = models.get_model(args, dataset_train)

    # Try to compile model if requested.
    if args.tune.compile_model:
        try:
            torch.set_float32_matmul_precision("high")
            model.compile(mode=args.tune.compile_mode)
        except:
            print(
                "Compilation of model failed. Continuing without compiling the model."
            )
            # To be safe reset the model.
            model = models.get_model(args, dataset_train)

    # Send model to device.
    model.to(device)

    print(f"Trial {trial.number}:", trial.params)

    # Optimizer.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.train.learning_rate,
        weight_decay=args.train.weight_decay,
        betas=(args.train.beta_1, args.train.beta_2),
        eps=args.train.eps,
    )

    # Scheduler.
    if args.train.scheduler == "cosine_warm_restarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            args.train.cosine_warm_restarts_t_0,
            T_mult=args.train.cosine_warm_restarts_t_mult,
        )
    elif args.train.scheduler == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            "min",
            factor=args.train.reduce_on_plateau_factor,
            patience=args.train.reduce_on_plateau_patience,
        )
    elif args.train.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.train.n_epochs
        )
    else:
        raise ValueError(
            "Unsupported scheduler. Options are: 'reduce_on_plateau', 'cosine', 'cosine_warm_restarts'"
        )

    # TODO: allow different settings for train and tune.
    # Set early stopping patience.
    # If early stopping is not activated, set it to a very large value.
    if args.train.early_stopping:
        patience_early_stopping = args.train.patience_early_stopping
    else:
        patience_early_stopping = 100 * args.train.n_epochs

    # Set gradient scaler (for AMP).
    grad_scaler = GradScaler(device=device)

    best_loss = 1e10
    best_epoch = -1

    # Loop over epochs.
    for epoch in range(args.tune.n_epochs):

        _ = train_loop(
            dataloader_train,
            model,
            optimizer,
            grad_scaler,
            args.pred_moments,
            send_to_device=args.lazy_loading,
            use_loss_skew=args.train.loss_skew,
            use_loss_kurt=args.train.loss_kurt,
            gauss_nllloss=args.train.gauss_nllloss,
            mse_loss=args.train.mse_loss,
        )
        val_loss = validation_loop(
            dataloader_val,
            model,
            args.pred_moments,
            send_to_device=args.lazy_loading,
            use_loss_skew=args.train.loss_skew,
            use_loss_kurt=args.train.loss_kurt,
            gauss_nllloss=args.train.gauss_nllloss,
            mse_loss=args.train.mse_loss,
        )

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        # Report current loss.
        trial.report(val_loss, epoch)

        if np.isnan(val_loss):
            print(f"Trial {trial.number}: trial aborted at epoch {epoch} (NaN value).")
            break

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch

        elif epoch - best_epoch > patience_early_stopping:
            print(
                f"Trial {trial.number}: early stopping triggered at epoch {epoch} with loss {val_loss}."
            )
            break

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            print(
                f"Trial {trial.number}: pruning triggered at epoch {epoch} with loss {val_loss}."
            )
            raise optuna.exceptions.TrialPruned()

    return val_loss
