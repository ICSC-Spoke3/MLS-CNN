from typing import Callable

import numpy as np
import optuna
import torch
from optuna.trial import TrialState
from rich import print
from torch.utils.data import DataLoader, Dataset, random_split

import models as models
from data import get_dataset_single_probe, get_datasets_multiprobe
from input_args import Inputs, suggest_args
from training import train_loop, validation_loop

device = "cuda" if torch.cuda.is_available() else "cpu"


def do_tune(args: Inputs) -> None:

    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(
            f"{args.output_dir}/optuna_journal.log"
        )
    )

    sampler = optuna.samplers.TPESampler(n_startup_trials=10, constant_liar=False)

    if args.tune.pruner == "hyperband":
        hyperband_max_resource = args.tune.n_epochs
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=args.tune.hyperband_min_resource,
            max_resource=hyperband_max_resource,
            reduction_factor=args.tune.hyperband_reduction_factor,
        )

        if args.verbose:
            print(
                f"Number of HyperBand brackets (pruners): {int(np.emath.logn(args.tune.hyperband_reduction_factor, hyperband_max_resource / args.tune.hyperband_min_resource)) + 1}"
            )

    else:

        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=args.tune.median_n_startup_trials,
            n_warmup_steps=args.tune.median_n_warmup_steps,
            n_min_trials=args.tune.median_n_min_trials,
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
        dataset, scaler_labels, scaler_data = get_dataset_single_probe(
            args.probes.probe_list[0], args, verbose=True
        )
    else:
        dataset, scaler_labels, scaler_data = get_datasets_multiprobe(
            args, verbose=True
        )
    # Split into training, validation, and test datasets.
    fraction_train = 1 - args.fraction_validation - args.fraction_test
    generator = torch.Generator().manual_seed(args.split_seed)
    dataset_train, dataset_val, dataset_test = random_split(
        dataset,
        [fraction_train, args.fraction_validation, args.fraction_test],
        generator=generator,
    )
    print(f"-------------------------------\n")

    objective_func = get_objective_func(args, dataset_train, dataset_val)
    study.optimize(objective_func, n_trials=args.tune.trials)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

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

    # Init. dataloaders.
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=args.train.batch_size,
        shuffle=True,
        #num_workers=args.n_threads,
        num_workers=0,
        drop_last=True,
        #pin_memory=True,
    )
    dataloader_val = DataLoader(
        dataset_val, batch_size=len(dataset_val), shuffle=False, 
        #num_workers=args.n_threads,
        num_workers=0,
        #pin_memory=True
    )

    # Init. model.
    model = models.get_model(args, dataset_train)
    if args.tune.compile_model:
        torch.set_float32_matmul_precision("high")
        model.compile(mode=args.tune.compile_mode)
    model.to(device, non_blocking=True)

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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        "min",
        factor=args.train.reduce_on_plateau_factor,
        patience=args.train.reduce_on_plateau_patience,
    )

    # Set early stopping patience.
    if args.train.patience_early_stopping_factor != 0:
        patience_early_stopping = int(
            args.train.patience_early_stopping_factor
            * args.train.reduce_on_plateau_patience
        )
    else:
        patience_early_stopping = args.train.patience_early_stopping

    best_loss = 1e5
    best_epoch = -1

    # Loop over epochs.
    for epoch in range(args.tune.n_epochs):

        train_loss = train_loop(
            dataloader_train,
            model,
            optimizer,
            verbose=False,
            send_to_device=args.lazy_loading,
            loss_skew=args.train.loss_skew,
            loss_kurt=args.train.loss_kurt,
            gauss_nllloss=args.train.gauss_nllloss,
        )
        val_loss = validation_loop(
            dataloader_val,
            model,
            send_to_device=args.lazy_loading,
            loss_skew=args.train.loss_skew,
            loss_kurt=args.train.loss_kurt,
            gauss_nllloss=args.train.gauss_nllloss,
        )

        scheduler.step(val_loss)

        # Report current loss.
        trial.report(val_loss, epoch)

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
