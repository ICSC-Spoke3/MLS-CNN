import os

import numpy as np
import numpy.typing as npt
import optuna
import torch
from rich import print
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim.swa_utils import SWALR, AveragedModel
from torch.utils.data import DataLoader, random_split
from torchinfo import summary

import models as models
import plot as plot
from data import (
    AugmentedDensityFieldDataset,
    get_dataset_single_probe,
    get_datasets_multiprobe,
)
from input_args import Inputs, suggest_args

device = "cuda" if torch.cuda.is_available() else "cpu"


def do_train(args: Inputs) -> None:

    # Load best parameters from previous tuning run.
    if args.train.train_from_tune:

        print("Using parameters from previous tuning run:")
        if args.train.tune_dir == "output_dir":
            tune_dir = args.output_dir
        else:
            tune_dir = args.train.tune_dir
        storage = optuna.storages.JournalStorage(
            optuna.storages.journal.JournalFileBackend(f"{tune_dir}/optuna_journal.log")
        )
        study = optuna.load_study(study_name=args.train.study_name, storage=storage)
        best_trial = study.best_trial
        args = suggest_args(best_trial, args)

        print("\n-------------------------------\n")
        print(args.model_dump())
        print("\n-------------------------------\n")

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
    if "density_field" in args.probes.probe_list:
        if len(args.probes.probe_list) == 1:
            dataset_train = AugmentedDensityFieldDataset(
                dataset_train, args.probes.density_field.n_augment_flip
            )
        else:
            print("Ignoring data augmentation in multiprobe setup.\n")

    print("Train dataset length: ", len(dataset_train))
    print("Val dataset length: ", len(dataset_val))
    print("Test dataset length: ", len(dataset_test))
    print(f"-------------------------------\n")

    # Set torch seed to try getting reproducible results.
    # Set it at the same stage as in the tuning module.
    torch.manual_seed(0)

    # Init. dataloaders.
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=args.train.batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=int(os.environ['SLURM_CPUS_PER_TASK']) if args.lazy_loading else 0,
        pin_memory=args.lazy_loading,
    )
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=args.train.batch_size,
        drop_last=False,
        shuffle=False,
        num_workers=int(os.environ['SLURM_CPUS_PER_TASK']) if args.lazy_loading else 0,
        pin_memory=args.lazy_loading,
    )
    dataloader_test = DataLoader(
        dataset_test,
        batch_size=args.train.batch_size,
        drop_last=False,
        shuffle=False,
        num_workers=int(os.environ['SLURM_CPUS_PER_TASK']) if args.lazy_loading else 0,
        pin_memory=args.lazy_loading,
    )

    # Init. model.
    model = models.get_model(args, dataset_train)

    # Try to compile model if requested.
    if args.train.compile_model:
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

    # Model summary.
    data_sample = next(iter(dataloader_train))[0]
    if args.lazy_loading:
        if isinstance(data_sample, list):
            data_sample = [elt.to(device, non_blocking=True) for elt in data_sample]
        else:
            data_sample = data_sample.to(device, non_blocking=True)
    print(
        summary(
            model,
            input_data=[data_sample],
            batch_dim=args.train.batch_size,
            verbose=0,
            col_names=(
                "input_size",
                "output_size",
                "num_params",
                "params_percent",
                "kernel_size",
                # "groups",
                "mult_adds",
                # "trainable",
            ),
            row_settings=["var_names"],
        )
    )
    print("\n")

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

    # Set early stopping patience.
    # If early stopping is not activated, set it to a very large value.
    if args.train.early_stopping:
        patience_early_stopping = args.train.patience_early_stopping
    else:
        patience_early_stopping = 100 * args.train.n_epochs

    # Stochastic weight averaging (SWA).
    if args.train.swa:
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=args.train.swa_lr)

    # Exponential moving average (EMA).
    if args.train.ema:
        ema_model = AveragedModel(
            model,
            multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(
                args.train.ema_decay_rate
            ),
        )

    # Set gradient scaler (for AMP).
    grad_scaler = GradScaler(device=device)

    # Train model.
    train_history, val_history = training(
        dataloader_train,
        dataloader_val,
        model,
        optimizer,
        scheduler,
        grad_scaler,
        args.train.n_epochs,
        patience_early_stopping,
        args.pred_moments,
        args.output_dir,
        send_to_device=args.lazy_loading,
        loss_skew=args.train.loss_skew,
        loss_kurt=args.train.loss_kurt,
        gauss_nllloss=args.train.gauss_nllloss,
        mse_loss=args.train.mse_loss,
        swa=args.train.swa,
        swa_model=swa_model if args.train.swa else None,
        swa_scheduler=swa_scheduler if args.train.swa else None,
        swa_n_epochs=args.train.swa_n_epochs,
        ema=args.train.ema,
        ema_model=ema_model if args.train.ema else None,
    )

    if args.verbose:
        print("Making plots and saving results...")

    # Plot history.
    plot.plot_training_history(
        train_history,
        val_history,
        args.output_dir,
        yscale="linear",
    )

    # Save training history.
    filename = "history.dat"
    np.savetxt(
        f"{args.output_dir}/{filename}",
        np.vstack([train_history, val_history]).transpose(),
        header="Train Val",
    )

    # Load best model.
    model.load_state_dict(
        torch.load(
            f"{args.output_dir}/best_model.pth",
            weights_only=True,
            map_location=torch.device(device),
        )
    )

    # Load SWA model.
    if args.train.swa:

        swa_model.load_state_dict(
            torch.load(
                f"{args.output_dir}/swa_model.pth",
                weights_only=True,
                map_location=torch.device(device),
            )
        )

    # Load EMA model.
    if args.train.ema:

        ema_model.load_state_dict(
            torch.load(
                f"{args.output_dir}/ema_model.pth",
                weights_only=True,
                map_location=torch.device(device),
            )
        )

    # Evaluate model.
    target_train, pred_train = eval_model(
        model,
        dataloader_train,
        scaler_labels,
        args.pred_moments,
        "training",
        args.output_dir,
        send_to_device=args.lazy_loading,
    )
    target_val, pred_val = eval_model(
        model,
        dataloader_val,
        scaler_labels,
        args.pred_moments,
        "validation",
        args.output_dir,
        send_to_device=args.lazy_loading,
    )
    target_test, pred_test = eval_model(
        model,
        dataloader_test,
        scaler_labels,
        args.pred_moments,
        "test",
        args.output_dir,
        send_to_device=args.lazy_loading,
    )

    # Predictions vs targets.
    ## Parameter names and labels.
    param_labels_dict = {
        "Omega_m": r"$\Omega_{\rm m}$",
        "sigma8": r"$\sigma_8$",
        "S8": r"$S_8$",
        "h": r"$h$",
        "n_s": r"$n_s$",
        "Omega_b": r"$\Omega_{\rm b}$",
        "w0": r"$w_0$",
        "wa": r"$w_a$",
        "xlf_a": r"$a_\mathrm{BA}$",
        "xlf_b": r"$b_\mathrm{BA}$",
        "xlf_c": r"$c_\mathrm{BA}$",
        "xlf_sigma": r"$\sigma_\mathrm{BA}$",
    }
    param_names = args.cosmo_params_names
    if args.xlum_sobol_n_models > 0:
        param_names += ["xlf_a", "xlf_b", "xlf_c", "xlf_sigma"]

    plot.plot_pred_vs_target(
        target_train,
        pred_train,
        "training",
        param_names,
        param_labels_dict,
        args.output_dir,
        plot_std=args.pred_moments,
    )
    plot.plot_pred_vs_target(
        target_val,
        pred_val,
        "validation",
        param_names,
        param_labels_dict,
        args.output_dir,
        plot_std=args.pred_moments,
    )
    plot.plot_pred_vs_target(
        target_test,
        pred_test,
        "test",
        param_names,
        param_labels_dict,
        args.output_dir,
        plot_std=args.pred_moments,
    )

    # Same for S8 if not in base parameters.
    if ("S8" not in param_names) and ("Omega_m" in param_names) and ("sigma8" in param_names):
        target_train_S8 = target_train[:, 1] * np.sqrt(target_train[:, 0] / 0.3)
        pred_train_S8 = pred_train[:, 1] * np.sqrt(pred_train[:, 0] / 0.3)
        target_train_S8 = target_train_S8.reshape(-1, 1)
        pred_train_S8 = pred_train_S8.reshape(-1, 1)
        plot.plot_pred_vs_target(
            target_train_S8,
            pred_train_S8,
            "training",
            ["S8"],
            {"S8": r"$S_8$"},
            args.output_dir,
            plot_std=False,
        )

        target_val_S8 = target_val[:, 1] * np.sqrt(target_val[:, 0] / 0.3)
        pred_val_S8 = pred_val[:, 1] * np.sqrt(pred_val[:, 0] / 0.3)
        target_val_S8 = target_val_S8.reshape(-1, 1)
        pred_val_S8 = pred_val_S8.reshape(-1, 1)
        plot.plot_pred_vs_target(
            target_val_S8,
            pred_val_S8,
            "validation",
            ["S8"],
            {"S8": r"$S_8$"},
            args.output_dir,
            plot_std=False,
        )

        target_test_S8 = target_test[:, 1] * np.sqrt(target_test[:, 0] / 0.3)
        pred_test_S8 = pred_test[:, 1] * np.sqrt(pred_test[:, 0] / 0.3)
        target_test_S8 = target_test_S8.reshape(-1, 1)
        pred_test_S8 = pred_test_S8.reshape(-1, 1)
        plot.plot_pred_vs_target(
            target_test_S8,
            pred_test_S8,
            "test",
            ["S8"],
            {"S8": r"$S_8$"},
            args.output_dir,
            plot_std=False,
        )

    # Same for sigma8 if not in base parameters.
    if ("sigma8" not in param_names) and ("Omega_m" in param_names) and ("S8" in param_names):
        target_train_sigma8 = target_train[:, 1] / np.sqrt(target_train[:, 0] / 0.3)
        pred_train_sigma8 = pred_train[:, 1] / np.sqrt(pred_train[:, 0] / 0.3)
        target_train_sigma8 = target_train_sigma8.reshape(-1, 1)
        pred_train_sigma8 = pred_train_sigma8.reshape(-1, 1)
        plot.plot_pred_vs_target(
            target_train_sigma8,
            pred_train_sigma8,
            "training",
            ["sigma8"],
            {"sigma8": r"$\sigma_8$"},
            args.output_dir,
            plot_std=False,
        )

        target_val_sigma8 = target_val[:, 1] / np.sqrt(target_val[:, 0] / 0.3)
        pred_val_sigma8 = pred_val[:, 1] / np.sqrt(pred_val[:, 0] / 0.3)
        target_val_sigma8 = target_val_sigma8.reshape(-1, 1)
        pred_val_sigma8 = pred_val_sigma8.reshape(-1, 1)
        plot.plot_pred_vs_target(
            target_val_sigma8,
            pred_val_sigma8,
            "validation",
            ["sigma8"],
            {"sigma8": r"$\sigma_8$"},
            args.output_dir,
            plot_std=False,
        )

        target_test_sigma8 = target_test[:, 1] / np.sqrt(target_test[:, 0] / 0.3)
        pred_test_sigma8 = pred_test[:, 1] / np.sqrt(pred_test[:, 0] / 0.3)
        target_test_sigma8 = target_test_sigma8.reshape(-1, 1)
        pred_test_sigma8 = pred_test_sigma8.reshape(-1, 1)
        plot.plot_pred_vs_target(
            target_test_sigma8,
            pred_test_sigma8,
            "test",
            ["sigma8"],
            {"sigma8": r"$\sigma_8$"},
            args.output_dir,
            plot_std=False,
        )

    # SAME FOR SWA.
    if args.train.swa:
        # Evaluate model.
        target_train, pred_train = eval_model(
            swa_model,
            dataloader_train,
            scaler_labels,
            args.pred_moments,
            "training",
            args.output_dir,
            send_to_device=args.lazy_loading,
        )
        target_val, pred_val = eval_model(
            swa_model,
            dataloader_val,
            scaler_labels,
            args.pred_moments,
            "validation",
            args.output_dir,
            send_to_device=args.lazy_loading,
        )
        target_test, pred_test = eval_model(
            swa_model,
            dataloader_test,
            scaler_labels,
            args.pred_moments,
            "test",
            args.output_dir,
            send_to_device=args.lazy_loading,
        )

        plot.plot_pred_vs_target(
            target_train,
            pred_train,
            "training_swa",
            param_names,
            param_labels_dict,
            args.output_dir,
            plot_std=args.pred_moments,
        )
        plot.plot_pred_vs_target(
            target_val,
            pred_val,
            "validation_swa",
            param_names,
            param_labels_dict,
            args.output_dir,
            plot_std=args.pred_moments,
        )
        plot.plot_pred_vs_target(
            target_test,
            pred_test,
            "test_swa",
            param_names,
            param_labels_dict,
            args.output_dir,
            plot_std=args.pred_moments,
        )

        # Same for S8 if not in base parameters.
        if ("S8" not in param_names) and ("Omega_m" in param_names) and ("sigma8" in param_names):
            target_train_S8 = target_train[:, 1] * np.sqrt(target_train[:, 0] / 0.3)
            pred_train_S8 = pred_train[:, 1] * np.sqrt(pred_train[:, 0] / 0.3)
            target_train_S8 = target_train_S8.reshape(-1, 1)
            pred_train_S8 = pred_train_S8.reshape(-1, 1)
            plot.plot_pred_vs_target(
                target_train_S8,
                pred_train_S8,
                "training_swa",
                ["S8"],
                {"S8": r"$S_8$"},
                args.output_dir,
                plot_std=False,
            )

            target_val_S8 = target_val[:, 1] * np.sqrt(target_val[:, 0] / 0.3)
            pred_val_S8 = pred_val[:, 1] * np.sqrt(pred_val[:, 0] / 0.3)
            target_val_S8 = target_val_S8.reshape(-1, 1)
            pred_val_S8 = pred_val_S8.reshape(-1, 1)
            plot.plot_pred_vs_target(
                target_val_S8,
                pred_val_S8,
                "validation_swa",
                ["S8"],
                {"S8": r"$S_8$"},
                args.output_dir,
                plot_std=False,
            )

            target_test_S8 = target_test[:, 1] * np.sqrt(target_test[:, 0] / 0.3)
            pred_test_S8 = pred_test[:, 1] * np.sqrt(pred_test[:, 0] / 0.3)
            target_test_S8 = target_test_S8.reshape(-1, 1)
            pred_test_S8 = pred_test_S8.reshape(-1, 1)
            plot.plot_pred_vs_target(
                target_test_S8,
                pred_test_S8,
                "test_swa",
                ["S8"],
                {"S8": r"$S_8$"},
                args.output_dir,
                plot_std=False,
            )

        # Same for sigma8 if not in base parameters.
        if ("sigma8" not in param_names) and ("Omega_m" in param_names) and ("S8" in param_names):
            target_train_sigma8 = target_train[:, 1] / np.sqrt(target_train[:, 0] / 0.3)
            pred_train_sigma8 = pred_train[:, 1] / np.sqrt(pred_train[:, 0] / 0.3)
            target_train_sigma8 = target_train_sigma8.reshape(-1, 1)
            pred_train_sigma8 = pred_train_sigma8.reshape(-1, 1)
            plot.plot_pred_vs_target(
                target_train_sigma8,
                pred_train_sigma8,
                "training_swa",
                ["sigma8"],
                {"sigma8": r"$\sigma_8$"},
                args.output_dir,
                plot_std=False,
            )

            target_val_sigma8 = target_val[:, 1] / np.sqrt(target_val[:, 0] / 0.3)
            pred_val_sigma8 = pred_val[:, 1] / np.sqrt(pred_val[:, 0] / 0.3)
            target_val_sigma8 = target_val_sigma8.reshape(-1, 1)
            pred_val_sigma8 = pred_val_sigma8.reshape(-1, 1)
            plot.plot_pred_vs_target(
                target_val_sigma8,
                pred_val_sigma8,
                "validation_swa",
                ["sigma8"],
                {"sigma8": r"$\sigma_8$"},
                args.output_dir,
                plot_std=False,
            )

            target_test_sigma8 = target_test[:, 1] / np.sqrt(target_test[:, 0] / 0.3)
            pred_test_sigma8 = pred_test[:, 1] / np.sqrt(pred_test[:, 0] / 0.3)
            target_test_sigma8 = target_test_sigma8.reshape(-1, 1)
            pred_test_sigma8 = pred_test_sigma8.reshape(-1, 1)
            plot.plot_pred_vs_target(
                target_test_sigma8,
                pred_test_sigma8,
                "test_swa",
                ["sigma8"],
                {"sigma8": r"$\sigma_8$"},
                args.output_dir,
                plot_std=False,
            )

    # SAME FOR EMA.
    if args.train.ema:
        # Evaluate model.
        target_train, pred_train = eval_model(
            ema_model,
            dataloader_train,
            scaler_labels,
            args.pred_moments,
            "training",
            args.output_dir,
            send_to_device=args.lazy_loading,
        )
        target_val, pred_val = eval_model(
            ema_model,
            dataloader_val,
            scaler_labels,
            args.pred_moments,
            "validation",
            args.output_dir,
            send_to_device=args.lazy_loading,
        )
        target_test, pred_test = eval_model(
            ema_model,
            dataloader_test,
            scaler_labels,
            args.pred_moments,
            "test",
            args.output_dir,
            send_to_device=args.lazy_loading,
        )

        plot.plot_pred_vs_target(
            target_train,
            pred_train,
            "training_ema",
            param_names,
            param_labels_dict,
            args.output_dir,
            plot_std=args.pred_moments,
        )
        plot.plot_pred_vs_target(
            target_val,
            pred_val,
            "validation_ema",
            param_names,
            param_labels_dict,
            args.output_dir,
            plot_std=args.pred_moments,
        )
        plot.plot_pred_vs_target(
            target_test,
            pred_test,
            "test_ema",
            param_names,
            param_labels_dict,
            args.output_dir,
            plot_std=args.pred_moments,
        )

        # Same for S8 if not in base parameters.
        if ("S8" not in param_names) and ("Omega_m" in param_names) and ("sigma8" in param_names):
            target_train_S8 = target_train[:, 1] * np.sqrt(target_train[:, 0] / 0.3)
            pred_train_S8 = pred_train[:, 1] * np.sqrt(pred_train[:, 0] / 0.3)
            target_train_S8 = target_train_S8.reshape(-1, 1)
            pred_train_S8 = pred_train_S8.reshape(-1, 1)
            plot.plot_pred_vs_target(
                target_train_S8,
                pred_train_S8,
                "training_ema",
                ["S8"],
                {"S8": r"$S_8$"},
                args.output_dir,
                plot_std=False,
            )

            target_val_S8 = target_val[:, 1] * np.sqrt(target_val[:, 0] / 0.3)
            pred_val_S8 = pred_val[:, 1] * np.sqrt(pred_val[:, 0] / 0.3)
            target_val_S8 = target_val_S8.reshape(-1, 1)
            pred_val_S8 = pred_val_S8.reshape(-1, 1)
            plot.plot_pred_vs_target(
                target_val_S8,
                pred_val_S8,
                "validation_ema",
                ["S8"],
                {"S8": r"$S_8$"},
                args.output_dir,
                plot_std=False,
            )

            target_test_S8 = target_test[:, 1] * np.sqrt(target_test[:, 0] / 0.3)
            pred_test_S8 = pred_test[:, 1] * np.sqrt(pred_test[:, 0] / 0.3)
            target_test_S8 = target_test_S8.reshape(-1, 1)
            pred_test_S8 = pred_test_S8.reshape(-1, 1)
            plot.plot_pred_vs_target(
                target_test_S8,
                pred_test_S8,
                "test_ema",
                ["S8"],
                {"S8": r"$S_8$"},
                args.output_dir,
                plot_std=False,
            )

        # Same for sigma8 if not in base parameters.
        if ("sigma8" not in param_names) and ("Omega_m" in param_names) and ("S8" in param_names):
            target_train_sigma8 = target_train[:, 1] / np.sqrt(target_train[:, 0] / 0.3)
            pred_train_sigma8 = pred_train[:, 1] / np.sqrt(pred_train[:, 0] / 0.3)
            target_train_sigma8 = target_train_sigma8.reshape(-1, 1)
            pred_train_sigma8 = pred_train_sigma8.reshape(-1, 1)
            plot.plot_pred_vs_target(
                target_train_sigma8,
                pred_train_sigma8,
                "training_ema",
                ["sigma8"],
                {"sigma8": r"$\sigma_8$"},
                args.output_dir,
                plot_std=False,
            )

            target_val_sigma8 = target_val[:, 1] / np.sqrt(target_val[:, 0] / 0.3)
            pred_val_sigma8 = pred_val[:, 1] / np.sqrt(pred_val[:, 0] / 0.3)
            target_val_sigma8 = target_val_sigma8.reshape(-1, 1)
            pred_val_sigma8 = pred_val_sigma8.reshape(-1, 1)
            plot.plot_pred_vs_target(
                target_val_sigma8,
                pred_val_sigma8,
                "validation_ema",
                ["sigma8"],
                {"sigma8": r"$\sigma_8$"},
                args.output_dir,
                plot_std=False,
            )

            target_test_sigma8 = target_test[:, 1] / np.sqrt(target_test[:, 0] / 0.3)
            pred_test_sigma8 = pred_test[:, 1] / np.sqrt(pred_test[:, 0] / 0.3)
            target_test_sigma8 = target_test_sigma8.reshape(-1, 1)
            pred_test_sigma8 = pred_test_sigma8.reshape(-1, 1)
            plot.plot_pred_vs_target(
                target_test_sigma8,
                pred_test_sigma8,
                "test_ema",
                ["sigma8"],
                {"sigma8": r"$\sigma_8$"},
                args.output_dir,
                plot_std=False,
            )

    if args.verbose:
        print("...done!")


def checkpoint(model: nn.Module, filename) -> None:
    torch.save(model.state_dict(), filename)


# TODO: implement gradient accumulation.
# TODO: implement gradient norm clipping.
# Be careful with AMP compatibility.
def train_loop(
    dataloader: DataLoader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    grad_scaler: torch.amp.GradScaler,
    pred_moments: bool,
    send_to_device: bool = True,
    use_loss_skew=False,
    use_loss_kurt=False,
    gauss_nllloss=False,
    mse_loss=False,
) -> float:

    assert dataloader.batch_size is not None

    num_batches = len(dataloader)

    current_loss = 0.0

    model.train()

    for _, (X, y) in enumerate(dataloader):

        if send_to_device:
            if isinstance(X, list):
                X = [elt.to(device, non_blocking=True) for elt in X]
            else:
                X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device):

            pred = model(X)

            if pred_moments:

                n_pred = int(pred.shape[1] / 2)
                pred_means = pred[:, :n_pred]
                pred_vars = torch.exp(pred[:, n_pred : 2 * n_pred])

            else:

                pred_means = pred

            del pred

            if gauss_nllloss:

                if not pred_moments:
                    raise ValueError(
                        "GaussianNLLLoss is incompatible with pred_moments=False."
                    )

                loss_fn = nn.GaussianNLLLoss()
                loss = loss_fn(pred_means, y, pred_vars)

                del pred_means
                del pred_vars

            elif mse_loss:

                loss_fn = nn.MSELoss()
                loss = loss_fn(pred_means, y)

                del pred_means

            else:

                loss_mean = torch.sum(
                    torch.log(torch.mean((pred_means - y) ** 2, dim=0)), dim=0
                )

                if pred_moments:

                    loss_var = torch.sum(
                        torch.log(
                            torch.mean(((pred_means - y) ** 2 - pred_vars) ** 2, dim=0)
                        ),
                        dim=0,
                    )

                    del pred_vars

                    loss_skew = (
                        torch.sum(
                            torch.log(torch.mean(((pred_means - y) ** 3) ** 2, dim=0)),
                            dim=0,
                        )
                        if use_loss_skew
                        else 0
                    )
                    loss_kurt = (
                        torch.sum(
                            torch.log(
                                torch.mean(((pred_means - y) ** 4 - 3) ** 2, dim=0)
                            ),
                            dim=0,
                        )
                        if use_loss_kurt
                        else 0
                    )

                else:

                    loss_var = 0
                    loss_skew = 0
                    loss_kurt = 0

                del pred_means

                loss = loss_mean + loss_var + loss_skew + loss_kurt

                del loss_mean
                del loss_var
                del loss_skew
                del loss_kurt

        grad_scaler.scale(loss).backward()
        grad_scaler.step(optimizer)

        grad_scaler.update()

        current_loss += loss.item()

    return current_loss / num_batches


def validation_loop(
    dataloader: DataLoader,
    model: nn.Module,
    pred_moments: bool,
    send_to_device: bool = True,
    use_loss_skew=False,
    use_loss_kurt=False,
    gauss_nllloss=False,
    mse_loss=False,
) -> float:

    num_batches = len(dataloader)

    val_loss = 0

    model.eval()

    with torch.no_grad():

        for X, y in dataloader:

            if send_to_device:
                if isinstance(X, list):
                    X = [elt.to(device, non_blocking=True) for elt in X]
                else:
                    X = X.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

            pred = model(X)

            if pred_moments:

                n_pred = int(pred.shape[1] / 2)
                pred_means = pred[:, :n_pred]
                pred_vars = torch.exp(pred[:, n_pred : 2 * n_pred])

            else:

                pred_means = pred

            del pred

            if gauss_nllloss:

                if not pred_moments:
                    raise ValueError(
                        "GaussianNLLLoss is incompatible with pred_moments=False."
                    )

                loss_fn = nn.GaussianNLLLoss()
                loss = loss_fn(pred_means, y, pred_vars)

                del pred_means
                del pred_vars

            elif mse_loss:

                loss_fn = nn.MSELoss()
                loss = loss_fn(pred_means, y)

                del pred_means

            else:

                loss_mean = torch.sum(
                    torch.log(torch.mean((pred_means - y) ** 2, dim=0)), dim=0
                )

                if pred_moments:

                    loss_var = torch.sum(
                        torch.log(
                            torch.mean(((pred_means - y) ** 2 - pred_vars) ** 2, dim=0)
                        ),
                        dim=0,
                    )

                    del pred_vars

                    loss_skew = (
                        torch.sum(
                            torch.log(torch.mean(((pred_means - y) ** 3) ** 2, dim=0)),
                            dim=0,
                        )
                        if use_loss_skew
                        else 0
                    )
                    loss_kurt = (
                        torch.sum(
                            torch.log(
                                torch.mean(((pred_means - y) ** 4 - 3) ** 2, dim=0)
                            ),
                            dim=0,
                        )
                        if use_loss_kurt
                        else 0
                    )

                else:

                    loss_var = 0
                    loss_skew = 0
                    loss_kurt = 0

                del pred_means

                loss = loss_mean + loss_var + loss_skew + loss_kurt

                del loss_mean
                del loss_var
                del loss_skew
                del loss_kurt

            val_loss += loss.item()

    val_loss /= num_batches

    return val_loss


def training(
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    grad_scaler: torch.amp.GradScaler,
    epochs: int,
    patience: int,
    pred_moments: bool,
    output: str,
    send_to_device: bool = True,
    loss_skew=False,
    loss_kurt=False,
    gauss_nllloss=False,
    mse_loss=False,
    swa: bool = False,
    swa_model: None | torch.optim.swa_utils.AveragedModel = None,
    swa_scheduler: None | torch.optim.swa_utils.SWALR = None,
    swa_n_epochs: int = 10,
    ema: bool = False,
    ema_model: None | torch.optim.swa_utils.AveragedModel = None,
    verbose: bool = True,
) -> tuple[list[float], list[float]]:

    best_loss = 1e10
    best_epoch = -1
    val_history = []
    train_history = []

    for t in range(epochs):

        if verbose:
            print(f"Epoch {t+1}\n-------------------------------")

            print(f"LR: {scheduler.get_last_lr()}")

        train_loss = train_loop(
            train_dataloader,
            model,
            optimizer,
            grad_scaler,
            pred_moments,
            send_to_device=send_to_device,
            use_loss_skew=loss_skew,
            use_loss_kurt=loss_kurt,
            gauss_nllloss=gauss_nllloss,
            mse_loss=mse_loss,
        )
        val_loss = validation_loop(
            val_dataloader,
            model,
            pred_moments,
            send_to_device=send_to_device,
            use_loss_skew=loss_skew,
            use_loss_kurt=loss_kurt,
            gauss_nllloss=gauss_nllloss,
            mse_loss=mse_loss,
        )

        train_history.append(train_loss)
        val_history.append(val_loss)

        if ema:
            assert ema_model is not None
            ema_model.update_parameters(model)

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = t
            checkpoint(model, f"{output}/best_model.pth")
            if verbose:
                print("New best model")
        elif t - best_epoch > patience:
            if verbose:
                print(f"Current epoch training loss: {train_loss}")
                print(f"Current epoch validation loss: {val_loss}")
                print(f"Best validation loss: {best_loss} (epoch {best_epoch+1})")
                print(f"-------------------------------\n")
                print(f"Early stopped training at epoch {t+1}")
                print(f"Best validation loss: {best_loss} (epoch {best_epoch+1})")
                print(f"-------------------------------\n")
            break

        if verbose:
            print(f"Current epoch training loss: {train_loss}")
            print(f"Current epoch validation loss: {val_loss}")
            print(f"Best validation loss: {best_loss} (epoch {best_epoch+1})")
            print(f"-------------------------------\n")

    if ema:

        assert ema_model is not None

        # Update batch_norm statistics for the ema_model at the end.
        torch.optim.swa_utils.update_bn(train_dataloader, ema_model)

        # Use ema_model to make predictions on val data.
        val_loss_ema = validation_loop(
            val_dataloader,
            ema_model,
            pred_moments,
            send_to_device=send_to_device,
            use_loss_skew=loss_skew,
            use_loss_kurt=loss_kurt,
            gauss_nllloss=gauss_nllloss,
            mse_loss=mse_loss,
        )
        if verbose:
            print(f"Final validation loss for EMA model: {val_loss_ema}.")
            print(f"-------------------------------\n")

        checkpoint(ema_model, f"{output}/ema_model.pth")

    if swa:

        assert swa_model is not None
        assert swa_scheduler is not None

        print("---------- Stochastic Weight Averaging ----------\n")

        for t in range(swa_n_epochs):

            if verbose:
                print(f"Epoch {t+1}\n-------------------------------")

                print(f"LR: {swa_scheduler.get_last_lr()}")

            train_loss = train_loop(
                train_dataloader,
                model,
                optimizer,
                grad_scaler,
                pred_moments,
                send_to_device=send_to_device,
                use_loss_skew=loss_skew,
                use_loss_kurt=loss_kurt,
                gauss_nllloss=gauss_nllloss,
                mse_loss=mse_loss,
            )
            val_loss = validation_loop(
                val_dataloader,
                model,
                pred_moments,
                send_to_device=send_to_device,
                use_loss_skew=loss_skew,
                use_loss_kurt=loss_kurt,
                gauss_nllloss=gauss_nllloss,
                mse_loss=mse_loss,
            )

            if verbose:
                print(f"Current epoch training loss: {train_loss}")
                print(f"Current epoch validation loss: {val_loss}")
                print(f"-------------------------------\n")

            swa_model.update_parameters(model)
            swa_scheduler.step()

        # Update batch_norm statistics for the swa_model at the end.
        torch.optim.swa_utils.update_bn(train_dataloader, swa_model)
        # Use swa_model to make predictions on val data.
        val_loss_swa = validation_loop(
            val_dataloader,
            swa_model,
            pred_moments,
            send_to_device=send_to_device,
            use_loss_skew=loss_skew,
            use_loss_kurt=loss_kurt,
            gauss_nllloss=gauss_nllloss,
            mse_loss=mse_loss,
        )
        if verbose:
            print(f"Final validation loss for SWA model: {val_loss_swa}.")
            print(f"-------------------------------\n")

        checkpoint(swa_model, f"{output}/swa_model.pth")

    if verbose:
        print("Done!")

    return train_history, val_history


def eval_model(
    model: nn.Module,
    dataloader: DataLoader,
    scaler_labels: StandardScaler,
    pred_moments: bool,
    set_name: str,
    output_dir: str,
    send_to_device: bool = True,
) -> tuple[npt.NDArray, npt.NDArray]:

    pred = []
    target = []

    model.eval()

    with torch.no_grad():

        for X, y in dataloader:

            target.append(y.detach().cpu().numpy())

            if send_to_device:
                if isinstance(X, list):
                    X = [elt.to(device, non_blocking=True) for elt in X]
                else:
                    X = X.to(device, non_blocking=True)

            pred.append(model(X).detach().cpu().numpy())

    if pred_moments:
        n_pred = int(pred[0].shape[1] / 2)
    else:
        n_pred = pred[0].shape[1]

    target = np.vstack(target)
    pred = np.vstack(pred)

    # Scale back.
    target = scaler_labels.inverse_transform(target)
    pred[:, :n_pred] = scaler_labels.inverse_transform(pred[:, :n_pred])
    if pred_moments:
        pred[:, n_pred:] = scaler_labels.scale_ * np.sqrt(np.exp(pred[:, n_pred:]))

    # Save data.
    np.savetxt(f"{output_dir}/targets_{set_name}_set.dat", target)
    # Save model predictions.
    np.savetxt(f"{output_dir}/predictions_{set_name}_set.dat", pred)

    return target, pred
