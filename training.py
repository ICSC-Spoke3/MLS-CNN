import numpy as np
import numpy.typing as npt
import optuna
import torch
from rich import print
from sklearn.preprocessing import StandardScaler
from torch import nn

from torch.utils.data import DataLoader
from torchinfo import summary

import models as models
import plot as plot
from data import get_datasets_multiprobe
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
    dataset_train, dataset_val, dataset_test, scaler_labels, scaler_data = (
        get_datasets_multiprobe(args, verbose=True)
    )
    print(f"-------------------------------\n")

    # Init. dataloaders.
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=args.train.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    dataloader_val = DataLoader(
        dataset_val, batch_size=len(dataset_val), shuffle=False, num_workers=0
    )
    dataloader_test = DataLoader(
        dataset_test, batch_size=len(dataset_test), shuffle=False, num_workers=0
    )

    # Init. model.
    model = models.get_model(args, dataset_train)
    model.to(device)

    # Model summary.
    print(
        summary(
            model,
            input_data=[next(iter(dataloader_train))[0]],
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
        weight_decay=args.train.learning_rate,
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

    # Train model.
    train_history, val_history = training(
        dataloader_train,
        dataloader_val,
        model,
        optimizer,
        scheduler,
        args.train.n_epochs,
        patience_early_stopping,
        args.output_dir,
        loss_skew=args.train.loss_skew,
        loss_kurt=args.train.loss_kurt,
        gauss_nllloss=args.train.gauss_nllloss,
    )

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

    # Evaluate model.
    target_train, pred_train = eval_model(
        model, dataloader_train, scaler_labels, "training", args.output_dir
    )
    target_val, pred_val = eval_model(
        model, dataloader_val, scaler_labels, "validation", args.output_dir
    )
    target_test, pred_test = eval_model(
        model, dataloader_test, scaler_labels, "test", args.output_dir
    )

    # Predictions vs targets.
    ## Parameter names and labels.
    if args.xlum_sobol_n_models > 0:
        param_names = ["omega_m", "sigma8", "xlf_a", "xlf_b", "xlf_c", "xlf_sigma"]
        param_labels = [
            r"$\Omega_{\rm m}$",
            r"$\sigma_8$",
            r"$a_\mathrm{BA}$",
            r"$b_\mathrm{BA}$",
            r"$c_\mathrm{BA}$",
            r"$\sigma_\mathrm{BA}$",
        ]
    else:
        param_names = ["omega_m", "sigma8"]
        param_labels = [
            r"$\Omega_{\rm m}$",
            r"$\sigma_8$",
        ]

    plot.plot_pred_vs_target(
        target_train,
        pred_train,
        "training",
        param_names,
        param_labels,
        args.output_dir,
        plot_std=True,
    )
    plot.plot_pred_vs_target(
        target_val,
        pred_val,
        "validation",
        param_names,
        param_labels,
        args.output_dir,
        plot_std=True,
    )
    plot.plot_pred_vs_target(
        target_test,
        pred_test,
        "test",
        param_names,
        param_labels,
        args.output_dir,
        plot_std=True,
    )

    # Same for S8.
    target_train_S8 = target_train[:, 1] * np.sqrt(target_train[:, 0] / 0.3)
    pred_train_S8 = pred_train[:, 1] * np.sqrt(pred_train[:, 0] / 0.3)
    target_train_S8 = target_train_S8.reshape(-1, 1)
    pred_train_S8 = pred_train_S8.reshape(-1, 1)
    plot.plot_pred_vs_target(
        target_train_S8,
        pred_train_S8,
        "training",
        ["S8"],
        [r"$S_8$"],
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
        [r"$S_8$"],
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
        [r"$S_8$"],
        args.output_dir,
        plot_std=False,
    )


def checkpoint(model: nn.Module, filename) -> None:
    torch.save(model.state_dict(), filename)


def train_loop(
    dataloader: DataLoader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_skew=False,
    loss_kurt=False,
    gauss_nllloss=False,
    verbose: bool = True,
) -> float:

    assert dataloader.batch_size is not None

    num_batches = len(dataloader)
    size = num_batches * dataloader.batch_size

    current_loss = 0.0

    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X = [elt.to(device) for elt in X]
        y = y.to(device)

        optimizer.zero_grad()

        pred = model(X)
        n_pred = int(pred.shape[1] / 2)

        pred_means = pred[:, :n_pred]
        pred_vars = torch.exp(pred[:, n_pred : 2 * n_pred])
        # loss_primary = torch.mean(torch.sum((values - y) ** 2, dim=1), dim=0)
        # loss_secondary = torch.mean(
        #     torch.sum(((values - y) ** 2 - stds**2) ** 2, dim=1), dim=0
        # )
        # loss = torch.log(loss_primary) + torch.log(loss_secondary)

        if gauss_nllloss:

            loss_fn = nn.GaussianNLLLoss()
            loss = loss_fn(pred_means, y, pred_vars)

        else:

            loss_mean = torch.sum(
                torch.log(torch.mean((pred_means - y) ** 2, dim=0)), dim=0
            )
            loss_var = torch.sum(
                torch.log(torch.mean(((pred_means - y) ** 2 - pred_vars) ** 2, dim=0)),
                dim=0,
            )
            loss_skew = (
                torch.sum(
                    torch.log(torch.mean(((pred_means - y) ** 3) ** 2, dim=0)), dim=0
                )
                if loss_skew
                else 0
            )
            loss_kurt = (
                torch.sum(
                    torch.log(torch.mean(((pred_means - y) ** 4 - 3) ** 2, dim=0)),
                    dim=0,
                )
                if loss_kurt
                else 0
            )
            loss = loss_mean + loss_var + loss_skew + loss_kurt

        loss.backward()
        optimizer.step()

        current_loss += loss.item()

        if batch % 10 == 9:
            loss, current = loss.item(), (batch + 1) * len(X[0])
            if verbose:
                print(f"batch {batch+1} loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    # TODO maybe at some point report the average loss over N last batches instead of all batches.
    return current_loss / num_batches


def validation_loop(
    dataloader: DataLoader,
    model: nn.Module,
    loss_skew=False,
    loss_kurt=False,
    gauss_nllloss=False,
) -> float:

    num_batches = len(dataloader)

    val_loss = 0

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X = [elt.to(device) for elt in X]
            y = y.to(device)

            pred = model(X)
            n_pred = int(pred.shape[1] / 2)

            pred_means = pred[:, :n_pred]
            pred_vars = torch.exp(pred[:, n_pred : 2 * n_pred])
            # loss_primary = torch.mean(torch.sum((values - y) ** 2, dim=1), dim=0)
            # loss_secondary = torch.mean(
            #     torch.sum(((values - y) ** 2 - stds**2) ** 2, dim=1), dim=0
            # )
            # loss = torch.log(loss_primary) + torch.log(loss_secondary)
            if gauss_nllloss:

                loss_fn = nn.GaussianNLLLoss()
                loss = loss_fn(pred_means, y, pred_vars)

            else:

                loss_mean = torch.sum(
                    torch.log(torch.mean((pred_means - y) ** 2, dim=0)), dim=0
                )
                loss_var = torch.sum(
                    torch.log(
                        torch.mean(((pred_means - y) ** 2 - pred_vars) ** 2, dim=0)
                    ),
                    dim=0,
                )
                loss_skew = (
                    torch.sum(
                        torch.log(torch.mean(((pred_means - y) ** 3) ** 2, dim=0)),
                        dim=0,
                    )
                    if loss_skew
                    else 0
                )
                loss_kurt = (
                    torch.sum(
                        torch.log(torch.mean(((pred_means - y) ** 4 - 3) ** 2, dim=0)),
                        dim=0,
                    )
                    if loss_kurt
                    else 0
                )
                loss = loss_mean + loss_var + loss_skew + loss_kurt

            val_loss += loss.item()

    val_loss /= num_batches

    return val_loss


def training(
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    epochs: int,
    patience: int,
    output: str,
    loss_skew=False,
    loss_kurt=False,
    gauss_nllloss=False,
    verbose: bool = True,
) -> tuple[list[float], list[float]]:

    best_loss = 1e5
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
            loss_skew=loss_skew,
            loss_kurt=loss_kurt,
            gauss_nllloss=gauss_nllloss,
            verbose=verbose,
        )
        val_loss = validation_loop(
            val_dataloader,
            model,
            loss_skew=loss_skew,
            loss_kurt=loss_kurt,
            gauss_nllloss=gauss_nllloss,
        )

        train_history.append(train_loss)
        val_history.append(val_loss)

        scheduler.step(val_loss)

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

    if verbose:
        print("Done!")

    return train_history, val_history


def eval_model(
    model: nn.Module,
    dataloader: DataLoader,
    scaler_labels: StandardScaler,
    set_name: str,
    output_dir: str,
) -> tuple[npt.NDArray, npt.NDArray]:

    pred = []
    target = []

    model.eval()

    for X, y in dataloader:

        target.append(y.detach().numpy())

        X = [elt.to(device) for elt in X]
        pred.append(model(X).detach().numpy())

    n_pred = int(pred[0].shape[1] / 2)

    target = np.vstack(target)
    pred = np.vstack(pred)

    # Scale back.
    target = scaler_labels.inverse_transform(target)
    pred[:, :n_pred] = scaler_labels.inverse_transform(pred[:, :n_pred])
    pred[:, n_pred:] = scaler_labels.scale_ * np.sqrt(np.exp(pred[:, n_pred:]))

    # Save data.
    np.savetxt(f"{output_dir}/targets_{set_name}_set.dat", target)
    # Save model predictions.
    np.savetxt(f"{output_dir}/predictions_{set_name}_set.dat", pred)

    return target, pred
