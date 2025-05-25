import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from sklearn.metrics import r2_score


def plot_training_history(
    train_history: npt.ArrayLike,
    val_history: npt.ArrayLike,
    output_dir: str,
    yscale: str,
) -> None:

    # Plot training history.
    plt.plot(train_history, label="training set")
    plt.plot(val_history, label="validation set")

    plt.xlabel("epochs")
    plt.ylabel("loss")

    plt.yscale(yscale)

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    figname1 = "history_loss.pdf"
    figname2 = "history_loss.png"
    plt.savefig(f"{output_dir}/{figname1}", bbox_inches="tight")
    plt.savefig(f"{output_dir}/{figname2}", bbox_inches="tight")

    plt.close()


def plot_pred_vs_target(
    target: npt.NDArray,
    pred: npt.NDArray,
    set_name: str,
    param_names: list[str],
    param_labels: list[str],
    output_dir: str,
    plot_std: bool,
) -> None:

    n_param = len(param_names)

    r2 = r2_score(target, pred[:, :n_param], multioutput="raw_values")

    for i, name in enumerate(param_names):

        fig, ax = plt.subplots()

        if plot_std:
            plt.errorbar(target[:, i], pred[:, i], pred[:, n_param + i], fmt=".")
        else:
            plt.plot(target[:, i], pred[:, i], ".")

        plt.plot(target[:, i], target[:, i], "-k")

        mean_abs_rel_err = (
            np.mean(np.abs((pred[:, i] - target[:, i]) / target[:, i])) * 100
        )
        mean_bias = np.mean(pred[:, i] - target[:, i])
        mean_bias_sign = np.sign(mean_bias)
        mean_bias_num1 = 10 ** (
            np.log10(np.abs(mean_bias)) - np.floor(np.log10(np.abs(mean_bias)))
        )
        mean_bias_num2 = np.floor(np.log10(np.abs(mean_bias)))
        if plot_std:
            chi2 = np.mean((pred[:, i] - target[:, i]) ** 2 / pred[:, n_param + i] ** 2)

        if plot_std:
            textstr = "\n".join(
                (
                    rf"$R^2={r2[i]:.2f}$",
                    rf"$\epsilon={mean_abs_rel_err:.2f}\%$",
                    rf"$\mathrm{{b}}={mean_bias_sign*mean_bias_num1:.2f}\cdot10^{{{mean_bias_num2:.0f}}}$",
                    rf"$\chi^2={chi2:.2f}$",
                )
            )
        else:
            textstr = "\n".join(
                (
                    rf"$R^2={r2[i]:.2f}$",
                    rf"$\epsilon={mean_abs_rel_err:.2f}\%$",
                    rf"$\mathrm{{b}}={mean_bias_sign*mean_bias_num1:.2f}\cdot10^{{{mean_bias_num2:.0f}}}$",
                )
            )
        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle="round", alpha=0.5)

        # place a text box in upper left in axes coords
        ax.text(
            0.05,
            0.95,
            textstr,
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment="top",
            bbox=props,
        )

        plt.xlabel("True")
        plt.ylabel("Prediction")

        plt.title(param_labels[i] + f" ({set_name} set)")

        plt.grid(True)
        plt.tight_layout()

        plt.savefig(
            f"{output_dir}/{name}_pred_vs_true_{set_name}_set.pdf", bbox_inches="tight"
        )
        plt.savefig(
            f"{output_dir}/{name}_pred_vs_true_{set_name}_set.png", bbox_inches="tight"
        )

        plt.close()
