import numpy as np
import numpy.typing as npt
import torch
from rich import print
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import Lambda

from input_args import Inputs

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_params_cosmo_xlum(
    cosmo_params: npt.NDArray,
    cosmo_models: npt.NDArray,
    n_models_xlum: int,
    xlum_params_file: str,
) -> npt.NDArray:

    # Number of cosmo models.
    n_models_cosmo = cosmo_models.shape[0]
    # Total number of models.
    n_models_total = n_models_cosmo * n_models_xlum

    # Cosmo parameter space dimension.
    ndim_cosmo = cosmo_params.shape[1]

    # TODO read from input file.
    ndim_xlum = 4
    ndim_total = ndim_cosmo + ndim_xlum

    # Init. final param array.
    params = np.zeros((n_models_total, ndim_total))

    # Select wanted cosmo params.
    cosmo_params = cosmo_params[cosmo_models - 1]

    # Repeat cosmo. params.
    cosmo_params = np.repeat(cosmo_params, n_models_xlum, axis=0)

    # Put cosmo params into final array.
    params[:, :ndim_cosmo] = cosmo_params

    # Read xlum params file.
    xlum_params_dict = np.load(xlum_params_file)

    # Gather xlum params for each cosmo model.
    xlum_params = []
    for m in cosmo_models:
        xlum_params.append(xlum_params_dict[f"model{m:05}_{m:05}"][:n_models_xlum, :])
    xlum_params = np.vstack(xlum_params)

    # Put xlum params into final array.
    params[:, ndim_cosmo:] = xlum_params

    return params


model


def read_cosmo_params(
    cosmo_params_file: str, params_names: list | None = None
) -> npt.NDArray:

    data = np.genfromtxt(cosmo_params_file)

    if params_names is None:
        return data

    params_idx = {
        "Omega_m": 0,
        "sigma8": 1,
        "h": 2,
        "n_s": 3,
        "Omega_b": 4,
        "w0": 5,
        "wa": 6,
    }

    cosmo_params = np.zeros((data.shape[0], len(params_names)))

    for i, name in enumerate(params_names):
        if name == "S8":
            cosmo_params[:, i] = data[:, params_idx["sigma8"]] * np.sqrt(
                data[:, params_idx["Omega_m"]] / 0.3
            )
        else:
            cosmo_params[:, i] = data[:, params_idx[name]]

    return cosmo_params


def get_cosmo_models_numbers(
    mobs_type: str, fraction: float = 1, seed: int | None = None
) -> npt.NDArray:

    # First and last model number.
    MODEL_MIN = 1
    if mobs_type == "mass":
        MODEL_MAX = 2048
    else:
        MODEL_MAX = 2048

    # Failed models.
    FAILED_MODELS = np.array([573, 644, 693, 813, 1348, 1700, 2012])

    # List of models.
    models = np.array([i for i in range(MODEL_MIN, MODEL_MAX + 1)])
    models = models[~np.isin(models, FAILED_MODELS)]

    # Select a fraction of models.
    if fraction < 1:
        rng = np.random.default_rng(seed)
        rng.shuffle(models)
        n = int(fraction * models.shape[0])
        models = models[:n]

    return models


class PinocchioNumberCountsDataset(Dataset):

    def __init__(
        self,
        cosmo_params_file: str,
        cosmo_params_names: list[str],
        data_dir: str,
        mobs_bins: npt.ArrayLike,
        redshift: npt.ArrayLike,
        mobs_type: str = "mass",
        xlum_sobol_n_models=0,
        xlum_params_file: str | None = None,
        cumulative: bool = False,
        transform=torch.from_numpy,
        target_transform=torch.from_numpy,
        fraction: float = 1,
        seed: int | None = None,
        lazy_loading: bool = False,
    ) -> None:

        if not cumulative:
            raise NotImplementedError(
                "differential mass bins for number counts not yet implemented."
            )

        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform

        self.mobs_min = np.ravel(np.array([mobs_bins]))
        self.redshift = np.ravel(np.array([redshift]))

        # Number of sobol sequence models for xlum.
        self.xlum_sobol_n_models = xlum_sobol_n_models

        # Use xlum sobol sequence or not.
        if self.xlum_sobol_n_models > 0 and mobs_type == "xlum":
            self.xlum_sobol = True
        else:
            self.xlum_sobol = False

        # Get models numbers.
        self.cosmo_models = get_cosmo_models_numbers(mobs_type, fraction, seed)

        # Cosmological parameters of each data file.
        cosmo_params = read_cosmo_params(cosmo_params_file, cosmo_params_names)

        # Labels are cosmo params + xlum params.
        if self.xlum_sobol:

            if xlum_params_file is None:
                raise ValueError("You must provide a value for `xlum_params_file`.")

            self.labels = get_params_cosmo_xlum(
                cosmo_params,
                self.cosmo_models,
                self.xlum_sobol_n_models,
                xlum_params_file,
            )
        # No xlum parameters in the labels.
        else:
            self.labels = cosmo_params[self.cosmo_models - 1, :]

        self.labels = self.target_transform(self.labels).type(torch.float32)

        if mobs_type == "mass":
            self.mobs_flag = "mass"
        elif mobs_type == "xlum":
            self.mobs_flag = "xlum"
        else:
            raise ValueError("Wrong value for mobs_type. Must be one of: mass, xlum.")

        self.lazy_loading = lazy_loading

        if not self.lazy_loading:
            self.data = []
            for i in range(len(self.labels)):
                self.labels[i].to(device, non_blocking=True)
                self.data.append(self.read_data(i).to(device, non_blocking=True))

    def __len__(self):

        return len(self.labels)

    def __getitem__(self, idx):

        if self.lazy_loading:
            data = self.read_data(idx)
        else:
            data = self.data[idx]

        label = self.labels[idx]

        return data, label

    def read_data(self, idx):

        if self.xlum_sobol:
            cm_idx = idx // self.xlum_sobol_n_models
            xm_idx = idx % self.xlum_sobol_n_models
        else:
            cm_idx = idx

        data = []

        for z in self.redshift:

            if self.xlum_sobol:
                data_path = f"{self.data_dir}/{self.mobs_flag}/z_{z:.4f}/pinocchio.model{self.cosmo_models[cm_idx]:05}_{self.cosmo_models[cm_idx]:05}_{xm_idx}.number_counts.dat"
            else:
                data_path = f"{self.data_dir}/{self.mobs_flag}/z_{z:.4f}/pinocchio.model{self.cosmo_models[cm_idx]:05}_{self.cosmo_models[cm_idx]:05}.number_counts.dat"

            data_tmp = np.genfromtxt(data_path)

            for m in self.mobs_min:
                data.append(np.log10(1 + data_tmp[data_tmp[:, 0] == m, 1]))

        data = np.ravel(data)

        data = self.transform(data).type(torch.float32)

        return data


class PinocchioPowerSpectrumDataset(Dataset):

    def __init__(
        self,
        cosmo_params_file: str,
        cosmo_params_names: list[str],
        data_dir: str,
        mobs_min: npt.ArrayLike,
        redshift: npt.ArrayLike,
        mobs_type: str = "mass",
        xlum_sobol_n_models=0,
        xlum_params_file: str | None = None,
        transform=torch.from_numpy,
        target_transform=torch.from_numpy,
        fraction: float = 1,
        seed: int | None = None,
        lazy_loading: bool = False,
    ) -> None:

        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform

        self.mobs_min = np.ravel(np.array([mobs_min]))
        self.redshift = np.ravel(np.array([redshift]))

        # Number of sobol sequence models for xlum.
        self.xlum_sobol_n_models = xlum_sobol_n_models

        # Use xlum sobol sequence or not.
        if self.xlum_sobol_n_models > 0 and mobs_type == "xlum":
            self.xlum_sobol = True
        else:
            self.xlum_sobol = False

        # Get cosmo models numbers.
        self.cosmo_models = get_cosmo_models_numbers(mobs_type, fraction, seed)

        # Cosmological parameters of each data file.
        cosmo_params = read_cosmo_params(cosmo_params_file, cosmo_params_names)

        # Labels are cosmo params + xlum params.
        if self.xlum_sobol:

            if xlum_params_file is None:
                raise ValueError("You must provide a value for `xlum_params_file`.")

            self.labels = get_params_cosmo_xlum(
                cosmo_params,
                self.cosmo_models,
                self.xlum_sobol_n_models,
                xlum_params_file,
            )
        # No xlum parameters in the labels.
        else:
            self.labels = cosmo_params[self.cosmo_models - 1, :]

        self.labels = self.target_transform(self.labels).type(torch.float32)

        if mobs_type == "mass":
            self.mobs_flag = "cut_mass"
        elif mobs_type == "xlum":
            self.mobs_flag = "cut_xlum"
        else:
            raise ValueError("Wrong value for mobs_type. Must be one of: mass, xlum.")

        self.lazy_loading = lazy_loading

        if not self.lazy_loading:
            self.data = []
            for i in range(len(self.labels)):
                self.labels[i].to(device, non_blocking=True)
                self.data.append(self.read_data(i).to(device, non_blocking=True))

    def __len__(self):

        return len(self.labels)

    def __getitem__(self, idx):

        if self.lazy_loading:
            data = self.read_data(idx)
        else:
            data = self.data[idx]

        label = self.labels[idx]

        return data, label

    def read_data(self, idx):

        if self.xlum_sobol:
            cm_idx = idx // self.xlum_sobol_n_models
            xm_idx = idx % self.xlum_sobol_n_models
        else:
            cm_idx = idx

        data = []

        for z in self.redshift:

            if self.xlum_sobol:
                data_path = f"{self.data_dir}/{self.mobs_flag}/z_{z:.4f}/pinocchio.model{self.cosmo_models[cm_idx]:05}_{self.cosmo_models[cm_idx]:05}_{xm_idx}.power_spectrum.npz"
            else:
                data_path = f"{self.data_dir}/{self.mobs_flag}/z_{z:.4f}/pinocchio.model{self.cosmo_models[cm_idx]:05}_{self.cosmo_models[cm_idx]:05}.power_spectrum.npz"

            with np.load(data_path) as data_read:

                for m in self.mobs_min:

                    data.append(np.log10(1 + data_read[f"{self.mobs_flag}_{m:.1e}"]))

        data = np.ravel(data)

        data = self.transform(data).type(torch.float32)

        return data


class PinocchioDensityFieldDataset(Dataset):

    def __init__(
        self,
        cosmo_params_file: str,
        cosmo_params_names: list[str],
        data_dir: str,
        mobs_min: npt.ArrayLike,
        redshift: npt.ArrayLike,
        mobs_type: str = "mass",
        xlum_sobol_n_models=0,
        xlum_params_file: str | None = None,
        overdensity=False,
        transform=torch.from_numpy,
        target_transform=torch.from_numpy,
        fraction: float = 1,
        seed: int | None = None,
        lazy_loading: bool = False,
    ) -> None:

        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform

        self.mobs_min = np.ravel(np.array([mobs_min]))
        self.redshift = np.ravel(np.array([redshift]))

        # If True, use overdensity of objects instead of density.
        self.overdensity = overdensity

        # Number of sobol sequence models for xlum.
        self.xlum_sobol_n_models = xlum_sobol_n_models

        # Use xlum sobol sequence or not.
        if self.xlum_sobol_n_models > 0 and mobs_type == "xlum":
            self.xlum_sobol = True
        else:
            self.xlum_sobol = False

        # Get cosmo models numbers.
        self.cosmo_models = get_cosmo_models_numbers(mobs_type, fraction, seed)

        # Cosmological parameters of each data file.
        cosmo_params = read_cosmo_params(cosmo_params_file, cosmo_params_names)

        # Labels are cosmo params + xlum params.
        if self.xlum_sobol:

            if xlum_params_file is None:
                raise ValueError("You must provide a value for `xlum_params_file`.")

            self.labels = get_params_cosmo_xlum(
                cosmo_params,
                self.cosmo_models,
                self.xlum_sobol_n_models,
                xlum_params_file,
            )
        # No xlum parameters in the labels.
        else:
            self.labels = cosmo_params[self.cosmo_models - 1, :]

        self.labels = self.target_transform(self.labels).type(torch.float32)

        if mobs_type == "mass":
            self.mobs_flag = "cut_mass"
        elif mobs_type == "xlum":
            self.mobs_flag = "cut_xlum"
        else:
            raise ValueError("Wrong value for mobs_type. Must be one of: mass, xlum.")

        self.lazy_loading = lazy_loading

        if not self.lazy_loading:
            self.data = []
            for i in range(len(self.labels)):
                self.labels[i].to(device, non_blocking=True)
                self.data.append(self.read_data(i).to(device, non_blocking=True))

    def __len__(self):

        return len(self.labels)

    def __getitem__(self, idx):

        if self.lazy_loading:
            data = self.read_data(idx)
        else:
            data = self.data[idx]

        label = self.labels[idx]

        return data, label

    def read_data(self, idx):

        if self.xlum_sobol:
            cm_idx = idx // self.xlum_sobol_n_models
            xm_idx = idx % self.xlum_sobol_n_models
        else:
            cm_idx = idx

        data = []

        for z in self.redshift:

            if self.xlum_sobol:
                data_path = f"{self.data_dir}/{self.mobs_flag}/z_{z:.4f}/pinocchio.model{self.cosmo_models[cm_idx]:05}_{self.cosmo_models[cm_idx]:05}_{xm_idx}.density_field.npz"
            else:
                data_path = f"{self.data_dir}/{self.mobs_flag}/z_{z:.4f}/pinocchio.model{self.cosmo_models[cm_idx]:05}_{self.cosmo_models[cm_idx]:05}.density_field.npz"

            with np.load(data_path) as data_read:

                for m in self.mobs_min:

                    data_tmp = data_read[f"{self.mobs_flag}_{m:.1e}"]

                    if self.overdensity:
                        data_tmp /= np.mean(data_tmp)
                        data_tmp -= 1

                    data_tmp = np.log10(1 + data_tmp)

                    ndim = data_tmp.ndim
                    if ndim == 2:
                        data_tmp = data_tmp.reshape(
                            (1, data_tmp.shape[0], data_tmp.shape[1])
                        )
                    elif ndim == 3:
                        data_tmp = data_tmp.reshape(
                            (
                                1,
                                data_tmp.shape[0],
                                data_tmp.shape[1],
                                data_tmp.shape[2],
                            )
                        )
                    else:
                        raise RuntimeError("Wrong dime for input images: ", ndim)

                    data.append(data_tmp)

        data = np.vstack(data)

        data = self.transform(data).type(torch.float32)

        return data


class MultiProbeDataset(Dataset):

    def __init__(self, dataset_list: list[PinocchioDensityFieldDataset]) -> None:

        super().__init__()

        self.dataset_list = dataset_list
        self.labels = self.dataset_list[0].labels

    def __len__(self):

        return len(self.labels)

    def __getitem__(self, idx):

        data_list = []

        for dataset in self.dataset_list:

            data_list.append(dataset[idx][0])

        label = self.labels[idx]

        return data_list, label


def get_dataset(probe: str, args: Inputs, verbose: bool = True, **kwargs) -> Dataset:

    if probe == "density_field":

        data_dir = f"{args.probes.data_dir_root}/{args.probes.density_field.data_dir}"

        if verbose:
            print(f"Reading data for probe={probe} from {data_dir}.")

        dataset = PinocchioDensityFieldDataset(
            args.cosmo_params_file,
            args.cosmo_params_names,
            data_dir,
            args.probes.density_field.mobs_min,
            args.probes.density_field.redshift,
            mobs_type=args.probes.density_field.mobs_type,
            xlum_sobol_n_models=args.xlum_sobol_n_models,
            xlum_params_file=args.xlum_params_file,
            overdensity=args.probes.density_field.overdensity,
            fraction=args.fraction_total,
            seed=args.split_seed,
            lazy_loading=args.lazy_loading,
            **kwargs,
        )

    elif probe == "power_spectrum":

        data_dir = f"{args.probes.data_dir_root}/{args.probes.power_spectrum.data_dir}"

        if verbose:
            print(f"Reading data for probe={probe} from {data_dir}.")

        dataset = PinocchioPowerSpectrumDataset(
            args.cosmo_params_file,
            args.cosmo_params_names,
            data_dir,
            args.probes.power_spectrum.mobs_min,
            args.probes.power_spectrum.redshift,
            mobs_type=args.probes.power_spectrum.mobs_type,
            xlum_sobol_n_models=args.xlum_sobol_n_models,
            xlum_params_file=args.xlum_params_file,
            fraction=args.fraction_total,
            seed=args.split_seed,
            lazy_loading=args.lazy_loading,
            **kwargs,
        )

    elif probe == "number_counts":

        data_dir = f"{args.probes.data_dir_root}/{args.probes.number_counts.data_dir}"

        if verbose:
            print(f"Reading data for probe={probe} from {data_dir}.")

        dataset = PinocchioNumberCountsDataset(
            args.cosmo_params_file,
            args.cosmo_params_names,
            data_dir,
            args.probes.number_counts.mobs_min,
            args.probes.number_counts.redshift,
            mobs_type=args.probes.number_counts.mobs_type,
            xlum_sobol_n_models=args.xlum_sobol_n_models,
            xlum_params_file=args.xlum_params_file,
            cumulative=args.probes.number_counts.cumulative,
            fraction=args.fraction_total,
            seed=args.split_seed,
            lazy_loading=args.lazy_loading,
            **kwargs,
        )
    else:
        raise ValueError(f"Unsupported probe: {probe}.")

    return dataset


def get_dataset_single_probe(
    probe: str, args: Inputs, verbose: bool = True
) -> tuple[Dataset, StandardScaler, StandardScaler | tuple[float, float]]:

    # Init. full dataset.
    dataset = get_dataset(probe, args, verbose=True)

    # Split into training, validation, and test datasets.
    fraction_train = 1 - args.fraction_validation - args.fraction_test
    generator = torch.Generator().manual_seed(args.split_seed)
    dataset_train, _, _ = random_split(
        dataset,
        [fraction_train, args.fraction_validation, args.fraction_test],
        generator=generator,
    )

    # Compute mean and standard deviation of all outputs from training set.
    dataloader_train = DataLoader(dataset_train, batch_size=len(dataset_train))

    # Compute mean and std over all input features.
    mean = next(iter(dataloader_train))[0].mean()
    std = next(iter(dataloader_train))[0].std()

    # # Save mean and std for later.
    # filename = f"dataset_standardization_{probe}_output.dat"
    # np.savetxt(
    #     f"{args.output_dir}/{filename}",
    #     [mean, std],
    #     header="Mean Std",
    # )

    # Transform to standardize data.
    transform_normalize = Lambda(lambda x: (torch.from_numpy(x) - mean) / std)

    mean = mean.detach().cpu().numpy()
    std = std.detach().cpu().numpy()

    if verbose:
        print(f"Standardizing dataset with: mean={mean}, std={std}.")

    scaler_data = (float(mean), float(std))

    # Fit StandardScaler to labels of training set.
    scaler_labels = StandardScaler()
    scaler_labels.fit(next(iter(dataloader_train))[1])

    if verbose:
        print(
            f"Standardizing dataset labels with: mean={scaler_labels.mean_}, std={scaler_labels.scale_}."
        )

    # Transform to standardize labels.
    target_transform_normalize = Lambda(
        lambda x: torch.from_numpy(scaler_labels.transform(x))
    )

    # Init. new dataset with standardization.
    dataset = get_dataset(
        probe,
        args,
        verbose=False,
        transform=transform_normalize,
        target_transform=target_transform_normalize,
    )

    return dataset, scaler_labels, scaler_data


def get_datasets_multiprobe(args: Inputs, verbose: bool = True) -> tuple[
    Dataset,
    StandardScaler,
    list[StandardScaler | tuple[float, float]],
]:

    dataset_list = []
    scaler_data_list = []

    for probe in args.probes.probe_list:

        dataset, scaler_labels, scaler_data = get_dataset_single_probe(
            probe, args, verbose
        )

        dataset_list.append(dataset)
        scaler_data_list.append(scaler_data)

    dataset = MultiProbeDataset(dataset_list)

    return dataset, scaler_labels, scaler_data_list
