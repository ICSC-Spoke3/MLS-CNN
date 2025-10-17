from abc import ABC, abstractmethod
import os

import numpy as np
import numpy.typing as npt
import torch
from abacusnbody.metadata import get_meta
from pandas import read_csv
from rich import print
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import Lambda

from input_args import _SIM_TYPES, Inputs

device = "cuda" if torch.cuda.is_available() else "cpu"

_BA_FIDUCIAL = np.array([-1.36, 1.88, -0.29, 0.328])


def get_params_cosmo_xlum(
    cosmo_params: npt.NDArray,
    cosmo_models: npt.NDArray,
    n_models_xlum: int,
    xlum_params_file: str,
    sim_type: _SIM_TYPES,
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

    # Repeat cosmo. params.
    cosmo_params = np.repeat(cosmo_params, n_models_xlum, axis=0)

    # Put cosmo params into final array.
    params[:, :ndim_cosmo] = cosmo_params

    # Gather xlum params for each cosmo model.
    xlum_params = []

    if xlum_params_file == "fiducial":

        for m in cosmo_models:
            xlum_params.append(_BA_FIDUCIAL.reshape(1, -1))

    else:

        # Read xlum params file.
        xlum_params_dict = np.load(xlum_params_file)

        for m in cosmo_models:
            if sim_type == "abacus":
                xlum_params.append(
                    xlum_params_dict[f"AbacusSummit_base_c{m:03}_ph000"][
                        :n_models_xlum, :
                    ]
                )
            else:
                xlum_params.append(
                    xlum_params_dict[f"model{m:05}_{m:05}"][:n_models_xlum, :]
                )

    xlum_params = np.vstack(xlum_params)

    # Put xlum params into final array.
    params[:, ndim_cosmo:] = xlum_params

    return params


def read_cosmo_params(
    cosmo_params_file: str,
    params_names: list[str],
    model_numbers: npt.NDArray,
    sim_type: _SIM_TYPES = "pinocchio",
) -> npt.NDArray:

    cosmo_params = np.zeros((model_numbers.shape[0], len(params_names)))

    if sim_type in ["pinocchio", "pinocchio_lcdm", "pinocchio_fiducial"]:

        params_idx_pinocchio = {
            "Omega_m": 0,
            "sigma8": 1,
            "h": 2,
            "n_s": 3,
            "Omega_b": 4,
            "w0": 5,
            "wa": 6,
        }

        data = np.genfromtxt(cosmo_params_file)
        if data.ndim == 1:
            data = data.reshape(1, -1)

        for i, name in enumerate(params_names):
            if name == "S8":
                cosmo_params[:, i] = data[
                    model_numbers - 1, params_idx_pinocchio["sigma8"]
                ] * np.sqrt(
                    data[model_numbers - 1, params_idx_pinocchio["Omega_m"]] / 0.3
                )
            else:
                cosmo_params[:, i] = data[model_numbers - 1, params_idx_pinocchio[name]]

    elif sim_type == "abacus":

        df = read_csv(cosmo_params_file)

        params_dict_cosmo_file = {
            "sigma8": "sigma8_m",
            "h": "h",
            "n_s": "n_s",
        }

        for j, m in enumerate(model_numbers):

            metadata = get_meta(f"AbacusSummit_base_c{m:03}_ph000")

            for i, name in enumerate(params_names):

                if name == "Omega_m":
                    cosmo_params[j, i] = metadata["Omega_M"]

                elif name in params_dict_cosmo_file:
                    cosmo_params[j, i] = df[df.root == f"abacus_cosm{m:03}"][
                        params_dict_cosmo_file[name]
                    ].array[0]

                elif name == "S8":
                    omega_m = metadata["Omega_M"]
                    sigma8 = df[df.root == f"abacus_cosm{m:03}"][
                        params_dict_cosmo_file["sigma8"]
                    ].array[0]
                    cosmo_params[j, i] = sigma8 * np.sqrt(omega_m / 0.3)

                elif name == "Omega_b":
                    omegabh2 = df[df.root == f"abacus_cosm{m:03}"]["omega_b"].array[0]
                    h = df[df.root == f"abacus_cosm{m:03}"]["h"]
                    cosmo_params[j, i] = omegabh2 / h / h

                else:
                    raise ValueError(f"Parameter {name} not yet supported for Abacus.")

    return cosmo_params


def get_cosmo_models_numbers(
    fraction: float = 1,
    seed: int | None = None,
    sim_type: _SIM_TYPES = "pinocchio",
) -> npt.NDArray:

    if sim_type == "pinocchio":
        # First and last model number.
        MODEL_MIN = 1
        MODEL_MAX = 2048

        # Failed models.
        FAILED_MODELS = np.array([573, 644, 693, 813, 1348, 1700, 2012])

        # List of models.
        models = np.array([i for i in range(MODEL_MIN, MODEL_MAX + 1)])
        models = models[~np.isin(models, FAILED_MODELS)]

    elif sim_type == "pinocchio_lcdm":
        # First and last model number.
        MODEL_MIN = 1
        MODEL_MAX = 4096

        # Failed models.
        FAILED_MODELS = np.array([])

        # List of models.
        models = np.array([i for i in range(MODEL_MIN, MODEL_MAX + 1)])
        models = models[~np.isin(models, FAILED_MODELS)]

    elif sim_type == "pinocchio_fiducial":

        # Number of sims.
        n_sims = 1000

        # List of models.
        models = np.array([1 for _ in range(n_sims)])

    elif sim_type == "abacus":
        # Only models that vary omega_cdm, omega_b, n_s, A_s.
        # Fiducial.
        models = [0]

        # Linear derivative grid.
        models += [i for i in range(100, 106)]
        models += [i for i in range(112, 114)]
        models += [i for i in range(116, 121)]
        models += [i for i in range(125, 127)]

        # Broad emulator grid.
        models += [i for i in range(130, 147)]

        models = np.array(models)

    # Select a fraction of models.
    if fraction < 1:
        rng = np.random.default_rng(seed)
        rng.shuffle(models)
        n = int(fraction * models.shape[0])
        models = models[:n]

    return models


class BaseDataset(ABC, Dataset):
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
        transform=torch.from_numpy,
        target_transform=torch.from_numpy,
        fraction: float = 1,
        seed: int | None = None,
        lazy_loading: bool = False,
        sim_type: _SIM_TYPES = "pinocchio",
    ) -> None:

        self.sim_type = sim_type

        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform

        self.mobs_bins = np.ravel(np.array([mobs_bins]))
        self.redshift = np.ravel(np.array([redshift]))

        self.xlum_sobol_n_models = xlum_sobol_n_models

        # Get models numbers.
        self.cosmo_models = get_cosmo_models_numbers(fraction, seed, sim_type=sim_type)

        # Cosmological parameters of each data file.
        cosmo_params = read_cosmo_params(
            cosmo_params_file, cosmo_params_names, self.cosmo_models, sim_type=sim_type
        )

        # Labels are cosmo params + xlum params.
        if mobs_type == "xlum" and self.xlum_sobol_n_models > 0:

            self.xlum_sobol = True

            if xlum_params_file is None:
                raise ValueError("You must provide a value for `xlum_params_file`.")

            self.labels = get_params_cosmo_xlum(
                cosmo_params,
                self.cosmo_models,
                xlum_sobol_n_models,
                xlum_params_file,
                self.sim_type,
            )

        # Labels are cosmo params + fiducial xlum params.
        elif mobs_type == "xlum" and self.xlum_sobol_n_models == 0:

            self.xlum_sobol = False

            self.labels = get_params_cosmo_xlum(
                cosmo_params, self.cosmo_models, 1, "fiducial", self.sim_type
            )

        # No xlum parameters in the labels.
        else:

            self.xlum_sobol = False

            self.labels = cosmo_params

        self.labels = self.target_transform(self.labels).type(torch.float32)

        self.mobs_type = mobs_type

        self.lazy_loading = lazy_loading

        if not self.lazy_loading:
            self.labels = self.labels.to(device, non_blocking=True)
            self.data = []
            for i in range(len(self.labels)):
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

    @abstractmethod
    def read_data(self, idx) -> torch.Tensor:
        pass


class NumberCountsDataset(BaseDataset):

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
        sim_type: _SIM_TYPES = "pinocchio",
    ) -> None:

        if not cumulative:
            raise NotImplementedError(
                "differential mass bins for number counts not yet implemented."
            )

        super(NumberCountsDataset, self).__init__(
            cosmo_params_file,
            cosmo_params_names,
            data_dir,
            mobs_bins,
            redshift,
            mobs_type=mobs_type,
            xlum_sobol_n_models=xlum_sobol_n_models,
            xlum_params_file=xlum_params_file,
            transform=transform,
            target_transform=target_transform,
            fraction=fraction,
            seed=seed,
            lazy_loading=lazy_loading,
            sim_type=sim_type,
        )

    def read_data(self, idx):

        if self.xlum_sobol:
            cm_idx = idx // self.xlum_sobol_n_models
            xm_idx = idx % self.xlum_sobol_n_models
            xm_suffix = f"_{xm_idx}"
        else:
            cm_idx = idx
            xm_suffix = ""

        if self.sim_type == "pinocchio" or self.sim_type == "pinocchio_lcdm":
            filename = f"pinocchio.model{self.cosmo_models[cm_idx]:05}_{self.cosmo_models[cm_idx]:05}{xm_suffix}.number_counts.dat"
        elif self.sim_type == "pinocchio_fiducial":
            filename = (
                f"pinocchio.model00001_{idx+10001:05}{xm_suffix}.number_counts.dat"
            )
        elif self.sim_type == "abacus":
            filename = f"abacus.model{self.cosmo_models[cm_idx]:05}_00000{xm_suffix}.number_counts.dat"

        data = []

        for z in self.redshift:

            data_path = f"{self.data_dir}/{self.mobs_type}/z_{z:.4f}/{filename}"

            data_tmp = np.genfromtxt(data_path)

            for m in self.mobs_bins:
                data.append(np.log10(1 + data_tmp[data_tmp[:, 0] == m, 1]))

        data = np.ravel(data)

        data = self.transform(data).type(torch.float32)

        return data


class PowerSpectrumDataset(BaseDataset):

    def __init__(
        self,
        cosmo_params_file: str,
        cosmo_params_names: list[str],
        data_dir: str,
        mobs_bins: npt.ArrayLike,
        redshift: npt.ArrayLike,
        kmax: float | None = None,
        mobs_type: str = "mass",
        xlum_sobol_n_models=0,
        xlum_params_file: str | None = None,
        transform=torch.from_numpy,
        target_transform=torch.from_numpy,
        fraction: float = 1,
        seed: int | None = None,
        lazy_loading: bool = False,
        sim_type: _SIM_TYPES = "pinocchio",
    ) -> None:

        self.kmax = kmax

        super(PowerSpectrumDataset, self).__init__(
            cosmo_params_file,
            cosmo_params_names,
            data_dir,
            mobs_bins,
            redshift,
            mobs_type=mobs_type,
            xlum_sobol_n_models=xlum_sobol_n_models,
            xlum_params_file=xlum_params_file,
            transform=transform,
            target_transform=target_transform,
            fraction=fraction,
            seed=seed,
            lazy_loading=lazy_loading,
            sim_type=sim_type,
        )

    def read_data(self, idx):

        if self.xlum_sobol:
            cm_idx = idx // self.xlum_sobol_n_models
            xm_idx = idx % self.xlum_sobol_n_models
            xm_suffix = f"_{xm_idx}"
        else:
            cm_idx = idx
            xm_suffix = ""

        if self.sim_type == "pinocchio" or self.sim_type == "pinocchio_lcdm":
            filename = f"pinocchio.model{self.cosmo_models[cm_idx]:05}_{self.cosmo_models[cm_idx]:05}{xm_suffix}.power_spectrum.npz"
        elif self.sim_type == "pinocchio_fiducial":
            filename = (
                f"pinocchio.model00001_{idx+10001:05}{xm_suffix}.power_spectrum.npz"
            )
        elif self.sim_type == "abacus":
            filename = f"abacus.model{self.cosmo_models[cm_idx]:05}_00000{xm_suffix}.power_spectrum.npz"

        data = []

        for z in self.redshift:

            data_path = f"{self.data_dir}/{self.mobs_type}/z_{z:.4f}/{filename}"

            with np.load(data_path) as data_read:

                for m in self.mobs_bins:

                    if self.kmax is not None:
                        kcut = data_read["k"] <= self.kmax
                        data_sel = data_read[f"{self.mobs_type}_{m:.2e}"][kcut]
                    else:
                        data_sel = data_read[f"{self.mobs_type}_{m:.2e}"]

                    data.append(np.log10(data_sel))

        data = np.ravel(data)

        data = self.transform(data).type(torch.float32)

        return data


class DensityFieldDataset(BaseDataset):

    def __init__(
        self,
        cosmo_params_file: str,
        cosmo_params_names: list[str],
        data_dir: str,
        mobs_bins: npt.ArrayLike,
        redshift: npt.ArrayLike,
        overdensity: bool = False,
        mobs_type: str = "mass",
        xlum_sobol_n_models=0,
        xlum_params_file: str | None = None,
        transform=torch.from_numpy,
        target_transform=torch.from_numpy,
        fraction: float = 1,
        seed: int | None = None,
        lazy_loading: bool = False,
        sim_type: _SIM_TYPES = "pinocchio",
    ) -> None:

        self.overdensity = overdensity

        super(DensityFieldDataset, self).__init__(
            cosmo_params_file,
            cosmo_params_names,
            data_dir,
            mobs_bins,
            redshift,
            mobs_type=mobs_type,
            xlum_sobol_n_models=xlum_sobol_n_models,
            xlum_params_file=xlum_params_file,
            transform=transform,
            target_transform=target_transform,
            fraction=fraction,
            seed=seed,
            lazy_loading=lazy_loading,
            sim_type=sim_type,
        )

    def read_data(self, idx, n_augment=0):

        if self.xlum_sobol:
            cm_idx = idx // self.xlum_sobol_n_models
            xm_idx = idx % self.xlum_sobol_n_models
            xm_suffix = f"_{xm_idx}"
        else:
            cm_idx = idx
            xm_suffix = ""

        if n_augment > 0:

            n_augment_str = f".augmented_{n_augment:03}"
        else:

            n_augment_str = ""

        if self.sim_type == "pinocchio" or self.sim_type == "pinocchio_lcdm":
            filename = f"pinocchio.model{self.cosmo_models[cm_idx]:05}_{self.cosmo_models[cm_idx]:05}{xm_suffix}.density_field{n_augment_str}.npz"
        elif self.sim_type == "pinocchio_fiducial":
            filename = f"pinocchio.model00001_{idx+10001:05}{xm_suffix}.density_field{n_augment_str}.npz"
        elif self.sim_type == "abacus":
            filename = f"abacus.model{self.cosmo_models[cm_idx]:05}_00000{xm_suffix}.density_field{n_augment_str}.npz"

        data = []

        for z in self.redshift:

            data_path = f"{self.data_dir}/{self.mobs_type}/z_{z:.4f}/{filename}"

            with np.load(data_path) as data_read:

                for m in self.mobs_bins:

                    data_tmp = data_read[f"{self.mobs_type}_{m:.2e}"]

                    if self.overdensity:

                        mean = np.mean(data_tmp)

                        if mean != 0:

                            data_tmp /= mean
                            data_tmp -= 1

                    else:

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
                        raise RuntimeError("Wrong dimension for input images: ", ndim)

                    data.append(data_tmp)

        data = np.vstack(data)

        data = self.transform(data).type(torch.float32)

        return data


class AugmentedDensityFieldDataset(Dataset):

    def __init__(
        self, dataset: DensityFieldDataset, n_augment_flip: int, do_flip: bool = True
    ) -> None:

        self.dataset = dataset

        self.n_augment_flip = n_augment_flip

        self.do_flip = do_flip

    def __len__(self):

        return len(self.dataset) * (self.n_augment_flip + 1)

    def __getitem__(self, idx):

        idx_base = int(idx % len(self.dataset))

        data, label = self.dataset[idx_base]

        if self.do_flip:

            idx_flip_list_3d = [[], [1], [2], [3], [1, 2], [1, 3], [2, 3], [1, 2, 3]]
            idx_flip = int(idx // len(self.dataset))

            data = torch.flip(data, idx_flip_list_3d[idx_flip])

        return data, label


class AugmentedMultiProbeDataset(Dataset):

    def __init__(self, multiprobe_dataset, n_augment_flip, n_flip_probe_idx) -> None:

        self.multiprobe_dataset = multiprobe_dataset
        self.n_augment_flip = n_augment_flip
        self.n_flip_probe_idx = n_flip_probe_idx

    def __len__(self):

        return len(self.multiprobe_dataset) * (self.n_augment_flip + 1)

    def __getitem__(self, idx):

        idx_flip_list_3d = [[], [1], [2], [3], [1, 2], [1, 3], [2, 3], [1, 2, 3]]

        idx_base = int(idx % len(self.multiprobe_dataset))
        idx_flip = int(idx // len(self.multiprobe_dataset))

        data_list, label = self.multiprobe_dataset[idx_base]

        data_list[self.n_flip_probe_idx] = torch.flip(
            data_list[self.n_flip_probe_idx], idx_flip_list_3d[idx_flip]
        )

        return data_list, label


# class AugmentedDensityFieldDataset(Dataset):
#
#   def __init__(self, dataset: DensityFieldDataset, n_augment: int) -> None:
#
#       self.dataset = dataset
#
#       self.n_augment = n_augment
#
#       if not self.dataset.dataset.lazy_loading:
#           self.data = []
#           for i in range(len(self.dataset) * (self.n_augment + 1)):
#               self.data.append(self.read_data(i).to(device, non_blocking=True))
#
#   def __len__(self):
#
#       return len(self.dataset) * (self.n_augment + 1)
#
#   def __getitem__(self, idx):
#
#       if self.dataset.dataset.lazy_loading:
#           data = self.read_data(idx)
#       else:
#           data = self.data[idx]
#
#       label = self.read_label(idx)
#
#       return data, label
#
#   def read_label(self, idx):
#
#       idx_base = int(idx % len(self.dataset))
#
#       _, label = self.dataset[idx_base]
#
#       return label
#
#   def read_data(self, idx):
#
#       idx_base = int(idx % len(self.dataset))
#       idx_augment = int(idx // len(self.dataset))
#
#       if idx_augment > 0:
#
#           data = self.dataset.dataset.read_data(idx_base, n_augment=idx_augment)
#
#       else:
#
#           data, _ = self.dataset[idx_base]
#
#       return data


class MultiProbeDataset(Dataset):

    def __init__(self, dataset_list: list[BaseDataset]) -> None:

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

        dataset = DensityFieldDataset(
            args.cosmo_params_file,
            args.cosmo_params_names,
            data_dir,
            args.probes.density_field.mobs_min,
            args.probes.density_field.redshift,
            overdensity=args.probes.density_field.overdensity,
            mobs_type=args.probes.density_field.mobs_type,
            xlum_sobol_n_models=args.xlum_sobol_n_models,
            xlum_params_file=args.xlum_params_file,
            fraction=args.fraction_total,
            seed=args.split_seed,
            lazy_loading=args.lazy_loading,
            sim_type=args.sim_type,
            **kwargs,
        )

    elif probe == "power_spectrum":

        data_dir = f"{args.probes.data_dir_root}/{args.probes.power_spectrum.data_dir}"

        if verbose:
            print(f"Reading data for probe={probe} from {data_dir}.")

        dataset = PowerSpectrumDataset(
            args.cosmo_params_file,
            args.cosmo_params_names,
            data_dir,
            args.probes.power_spectrum.mobs_min,
            args.probes.power_spectrum.redshift,
            kmax=args.probes.power_spectrum.kmax,
            mobs_type=args.probes.power_spectrum.mobs_type,
            xlum_sobol_n_models=args.xlum_sobol_n_models,
            xlum_params_file=args.xlum_params_file,
            fraction=args.fraction_total,
            seed=args.split_seed,
            lazy_loading=args.lazy_loading,
            sim_type=args.sim_type,
            **kwargs,
        )

    elif probe == "number_counts":

        data_dir = f"{args.probes.data_dir_root}/{args.probes.number_counts.data_dir}"

        if verbose:
            print(f"Reading data for probe={probe} from {data_dir}.")

        dataset = NumberCountsDataset(
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
            sim_type=args.sim_type,
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

    # Augment dataset for CNN.
    if probe == "density_field":
        dataset_train = AugmentedDensityFieldDataset(
            dataset_train, args.probes.density_field.n_augment_flip
        )

    # Compute mean and standard deviation of all outputs from training set.
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=2**args.train.batch_size_two_power,
        num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]) if args.lazy_loading else 0,
    )

    # Compute mean and std over all input features.
    mean = 0
    meansq = 0
    for data, _ in dataloader_train:
        mean += data.mean()
        meansq += (data**2).mean()
    std = torch.sqrt(meansq - mean**2)

    # Transform to standardize data.
    transform_normalize = Lambda(lambda x: (torch.from_numpy(x) - mean) / std)

    mean = mean.detach().cpu().numpy()
    std = std.detach().cpu().numpy()

    if verbose:
        print(f"Standardizing dataset with: mean={mean}, std={std}.")

    # Save dataset scaling for later.
    mean_array = np.array(mean)
    scale_array = np.array(std)
    scaling_data = np.vstack((mean_array, scale_array)).T
    filename = f"dataset_standardization_{probe}.dat"
    np.savetxt(
        f"{args.output_dir}/{filename}",
        scaling_data,
        header="Mean Std",
    )

    scaler_data = (float(mean), float(std))

    # Fit StandardScaler to labels of training set.
    scaler_labels = StandardScaler()
    scaler_labels.fit(next(iter(dataloader_train))[1].detach().cpu().numpy())

    if verbose:
        print(
            f"Standardizing dataset labels with: mean={scaler_labels.mean_}, std={scaler_labels.scale_}."
        )

    # Save labels scaling for later.
    mean_array = np.array(scaler_labels.mean_)
    scale_array = np.array(scaler_labels.scale_)
    scaling_data = np.vstack((mean_array, scale_array)).T
    filename = f"dataset_standardization_labels.dat"
    np.savetxt(
        f"{args.output_dir}/{filename}",
        scaling_data,
        header="Mean Std",
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
