import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset

from input_args import Inputs


class FC_regressor(nn.Module):

    def __init__(
        self,
        feature_extractor_list: nn.ModuleList,
        unit_list,
        output_size: int,
        batch_norm: bool,
        dropout: float,
    ):
        super().__init__()

        self.feature_extractor_list = feature_extractor_list

        self.batch_norm = batch_norm

        # Activation.
        activation = nn.ReLU

        # Feature size.
        n_features = 0
        for extractor in self.feature_extractor_list:
            *_, last_layer = extractor.modules()
            n_features += last_layer.out_features

        # Input module.
        self.input_module = nn.Sequential(
            nn.BatchNorm1d(n_features), activation(), nn.Dropout(p=dropout)
        )

        # FC module.
        self.fc_module = nn.Sequential()
        units_previous = n_features

        for i, unit in enumerate(unit_list):

            self.fc_module.add_module(f"fc_{i+1}", nn.Linear(units_previous, unit))
            self.fc_module.add_module(f"activation_{i+1}", activation())
            self.fc_module.add_module(f"dropout_{i+1}", nn.Dropout(p=dropout))

            units_previous = unit

        # Output layer.
        # self.fc_output_mean = nn.Linear(units_previous, output_size)
        # self.fc_output_std = nn.Linear(units_previous, output_size)
        self.fc_output = nn.Linear(units_previous, output_size * 2)

    def forward(self, x_list):

        feature_list = []
        for x, feature_extractor in zip(x_list, self.feature_extractor_list):
            feature_list.append(feature_extractor(x))

        x = torch.cat(feature_list, dim=-1)

        # Input module.
        x = self.input_module(x)

        # FC module.
        x = self.fc_module(x)

        # Output layer.
        # x = torch.cat(
        #     (self.fc_output_mean(x), F.softplus(self.fc_output_std(x))), dim=-1
        # )
        x = self.fc_output(x)

        return x


def get_CNN_extractor_2(
    dim, in_nside: int, in_channels: int, channels_base: int, batch_norm: bool
):

    # Activation function.
    activation = nn.ReLU

    # Convolution settings.
    # This window conserves the image size.
    conv_kernel_1 = 3
    conv_stride_1 = 1
    conv_padding_1 = 1

    # This window divides the image nside by two.
    conv_kernel_2 = 2
    conv_stride_2 = 2
    conv_padding_2 = 0

    padding_mode = "circular"

    num_conv_layers = int(np.log2(in_nside / 4))

    # Batch norm.
    if dim == 2:
        conv_layer = nn.Conv2d
        batch_norm_layer = nn.BatchNorm2d
        pool_layer = nn.MaxPool2d
    elif dim == 3:
        conv_layer = nn.Conv3d
        batch_norm_layer = nn.BatchNorm3d
        pool_layer = nn.MaxPool3d
    else:
        raise ValueError("Wrong dimension value: ", dim)

    module = nn.Sequential()

    n_channels_previous = in_channels

    for i in range(num_conv_layers):

        n_channels = 2 ** (i + 1) * channels_base

        module.add_module(
            f"conv_{i+1}",
            conv_layer(
                n_channels_previous,
                n_channels,
                conv_kernel_1,
                conv_stride_1,
                conv_padding_1,
                padding_mode=padding_mode,
            ),
        )
        if batch_norm:
            module.add_module(f"batch_norm_{i+1}_1", batch_norm_layer(n_channels))
        module.add_module(f"activation_{i+1}_1", activation())

        module.add_module(
            f"pool_{i+1}",
            pool_layer(
                conv_kernel_2,
                conv_stride_2,
                conv_padding_2,
            ),
        )

        n_channels_previous = n_channels

    # Last convolution.
    n_channels_final = 2 ** (num_conv_layers + 1) * channels_base
    module.add_module(
        "conv_final",
        conv_layer(
            n_channels_previous,
            n_channels_final,
            4,
            1,
            0,
            padding_mode=padding_mode,
        ),
    )
    if batch_norm:
        module.add_module(f"batch_norm_final", batch_norm_layer(n_channels_final))
    module.add_module(f"activation_final", activation())

    # Flatten.
    module.add_module("flatten", nn.Flatten(start_dim=1))

    # FC (output layer).
    module.add_module(
        "fc_final",
        nn.Linear(n_channels_final, n_channels_final),
    )

    return module


def get_CNN_extractor(
    dim, in_nside: int, in_channels: int, channels_base: int, batch_norm: bool
):

    # Activation function.
    activation = nn.ReLU

    # Convolution settings.
    # This window conserves the image size.
    conv_kernel_1 = 3
    conv_stride_1 = 1
    conv_padding_1 = 1

    # This window divides the image nside by two.
    conv_kernel_2 = 2
    conv_stride_2 = 2
    conv_padding_2 = 0

    padding_mode = "circular"

    num_conv_layers = int(np.log2(in_nside / 4))

    # Batch norm.
    if dim == 2:
        conv_layer = nn.Conv2d
        batch_norm_layer = nn.BatchNorm2d
    elif dim == 3:
        conv_layer = nn.Conv3d
        batch_norm_layer = nn.BatchNorm3d
    else:
        raise ValueError("Wrong dimension value: ", dim)

    module = nn.Sequential()

    n_channels_previous = in_channels

    for i in range(num_conv_layers):

        n_channels = 2 ** (i + 1) * channels_base

        module.add_module(
            f"conv_{i+1}_1",
            conv_layer(
                n_channels_previous,
                n_channels,
                conv_kernel_1,
                conv_stride_1,
                conv_padding_1,
                padding_mode=padding_mode,
            ),
        )
        if batch_norm:
            module.add_module(f"batch_norm_{i+1}_1", batch_norm_layer(n_channels))
        module.add_module(f"activation_{i+1}_1", activation())

        module.add_module(
            f"conv_{i+1}_2",
            conv_layer(
                n_channels,
                n_channels,
                conv_kernel_1,
                conv_stride_1,
                conv_padding_1,
                padding_mode=padding_mode,
            ),
        )
        if batch_norm:
            module.add_module(f"batch_norm_{i+1}_2", batch_norm_layer(n_channels))
        module.add_module(f"activation_{i+1}_2", activation())

        module.add_module(
            f"conv_{i+1}_final",
            conv_layer(
                n_channels,
                n_channels,
                conv_kernel_2,
                conv_stride_2,
                conv_padding_2,
                padding_mode=padding_mode,
            ),
        )
        if batch_norm:
            module.add_module(f"batch_norm_{i+1}_final", batch_norm_layer(n_channels))
        module.add_module(f"activation_{i+1}_final", activation())

        n_channels_previous = n_channels

    # Last convolution.
    n_channels_final = 2 ** (num_conv_layers + 1) * channels_base
    module.add_module(
        "conv_final",
        conv_layer(
            n_channels_previous,
            n_channels_final,
            4,
            1,
            0,
            padding_mode=padding_mode,
        ),
    )
    if batch_norm:
        module.add_module(f"batch_norm_final", batch_norm_layer(n_channels_final))
    module.add_module(f"activation_final", activation())

    # Flatten.
    module.add_module("flatten", nn.Flatten(start_dim=1))

    # FC (output layer).
    module.add_module(
        "fc_final",
        nn.Linear(n_channels_final, n_channels_final),
    )

    return module


def get_FC_extractor(input_size, unit_list, batch_norm, dropout):

    # Activation function.
    activation = nn.ReLU

    module = nn.Sequential()

    units_previous = input_size

    for i, unit in enumerate(unit_list):

        module.add_module(f"fc_{i+1}", nn.Linear(units_previous, unit))

        if i < len(unit_list) - 1:
            if batch_norm:
                module.add_module(f"batch_norm_{i+1}", nn.BatchNorm1d(unit))
            module.add_module(f"activation_{i+1}", activation())
            module.add_module(f"dropout_{i+1}", nn.Dropout(p=dropout))

        units_previous = unit

    return module


def get_model(args: Inputs, dataset: Dataset):

    feature_extractor_list = []

    for i, probe in enumerate(args.probes.probe_list):

        if probe == "density_field":

            # Number of input channels.
            in_channels = dataset[0][0][i].shape[0]
            # Size of input images.
            nside = dataset[0][0][i].shape[1]
            # Dimension of input images (without channel dim).
            dim = dataset[0][0][i].dim() - 1

            if args.train.cnn_type == 1:
                cnn_extractor = get_CNN_extractor(
                    dim,
                    nside,
                    in_channels,
                    args.train.density_field_n_channels_base,
                    args.train.batch_norm,
                )
            elif args.train.cnn_type == 2:
                cnn_extractor = get_CNN_extractor_2(
                    dim,
                    nside,
                    in_channels,
                    args.train.density_field_n_channels_base,
                    args.train.batch_norm,
                )
            else:
                raise ValueError("Wrong value for `cnn_type`. Must be on of: 1, 2.")

            feature_extractor_list.append(cnn_extractor)

        elif probe == "power_spectrum":

            # Input vector size.
            input_size = dataset[0][0][i].shape[0]

            units_list = [
                args.train.ps_fc_units_per_layer for i in range(args.train.ps_fc_layers)
            ]
            feature_extractor_list.append(
                get_FC_extractor(
                    input_size,
                    units_list,
                    args.train.batch_norm,
                    args.train.dropout,
                )
            )

        elif probe == "number_counts":

            # Input vector size.
            input_size = dataset[0][0][i].shape[0]

            units_list = [
                args.train.nc_fc_units_per_layer for i in range(args.train.nc_fc_layers)
            ]
            feature_extractor_list.append(
                get_FC_extractor(
                    input_size,
                    units_list,
                    args.train.batch_norm,
                    args.train.dropout,
                )
            )

        else:
            raise ValueError(f"Unsupported probe: {probe}.")

    feature_extractor_list = nn.ModuleList(feature_extractor_list)
    units_list = [
        args.train.regressor_fc_units_per_layer
        for i in range(args.train.regressor_fc_layers)
    ]
    output_size = dataset[0][1].shape[0]
    model = FC_regressor(
        feature_extractor_list,
        units_list,
        output_size,
        args.train.batch_norm,
        args.train.dropout,
    )

    return model


def feature_extractor_type(probe: str) -> str:

    cnn_probes = ["density_field"]
    fc_probes = ["power_spectrum", "number_counts"]

    if probe in cnn_probes:
        return "cnn"
    elif probe in fc_probes:
        return "fc"
    else:
        raise ValueError(f"Unsupported probe: {probe}.")
