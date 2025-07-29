import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from input_args import Inputs


class FCRegressorMultiProbe(nn.Module):

    def __init__(
        self,
        feature_extractor_list: nn.ModuleList,
        n_hidden_layers: int,
        n_units_per_hidden_layer: int,
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

        for i in range(n_hidden_layers):

            if i == 0:
                self.fc_module.add_module(
                    f"fc_{i+1}",
                    nn.Linear(n_features, n_units_per_hidden_layer),
                )
            else:
                self.fc_module.add_module(
                    f"fc_{i+1}",
                    nn.Linear(n_units_per_hidden_layer, n_units_per_hidden_layer),
                )
            self.fc_module.add_module(f"activation_{i+1}", activation())
            self.fc_module.add_module(f"dropout_{i+1}", nn.Dropout(p=dropout))

        # Output layer.
        self.fc_output = nn.Linear(n_units_per_hidden_layer, output_size * 2)

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
        x = self.fc_output(x)

        return x


class FCRegressorSingleProbe(nn.Module):

    def __init__(
        self,
        feature_extractor: nn.Module,
        n_hidden_layers: int,
        n_units_per_hidden_layer: int,
        output_size: int,
        batch_norm: bool,
        dropout: float,
    ):
        super().__init__()

        self.feature_extractor = feature_extractor

        self.batch_norm = batch_norm

        # Activation.
        activation = nn.ReLU

        # Feature size.
        *_, module1_last_layer = self.feature_extractor.modules()
        n_features = module1_last_layer.out_features

        # Input module.
        self.input_module = nn.Sequential(
            nn.BatchNorm1d(n_features), activation(), nn.Dropout(p=dropout)
        )

        # FC module.
        self.fc_module = nn.Sequential()

        for i in range(n_hidden_layers):

            if i == 0:
                self.fc_module.add_module(
                    f"fc_{i+1}",
                    nn.Linear(n_features, n_units_per_hidden_layer),
                )
            else:
                self.fc_module.add_module(
                    f"fc_{i+1}",
                    nn.Linear(n_units_per_hidden_layer, n_units_per_hidden_layer),
                )
            self.fc_module.add_module(f"activation_{i+1}", activation())
            self.fc_module.add_module(f"dropout_{i+1}", nn.Dropout(p=dropout))

        # Output layer.
        self.fc_output = nn.Linear(n_units_per_hidden_layer, output_size * 2)

    def forward(self, x):

        # Module1 on x1.
        x = self.feature_extractor(x)

        # Input module.
        x = self.input_module(x)

        # FC module.
        x = self.fc_module(x)

        # Output layer.
        x = self.fc_output(x)

        return x


def get_cnn_extractor(
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

    # padding_mode = "circular"
    padding_mode = "zeros"

    num_conv_layers = int(np.log2(in_nside / 2))

    # Batch norm.
    if dim == 2:
        conv_layer = nn.Conv2d
        batch_norm_layer = nn.BatchNorm2d
        # pool_layer = nn.MaxPool2d
    elif dim == 3:
        conv_layer = nn.Conv3d
        batch_norm_layer = nn.BatchNorm3d
        # pool_layer = nn.MaxPool3d
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
                n_channels_previous,
                n_channels,
                conv_kernel_2,
                conv_stride_2,
                conv_padding_2,
                padding_mode=padding_mode,
            ),
        )
        if batch_norm:
            module.add_module(f"batch_norm_{i+1}_2", batch_norm_layer(n_channels))
        module.add_module(f"activation_{i+1}_2", activation())

        n_channels_previous = n_channels

    # Last convolution.
    n_channels_final = 2 ** (num_conv_layers + 1) * channels_base
    module.add_module(
        "conv_final",
        conv_layer(
            n_channels_previous,
            n_channels_final,
            2,
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


def get_fc_extractor(
    input_size, n_hidden_layers, n_units_per_hidden_layer, batch_norm, dropout
):

    # Activation function.
    activation = nn.ReLU

    module = nn.Sequential()

    for i in range(n_hidden_layers):

        if i == 0:
            module.add_module(
                f"fc_{i+1}", nn.Linear(input_size, n_units_per_hidden_layer)
            )
        else:
            module.add_module(
                f"fc_{i+1}",
                nn.Linear(n_units_per_hidden_layer, n_units_per_hidden_layer),
            )

        if i < n_hidden_layers - 1:
            if batch_norm:
                module.add_module(
                    f"batch_norm_{i+1}", nn.BatchNorm1d(n_units_per_hidden_layer)
                )
            module.add_module(f"activation_{i+1}", activation())
            module.add_module(f"dropout_{i+1}", nn.Dropout(p=dropout))

    return module


def get_model(args: Inputs, dataset: Dataset):

    feature_extractor_list = []

    for i, probe in enumerate(args.probes.probe_list):

        if probe == "density_field":

            if len(args.probes.probe_list) > 1:
                # Number of input channels.
                in_channels = dataset[0][0][i].shape[0]
                # Size of input images.
                nside = dataset[0][0][i].shape[1]
                # Dimension of input images (without channel dim).
                dim = dataset[0][0][i].dim() - 1
            else:
                # Number of input channels.
                in_channels = dataset[0][0].shape[0]
                # Size of input images.
                nside = dataset[0][0].shape[1]
                # Dimension of input images (without channel dim).
                dim = dataset[0][0].dim() - 1

            cnn_extractor = get_cnn_extractor(
                dim,
                nside,
                in_channels,
                args.train.density_field_n_channels_base,
                args.train.batch_norm,
            )

            feature_extractor_list.append(cnn_extractor)

        elif probe == "power_spectrum":

            if len(args.probes.probe_list) > 1:
                # Input vector size.
                input_size = dataset[0][0][i].shape[0]
            else:
                # Input vector size.
                input_size = dataset[0][0].shape[0]

            feature_extractor_list.append(
                get_fc_extractor(
                    input_size,
                    args.train.ps_fc_layers,
                    args.train.ps_fc_units_per_layer,
                    args.train.batch_norm,
                    args.train.dropout,
                )
            )

        elif probe == "number_counts":

            if len(args.probes.probe_list) > 1:
                # Input vector size.
                input_size = dataset[0][0][i].shape[0]
            else:
                # Input vector size.
                input_size = dataset[0][0].shape[0]

            feature_extractor_list.append(
                get_fc_extractor(
                    input_size,
                    args.train.nc_fc_layers,
                    args.train.nc_fc_units_per_layer,
                    args.train.batch_norm,
                    args.train.dropout,
                )
            )

        else:
            raise ValueError(f"Unsupported probe: {probe}.")

    output_size = dataset[0][1].shape[0]

    if len(args.probes.probe_list) == 1:
        model = FCRegressorSingleProbe(
            *feature_extractor_list,
            n_hidden_layers=args.train.regressor_fc_layers,
            n_units_per_hidden_layer=args.train.regressor_fc_units_per_layer,
            output_size=output_size,
            batch_norm=args.train.batch_norm,
            dropout=args.train.dropout,
        )
    else:
        feature_extractor_list = nn.ModuleList(feature_extractor_list)
        model = FCRegressorMultiProbe(
            feature_extractor_list,
            n_hidden_layers=args.train.regressor_fc_layers,
            n_units_per_hidden_layer=args.train.regressor_fc_units_per_layer,
            output_size=output_size,
            batch_norm=args.train.batch_norm,
            dropout=args.train.dropout,
        )

    return model
