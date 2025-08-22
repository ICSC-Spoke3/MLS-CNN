import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from input_args import Inputs

activation_dict = {
    "relu": nn.ReLU,
    "hardswish": nn.Hardswish,
    "leakyrelu": nn.LeakyReLU,
    "gelu": nn.GELU,
    "celu": nn.CELU,
    "selu": nn.SELU,
}


class FCRegressorMultiProbe(nn.Module):

    def __init__(
        self,
        feature_extractor_list: nn.ModuleList,
        n_hidden_layers: int,
        n_units_per_hidden_layer: int,
        output_size: int,
        batch_norm: bool,
        dropout: float,
        activation_func: str,
    ):

        super().__init__()

        self.feature_extractor_list = feature_extractor_list

        # Activation function.
        activation = activation_dict[activation_func]

        # Use or not bias.
        if batch_norm:
            use_bias = False
        else:
            use_bias = True

        # FC module.
        self.fc_module = nn.Sequential()

        for i in range(n_hidden_layers):

            self.fc_module.add_module(
                f"fc_{i+1}",
                nn.LazyLinear(n_units_per_hidden_layer, bias=use_bias),
            )
            if batch_norm:
                self.fc_module.add_module(
                    f"batch_norm_{i+1}", nn.BatchNorm1d(n_units_per_hidden_layer)
                )
            self.fc_module.add_module(f"activation_{i+1}", activation())
            self.fc_module.add_module(f"dropout_{i+1}", nn.Dropout(p=dropout))

        # Output layer.
        self.fc_output = nn.Linear(n_units_per_hidden_layer, output_size * 2)

    def forward(self, x_list):

        # Feature extractor modules.
        feature_list = []
        for x, feature_extractor in zip(x_list, self.feature_extractor_list):
            feature_list.append(feature_extractor(x))

        x = torch.cat(feature_list, dim=-1)

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
        activation_func: str,
    ):

        super().__init__()

        self.feature_extractor = feature_extractor

        # Activation function.
        activation = activation_dict[activation_func]

        # Use or not bias.
        if batch_norm:
            use_bias = False
        else:
            use_bias = True

        # FC module.
        self.fc_module = nn.Sequential()

        for i in range(n_hidden_layers):

            self.fc_module.add_module(
                f"fc_{i+1}",
                nn.LazyLinear(n_units_per_hidden_layer, bias=use_bias),
            )
            if batch_norm:
                self.fc_module.add_module(
                    f"batch_norm_{i+1}", nn.BatchNorm1d(n_units_per_hidden_layer)
                )
            self.fc_module.add_module(f"activation_{i+1}", activation())
            self.fc_module.add_module(f"dropout_{i+1}", nn.Dropout(p=dropout))

        # Output layer.
        self.fc_output = nn.Linear(n_units_per_hidden_layer, output_size * 2)

    def forward(self, x):

        # Feature extractor module.
        x = self.feature_extractor(x)

        # FC module.
        x = self.fc_module(x)

        # Output layer.
        x = self.fc_output(x)

        return x


def get_cnn_extractor(
    dim,
    in_nside: int,
    in_channels: int,
    channels_first: int,
    channels_factor: int,
    final_nside: int,
    batch_norm: bool,
    activation_func: str,
):

    if final_nside >= in_nside:
        raise ValueError(f"`finale_nside` must be strictly smaller than `in_nside`.")

    # Activation function.
    activation = activation_dict[activation_func]

    # Convolution settings.
    # This window conserves the image size.
    conv_kernel_1 = 3
    conv_stride_1 = 1
    conv_padding_1 = 1

    # This window divides the image nside by two.
    conv_kernel_2 = 2
    conv_stride_2 = 2
    conv_padding_2 = 0

    padding_mode = "zeros"

    num_conv_layers = int(np.log2(in_nside / final_nside))

    # Batch norm.
    if dim == 2:
        conv_layer = nn.Conv2d
        batch_norm_layer = nn.BatchNorm2d
        pool_layer = nn.AvgPool2d
    elif dim == 3:
        conv_layer = nn.Conv3d
        batch_norm_layer = nn.BatchNorm3d
        pool_layer = nn.AvgPool3d
    else:
        raise ValueError("Wrong dimension value: ", dim)

    # Use or not bias.
    if batch_norm:
        use_bias = False
    else:
        use_bias = True

    module = nn.Sequential()

    n_channels_previous = in_channels

    for i in range(num_conv_layers):

        n_channels = channels_factor**i * channels_first

        module.add_module(
            f"conv_{i+1}",
            conv_layer(
                n_channels_previous,
                n_channels,
                conv_kernel_1,
                conv_stride_1,
                conv_padding_1,
                padding_mode=padding_mode,
                bias=use_bias,
            ),
        )
        if batch_norm:
            module.add_module(f"batch_norm_{i+1}", batch_norm_layer(n_channels))
        module.add_module(f"activation_{i+1}", activation())

        module.add_module(
            f"avg_pool_{i+1}",
            pool_layer(
                conv_kernel_2,
                conv_stride_2,
                conv_padding_2,
            ),
        )

        n_channels_previous = n_channels

    # Global average pooling.
    module.add_module("global_avg_pool", nn.AdaptiveAvgPool3d(1))

    # Flatten.
    module.add_module("flatten", nn.Flatten(start_dim=1))

    return module


def get_fc_extractor(
    n_hidden_layers, n_units_per_hidden_layer, batch_norm, dropout, activation_func
):

    # Activation function.
    activation = activation_dict[activation_func]

    # Use or not bias.
    if batch_norm:
        use_bias = False
    else:
        use_bias = True

    module = nn.Sequential()

    for i in range(n_hidden_layers):

        module.add_module(
            f"fc_{i+1}", nn.LazyLinear(n_units_per_hidden_layer, bias=use_bias)
        )

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
                args.train.density_field_n_channels_first,
                2,
                args.train.density_field_final_nside,
                args.train.density_field_batch_norm,
                args.train.density_field_activation,
            )

            feature_extractor_list.append(cnn_extractor)

        elif probe == "power_spectrum":

            feature_extractor_list.append(
                get_fc_extractor(
                    args.train.ps_fc_layers,
                    args.train.ps_fc_units_per_layer,
                    args.train.ps_batch_norm,
                    args.train.dropout,
                    args.train.ps_activation,
                )
            )

        elif probe == "number_counts":

            feature_extractor_list.append(
                get_fc_extractor(
                    args.train.nc_fc_layers,
                    args.train.nc_fc_units_per_layer,
                    args.train.nc_batch_norm,
                    args.train.dropout,
                    args.train.nc_activation,
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
            batch_norm=args.train.regressor_batch_norm,
            dropout=args.train.dropout,
            activation_func=args.train.regressor_activation,
        )
    else:
        feature_extractor_list = nn.ModuleList(feature_extractor_list)
        model = FCRegressorMultiProbe(
            feature_extractor_list,
            n_hidden_layers=args.train.regressor_fc_layers,
            n_units_per_hidden_layer=args.train.regressor_fc_units_per_layer,
            output_size=output_size,
            batch_norm=args.train.regressor_batch_norm,
            dropout=args.train.dropout,
            activation_func=args.train.regressor_activation,
        )

    return model
