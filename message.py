import activation
import torch
import utils
import torch_geometric as pyg
from typing import NamedTuple


class ConvConfiguration(NamedTuple):
    num_convs: int
    hidden_dim: int


class GINConvBlock(torch.nn.Module):
    def __init__(self, in_dim, config: ConvConfiguration, act_fn, num_hidden_layers, dropout, do_feature_norm, do_conv_norm, do_residual):
        """
        A single convolution block containing projection, GINConv, optional normalization, activation, dropout, and residual connection.

        Args:
            in_dim (int): Input dimension for this block.
            config (ConvConfiguration): Configuration for the convolution block.
            act_fn (callable): Activation function.
            num_hidden_layers (int): Number of hidden layers in the MLP of GINConv.
            dropout (float): Dropout rate to apply.
            do_feature_norm (bool): Whether to apply feature normalization for this block.
            do_conv_norm (bool): Whether to apply normalization after convolution.
            do_residual (bool): Whether to apply residual connections.
        """
        super().__init__()

        self.feature_norm = torch.nn.BatchNorm1d(in_dim) if do_feature_norm else None

        self.projection = torch.nn.Sequential(
            torch.nn.Linear(in_dim, config.hidden_dim),
            act_fn(),
            torch.nn.Dropout(dropout)
        )

        self.conv = pyg.nn.GINConv(
            utils.build_layers(
                in_dim=config.hidden_dim,
                hidden_dim=config.hidden_dim,
                out_dim=config.hidden_dim,
                act_fn=act_fn,
                num_hidden_layers=num_hidden_layers
            )
        )

        self.norm = torch.nn.BatchNorm1d(config.hidden_dim) if do_conv_norm else None
        self.act_fn = act_fn
        self.dropout = torch.nn.Dropout(dropout)
        self.num_convs = config.num_convs
        self.do_residual = do_residual

    def forward(self, x, edge_index, residual):
        """
        Forward pass for the convolution block.

        Args:
            x (torch.Tensor): Node features.
            edge_index (torch.Tensor): Edge connectivity.
            residual (torch.Tensor): Residual features to add.

        Returns:
            torch.Tensor: Updated node features.
        """
        x = self.projection(x)
        residual = x.clone()  # Use the projected input as the residual

        for _ in range(self.num_convs):
            x = self.conv(x, edge_index)
            if self.norm:
                x = self.norm(x)
            x = self.act_fn()(x)
            x = self.dropout(x)

        if self.do_residual:
            x = x + residual  # Add the residual (projected input)

        return x

def test_gin_conv_block():
    # Mock configuration and inputs
    in_dim = 16
    config = ConvConfiguration(num_convs=2, hidden_dim=32)
    num_hidden_layers = 2
    dropout = 0.1
    do_feature_norm = True
    do_conv_norm = True
    do_residual = True

    block = GINConvBlock(
        in_dim=in_dim,
        config=config,
        act_fn=activation.get_act_fn("relu"),
        num_hidden_layers=num_hidden_layers,
        dropout=dropout,
        do_feature_norm=do_feature_norm,
        do_conv_norm=do_conv_norm,
        do_residual=do_residual
    )

    # Mock inputs
    x = torch.rand(5, in_dim)  # 5 nodes with in_dim features
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 0], [1, 2, 3, 4, 0, 1]])  # Simple cyclic graph
    residual = x.clone()

    # Forward pass
    output = block(x, edge_index, residual)

    assert output.shape == (5, config.hidden_dim)
    print("GINConvBlock test passed!")

if __name__ == "__main__":
    test_gin_conv_block()
