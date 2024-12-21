import torch
import torch_geometric as pyg
import utils
import activation
import aggregate
import message
from typing import List

class GCN(torch.nn.Module):

    def __init__(self, in_features, conv_configurations: List[message.ConvConfiguration],
                 num_hidden_layers, dropout, do_two_stage, num_heads, num_sabs,
                 notes_dim, act_mode, do_feature_norm, do_conv_norm, do_residual, **kwargs):
        """
        Initializes the GCN model.

        Args:
            in_features (int): Number of input features for each node.
            conv_configurations (List[ConvConfiguration]): List of configurations for convolution blocks.
            num_hidden_layers (int): Number of hidden layers in the MLP of GINConv.
            dropout (float): Dropout rate to apply.
            notes_dim (int): Output dimension for note predictions.
            do_feature_norm (bool): Whether to apply feature normalization.
            do_conv_norm (bool): Whether to apply normalization after convolution.
            do_residual (bool): Whether to apply residual connections at the GCN level.
        """
        super().__init__()

        self.do_residual = do_residual

        # Activation function
        self.act_fn = activation.get_act_fn(act_mode)

        # GNN layers with configurations
        self.conv_blocks = torch.nn.ModuleList()
        current_dim = in_features
        for config in conv_configurations:
            self.conv_blocks.append(
                message.GINConvBlock(
                    in_dim=current_dim,
                    config=config,
                    act_fn=self.act_fn,
                    num_hidden_layers=num_hidden_layers,
                    dropout=dropout,
                    do_feature_norm=do_feature_norm,
                    do_conv_norm=do_conv_norm,
                    do_residual=do_residual
                )
            )
            current_dim = config.hidden_dim

        self.readout = aggregate.BlendAggregator(
            do_two_stage,
            in_channels=current_dim,
            heads=num_heads,
            num_sabs=num_sabs,
            dropout=dropout
        )

        self.notes_predictor = torch.nn.Linear(current_dim, notes_dim)

    def forward(self, graph):
        """
        Defines the forward pass of the GCN.

        Args:
            graph: A PyG data object with attributes `x` (node features) and `edge_index` (edge connectivity).

        Returns:
            dict: Contains 'embed' (node embeddings) and 'logits' (note predictions).
        """
        x = graph.x
        residual = None
        for conv_block in self.conv_blocks:
            x = conv_block(x, graph.edge_index, residual)
            if self.do_residual:
                residual = x

        # Save the embedding
        x = self.readout(x, graph)
        embedding = x.clone()

        # Predict notes using the final embedding
        predictions = self.notes_predictor(x)

        return {"embed": embedding, "logits": predictions}

if __name__ == "__main__":
    # Mock configurations and inputs
    in_features = 16
    conv_configurations = [
        message.ConvConfiguration(num_convs=2, hidden_dim=32),
        message.ConvConfiguration(num_convs=3, hidden_dim=64)
    ]
    num_hidden_layers = 2
    dropout = 0.1
    do_two_stage = True
    num_heads = 4
    num_sabs = 2
    notes_dim = 10
    act_mode = "relu"
    do_feature_norm = True
    do_conv_norm = True
    do_residual = True

    model = GCN(
        in_features=in_features,
        conv_configurations=conv_configurations,
        num_hidden_layers=num_hidden_layers,
        dropout=dropout,
        do_two_stage=do_two_stage,
        num_heads=num_heads,
        num_sabs=num_sabs,
        notes_dim=notes_dim,
        act_mode=act_mode,
        do_feature_norm=do_feature_norm,
        do_conv_norm=do_conv_norm,
        do_residual=do_residual
    )

    # Mock graph input
    graph = pyg.data.Data(
        x=torch.rand(5, in_features),  # 5 nodes with in_features each
        edge_index=torch.tensor([[0, 1, 2, 3, 4, 0], [1, 2, 3, 4, 0, 1]]),  # Simple cyclic graph
        mol_batch=torch.tensor([0,0,0,0,0]),
        blend_batch=torch.tensor([0]),
    )

    # Forward pass
    output = model(graph)

    assert "embed" in output
    assert "logits" in output
    assert output["embed"].shape == (1, conv_configurations[-1].hidden_dim)
    assert output["logits"].shape == (1, notes_dim)

    print("Test passed!")
