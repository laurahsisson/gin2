import torch
import torch_geometric as pyg
import utils
import activation
import aggregate


class GINConvBlock(torch.nn.Module):
    def __init__(self, hidden_dim, act_fn, num_hidden_layers, do_conv_norm, dropout, do_residual):
        """
        A single convolution block containing GINConv, optional normalization, activation, dropout, and residual connection.

        Args:
            hidden_dim (int): Dimension of hidden layers.
            act_fn (callable): Activation function.
            num_hidden_layers (int): Number of hidden layers in the MLP of GINConv.
            do_conv_norm (bool): Whether to apply normalization.
            dropout (float): Dropout rate to apply.
            do_residual (bool): Whether to apply residual connections.
        """
        super().__init__()

        self.conv = pyg.nn.GINConv(
            utils.build_layers(
                in_dim=hidden_dim,
                hidden_dim=hidden_dim,
                out_dim=hidden_dim,
                act_fn=act_fn,
                num_hidden_layers=num_hidden_layers
            )
        )

        self.norm = torch.nn.BatchNorm1d(hidden_dim) if do_conv_norm else None
        self.act_fn = act_fn
        self.dropout = torch.nn.Dropout(dropout)
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
        x = self.conv(x, edge_index)
        if self.norm:
            x = self.norm(x)
        x = self.act_fn()(x)
        x = self.dropout(x)

        if self.do_residual:
            x = x + residual

        return x


class GCN(torch.nn.Module):

    def __init__(self, in_features, num_convs, num_mpnns, num_hidden_layers,
                 hidden_dim, dropout, do_two_stage, num_heads, num_sabs,
                 notes_dim, act_mode, do_feature_norm, do_conv_norm, do_residual, **kwargs):
        """
        Initializes the GCN model.

        Args:
            in_features (int): Number of input features for each node.
            num_convs (int): Number of convolutional layers per GINConv block.
            num_hidden_layers (int): Number of hidden layers in the MLP of GINConv.
            hidden_dim (int): Dimension of hidden layers.
            dropout (float): Dropout rate to apply.
            notes_dim (int): Output dimension for note predictions.
            do_feature_norm (bool): Whether to apply feature normalization.
            do_conv_norm (bool): Whether to apply convolution normalization.
            do_residual (bool): Whether to apply residual connections.
        """
        super().__init__()

        self.do_feature_norm = do_feature_norm

        # Activation function
        self.act_fn = activation.get_act_fn(act_mode)

        # Feature normalization
        if self.do_feature_norm:
            self.feature_norm = torch.nn.BatchNorm1d(in_features)

        # Initial projection of node features to hidden dimension
        self.project_node_feats = torch.nn.Sequential(
            torch.nn.Linear(in_features, hidden_dim),
            self.act_fn(),
            torch.nn.Dropout(dropout)
        )

        # GNN layers with hidden state normalization
        self.conv_blocks = torch.nn.ModuleList([
            GINConvBlock(
                hidden_dim=hidden_dim,
                act_fn=self.act_fn,
                num_hidden_layers=num_hidden_layers,
                do_conv_norm=do_conv_norm,
                dropout=dropout,
                do_residual=do_residual
            ) for _ in range(num_mpnns)
        ])

        self.readout = aggregate.BlendAggregator(
            do_two_stage,
            in_channels=hidden_dim,
            heads=num_heads,
            num_sabs=num_sabs,
            dropout=dropout
        )

        self.notes_predictor = torch.nn.Linear(hidden_dim, notes_dim)

    def forward(self, graph):
        """
        Defines the forward pass of the GCN.

        Args:
            graph: A PyG data object with attributes `x` (node features) and `edge_index` (edge connectivity).

        Returns:
            dict: Contains 'embed' (node embeddings) and 'logits' (note predictions).
        """
        # Feature normalization
        if self.do_feature_norm:
            x = self.feature_norm(graph.x)
        else:
            x = graph.x

        # Initial projection of node features
        x = self.project_node_feats(x)

        # Residual connections and graph convolutions
        residual = x
        for conv_block in self.conv_blocks:
            x = conv_block(x, graph.edge_index, residual)
            residual = x

        # Save the embedding
        x = self.readout(x, graph)
        embedding = x.clone()

        # Predict notes using the final embedding
        predictions = self.notes_predictor(x)

        return {"embed": embedding, "logits": predictions}
