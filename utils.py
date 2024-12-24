import torch
import copy

def count_parameters(module: torch.nn.Module):
    return sum(p.numel() for p in module.parameters())


def make_scheduler(optimizer, warmup, total_steps):
    warmup_steps = int(warmup * total_steps)
    cooldown_steps = total_steps - warmup_steps
    # Define the warmup and cooldown schedulers
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-10,
        end_factor=1.0,
        total_iters=warmup_steps)
    cooldown_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=1e-10,
        total_iters=cooldown_steps)
    # Chain the schedulers together with SequentialLR
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cooldown_scheduler],
        milestones=[warmup_steps])


def build_layers(in_dim, hidden_dim, out_dim, act_fn, num_hidden_layers):
    layers = []

    # If there are no hidden layers, just add the output layer
    if num_hidden_layers == 0:
        layers.append(torch.nn.Linear(in_dim, out_dim))
    else:
        # Input layer
        layers.append(torch.nn.Linear(in_dim, hidden_dim))
        layers.append(act_fn())

        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
            layers.append(act_fn())

        # Output layer
        layers.append(torch.nn.Linear(hidden_dim, out_dim))

    return torch.nn.Sequential(*layers)

def add_global_node(graph):
    graph = copy.deepcopy(graph)
    # Add a global node feature, initialized as zeros or some value
    global_node_feature = torch.zeros((1, graph.x.size(1)))
    graph.x = torch.cat([graph.x, global_node_feature], dim=0)
    
    # Create edges from the global node to all nodes
    num_nodes = graph.x.size(0) - 1  # Exclude the newly added global node
    global_node_index = num_nodes
    global_edges = torch.cat([
        torch.arange(num_nodes).unsqueeze(0).repeat(2, 1),
        torch.tensor([[global_node_index], [global_node_index]])
    ], dim=1)
    global_edges = torch.cat([global_edges, torch.flip(global_edges, [0])], dim=1)
    graph.edge_index = torch.cat([graph.edge_index, global_edges], dim=1)
    
    # Optionally add attributes for global edges (e.g., zeros or specific values)
    if graph.edge_attr is not None:
        global_edge_attr = torch.zeros((global_edges.size(1), graph.edge_attr.size(1)))
        graph.edge_attr = torch.cat([graph.edge_attr, global_edge_attr], dim=0)
    
    # Update mol_batch and blend_batch
    global_mol_index = torch.amax(graph.mol_batch)+1
    global_mol_batch = torch.tensor([global_mol_index])  # Assign the global node to a new molecule
    
    global_blend_batch = torch.tensor([0])  # Assign the global node to the first blend
    graph.mol_batch = torch.cat([graph.mol_batch, global_mol_batch], dim=0)
    graph.blend_batch = torch.cat([graph.blend_batch, global_blend_batch], dim=0)
    
    return graph