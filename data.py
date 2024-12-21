import torch_geometric as tg
import torch
import numpy as np
from tqdm.notebook import tqdm
import torch
from ogb.utils import smiles2graph
from torch_geometric.data import InMemoryDataset, download_url, Data
from torch_geometric.loader import DataLoader
import numpy as np
import data

INDEX_KEYS = {"edge_index", 'edge_feat', 'node_feat'}

def smiles2torch(smiles):
    graph = smiles2graph(smiles)
    for key in INDEX_KEYS:
        graph[key] = torch.from_numpy(graph[key])
    return Data(x=graph["node_feat"].float(),
                edge_attr=graph["edge_feat"].float(),
                edge_index=graph["edge_index"])


# For use in the backbone GNN model, holds an arbitrary number of
# molecules.
class BlendData(tg.data.Data):
    def __inc__(self, key, value, *args, **kwargs):
        # Used for indexing the molecule into each batch
        # Each blend has only 1 blend (by definition)
        if key == 'blend_batch':
            return 1
        return super().__inc__(key, value, *args, **kwargs)

    @classmethod
    def combine_graphs(cls, graphs):
        # Start with empty tensors for concatenation
        x_list, edge_index_list, edge_attr_list = [], [], []
        mol_batch_list = []
        current_node_index = 0

        for i, graph in enumerate(graphs):
            x_list.append(graph.x)
            edge_attr_list.append(graph.edge_attr)
            edge_index_list.append(graph.edge_index + current_node_index)
            
            # `mol_batch` needs to mark the molecule index for each node
            mol_batch_list.append(torch.full((graph.x.size(0),), i, dtype=torch.long))

            # Update node index offset for the next graph
            current_node_index += graph.x.size(0)

        # Concatenate all lists into single tensors
        x = torch.cat(x_list, dim=0)
        edge_attr = torch.cat(edge_attr_list, dim=0)
        edge_index = torch.cat(edge_index_list, dim=1)
        mol_batch = torch.cat(mol_batch_list, dim=0)
        
        # Create a blend_batch tensor of zeros for each node
        blend_batch = torch.zeros(len(graphs), dtype=torch.long)

        return cls(x=x, edge_attr=edge_attr, edge_index=edge_index, mol_batch=mol_batch, blend_batch=blend_batch)


def convert(datapoint):
    return {
        "mol1": datapoint["edge"][0],
        "mol2": datapoint["edge"][1],
        "blend_notes": datapoint["blend_notes"]
    }

def make_blend_dataset(pair_dataset,
         all_notes=None,
         convert_first=False,
         disable_tqdm=False,
         limit=None):
    if convert_first:
        pair_dataset = [convert(d) for d in pair_dataset]

    if all_notes is None:
        all_notes = set()
        for d in pair_dataset:
            all_notes.update(d["blend_notes"])
        all_notes = list(all_notes)

    # Create a dictionary mapping each label to a unique index
    label_to_index = {label: idx for idx, label in enumerate(all_notes)}

    def multi_hot(notes):
        # Initialize a zero tensor of the appropriate size
        multi_hot_vector = torch.zeros(len(all_notes))

        # Set the corresponding positions in the tensor to 1 for each label of the item
        for label in notes:
            if not label in label_to_index:
                continue
            index = label_to_index[label]
            multi_hot_vector[index] = 1
        return multi_hot_vector

    all_multihots = dict()
    for d in pair_dataset:
        all_multihots[(d["mol1"], d["mol2"])] = multi_hot(d["blend_notes"])

    all_smiles = set()
    for d in pair_dataset:
        all_smiles.add(d["mol1"])
        all_smiles.add(d["mol2"])

    errored = 0
    graph_data = dict()
    for smiles in all_smiles:
        try:
            graph_data[smiles] = smiles2torch(smiles)
        except AttributeError as e:
            print(e)
            errored += 1

    pair_to_data = dict()
    for i, d in enumerate(
            tqdm(pair_dataset, smoothing=0, disable=disable_tqdm)):
        if not d["mol1"] in graph_data or not d["mol2"] in graph_data:
            continue
        pair = (d["mol1"], d["mol2"])
        g1 = graph_data[d["mol1"]]
        g2 = graph_data[d["mol2"]]
        pair_to_data[pair] = BlendData.combine_graphs([g1, g2])

    valid_pairs = set(pair_to_data.keys()).intersection(
        set(all_multihots.keys()))

    dataset = []
    for (pair, graph) in pair_to_data.items():
        dataset.append({
            "pair": pair,
            "graph": graph,
            "notes": all_multihots[pair]
        })

    return np.array(dataset)
