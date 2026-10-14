from typing import Tuple
import torch
from torch_geometric.utils import scatter
from etflow.commons.utils import extend_graph_order_radius

def center_of_mass(x, dim=0, batch=None):
    num_nodes = x.size(0)

    if batch is None:
        batch = torch.zeros(num_nodes, dtype=torch.long, device=x.device)

    x_com = scatter(x, batch, dim=dim, reduce="mean")[batch]
    return x - x_com

def extend_bond_index(
    pos: torch.Tensor,
    bond_index: torch.Tensor,
    batch: torch.Tensor,
    bond_attr: torch.Tensor,
    device: torch.device,
    one_hot: bool = False,
    one_hot_types: int = 5,
    cutoff: float = 10.0,
    max_num_neighbors: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if bond_attr is None:
        bond_type = torch.ones(bond_index.shape[1], dtype=torch.long, device=device)
        # all molecular graph edges are type 1, radius based become 0
    else:
        bond_type = bond_attr.view(-1).long() + 1  # we reserve 0 for radius based edges
        assert (
            bond_type.shape[0] == bond_index.shape[1]
        ), "Edge type should have same shape as number of edges."

    edge_index, edge_type = extend_graph_order_radius(
        pos=pos,
        edge_index=bond_index,
        edge_type=bond_type,
        batch=batch,
        cutoff=cutoff,
        max_num_neighbors=max_num_neighbors,
        extend_radius=True,
    )
    assert (
        bond_index.shape[1] == (edge_type > 0).sum().item()
    ), "Edge Type should be greater than 0 when edge is a molecular bond."

    # make one_hot if provided
    if one_hot:
        # +1 to account for radius based edges
        edge_type = torch.nn.functional.one_hot(
            edge_type, num_classes=one_hot_types + 1
        ).float()

    return edge_index, edge_type
