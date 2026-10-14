from pathlib import Path
from typing import Optional
import datamol as dm
import torch
from loguru import logger as log
from torch_geometric.data import Data
from etflow.commons.io import load_pkl

def check_disconnected_components(mol):
    """Check for disconnected components using Union-Find algorithm."""
    # Initialize parent array for union-find
    n_nodes = mol.GetNumAtoms()
    parent = list(range(n_nodes))

    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])  # Add reverse edge for undirected graph
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        parent[find(x)] = find(y)

    # Process all edges
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        union(src, dst)

    # Count unique components
    components = {}
    for node in range(n_nodes):
        root = find(node)
        if root not in components:
            components[root] = []
        components[root].append(node)

    return list(components.values())

def process_test_mol(mols: list[dm.Mol], partition: str):
    """Process a single test molecule into a relevant PyG data object."""
    try:
        mol = mols[0]
        smiles = dm.to_smiles(
            mol,
            canonical=False,
            explicit_hs=True,
            with_atom_indices=True,
            isomeric=True,
        )

        atomic_numbers = torch.tensor(
            [atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.long
        )
        atomic_charges = torch.tensor(
            [atom.GetFormalCharge() for atom in mol.GetAtoms()], dtype=torch.long
        )

        # get conformer positions from mol.GetConformer().GetPositions()
        pos = torch.tensor([mol.GetConformer().GetPositions() for mol in mols]).float()
        data = Data(
            atomic_numbers=atomic_numbers,
            charges=atomic_charges,
            pos=pos,
            smiles=smiles,
            subset=partition,
        )
        return data
    except Exception as e:
        log.warning(f"Skipping: {smiles} due to {e}")
        return None

def process_mol(
    pkl_path: Path,
    partition: str,
    split: str,
    keep_top_n: int = 30,
) -> Optional[Data]:
    """Process a single molecule with all its conformers into one PyG Data object."""
    try:
        # Load molecule data
        mol_dict = load_pkl(pkl_path)
        confs = mol_dict["conformers"]
        mols = [conf["rd_mol"] for conf in confs]

        # Get SMILES with atom indices (use first mol as they're all same)
        smiles = dm.to_smiles(
            mols[0],
            canonical=False,
            explicit_hs=True,
            with_atom_indices=True,
            isomeric=True,
        )

        # Get chemical information (same for all conformers)
        mol = mols[0]
        atomic_numbers = torch.tensor(
            [atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.long
        )
        atomic_charges = torch.tensor(
            [atom.GetFormalCharge() for atom in mol.GetAtoms()], dtype=torch.long
        )

        # Sort conformers by Boltzmann weight if train/val split
        if split in ["train", "val"]:
            confs = sorted(confs, key=lambda x: x["boltzmannweight"], reverse=True)
            confs = confs[:keep_top_n]  # Keep top 30 conformers

        # Collect conformer positions and energies
        positions = []
        energies = []
        boltzmann_weights = []  # Also store the weights

        for conf in confs:
            mol = conf["rd_mol"]
            pos = torch.from_numpy(mol.GetConformer().GetPositions()).float()
            energy = torch.tensor([conf["totalenergy"]]).float()
            weight = torch.tensor([conf["boltzmannweight"]]).float()

            positions.append(pos)
            energies.append(energy)
            boltzmann_weights.append(weight)

        # Stack conformer data
        positions = torch.stack(positions)  # [num_conformers, num_atoms, 3]
        energies = torch.stack(energies)  # [num_conformers, 1]
        boltzmann_weights = torch.stack(boltzmann_weights)  # [num_conformers, 1]

        # Check for disconnected components
        components = check_disconnected_components(mol)
        if len(components) > 1:
            log.warning(f"Skipping {smiles} due to disconnected components")
            return None

        # Create single PyG Data object with all conformers
        data = Data(
            atomic_numbers=atomic_numbers,  # [num_atoms]
            charges=atomic_charges,  # [num_atoms]
            pos=positions,  # [num_conformers, num_atoms, 3]
            energy=energies,  # [num_conformers, 1]
            smiles=smiles,
            subset=partition,
        )

        return data

    except Exception as e:
        log.warning(f"Skipping due to {e}")
        return None
