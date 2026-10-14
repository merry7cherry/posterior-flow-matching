import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Batch, Data
from etflow.commons.featurization import MoleculeFeaturizer
from etflow.models.pfm_model import PosteriorFlow
from etflow.models.pfm_probability import center


def molecule(smiles="CCO"):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=42)
    smiles = Chem.MolToSmiles(mol, canonical=False, allHsExplicit=True)
    f = MoleculeFeaturizer()
    mol = f.get_mol(smiles)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    edge, attr = f.get_edge_index(smiles, False)
    return Data(
        pos=torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float),
        atomic_numbers=torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()]),
        edge_index=edge,
        edge_attr=attr,
        node_attr=f.get_atom_features(smiles),
        smiles=smiles,
    )


def model():
    return PosteriorFlow(
        hidden_channels=32,
        num_layers=2,
        num_heads=4,
        node_attr_dim=10,
        edge_attr_dim=1,
        so3_equivariant=True,
        clip_during_norm=True,
    )


def test_model_backward_equivariance_and_sde(tmp_path):
    torch.manual_seed(7)
    m = model()
    b = Batch.from_data_list([molecule(), molecule("CC")])
    x = center(b.pos, b.batch)
    t = torch.full((2, 1), 0.4)
    args = (b.atomic_numbers, t, x, b.edge_index, b.batch, b.node_attr, b.edge_attr)
    v, lv = m.predict_posterior(*args)
    # Ensure the scalar head actually depends on coordinates for the derivative test.
    torch.nn.init.normal_(m.network.variance_head[-1].weight, std=0.02)
    v, lv = m.predict_posterior(*args)
    Q = torch.linalg.qr(torch.randn(3, 3))[0]
    if torch.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    vr, lvr = m.predict_posterior(
        b.atomic_numbers, t, x @ Q, b.edge_index, b.batch, b.node_attr, b.edge_attr
    )
    torch.testing.assert_close(vr, v @ Q, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(lvr, lv, atol=2e-4, rtol=2e-4)
    m.log = lambda *a, **kw: None
    loss = m.generic_step(b, 0, "train")
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    for mode in ["ode", "sde"]:
        with torch.no_grad():
            y, frames = m.sample(
                b.atomic_numbers,
                b.edge_index,
                b.batch,
                b.node_attr,
                b.edge_attr,
                n_timesteps=3,
                sampler_type=mode,
                return_trajectory=True,
            )
        assert torch.isfinite(y).all() and frames
        torch.testing.assert_close(center(y, b.batch), y, atol=1e-6, rtol=0)
    path = tmp_path / "weights.pt"
    torch.save(m.state_dict(), path)
    restored = model()
    restored.load_state_dict(torch.load(path, weights_only=True))
    for a, c in zip(m.parameters(), restored.parameters()):
        torch.testing.assert_close(a, c)
