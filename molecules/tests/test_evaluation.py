"""Sample outputs feed the native evaluator directly."""
import argparse
import numpy as np
import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data
from etflow.models.pfm_model import PosteriorFlow
from etflow.cli.sample import generate
from etflow.cli.evaluate import evaluate_metrics, CovMatEvaluator, print_covmat_results


@pytest.fixture
def experiment(tmp_path):
    torch.manual_seed(9)
    data_dir = tmp_path/'data'
    folder = data_dir/'qm9/test'
    folder.mkdir(parents=True)
    for i, smiles in enumerate(['CC', 'CCO']):
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        AllChem.EmbedMolecule(mol, randomSeed=42)
        for j, atom in enumerate(mol.GetAtoms()):
            atom.SetAtomMapNum(j+1)
        pos = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float32)
        torch.save(Data(smiles=Chem.MolToSmiles(mol, canonical=False, allHsExplicit=True),
            pos=torch.stack([pos, pos+.01]),
            atomic_numbers=torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()])), folder/f'{i}.pt')
    model_args = dict(hidden_channels=32, num_layers=2, num_heads=4, node_attr_dim=10,
                      edge_attr_dim=1, so3_equivariant=True, clip_during_norm=True)
    cfg = dict(model_args=model_args, seed=42, data_dir=str(data_dir),
               sampling=dict(batch_size=2, n_timesteps=2, terminal_time=.99, diffusion_scale=.3, probes=2))
    checkpoint = tmp_path/'model.ckpt'
    torch.save(dict(pfm_config=cfg, state_dict=PosteriorFlow(**model_args).state_dict()), checkpoint)
    return argparse.Namespace(checkpoint=checkpoint, data_dir=data_dir, output=tmp_path/'generated.pt',
                              mode='ode', steps=None, max_molecules=None, diffusion_scale=None, device='cpu')


def test_sample_and_native_metrics(experiment):
    packed = generate(experiment)
    saved = torch.load(experiment.output, weights_only=False)
    for a, b in zip(packed, saved):
        np.testing.assert_array_equal(a.pos_gen, b.pos_gen)
        assert b.pos_gen.shape == (4, len(b.rdmol.GetAtoms()), 3)
        assert b.endpoint_variance.shape == (len(b.rdmol.GetAtoms()),)
    report = evaluate_metrics(experiment.output, experiment.output.parent/'metrics')
    native, _ = CovMatEvaluator(num_workers=1, use_force_field=False)(packed)
    _, matching = print_covmat_results(native)
    for key, value in matching.items():
        assert report[key] == value
    row = int(np.argmin(abs(native['thresholds']-.5)))
    assert report['COV-R_mean'] == native['CoverageR'][:, row].mean()
    assert report['COV-P_mean'] == native['CoverageP'][:, row].mean()


def test_sde_zero_noise_matches_ode(experiment):
    ode = generate(experiment)
    experiment.mode = 'sde'
    experiment.diffusion_scale = 0
    sde = generate(experiment)
    for a, b in zip(ode, sde):
        np.testing.assert_array_equal(a.pos_gen, b.pos_gen)
