"""Preparation keeps held-out molecules and excludes matching training graphs."""
import pickle
import numpy as np
import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from etflow.cli.prepare import prepare
from etflow.data.prepare import process_mol, process_test_mol


def test_native_preparation_split(tmp_path):
    raw, metadata, output = [tmp_path/name for name in ['raw', 'metadata', 'processed']]
    raw.mkdir(); metadata.mkdir()
    mols = []
    for i, smiles in enumerate(['CC', 'CCO', 'CC']):
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        AllChem.EmbedMolecule(mol, randomSeed=42)
        mols.append(mol)
        with (raw/f'{i}.pickle').open('wb') as f:
            pickle.dump(dict(conformers=[dict(rd_mol=mol, boltzmannweight=1.)]), f)
    np.save(metadata/'split.npy', np.array([np.array([0, 1]), np.array([2]), np.array([], dtype=int)], dtype=object))
    test = Chem.AddHs(Chem.MolFromSmiles('CCC'))
    AllChem.EmbedMolecule(test, randomSeed=42)
    with (metadata/'test_mols.pkl').open('wb') as f:
        pickle.dump({'CCC': [test]}, f)
    assert prepare(raw, metadata, output) == {'train': 1, 'val': 1, 'test': 1}
    assert len(list((output/'qm9/train').glob('*.pt'))) == 1


@pytest.mark.parametrize('split, expected', [
    ('train', [1, 2]), ('val', [1, 2]), ('test', [0, 1, 2]),
])
def test_boltzmann_conformer_selection(tmp_path, split, expected):
    mol = Chem.AddHs(Chem.MolFromSmiles('CCO'))
    AllChem.EmbedMolecule(mol, randomSeed=42)
    positions = mol.GetConformer().GetPositions()
    conformers = []
    for index, weight in enumerate([0.1, 0.6, 0.3]):
        conformer = Chem.Mol(mol)
        for atom, position in enumerate(positions + index):
            conformer.GetConformer().SetAtomPosition(atom, position)
        conformers.append(dict(rd_mol=conformer, boltzmannweight=weight))
    path = tmp_path/'molecule.pickle'
    with path.open('wb') as f:
        pickle.dump(dict(conformers=conformers), f)

    data = process_mol(path, 'qm9', split, keep_top_n=2)
    expected_pos = torch.tensor(np.stack([positions + i for i in expected])).float()
    torch.testing.assert_close(data.pos, expected_pos)
    assert data.atomic_numbers.tolist() == [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    if split == 'test':
        test_data = process_test_mol([conf['rd_mol'] for conf in conformers], 'qm9')
        torch.testing.assert_close(test_data.pos, expected_pos)
