"""Preparation keeps held-out molecules and excludes matching training graphs."""
import pickle
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from etflow.cli.prepare import prepare


def test_native_preparation_split(tmp_path):
    raw, metadata, output = [tmp_path/name for name in ['raw', 'metadata', 'processed']]
    raw.mkdir(); metadata.mkdir()
    mols = []
    for i, smiles in enumerate(['CC', 'CCO', 'CC']):
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        AllChem.EmbedMolecule(mol, randomSeed=42)
        mols.append(mol)
        with (raw/f'{i}.pickle').open('wb') as f:
            pickle.dump(dict(conformers=[dict(rd_mol=mol, totalenergy=0., boltzmannweight=1.)]), f)
    np.save(metadata/'split.npy', np.array([np.array([0, 1]), np.array([2]), np.array([], dtype=int)], dtype=object))
    test = Chem.AddHs(Chem.MolFromSmiles('CCC'))
    AllChem.EmbedMolecule(test, randomSeed=42)
    with (metadata/'test_mols.pkl').open('wb') as f:
        pickle.dump({'CCC': [test]}, f)
    assert prepare(raw, metadata, output) == {'train': 1, 'val': 1, 'test': 1}
    assert len(list((output/'qm9/train').glob('*.pt'))) == 1
