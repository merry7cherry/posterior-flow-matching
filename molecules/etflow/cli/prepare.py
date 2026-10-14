"""Prepare GEOM-QM9 using the published sorted-pickle split convention."""
import argparse
import json
from pathlib import Path
import pickle

import numpy as np
import torch
from rdkit import Chem
from tqdm import tqdm
from etflow.data.prepare import process_mol, process_test_mol
from etflow.commons.featurization import MoleculeFeaturizer


def molecular_smiles(data):
    mol = MoleculeFeaturizer().get_mol(data.smiles)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=True)


def prepare(raw_dir, metadata_dir, output_dir):
    files = sorted(raw_dir.glob('*.pickle'))
    train_indices, val_indices, _ = np.load(metadata_dir/'split.npy', allow_pickle=True)
    with (metadata_dir/'test_mols.pkl').open('rb') as f:
        test_molecules = pickle.load(f)
    counts, held_out = {}, set()
    # Validation/test are the supplied held-out sets. Exclude duplicate training graphs.
    for split in ['test', 'val', 'train']:
        folder = output_dir/'qm9'/split
        folder.mkdir(parents=True, exist_ok=True)
        counts[split] = 0
        records = (enumerate(test_molecules.values()) if split == 'test'
                   else ((int(i), files[int(i)]) for i in sorted(val_indices if split == 'val' else train_indices)))
        for index, value in tqdm(records, desc=split):
            data = process_test_mol(value, 'qm9') if split == 'test' else process_mol(value, 'qm9', split)
            if data is None:
                continue
            smiles = molecular_smiles(data)
            if split == 'train' and smiles in held_out:
                continue
            if split != 'train':
                held_out.add(smiles)
            torch.save(data, folder/f'{index:06d}.pt')
            counts[split] += 1
    (output_dir/'counts.json').write_text(json.dumps(counts, indent=2))
    print(json.dumps(counts))
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', required=True, type=Path)
    parser.add_argument('--metadata-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    prepare(args.raw_dir, args.metadata_dir, args.output_dir)


if __name__ == '__main__':
    main()
