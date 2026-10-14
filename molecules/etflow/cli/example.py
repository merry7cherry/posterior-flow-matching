"""Create a tiny RDKit-embedded demo dataset, not the GEOM benchmark."""
import argparse
from pathlib import Path
import json
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError('Choose an empty output directory')
    # Disjoint molecular graphs in each split; generated conformers are fixtures.
    splits = {'train': ['CC', 'CCO', 'CCN', 'CO'], 'val': ['CCC', 'CCCO'], 'test': ['CCOC', 'CCCN']}
    for split, smiles_list in splits.items():
        folder = args.output_dir / 'qm9' / split
        folder.mkdir(parents=True, exist_ok=True)
        for i, smiles in enumerate(smiles_list):
            mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
            if len(AllChem.EmbedMultipleConfs(mol, numConfs=2, randomSeed=42)) != 2:
                raise RuntimeError('RDKit failed to embed example')
            for j, atom in enumerate(mol.GetAtoms()):
                atom.SetAtomMapNum(j + 1)
            torch.save(Data(smiles=Chem.MolToSmiles(mol, canonical=False, allHsExplicit=True),
                pos=torch.tensor([c.GetPositions().tolist() for c in mol.GetConformers()], dtype=torch.float32),
                atomic_numbers=torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()])), folder / f'{i}.pt')
    (args.output_dir / 'example.json').write_text(json.dumps({'kind': 'RDKit synthetic demo; not GEOM', 'splits': splits}, indent=2))


if __name__ == '__main__':
    main()
