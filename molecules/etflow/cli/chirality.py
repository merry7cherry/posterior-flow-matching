"""Conservative whole-conformer reflection using input stereo, never reference RMSD."""

import argparse
from collections import Counter
import copy
from pathlib import Path

import numpy as np
from rdkit import Chem
import torch

TETRA = {Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW}
POLICY = dict(name="input-stereo-global-reflection-v1", axis=0,
              normalized_volume_min=1e-3, unresolved="keep", reference_selection=False)


class StereoCheck:
    def __init__(self, template):
        self.expected = {a.GetIdx(): a.GetChiralTag() for a in template.GetAtoms()
                         if a.GetChiralTag() in TETRA}
        self.neighbors = {i: [a.GetIdx() for a in template.GetAtomWithIdx(i).GetNeighbors()]
                          for i in self.expected}
        self.mol = Chem.Mol(template)
        self.mol.RemoveAllConformers()
        conf = Chem.Conformer(template.GetNumAtoms())
        conf.Set3D(True)
        self.mol.AddConformer(conf, assignId=True)

    def classify(self, positions):
        if not self.expected:
            return "no_specified_centers"
        pos = np.asarray(positions, dtype=np.float64)
        if pos.shape != (self.mol.GetNumAtoms(), 3) or not np.isfinite(pos).all():
            raise ValueError("Invalid stereo coordinates")
        for i, neighbors in self.neighbors.items():
            if len(neighbors) != 4:
                return "unsupported_center"
            points = pos[neighbors]
            scale = np.sqrt(np.mean(np.sum((points - pos[i]) ** 2, axis=1)))
            volume = abs(np.linalg.det(points[:3] - points[3]))
            if scale < 1e-12 or volume / scale**3 < POLICY["normalized_volume_min"]:
                return "near_planar"
        self.mol.GetConformer().SetPositions(pos)
        Chem.AssignAtomChiralTagsFromStructure(self.mol, confId=0, replaceExistingTags=True)
        observed = [self.mol.GetAtomWithIdx(i).GetChiralTag() for i in self.expected]
        if any(tag not in TETRA for tag in observed):
            return "undetermined"
        matches = [tag == expected for tag, expected in zip(observed, self.expected.values())]
        return "all_correct" if all(matches) else ("all_opposite" if not any(matches) else "mixed")


def correct(data):
    checker = StereoCheck(data.rdmol)
    result = copy.deepcopy(data)
    result.pos_gen = np.array(data.pos_gen, copy=True)
    stats, flipped = Counter(), []
    for j, pos in enumerate(data.pos_gen):
        before = checker.classify(pos)
        stats[before] += 1
        if before == "all_opposite":
            candidate = np.array(pos, copy=True)
            candidate[:, POLICY["axis"]] *= -1
            if checker.classify(candidate) != "all_correct":
                raise ValueError("Global reflection failed stereo revalidation")
            result.pos_gen[j] = candidate
            flipped.append(j)
    result.chirality = dict(counts=dict(stats), flipped=flipped)
    return result


def process(source, output):
    packed = torch.load(source, map_location='cpu', weights_only=False)
    corrected = [correct(data) for data in packed]
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(corrected, output)
    print(f'Corrected {sum(len(d.chirality["flipped"]) for d in corrected)} conformers')
    return corrected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    process(args.source, args.output)


if __name__ == "__main__":
    main()
