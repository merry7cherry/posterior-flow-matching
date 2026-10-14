import copy
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data
from etflow.cli.chirality import correct, StereoCheck, process

def stereo_data(smiles="C[C@H](O)[C@@H](F)Cl"):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=42) == 0
    pos = mol.GetConformer().GetPositions()
    return Data(smiles=smiles, rdmol=mol, pos_ref=np.array([pos]),
                pos_gen=np.array([pos, pos * [-1, 1, 1]]))

def test_reflection_preserves_geometry_and_uses_input_not_reference():
    original = stereo_data()
    expected = original.pos_gen.copy()
    output = correct(original)
    assert output.chirality["flipped"] == [1]
    assert output.chirality["counts"] == {"all_correct": 1, "all_opposite": 1}
    np.testing.assert_array_equal(original.pos_gen, expected)
    np.testing.assert_array_equal(output.pos_ref, original.pos_ref)
    np.testing.assert_allclose(output.pos_gen[0], output.pos_gen[1])
    for a, b in zip(original.pos_gen, output.pos_gen):
        np.testing.assert_allclose(np.linalg.norm(a[:, None] - a[None, :], axis=-1),
                                   np.linalg.norm(b[:, None] - b[None, :], axis=-1))
    changed_reference = copy.deepcopy(original)
    changed_reference.pos_ref *= [-1, 1, 1]
    np.testing.assert_array_equal(correct(changed_reference).pos_gen, output.pos_gen)

def test_mixed_planar_and_unspecified_are_kept():
    data = stereo_data()
    centers = list(StereoCheck(data.rdmol).expected)
    mol = Chem.Mol(data.rdmol)
    mol.GetAtomWithIdx(centers[0]).InvertChirality()
    data.rdmol = mol
    assert StereoCheck(mol).classify(data.pos_gen[0]) == "mixed"
    np.testing.assert_array_equal(correct(data).pos_gen, data.pos_gen)
    flat = stereo_data()
    flat.pos_gen[:, :, 2] = 0
    assert correct(flat).chirality["counts"] == {"near_planar": 2}
    unspecified = stereo_data("CC(O)C(F)Cl")
    assert correct(unspecified).chirality["counts"] == {"no_specified_centers": 2}


def test_direct_file_correction_keeps_raw_input(tmp_path):
    source, output = tmp_path/'raw.pt', tmp_path/'corrected.pt'
    data = stereo_data()
    torch.save([data], source)
    process(source, output)
    corrected = torch.load(output, weights_only=False)[0]
    original = torch.load(source, weights_only=False)[0]
    assert StereoCheck(corrected.rdmol).classify(corrected.pos_gen[1]) == 'all_correct'
    np.testing.assert_array_equal(original.pos_gen, data.pos_gen)
