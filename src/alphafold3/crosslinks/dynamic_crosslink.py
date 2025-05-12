import rdkit.Chem as rd_chem
from rdkit.Chem import AllChem
from alphafold3.data.tools.rdkit_utils import assign_atom_names_from_graph
from alphafold3.cpp import cif_dict
from rdkit.Chem import Descriptors

from typing import Dict, Union
import re
import functools
import collections
from collections.abc import Mapping
import numpy as np

_RDKIT_MMCIF_TO_BOND_TYPE: Mapping[str, rd_chem.BondType] = {
    'SING': rd_chem.BondType.SINGLE,
    'DOUB': rd_chem.BondType.DOUBLE,
    'TRIP': rd_chem.BondType.TRIPLE,
}

_RDKIT_BOND_TYPE_TO_MMCIF: Mapping[rd_chem.BondType, str] = {
    v: k for k, v in _RDKIT_MMCIF_TO_BOND_TYPE.items()
}

_RDKIT_BOND_STEREO_TO_MMCIF: Mapping[rd_chem.BondStereo, str] = {
    rd_chem.BondStereo.STEREONONE: 'N',
    rd_chem.BondStereo.STEREOE: 'E',
    rd_chem.BondStereo.STEREOZ: 'Z',
    rd_chem.BondStereo.STEREOCIS: 'Z',
    rd_chem.BondStereo.STEREOTRANS: 'E',
}

_CBETA_BOND = {
    "moltype": "protein",
    "atomtypes": [
        {"restype": restype, "atomname": "CB" if restype != "GLY" else "CA"}
        for restype in (
            "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU",
            "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"
        )
    ],
}

# EXPERIMENTAL
_LAST_SIDECHAIN_BOND = {
    "moltype": "protein",
    "atomtypes": [
        {"restype": restype, "atomname": atomname}
        for restype, atomname in {
            "ALA": "CB",
            "ARG": "NH2",
            "ASN": "ND2",
            "ASP": "OD2",
            "CYS": "SG",
            "GLN": "NE2",
            "GLU": "OE2",
            "GLY": "CA",  # Glycine has no side chain, using CA for consistency
            "HIS": "NE2",
            "ILE": "CD1",
            "LEU": "CD2",
            "LYS": "NZ",
            "MET": "CE",
            "PHE": "CZ",
            "PRO": "CD",
            "SER": "OG",
            "THR": "OG1",
            "TRP": "CH2",
            "TYR": "OH",
            "VAL": "CG2",
        }.items()
    ],
}

REGISTERED_LINK_TYPES ={}

def register_dynamic_link(link_type_name: str):
    """Decorator to register a definition function for a specific corsslinks types."""
    def decorator(func):
        REGISTERED_LINK_TYPES[link_type_name] = func
        return func
    return decorator

def issupported(xlinkname: str) -> bool:
    """Checks if dynamic croslink is supported"""
    link_type_name, _ = dynamic_xlinkname_processing(xlinkname)
    if not link_type_name or link_type_name not in REGISTERED_LINK_TYPES:
        return False
    else:
        return True

def dynamic_xlinkname_processing(xlinkname: str) ->  tuple[str, int] | tuple[None, None]:
    """Converts xlinkname to the link type and number of chains"""
    match = re.match(r'^([A-Za-z]+)(\d+)$', xlinkname)
    if match:
        link_type_name = match.group(1)
        num_segments = int(match.group(2))
        return link_type_name, num_segments
    else:
        return None, None

def random_conformer(mol: rd_chem.Mol,
                     num_conformers: int = 1):
    params = AllChem.ETKDGv3()  # Use the ETKDG method for conformer generation
    params.numThreads = 0  # Use all available threads
    AllChem.EmbedMultipleConfs(mol, numConfs=num_conformers, params=params)


def mol_to_ccd_cif_custom(
        mol: rd_chem.Mol,
        component_id: str,
        pdbx_smiles: str | None = None,
        include_hydrogens: bool = True,
) -> cif_dict.CifDict:
    """**Funciton adopted from the from alphafold3.data.tools.rdkit_utils import mol_to_ccd_cif.**

    Creates a CCD-like mmcif data block from an rdkit Mol object.

    Only a subset of associated mmcif fields is populated, but that is
    sufficient for further usage, e.g. in featurization code.

    Atom names can be specified via `atom_name` property. For atoms with
    unspecified value of that property, the name is assigned based on element type
    and the order in the Mol object.

    If the Mol object has associated conformers, atom positions from the first of
    them will be populated in the resulting mmcif file.

    Args:
        mol: An rdkit molecule.
        component_id: Name of the molecule to use in the resulting mmcif. That is
            equivalent to CCD code.
        pdbx_smiles: If specified, the value will be used to populate
            `_chem_comp.pdbx_smiles`.
        include_hydrogens: Whether to include atom and bond data involving
            hydrogens.

    Returns:
        An mmcif data block corresponding for the given rdkit molecule.

    Raises:
        UnsupportedMolBond: When a molecule contains a bond that can't be
            represented with mmcif.
    """
    mol = rd_chem.Mol(mol)
    if include_hydrogens:
        mol = rd_chem.AddHs(mol)
    rd_chem.Kekulize(mol)

    if mol.GetNumConformers() > 0:
        ideal_conformer = mol.GetConformer(0).GetPositions()
        ideal_conformer = np.vectorize(lambda x: f'{x:.3f}')(ideal_conformer)
    else:
        # No data will be populated in the resulting mmcif if the molecule doesn't
        # have any conformers attached to it.
        ideal_conformer = None

    mol_cif = collections.defaultdict(list)
    mol_cif['data_'] = [component_id]
    mol_cif['_chem_comp.id'] = [component_id]
    # modification
    mol_cif['_chem_comp.formula_weight'] = [f'{Descriptors.MolWt(mol):.2f}']
    mol_cif['_chem_comp.name'] = ['?']
    mol_cif['_chem_comp.formula'] = ['?']
    mol_cif['_chem_comp.type'] = ['non-polymer']
    mol_cif['_chem_comp.mon_nstd_parent_comp_id'] = ['?']
    mol_cif['_chem_comp.pdbx_synonyms'] = ['?']

    # IS NOT NECESSARY IN NEWR VERISON
    mol_cif['_pdbx_chem_comp_descriptor.type'] = ['?']
    mol_cif['_pdbx_chem_comp_descriptor.descriptor'] = ['?']

    if pdbx_smiles:
        mol_cif['_chem_comp.pdbx_smiles'] = [pdbx_smiles]

    mol = assign_atom_names_from_graph(mol, keep_existing_names=True)

    for atom_idx, atom in enumerate(mol.GetAtoms()):
        element = atom.GetSymbol()
        if not include_hydrogens and element in ('H', 'D'):
            continue

        mol_cif['_chem_comp_atom.comp_id'].append(component_id)
        mol_cif['_chem_comp_atom.atom_id'].append(atom.GetProp('atom_name'))
        mol_cif['_chem_comp_atom.type_symbol'].append(atom.GetSymbol().upper())
        mol_cif['_chem_comp_atom.charge'].append(str(atom.GetFormalCharge()))
        # modification
        mol_cif['_chem_comp_atom.pdbx_leaving_atom_flag'].append('N')

        if ideal_conformer is not None:
            coords = ideal_conformer[atom_idx]
            mol_cif['_chem_comp_atom.pdbx_model_Cartn_x_ideal'].append(coords[0])
            mol_cif['_chem_comp_atom.pdbx_model_Cartn_y_ideal'].append(coords[1])
            mol_cif['_chem_comp_atom.pdbx_model_Cartn_z_ideal'].append(coords[2])

    for bond in mol.GetBonds():
        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()
        if not include_hydrogens and (
                atom1.GetSymbol() in ('H', 'D') or atom2.GetSymbol() in ('H', 'D')
        ):
            continue
        mol_cif['_chem_comp_bond.comp_id'].append(component_id)
        mol_cif['_chem_comp_bond.atom_id_1'].append(
            bond.GetBeginAtom().GetProp('atom_name')
        )
        mol_cif['_chem_comp_bond.atom_id_2'].append(
            bond.GetEndAtom().GetProp('atom_name')
        )
        try:
            bond_type = bond.GetBondType()
            # Older versions of RDKit did not have a DATIVE bond type. Convert it to
            # SINGLE to match the AF3 training setup.
            if bond_type == rd_chem.BondType.DATIVE:
                bond_type = rd_chem.BondType.SINGLE
            mol_cif['_chem_comp_bond.value_order'].append(
                _RDKIT_BOND_TYPE_TO_MMCIF[bond_type]
            )
            mol_cif['_chem_comp_bond.pdbx_stereo_config'].append(
                _RDKIT_BOND_STEREO_TO_MMCIF[bond.GetStereo()]
            )
        except KeyError as e:
            raise UnsupportedMolBondError from e
        mol_cif['_chem_comp_bond.pdbx_aromatic_flag'].append(
            'Y' if bond.GetIsAromatic() else 'N'
        )
    return cif_dict.CifDict(mol_cif)

def rigid_smiles(n: int):
    if n < 1:
        raise ValueError("n must be at least 1")

    base = "c1ccccc1"  # Benzene ring
    if n == 1:
        return base, "RIGID1"

    smiles = base
    for i in range(2, n + 1):
        num = i if i < 10 else f'%{i}'
        smiles = f"c{num}c{smiles}cc{num}"

    return smiles

def ccd_from_smiles(smiles: str, name: str) -> str:
    mol = rd_chem.MolFromSmiles(smiles)
    mol = rd_chem.AddHs(mol)
    mol = assign_atom_names_from_graph(mol)
    random_conformer(mol)

    ccd_cif = mol_to_ccd_cif_custom(mol, name)

    for atom in mol.GetAtoms():
        atom.SetProp("atomLabel", str(atom.GetProp("atom_name")))

    return ccd_cif

def rigid_atoms_to_connect(n: int):
    atom1 = "C1"
    atom2 = f"C{2+2*n}"
    return  atom1, atom2

@register_dynamic_link("RIGID")
@functools.cache
def rigid_definition(xlinkname: str) -> Dict:
    """Creates rigid definition dictionary of the molecule with the given length"""
    _, n = dynamic_xlinkname_processing(xlinkname)

    smiles = rigid_smiles(n)
    ccd_cif = ccd_from_smiles(smiles, xlinkname)
    atom1, atom2 = rigid_atoms_to_connect(n)

    atom2_bond1, atom2_bond2 = (
        {"moltype": "ligand", "restype": xlinkname, "atomname": cross_atom}
        for cross_atom in (atom1, atom2)
    )

    bonds_dict = {
        "bond1": {
            "atom1": _LAST_SIDECHAIN_BOND,
            "atom2": atom2_bond1
        },
        "bond2": {
            "atom1": _LAST_SIDECHAIN_BOND,
            "atom2": atom2_bond2
        }
    }

    definition = {
        xlinkname: {
            "ccdCode": xlinkname,
            "userCCD": str(ccd_cif),
            **bonds_dict
        }
    }
    return definition

@register_dynamic_link("LINK")
@functools.cache
def flexlink_definition(xlinkname: str) -> Dict:
    """Creates definition dictionary of the poly-C molecule with the given length"""
    _, n = dynamic_xlinkname_processing(xlinkname)
    smiles = n*'C'
    ccd_cif = ccd_from_smiles(smiles, xlinkname)

    atom1, atom2 = "C1", f"C{n}"

    atom2_bond1, atom2_bond2 = (
        {"moltype": "ligand", "restype": xlinkname, "atomname": cross_atom}
        for cross_atom in (atom1, atom2)
    )

    bonds_dict = {
        "bond1": {
            "atom1": _LAST_SIDECHAIN_BOND,
            "atom2": atom2_bond1
        },
        "bond2": {
            "atom1": _LAST_SIDECHAIN_BOND,
            "atom2": atom2_bond2
        }
    }

    definition = {
        xlinkname: {
            "ccdCode": xlinkname,
            "userCCD": str(ccd_cif),
            **bonds_dict
        }
    }
    return definition
