import rdkit.Chem as rd_chem
from rdkit.Chem import AllChem
from alphafold3.data.tools.rdkit_utils import mol_to_ccd_cif, assign_atom_names_from_graph
from alphafold3.cpp import cif_dict

from typing import Dict, Union
import re
import functools


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
    """Checks if dynamic crosslink is supported"""
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
        if num_segments < 1:
            raise ValueError(f"Number of the {xlinkname} crosslink segments should at least 1")
        return link_type_name, num_segments
    else:
        return None, None

def random_conformer(mol: rd_chem.Mol,
                     num_conformers: int = 1):
    params = AllChem.ETKDGv3()
    params.numThreads = 0
    AllChem.EmbedMultipleConfs(mol, numConfs=num_conformers, params=params)

def rigid_smiles(n: int):
    if n < 1:
        raise ValueError("n must be at least 1")

    base = "c1ccccc1"  # Benzene ring
    if n == 1:
        return base

    smiles = base1
    for i in range(2, n + 1):
        num = i if i < 10 else f'%{i}'
        smiles = f"c{num}c{smiles}cc{num}"

    return smiles

def ccd_from_smiles(smiles: str, name: str) -> str:
    mol = rd_chem.MolFromSmiles(smiles)
    mol = rd_chem.AddHs(mol)
    mol = assign_atom_names_from_graph(mol)
    random_conformer(mol)
    ccd_cif = mol_to_ccd_cif(
        mol=mol,
        component_id=name,
        pdbx_smiles=smiles
    )
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
