from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    # Avoid circular import at runtime; use it only for type hints.
    from alphafold3.common.folding_input import ProteinChain, RnaChain, DnaChain, Ligand

from alphafold3.constants import atom_types
from alphafold3.structure import mmcif as mmcif_lib
from alphafold3.crosslinks import crosslink_definitions  # for LinkDefinition
from alphafold3.crosslinks import utils as xutils  # for Residue, restype_matches, get_unused_chain_id
from alphafold3.constants import residue_names

BondAtomId = Tuple[str, int, str]

def restype_matches(restype: str, resid: int, expected: list[str]) -> bool:
    return restype in expected or (resid == 1 and "NTER" in expected)

def atom_in_residue(atom: str, restype: str) -> bool:
    expected_atoms = atom_types.RESIDUE_ATOMS.get(restype, [])
    return atom in expected_atoms

def get_unused_chain_id(used_ids: list[str]) -> str:
    i = 1
    while True:
        candidate = mmcif_lib.int_id_to_str_id(i)
        if candidate not in used_ids:
            return candidate
        i += 1

class Residue:
    def __init__(self, chain_id: str, resid: int, restype: str):
        self.chain_id = chain_id
        self.resid = resid
        self.restype = restype

    def __repr__(self) -> str:
        return f"Residue({self.chain_id}:{self.resid} {self.restype})"

@dataclass(frozen=True)
class LinkHandler:
    """
    Combines residue validation and bond‐pair creation using an associated LinkDefinition.
    """
    link_definition: crosslink_definitions.LinkDefinition
    chains: "Sequence[ProteinChain | RnaChain | DnaChain | Ligand]" # lazy type annotation using forward reference
    residue_pairs: Sequence[Tuple[xutils.Residue, xutils.Residue]]

    def validate_and_get_bond_info(
            self,
            residue1: xutils.Residue,
            residue2: xutils.Residue,
        ) -> Tuple[str, str, str, str]:
        """
        Validates each residue from the pair against the expected residue types
        from the link definition, and returns the bonded atom names.

        Raises:
          ValueError if a residue's restype does not match the expected values.
        """
        expected1 = self.link_definition.get_expected_restypes_bond1()
        expected2 = self.link_definition.get_expected_restypes_bond2()
        if not xutils.restype_matches(residue1.restype, residue1.resid, expected1):
            raise ValueError(
                f"Residue {residue1.chain_id}:{residue1.resid} {residue1.restype} "
                f"does not match expected {expected1}"
            )
        if not xutils.restype_matches(residue2.restype, residue2.resid, expected2):
            raise ValueError(
                f"Residue {residue2.chain_id}:{residue2.resid} {residue2.restype} "
                f"does not match expected {expected2}"
            )
        bond1_atom1 = self.link_definition.get_bond_atom1(residue1.restype, residue1.resid, bond=1)
        bond2_atom1 = self.link_definition.get_bond_atom1(residue2.restype, residue2.resid, bond=2)
        bond1_atom2 = self.link_definition.get_bond_atom2(bond=1)
        bond2_atom2 = self.link_definition.get_bond_atom2(bond=2)
        return bond1_atom1, bond1_atom2, bond2_atom1, bond2_atom2

    def create_bonded_atom_pairs(
        self,
        used_chain_ids: List[str]
    ) -> Tuple[List[Tuple[BondAtomId, BondAtomId]], List[str]]:
        """
        For each residue pair, validates the residues and creates bonded atom pairs.
        For each pair ((residue1, residue2)):
          - Validates the residues using validate_and_get_bond_info.
          - Generates a new ligand chain ID.
          - Creates two bond pairs:
                ((residue1.chain_id, residue1.resid, bond_atom1),
                 (new_ligand_id, ligand_resid, bond_atom2))
                ((residue2.chain_id, residue2.resid, bond_atom3),
                 (new_ligand_id, ligand_resid, bond_atom4))
        Returns:
          A tuple of (list of bonded atom pairs, list of newly generated ligand chain IDs).
        """
        bond_pairs = []
        ligand_ids = []
        ligand_resid = 1
        id_to_chain = {chain.id: chain for chain in self.chains}
        # for residue1, residue2 in self.residue_pairs:
        for (chain_id1, resid1), (chain_id2, resid2) in self.residue_pairs:
            residue1 = xutils.Residue(
                chain_id1,
                resid1,
                residue_names.PROTEIN_COMMON_ONE_TO_THREE[
                    id_to_chain[chain_id1].sequence[resid1 - 1]
                ]
            )
            residue2 = xutils.Residue(
                chain_id2,
                resid2,
                residue_names.PROTEIN_COMMON_ONE_TO_THREE[
                    id_to_chain[chain_id2].sequence[resid2 - 1]
                ]
            )

            bond1_atom1, bond1_atom2, bond2_atom1, bond2_atom2 = \
                self.validate_and_get_bond_info(residue1, residue2)
            new_ligand_id = get_unused_chain_id(used_chain_ids)
            ligand_ids.append(new_ligand_id)
            used_chain_ids.append(new_ligand_id)

            pair1 = ((residue1.chain_id, residue1.resid, bond1_atom1),
                     (new_ligand_id, ligand_resid, bond1_atom2))
            pair2 = ((residue2.chain_id, residue2.resid, bond2_atom1),
                     (new_ligand_id, ligand_resid, bond2_atom2))
            bond_pairs.extend([pair1, pair2])
        return bond_pairs, ligand_ids