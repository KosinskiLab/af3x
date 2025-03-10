# This file is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# This statement does not change the license of any other file in this software package.
#

#
# TODO: * Currently the SMILES correspond to the full crosslinker, but we need to remove the functional groups
#       that are leaving after crosslinking reaction. But SMILES string are ignored in the current implementation.
#       * The specification format grew organically and might be messy. We should clean it up.
#

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

CROSSLINKS = {

    "azide-A-DSBSO": {
        "ccdCode": "azide-A-DSBSO",
        "userCCD": """data_azide-A-DSBSO
#
_chem_comp.id azide-A-DSBSO
_chem_comp.name '5,8-bis(oxidanyl)naphthalene-1,4-dione'
_chem_comp.type non-polymer
_chem_comp.formula 'C10 H6 O4'
_chem_comp.mon_nstd_parent_comp_id ?
_chem_comp.pdbx_synonyms ?
_chem_comp.formula_weight 190.152
#
loop_
_chem_comp_atom.comp_id
_chem_comp_atom.atom_id
_chem_comp_atom.alt_atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.charge
_chem_comp_atom.pdbx_align
_chem_comp_atom.pdbx_aromatic_flag
_chem_comp_atom.pdbx_leaving_atom_flag
_chem_comp_atom.pdbx_stereo_config
_chem_comp_atom.pdbx_backbone_atom_flag
_chem_comp_atom.pdbx_n_terminal_atom_flag
_chem_comp_atom.pdbx_c_terminal_atom_flag
_chem_comp_atom.model_Cartn_x
_chem_comp_atom.model_Cartn_y
_chem_comp_atom.model_Cartn_z
_chem_comp_atom.pdbx_model_Cartn_x_ideal
_chem_comp_atom.pdbx_model_Cartn_y_ideal
_chem_comp_atom.pdbx_model_Cartn_z_ideal
_chem_comp_atom.pdbx_component_atom_id
_chem_comp_atom.pdbx_component_comp_id
_chem_comp_atom.pdbx_ordinal
azide-A-DSBSO N1 N1 N -1 1 N N N N N N -4.842 7.334 -2.832 -4.842 7.334 -2.832 N1 azide-A-DSBSO 1
azide-A-DSBSO N2 N2 N +1 1 N N N N N N -4.379 6.559 -2.155 -4.379 6.559 -2.155 N2 azide-A-DSBSO 2
azide-A-DSBSO N3 N3 N 0 1 N N N N N N -3.740 5.740 -1.504 -3.740 5.740 -1.504 N3 azide-A-DSBSO 3
azide-A-DSBSO C4 C4 C 0 1 N N N N N N -3.165 6.245 -0.238 -3.165 6.245 -0.238 C4 azide-A-DSBSO 4
azide-A-DSBSO C5 C5 C 0 1 N N N N N N -2.206 5.236 0.362 -2.206 5.236 0.362 C5 azide-A-DSBSO 5
azide-A-DSBSO C6 C6 C 0 1 N N N N N N -0.989 4.963 -0.525 -0.989 4.963 -0.525 C6 azide-A-DSBSO 6
azide-A-DSBSO C7 C7 C 0 1 N N N N N N 0.155 4.221 0.151 0.155 4.221 0.151 C7 azide-A-DSBSO 7
azide-A-DSBSO C8 C8 C 0 1 N N N N N N 1.023 5.191 0.915 1.023 5.191 0.915 C8 azide-A-DSBSO 8
azide-A-DSBSO O9 O9 O 0 1 N N N N N N 1.002 3.560 -0.807 1.002 3.560 -0.807 O9 azide-A-DSBSO 9
azide-A-DSBSO C10 C10 C 0 1 N N N N N N 0.604 2.221 -1.136 0.604 2.221 -1.136 C10 azide-A-DSBSO 10
azide-A-DSBSO C11 C11 C 0 1 N N N N N N 0.628 1.302 0.087 0.628 1.302 0.087 C11 azide-A-DSBSO 11
azide-A-DSBSO C12 C12 C 0 1 N N N N N N 1.968 0.541 0.142 1.968 0.541 0.142 C12 azide-A-DSBSO 12
azide-A-DSBSO S13 S13 S 0 1 N N N N N N 3.434 1.605 0.097 3.434 1.605 0.097 S13 azide-A-DSBSO 13
azide-A-DSBSO C14 C14 C 0 1 N N N N N N 4.620 0.370 -0.471 4.620 0.370 -0.471 C14 azide-A-DSBSO 14
azide-A-DSBSO C15 C15 C 0 1 N N N N N N 5.148 -0.541 0.632 5.148 -0.541 0.632 C15 azide-A-DSBSO 15
azide-A-DSBSO C16 C16 C 0 1 N N N N N N 6.069 -1.570 0.044 6.069 -1.570 0.044 C16 azide-A-DSBSO 16
azide-A-DSBSO O25 O25 O 0 1 N N N N N N 6.783 -1.428 -0.912 6.783 -1.428 -0.912 O25 azide-A-DSBSO 17
azide-A-DSBSO O26 O26 O 0 1 N N N N N N 3.825 1.927 1.511 3.825 1.927 1.511 O26 azide-A-DSBSO 18
azide-A-DSBSO C27 C27 C 0 1 N N N N N N -0.533 0.302 -0.022 -0.533 0.302 -0.022 C27 azide-A-DSBSO 19
azide-A-DSBSO S28 S28 S 0 1 N N N N N N -0.600 -0.921 1.314 -0.600 -0.921 1.314 S28 azide-A-DSBSO 20
azide-A-DSBSO C29 C29 C 0 1 N N N N N N -2.381 -1.206 1.304 -2.381 -1.206 1.304 C29 azide-A-DSBSO 21
azide-A-DSBSO C30 C30 C 0 1 N N N N N N -2.872 -2.092 0.164 -2.872 -2.092 0.164 C30 azide-A-DSBSO 22
azide-A-DSBSO C31 C31 C 0 1 N N N N N N -4.354 -2.306 0.268 -4.354 -2.306 0.268 C31 azide-A-DSBSO 23
azide-A-DSBSO O40 O40 O 0 1 N N N N N N -5.138 -1.618 0.867 -5.138 -1.618 0.867 O40 azide-A-DSBSO 24
azide-A-DSBSO O41 O41 O 0 1 N N N N N N 0.023 -2.188 0.802 0.023 -2.188 0.802 O41 azide-A-DSBSO 25
azide-A-DSBSO C42 C42 C 0 1 N N N N N N 0.460 2.149 1.349 0.460 2.149 1.349 C42 azide-A-DSBSO 26
azide-A-DSBSO O43 O43 O 0 1 N N N N N N -0.380 3.285 1.104 -0.380 3.285 1.104 O43 azide-A-DSBSO 27
azide-A-DSBSO H4A H4A H 0 1 N N N N N N -2.654 7.200 -0.384 -2.654 7.200 -0.384 H4A azide-A-DSBSO 28
azide-A-DSBSO H4B H4B H 0 1 N N N N N N -3.985 6.420 0.463 -3.985 6.420 0.463 H4B azide-A-DSBSO 29
azide-A-DSBSO H5A H5A H 0 1 N N N N N N -2.749 4.305 0.544 -2.749 4.305 0.544 H5A azide-A-DSBSO 30
azide-A-DSBSO H5B H5B H 0 1 N N N N N N -1.893 5.619 1.336 -1.893 5.619 1.336 H5B azide-A-DSBSO 31
azide-A-DSBSO H6A H6A H 0 1 N N N N N N -0.629 5.903 -0.952 -0.629 5.903 -0.952 H6A azide-A-DSBSO 32
azide-A-DSBSO H6B H6B H 0 1 N N N N N N -1.339 4.385 -1.383 -1.339 4.385 -1.383 H6B azide-A-DSBSO 33
azide-A-DSBSO H8A H8A H 0 1 N N N N N N 1.943 4.719 1.270 1.943 4.719 1.270 H8A azide-A-DSBSO 34
azide-A-DSBSO H8B H8B H 0 1 N N N N N N 1.339 6.025 0.283 1.339 6.025 0.283 H8B azide-A-DSBSO 35
azide-A-DSBSO H8C H8C H 0 1 N N N N N N 0.513 5.579 1.801 0.513 5.579 1.801 H8C azide-A-DSBSO 36
azide-A-DSBSO H10A H10A H 0 1 N N N N N N -0.379 2.267 -1.616 -0.379 2.267 -1.616 H10A azide-A-DSBSO 37
azide-A-DSBSO H10B H10B H 0 1 N N N N N N 1.263 1.865 -1.933 1.263 1.865 -1.933 H10B azide-A-DSBSO 38
azide-A-DSBSO H12A H12A H 0 1 N N N N N N 2.072 -0.036 1.064 2.072 -0.036 1.064 H12A azide-A-DSBSO 39
azide-A-DSBSO H12B H12B H 0 1 N N N N N N 2.022 -0.144 -0.708 2.022 -0.144 -0.708 H12B azide-A-DSBSO 40
azide-A-DSBSO H14A H14A H 0 1 N N N N N N 4.187 -0.197 -1.298 4.187 -0.197 -1.298 H14A azide-A-DSBSO 41
azide-A-DSBSO H14B H14B H 0 1 N N N N N N 5.451 0.941 -0.893 5.451 0.941 -0.893 H14B azide-A-DSBSO 42
azide-A-DSBSO H15A H15A H 0 1 N N N N N N 4.342 -1.067 1.149 4.342 -1.067 1.149 H15A azide-A-DSBSO 43
azide-A-DSBSO H15B H15B H 0 1 N N N N N N 5.718 0.041 1.361 5.718 0.041 1.361 H15B azide-A-DSBSO 44
azide-A-DSBSO H27A H27A H 0 1 N N N N N N -1.481 0.844 -0.031 -1.481 0.844 -0.031 H27A azide-A-DSBSO 45
azide-A-DSBSO H27B H27B H 0 1 N N N N N N -0.458 -0.251 -0.962 -0.458 -0.251 -0.962 H27B azide-A-DSBSO 46
azide-A-DSBSO H29A H29A H 0 1 N N N N N N -2.603 -1.700 2.254 -2.603 -1.700 2.254 H29A azide-A-DSBSO 47
azide-A-DSBSO H29B H29B H 0 1 N N N N N N -2.896 -0.243 1.335 -2.896 -0.243 1.335 H29B azide-A-DSBSO 48
azide-A-DSBSO H30A H30A H 0 1 N N N N N N -2.386 -3.069 0.214 -2.386 -3.069 0.214 H30A azide-A-DSBSO 49
azide-A-DSBSO H30B H30B H 0 1 N N N N N N -2.671 -1.655 -0.817 -2.671 -1.655 -0.817 H30B azide-A-DSBSO 50
azide-A-DSBSO H42A H42A H 0 1 N N N N N N 1.400 2.480 1.793 1.400 2.480 1.793 H42A azide-A-DSBSO 51
azide-A-DSBSO H42B H42B H 0 1 N N N N N N -0.024 1.603 2.160 -0.024 1.603 2.160 H42B azide-A-DSBSO 52
#
loop_
_chem_comp_bond.comp_id
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_order
_chem_comp_bond.pdbx_aromatic_flag
_chem_comp_bond.pdbx_stereo_config
_chem_comp_bond.pdbx_ordinal
azide-A-DSBSO  N1   N2 DOUB N N 1
azide-A-DSBSO  N2   N3 DOUB N N 2
azide-A-DSBSO  N3   C4 SING N N 3
azide-A-DSBSO  C4   C5 SING N N 4
azide-A-DSBSO  C5   C6 SING N N 5
azide-A-DSBSO  C6   C7 SING N N 6
azide-A-DSBSO  C7   C8 SING N N 7
azide-A-DSBSO  C7   O9 SING N N 8
azide-A-DSBSO  O9  C10 SING N N 9
azide-A-DSBSO C10  C11 SING N N 10
azide-A-DSBSO C11  C12 SING N N 11
azide-A-DSBSO C12  S13 SING N N 12
azide-A-DSBSO S13  C14 SING N N 13
azide-A-DSBSO C14  C15 SING N N 14
azide-A-DSBSO C15  C16 SING N N 15
azide-A-DSBSO C16  O25 DOUB N N 16
azide-A-DSBSO S13  O26 DOUB N N 17
azide-A-DSBSO C11  C27 SING N N 18
azide-A-DSBSO C27  S28 SING N N 19
azide-A-DSBSO S28  C29 SING N N 20
azide-A-DSBSO C29  C30 SING N N 21
azide-A-DSBSO C30  C31 SING N N 22
azide-A-DSBSO C31  O40 DOUB N N 23
azide-A-DSBSO S28  O41 DOUB N N 24
azide-A-DSBSO C11  C42 SING N N 25
azide-A-DSBSO C42  O43 SING N N 26
azide-A-DSBSO O43   C7 SING N N 27
azide-A-DSBSO  C4  H4A SING N N 28
azide-A-DSBSO  C4  H4B SING N N 29
azide-A-DSBSO  C5  H5A SING N N 30
azide-A-DSBSO  C5  H5B SING N N 31
azide-A-DSBSO  C6  H6A SING N N 32
azide-A-DSBSO  C6  H6B SING N N 33
azide-A-DSBSO  C8  H8A SING N N 34
azide-A-DSBSO  C8  H8B SING N N 35
azide-A-DSBSO  C8  H8C SING N N 36
azide-A-DSBSO C10 H10A SING N N 37
azide-A-DSBSO C10 H10B SING N N 38
azide-A-DSBSO C12 H12A SING N N 39
azide-A-DSBSO C12 H12B SING N N 40
azide-A-DSBSO C14 H14A SING N N 41
azide-A-DSBSO C14 H14B SING N N 42
azide-A-DSBSO C15 H15A SING N N 43
azide-A-DSBSO C15 H15B SING N N 44
azide-A-DSBSO C27 H27A SING N N 45
azide-A-DSBSO C27 H27B SING N N 46
azide-A-DSBSO C29 H29A SING N N 47
azide-A-DSBSO C29 H29B SING N N 48
azide-A-DSBSO C30 H30A SING N N 49
azide-A-DSBSO C30 H30B SING N N 50
azide-A-DSBSO C42 H42A SING N N 51
azide-A-DSBSO C42 H42B SING N N 52
#
_pdbx_chem_comp_descriptor.type SMILES_CANONICAL
_pdbx_chem_comp_descriptor.descriptor 'CC1(OCC(CO1)(CS(=O)CCC(=O)ON2C(=O)CCC2=O)CS(=O)CCC(=O)ON3C(=O)CCC3=O)CCCN=[N+]=[N-]'
#
""",

        "bond1": {
            "atom1": {
                "moltype": "protein",
                "atomtypes": [
                    {
                        "restype": "LYS",
                        "atomname": "NZ"
                    },
                    {
                        "restype": "SER",
                        "atomname": "OG"
                    },
                    {
                        "restype": "THR",
                        "atomname": "OG1"
                    },
                    {
                        "restype": "TYR",
                        "atomname": "OH"
                    },
                    {
                        "restype": "NTER",
                        "atomname": "N"
                    }
                ],
            },
            "atom2": {
                "moltype": "ligand",
                "restype": "azide-A-DSBSO",
                "atomname": "C16",
            }
        },
        "bond2": {
            "atom1": {
                "moltype": "protein",
                "atomtypes": [
                    {
                        "restype": "LYS",
                        "atomname": "NZ"
                    },
                    {
                        "restype": "SER",
                        "atomname": "OG"
                    },
                    {
                        "restype": "THR",
                        "atomname": "OG1"
                    },
                    {
                        "restype": "TYR",
                        "atomname": "OH"
                    },
                    {
                        "restype": "NTER",
                        "atomname": "N"
                    }
                ],
            },
            "atom2": {
                "moltype": "ligand",
                "restype": "azide-A-DSBSO",
                "atomname": "C31",
            }
        }
    },  
    "DSSO": {
        "ccdCode": "DSSO",
        "userCCD": """data_DSSO
#
_chem_comp.id DSSO
_chem_comp.name 'Disuccinimidyl sulfoxide, DSSO crosslinker'
_chem_comp.type non-polymer
_chem_comp.formula 'C6 H8 O3 S'
_chem_comp.mon_nstd_parent_comp_id ?
_chem_comp.pdbx_synonyms ?
_chem_comp.formula_weight XXX
#
loop_
_chem_comp_atom.comp_id
_chem_comp_atom.atom_id
_chem_comp_atom.alt_atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.charge
_chem_comp_atom.pdbx_align
_chem_comp_atom.pdbx_aromatic_flag
_chem_comp_atom.pdbx_leaving_atom_flag
_chem_comp_atom.pdbx_stereo_config
_chem_comp_atom.pdbx_backbone_atom_flag
_chem_comp_atom.pdbx_n_terminal_atom_flag
_chem_comp_atom.pdbx_c_terminal_atom_flag
_chem_comp_atom.model_Cartn_x
_chem_comp_atom.model_Cartn_y
_chem_comp_atom.model_Cartn_z
_chem_comp_atom.pdbx_model_Cartn_x_ideal
_chem_comp_atom.pdbx_model_Cartn_y_ideal
_chem_comp_atom.pdbx_model_Cartn_z_ideal
_chem_comp_atom.pdbx_component_atom_id
_chem_comp_atom.pdbx_component_comp_id
_chem_comp_atom.pdbx_ordinal
DSSO O1 O1 O 0 1 N N N N N N 2.688 -0.277 -0.628 2.688 -0.277 -0.628 O1 DSSO 1
DSSO C2 C2 C 0 1 N N N N N N 3.015 0.124 0.457 3.015 0.124 0.457 C2 DSSO 2
DSSO C11 C11 C 0 1 N N N N N N 2.097 0.574 1.557 2.097 0.574 1.557 C11 DSSO 3
DSSO C12 C12 C 0 1 N N N N N N 0.915 1.376 1.023 0.915 1.376 1.023 C12 DSSO 4
DSSO S13 S13 S 0 1 N N N N N N -0.523 0.377 0.591 -0.523 0.377 0.591 S13 DSSO 5
DSSO C14 C14 C 0 1 N N N N N N -1.281 0.295 2.225 -1.281 0.295 2.225 C14 DSSO 6
DSSO C15 C15 C 0 1 N N N N N N -2.558 -0.536 2.183 -2.558 -0.536 2.183 C15 DSSO 7
DSSO C16 C16 C 0 1 N N N N N N -3.548 0.090 1.244 -3.548 0.090 1.244 C16 DSSO 8
DSSO O25 O25 O 0 1 N N N N N N -4.039 1.183 1.346 -4.039 1.183 1.346 O25 DSSO 9
DSSO O26 O26 O 0 1 N N N N N N 0.001 -1.003 0.312 0.001 -1.003 0.312 O26 DSSO 10
DSSO H11A H11A H 0 1 N N N N N N 2.696 1.212 2.211 2.696 1.212 2.211 H11A DSSO 11
DSSO H11B H11B H 0 1 N N N N N N 1.770 -0.289 2.143 1.770 -0.289 2.143 H11B DSSO 12
DSSO H12A H12A H 0 1 N N N N N N 0.600 2.119 1.760 0.600 2.119 1.760 H12A DSSO 13
DSSO H12B H12B H 0 1 N N N N N N 1.188 1.925 0.118 1.188 1.925 0.118 H12B DSSO 14
DSSO H14A H14A H 0 1 N N N N N N -0.567 -0.166 2.911 -0.567 -0.166 2.911 H14A DSSO 15
DSSO H14B H14B H 0 1 N N N N N N -1.483 1.316 2.559 -1.483 1.316 2.559 H14B DSSO 16
DSSO H15A H15A H 0 1 N N N N N N -2.368 -1.573 1.897 -2.368 -1.573 1.897 H15A DSSO 17
DSSO H15B H15B H 0 1 N N N N N N -3.026 -0.565 3.170 -3.026 -0.565 3.170 H15B DSSO 18
#
loop_
_chem_comp_bond.comp_id
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_order
_chem_comp_bond.pdbx_aromatic_flag
_chem_comp_bond.pdbx_stereo_config
_chem_comp_bond.pdbx_ordinal             
DSSO  O1   C2 DOUB N N 1
DSSO  C2  C11 SING N N 2
DSSO C11  C12 SING N N 3
DSSO C12  S13 SING N N 4
DSSO S13  C14 SING N N 5
DSSO C14  C15 SING N N 6
DSSO C15  C16 SING N N 7
DSSO C16  O25 DOUB N N 8
DSSO S13  O26 DOUB N N 9
DSSO C11 H11A SING N N 10
DSSO C11 H11B SING N N 11
DSSO C12 H12A SING N N 12
DSSO C12 H12B SING N N 13
DSSO C14 H14A SING N N 14
DSSO C14 H14B SING N N 15
DSSO C15 H15A SING N N 16
DSSO C15 H15B SING N N 17
#
_pdbx_chem_comp_descriptor.type SMILES_CANONICAL
_pdbx_chem_comp_descriptor.descriptor 'DUPA'
#
""",
        "bond1": {
            "atom1": {
                "moltype": "protein",
                "atomtypes": [
                    {
                        "restype": "LYS",
                        "atomname": "NZ"
                    },
                    {
                        "restype": "SER",
                        "atomname": "OG"
                    },
                    {
                        "restype": "THR",
                        "atomname": "OG1"
                    },
                    {
                        "restype": "TYR",
                        "atomname": "OH"
                    },
                    {
                        "restype": "NTER",
                        "atomname": "N"
                    }
                ],
            },
            "atom2": {
                "moltype": "ligand",
                "restype": "DSSO",
                "atomname": "C2",
            }
        },
        "bond2": {
            "atom1": {
                "moltype": "protein",
                "atomtypes": [
                    {
                        "restype": "LYS",
                        "atomname": "NZ"
                    },
                    {
                        "restype": "SER",
                        "atomname": "OG"
                    },
                    {
                        "restype": "THR",
                        "atomname": "OG1"
                    },
                    {
                        "restype": "TYR",
                        "atomname": "OH"
                    },
                    {
                        "restype": "NTER",
                        "atomname": "N"
                    }
                ],
            },
            "atom2": {
                "moltype": "ligand",
                "restype": "DSSO",
                "atomname": "C16",
            }
        },
    },
    "DSS": {
        "ccdCode": "DSS",
        "userCCD": """data_DSS
#
_chem_comp.id DSS
_chem_comp.name 'Disuccinimidyl suberate crosslink'
_chem_comp.type non-polymer
_chem_comp.formula 'C8 H12 O2'
_chem_comp.mon_nstd_parent_comp_id ?
_chem_comp.pdbx_synonyms ?
_chem_comp.formula_weight XXX
#
loop_
_chem_comp_atom.comp_id
_chem_comp_atom.atom_id
_chem_comp_atom.alt_atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.charge
_chem_comp_atom.pdbx_align
_chem_comp_atom.pdbx_aromatic_flag
_chem_comp_atom.pdbx_leaving_atom_flag
_chem_comp_atom.pdbx_stereo_config
_chem_comp_atom.pdbx_backbone_atom_flag
_chem_comp_atom.pdbx_n_terminal_atom_flag
_chem_comp_atom.pdbx_c_terminal_atom_flag
_chem_comp_atom.model_Cartn_x
_chem_comp_atom.model_Cartn_y
_chem_comp_atom.model_Cartn_z
_chem_comp_atom.pdbx_model_Cartn_x_ideal
_chem_comp_atom.pdbx_model_Cartn_y_ideal
_chem_comp_atom.pdbx_model_Cartn_z_ideal
_chem_comp_atom.pdbx_component_atom_id
_chem_comp_atom.pdbx_component_comp_id
_chem_comp_atom.pdbx_ordinal
DSS C9 C9 C 0 1 N N N N N N -3.367 0.031 0.314 -3.367 0.031 0.314 C9 DSS 1
DSS O10 O10 O 0 1 N N N N N N -3.617 1.206 0.300 -3.617 1.206 0.300 O10 DSS 2
DSS C11 C11 C 0 1 N N N N N N -2.284 -0.661 -0.461 -2.284 -0.661 -0.461 C11 DSS 3
DSS C12 C12 C 0 1 N N N N N N -1.591 0.258 -1.461 -1.591 0.258 -1.461 C12 DSS 4
DSS C13 C13 C 0 1 N N N N N N -0.556 -0.453 -2.323 -0.556 -0.453 -2.323 C13 DSS 5
DSS C14 C14 C 0 1 N N N N N N 0.784 -0.698 -1.643 0.784 -0.698 -1.643 C14 DSS 6
DSS C15 C15 C 0 1 N N N N N N 1.827 -1.310 -2.568 1.827 -1.310 -2.568 C15 DSS 7
DSS C16 C16 C 0 1 N N N N N N 3.119 -1.685 -1.853 3.119 -1.685 -1.853 C16 DSS 8
DSS C17 C17 C 0 1 N N N N N N 3.906 -0.497 -1.378 3.906 -0.497 -1.378 C17 DSS 9
DSS O18 O18 O 0 1 N N N N N N 4.059 0.540 -1.965 4.059 0.540 -1.965 O18 DSS 10
DSS H11A H11A H 0 1 N N N N N N -1.566 -1.051 0.265 -1.566 -1.051 0.265 H11A DSS 11
DSS H11B H11B H 0 1 N N N N N N -2.737 -1.499 -0.997 -2.737 -1.499 -0.997 H11B DSS 12
DSS H12A H12A H 0 1 N N N N N N -2.340 0.701 -2.122 -2.340 0.701 -2.122 H12A DSS 13
DSS H12B H12B H 0 1 N N N N N N -1.120 1.091 -0.931 -1.120 1.091 -0.931 H12B DSS 14
DSS H13A H13A H 0 1 N N N N N N -0.967 -1.405 -2.670 -0.967 -1.405 -2.670 H13A DSS 15
DSS H13B H13B H 0 1 N N N N N N -0.392 0.156 -3.216 -0.392 0.156 -3.216 H13B DSS 16
DSS H14A H14A H 0 1 N N N N N N 0.652 -1.367 -0.788 0.652 -1.367 -0.788 H14A DSS 17
DSS H14B H14B H 0 1 N N N N N N 1.147 0.253 -1.245 1.147 0.253 -1.245 H14B DSS 18
DSS H15A H15A H 0 1 N N N N N N 1.409 -2.215 -3.017 1.409 -2.215 -3.017 H15A DSS 19
DSS H15B H15B H 0 1 N N N N N N 2.041 -0.621 -3.389 2.041 -0.621 -3.389 H15B DSS 20
DSS H16A H16A H 0 1 N N N N N N 3.759 -2.237 -2.546 3.759 -2.237 -2.546 H16A DSS 21
DSS H16B H16B H 0 1 N N N N N N 2.897 -2.328 -0.998 2.897 -2.328 -0.998 H16B DSS 22
#
loop_
_chem_comp_bond.comp_id
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_order
_chem_comp_bond.pdbx_aromatic_flag
_chem_comp_bond.pdbx_stereo_config
_chem_comp_bond.pdbx_ordinal  
DSS  C9  O10 DOUB N N 1
DSS  C9  C11 SING N N 2
DSS C11  C12 SING N N 3
DSS C12  C13 SING N N 4
DSS C13  C14 SING N N 5
DSS C14  C15 SING N N 6
DSS C15  C16 SING N N 7
DSS C16  C17 SING N N 8
DSS C17  O18 DOUB N N 9
DSS C11 H11A SING N N 10
DSS C11 H11B SING N N 11
DSS C12 H12A SING N N 12
DSS C12 H12B SING N N 13
DSS C13 H13A SING N N 14
DSS C13 H13B SING N N 15
DSS C14 H14A SING N N 16
DSS C14 H14B SING N N 17
DSS C15 H15A SING N N 18
DSS C15 H15B SING N N 19
DSS C16 H16A SING N N 20
DSS C16 H16B SING N N 21
#
_pdbx_chem_comp_descriptor.type SMILES_CANONICAL
_pdbx_chem_comp_descriptor.descriptor 'C1CC(=O)N(C1=O)OC(=O)CCCCCCC(=O)ON2C(=O)CCC2=O'
#
""",
        "bond1": {
            "atom1": {
                "moltype": "protein",
                "atomtypes": [
                    {
                        "restype": "LYS",
                        "atomname": "NZ"
                    },
                    {
                        "restype": "SER",
                        "atomname": "OG"
                    },
                    {
                        "restype": "THR",
                        "atomname": "OG1"
                    },
                    {
                        "restype": "TYR",
                        "atomname": "OH"
                    },
                    {
                        "restype": "NTER",
                        "atomname": "N"
                    }
                ],
            },
            "atom2": {
                "moltype": "ligand",
                "restype": "DSSO",
                "atomname": "C9",
            }
        },
        "bond2": {
            "atom1": {
                "moltype": "protein",
                "atomtypes": [
                    {
                        "restype": "LYS",
                        "atomname": "NZ"
                    },
                    {
                        "restype": "SER",
                        "atomname": "OG"
                    },
                    {
                        "restype": "THR",
                        "atomname": "OG1"
                    },
                    {
                        "restype": "TYR",
                        "atomname": "OH"
                    },
                    {
                        "restype": "NTER",
                        "atomname": "N"
                    }
                ],
            },
            "atom2": {
                "moltype": "ligand",
                "restype": "DSSO",
                "atomname": "C17",
            }
        },
    },
    "BS3": {
        "ccdCode": "BS3",
        "userCCD": """data_BS3
#
_chem_comp.id BS3
_chem_comp.name 'Bis(sulfosuccinimidyl)suberate, BS3 crosslinker'
_chem_comp.type non-polymer
_chem_comp.formula 'C8 H12 O2'
_chem_comp.mon_nstd_parent_comp_id ?
_chem_comp.pdbx_synonyms ?
_chem_comp.formula_weight XXX
#
loop_
_chem_comp_atom.comp_id
_chem_comp_atom.atom_id
_chem_comp_atom.alt_atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.charge
_chem_comp_atom.pdbx_align
_chem_comp_atom.pdbx_aromatic_flag
_chem_comp_atom.pdbx_leaving_atom_flag
_chem_comp_atom.pdbx_stereo_config
_chem_comp_atom.pdbx_backbone_atom_flag
_chem_comp_atom.pdbx_n_terminal_atom_flag
_chem_comp_atom.pdbx_c_terminal_atom_flag
_chem_comp_atom.model_Cartn_x
_chem_comp_atom.model_Cartn_y
_chem_comp_atom.model_Cartn_z
_chem_comp_atom.pdbx_model_Cartn_x_ideal
_chem_comp_atom.pdbx_model_Cartn_y_ideal
_chem_comp_atom.pdbx_model_Cartn_z_ideal
_chem_comp_atom.pdbx_component_atom_id
_chem_comp_atom.pdbx_component_comp_id
_chem_comp_atom.pdbx_ordinal
BS3 C9 C9 C 0 1 N N N N N N -3.367 0.031 0.314 -3.367 0.031 0.314 C9 BS3 1
BS3 O10 O10 O 0 1 N N N N N N -3.617 1.206 0.300 -3.617 1.206 0.300 O10 BS3 2
BS3 C11 C11 C 0 1 N N N N N N -2.284 -0.661 -0.461 -2.284 -0.661 -0.461 C11 BS3 3
BS3 C12 C12 C 0 1 N N N N N N -1.591 0.258 -1.461 -1.591 0.258 -1.461 C12 BS3 4
BS3 C13 C13 C 0 1 N N N N N N -0.556 -0.453 -2.323 -0.556 -0.453 -2.323 C13 BS3 5
BS3 C14 C14 C 0 1 N N N N N N 0.784 -0.698 -1.643 0.784 -0.698 -1.643 C14 BS3 6
BS3 C15 C15 C 0 1 N N N N N N 1.827 -1.310 -2.568 1.827 -1.310 -2.568 C15 BS3 7
BS3 C16 C16 C 0 1 N N N N N N 3.119 -1.685 -1.853 3.119 -1.685 -1.853 C16 BS3 8
BS3 C17 C17 C 0 1 N N N N N N 3.906 -0.497 -1.378 3.906 -0.497 -1.378 C17 BS3 9
BS3 O18 O18 O 0 1 N N N N N N 4.059 0.540 -1.965 4.059 0.540 -1.965 O18 BS3 10
BS3 H11A H11A H 0 1 N N N N N N -1.566 -1.051 0.265 -1.566 -1.051 0.265 H11A BS3 11
BS3 H11B H11B H 0 1 N N N N N N -2.737 -1.499 -0.997 -2.737 -1.499 -0.997 H11B BS3 12
BS3 H12A H12A H 0 1 N N N N N N -2.340 0.701 -2.122 -2.340 0.701 -2.122 H12A BS3 13
BS3 H12B H12B H 0 1 N N N N N N -1.120 1.091 -0.931 -1.120 1.091 -0.931 H12B BS3 14
BS3 H13A H13A H 0 1 N N N N N N -0.967 -1.405 -2.670 -0.967 -1.405 -2.670 H13A BS3 15
BS3 H13B H13B H 0 1 N N N N N N -0.392 0.156 -3.216 -0.392 0.156 -3.216 H13B BS3 16
BS3 H14A H14A H 0 1 N N N N N N 0.652 -1.367 -0.788 0.652 -1.367 -0.788 H14A BS3 17
BS3 H14B H14B H 0 1 N N N N N N 1.147 0.253 -1.245 1.147 0.253 -1.245 H14B BS3 18
BS3 H15A H15A H 0 1 N N N N N N 1.409 -2.215 -3.017 1.409 -2.215 -3.017 H15A BS3 19
BS3 H15B H15B H 0 1 N N N N N N 2.041 -0.621 -3.389 2.041 -0.621 -3.389 H15B BS3 20
BS3 H16A H16A H 0 1 N N N N N N 3.759 -2.237 -2.546 3.759 -2.237 -2.546 H16A BS3 21
BS3 H16B H16B H 0 1 N N N N N N 2.897 -2.328 -0.998 2.897 -2.328 -0.998 H16B BS3 22
#
loop_
_chem_comp_bond.comp_id
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_order
_chem_comp_bond.pdbx_aromatic_flag
_chem_comp_bond.pdbx_stereo_config
_chem_comp_bond.pdbx_ordinal  
BS3  C9  O10 DOUB N N 1
BS3  C9  C11 SING N N 2
BS3 C11  C12 SING N N 3
BS3 C12  C13 SING N N 4
BS3 C13  C14 SING N N 5
BS3 C14  C15 SING N N 6
BS3 C15  C16 SING N N 7
BS3 C16  C17 SING N N 8
BS3 C17  O18 DOUB N N 9
BS3 C11 H11A SING N N 10
BS3 C11 H11B SING N N 11
BS3 C12 H12A SING N N 12
BS3 C12 H12B SING N N 13
BS3 C13 H13A SING N N 14
BS3 C13 H13B SING N N 15
BS3 C14 H14A SING N N 16
BS3 C14 H14B SING N N 17
BS3 C15 H15A SING N N 18
BS3 C15 H15B SING N N 19
BS3 C16 H16A SING N N 20
BS3 C16 H16B SING N N 21
#
_pdbx_chem_comp_descriptor.type SMILES_CANONICAL
_pdbx_chem_comp_descriptor.descriptor 'C1CC(=O)N(C1=O)OC(=O)CCCCCCC(=O)ON2C(=O)CCC2=O'
#
""",
        "bond1": {
            "atom1": {
                "moltype": "protein",
                "atomtypes": [
                    {
                        "restype": "LYS",
                        "atomname": "NZ"
                    },
                    {
                        "restype": "SER",
                        "atomname": "OG"
                    },
                    {
                        "restype": "THR",
                        "atomname": "OG1"
                    },
                    {
                        "restype": "TYR",
                        "atomname": "OH"
                    },
                    {
                        "restype": "NTER",
                        "atomname": "N"
                    }
                ],
            },
            "atom2": {
                "moltype": "ligand",
                "restype": "BS3",
                "atomname": "C9",
            }
        },
        "bond2": {
            "atom1": {
                "moltype": "protein",
                "atomtypes": [
                    {
                        "restype": "LYS",
                        "atomname": "NZ"
                    },
                    {
                        "restype": "SER",
                        "atomname": "OG"
                    },
                    {
                        "restype": "THR",
                        "atomname": "OG1"
                    },
                    {
                        "restype": "TYR",
                        "atomname": "OH"
                    },
                    {
                        "restype": "NTER",
                        "atomname": "N"
                    }
                ],
            },
            "atom2": {
                "moltype": "ligand",
                "restype": "BS3",
                "atomname": "C17",
            }
        }
    },
    "DSG": {
        "ccdCode": "DSG",
        "userCCD": """data_DSG
#
_chem_comp.id DSG
_chem_comp.name 'Disuccinimidyl sulfoxide, DSG Crosslinker'
_chem_comp.type non-polymer
_chem_comp.formula 'C5 H6 O2'
_chem_comp.mon_nstd_parent_comp_id ?
_chem_comp.pdbx_synonyms ?
_chem_comp.formula_weight XXX
#
loop_
_chem_comp_atom.comp_id
_chem_comp_atom.atom_id
_chem_comp_atom.alt_atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.charge
_chem_comp_atom.pdbx_align
_chem_comp_atom.pdbx_aromatic_flag
_chem_comp_atom.pdbx_leaving_atom_flag
_chem_comp_atom.pdbx_stereo_config
_chem_comp_atom.pdbx_backbone_atom_flag
_chem_comp_atom.pdbx_n_terminal_atom_flag
_chem_comp_atom.pdbx_c_terminal_atom_flag
_chem_comp_atom.model_Cartn_x
_chem_comp_atom.model_Cartn_y
_chem_comp_atom.model_Cartn_z
_chem_comp_atom.pdbx_model_Cartn_x_ideal
_chem_comp_atom.pdbx_model_Cartn_y_ideal
_chem_comp_atom.pdbx_model_Cartn_z_ideal
_chem_comp_atom.pdbx_component_atom_id
_chem_comp_atom.pdbx_component_comp_id
_chem_comp_atom.pdbx_ordinal
DSG C9 C9 C 0 1 N N N N N N -2.603 -0.972 0.951 -2.603 -0.972 0.951 C9 DSG 1
DSG O10 O10 O 0 1 N N N N N N -3.196 -0.697 1.960 -3.196 -0.697 1.960 O10 DSG 2
DSG C11 C11 C 0 1 N N N N N N -1.244 -1.601 0.857 -1.244 -1.601 0.857 C11 DSG 3
DSG C12 C12 C 0 1 N N N N N N -0.149 -0.599 0.516 -0.149 -0.599 0.516 C12 DSG 4
DSG C13 C13 C 0 1 N N N N N N 1.220 -1.259 0.428 1.220 -1.259 0.428 C13 DSG 5
DSG C14 C14 C 0 1 N N N N N N 2.317 -0.291 0.093 2.317 -0.291 0.093 C14 DSG 6
DSG O15 O15 O 0 1 N N N N N N 2.187 0.875 -0.168 2.187 0.875 -0.168 O15 DSG 7
DSG H11A H11A H 0 1 N N N N N N -1.035 -2.078 1.818 -1.035 -2.078 1.818 H11A DSG 8
DSG H11B H11B H 0 1 N N N N N N -1.284 -2.406 0.119 -1.284 -2.406 0.119 H11B DSG 9
DSG H12A H12A H 0 1 N N N N N N -0.382 -0.106 -0.432 -0.382 -0.106 -0.432 H12A DSG 10
DSG H12B H12B H 0 1 N N N N N N -0.129 0.194 1.268 -0.129 0.194 1.268 H12B DSG 11
DSG H13A H13A H 0 1 N N N N N N 1.204 -2.029 -0.347 1.204 -2.029 -0.347 H13A DSG 12
DSG H13B H13B H 0 1 N N N N N N 1.463 -1.726 1.386 1.463 -1.726 1.386 H13B DSG 13
#
loop_
_chem_comp_bond.comp_id
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_order
_chem_comp_bond.pdbx_aromatic_flag
_chem_comp_bond.pdbx_stereo_config
_chem_comp_bond.pdbx_ordinal 
DSG  C9  O10 DOUB N N 1
DSG  C9  C11 SING N N 2
DSG C11  C12 SING N N 3
DSG C12  C13 SING N N 4
DSG C13  C14 SING N N 5
DSG C14  O15 DOUB N N 6
DSG C11 H11A SING N N 7
DSG C11 H11B SING N N 8
DSG C12 H12A SING N N 9
DSG C12 H12B SING N N 10
DSG C13 H13A SING N N 11
DSG C13 H13B SING N N 12
#
_pdbx_chem_comp_descriptor.type SMILES_CANONICAL
_pdbx_chem_comp_descriptor.descriptor 'C1CC(=O)N(C1=O)OC(=O)CCCC(=O)ON2C(=O)CCC2=O'
#
""",
        "bond1": {
            "atom1": {
                "moltype": "protein",
                "atomtypes": [
                    {
                        "restype": "LYS",
                        "atomname": "NZ"
                    },
                    {
                        "restype": "SER",
                        "atomname": "OG"
                    },
                    {
                        "restype": "THR",
                        "atomname": "OG1"
                    },
                    {
                        "restype": "TYR",
                        "atomname": "OH"
                    },
                    {
                        "restype": "NTER",
                        "atomname": "N"
                    }
                ],
            },
            "atom2": {
                "moltype": "ligand",
                "restype": "DSG",
                "atomname": "C9",
            }
        },
        "bond2": {
            "atom1": {
                "moltype": "protein",
                "atomtypes": [
                    {
                        "restype": "LYS",
                        "atomname": "NZ"
                    },
                    {
                        "restype": "SER",
                        "atomname": "OG"
                    },
                    {
                        "restype": "THR",
                        "atomname": "OG1"
                    },
                    {
                        "restype": "TYR",
                        "atomname": "OH"
                    },
                    {
                        "restype": "NTER",
                        "atomname": "N"
                    }
                ],
            },
            "atom2": {
                "moltype": "ligand",
                "restype": "DSG",
                "atomname": "C14",
            }
        }
    },
    "BS2G": {
        "ccdCode": "BS2G",
        "userCCD": "data_BS2G\n#\n_chem_comp.formula                 ?\n_chem_comp.formula_weight          100.12\n_chem_comp.id                      BS2G\n_chem_comp.mon_nstd_parent_comp_id ?\n_chem_comp.name                    ?\n_chem_comp.pdbx_synonyms           ?\n_chem_comp.type                    non-polymer\n#\nloop_\n_chem_comp_atom.atom_id\n_chem_comp_atom.charge\n_chem_comp_atom.comp_id\n_chem_comp_atom.pdbx_leaving_atom_flag\n_chem_comp_atom.pdbx_model_Cartn_x_ideal\n_chem_comp_atom.pdbx_model_Cartn_y_ideal\n_chem_comp_atom.pdbx_model_Cartn_z_ideal\n_chem_comp_atom.type_symbol\nO1 0 BS2G N 2.765  -0.850 1.717  O \nC1 0 BS2G N 2.172  -0.628 0.651  C \nC2 0 BS2G N 0.793  -0.058 0.640  C \nC3 0 BS2G N 0.268  0.127  -0.760 C \nC4 0 BS2G N -1.148 0.714  -0.708 C \nC5 0 BS2G N -2.069 -0.194 0.031  C \nO2 0 BS2G N -2.087 -1.250 0.613  O \nH1 0 BS2G N 2.850  -0.927 -0.162 H \nH2 0 BS2G N 0.125  -0.754 1.180  H \nH3 0 BS2G N 0.799  0.890  1.204  H \nH4 0 BS2G N 0.905  0.884  -1.259 H \nH5 0 BS2G N 0.241  -0.846 -1.279 H \nH6 0 BS2G N -1.517 0.934  -1.715 H \nH7 0 BS2G N -1.052 1.698  -0.195 H \nH8 0 BS2G N -3.045 0.262  0.040  H \n#\nloop_\n_chem_comp_bond.atom_id_1\n_chem_comp_bond.atom_id_2\n_chem_comp_bond.comp_id\n_chem_comp_bond.pdbx_aromatic_flag\n_chem_comp_bond.pdbx_stereo_config\n_chem_comp_bond.value_order\nO1 C1 BS2G N N DOUB \nC1 C2 BS2G N N SING \nC2 C3 BS2G N N SING \nC3 C4 BS2G N N SING \nC4 C5 BS2G N N SING \nC5 O2 BS2G N N DOUB \nC1 H1 BS2G N N SING \nC2 H2 BS2G N N SING \nC2 H3 BS2G N N SING \nC3 H4 BS2G N N SING \nC3 H5 BS2G N N SING \nC4 H6 BS2G N N SING \nC4 H7 BS2G N N SING \nC5 H8 BS2G N N SING \n#\n_pdbx_chem_comp_descriptor.descriptor ?\n_pdbx_chem_comp_descriptor.type       ?\n#\n",
        "bond1": {
            "atom1": {
                "moltype": "protein",
                "atomtypes": [
                    {
                        "restype": "LYS",
                        "atomname": "NZ"
                    },
                    {
                        "restype": "SER",
                        "atomname": "OG"
                    },
                    {
                        "restype": "THR",
                        "atomname": "OG1"
                    },
                    {
                        "restype": "TYR",
                        "atomname": "OH"
                    },
                    {
                        "restype": "NTER",
                        "atomname": "N"
                    }
                ]
            },
            "atom2": {
                "moltype": "ligand",
                "restype": "BS2G",
                "atomname": "C1"
            }
        },
        "bond2": {
            "atom1": {
                "moltype": "protein",
                "atomtypes": [
                    {
                        "restype": "LYS",
                        "atomname": "NZ"
                    },
                    {
                        "restype": "SER",
                        "atomname": "OG"
                    },
                    {
                        "restype": "THR",
                        "atomname": "OG1"
                    },
                    {
                        "restype": "TYR",
                        "atomname": "OH"
                    },
                    {
                        "restype": "NTER",
                        "atomname": "N"
                    }
                ]
            },
            "atom2": {
                "moltype": "ligand",
                "restype": "BS2G",
                "atomname": "C5"
            }
        }
    },
    "DSBU": {
        "ccdCode": "DSBU",
        "userCCD": "data_DSBU\n#\n_chem_comp.formula                 ?\n_chem_comp.formula_weight          200.24\n_chem_comp.id                      DSBU\n_chem_comp.mon_nstd_parent_comp_id ?\n_chem_comp.name                    ?\n_chem_comp.pdbx_synonyms           ?\n_chem_comp.type                    non-polymer\n#\nloop_\n_chem_comp_atom.atom_id\n_chem_comp_atom.charge\n_chem_comp_atom.comp_id\n_chem_comp_atom.pdbx_leaving_atom_flag\n_chem_comp_atom.pdbx_model_Cartn_x_ideal\n_chem_comp_atom.pdbx_model_Cartn_y_ideal\n_chem_comp_atom.pdbx_model_Cartn_z_ideal\n_chem_comp_atom.type_symbol\nO1  0 DSBU N -4.331 -1.874 0.233  O \nC1  0 DSBU N -4.916 -0.846 0.483  C \nC2  0 DSBU N -4.724 0.608  0.263  C \nC3  0 DSBU N -3.443 0.899  -0.478 C \nC4  0 DSBU N -2.220 0.410  0.258  C \nN1  0 DSBU N -1.057 0.753  -0.564 N \nC5  0 DSBU N 0.279  0.443  -0.181 C \nO2  0 DSBU N 0.406  -0.148 0.913  O \nN2  0 DSBU N 1.380  0.783  -0.991 N \nC6  0 DSBU N 2.732  0.474  -0.612 C \nC7  0 DSBU N 2.980  -1.006 -0.430 C \nC8  0 DSBU N 4.431  -1.281 -0.028 C \nC9  0 DSBU N 4.784  -0.617 1.244  C \nO3  0 DSBU N 5.017  0.567  1.287  O \nH1  0 DSBU N -5.840 -1.099 1.021  H \nH2  0 DSBU N -5.556 0.955  -0.362 H \nH3  0 DSBU N -4.760 1.190  1.204  H \nH4  0 DSBU N -3.361 2.010  -0.563 H \nH5  0 DSBU N -3.461 0.501  -1.493 H \nH6  0 DSBU N -2.254 -0.689 0.410  H \nH7  0 DSBU N -2.168 0.881  1.271  H \nH8  0 DSBU N -1.152 1.255  -1.497 H \nH9  0 DSBU N 1.199  1.284  -1.907 H \nH10 0 DSBU N 2.949  1.048  0.308  H \nH11 0 DSBU N 3.445  0.820  -1.376 H \nH12 0 DSBU N 2.373  -1.323 0.454  H \nH13 0 DSBU N 2.794  -1.587 -1.331 H \nH14 0 DSBU N 4.527  -2.385 0.038  H \nH15 0 DSBU N 5.082  -0.915 -0.869 H \nH16 0 DSBU N 4.864  -1.112 2.222  H \n#\nloop_\n_chem_comp_bond.atom_id_1\n_chem_comp_bond.atom_id_2\n_chem_comp_bond.comp_id\n_chem_comp_bond.pdbx_aromatic_flag\n_chem_comp_bond.pdbx_stereo_config\n_chem_comp_bond.value_order\nO1 C1  DSBU N N DOUB \nC1 C2  DSBU N N SING \nC2 C3  DSBU N N SING \nC3 C4  DSBU N N SING \nC4 N1  DSBU N N SING \nN1 C5  DSBU N N SING \nC5 O2  DSBU N N DOUB \nC5 N2  DSBU N N SING \nN2 C6  DSBU N N SING \nC6 C7  DSBU N N SING \nC7 C8  DSBU N N SING \nC8 C9  DSBU N N SING \nC9 O3  DSBU N N DOUB \nC1 H1  DSBU N N SING \nC2 H2  DSBU N N SING \nC2 H3  DSBU N N SING \nC3 H4  DSBU N N SING \nC3 H5  DSBU N N SING \nC4 H6  DSBU N N SING \nC4 H7  DSBU N N SING \nN1 H8  DSBU N N SING \nN2 H9  DSBU N N SING \nC6 H10 DSBU N N SING \nC6 H11 DSBU N N SING \nC7 H12 DSBU N N SING \nC7 H13 DSBU N N SING \nC8 H14 DSBU N N SING \nC8 H15 DSBU N N SING \nC9 H16 DSBU N N SING \n#\n_pdbx_chem_comp_descriptor.descriptor ?\n_pdbx_chem_comp_descriptor.type       ?\n#\n",
        "bond1": {
            "atom1": {
                "moltype": "protein",
                "atomtypes": [
                    {
                        "restype": "LYS",
                        "atomname": "NZ"
                    },
                    {
                        "restype": "SER",
                        "atomname": "OG"
                    },
                    {
                        "restype": "THR",
                        "atomname": "OG1"
                    },
                    {
                        "restype": "TYR",
                        "atomname": "OH"
                    },
                    {
                        "restype": "NTER",
                        "atomname": "N"
                    }
                ]
            },
            "atom2": {
                "moltype": "ligand",
                "restype": "DSBU",
                "atomname": "C1"
            }
        },
        "bond2": {
            "atom1": {
                "moltype": "protein",
                "atomtypes": [
                    {
                        "restype": "LYS",
                        "atomname": "NZ"
                    },
                    {
                        "restype": "SER",
                        "atomname": "OG"
                    },
                    {
                        "restype": "THR",
                        "atomname": "OG1"
                    },
                    {
                        "restype": "TYR",
                        "atomname": "OH"
                    },
                    {
                        "restype": "NTER",
                        "atomname": "N"
                    }
                ]
            },
            "atom2": {
                "moltype": "ligand",
                "restype": "DSBU",
                "atomname": "C9"
            }
        }
    },
    "PHOX": {
        "ccdCode": "PHOX",
        "userCCD": "data_PHOX\n#\n_chem_comp.formula                 ?\n_chem_comp.formula_weight          214.11\n_chem_comp.id                      PHOX\n_chem_comp.mon_nstd_parent_comp_id ?\n_chem_comp.name                    ?\n_chem_comp.pdbx_synonyms           ?\n_chem_comp.type                    non-polymer\n#\nloop_\n_chem_comp_atom.atom_id\n_chem_comp_atom.charge\n_chem_comp_atom.comp_id\n_chem_comp_atom.pdbx_leaving_atom_flag\n_chem_comp_atom.pdbx_model_Cartn_x_ideal\n_chem_comp_atom.pdbx_model_Cartn_y_ideal\n_chem_comp_atom.pdbx_model_Cartn_z_ideal\n_chem_comp_atom.type_symbol\nO1 0 PHOX N -3.743 -0.665 1.180  O \nC1 0 PHOX N -2.616 -1.217 1.131  C \nC2 0 PHOX N -1.467 -0.533 0.601  C \nC3 0 PHOX N -1.703 0.748  0.163  C \nC4 0 PHOX N -0.637 1.478  -0.366 C \nC5 0 PHOX N -0.855 2.839  -0.839 C \nO2 0 PHOX N -2.018 3.316  -0.753 O \nC6 0 PHOX N 0.605  0.879  -0.423 C \nC7 0 PHOX N 0.803  -0.398 0.022  C \nP1 0 PHOX N 2.417  -1.204 -0.042 P \nO3 0 PHOX N 2.570  -1.989 1.247  O \nO4 0 PHOX N 2.452  -2.273 -1.351 O \nO5 0 PHOX N 3.666  -0.077 -0.103 O \nC8 0 PHOX N -0.237 -1.160 0.557  C \nH1 0 PHOX N -2.787 -2.198 1.549  H \nH2 0 PHOX N -2.669 1.228  0.202  H \nH3 0 PHOX N -0.008 3.378  -1.243 H \nH4 0 PHOX N 1.430  1.444  -0.832 H \nH5 0 PHOX N 1.599  -2.214 -1.877 H \nH6 0 PHOX N 3.328  0.789  0.256  H \nH7 0 PHOX N -0.129 -2.172 0.922  H \n#\nloop_\n_chem_comp_bond.atom_id_1\n_chem_comp_bond.atom_id_2\n_chem_comp_bond.comp_id\n_chem_comp_bond.pdbx_aromatic_flag\n_chem_comp_bond.pdbx_stereo_config\n_chem_comp_bond.value_order\nO1 C1 PHOX N N DOUB \nC1 C2 PHOX N N SING \nC2 C3 PHOX Y N DOUB \nC3 C4 PHOX Y N SING \nC4 C5 PHOX N N SING \nC5 O2 PHOX N N DOUB \nC4 C6 PHOX Y N DOUB \nC6 C7 PHOX Y N SING \nC7 P1 PHOX N N SING \nP1 O3 PHOX N N DOUB \nP1 O4 PHOX N N SING \nP1 O5 PHOX N N SING \nC7 C8 PHOX Y N DOUB \nC8 C2 PHOX Y N SING \nC1 H1 PHOX N N SING \nC3 H2 PHOX N N SING \nC5 H3 PHOX N N SING \nC6 H4 PHOX N N SING \nO4 H5 PHOX N N SING \nO5 H6 PHOX N N SING \nC8 H7 PHOX N N SING \n#\n_pdbx_chem_comp_descriptor.descriptor ?\n_pdbx_chem_comp_descriptor.type       ?\n#\n",
        "bond1": {
            "atom1": {
                "moltype": "protein",
                "atomtypes": [
                    {
                        "restype": "LYS",
                        "atomname": "NZ"
                    },
                    {
                        "restype": "SER",
                        "atomname": "OG"
                    },
                    {
                        "restype": "THR",
                        "atomname": "OG1"
                    },
                    {
                        "restype": "TYR",
                        "atomname": "OH"
                    },
                    {
                        "restype": "NTER",
                        "atomname": "N"
                    }
                ]
            },
            "atom2": {
                "moltype": "ligand",
                "restype": "PHOX",
                "atomname": "C1"
            }
        },
        "bond2": {
            "atom1": {
                "moltype": "protein",
                "atomtypes": [
                    {
                        "restype": "LYS",
                        "atomname": "NZ"
                    },
                    {
                        "restype": "SER",
                        "atomname": "OG"
                    },
                    {
                        "restype": "THR",
                        "atomname": "OG1"
                    },
                    {
                        "restype": "TYR",
                        "atomname": "OH"
                    },
                    {
                        "restype": "NTER",
                        "atomname": "N"
                    }
                ]
            },
            "atom2": {
                "moltype": "ligand",
                "restype": "PHOX",
                "atomname": "C5"
            }
        }
    },
    "BSPEG5": {
        "ccdCode": "BSPEG5",
        "userCCD": "data_BSPEG5\n#\n_chem_comp.formula                 ?\n_chem_comp.formula_weight          306.35\n_chem_comp.id                      BSPEG5\n_chem_comp.mon_nstd_parent_comp_id ?\n_chem_comp.name                    ?\n_chem_comp.pdbx_synonyms           ?\n_chem_comp.type                    non-polymer\n#\nloop_\n_chem_comp_atom.atom_id\n_chem_comp_atom.charge\n_chem_comp_atom.comp_id\n_chem_comp_atom.pdbx_leaving_atom_flag\n_chem_comp_atom.pdbx_model_Cartn_x_ideal\n_chem_comp_atom.pdbx_model_Cartn_y_ideal\n_chem_comp_atom.pdbx_model_Cartn_z_ideal\n_chem_comp_atom.type_symbol\nO1  0 BSPEG5 N -7.534 0.670  2.666  O \nC1  0 BSPEG5 N -8.110 1.318  1.789  C \nC2  0 BSPEG5 N -7.254 1.519  0.590  C \nC3  0 BSPEG5 N -5.936 0.835  0.879  C \nO2  0 BSPEG5 N -5.047 0.959  -0.184 O \nC4  0 BSPEG5 N -3.865 0.277  0.221  C \nC5  0 BSPEG5 N -2.823 0.337  -0.842 C \nO3  0 BSPEG5 N -1.671 -0.309 -0.464 O \nC6  0 BSPEG5 N -1.821 -1.644 -0.191 C \nC7  0 BSPEG5 N -0.511 -2.280 0.183  C \nO4  0 BSPEG5 N 0.415  -2.207 -0.806 O \nC8  0 BSPEG5 N 0.795  -0.979 -1.229 C \nC9  0 BSPEG5 N 1.899  -1.201 -2.271 C \nO5  0 BSPEG5 N 2.984  -1.859 -1.773 O \nC10 0 BSPEG5 N 3.659  -1.283 -0.735 C \nC11 0 BSPEG5 N 4.226  0.061  -1.025 C \nO6  0 BSPEG5 N 4.905  0.631  0.032  O \nC12 0 BSPEG5 N 5.984  -0.101 0.477  C \nC13 0 BSPEG5 N 6.619  0.650  1.629  C \nC14 0 BSPEG5 N 7.064  1.999  1.140  C \nO7  0 BSPEG5 N 6.887  2.326  -0.029 O \nH1  0 BSPEG5 N -9.127 1.663  1.960  H \nH2  0 BSPEG5 N -7.672 1.053  -0.327 H \nH3  0 BSPEG5 N -7.095 2.595  0.378  H \nH4  0 BSPEG5 N -6.148 -0.255 0.979  H \nH5  0 BSPEG5 N -5.431 1.264  1.766  H \nH6  0 BSPEG5 N -3.492 0.711  1.167  H \nH7  0 BSPEG5 N -4.233 -0.784 0.344  H \nH8  0 BSPEG5 N -3.184 -0.206 -1.768 H \nH9  0 BSPEG5 N -2.549 1.379  -1.125 H \nH10 0 BSPEG5 N -2.273 -2.192 -1.064 H \nH11 0 BSPEG5 N -2.588 -1.823 0.628  H \nH12 0 BSPEG5 N -0.735 -3.363 0.440  H \nH13 0 BSPEG5 N -0.120 -1.791 1.129  H \nH14 0 BSPEG5 N 0.013  -0.424 -1.840 H \nH15 0 BSPEG5 N 1.126  -0.339 -0.393 H \nH16 0 BSPEG5 N 2.123  -0.252 -2.799 H \nH17 0 BSPEG5 N 1.433  -1.873 -3.023 H \nH18 0 BSPEG5 N 4.533  -1.974 -0.545 H \nH19 0 BSPEG5 N 3.084  -1.282 0.226  H \nH20 0 BSPEG5 N 4.812  0.010  -1.961 H \nH21 0 BSPEG5 N 3.388  0.771  -1.264 H \nH22 0 BSPEG5 N 5.623  -1.068 0.925  H \nH23 0 BSPEG5 N 6.770  -0.300 -0.255 H \nH24 0 BSPEG5 N 5.842  0.811  2.389  H \nH25 0 BSPEG5 N 7.514  0.100  2.013  H \nH26 0 BSPEG5 N 7.523  2.549  1.964  H \n#\nloop_\n_chem_comp_bond.atom_id_1\n_chem_comp_bond.atom_id_2\n_chem_comp_bond.comp_id\n_chem_comp_bond.pdbx_aromatic_flag\n_chem_comp_bond.pdbx_stereo_config\n_chem_comp_bond.value_order\nO1  C1  BSPEG5 N N DOUB \nC1  C2  BSPEG5 N N SING \nC2  C3  BSPEG5 N N SING \nC3  O2  BSPEG5 N N SING \nO2  C4  BSPEG5 N N SING \nC4  C5  BSPEG5 N N SING \nC5  O3  BSPEG5 N N SING \nO3  C6  BSPEG5 N N SING \nC6  C7  BSPEG5 N N SING \nC7  O4  BSPEG5 N N SING \nO4  C8  BSPEG5 N N SING \nC8  C9  BSPEG5 N N SING \nC9  O5  BSPEG5 N N SING \nO5  C10 BSPEG5 N N SING \nC10 C11 BSPEG5 N N SING \nC11 O6  BSPEG5 N N SING \nO6  C12 BSPEG5 N N SING \nC12 C13 BSPEG5 N N SING \nC13 C14 BSPEG5 N N SING \nC14 O7  BSPEG5 N N DOUB \nC1  H1  BSPEG5 N N SING \nC2  H2  BSPEG5 N N SING \nC2  H3  BSPEG5 N N SING \nC3  H4  BSPEG5 N N SING \nC3  H5  BSPEG5 N N SING \nC4  H6  BSPEG5 N N SING \nC4  H7  BSPEG5 N N SING \nC5  H8  BSPEG5 N N SING \nC5  H9  BSPEG5 N N SING \nC6  H10 BSPEG5 N N SING \nC6  H11 BSPEG5 N N SING \nC7  H12 BSPEG5 N N SING \nC7  H13 BSPEG5 N N SING \nC8  H14 BSPEG5 N N SING \nC8  H15 BSPEG5 N N SING \nC9  H16 BSPEG5 N N SING \nC9  H17 BSPEG5 N N SING \nC10 H18 BSPEG5 N N SING \nC10 H19 BSPEG5 N N SING \nC11 H20 BSPEG5 N N SING \nC11 H21 BSPEG5 N N SING \nC12 H22 BSPEG5 N N SING \nC12 H23 BSPEG5 N N SING \nC13 H24 BSPEG5 N N SING \nC13 H25 BSPEG5 N N SING \nC14 H26 BSPEG5 N N SING \n#\n_pdbx_chem_comp_descriptor.descriptor ?\n_pdbx_chem_comp_descriptor.type       ?\n#\n",
        "bond1": {
            "atom1": {
                "moltype": "protein",
                "atomtypes": [
                    {
                        "restype": "LYS",
                        "atomname": "NZ"
                    },
                    {
                        "restype": "SER",
                        "atomname": "OG"
                    },
                    {
                        "restype": "THR",
                        "atomname": "OG1"
                    },
                    {
                        "restype": "TYR",
                        "atomname": "OH"
                    },
                    {
                        "restype": "NTER",
                        "atomname": "N"
                    }
                ]
            },
            "atom2": {
                "moltype": "ligand",
                "restype": "BSPEG5",
                "atomname": "C1"
            }
        },
        "bond2": {
            "atom1": {
                "moltype": "protein",
                "atomtypes": [
                    {
                        "restype": "LYS",
                        "atomname": "NZ"
                    },
                    {
                        "restype": "SER",
                        "atomname": "OG"
                    },
                    {
                        "restype": "THR",
                        "atomname": "OG1"
                    },
                    {
                        "restype": "TYR",
                        "atomname": "OH"
                    },
                    {
                        "restype": "NTER",
                        "atomname": "N"
                    }
                ]
            },
            "atom2": {
                "moltype": "ligand",
                "restype": "BSPEG5",
                "atomname": "C14"
            }
        }
    },
    "BSPEG9": {
        "ccdCode": "BSPEG9",
        "userCCD": "data_BSPEG9\n#\n_chem_comp.formula                 ?\n_chem_comp.formula_weight          482.57\n_chem_comp.id                      BSPEG9\n_chem_comp.mon_nstd_parent_comp_id ?\n_chem_comp.name                    ?\n_chem_comp.pdbx_synonyms           ?\n_chem_comp.type                    non-polymer\n#\nloop_\n_chem_comp_atom.atom_id\n_chem_comp_atom.charge\n_chem_comp_atom.comp_id\n_chem_comp_atom.pdbx_leaving_atom_flag\n_chem_comp_atom.pdbx_model_Cartn_x_ideal\n_chem_comp_atom.pdbx_model_Cartn_y_ideal\n_chem_comp_atom.pdbx_model_Cartn_z_ideal\n_chem_comp_atom.type_symbol\nO1  0 BSPEG9 N 9.901   1.725  5.164  O \nC1  0 BSPEG9 N 10.882  1.355  4.544  C \nC2  0 BSPEG9 N 11.044  1.845  3.150  C \nC3  0 BSPEG9 N 10.041  1.042  2.293  C \nO2  0 BSPEG9 N 8.805   1.335  2.823  O \nC4  0 BSPEG9 N 7.809   0.653  2.155  C \nC5  0 BSPEG9 N 7.744   1.008  0.716  C \nO3  0 BSPEG9 N 6.756   0.361  -0.018 O \nC6  0 BSPEG9 N 5.497   0.626  0.439  C \nC7  0 BSPEG9 N 4.421   -0.069 -0.402 C \nO4  0 BSPEG9 N 4.462   0.322  -1.697 O \nC8  0 BSPEG9 N 3.544   -0.201 -2.540 C \nC9  0 BSPEG9 N 3.505   -1.674 -2.731 C \nO5  0 BSPEG9 N 3.251   -2.465 -1.668 O \nC10 0 BSPEG9 N 2.054   -2.391 -1.029 C \nC11 0 BSPEG9 N 0.918   -2.742 -1.932 C \nO6  0 BSPEG9 N -0.284  -2.685 -1.294 O \nC12 0 BSPEG9 N -0.647  -1.481 -0.782 C \nC13 0 BSPEG9 N -0.762  -0.430 -1.842 C \nO7  0 BSPEG9 N -1.683  -0.684 -2.827 O \nC14 0 BSPEG9 N -2.982  -0.832 -2.392 C \nC15 0 BSPEG9 N -3.553  0.345  -1.677 C \nO8  0 BSPEG9 N -4.858  0.166  -1.257 O \nC16 0 BSPEG9 N -5.759  -0.078 -2.260 C \nC17 0 BSPEG9 N -7.165  -0.230 -1.667 C \nO9  0 BSPEG9 N -7.452  0.950  -1.035 O \nC18 0 BSPEG9 N -8.719  0.973  -0.474 C \nC19 0 BSPEG9 N -8.981  -0.066 0.558  C \nO10 0 BSPEG9 N -8.201  -0.074 1.660  O \nC20 0 BSPEG9 N -8.146  0.986  2.489  C \nC21 0 BSPEG9 N -9.388  1.435  3.167  C \nC22 0 BSPEG9 N -10.460 1.862  2.238  C \nO11 0 BSPEG9 N -11.307 2.202  1.403  O \nH1  0 BSPEG9 N 11.692  0.687  4.880  H \nH2  0 BSPEG9 N 12.036  1.695  2.725  H \nH3  0 BSPEG9 N 10.734  2.923  3.085  H \nH4  0 BSPEG9 N 10.197  1.364  1.244  H \nH5  0 BSPEG9 N 10.341  -0.008 2.344  H \nH6  0 BSPEG9 N 8.025   -0.437 2.327  H \nH7  0 BSPEG9 N 6.817   0.915  2.634  H \nH8  0 BSPEG9 N 8.675   0.869  0.136  H \nH9  0 BSPEG9 N 7.475   2.104  0.656  H \nH10 0 BSPEG9 N 5.341   1.720  0.460  H \nH11 0 BSPEG9 N 5.333   0.236  1.467  H \nH12 0 BSPEG9 N 4.576   -1.165 -0.331 H \nH13 0 BSPEG9 N 3.459   0.278  0.053  H \nH14 0 BSPEG9 N 3.773   0.232  -3.569 H \nH15 0 BSPEG9 N 2.509   0.226  -2.347 H \nH16 0 BSPEG9 N 4.517   -1.954 -3.146 H \nH17 0 BSPEG9 N 2.799   -1.872 -3.590 H \nH18 0 BSPEG9 N 1.905   -1.443 -0.485 H \nH19 0 BSPEG9 N 2.067   -3.174 -0.210 H \nH20 0 BSPEG9 N 1.044   -3.808 -2.287 H \nH21 0 BSPEG9 N 0.963   -2.130 -2.858 H \nH22 0 BSPEG9 N -1.643  -1.645 -0.262 H \nH23 0 BSPEG9 N -0.005  -1.070 0.023  H \nH24 0 BSPEG9 N -1.008  0.536  -1.308 H \nH25 0 BSPEG9 N 0.221   -0.316 -2.356 H \nH26 0 BSPEG9 N -3.018  -1.744 -1.769 H \nH27 0 BSPEG9 N -3.595  -0.983 -3.323 H \nH28 0 BSPEG9 N -2.956  0.647  -0.789 H \nH29 0 BSPEG9 N -3.551  1.237  -2.339 H \nH30 0 BSPEG9 N -5.815  0.765  -2.972 H \nH31 0 BSPEG9 N -5.545  -0.987 -2.863 H \nH32 0 BSPEG9 N -7.065  -1.074 -0.942 H \nH33 0 BSPEG9 N -7.892  -0.484 -2.458 H \nH34 0 BSPEG9 N -9.417  0.767  -1.342 H \nH35 0 BSPEG9 N -8.886  2.021  -0.154 H \nH36 0 BSPEG9 N -10.087 -0.127 0.784  H \nH37 0 BSPEG9 N -8.809  -1.061 0.039  H \nH38 0 BSPEG9 N -7.430  0.701  3.324  H \nH39 0 BSPEG9 N -7.610  1.881  2.039  H \nH40 0 BSPEG9 N -9.205  2.328  3.831  H \nH41 0 BSPEG9 N -9.822  0.682  3.857  H \nH42 0 BSPEG9 N -11.427 2.219  2.517  H \n#\nloop_\n_chem_comp_bond.atom_id_1\n_chem_comp_bond.atom_id_2\n_chem_comp_bond.comp_id\n_chem_comp_bond.pdbx_aromatic_flag\n_chem_comp_bond.pdbx_stereo_config\n_chem_comp_bond.value_order\nO1  C1  BSPEG9 N N DOUB \nC1  C2  BSPEG9 N N SING \nC2  C3  BSPEG9 N N SING \nC3  O2  BSPEG9 N N SING \nO2  C4  BSPEG9 N N SING \nC4  C5  BSPEG9 N N SING \nC5  O3  BSPEG9 N N SING \nO3  C6  BSPEG9 N N SING \nC6  C7  BSPEG9 N N SING \nC7  O4  BSPEG9 N N SING \nO4  C8  BSPEG9 N N SING \nC8  C9  BSPEG9 N N SING \nC9  O5  BSPEG9 N N SING \nO5  C10 BSPEG9 N N SING \nC10 C11 BSPEG9 N N SING \nC11 O6  BSPEG9 N N SING \nO6  C12 BSPEG9 N N SING \nC12 C13 BSPEG9 N N SING \nC13 O7  BSPEG9 N N SING \nO7  C14 BSPEG9 N N SING \nC14 C15 BSPEG9 N N SING \nC15 O8  BSPEG9 N N SING \nO8  C16 BSPEG9 N N SING \nC16 C17 BSPEG9 N N SING \nC17 O9  BSPEG9 N N SING \nO9  C18 BSPEG9 N N SING \nC18 C19 BSPEG9 N N SING \nC19 O10 BSPEG9 N N SING \nO10 C20 BSPEG9 N N SING \nC20 C21 BSPEG9 N N SING \nC21 C22 BSPEG9 N N SING \nC22 O11 BSPEG9 N N DOUB \nC1  H1  BSPEG9 N N SING \nC2  H2  BSPEG9 N N SING \nC2  H3  BSPEG9 N N SING \nC3  H4  BSPEG9 N N SING \nC3  H5  BSPEG9 N N SING \nC4  H6  BSPEG9 N N SING \nC4  H7  BSPEG9 N N SING \nC5  H8  BSPEG9 N N SING \nC5  H9  BSPEG9 N N SING \nC6  H10 BSPEG9 N N SING \nC6  H11 BSPEG9 N N SING \nC7  H12 BSPEG9 N N SING \nC7  H13 BSPEG9 N N SING \nC8  H14 BSPEG9 N N SING \nC8  H15 BSPEG9 N N SING \nC9  H16 BSPEG9 N N SING \nC9  H17 BSPEG9 N N SING \nC10 H18 BSPEG9 N N SING \nC10 H19 BSPEG9 N N SING \nC11 H20 BSPEG9 N N SING \nC11 H21 BSPEG9 N N SING \nC12 H22 BSPEG9 N N SING \nC12 H23 BSPEG9 N N SING \nC13 H24 BSPEG9 N N SING \nC13 H25 BSPEG9 N N SING \nC14 H26 BSPEG9 N N SING \nC14 H27 BSPEG9 N N SING \nC15 H28 BSPEG9 N N SING \nC15 H29 BSPEG9 N N SING \nC16 H30 BSPEG9 N N SING \nC16 H31 BSPEG9 N N SING \nC17 H32 BSPEG9 N N SING \nC17 H33 BSPEG9 N N SING \nC18 H34 BSPEG9 N N SING \nC18 H35 BSPEG9 N N SING \nC19 H36 BSPEG9 N N SING \nC19 H37 BSPEG9 N N SING \nC20 H38 BSPEG9 N N SING \nC20 H39 BSPEG9 N N SING \nC21 H40 BSPEG9 N N SING \nC21 H41 BSPEG9 N N SING \nC22 H42 BSPEG9 N N SING \n#\n_pdbx_chem_comp_descriptor.descriptor ?\n_pdbx_chem_comp_descriptor.type       ?\n#\n",
        "bond1": {
            "atom1": {
                "moltype": "protein",
                "atomtypes": [
                    {
                        "restype": "LYS",
                        "atomname": "NZ"
                    },
                    {
                        "restype": "SER",
                        "atomname": "OG"
                    },
                    {
                        "restype": "THR",
                        "atomname": "OG1"
                    },
                    {
                        "restype": "TYR",
                        "atomname": "OH"
                    },
                    {
                        "restype": "NTER",
                        "atomname": "N"
                    }
                ]
            },
            "atom2": {
                "moltype": "ligand",
                "restype": "BSPEG9",
                "atomname": "C1"
            }
        },
        "bond2": {
            "atom1": {
                "moltype": "protein",
                "atomtypes": [
                    {
                        "restype": "LYS",
                        "atomname": "NZ"
                    },
                    {
                        "restype": "SER",
                        "atomname": "OG"
                    },
                    {
                        "restype": "THR",
                        "atomname": "OG1"
                    },
                    {
                        "restype": "TYR",
                        "atomname": "OH"
                    },
                    {
                        "restype": "NTER",
                        "atomname": "N"
                    }
                ]
            },
            "atom2": {
                "moltype": "ligand",
                "restype": "BSPEG9",
                "atomname": "C22"
            }
        }
    },
    "S-S": {
        "ccdCode": "S-S",
        "userCCD": """data_S-S
#
_chem_comp.id S-S
_chem_comp.name 'Disulfide bond'
_chem_comp.type non-polymer
_chem_comp.formula 'S2'
_chem_comp.mon_nstd_parent_comp_id ?
_chem_comp.pdbx_synonyms ?
_chem_comp.formula_weight XXX
#
loop_
_chem_comp_atom.comp_id
_chem_comp_atom.atom_id
_chem_comp_atom.alt_atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.charge
_chem_comp_atom.pdbx_align
_chem_comp_atom.pdbx_aromatic_flag
_chem_comp_atom.pdbx_leaving_atom_flag
_chem_comp_atom.pdbx_stereo_config
_chem_comp_atom.pdbx_backbone_atom_flag
_chem_comp_atom.pdbx_n_terminal_atom_flag
_chem_comp_atom.pdbx_c_terminal_atom_flag
_chem_comp_atom.model_Cartn_x
_chem_comp_atom.model_Cartn_y
_chem_comp_atom.model_Cartn_z
_chem_comp_atom.pdbx_model_Cartn_x_ideal
_chem_comp_atom.pdbx_model_Cartn_y_ideal
_chem_comp_atom.pdbx_model_Cartn_z_ideal
_chem_comp_atom.pdbx_component_atom_id
_chem_comp_atom.pdbx_component_comp_id
_chem_comp_atom.pdbx_ordinal
S-S S1 S1 S 0 1 N N N N N N -0.943  0.513 -0.372 -0.943  0.513 -0.372 S1 S-S 1
S-S S2 S2 S 0 1 N N N N N N 0.924  0.463  0.471 0.924  0.463  0.471 S2 S-S 2
#
loop_
_chem_comp_bond.comp_id
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_order
_chem_comp_bond.pdbx_aromatic_flag
_chem_comp_bond.pdbx_stereo_config
_chem_comp_bond.pdbx_ordinal 
S-S  S1  S2 SING N N 1
#
_pdbx_chem_comp_descriptor.type SMILES_CANONICAL
_pdbx_chem_comp_descriptor.descriptor 'S-S'
#
""",
        "bond1": {
            "atom1": {
                "moltype": "protein",
                "atomtypes": [
                    {
                        "restype": "ALA",
                        "atomname": "CB"
                    }
                ],
            },
            "atom2": {
                "moltype": "ligand",
                "restype": "S-S",
                "atomname": "S1",
            }
        },
        "bond2": {
            "atom1": {
                "moltype": "protein",
                "atomtypes": [
                    {
                        "restype": "ALA",
                        "atomname": "CB"
                    }
                ],
            },
            "atom2": {
                "moltype": "ligand",
                "restype": "S-S",
                "atomname": "S2",
            }
        }
    }

#TODO: add more crosslinkers from https://www.rappsilberlab.org/software/xisearch/#crosslinker-selection-presets

}

@dataclass(frozen=True)
class LinkDefinition:
    name: str
    ccd_code: str
    user_ccd: str
    bond1: Dict[str, Any]
    bond2: Dict[str, Any]

    @classmethod
    def from_dict(cls, name: str, definition: Dict[str, Any]) -> "LinkDefinition":
        return cls(
            name=name,
            ccd_code=definition["ccdCode"],
            user_ccd=definition["userCCD"],
            bond1=definition["bond1"],
            bond2=definition["bond2"],
        )

    def get_expected_restypes_bond1(self) -> List[str]:
        return [atom["restype"] for atom in self.bond1["atom1"]["atomtypes"]]

    def get_expected_restypes_bond2(self) -> List[str]:
        return [atom["restype"] for atom in self.bond2["atom1"]["atomtypes"]]

    def get_bond_atom1(self, restype: str, resid: int, bond: int = 1) -> str:
        """
        Returns the bonded atom name from the first atom of the specified bond.
        If no match is found, and resid is 1, looks for type 'NTER'.
        """
        atomtypes = (
            self.bond1["atom1"]["atomtypes"] if bond == 1 else self.bond2["atom1"]["atomtypes"]
        )
        for atom in atomtypes:
            if atom["restype"] == restype:
                return atom["atomname"]
        for atom in atomtypes:
            if atom["restype"] == "NTER" and resid == 1:
                return atom["atomname"]
        raise ValueError(f"No valid bond atom found for restype {restype} at residue {resid}")

    def get_bond_atom2(self, bond: int = 1) -> str:
        """
        Returns the bonded atom name from the second atom of the specified bond.
        """
        return self.bond1["atom2"]["atomname"] if bond == 1 else self.bond2["atom2"]["atomname"]


def create_link_definitions() -> Dict[str, LinkDefinition]:
    """
    Factory that converts the CROSSLINKS dictionary into LinkDefinition instances.
    """
    link_defs = {}
    for name, definition in CROSSLINKS.items():
        link_defs[name] = LinkDefinition.from_dict(name, definition)
    return link_defs

def get_link_definition(linkset_name: str) -> LinkDefinition:
    """Create and return a LinkDefinition instance for the specified linkset."""
    return LinkDefinition.from_dict(linkset_name, CROSSLINKS[linkset_name])
