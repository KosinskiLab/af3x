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
