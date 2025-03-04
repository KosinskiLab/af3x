import argparse
import json
import glob
import numpy as np

from alphafold3 import structure
from alphafold3.common import folding_input
from alphafold3.crosslinks import crosslink_definitions


def parse_args():
    parser = argparse.ArgumentParser(description="""Evaluate agreement of structures with crosslinks.
    Distances will be calculated between crosslinked atoms. 
    WARNING: this script has been only tested on AlphaFold3 mmcif models.
    """)
    parser.add_argument('-f', '--files', nargs='+', required=True,
                        help='paths to mmcif files, supports wildcards')
    parser.add_argument('-j', '--input_json', required=True,
                        help='path to input json file in AF3x format')
    parser.add_argument('-t', '--threshold', type=float,
                        help='single threshold for all crosslinks')
    parser.add_argument('-T', '--thresholds',
                        help='thresholds for specific crosslink types, '
                             'format: <xlink>:<val>_<xlink>:<val> (e.g. DSS:30_DSG:20)')
    parser.add_argument('-s', '--save_distances_csv', action='store_true',
                        help='save distances to CSV files for each input mmcif')
    parser.add_argument('--use_calpha', action='store_true',
                        help='calculate distances between Calpha atoms of crosslinked residues instead of crosslinked atoms')
    parser.add_argument('--summary_out_csv', type=str,
                        help='path to summary output CSV file')
    parser.add_argument('--find_shortest_homooligomeric', action='store_true',
                        help='find the shortest crosslink for homooligomeric proteins')
    parser.add_argument('--extract_avg_plddt', action='store_true',
                        help='extract the average pLDDT score for each structure and include in summary CSV')
    parser.add_argument('--extract_avg_plddt_calpha', action='store_true',
                        help='extract the average pLDDT score of Calpha atoms for each structure and include in summary CSV')

    args = parser.parse_args()
    
    if (args.extract_avg_plddt or args.extract_avg_plddt_calpha) and not args.summary_out_csv:
        raise ValueError("--summary_out_csv must be defined if --extract_avg_plddt or --extract_avg_plddt_calpha is specified")
    
    return args


def load_thresholds(args, fold_input):
    if args.threshold and args.thresholds:
        raise ValueError("use either --threshold or --thresholds, not both")

    if not args.threshold and not args.thresholds:
        raise ValueError("specify either --threshold or --thresholds")

    thresholds = {}
    if args.thresholds:
        for t in args.thresholds.split('_'):
            crosslink_type, val = t.split(':')
            thresholds[crosslink_type] = float(val)
    else:
        for xl_def in fold_input.crosslinks:
            thresholds[xl_def['name']] = args.threshold
    return thresholds


def load_mmcif_files(file_patterns):
    mmcif_files = []
    for pattern in file_patterns:
        mmcif_files.extend(glob.glob(pattern))
    return mmcif_files


def load_fold_input(input_json_path):
    with open(input_json_path, 'r') as json_file:
        return folding_input.Input.from_json(json_file.read(), expand_crosslinks=False)


def extract_atom_map(struc):
    atom_map = {}
    for at in struc.iter_atoms():
        atom_map[(at['chain_id'], int(at['res_auth_seq_id']), at['atom_name'])] = at
    return atom_map

def find_homooligomeric_chains(fold_input):
    seq_to_chains = {}
    for chain in fold_input.chains:
        if hasattr(chain, 'sequence'):
            seq = chain.sequence
            seq_to_chains.setdefault(seq, []).append(chain.id)
    homooligomeric_chain_groups = []
    for seq, chains in seq_to_chains.items():
        if len(chains) > 1:
            homooligomeric_chain_groups.append(chains)
    return homooligomeric_chain_groups

def find_atoms_for_xlinks(fold_input, atom_map, use_calpha=False, find_shortest_homooligomeric=False):
    if find_shortest_homooligomeric:
        homooligomeric_chains = find_homooligomeric_chains(fold_input)

    xlinks = []
    xlink_count = 0
    for xl_def in fold_input.crosslinks:
        crosslink_def = crosslink_definitions.CROSSLINKS[xl_def['name']]
        for xlink in xl_def['residue_pairs']:
            xlink_count += 1    
            atom_groups = []
            for i in range(2):
                chain, resid = xlink[i]
                if find_shortest_homooligomeric:
                    for chain_group in homooligomeric_chains:
                        if chain in chain_group:
                            chains = chain_group
                            break
                else:
                    chains = [chain]

                atoms = []
                for chain in chains:
                    if use_calpha:
                        atom = atom_map.get((chain, resid, 'CA'))
                        if atom:
                            atoms.append(atom)
                    else:
                        atomtypes = crosslink_def[f"bond{i+1}"]["atom1"]["atomtypes"]
                        found = False
                        for atomtype_def in atomtypes:
                            def_resname = atomtype_def["restype"]
                            def_atom_name = atomtype_def["atomname"]
                            atom = atom_map.get((chain, resid, def_atom_name))
                            if atom and atom['res_name'] == def_resname:
                                atoms.append(atom)
                                found = True
                                break
                        if not found:
                            print(f"could not find atom for residue {chain}:{resid} in crosslink {xl_def['name']}")
                            break

                if len(atoms) != len(chains):
                    print(f"could not find atom for residue {chain}:{resid} in crosslink {xl_def['name']}")
                    break
                atom_groups.append(atoms)

            if len(atom_groups) == 2:
                xlinks.append({
                    "residue_pair": xlink,
                    "atoms1": atom_groups[0],
                    "atoms2": atom_groups[1]
                })

    assert len(xlinks) == xlink_count
    return xlinks

def compute_distances(xlinks_with_atoms):
    distances = []
    for xlink in xlinks_with_atoms:
        min_dist = float('inf')
        for atom1 in xlink['atoms1']:
            for atom2 in xlink['atoms2']:
                atom1_coords = np.array([atom1['atom_x'], atom1['atom_y'], atom1['atom_z']])
                atom2_coords = np.array([atom2['atom_x'], atom2['atom_y'], atom2['atom_z']])
                dist = np.linalg.norm(atom1_coords - atom2_coords)
                if dist < min_dist:
                    min_dist = dist
        distances.append(min_dist)
    return distances


def evaluate_crosslinks(fold_input, distances, thresholds):
    satisfied_count = 0
    idx = 0
    for xl_def in fold_input.crosslinks:
        threshold = thresholds[xl_def['name']]
        for _ in xl_def['residue_pairs']:
            if distances[idx] <= threshold:
                satisfied_count += 1
            idx += 1
    return satisfied_count

def extract_avg_plddt(struc, calpha=False):
    plddts = []
    for at in struc.iter_atoms():
        if calpha:
            if at['atom_name'] == 'CA':
                plddts.append(at['atom_b_factor'])
        else:
            plddts.append(at['atom_b_factor'])

    return float(np.mean(plddts))

def main():
    args = parse_args()
    fold_input = load_fold_input(args.input_json)
    thresholds = load_thresholds(args, fold_input)
    mmcif_files = load_mmcif_files(args.files)


    FIELDS = ["mmcif_file", "satisfied_count", "total_crosslinks", "percentage_satisfied"]
    if args.extract_avg_plddt:
        FIELDS.append("avg_plddt")
    if args.extract_avg_plddt_calpha:
        FIELDS.append("avg_plddt_calpha")

    eval_results = []
    for mmcif_file in mmcif_files:
        print(f"evaluating crosslinks in {mmcif_file}")
        with open(mmcif_file) as f:
            mmcif_str = f.read()

        struc = structure.from_mmcif(mmcif_str, include_bonds=True)
        struc = struc.copy_and_update(residues=struc.present_residues)

        atom_map = extract_atom_map(struc)
        xlinks_with_atoms = find_atoms_for_xlinks(fold_input, atom_map, args.use_calpha, find_shortest_homooligomeric=args.find_shortest_homooligomeric)
        distances = compute_distances(xlinks_with_atoms)
        satisfied_count = evaluate_crosslinks(fold_input, distances, thresholds)

        if args.extract_avg_plddt:
            avg_plddt = extract_avg_plddt(struc, calpha=False)
        if args.extract_avg_plddt_calpha:
            avg_plddt_calpha = extract_avg_plddt(struc, calpha=True)

        total_crosslinks = len(xlinks_with_atoms)
        percentage_satisfied = (satisfied_count / total_crosslinks) * 100
        print(f"number of crosslinks that satisfy the thresholds: {satisfied_count} out of {total_crosslinks}")
        print(f"percentage of crosslinks that satisfy the thresholds: {percentage_satisfied:.2f}%")

        if args.summary_out_csv:
            eval_results.append({
                "mmcif_file": mmcif_file,
                "satisfied_count": satisfied_count,
                "total_crosslinks": total_crosslinks,
                "percentage_satisfied": percentage_satisfied
            })

        if args.extract_avg_plddt:
            eval_results[-1]["avg_plddt"] = avg_plddt
        if args.extract_avg_plddt_calpha:
            eval_results[-1]["avg_plddt_calpha"] = avg_plddt_calpha

        if args.save_distances_csv:
            csv_file = mmcif_file.replace('.cif', '_crosslinks.csv')
            with open(csv_file, 'w') as f:
                f.write("chain1,resid1,chain2,resid2,distance\n")
                for xlink, dist in zip(xlinks_with_atoms, distances):
                    chain1, resid1 = xlink['residue_pair'][0]
                    chain2, resid2 = xlink['residue_pair'][1]
                    f.write(f"{chain1},{resid1},{chain2},{resid2},{dist}\n")

    if args.summary_out_csv:
        with open(args.summary_out_csv, 'w') as f:
            heading = ",".join(FIELDS)
            f.write(heading + "\n")
            
            for result in eval_results:
                row = ",".join(f"{result[field]:.2f}" if isinstance(result[field], float) else str(result[field]) for field in FIELDS)
                f.write(row + "\n")

if __name__ == "__main__":
    main()
