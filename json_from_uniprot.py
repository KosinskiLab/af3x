import argparse
import json
import requests
import string
import itertools

#TODO: ChatGPT, revise and test this script

def id_generator():
    """Yield single-character IDs from A to Z, then AA, AB, etc."""
    base = string.ascii_uppercase
    for length in range(1, 3):  # Support IDs of length 1 or 2 (A-Z, AA-ZZ)
        for id_ in map("".join, itertools.product(base, repeat=length)):
            yield id_

def fetch_sequence(uniprot_id):
    """Fetch the protein sequence for a given UniProt ID from the UniProt API."""
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    try:
        response = requests.get(url)
        response.raise_for_status()
        fasta_data = response.text
        sequence = "".join(fasta_data.split("\n")[1:])  # Skip the FASTA header
        return sequence
    except requests.exceptions.RequestException as e:
        print(f"Error fetching sequence for {uniprot_id}: {e}")
        return None

def construct_json(uniprot_ids, job_name):
    """Construct the JSON structure."""
    id_gen = id_generator()  # Initialize ID generator
    sequences = []
    for uniprot_id in uniprot_ids:
        sequence = fetch_sequence(uniprot_id)
        if sequence:
            sequences.append({
                "protein": {
                    "id": next(id_gen),  # Assign a unique chain-like ID
                    "sequence": sequence
                }
            })
        else:
            print(f"Skipping {uniprot_id} due to missing sequence.")
    
    json_structure = {
        "name": job_name if job_name else "_".join(uniprot_ids),
        "modelSeeds": [1],
        "sequences": sequences,
        "dialect": "alphafold3",
        "version": 1
    }
    return json_structure

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate JSON for a list of UniProt IDs with sequences.")
    parser.add_argument(
        "uniprot_ids",
        nargs="+",
        help="List of UniProt IDs"
    )
    parser.add_argument(
        "--job_name",
        default=None,
        help="Optional job name (default is UniProt IDs joined with '_')"
    )
    parser.add_argument(
        "--output",
        default="output.json",
        help="Output JSON file name (default is 'output.json')"
    )
    args = parser.parse_args()

    # Construct JSON
    json_data = construct_json(args.uniprot_ids, args.job_name)

    # Write JSON to file
    with open(args.output, "w") as json_file:
        json.dump(json_data, json_file, indent=4)
    
    print(f"JSON written to {args.output}")

if __name__ == "__main__":
    main()
