"""
Script create a curated by user crosslinker:
1) makes a picture of the custom ccd crosslinker (in order to choose the connecting atoms)
2) Creates the final element fo the dictionary that should be put in CROSSLINKS dictionary in crosslink_definitions.py
"""

from dynamic_crosslink import ccd_from_smiles

if __name__ == '__main__':
    smiles, name = "CCCCC(=O)NCCCCCCCCCCC=O", "SDA25A"
    smiles, name = "CCCCC(=O)NCCCCCCCCC(=O)NCCCC=O", "SDA25A"
    smiles, name = "CCCCC(=O)NCCCCCCCCCCC(=O)NCCCC=O", "SDA25A"
    cross_ccd = ccd_from_smiles(smiles, name)
    print(str(cross_ccd))

