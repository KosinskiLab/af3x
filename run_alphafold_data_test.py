# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Tests the AlphaFold 3 data pipeline."""

import contextlib
import datetime
import difflib
import functools
import hashlib
import json
import os
import pathlib
import pickle
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
from alphafold3 import structure
from alphafold3.common import folding_input
from alphafold3.common import resources
from alphafold3.common.testing import data as testing_data
from alphafold3.constants import chemical_components
from alphafold3.data import featurisation
from alphafold3.data import pipeline
from alphafold3.model import features
from alphafold3.model.atom_layout import atom_layout
import jax
import numpy as np

import run_alphafold
import shutil


_JACKHMMER_BINARY_PATH = shutil.which('jackhmmer')
_NHMMER_BINARY_PATH = shutil.which('nhmmer')
_HMMALIGN_BINARY_PATH = shutil.which('hmmalign')
_HMMSEARCH_BINARY_PATH = shutil.which('hmmsearch')
_HMMBUILD_BINARY_PATH = shutil.which('hmmbuild')


@contextlib.contextmanager
def _output(name: str):
  with open(result_path := f'{absltest.TEST_TMPDIR.value}/{name}', "wb") as f:
    yield result_path, f


@functools.singledispatch
def _hash_data(x: Any, /) -> str:
  if x is None:
    return '<<None>>'
  return _hash_data(json.dumps(x).encode('utf-8'))


@_hash_data.register
def _(x: bytes, /) -> str:
  return hashlib.sha256(x).hexdigest()


@_hash_data.register
def _(x: jax.Array) -> str:
  return _hash_data(jax.device_get(x))


@_hash_data.register
def _(x: np.ndarray) -> str:
  if x.dtype == object:
    return ';'.join(map(_hash_data, x.ravel().tolist()))
  return _hash_data(x.tobytes())


@_hash_data.register
def _(_: structure.Structure) -> str:
  return '<<structure>>'


@_hash_data.register
def _(_: atom_layout.AtomLayout) -> str:
  return '<<atom-layout>>'


def _generate_diff(actual: str, expected: str) -> str:
  return '\n'.join(
      difflib.unified_diff(
          expected.split('\n'),
          actual.split('\n'),
          fromfile='expected',
          tofile='actual',
          lineterm='',
      )
  )


class DataPipelineTest(parameterized.TestCase):
  """Test AlphaFold 3 inference."""

  def setUp(self):
    super().setUp()
    small_bfd_database_path = testing_data.Data(
        resources.ROOT
        / 'test_data/miniature_databases/bfd-first_non_consensus_sequences__subsampled_1000.fasta'
    ).path()
    mgnify_database_path = testing_data.Data(
        resources.ROOT
        / 'test_data/miniature_databases/mgy_clusters__subsampled_1000.fa'
    ).path()
    uniprot_cluster_annot_database_path = testing_data.Data(
        resources.ROOT
        / 'test_data/miniature_databases/uniprot_all__subsampled_1000.fasta'
    ).path()
    uniref90_database_path = testing_data.Data(
        resources.ROOT
        / 'test_data/miniature_databases/uniref90__subsampled_1000.fasta'
    ).path()
    ntrna_database_path = testing_data.Data(
        resources.ROOT
        / 'test_data/miniature_databases/nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq__subsampled_1000.fasta'
    ).path()
    rfam_database_path = testing_data.Data(
        resources.ROOT
        / 'test_data/miniature_databases/rfam_14_4_clustered_rep_seq__subsampled_1000.fasta'
    ).path()
    rna_central_database_path = testing_data.Data(
        resources.ROOT
        / 'test_data/miniature_databases/rnacentral_active_seq_id_90_cov_80_linclust__subsampled_1000.fasta'
    ).path()
    pdb_database_path = testing_data.Data(
        resources.ROOT / 'test_data/miniature_databases/pdb_mmcif'
    ).path()
    seqres_database_path = testing_data.Data(
        resources.ROOT
        / 'test_data/miniature_databases/pdb_seqres_2022_09_28__subsampled_1000.fasta'
    ).path()

    self._data_pipeline_config = pipeline.DataPipelineConfig(
        jackhmmer_binary_path=_JACKHMMER_BINARY_PATH,
        nhmmer_binary_path=_NHMMER_BINARY_PATH,
        hmmalign_binary_path=_HMMALIGN_BINARY_PATH,
        hmmsearch_binary_path=_HMMSEARCH_BINARY_PATH,
        hmmbuild_binary_path=_HMMBUILD_BINARY_PATH,
        small_bfd_database_path=small_bfd_database_path,
        mgnify_database_path=mgnify_database_path,
        uniprot_cluster_annot_database_path=uniprot_cluster_annot_database_path,
        uniref90_database_path=uniref90_database_path,
        ntrna_database_path=ntrna_database_path,
        rfam_database_path=rfam_database_path,
        rna_central_database_path=rna_central_database_path,
        pdb_database_path=pdb_database_path,
        seqres_database_path=seqres_database_path,
        max_template_date=datetime.date(2021, 9, 30),
    )
    test_input = {
        'name': '5tgy',
        'modelSeeds': [1234],
        'sequences': [
            {
                'protein': {
                    'id': 'P',
                    'sequence': (
                        'SEFEKLRQTGDELVQAFQRLREIFDKGDDDSLEQVLEEIEELIQKHRQLFDNRQEAADTEAAKQGDQWVQLFQRFREAIDKGDKDSLEQLLEELEQALQKIRELAEKKN'
                    ),
                    'modifications': [],
                    'unpairedMsa': None,
                    'pairedMsa': None,
                }
            },
            {'ligand': {'id': 'LL', 'ccdCodes': ['7BU']}},
        ],
        'dialect': folding_input.JSON_DIALECT,
        'version': folding_input.JSON_VERSION,
    }
    self._test_input_json = json.dumps(test_input)

  def compare_golden(self, result_path: str) -> None:
    filename = os.path.split(result_path)[1]
    golden_path = testing_data.Data(
        resources.ROOT / f'test_data/{filename}'
    ).path()
    with open(golden_path, 'r') as golden_file:
      golden_text = golden_file.read().rstrip('\n')
    with open(result_path, 'r') as result_file:
      result_text = result_file.read().rstrip('\n')

    diff = _generate_diff(result_text, golden_text)

    self.assertEqual(diff, "", f"Result differs from golden:\n{diff}")

  def test_config(self):
    model_config = run_alphafold.make_model_config()
    model_config_as_str = json.dumps(
        model_config.as_dict(), sort_keys=True, indent=2
    )
    with _output('model_config.json') as (result_path, output):
      output.write(model_config_as_str.encode('utf-8'))
    self.compare_golden(result_path)

  def test_featurisation(self):
    """Run featurisation and assert that the output is as expected."""
    fold_input = folding_input.Input.from_json(self._test_input_json)
    data_pipeline = pipeline.DataPipeline(self._data_pipeline_config)
    full_fold_input = data_pipeline.process(fold_input)
    featurised_example = featurisation.featurise_input(
        full_fold_input,
        ccd=chemical_components.Ccd(),
        buckets=None,
    )
    del featurised_example[0]['ref_pos']  # Depends on specific RDKit version.

    with _output('featurised_example.pkl') as (_, output):
      output.write(pickle.dumps(featurised_example))
    featurised_example = jax.tree_util.tree_map(_hash_data, featurised_example)
    with _output('featurised_example.json') as (result_path, output):
      output.write(
          json.dumps(featurised_example, sort_keys=True, indent=2).encode(
              'utf-8'
          )
      )
    self.compare_golden(result_path)

  def test_write_input_json(self):
    fold_input = folding_input.Input.from_json(self._test_input_json)
    output_dir = self.create_tempdir().full_path
    run_alphafold.write_fold_input_json(fold_input, output_dir)
    with open(
        os.path.join(output_dir, f'{fold_input.sanitised_name()}_data.json'),
        'rt',
    ) as f:
      actual_fold_input = folding_input.Input.from_json(f.read())

    self.assertEqual(actual_fold_input, fold_input)

  def test_process_fold_input_runs_only_data_pipeline(self):
    fold_input = folding_input.Input.from_json(self._test_input_json)
    output_dir = self.create_tempdir().full_path
    run_alphafold.process_fold_input(
        fold_input=fold_input,
        data_pipeline_config=self._data_pipeline_config,
        model_runner=None,
        output_dir=output_dir,
    )
    with open(
        os.path.join(output_dir, f'{fold_input.sanitised_name()}_data.json'),
        'rt',
    ) as f:
      actual_fold_input = folding_input.Input.from_json(f.read())

    featurisation.validate_fold_input(actual_fold_input)

  @parameterized.product(num_db_dirs=tuple(range(1, 3)))
  def test_replace_db_dir(self, num_db_dirs: int) -> None:
    """Test that the db_dir is replaced correctly."""
    db_dirs = [pathlib.Path(self.create_tempdir()) for _ in range(num_db_dirs)]
    db_dirs_posix = [db_dir.as_posix() for db_dir in db_dirs]

    for i, db_dir in enumerate(db_dirs):
      for j in range(i + 1):
        (db_dir / f'filename{j}.txt').write_text(f'hello world {i}')

    for i in range(num_db_dirs):
      self.assertEqual(
          pathlib.Path(
              run_alphafold.replace_db_dir(
                  f'${{DB_DIR}}/filename{i}.txt', db_dirs_posix
              )
          ).read_text(),
          f'hello world {i}',
      )
    with self.assertRaises(FileNotFoundError):
      run_alphafold.replace_db_dir(
          f'${{DB_DIR}}/filename{num_db_dirs}.txt', db_dirs_posix
      )

class CrosslinkDataPipelineTest(DataPipelineTest):
  """Test AlphaFold 3 inference."""

  def setUp(self):
    super().setUp()
    test_input = {
        'name': '5tgy',
        'modelSeeds': [1234],
        'sequences': [
            {
                'protein': {
                    'id': 'A',
                    'sequence': 'SEFEKLRQTGDELVQAFQRLREIFDKGDDDSLEQVLEEIEELIQKHRQLFDNRQEAADTEAAKQGDQWVQLFQRFREAIDKGDKDSLEQLLEELEQALQKIRELAEKKN',
                    'modifications': [],
                    'unpairedMsa': None,
                    'pairedMsa': None,
                }
            },
            {
                'protein': {
                    'id': 'B',
                    'sequence': 'SEFEKLRQTGDELVQAFQRLREIFDKGDDDSLEQVLEEIEELIQKHRQLFDNRQEAADTEAAKQGDQWVQLFQRFREAIDKGDKDSLEQLLEELEQALQKIRELAEKKN',
                    'modifications': [],
                    'unpairedMsa': None,
                    'pairedMsa': None,
                }
            }
        ],
        'crosslinks': [
            {
                'name': 'azide-A-DSBSO',
                'residue_pairs': [
                  (("A", 5), ("B", 5)),
                  (("A", 81), ("B", 81)),
                ]
            }
        ],
        'dialect': folding_input.JSON_DIALECT,
        'version': folding_input.JSON_VERSION,
    }
    self._test_input_json = json.dumps(test_input)
    
    self.expected_bonded_atom_pairs = (
        (('A', 5, 'NZ'), ('C', 1, 'C16')),
        (('B', 5, 'NZ'), ('C', 1, 'C31')),
        (('A', 81, 'NZ'), ('D', 1, 'C16')),
        (('B', 81, 'NZ'), ('D', 1, 'C31')),
      )
    
    self.expected_ligands = [
      {
        "id": "C",
        "ccd_ids": ("azide-A-DSBSO",)
      },
      {
        "id": "D",
        "ccd_ids": ("azide-A-DSBSO",)
     }
    ]

  def compare_golden(self, result_path: str) -> None:
    raise NotImplementedError
  
  @absltest.skip('Skipping test becuase compare_golden is not implemented')
  def test_config(self):
    return super().test_config()

  @absltest.skip('Skipping test becuase compare_golden is not implemented')
  def test_featurisation(self):
    return super().test_featurisation()

  def _check_crosslinks(self, fold_input):
    self.assertIsInstance(fold_input.crosslinks, list)
    for link_set in fold_input.crosslinks:
      self.assertIsInstance(link_set, dict)

  def test_add_xlinks(self):
    fold_input = folding_input.Input.from_json(self._test_input_json)
    fold_input = fold_input.expand_links()
    self.assertEqual(len(fold_input.bonded_atom_pairs), len(self.expected_bonded_atom_pairs))
    self.assertEqual(
      fold_input.bonded_atom_pairs,
      self.expected_bonded_atom_pairs
    )
    self.assertTrue(fold_input.user_ccd)
    for expected_ligand, ligand in zip(self.expected_ligands, fold_input.ligands):
      self.assertEqual(expected_ligand["id"], ligand.id)
      self.assertEqual(expected_ligand["ccd_ids"], ligand.ccd_ids)

    output_dir = self.create_tempdir()
    run_alphafold.write_fold_input_json(fold_input, output_dir)
    with open(
        os.path.join(output_dir, f'{fold_input.sanitised_name()}_data.json'),
        'rt',
    ) as f:
      out_json = json.load(f)

  def test_expand_links_tracks_crosslinker_chain_ids(self):
    fold_input = folding_input.Input.from_json(self._test_input_json)
    fold_input = fold_input.expand_links()

    ligand_ids = frozenset(ligand.id for ligand in fold_input.ligands)
    self.assertEqual(fold_input.crosslinker_chain_ids, ligand_ids)
    self.assertTrue(all(ligand.is_crosslinker for ligand in fold_input.ligands))


class TokenFeaturesCompatibilityTest(absltest.TestCase):

  def test_from_data_dict_defaults_is_crosslinker_to_false(self):
    batch = {
        'residue_index': np.array([1, 2], dtype=np.int32),
        'token_index': np.array([1, 2], dtype=np.int32),
        'aatype': np.array([0, 0], dtype=np.int32),
        'seq_mask': np.array([True, True], dtype=bool),
        'entity_id': np.array([1, 1], dtype=np.int32),
        'asym_id': np.array([1, 1], dtype=np.int32),
        'sym_id': np.array([1, 1], dtype=np.int32),
        'seq_length': np.array(2, dtype=np.int32),
        'is_protein': np.array([True, True], dtype=bool),
        'is_rna': np.array([False, False], dtype=bool),
        'is_dna': np.array([False, False], dtype=bool),
        'is_ligand': np.array([False, False], dtype=bool),
        'is_nonstandard_polymer_chain': np.array([False, False], dtype=bool),
        'is_water': np.array([False, False], dtype=bool),
    }

    token_features = features.TokenFeatures.from_data_dict(batch)

    np.testing.assert_array_equal(
        np.asarray(token_features.is_crosslinker),
        np.array([False, False], dtype=bool),
    )

class CrosslinkDataPipelineTest4G3Y(CrosslinkDataPipelineTest):
  """Test AlphaFold 3 inference."""

  def setUp(self):
    super().setUp()
    fn = testing_data.Data(
        resources.ROOT
        / 'test_data/crosslinks/4G3Y/4g3y_input.json').path()
    with open(fn, 'r') as f:
      self._test_input_json = f.read()

      self.expected_bonded_atom_pairs_len = 24

  def test_add_xlinks(self):
    fold_input = folding_input.Input.from_json(self._test_input_json)
    fold_input_expanded = fold_input.expand_links()

    self.assertEqual(len(fold_input_expanded.bonded_atom_pairs), self.expected_bonded_atom_pairs_len)

    fold_input_noverlap = fold_input.remove_overlapping_crosslinks()
    self.assertEqual(len(fold_input_noverlap.crosslinks[0]["residue_pairs"]), 4)

    sampled_crosslinks = fold_input.sample_crosslink_combinations(1)
    self.assertEqual(len(sampled_crosslinks), 12)
    for sample in sampled_crosslinks:
      self.assertEqual(len(sample.crosslinks[0]["residue_pairs"]), 1)
      self._check_crosslinks(sample)
    self.assertTrue(
      any("4G3Y_DSSO_B208-C129" in sample.name for sample in sampled_crosslinks)
    )

    sampled_crosslinks = fold_input.sample_crosslink_combinations(2)
    self.assertEqual(len(sampled_crosslinks), 66)
    for sample in sampled_crosslinks:
      self.assertEqual(len(sample.crosslinks[0]["residue_pairs"]), 2)
      self._check_crosslinks(sample)
    self.assertTrue(
      any("4G3Y_DSSO_B208-C129_DSSO_A145-C129" in sample.name for sample in sampled_crosslinks)
    )

    sampled_crosslinks = fold_input.sample_crosslink_combinations(12)
    self.assertEqual(len(sampled_crosslinks), 1)

    with self.assertRaises(ValueError, msg="num_samples cannot be larger than the number of available crosslinks."):
        sampled_crosslinks = fold_input.sample_crosslink_combinations(13)

    sampled_crosslinks = fold_input.sample_crosslink_combinations(0)
    self.assertEqual(len(sampled_crosslinks), 1)

class DisulfideDataPipelineTest8WW0(DataPipelineTest):
  """Test AlphaFold 3 inference."""

  def setUp(self):
    super().setUp()
    fn = testing_data.Data(
        resources.ROOT
        / 'test_data/crosslinks/8WW0/8WW0_data_disulfide.json').path()
    with open(fn, 'r') as f:
      self._test_input_json = f.read()

      self.expected_bonded_atom_pairs_len = 8

  def compare_golden(self, result_path: str) -> None:
    raise NotImplementedError
  
  @absltest.skip('Skipping test becuase compare_golden is not implemented')
  def test_config(self):
    return super().test_config()

  @absltest.skip('Skipping test becuase compare_golden is not implemented')
  def test_featurisation(self):
    return super().test_featurisation()

  def test_add_disulfide_bonds_expand(self):
    fold_input = folding_input.Input.from_json(self._test_input_json)
    fold_input = fold_input.expand_links()
    chain = fold_input.protein_chains[0]

    self.assertEqual(chain.sequence, 'SAPGEANAHWELFAEEGRLATGYRHAVAPPSA')
    self.assertStartsWith(chain.unpaired_msa, '>query\nSAPGEANAHWELFAEEGRLATGYRHAVAPPSA')
    self.assertStartsWith(chain.paired_msa, '>query\nSAPGEANAHWELFAEEGRLATGYRHAVAPPSA')
    self.assertEqual(len(fold_input.bonded_atom_pairs), self.expected_bonded_atom_pairs_len)
    self.assertEqual(len(fold_input.ligands), 4)
    # self.assertIsNone(fold_input.disulfide_bonds)

  def test_add_disulfide_bonds_noexpand(self):
    fold_input = folding_input.Input.from_json(self._test_input_json)
    chain = fold_input.protein_chains[0]
    self.assertEqual(chain.sequence, 'SCPGECNCHWELFCEEGRLCTGYRHCVCPPSC')
    self.assertStartsWith(chain.unpaired_msa, '>query\nSCPGECNCHWELFCEEGRLCTGYRHCVCPPSC')
    self.assertStartsWith(chain.paired_msa, '>query\nSCPGECNCHWELFCEEGRLCTGYRHCVCPPSC')
    self.assertIsNone(fold_input.bonded_atom_pairs)
    self.assertEqual(len(fold_input.ligands), 0)
    # self.assertIsNotNone(fold_input.disulfide_bonds)

class CrosslinkErrorHandlingTest(absltest.TestCase):
  """Tests that expand_links raises clearly on invalid crosslink inputs."""

  # Sequence with LYS residues (required by NHS-ester crosslinkers like DSSO).
  # Position 5 = K (LYS), position 81 = K (LYS).
  _LYS_SEQUENCE = (
      'SEFEKLRQTGDELVQAFQRLREIFDKGDDDSLEQVLEEIEELIQKHRQLFDNRQEAADT'
      'EAAKQGDQWVQLFQRFREAIDKGDKDSLEQLLEELEQALQKIRELAEKKN'
  )
  # All-Ala sequence (10 aa): contains no LYS, SER, or THR.
  _ALA_SEQUENCE = 'AAAAAAAAAA'

  def _make_json(self, seq_a, seq_b, crosslinks):
    return json.dumps({
        'name': 'test',
        'modelSeeds': [1],
        'sequences': [
            {'protein': {'id': 'A', 'sequence': seq_a, 'modifications': [],
                         'unpairedMsa': None, 'pairedMsa': None}},
            {'protein': {'id': 'B', 'sequence': seq_b, 'modifications': [],
                         'unpairedMsa': None, 'pairedMsa': None}},
        ],
        'crosslinks': crosslinks,
        'dialect': folding_input.JSON_DIALECT,
        'version': folding_input.JSON_VERSION,
    })

  def test_invalid_crosslinker_name_raises(self):
    """expand_links raises ValueError for an unrecognised crosslinker name."""
    json_str = self._make_json(
        self._LYS_SEQUENCE, self._LYS_SEQUENCE,
        [{'name': 'NONEXISTENT_XL', 'residue_pairs': [(('A', 5), ('B', 5))]}],
    )
    fold_input = folding_input.Input.from_json(json_str)
    with self.assertRaises(ValueError):
      fold_input.expand_links()

  def test_wrong_residue_type_raises(self):
    """expand_links raises ValueError when the residue type is incompatible with the crosslinker.

    DSSO targets LYS, SER, THR, TYR, or the N-terminus (NTER at position 1).
    An all-Ala sequence at position 2 (not the N-terminus) has no valid attachment site.
    """
    json_str = self._make_json(
        self._ALA_SEQUENCE, self._ALA_SEQUENCE,
        [{'name': 'DSSO', 'residue_pairs': [(('A', 2), ('B', 2))]}],
    )
    fold_input = folding_input.Input.from_json(json_str)
    with self.assertRaises(ValueError):
      fold_input.expand_links()

  def test_residue_out_of_range_raises(self):
    """expand_links raises when a residue index exceeds the chain length."""
    # _ALA_SEQUENCE is 10 residues; resid 11 is out of range.
    json_str = self._make_json(
        self._ALA_SEQUENCE, self._ALA_SEQUENCE,
        [{'name': 'DSSO', 'residue_pairs': [(('A', 11), ('B', 1))]}],
    )
    fold_input = folding_input.Input.from_json(json_str)
    with self.assertRaises((ValueError, IndexError)):
      fold_input.expand_links()

  def test_no_crosslinks_expand_links_is_noop(self):
    """expand_links with no crosslinks returns the input unchanged (no new ligands or bonds)."""
    json_str = self._make_json(self._LYS_SEQUENCE, self._LYS_SEQUENCE, [])
    fold_input = folding_input.Input.from_json(json_str)
    expanded = fold_input.expand_links()
    # Bug was: UnboundLocalError because user_ccd/bonded_atom_pairs were only
    # set inside the `if all_links:` block but referenced unconditionally in the return.
    self.assertIsNone(expanded.bonded_atom_pairs)
    self.assertEqual(len(expanded.ligands), 0)
    self.assertEqual(len(expanded.protein_chains), 2)


# Minimal valid user CCD entry for a two-atom ligand.
_USER_CCD = """data_MYL
#
_chem_comp.formula                 "C H4 O"
_chem_comp.formula_weight          32.04
_chem_comp.id                      MYL
_chem_comp.mon_nstd_parent_comp_id ?
_chem_comp.name                    "Test ligand"
_chem_comp.pdbx_synonyms           ?
_chem_comp.type                    non-polymer
#
loop_
_chem_comp_atom.atom_id
_chem_comp_atom.charge
_chem_comp_atom.comp_id
_chem_comp_atom.pdbx_model_Cartn_x_ideal
_chem_comp_atom.pdbx_model_Cartn_y_ideal
_chem_comp_atom.pdbx_model_Cartn_z_ideal
_chem_comp_atom.type_symbol
C1 0 MYL 0.000 0.000 0.000 C
O1 0 MYL 1.400 0.000 0.000 O
#
loop_
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.comp_id
_chem_comp_bond.pdbx_aromatic_flag
_chem_comp_bond.value_order
C1 O1 MYL N SING
#
"""


class CrosslinkWithBondedAtomPairsTest(absltest.TestCase):
  """expand_links must keep user-supplied bonds when adding XL bonds.

  __post_init__ coerces bonded_atom_pairs to a tuple, so expand_links has to
  copy it into a list before extending. Inputs carrying a userCCD hit this first
  because naming atoms in a userCCD is what makes them bondable, but the trigger
  is bondedAtomPairs, not the userCCD.
  """

  _SEQUENCE = CrosslinkErrorHandlingTest._LYS_SEQUENCE

  def _make_input(self, *, user_ccd=None, ligand=None, bonds=None,
                  crosslinks=None, disulfide_bonds=None):
    sequences = [
        {'protein': {'id': 'A', 'sequence': self._SEQUENCE, 'modifications': [],
                     'unpairedMsa': None, 'pairedMsa': None}},
        {'protein': {'id': 'B', 'sequence': self._SEQUENCE, 'modifications': [],
                     'unpairedMsa': None, 'pairedMsa': None}},
    ]
    if ligand is not None:
      sequences.append({'ligand': ligand})
    raw = {
        'name': 'test',
        'modelSeeds': [1],
        'sequences': sequences,
        'dialect': folding_input.JSON_DIALECT,
        'version': folding_input.JSON_VERSION,
    }
    if user_ccd is not None:
      raw['userCCD'] = user_ccd
    if bonds is not None:
      raw['bondedAtomPairs'] = bonds
    if crosslinks is not None:
      raw['crosslinks'] = crosslinks
    if disulfide_bonds is not None:
      raw['disulfide_bonds'] = disulfide_bonds
    return folding_input.Input.from_json(json.dumps(raw))

  _DSSO = [{'name': 'DSSO', 'residue_pairs': [(('A', 5), ('B', 5))]}]
  _USER_BOND = [(('A', 1, 'CB'), ('L', 1, 'C1'))]
  _LIGAND = {'id': 'L', 'ccdCodes': ['MYL']}

  def test_user_ccd_bond_survives_crosslink_expansion(self):
    """A userCCD ligand bond and the XL bonds must coexist."""
    fold_input = self._make_input(
        user_ccd=_USER_CCD, ligand=self._LIGAND,
        bonds=self._USER_BOND, crosslinks=self._DSSO,
    )
    self.assertIsInstance(fold_input.bonded_atom_pairs, tuple)

    expanded = fold_input.expand_links()

    # One user bond plus two XL bonds (one per crosslinked residue).
    self.assertLen(expanded.bonded_atom_pairs, 3)
    self.assertIn((('A', 1, 'CB'), ('L', 1, 'C1')), expanded.bonded_atom_pairs)
    self.assertEqual(expanded.crosslinker_chain_ids, frozenset({'C'}))
    # The user CCD entry must survive alongside the appended crosslinker entry.
    ccd = chemical_components.cached_ccd(user_ccd=expanded.user_ccd)
    self.assertIn('MYL', ccd)
    self.assertIn('DSSO', ccd)

  def test_bonded_atom_pairs_without_user_ccd(self):
    """The trigger is bondedAtomPairs, not the userCCD."""
    expanded = self._make_input(
        ligand={'id': 'L', 'ccdCodes': ['ATP']},
        bonds=[(('A', 1, 'CB'), ('L', 1, 'PA'))],
        crosslinks=self._DSSO,
    ).expand_links()

    self.assertLen(expanded.bonded_atom_pairs, 3)
    self.assertIn((('A', 1, 'CB'), ('L', 1, 'PA')), expanded.bonded_atom_pairs)

  def test_bonded_atom_pairs_with_disulfide_bonds(self):
    """Disulfide expansion goes through the same code path."""
    expanded = self._make_input(
        user_ccd=_USER_CCD, ligand=self._LIGAND, bonds=self._USER_BOND,
        disulfide_bonds=[{'residue_pairs': [(('A', 6), ('A', 14))]}],
    ).expand_links()

    self.assertIn((('A', 1, 'CB'), ('L', 1, 'C1')), expanded.bonded_atom_pairs)

  def test_expand_links_does_not_mutate_input(self):
    """expand_links must not extend the caller's bonded_atom_pairs in place."""
    fold_input = self._make_input(
        user_ccd=_USER_CCD, ligand=self._LIGAND,
        bonds=self._USER_BOND, crosslinks=self._DSSO,
    )
    fold_input.expand_links()
    self.assertLen(fold_input.bonded_atom_pairs, 1)


class CrosslinkDataJsonReuseTest(absltest.TestCase):
  """The written _data.json must stay reusable as input for another run.

  process_fold_input writes the JSON before expanding, so expansion products
  (crosslinker ligands, XL bonds, appended CCD entries) are never serialised.
  Re-running on the written file must reproduce the same expansion.
  """

  def setUp(self):
    super().setUp()
    fn = testing_data.Data(
        resources.ROOT / 'test_data/crosslinks/9G5K/9G5K_input.json').path()
    with open(fn, 'r') as f:
      self._fold_input = folding_input.Input.from_json(f.read())

  def _written_json(self) -> str:
    output_dir = self.create_tempdir().full_path
    run_alphafold.process_fold_input(
        fold_input=self._fold_input,
        data_pipeline_config=None,
        model_runner=None,
        output_dir=output_dir,
    )
    path = os.path.join(
        output_dir, f'{self._fold_input.sanitised_name()}_data.json')
    with open(path, 'r') as f:
      return f.read()

  def test_written_json_matches_input(self):
    self.assertEqual(self._written_json(), self._fold_input.to_json())

  def test_written_json_has_no_expansion_products(self):
    written = json.loads(self._written_json())
    self.assertIsNone(written['bondedAtomPairs'])
    self.assertIsNone(written['userCCD'])
    self.assertEmpty([s for s in written['sequences'] if 'ligand' in s])
    self.assertEqual(
        [xl['name'] for xl in written['crosslinks']], ['azide-A-DSBSO'])

  def test_reused_json_expands_identically(self):
    reused = folding_input.Input.from_json(self._written_json())
    self.assertEqual(
        reused.expand_links().to_json(),
        self._fold_input.expand_links().to_json(),
    )


if __name__ == '__main__':
  absltest.main()
