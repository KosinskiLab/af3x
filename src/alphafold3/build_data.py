# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Script for building intermediate data."""

from importlib import resources
import os
import pathlib
import site

import alphafold3.constants.converters
from alphafold3.constants.converters import ccd_pickle_gen
from alphafold3.constants.converters import chemical_component_sets_gen


def build_data():
  """Builds intermediate data."""
  libcifpp_data_dir = os.environ.get('LIBCIFPP_DATA_DIR')
  if libcifpp_data_dir:
    cif_path = pathlib.Path(libcifpp_data_dir) / 'components.cif'
  else:
    for site_path in site.getsitepackages():
      path = pathlib.Path(site_path) / 'share/libcifpp/components.cif'
      if path.exists():
        cif_path = path
        break
    else:
      raise ValueError(
          'Could not find components.cif. If libcifpp is installed in a'
          ' non-standard location, please set the LIBCIFPP_DATA_DIR environment'
          ' variable to the directory where libcifpp is installed.'
      )

  out_root = resources.files(alphafold3.constants.converters)
  ccd_pickle_path = out_root.joinpath('ccd.pickle')
  chemical_component_sets_pickle_path = out_root.joinpath(
      'chemical_component_sets.pickle'
  )
  ccd_pickle_gen.main(['', str(cif_path), str(ccd_pickle_path)])
  chemical_component_sets_gen.main(
      ['', str(chemical_component_sets_pickle_path)]
  )
