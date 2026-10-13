"""AF3x code version stamp for run outputs.

Writes: <output_dir>/<job_name>_af3x_version.json (keys: see get_version_info).
Git fields are None when the package is not run from a git checkout.
AF3_UPSTREAM_COMMIT: upstream AlphaFold3 commit last merged; update at each sync.
"""

from collections.abc import Mapping
import datetime
import functools
import hashlib
import importlib.metadata
import json
import os
import pathlib
import platform
import subprocess
from typing import Any

from alphafold3 import version
from alphafold3.constants import chemical_components


AF3_UPSTREAM_COMMIT = '87bd9e678d9acacc4aa9baa05e820f32b80e1b49'

_PACKAGES = (
    'jax',
    'jaxlib',
    'jax-cuda12-plugin',
    'tokamax',
    'dm-haiku',
    'rdkit',
    'numpy',
)

# src/alphafold3/common/af3x_version.py -> repository root.
_SOURCE_ROOT = pathlib.Path(__file__).resolve().parents[3]


def _git(*args: str) -> str | None:
  try:
    result = subprocess.run(
        ['git', '-C', str(_SOURCE_ROOT), *args],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
  except (OSError, subprocess.SubprocessError):
    return None
  return result.stdout.strip()


def _package_version(name: str) -> str | None:
  try:
    return importlib.metadata.version(name)
  except importlib.metadata.PackageNotFoundError:
    return None


def _ccd_file() -> str | None:
  # Pickle before the AF3 sync, MessagePack after it.
  for attr in ('_CCD_MSGPACK_FILE', '_CCD_PICKLE_FILE'):
    path = getattr(chemical_components, attr, None)
    if path is not None:
      return str(path)
  return None


def _sha256(path: str) -> str | None:
  digest = hashlib.sha256()
  try:
    with open(path, 'rb') as f:
      for chunk in iter(lambda: f.read(1 << 24), b''):
        digest.update(chunk)
  except OSError:
    return None
  return digest.hexdigest()


@functools.cache
def get_version_info() -> Mapping[str, Any]:
  """Returns code, package and CCD versions of the running AF3x."""
  toplevel = _git('rev-parse', '--show-toplevel')
  in_checkout = toplevel is not None and pathlib.Path(toplevel) == _SOURCE_ROOT
  commit = _git('rev-parse', 'HEAD') if in_checkout else None
  status = (
      _git('status', '--porcelain', '--untracked-files=no')
      if in_checkout
      else None
  )
  branch = _git('rev-parse', '--abbrev-ref', 'HEAD') if in_checkout else None
  ccd_file = _ccd_file()
  return {
      'af3x_commit': commit,
      'af3x_dirty': None if status is None else bool(status),
      'af3x_branch': branch,
      'source_dir': str(_SOURCE_ROOT),
      'af3_upstream_commit': AF3_UPSTREAM_COMMIT,
      'alphafold3_version': version.__version__,
      'python': platform.python_version(),
      'packages': {name: _package_version(name) for name in _PACKAGES},
      'ccd_file': ccd_file,
      'ccd_sha256': _sha256(ccd_file) if ccd_file else None,
  }


def short_id() -> str:
  """Returns '<12-char commit>[-dirty]', or 'unknown' outside a git checkout."""
  info = get_version_info()
  if info['af3x_commit'] is None:
    return 'unknown'
  return info['af3x_commit'][:12] + ('-dirty' if info['af3x_dirty'] else '')


def write_version_info(output_dir: os.PathLike[str] | str, job_name: str) -> str:
  """Writes <job_name>_af3x_version.json into output_dir; returns its path."""
  info = dict(get_version_info())
  info['written_at'] = datetime.datetime.now().isoformat(timespec='seconds')
  path = os.path.join(output_dir, f'{job_name}_af3x_version.json')
  with open(path, 'wt') as f:
    json.dump(info, f, indent=2)
  return path
