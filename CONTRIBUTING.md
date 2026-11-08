# Contributing to AF3x

## Questions and bug reports

Open an issue at [KosinskiLab/af3x](https://github.com/KosinskiLab/af3x/issues). Include the AF3x version (the
`<job>_af3x_version.json` file of the run, or `git describe --tags --match 'af3x-v*'`), the input JSON and the error message.

If the problem also occurs with plain AlphaFold 3 (no crosslinks), report it to
[AlphaFold 3](https://github.com/google-deepmind/alphafold3/issues) instead.

## Pull requests

Pull requests with bug fixes, new crosslinkers and documentation improvements are welcome. Open them against the
`experimental` branch, with a short description of the change and how you tested it:

- run the CPU tests: `python run_alphafold_data_test.py` and
  `python -m pytest --ignore=run_alphafold_test.py` (do not pass `-v`: absl reads it as its own flag);
- for a new or changed crosslinker, include an example input JSON that uses it.

## AI-generated code

AI tools may be used for code, documentation and pull request descriptions, as long as:

1. it is declared in the pull request description;
2. you have reviewed the code yourself;
3. you have tested the change. We may ask you to run a prediction to check it.

Do not submit pull requests with invented test results.

## Licence

AF3x is licensed under the [Apache License 2.0](LICENSE). By submitting a contribution you agree that it is
licensed under the same terms (section 5 of the licence). No separate contributor agreement is needed.
