# Drug Discovery: SMILES VAE

[![CI](https://github.com/bhargavmanchala/Drug_Discovery/actions/workflows/ci.yml/badge.svg)](https://github.com/bhargavmanchala/Drug_Discovery/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)

A small convolutional **variational autoencoder (VAE)** that learns from
[SMILES](https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system)
strings and generates new candidate molecules. Generated strings are checked
and profiled with [RDKit](https://www.rdkit.org/).

This is an educational demo of the generative-chemistry workflow, not a
production drug discovery system.

## Table of contents

- [How it works](#how-it-works)
- [Project structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Notebooks](#notebooks)
- [Results and limitations](#results-and-limitations)
- [Development](#development)
- [Roadmap](#roadmap)
- [License](#license)

## How it works

1. **Encode**: each SMILES is one-hot encoded character by character and
   padded to the longest string using an explicit pad token.
2. **Encoder**: two `Conv1D` layers, a dense layer, then `z_mean` and
   `z_log_var` heads (default latent size 2).
3. **Sample**: the reparameterisation trick, `z = mean + exp(0.5 * log_var) * eps`.
4. **Decoder**: dense layers, then three `Conv1DTranspose` layers with a softmax
   over the character vocabulary.
5. **Loss**: categorical cross-entropy plus a KL divergence term scaled by
   `kl_weight` (default 0.1).
6. **Generate**: sample latent vectors from `N(0, I)`, decode, and cut each
   string at the first pad token.
7. **Analyse**: RDKit reports validity, molecular weight, LogP, and hydrogen
   bond donors and acceptors.

## Project structure

```
.
├── src/smiles_vae/        # installable package
│   ├── data.py            # loading, vocabulary, one-hot encode/decode
│   ├── model.py           # encoder, decoder, VAE
│   ├── pipeline.py        # train() and generate() helpers
│   ├── analysis.py        # RDKit validity and descriptors
│   └── cli.py             # command line entry point
├── tests/                 # pytest suite
├── data/sample_smiles.txt # five-molecule demo dataset
├── notebooks/             # demo and original walkthrough notebooks
├── docs/project_report.pdf
├── legacy/                # original Colab export, kept for provenance
├── .github/workflows/     # CI (ruff + pytest)
├── pyproject.toml
├── CHANGELOG.md
├── CONTRIBUTING.md
└── LICENSE
```

## Installation

```bash
git clone https://github.com/bhargavmanchala/Drug_Discovery.git
cd Drug_Discovery
python -m venv .venv
.venv\Scripts\activate        # macOS/Linux: source .venv/bin/activate
pip install -e .
```

For development tools (pytest, ruff, nbformat) use `pip install -e ".[dev]"`.

## Usage

Command line:

```bash
smiles-vae --data data/sample_smiles.txt --epochs 25 --num-samples 10 --plot loss.png
```

Equivalent: `python -m smiles_vae ...`. Run `smiles-vae --help` for all options
(`--latent-dim`, `--kl-weight`, `--batch-size`, `--learning-rate`, `--seed`).

Python API:

```python
from smiles_vae.data import DEFAULT_SMILES
from smiles_vae.pipeline import train, generate
from smiles_vae.analysis import describe, validity_rate

t = train(DEFAULT_SMILES, epochs=25, seed=42)
molecules = generate(t, n=10, seed=42)
print(molecules, validity_rate(molecules))
print(describe("CCO"))
```

Dataset format: a text file with one SMILES per line. Blank lines and lines
starting with `#` are ignored, as is any text after the first token.

## Notebooks

| Notebook | Purpose |
| --- | --- |
| [`notebooks/smiles_vae_demo.ipynb`](notebooks/smiles_vae_demo.ipynb) | Train and generate using the package |
| [`notebooks/drug_discovery_smiles_original.ipynb`](notebooks/drug_discovery_smiles_original.ipynb) | The original step-by-step walkthrough |

The original Colab notebook is also available
[here](https://colab.research.google.com/drive/1VCyw-wDqRv2e03lCjmeXkNhdD7By6lGW?usp=sharing).
The project report is in [`docs/project_report.pdf`](docs/project_report.pdf).

## Results and limitations

The original experiments used five training molecules (ethanol, acetic acid,
propane, ethylamine, diethyl ether). Observed behaviour:

| Epochs | Outcome |
| --- | --- |
| 25 | Short, mostly invalid strings such as `CCCCNNO` |
| 75 to 100 | Output collapses to a single repeated string |
| 250 | Severe overfitting, for example `OOOOOOO` |

With so little data the model cannot learn SMILES grammar, so validity is low.
For meaningful results use thousands of molecules (for example ZINC or ChEMBL
subsets) and consider grammar-aware representations such as SELFIES.

## Development

```bash
pip install -e ".[dev]"
ruff check .
pytest
```

CI runs the same two commands on Python 3.10 and 3.11. See
[CONTRIBUTING.md](CONTRIBUTING.md) and [CHANGELOG.md](CHANGELOG.md).

## Roadmap

- Larger datasets and a train/validation/test split
- SELFIES encoding to guarantee valid molecules
- Latent space visualisation and property-guided optimisation
- Model save and load

## License

Released under the [MIT License](LICENSE).
