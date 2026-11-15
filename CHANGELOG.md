# Changelog

## 0.1.0

- Restructured the Colab export into an installable `smiles_vae` package under `src/`.
- Added an explicit pad token, so padding no longer produces all-zero targets.
- Added seeding, a CLI, RDKit validity reporting, tests and CI.
- Added a demo notebook and kept the original Colab script under `legacy/`.
- Added the MIT license.
