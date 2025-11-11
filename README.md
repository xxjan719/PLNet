# PLNet: Persistent Laplacian Neural Network for Protein-Protein Binding Free Energy Prediction

## Description
```
This manual is for the code implementation of paper "PLNet: Persistent Laplacian Neural Network for Protein-Protein Binding Free Energy Prediction"
```

## Comments
- Primary dataset: place `P2P.csv` in the `data/` directory (i.e. `data/P2P.csv`).
- The repository includes tooling and an optional virtual environment under `env/`; add generated environments and other local artifacts to `.gitignore` so they are not committed.
- Start by running `src/config.py` — setup and configuration instructions will be printed to the terminal.
- The `bin/` directory holds external executables (e.g. JACKAL, SCAP, ProFix). You can populate `bin/` yourself following the instructions printed by `src/config.py`, or request prebuilt binaries from xingjianxu@ufl.edu.
- All datasets support feature extraction including ESM embeddings, biophysical properties, and pairwise similarity matrices

## Dependencies
- fair-esm (2.0.0): For protein language modeling
- torch (2.0.0): For deep learning operations
- biopython (1.79): For sequence analysis
- numpy (1.21.0): For numerical computations
- pandas (1.3.0): For data manipulation
- requests (2.26.0): For PDB file downloading
- tqdm (4.62.0): For progress tracking
- ProFix: For PDB file processing
- Jackal: For protein structure analysis
- **SCAP**: Side Chain Analysis Program for protein mutation generation (must be installed in system PATH)

## Usage

### Install Environment and package
```python
python src/config.py
```
Then redo again for install package.

### Data Generation for P2P
```python
python src/data_preparation.py  # Prepare all P2P and generate the feature for ESM and Persistent Homology
```



