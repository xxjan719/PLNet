# PLNet: Persistent Laplacian Neural Network for Protein-Protein Binding Free Energy Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

This repository contains the code implementation for the paper **"PLNet: Persistent Laplacian Neural Network for Protein-Protein Binding Free Energy Prediction"**.

PLNet is a machine learning framework that predicts protein-protein binding free energy using persistent Laplacian features, ESM embeddings, and biophysical properties.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation-and-setup)
- [Usage](#usage)
- [Data Preparation](#data-generation-for-p2p)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

## Features

- **Multiple Feature Types**: Supports ESM embeddings, biophysical properties, and pairwise similarity matrices
- **Multiple Datasets**: Supports V2020, WT, MT, SKempi_V2, P2P, and Antibody-Antigen datasets
- **Automated Setup**: Automated environment and package management via `config.py`
- **Persistent Homology**: Integration with GUDHI for topological data analysis
- **Flexible Models**: Gradient Boosting Decision Trees (GBDT) with cross-validation

## Requirements

### System Requirements
- Python 3.8 or higher
- Linux/macOS/Windows
- At least 4GB RAM (8GB+ recommended for large datasets)

### Prerequisites
- **Primary dataset**: `P2P.csv` must be placed in the `data/` directory (i.e. `data/P2P.csv`)
- **Virtual environment**: The repository includes tooling and an optional virtual environment under `env/`; add generated environments and other local artifacts to `.gitignore` so they are not committed
- **External tools**: The `bin/` directory holds external executables (e.g. JACKAL, SCAP, ProFix). You can populate `bin/` yourself following the instructions printed by `src/config.py`, or request prebuilt binaries from xingjianxu@ufl.edu
- **Setup script**: Start by running `src/config.py` — setup and configuration instructions will be printed to the terminal

## Dependencies

### Python Packages
The following Python packages are automatically checked and installed by `config.py`:
- fair-esm (2.0.0): For protein language modeling
- torch (2.0.0): For deep learning operations
- biopython (1.79): For sequence analysis
- numpy (1.21.0): For numerical computations
- requests (2.26.0): For PDB file downloading
- tqdm (4.62.0): For progress tracking
- gudhi (3.7.1): Topological data analysis library
- scikit-learn (1.0.2): Machine learning library
- matplotlib (3.4.3): Plotting library
- seaborn (0.11.2): Statistical data visualization
- scipy (1.7.1): Scientific computing
- cython (0.29.0): Cython for C++ extensions
- pdb2pqr (3.6.1): PDB2PQR tool

### External Tools
- **ProFix**: For PDB file processing (part of JACKAL)
- **SCAP**: Side Chain Analysis Program for protein mutation generation (part of JACKAL)
- **JACKAL**: Protein structure analysis suite

## Usage

### Installation and Setup

The `config.py` script automates the setup process. Run it to:

1. **Check/Create Virtual Environment**
   - If not in a virtual environment, the script will prompt you to create one
   - A virtual environment named `env/` will be created in the project root
   - You'll need to activate it before continuing:
     ```bash
     # On Linux/macOS:
     source env/bin/activate
     
     # On Windows:
     env\Scripts\activate
     ```

2. **Install Python Packages**
   - After activating the virtual environment, run `config.py` again
   - The script will automatically check and install all required Python packages
   - Missing packages will be installed, outdated packages will be upgraded

3. **Check External Tools**
   - The script verifies if JACKAL tools (profix, scap) are available
   - Tools can be installed locally in `bin/` directory or globally in system PATH
   - If missing, instructions for installation will be provided

4. **Compile Cython Extensions**
   - Automatically compiles Cython extensions if available

5. **Generate Data Tables**
   - Reads `P2P.csv` from `data/` directory
   - Generates dataset-specific tables for V2020, WT, and MT datasets

**Complete Installation Steps:**

```bash
# Step 1: Run config.py (will prompt for virtual environment creation)
python src/config.py

# Step 2: Activate the virtual environment (if created)
source env/bin/activate  # Linux/macOS
# OR
env\Scripts\activate     # Windows

# Step 3: Run config.py again to install packages
python src/config.py

# Step 4: Verify installation completed successfully
# The script will report status of all components
```

**Note:** If a virtual environment is created in step 1, you must activate it and run `python src/config.py` again to complete package installation. The script will exit after creating the environment to allow you to activate it first.

### Data Generation for P2P

After installation, prepare your data and generate features:

```bash
python src/data_preparation.py  # Prepare all P2P and generate features for ESM and Persistent Homology
```

This script will:
- Process PDB files from the dataset directories
- Generate ESM embeddings using fair-esm
- Compute biophysical properties
- Calculate persistent homology features
- Create feature tables for model training

### Model Training and Prediction

Train models using the machine learning pipeline:

```bash
python src/machine_learning.py
```

The script will prompt you to:
1. Select a dataset (V2020, WT, MT, V2020+WT, V2020+MT, SKempi_V2, P2P, or Antibody-Antigen)
2. Select a feature method (biophysics, esm, PPI, or All)
3. Run cross-validation and generate predictions

Results will be saved to log files in the `results/` directory.

## Troubleshooting

### Common Issues

**Issue: Virtual environment not activating**
- **Solution**: Make sure you're using the correct activation command for your operating system
- Linux/macOS: `source env/bin/activate`
- Windows: `env\Scripts\activate`

**Issue: Missing external tools (JACKAL, SCAP, ProFix)**
- **Solution**: Follow the instructions provided by `config.py` or contact xingjianxu@ufl.edu for prebuilt binaries
- Tools can be installed locally in `bin/` directory or globally in system PATH

**Issue: Package installation fails**
- **Solution**: Ensure you're in an activated virtual environment and have internet connectivity
- Try upgrading pip: `pip install --upgrade pip`
- Some packages may require system-level dependencies (e.g., C++ compilers for Cython)

**Issue: P2P.csv not found**
- **Solution**: Ensure `P2P.csv` is placed in the `data/` directory before running `config.py`
- The file should contain columns: partnerA, partnerB, source, suffix, ID, DDG

**Issue: Cython compilation errors**
- **Solution**: Ensure you have a C++ compiler installed (gcc/clang on Linux/macOS, Visual Studio on Windows)
- Cython extensions are optional; the code will work without them but may be slower

## Citation

If you use PLNet in your research, please cite:

```bibtex
@article{plnet2025,
  title={PLNet: Persistent Laplacian Neural Network for Protein-Protein Binding Free Energy Prediction},
  author={Xu, Xingjian},
  journal={[Journal Name]},
  year={2025}
}
```

*Note: Please update with the actual citation once the paper is published.*

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2025 Xingjian Xu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

## Contact

For questions, issues, or requests for prebuilt binaries:
- **Email**: xingjianxu@ufl.edu
- **Issues**: Please use the GitHub Issues page for bug reports and feature requests

## Acknowledgments

- JACKAL tools (ProFix, SCAP) for protein structure analysis
- ESM (Evolutionary Scale Modeling) for protein language modeling
- GUDHI for persistent homology computations
- All contributors and users of this project

