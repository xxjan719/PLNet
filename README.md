# Protein-Protein Interaction Binding Affinity

## Description
```bash
PL_PPI_BA_prediction/
│
├── bin/                           # Executables and binary files
│   ├── profix                     # ProFix executable for PDB structure processing
│   ├── test_profix.py             # Test script for ProFix functionality
│   ├── test_mutation_structure.py # Test script for mutation structure generation
│   ├── test_mutation_structure_standalone.py # Standalone mutation testing
│   ├── jackal_64bit/              # Jackal executable directory
│   ├── jackal_64bit.tar.gz        # Jackal archive
│   └── jackal.dir                 # Jackal configuration
│
├── data/                          # Dataset files and directories
│   ├── P2P.csv                    # Main protein-protein interaction dataset
│   ├── PDBbind_v2020_PP_anti.csv  # PDBbind v2020 processed data
│   ├── skempi_v2                  # SKEMPI v2 dataset reference
│   ├── skempi_v2_wt_anti.csv      # SKEMPI v2 wild-type processed data
│   └── Example/                   # Example PDB files for testing
│
├── src/                           # Source code
│   ├── data_generation.py         # Main data processing pipeline
│   ├── machine_learning.py        # Machine learning models and training
│   └── utils/                     # Utility modules
│       ├── __init__.py            # Package initialization
│       ├── constant.py            # Constants and configuration
│       ├── feature_parser.py      # Protein feature extraction (ESM, biophysics)
│       ├── helper.py              # General utility functions
│       ├── mutation_structure.py  # Protein mutation structure generation
│       ├── pairwise_parser.py     # Protein sequence comparison and similarity
│       ├── persistent.py          # Persistent homology calculations
│       ├── plot.py                # Visualization and plotting functions
│       ├── table_generation.py    # Dataset table generation utilities
│       └── pyprotein/             # C++ extensions for protein analysis
│           ├── prot.cpp           # C++ source code
│           └── prot.cpython-*.so  # Compiled extensions
│
├── config/                        # Configuration files
│   ├── argparser.py               # Command-line argument parsing
│   ├── packages.py                # Package dependencies and management
│   └── path.py                    # Path configurations
│
├── results/                       # Output directory for results
└── .gitignore                     # Git ignore rules
```

**Note:**
- The project supports three main datasets: PDBbind v2020, SKEMPI v2 wild-type (WT), and SKEMPI v2 mutant (MT)
- Currently, only the WT dataset has been fully processed with ProFix and is available in `data/WT/`
- V2020 and MT datasets require additional processing to generate ProFix-processed structures
- The system includes comprehensive mutation structure generation capabilities using SCAP
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

### Mutation Structure Generation
```python
# Single mutation generation
from src.utils.mutation_structure import generate_mutation_structure

# Generate a mutation
mt_path = generate_mutation_structure(
    wt_pdb_path="wild_type.pdb",
    output_base_name="mutation_output",
    mutation_chain="A",
    mutation_residue_id="83",
    mutation_residue="A"  # Alanine
)

# Generate mutation from string format "PDBID_ChainID_WildResidue_ResidueID_MutateResidue"
from src.utils.mutation_structure import generate_mutation_structure_from_string, parse_mutation_string

# Parse mutation string (e.g., "1JTG_A_N_100_A" means PDB 1JTG, Chain A, Wild residue N, Residue ID 100, Mutate to A)
mutation_info = parse_mutation_string("1JTG_A_N_100_A")
print(mutation_info)  # {'pdb_id': '1JTG', 'chain_id': 'A', 'wild_residue': 'N', ...}

# Generate mutation directly from string
mt_path = generate_mutation_structure_from_string(
    mutation_string="1JTG_A_N_100_A",
    wt_pdb_path="1JTG.pdb",
    output_base_name="mutation_output"
)

# Batch mutation generation
from src.utils.mutation_structure import MutationStructureGenerator

generator = MutationStructureGenerator()

# Batch from individual parameters
mutations = [
    {
        'wt_pdb_path': 'protein1.pdb',
        'output_base_name': 'mut1',
        'mutation_chain': 'A',
        'mutation_residue_id': '83',
        'mutation_residue': 'A'
    },
    # ... more mutations
]
results = generator.batch_generate_mutations(mutations)

# Batch from mutation strings
mutation_strings = ["1JTG_A_N_100_A", "2ABC_B_L_45_G"]
wt_pdb_paths = ["1JTG.pdb", "2ABC.pdb"]
output_names = ["mut1", "mut2"]
results = generator.batch_generate_mutations_from_strings(
    mutation_strings, wt_pdb_paths, output_names
)

# Test mutation generation
python bin/test_mutation_structure.py --test deps  # Test dependencies
python bin/test_mutation_structure.py --test parse  # Test string parsing
python bin/test_mutation_structure.py --test single --wt-pdb your_protein.pdb  # Test single mutation
python bin/test_mutation_structure.py --test string --wt-pdb your_protein.pdb --mutation-string "1JTG_A_N_100_A"  # Test string mutation
```

### Data Processing with ProFix
```python
# Data Processing with ProFix
from src.utils.profix_handler import ProFixHandler

# Process a single PDB file
success = ProFixHandler.process_pdb("path/to/file.pdb")

# Batch process multiple files
results = ProFixHandler.batch_process_pdbs(["file1.pdb", "file2.pdb"])

# Validate processed files
is_valid, message = ProFixHandler.validate_pdb("processed_file.pdb")

# Data Generation and 10-Fold Pairwise Matrix
python src/data_generation.py  # Process all datasets and generate pairwise matrices/plots
```

## Comments
- Variable names: Follow Python naming conventions
- Code structure: Modular design with clear separation of concerns
- Documentation: Comprehensive docstrings and comments
- Path handling: Use centralized path configuration
- Binary management: Keep executables in bin directory
- Error handling: Comprehensive error checking and reporting
- **External tools**: SCAP and ProFix must be installed and available in system PATH
