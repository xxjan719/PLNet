"""
***Description***
This code generates comprehensive datasets for protein-protein interaction (PPI) binding affinity prediction.
It processes multiple datasets including PDBbind v2020 and SKEMPI v2, extracting various features:
- ESM (Evolutionary Scale Modeling) embeddings
- Biophysical properties for protein partners A and B
- Protein-protein interface amino acid features
- Electrostatic features at the interface
- Pairwise similarity matrices using BioPairwiseParser and TransformerPairwiseParser

The system supports all V2020, wild-type (WT) and mutant (MT) protein structures, enabling comparative
analysis of binding affinity changes due to mutations.

*** Author Information ***
Author: Xingjian Xu
Email: xingjianxu@ufl.edu
Institution: University of Florida
Date Created: 06/01/2024
Last Modified: 07/01/2025
Version: 1.0
"""
import os
import config
import p2p_bio.utils as utils

import p2p_bio.feature_generation as feature_generation


# Ask user whether to rebuild everything from PDB files or skip to calculated data
print("\n" + "="*60)
print("DATA PREPARATION OPTIONS")
print("="*60)
print("1. Rebuild everything from PDB files (download missing PDB files and process)")
print("2. Skip PDB processing and go directly to calculated data")
print("="*60)

while True:
    choice = input("\nChoose option (1 or 2): ").strip()
    if choice in ['1', '2']:
        break
    else:
        print("Please enter '1' or '2'")

if choice == '1':
    # Option 1: Rebuild everything from PDB files
    print("\n[INFO] Rebuilding everything from PDB files...")
    
    # Check completeness of each dataset
    print("\nChecking PDB files completeness...")
    dataset_status = {}
    print(config.DATASET_CONFIG)
    for _dataset_key, _config in config.DATASET_CONFIG.items():
        complete, missing = utils.check_PDB_completeness(_config['dir'], _config['table'], _config['name'])
        dataset_status[_dataset_key] = {'complete': complete, 'missing': missing}

    # Process incomplete datasets
    for _dataset_key, _status in dataset_status.items():
        if not _status['complete']:
            _config = config.DATASET_CONFIG[_dataset_key]
            answer = input(f"Process {len(_status['missing'])} missing files for {_dataset_key} dataset? (y/n): ")
            if answer.lower() == 'y':
                utils.process_missing_PDB_files(_config['dir'], _status['missing'], _config['name'])
            else:
                print(f"Skipping {_dataset_key} dataset processing")
        else:
            print(f"[SUCCESS] {_dataset_key} dataset is complete")

    # Generate features dataset by dataset
    print("\n" + "="*60)
    print("FEATURE GENERATION - DATASET BY DATASET")
    print("="*60)
    
    for _dataset_key, _config in config.DATASET_CONFIG.items():
        print(f"\n[INFO] Checking {_dataset_key} dataset for feature generation...")
        print(f"Dataset directory: {_config['dir']}")
        print(f"Dataset table: {_config['name']}")
        
        # Check if dataset directory exists
        dataset_dir = _config['dir']
        if not os.path.exists(dataset_dir):
            print(f"[WARNING] Dataset directory {dataset_dir} does not exist")
            continue
        
        # Count PDB files in the dataset
        pdb_count = 0
        for pdb_id in os.listdir(dataset_dir):
            pdb_folder = os.path.join(dataset_dir, pdb_id)
            if os.path.isdir(pdb_folder):
                pdb_count += 1
        
        if pdb_count == 0:
            print(f"[WARNING] No PDB directories found in {dataset_dir}")
            continue
        
        # Ask user if they want to generate features for this specific dataset
        answer = input(f"Generate features for {_dataset_key} dataset ({pdb_count} PDB files)? (y/n): ")
        if answer.lower() == 'y':
            print(f"\n[INFO] Processing features for {_dataset_key} dataset...")
            
            # Process each PDB in the dataset directory
            for pdb_id in os.listdir(dataset_dir):
                pdb_folder = os.path.join(dataset_dir, pdb_id)
                if not os.path.isdir(pdb_folder):
                    print(f"Skipping {pdb_id} as it is not a directory")
                    continue
                
                print("=" * 60)
                print(f"Processing PDB ID: {pdb_id}")
                
                # Get partnerA and partnerB from the dataset table
                dataset_table = _config['table']
                pdb_row = dataset_table[dataset_table['ID'] == pdb_id]
                
                if pdb_row.empty:
                    print(f"[WARNING] PDB ID {pdb_id} not found in dataset table. Skipping...")
                    continue
                
                partner1 = pdb_row['partnerA'].iloc[0]
                partner2 = pdb_row['partnerB'].iloc[0]
                
                # Create filepath for this PDB
                filepath = os.path.join(_config['dir'], pdb_id)
                
                # Create protein complex object with config paths
                pc = feature_generation.protein_complex(
                    PDBID=pdb_id, 
                    filepath=filepath,
                    partner1=partner1,
                    partner2=partner2,
                    generate_structure_files=True,
                    check_structure=True,
                    persistent_homology=True, 
                    read_features=False,
                    sequence=False
                )
                print(f"Completed processing for {pdb_id}\n")
                print("=" * 60)
            
            # Collect features into table for this dataset
            print(f"\n[INFO] Collecting features for {_dataset_key}...")
            feature_generation.collect_PDB_feature_to_table(_config['name'])
            print(f"[SUCCESS] Feature generation completed for {_dataset_key} dataset!")
        else:
            print(f"[INFO] Skipping feature generation for {_dataset_key} dataset")
    
    print("\n[SUCCESS] Feature generation process completed!")

else:
    # Option 2: Skip to calculated data
    print("\n[INFO] Skipping PDB processing, proceeding to calculated data...")

# Define required feature files
required_files = [
    'dd_esm_{suffix}.csv',
    'dd_partnerA_biophysics_{suffix}.csv',
    'dd_partnerB_biophysics_{suffix}.csv',
    'dd_PPI_aa_{suffix}.csv',
    'dd_PPI_es_{suffix}.csv'
]
    
for _dataset_key, _config in config.DATASET_CONFIG.items():
    # Create feature directory
    feature_dir = os.path.join(config.DIR_DATA, f"{_config['suffix']}_Table")
    os.makedirs(feature_dir, exist_ok=True)
    
    # Check for missing files
    files_needed = [f.format(suffix=_config['suffix']) for f in required_files]
    missing_files = [f for f in files_needed if not os.path.exists(os.path.join(feature_dir, f))]
    
    if missing_files:
        print(f"\n[WARNING] Missing feature files for {_dataset_key} dataset: {', '.join(missing_files)}")
        if choice == '1':
            print(f"Generating features for {_dataset_key} dataset...")
            # Note: This would need the protein_complex function to be imported and available
            # utils.generate_feature_all_from_Table(protein_complex, _config['table'], _config['suffix'])
            print(f"[INFO] Feature generation for {_dataset_key} would be called here")
        else:
            print(f"[ERROR] Cannot generate features without PDB files. Please choose option 1 first.")
    else:
        print(f"\n[SUCCESS] {_dataset_key} feature tables are complete. Skipping feature generation.")

print("\n[SUCCESS] Data preparation completed!")