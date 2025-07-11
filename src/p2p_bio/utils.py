import os,re,logging,sys,requests
import pandas as pd
from pathlib import Path
from Bio.PDB.PDBIO import PDBIO
from tqdm import tqdm
import shutil
import subprocess
from datetime import datetime

def download_pdb(pdb_id, save_path):
    """
     Download a PDB file from the Protein Data Bank      
    Args:
        pdb_id (str): The PDB ID to download
        save_path (str): Path where to save the PDB file
    Returns:
        bool: True if download was successful, False otherwise
    """
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes
            
        # Get total file size
        total_size = int(response.headers.get('content-length', 0))
            
        # Create progress bar
        with open(save_path, 'wb') as f, tqdm(
                desc=f"Downloading {pdb_id}",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
              for data in response.iter_content(chunk_size=1024):
                    size = f.write(data)
                    pbar.update(size)
        return True
    except requests.exceptions.RequestException as e:
            print(f"Error downloading {pdb_id}: {str(e)}")
            return False


def check_PDB_completeness(dataset_dir, table, dataset_name):
    """Check if all PDB files and their ProFix outputs exist in the dataset directory."""
    expected_files = set(table['ID'].values)
    existing_files = set()
    # Check for PDB files in nested directory structure: dataset_dir/PDBid/PDBid.pdb
    for pdb_id in expected_files:
        pdb_file_path = os.path.join(dataset_dir, pdb_id, f"{pdb_id}.pdb")
        if os.path.exists(pdb_file_path):
            existing_files.add(pdb_id)
    
    missing_files = expected_files - existing_files
    extra_files = existing_files - expected_files
    
    print(f"\nChecking {dataset_name} dataset completeness:")
    print(f"Expected files: {len(expected_files)}")
    print(f"Existing PDB files: {len(existing_files)}")
    print(f"Missing files: {len(missing_files)}")
    print(f"Extra files: {len(extra_files)}")
    
    return len(missing_files) == 0, list(missing_files)

def check_profix_PDB_completeness(dataset_dir, table, dataset_name):
    """Check if all ProFix outputs exist in the dataset directory."""
    expected_files = set(table['ID'].values)
    existing_files = set()
    
    # Check for ProFix outputs in nested directory structure: dataset_dir/PDBid/PDBid_fix.pdb
    for pdb_id in expected_files:
        profix_file_path = os.path.join(dataset_dir, pdb_id, f"{pdb_id}_fix.pdb")
        if os.path.exists(profix_file_path):
            existing_files.add(pdb_id)
    
    missing_files = expected_files - existing_files
    extra_files = existing_files - expected_files
    
    print(f"\nChecking ProFix outputs for {dataset_name} dataset:")
    print(f"Expected files: {len(expected_files)}")
    print(f"Existing ProFix outputs: {len(existing_files)}")
    print(f"Missing ProFix outputs: {len(missing_files)}")
    print(f"Extra ProFix outputs: {len(extra_files)}")
    
    return len(missing_files) == 0, list(missing_files)



def process_missing_PDB_files(dataset_dir, missing_files, dataset_name):
    """Download PDB files for a dataset."""
    print(f"\nProcessing missing files for {dataset_name} dataset...")
    
    # Determine dataset type and PDB ID handling
    if dataset_name == 'SKEMPI_V2_MT':
        # For SKEMPI_V2_MT, extract first 4 characters as PDB ID
        print("[INFO] SKEMPI_V2_MT dataset detected - extracting PDB IDs from mutation format")
        pdb_id_mapping = {}
        for mutation_id in sorted(missing_files):
            # Extract first 4 characters as PDB ID (e.g., "1OGA" from "1OGA_A_A_69_G")
            pdb_id = mutation_id[:4]
            pdb_id_mapping[mutation_id] = pdb_id
    else:
        # For PDBBIND_V2020_PP and SKEMPI_V2_WT, use PDB ID directly
        pdb_id_mapping = {pdb_id: pdb_id for pdb_id in missing_files}
    
     # Download PDB files
    for original_id, pdb_id in pdb_id_mapping.items():
        # Create directory structure: dataset_dir/original_id/pdb_id.pdb
        pdb_folder = os.path.join(dataset_dir, original_id)
        os.makedirs(pdb_folder, exist_ok=True)
        pdb_file = os.path.join(pdb_folder, f'{pdb_id}.pdb')
        
        print(f"\nProcessing {original_id} (PDB ID: {pdb_id})...")
        
        # Check if PDB file already exists
        if os.path.exists(pdb_file):
            print(f"[SUCCESS] PDB file for {pdb_id} already exists at {pdb_file}")
            
            # Check if file is not empty and readable
            if os.path.getsize(pdb_file) == 0:
                print(f"[WARNING] {pdb_file} is empty. Attempting to redownload...")
                download_success = download_pdb(pdb_id, pdb_file)
                if not download_success:
                    print(f"[ERROR] Failed to download {pdb_id}")
                    continue
        else:
            print(f"Downloading PDB file for {pdb_id}...")
            download_success = download_pdb(pdb_id, pdb_file)
            if not download_success:
                print(f"[ERROR] Failed to download {pdb_id}")
                continue
            
            print(f"[SUCCESS] Downloaded {pdb_id}.pdb to {pdb_file}")
    
    print(f"\n[SUCCESS] Completed PDB file downloads for {dataset_name} dataset")


def remove_non_interacting_residues(primary_partner, secondary_partner,cut_off, structure):
    for ichain in primary_partner:
        iresidue_ids_to_remove = []
        for iresidue in structure[0][ichain]:
            if iresidue.id[0] != ' ' or iresidue.resname == 'HOH':continue
            if 'CA' not in iresidue:continue
            flag_not_remove = False            
            for jchain in secondary_partner:
                for jresidue in structure[0][jchain]:
                    if jresidue.id[0] != ' ' or jresidue.resname == 'HOH':continue
                    if 'CA' not in jresidue:continue
                    dist = jresidue['CA'] - iresidue['CA']
                    if dist < cut_off:
                        flag_not_remove = True
                        break
                    if flag_not_remove:
                        break
                if not flag_not_remove:
                    iresidue_ids_to_remove.append(iresidue.id)
        for iresidue_id in iresidue_ids_to_remove:
            if iresidue_id in structure[0][ichain]:
                structure[0][ichain].detach_child(iresidue_id)

def use_profix(PDBID:str, filepath:str, savepath:str = None)->str:
    """
    Process PDB file with ProFix and rename the output file.
    This function will:
    1. Try to use local bin/profix first, fall back to global profix if not available
    2. Run ProFix on the PDB file to create a _fix.pdb file
    3. Rename the _fix.pdb file to the original name
    
    Args:
        PDBID: The PDB ID of the protein structure
        filepath: The directory containing the PDB file
        savepath: Optional directory to save the processed file (defaults to filepath)
    
    Returns:
        str: Path to the processed PDB file
    """
    # If savepath is not provided, use filepath
    if savepath is None:
        savepath = filepath
        
    pdb_file_path = os.path.join(filepath, f'{PDBID}.pdb') 
    temp_fix_path = os.path.join(filepath, f'{PDBID}_fix.pdb')
        
    # Check if input file exists
    if not os.path.exists(pdb_file_path):
        raise FileNotFoundError(f"PDB file not found: {pdb_file_path}")
        
    # Get project paths
    project_root = Path(__file__).parent.parent.parent
    bin_dir = project_root / "bin"
    profix_path = bin_dir / "profix"
    
    print(f"Processing {pdb_file_path} with ProFix...")
    
    # Copy jackal.dir from main directory to working PDB directory if it exists
    jackal_dir_source = bin_dir / "jackal.dir"
    jackal_dir_dest = Path(filepath) / "jackal.dir"
    
    if jackal_dir_source.exists():
        try:
            shutil.copy2(str(jackal_dir_source), str(jackal_dir_dest))
            print(f"[INFO] Copied jackal.dir to working directory: {jackal_dir_dest}")
        except Exception as e:
            print(f"[WARNING] Failed to copy jackal.dir: {str(e)}")
    else:
        print(f"[WARNING] jackal.dir not found in {bin_dir}")
    
    # Try local profix first, then global profix
    use_local_profix = False
    if profix_path.exists():
        try:
            os.chmod(profix_path, 0o755)
            print(f"Using local profix: {profix_path}")
            use_local_profix = True
        except Exception as e:
            print(f"Failed to set executable permissions for local profix: {str(e)}")
            use_local_profix = False
    else:
        print("Local profix not found, trying global profix...")
        use_local_profix = False
    
    try:
        # Save current working directory
        original_dir = os.getcwd()
        
        if use_local_profix:
            # Use local profix from bin directory
            os.chdir(str(bin_dir))
            print(f"Changed working directory to {bin_dir}")
            relative_pdb_path = os.path.relpath(pdb_file_path, bin_dir)
            cmd = ["./profix", "-fix", "0", relative_pdb_path]
        else:
            # Use global profix
            print("Using global profix command")
            cmd = ["profix", "-fix", "0", pdb_file_path]
        
        print(f"Executing command: {' '.join(cmd)}")

        # Run the command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )

        # Debug output
        print(f"ProFix exit code: {result.returncode}")
        
        # Check for output file in multiple locations
        pdb_id = Path(pdb_file_path).stem  # Get filename without extension
        
        # Create comprehensive list of possible output locations
        possible_outputs = []
        
        if use_local_profix:
            # Check in bin directory and relative to bin directory
            possible_outputs.extend([
                bin_dir / f"{pdb_id}_fix.pdb",
                bin_dir / f"{pdb_id.upper()}_fix.pdb",
                Path(f"{pdb_id}_fix.pdb"),  # Relative to bin directory
                Path(f"{pdb_id.upper()}_fix.pdb")  # Relative to bin directory
            ])
        else:
            # Check in multiple locations for global profix
            possible_outputs.extend([
                Path(filepath) / f"{pdb_id}_fix.pdb",
                Path(filepath) / f"{pdb_id.upper()}_fix.pdb",
                Path.cwd() / f"{pdb_id}_fix.pdb",  # Current working directory
                Path.cwd() / f"{pdb_id.upper()}_fix.pdb",  # Current working directory
                Path(filepath).parent / f"{pdb_id}_fix.pdb",  # Parent of filepath
                Path(filepath).parent / f"{pdb_id.upper()}_fix.pdb",  # Parent of filepath
            ])
        
        # Also check for files without the "_fix" suffix (some versions of ProFix don't add it)
        possible_outputs.extend([
            Path(filepath) / f"{pdb_id}.pdb",
            Path(filepath) / f"{pdb_id.upper()}.pdb",
            Path.cwd() / f"{pdb_id}.pdb",
            Path.cwd() / f"{pdb_id.upper()}.pdb",
        ])

        print("Checking for output files:")
        output_found = False
        target_file_path = None
        
        for output_path_candidate in possible_outputs:
            # Determine the full path based on whether we're using local profix
            if use_local_profix and not output_path_candidate.is_absolute():
                full_path = bin_dir / output_path_candidate
            else:
                full_path = output_path_candidate
            
            exists = full_path.exists()
            print(f"  - {full_path}: {'EXISTS' if exists else 'NOT FOUND'}")
            
            if exists:
                # Always move the file to the target working directory (filepath)
                try:
                    # Determine the target path in the working directory
                    if savepath != filepath:
                        # If savepath is different, use savepath
                        os.makedirs(savepath, exist_ok=True)
                        target_file_path = os.path.join(savepath, f'{PDBID}.pdb')
                    else:
                        # Otherwise, use the original filepath
                        target_file_path = pdb_file_path
                    
                    # Copy the ProFix output to the target location
                    shutil.copy2(str(full_path), target_file_path)
                    print(f"[SUCCESS] Moved ProFix output to target directory: {target_file_path}")
                    
                    # Clean up the original ProFix output file if it's not in the target directory
                    if str(full_path) != str(target_file_path):
                        try:
                            os.remove(str(full_path))
                            print(f"[INFO] Cleaned up temporary ProFix output: {full_path}")
                        except Exception as e:
                            print(f"[WARNING] Could not remove temporary file {full_path}: {str(e)}")
                    
                    output_found = True
                    break
                except Exception as e:
                    print(f"[WARNING] Error moving output file: {str(e)}")

        # Restore original directory
        os.chdir(original_dir)

        if not output_found:
            # Print debug information
            print(f"[DEBUG] ProFix stdout: {result.stdout}")
            print(f"[DEBUG] ProFix stderr: {result.stderr}")
            print(f"[DEBUG] Current working directory: {Path.cwd()}")
            print(f"[DEBUG] Input file path: {pdb_file_path}")
            print(f"[DEBUG] Filepath directory: {filepath}")
            print(f"[DEBUG] Target file path: {target_file_path}")
            raise RuntimeError(f"Failed to find ProFix output file. Exit code: {result.returncode}")

        # Return the appropriate path
        if savepath != filepath:
            return os.path.join(savepath, f'{PDBID}.pdb')
        else:
            return pdb_file_path

    except Exception as e:
        print(f"[ERROR] Error executing ProFix: {str(e)}")
        # Restore original directory in case of error
        try:
            os.chdir(original_dir)
        except:
            pass
        raise RuntimeError(f"ProFix processing failed: {str(e)}")

        


def save_pdb(structure, filepath):
    io = PDBIO()
    io.set_structure(structure)
    io.save(filepath)
    pass



def setup_logging(dataset_name, method_input):
    """Set up logging to both console and file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    PROJECT_DIR = Path(__file__).resolve().parent.parent.parent  # Adjust as needed
    log_dir = os.path.join(PROJECT_DIR, 'results', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_filename = f"gbdt_results_{dataset_name}_{method_input}_{timestamp}.txt"
    log_filepath = os.path.join(log_dir, log_filename)
    
    # Configure logging
    logger = logging.getLogger('gbdt_training')
    logger.setLevel(logging.INFO)
    
    # File handler
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Format
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_filepath


def generate_feature_all_from_Table(protein_complex:object, TABLE_GENERATION:pd.DataFrame, data_type:str='MT'):
    if data_type == 'MT':
        ID = TABLE_GENERATION['ID_clean']
    else:
        ID = TABLE_GENERATION['ID']
    
    current_file_path = os.path.abspath(__file__)
    utils_dir = os.path.dirname(current_file_path)
    src_dir = os.path.dirname(utils_dir)
    project_root = os.path.dirname(src_dir)
    RESULT_root = os.path.join(project_root, 'results', data_type)
    
    # Create the results directory if it doesn't exist
    os.makedirs(RESULT_root, exist_ok=True)
    
    for pdb_id in ID:
        pdb_folder = os.path.join(RESULT_root, pdb_id)
        
        # Case 1: PDB ID folder doesn't exist at all
        if not os.path.exists(pdb_folder):
            print(f"Creating new folder and files for {pdb_id}")
            protein_complex(PDBId=pdb_id, 
                            SAVEPATH=None,
                            generate_structure_files=True,
                            check_structure=True,
                            persistent_homology=True, 
                            read_features=False,
                            sequence=False)
        
        # Case 2: PDB ID folder exists but missing required files
        else:
            required_files_1 = [
                f"{pdb_id}.pdb",
                f"{pdb_id}_partner1.pdb",
                f"{pdb_id}_partner2.pdb"
            ]
            required_files_persistent_homology = [
                f"{pdb_id}_persistent_homology_aa.txt",
                f"{pdb_id}_persistent_homology_es.txt",
            ]
            required_files_esm = []
            missing_files_1 = [f for f in required_files_1 if not os.path.exists(os.path.join(pdb_folder, f))]
            missing_files_persistent_homology = [f for f in required_files_persistent_homology if not os.path.exists(os.path.join(pdb_folder, f))]
            if missing_files_1:
                print(f"Folder for {pdb_id} exists but missing files: {', '.join(missing_files_1)}")
                protein_complex(PDBId=pdb_id, 
                                SAVEPATH=pdb_folder,  # Use existing folder
                                generate_structure_files=True,
                                check_structure=True,
                                persistent_homology=False,  # Only generate structure files
                                read_features=False,
                                sequence=False)
            elif missing_files_persistent_homology:
                print(f"Folder for {pdb_id} exists but missing persistent homology files: {', '.join(missing_files_persistent_homology)}")
                protein_complex(PDBId=pdb_id, 
                                SAVEPATH=pdb_folder,  # Use existing folder
                                generate_structure_files=False,
                                check_structure=False,
                                persistent_homology=True,  # Only generate persistent homology files
                                read_features=False,
                                sequence=False)
            else:
                print(f"All required files already exist for {pdb_id}".center(50, '-'))





