"""
Mutation Structure Generation Module

This module provides functionality to generate mutated protein structures from wild-type structures
using SCAP (Side Chain Analysis Program). It simplifies the complex PPIstructure.py implementation
to focus on the core mutation generation functionality.

Dependencies:
- SCAP: Side Chain Analysis Program (must be installed in system PATH)
- ProFix: Protein structure fixing tool (must be installed in system PATH)
- pdb2pqr: For generating PQR files with charge assignments (must be installed in system PATH)
- Biopython: For PDB file handling
"""

import os
import re
import sys
import warnings
import subprocess
from typing import Tuple, Optional, List, Dict
from Bio.PDB import PDBParser, PDBIO

# Try to import amino acid conversion functions, with fallback
try:
    from Bio.PDB.Polypeptide import three_to_one, one_to_three
except ImportError:
    # Fallback: define our own amino acid conversion dictionaries
    three_to_one = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
        'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
        'SEF': 'S'  # Non-canonical serine
    }
    
    one_to_three = {v: k for k, v in three_to_one.items()}

# Import from constant module
# Amino acid definitions
AMINO_ACIDS = ['ARG', 'HIS', 'LYS', 'ASP', 'GLU', 'SER', 'THR', 'ASN', 'GLN', 'CYS',
               'SEF', 'GLY', 'PRO', 'ALA', 'VAL', 'ILE', 'LEU', 'MET', 'PHE', 'TYR', 'TRP']

# Non-canonical amino acid mappings
NON_CANONICAL_AA = {
    'LLP': 'LYS', 'M3P': 'LYS', 'MSE': 'MET', 'F2F': 'PHE', 'CGU': 'GLU',
    'MYL': 'LYS', 'TPO': 'THR', 'HSE': 'HIS'
}

default_cutoff = 16.0

AminoA_index = {
    'ARG': 2, 'HIS': 2, 'LYS': 2, 'ASP': 3, 'GLU': 3,
    'SER': 1, 'THR': 1, 'ASN': 1, 'GLN': 1, 'CYS': 4,
    'SEF': 4, 'GLY': 4, 'PRO': 4, 'ALA': 0, 'VAL': 0,
    'ILE': 0, 'LEU': 0, 'MET': 0, 'PHE': 0, 'TYR': 0,
    'TRP': 0
}

warnings.filterwarnings('ignore')


def parse_mutation_string(mutation_string: str) -> Dict[str, str]:
    """
    Parse mutation information from string format "PDBID_ChainID_WildResidue_ResidueID_MutateResidue".
    
    Args:
        mutation_string: Mutation string in format "1JTG_A_N_100_A"
        
    Returns:
        Dictionary with parsed mutation information:
        {
            'pdb_id': '1JTG',
            'chain_id': 'A', 
            'wild_residue': 'N',
            'residue_id': '100',
            'mutate_residue': 'A'
        }
        
    Raises:
        ValueError: If mutation string format is invalid
    """
    parts = mutation_string.split('_')
    if len(parts) != 5:
        raise ValueError(f"Invalid mutation string format. Expected 5 parts separated by '_', got {len(parts)} parts: {mutation_string}")
    
    pdb_id, chain_id, wild_residue, residue_id, mutate_residue = parts
    
    # Validate residue codes
    if len(wild_residue) == 1:
        # Convert one-letter to three-letter if needed
        try:
            wild_residue_three = one_to_three[wild_residue]
        except KeyError:
            raise ValueError(f"Invalid wild residue code: {wild_residue}")
    elif len(wild_residue) == 3:
        wild_residue_three = wild_residue
    else:
        raise ValueError(f"Invalid wild residue format: {wild_residue}")
    
    if len(mutate_residue) == 1:
        # Convert one-letter to three-letter if needed
        try:
            mutate_residue_three = one_to_three[mutate_residue]
        except KeyError:
            raise ValueError(f"Invalid mutation residue code: {mutate_residue}")
    elif len(mutate_residue) == 3:
        mutate_residue_three = mutate_residue
    else:
        raise ValueError(f"Invalid mutation residue format: {mutate_residue}")
    
    return {
        'pdb_id': pdb_id,
        'chain_id': chain_id,
        'wild_residue': wild_residue,
        'wild_residue_three': wild_residue_three,
        'residue_id': residue_id,
        'mutate_residue': mutate_residue,
        'mutate_residue_three': mutate_residue_three
    }


class MutationStructureGenerator:
    """
    A simplified class for generating mutated protein structures using SCAP.
    
    This class handles the core functionality of:
    1. Loading and validating wild-type PDB structures
    2. Processing structures with ProFix
    3. Generating mutations using SCAP
    4. Saving the resulting mutant structures
    5. Generating PQR files for electrostatic calculations
    """
    
    def __init__(self, working_directory: str = "."):
        """
        Initialize the mutation structure generator.
        
        Args:
            working_directory: Directory where temporary files will be created
        """
        self.working_directory = working_directory
        self._validate_dependencies()
    
    def _validate_dependencies(self) -> None:
        """Validate that required external tools (SCAP, ProFix, pdb2pqr) are available."""
        try:
            subprocess.run(['scap', '--help'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("SCAP not found in system PATH. Please install SCAP.")
        
        try:
            subprocess.run(['profix', '--help'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("ProFix not found in system PATH. Please install ProFix.")
        
        try:
            subprocess.run(['pdb2pqr', '--help'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("pdb2pqr not found in system PATH. Please install pdb2pqr.")
    
    def _parse_residue_id(self, res_id: str) -> Tuple[str, int, str]:
        """
        Parse residue ID string to tuple format.
        
        Args:
            res_id: Residue ID string (e.g., "83" or "83A")
            
        Returns:
            Tuple of (hetero flag, residue number, insertion code)
        """
        res_id_match = re.search(r'([a-zA-Z])?(\d+)([a-zA-Z])?', res_id)
        if res_id_match:
            hetero_flag = res_id_match.group(1) or ' '
            residue_number = int(res_id_match.group(2))
            insertion_code = res_id_match.group(3) or ' '
            return hetero_flag, residue_number, insertion_code
        else:
            raise ValueError(f"Invalid residue ID format: {res_id}")
    
    def _generate_pqr_files(self, pdb_file_path: str, output_base_name: str) -> str:
        """
        Generate PQR files from PDB files using pdb2pqr.
        
        Args:
            pdb_file_path: Path to the PDB file
            output_base_name: Base name for output files
            
        Returns:
            Path to the generated PQR file
            
        Raises:
            RuntimeError: If pdb2pqr fails to generate PQR files
        """
        # Generate PQR file for the input PDB
        pqr_file_path = f"{output_base_name}.pqr"
        
        try:
            if not os.path.exists(pqr_file_path):
                subprocess.run([
                    'pdb2pqr', 
                    '--ff=amber', 
                    '--ph-calc-method=propka', 
                    '--chain', 
                    '--with-ph=7.0', 
                    pdb_file_path, 
                    pqr_file_path
                ], check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"pdb2pqr failed to generate PQR file: {e}")
        except FileNotFoundError:
            raise RuntimeError("pdb2pqr is not installed, please install it first")
        
        return pqr_file_path
    
    def _clean_structure(self, structure, target_chains: List[str]) -> None:
        """
        Clean structure by removing non-canonical residues and unwanted chains.
        
        Args:
            structure: Biopython structure object
            target_chains: List of chain IDs to keep
        """
        # Remove unwanted chains
        chains_to_remove = []
        for chain in structure[0]:
            if chain.id not in target_chains:
                chains_to_remove.append(chain.id)
        
        for chain_id in chains_to_remove:
            structure[0].detach_child(chain_id)
        
        # Replace non-canonical residues and remove non-amino acid residues
        for chain_id in target_chains:
            if chain_id in structure[0]:
                residues_to_remove = []
                for residue in structure[0][chain_id]:
                    if residue.resname in NON_CANONICAL_AA:
                        residue.resname = NON_CANONICAL_AA[residue.resname]
                    elif residue.resname not in AMINO_ACIDS:
                        residues_to_remove.append(residue.id)
                
                for residue_id in residues_to_remove:
                    structure[0][chain_id].detach_child(residue_id)
    
    def _remove_insertion_codes(self, structure, target_chains: List[str], 
                               mutation_res_id: Tuple, mutation_chain: str) -> Tuple[int, None]:
        """
        Remove insertion codes and renumber residues sequentially.
        
        Args:
            structure: Biopython structure object
            target_chains: List of chain IDs to process
            mutation_res_id: Original mutation residue ID
            mutation_chain: Chain ID where mutation occurs
            
        Returns:
            Tuple of (new residue ID, None)
        """
        new_res_id = None
        
        for chain_id in target_chains:
            if chain_id in structure[0]:
                # First pass: mark old residues
                for residue in structure[0][chain_id]:
                    residue_id_list = list(residue.id)
                    residue_id_list[0] = 'Old'
                    residue.id = tuple(residue_id_list)
                
                # Second pass: renumber sequentially
                for idx, residue in enumerate(structure[0][chain_id]):
                    residue_id_list = list(residue.id)
                    residue_id_list[0] = ' '
                    residue_id_list[1] = idx + 1
                    residue_id_list[2] = ' '
                    residue.id = tuple(residue_id_list)
                    
                    # Track mutation residue
                    if (chain_id == mutation_chain and 
                        residue_id_list[1] == mutation_res_id[1]):
                        new_res_id = idx + 1
        
        return new_res_id, None
    
    def _run_profix_scap(self, base_filename: str, mutation_chain: str, 
                        residue_id: int, mutation_residue: str) -> None:
        """
        Run ProFix and SCAP to generate mutant structure.
        
        Args:
            base_filename: Base filename for the structure
            mutation_chain: Chain ID where mutation occurs
            residue_id: Residue ID where mutation occurs
            mutation_residue: New residue type (one-letter code)
        """
        # Run ProFix to fix the structure
        profix_cmd = f"profix -fix 0 {base_filename}_WT.pdb"
        result = subprocess.run(profix_cmd.split(), capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ProFix failed: {result.stderr}")
        
        # Rename fixed file
        os.rename(f"{base_filename}_WT_fix.pdb", f"{base_filename}_WT.pdb")
        
        # Create SCAP input file
        scap_input = f"{mutation_chain},{residue_id},{mutation_residue}"
        with open("tmp_scap.list", "w") as f:
            f.write(scap_input)
        
        # Run SCAP to generate mutation
        scap_cmd = f"scap -ini 20 -min 4 {base_filename}_WT.pdb tmp_scap.list"
        result = subprocess.run(scap_cmd.split(), capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"SCAP failed: {result.stderr}")
        
        # Rename output file
        os.rename(f"{base_filename}_WT_scap.pdb", f"{base_filename}_MT.pdb")
        
        # Clean up temporary file
        if os.path.exists("tmp_scap.list"):
            os.remove("tmp_scap.list")
    
    def generate_mutation_from_string(self, 
                                    mutation_string: str,
                                    wt_pdb_path: str,
                                    output_base_name: str,
                                    target_chains: Optional[List[str]] = None) -> str:
        """
        Generate a mutated protein structure from a mutation string.
        
        Args:
            mutation_string: Mutation string in format "PDBID_ChainID_WildResidue_ResidueID_MutateResidue"
            wt_pdb_path: Path to wild-type PDB file
            output_base_name: Base name for output files
            target_chains: List of chain IDs to include (if None, includes all chains)
            
        Returns:
            Path to the generated mutant PDB file
            
        Raises:
            ValueError: If input parameters are invalid
            RuntimeError: If external tools fail
        """
        # Parse mutation string
        mutation_info = parse_mutation_string(mutation_string)
        
        # Validate that wild residue matches what's expected
        if mutation_info['wild_residue'] != mutation_info['wild_residue']:
            print(f"Warning: Wild residue in mutation string ({mutation_info['wild_residue']}) "
                  f"may not match actual residue in structure")
        
        # Generate mutation using parsed information
        return self.generate_mutation(
            wt_pdb_path=wt_pdb_path,
            output_base_name=output_base_name,
            mutation_chain=mutation_info['chain_id'],
            mutation_residue_id=mutation_info['residue_id'],
            mutation_residue=mutation_info['mutate_residue'],
            target_chains=target_chains
        )
    
    def generate_mutation(self, 
                         wt_pdb_path: str,
                         output_base_name: str,
                         mutation_chain: str,
                         mutation_residue_id: str,
                         mutation_residue: str,
                         target_chains: Optional[List[str]] = None) -> str:
        """
        Generate a mutated protein structure from a wild-type structure.
        
        This method generates:
        1. Wild-type PDB file: {output_base_name}_WT.pdb
        2. Mutant PDB file: {output_base_name}_MT.pdb
        3. Wild-type PQR file: {output_base_name}_WT.pqr
        4. Mutant PQR file: {output_base_name}_MT.pqr
        
        Args:
            wt_pdb_path: Path to wild-type PDB file
            output_base_name: Base name for output files
            mutation_chain: Chain ID where mutation occurs
            mutation_residue_id: Residue ID where mutation occurs (e.g., "83" or "83A")
            mutation_residue: New residue type (one-letter code, e.g., "A")
            target_chains: List of chain IDs to include (if None, includes all chains)
            
        Returns:
            Path to the generated mutant PDB file
            
        Raises:
            ValueError: If input parameters are invalid
            RuntimeError: If external tools fail
        """
        # Validate inputs
        if not os.path.exists(wt_pdb_path):
            raise ValueError(f"Wild-type PDB file not found: {wt_pdb_path}")
        
        if mutation_residue not in [three_to_one(aa) for aa in AMINO_ACIDS]:
            raise ValueError(f"Invalid mutation residue: {mutation_residue}")
        
        # Parse residue ID
        try:
            res_id_tuple = self._parse_residue_id(mutation_residue_id)
        except ValueError as e:
            raise ValueError(f"Invalid residue ID format: {mutation_residue_id}")
        
        # Load structure
        parser = PDBParser(PERMISSIVE=1)
        try:
            structure = parser.get_structure("WT", wt_pdb_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load PDB structure: {e}")
        
        # Determine target chains
        if target_chains is None:
            target_chains = [chain.id for chain in structure[0]]
        
        if mutation_chain not in target_chains:
            raise ValueError(f"Mutation chain {mutation_chain} not in target chains {target_chains}")
        
        # Clean structure
        self._clean_structure(structure, target_chains)
        
        # Remove insertion codes and renumber
        new_res_id, _ = self._remove_insertion_codes(structure, target_chains, 
                                                    res_id_tuple, mutation_chain)
        
        if new_res_id is None:
            raise ValueError(f"Mutation residue {mutation_residue_id} not found in chain {mutation_chain}")
        
        # Save wild-type structure
        wt_output_path = f"{output_base_name}_WT.pdb"
        io = PDBIO()
        io.set_structure(structure)
        io.save(wt_output_path)
        
        # Generate mutation using ProFix and SCAP
        self._run_profix_scap(output_base_name, mutation_chain, new_res_id, mutation_residue)
        
        # Return path to mutant structure
        mt_output_path = f"{output_base_name}_MT.pdb"
        if not os.path.exists(mt_output_path):
            raise RuntimeError("Failed to generate mutant structure file")
        
        # Generate PQR files for both wild-type and mutant structures
        try:
            # Generate PQR for wild-type structure
            wt_pqr_path = self._generate_pqr_files(wt_output_path, f"{output_base_name}_WT")
            print(f"Generated wild-type PQR file: {wt_pqr_path}")
            
            # Generate PQR for mutant structure
            mt_pqr_path = self._generate_pqr_files(mt_output_path, f"{output_base_name}_MT")
            print(f"Generated mutant PQR file: {mt_pqr_path}")
            
        except Exception as e:
            print(f"Warning: Failed to generate PQR files: {e}")
            print("PDB files were generated successfully, but PQR generation failed.")
        
        return mt_output_path
    
    def batch_generate_mutations_from_strings(self, 
                                            mutation_strings: List[str],
                                            wt_pdb_paths: List[str],
                                            output_base_names: List[str],
                                            target_chains: Optional[List[List[str]]] = None) -> List[str]:
        """
        Generate multiple mutations from mutation strings in batch.
        
        Args:
            mutation_strings: List of mutation strings in format "PDBID_ChainID_WildResidue_ResidueID_MutateResidue"
            wt_pdb_paths: List of paths to wild-type PDB files
            output_base_names: List of output base names
            target_chains: Optional list of target chain lists for each mutation
            
        Returns:
            List of paths to generated mutant PDB files
        """
        if len(mutation_strings) != len(wt_pdb_paths) or len(mutation_strings) != len(output_base_names):
            raise ValueError("All input lists must have the same length")
        
        if target_chains is None:
            target_chains = [None] * len(mutation_strings)
        elif len(target_chains) != len(mutation_strings):
            raise ValueError("target_chains list must have the same length as other input lists")
        
        results = []
        
        for i, (mutation_string, wt_pdb_path, output_base_name, target_chain) in enumerate(
            zip(mutation_strings, wt_pdb_paths, output_base_names, target_chains)):
            try:
                print(f"Processing mutation {i+1}/{len(mutation_strings)}: {mutation_string}")
                
                # Generate mutation
                mt_path = self.generate_mutation_from_string(
                    mutation_string=mutation_string,
                    wt_pdb_path=wt_pdb_path,
                    output_base_name=output_base_name,
                    target_chains=target_chain
                )
                
                results.append(mt_path)
                print(f"Successfully generated: {mt_path}")
                
            except Exception as e:
                print(f"Failed to process mutation {i+1}: {e}")
                results.append(None)
        
        return results
    
    def batch_generate_mutations(self, 
                                mutations: List[dict],
                                output_directory: str = ".") -> List[str]:
        """
        Generate multiple mutations in batch.
        
        Args:
            mutations: List of mutation dictionaries with keys:
                      - wt_pdb_path: Path to wild-type PDB
                      - output_base_name: Base name for output files
                      - mutation_chain: Chain ID for mutation
                      - mutation_residue_id: Residue ID for mutation
                      - mutation_residue: New residue type
                      - target_chains: Optional list of target chains
            output_directory: Directory to save output files
            
        Returns:
            List of paths to generated mutant PDB files
        """
        results = []
        
        for i, mutation in enumerate(mutations):
            try:
                print(f"Processing mutation {i+1}/{len(mutations)}: {mutation}")
                
                # Set output path
                output_base_name = os.path.join(output_directory, mutation['output_base_name'])
                
                # Generate mutation
                mt_path = self.generate_mutation(
                    wt_pdb_path=mutation['wt_pdb_path'],
                    output_base_name=output_base_name,
                    mutation_chain=mutation['mutation_chain'],
                    mutation_residue_id=mutation['mutation_residue_id'],
                    mutation_residue=mutation['mutation_residue'],
                    target_chains=mutation.get('target_chains')
                )
                
                results.append(mt_path)
                print(f"Successfully generated: {mt_path}")
                
            except Exception as e:
                print(f"Failed to process mutation {i+1}: {e}")
                results.append(None)
        
        return results


def generate_mutation_structure_from_string(mutation_string: str,
                                           wt_pdb_path: str,
                                           output_base_name: str,
                                           target_chains: Optional[List[str]] = None,
                                           working_directory: str = ".") -> str:
    """
    Convenience function to generate a single mutation structure from a mutation string.
    
    Args:
        mutation_string: Mutation string in format "PDBID_ChainID_WildResidue_ResidueID_MutateResidue"
        wt_pdb_path: Path to wild-type PDB file
        output_base_name: Base name for output files
        target_chains: Optional list of target chains
        working_directory: Working directory for temporary files
        
    Returns:
        Path to the generated mutant PDB file
    """
    generator = MutationStructureGenerator(working_directory)
    return generator.generate_mutation_from_string(
        mutation_string=mutation_string,
        wt_pdb_path=wt_pdb_path,
        output_base_name=output_base_name,
        target_chains=target_chains
    )


def generate_mutation_structure(wt_pdb_path: str,
                               output_base_name: str,
                               mutation_chain: str,
                               mutation_residue_id: str,
                               mutation_residue: str,
                               target_chains: Optional[List[str]] = None,
                               working_directory: str = ".") -> str:
    """
    Convenience function to generate a single mutation structure.
    
    Args:
        wt_pdb_path: Path to wild-type PDB file
        output_base_name: Base name for output files
        mutation_chain: Chain ID where mutation occurs
        mutation_residue_id: Residue ID where mutation occurs
        mutation_residue: New residue type (one-letter code)
        target_chains: Optional list of target chains
        working_directory: Working directory for temporary files
        
    Returns:
        Path to the generated mutant PDB file
    """
    generator = MutationStructureGenerator(working_directory)
    return generator.generate_mutation(
        wt_pdb_path=wt_pdb_path,
        output_base_name=output_base_name,
        mutation_chain=mutation_chain,
        mutation_residue_id=mutation_residue_id,
        mutation_residue=mutation_residue,
        target_chains=target_chains
    ) 