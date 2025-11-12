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

Author: Jiahui Chen, Xingjian Xu
Email: 06/04/2024
LastEditTime: 11/10/2025

"""

import os
import re
import sys
import warnings
import subprocess
import shutil
import glob
from typing import Tuple, Optional, List, Dict
from Bio.PDB import PDBParser, PDBIO


warnings.filterwarnings('ignore')
# Try to import amino acid conversion functions, with fallback
try:
    from .constant import default_cutoff, AMINO_ACIDS, NON_CANONICAL_AA, AminoA_index
    from Bio.PDB.Polypeptide import three_to_one, one_to_three
except:
    # When this module is executed as a script (not as a package) the relative import fails.
    # Add the parent 'src' directory to sys.path so absolute imports work when running the file directly.
    try:
        src_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if src_parent not in sys.path:
            sys.path.insert(0, src_parent)
    except Exception:
        pass
    from p2p_bio.constant import three_to_one, one_to_three, default_cutoff, AMINO_ACIDS, NON_CANONICAL_AA, AminoA_index
    
# Ensure `three_to_one` is a callable function. Some modules provide a dict instead of a function.
try:
    if isinstance(three_to_one, dict):
        _three_map = three_to_one
        def three_to_one(code: str) -> str:
            if not code:
                return '?'
            return _three_map.get(code.upper(), _three_map.get(code, '?'))
except NameError:
    pass


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
        # Helper to find executable in PATH or project/bin
        def find_executable(cmd_name: str) -> Optional[str]:
            # 1) system PATH
            path = shutil.which(cmd_name)
            if path:
                return path
            # 2) project bin directory (src/../bin)
            try:
                # Compute repository root (three levels up from this file: src/p2p_bio -> src -> repo)
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                candidate = os.path.join(project_root, 'bin', cmd_name)
                if os.path.exists(candidate) and os.access(candidate, os.X_OK):
                    return candidate
            except Exception:
                pass
            return None

        # Find SCAP
        scap_path = find_executable('scap')
        if not scap_path:
            raise RuntimeError("SCAP not found in system PATH or project 'bin/'. Please install SCAP or place its executable in the project's bin/ directory.")
        self.scap_path = scap_path

        # Find ProFix
        profix_path = find_executable('profix')
        if not profix_path:
            raise RuntimeError("ProFix not found in system PATH or project 'bin/'. Please install ProFix or place its executable in the project's bin/ directory.")
        self.profix_path = profix_path

        # Find pdb2pqr (may be installed via pip/conda)
        pdb2pqr_path = find_executable('pdb2pqr')
        if not pdb2pqr_path:
            raise RuntimeError("pdb2pqr not found in system PATH or project 'bin/'. Please install pdb2pqr.")
        self.pdb2pqr_path = pdb2pqr_path
        # Try to read jackal.dir from project bin to set JACKALDIR for SCAP if available
        try:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            jackal_dir_file = os.path.join(project_root, 'bin', 'jackal.dir')
            if os.path.exists(jackal_dir_file):
                with open(jackal_dir_file, 'r') as f:
                    content = f.read().strip()
                
                # Parse jackal.dir format: can be either a simple path or formatted as:
                # pdb: .
                # library: /path/to/library
                # key: KEY
                jackal_path = None
                if 'library:' in content:
                    # Parse formatted version
                    for line in content.split('\n'):
                        line = line.strip()
                        if line.startswith('library:'):
                            jackal_path = line.split('library:', 1)[1].strip()
                            break
                else:
                    # Simple path format (backward compatibility)
                    jackal_path = content
                
                if jackal_path:
                    # If jackal.dir points to a subfolder like .../library, use its parent
                    jp = jackal_path
                    try:
                        if jp.endswith(os.sep + 'library') or jp.endswith('/library'):
                            jp = os.path.dirname(jp)
                    except Exception:
                        pass
                    self.jackal_dir = jp
                else:
                    self.jackal_dir = None
            else:
                self.jackal_dir = None
        except Exception:
            self.jackal_dir = None
    
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
                    self.pdb2pqr_path,
                    '--ff=AMBER',
                    '--titration-state-method=propka',
                    '--keep-chain',
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

        # We'll build a mapping from original (number,insertion) -> new sequential id
        for chain_id in target_chains:
            if chain_id in structure[0]:
                # Collect original residue identity information in order
                original_info = []
                for residue in structure[0][chain_id]:
                    # residue.id is typically a tuple like (' ', seq_number, insertion)
                    try:
                        _flag, orig_num, orig_ins = residue.id
                    except Exception:
                        # Fallback: try to coerce
                        _flag = residue.id[0] if len(residue.id) > 0 else ' '
                        orig_num = residue.id[1] if len(residue.id) > 1 else None
                        orig_ins = residue.id[2] if len(residue.id) > 2 else ' '
                    original_info.append((orig_num, orig_ins))

                # Now renumber sequentially and find which new index corresponds to the original mutation id
                for idx, residue in enumerate(list(structure[0][chain_id])):
                    residue_id_list = list(residue.id)
                    # assign new sequential numbering and clear insertion code
                    residue_id_list[0] = ' '
                    residue_id_list[1] = idx + 1
                    residue_id_list[2] = ' '
                    residue.id = tuple(residue_id_list)

                    # Compare against original info captured earlier
                    try:
                        orig_num, orig_ins = original_info[idx]
                    except Exception:
                        orig_num, orig_ins = None, ' '

                    # mutation_res_id is a tuple (hetero_flag, number, insertion)
                    try:
                        mut_num = mutation_res_id[1]
                        mut_ins = mutation_res_id[2]
                    except Exception:
                        mut_num = mutation_res_id
                        mut_ins = ' '

                    # Match by original residue number and insertion code (if provided)
                    if chain_id == mutation_chain and orig_num is not None:
                        if orig_num == mut_num and (mut_ins == ' ' or str(orig_ins) == str(mut_ins)):
                            new_res_id = idx + 1

        return new_res_id, None
    
    def _run_profix_scap(self, wt_pdb_path: str, mutation_chain: str,
                        residue_id: int, mutation_residue: str,
                        desired_output_name: Optional[str] = None) -> str:
        """
        Run ProFix and SCAP to generate mutant structure.

        Operates in the directory containing the input WT PDB so that
        temporary files (e.g. tmp_scap.list) and outputs live alongside
        the original PDB (for example, data/Example).

        Returns the path to the generated mutant PDB.
        """
        wt_pdb_path = os.path.abspath(wt_pdb_path)
        wt_dir = os.path.dirname(wt_pdb_path) or os.getcwd()
        base = os.path.splitext(os.path.basename(wt_pdb_path))[0]

        # Ensure we operate in the WT directory so SCAP output and tmp files end up there
        cwd = os.getcwd()
        try:
            os.chdir(wt_dir)

            # Run ProFix on the WT pdb file (may modify in-place or produce variant names)
            profix_cmd = [self.profix_path, '-fix', '0', wt_pdb_path]
            result = subprocess.run(profix_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"ProFix failed: {result.stderr}\ncmd: {' '.join(profix_cmd)}")

            # Candidate fixed filenames
            candidates = [
                f"{base}_WT_fix.pdb",
                f"{base}_fix.pdb",
                f"{base}_WT_fix.PDB",
                f"{base}_fix.PDB",
                os.path.basename(wt_pdb_path),
            ]

            fixed_path = None
            for cand in candidates:
                if os.path.exists(cand):
                    fixed_path = os.path.abspath(cand)
                    break

            if fixed_path is None:
                raise RuntimeError(f"ProFix did not produce expected fixed file. Candidates checked: {candidates}. Dir listing: {os.listdir(wt_dir)}")

            # Create SCAP input file next to WT
            scap_list = os.path.join(wt_dir, 'tmp_scap.list')
            with open(scap_list, 'w') as f:
                f.write(f"{mutation_chain},{residue_id},{mutation_residue}\n")

            # Run SCAP; pass JACKALDIR if available
            scap_cmd = [self.scap_path, '-ini', '20', '-min', '4', fixed_path, scap_list]
            env = os.environ.copy()
            if hasattr(self, 'jackal_dir') and self.jackal_dir:
                env['JACKALDIR'] = self.jackal_dir

            result = subprocess.run(scap_cmd, capture_output=True, text=True, env=env)
            if result.returncode != 0:
                raise RuntimeError(f"SCAP failed: {result.stderr}\nstdout: {result.stdout}\ncmd: {' '.join(scap_cmd)}")

            # Find SCAP output in WT directory
            patterns = [
                f"{base}*_scap*.pdb",
                f"{base}*_SCAP*.pdb",
                f"{base}*_WT_scap*.pdb",
                f"{base}_WT_scap.pdb",
                f"{base}_scap.pdb",
                f"{base}_MT.pdb",
                f"{base}*_MT*.pdb",
                f"{base}*scap*.pdb",
            ]

            scap_candidates = []
            for pat in patterns:
                scap_candidates.extend(glob.glob(pat))

            # remove duplicates preserving order
            seen = set(); scap_candidates = [x for x in scap_candidates if not (x in seen or seen.add(x))]

            if not scap_candidates:
                raise RuntimeError(f"SCAP produced no expected outputs in {wt_dir}. Dir listing: {os.listdir(wt_dir)}\nSCAP stdout: {result.stdout!r}\nSCAP stderr: {result.stderr!r}")

            # Choose first candidate and rename to desired final name.
            src = os.path.abspath(scap_candidates[0])
            if desired_output_name:
                # strip any extension
                desired_base = os.path.splitext(os.path.basename(desired_output_name))[0]
                final_mt = os.path.join(wt_dir, f"{desired_base}.pdb")
            else:
                final_mt = os.path.join(wt_dir, f"{base}_MT.pdb")

            if os.path.abspath(src) != os.path.abspath(final_mt):
                try:
                    os.replace(src, final_mt)
                except Exception as e:
                    raise RuntimeError(f"Failed to move SCAP output {src} -> {final_mt}: {e}")

            # cleanup tmp list
            try:
                if os.path.exists(scap_list):
                    os.remove(scap_list)
            except Exception:
                pass

            return final_mt

        finally:
            os.chdir(cwd)
    
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
        # If user provided a directory as wt_pdb_path, try to locate a PDB file inside.
        if os.path.isdir(wt_pdb_path):
            # Prefer a file named by the PDB ID (e.g., 1A22.pdb)
            pdb_id = mutation_info.get('pdb_id')
            candidate = os.path.join(wt_pdb_path, f"{pdb_id}.pdb") if pdb_id else None
            if candidate and os.path.isfile(candidate):
                wt_pdb_path = candidate
            else:
                # fallback: find the first .pdb file in the directory
                files = [f for f in os.listdir(wt_pdb_path) if f.lower().endswith('.pdb')]
                if len(files) == 0:
                    raise ValueError(f"No PDB file found in directory: {wt_pdb_path}. Please provide a path to a PDB file.")
                if len(files) > 1:
                    print(f"[WARNING] Multiple PDB files found in {wt_pdb_path}; using {files[0]}")
                wt_pdb_path = os.path.join(wt_pdb_path, files[0])
        
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
        
        # Save wild-type structure next to the input WT PDB (don't force a '_WT' suffix)
        wt_dir = os.path.dirname(os.path.abspath(wt_pdb_path)) or os.getcwd()
        base_name = os.path.splitext(os.path.basename(wt_pdb_path))[0]
        wt_output_path = os.path.join(wt_dir, f"{base_name}.pdb")
        io = PDBIO()
        io.set_structure(structure)
        io.save(wt_output_path)

        # Generate mutation using ProFix and SCAP (operate in WT directory)
        # Use the provided output_base_name (if it's a simple name like '1A22_A_M_14_A')
        desired_name = None
        try:
            # prefer output_base_name if it's simple (no path)
            desired_name = os.path.basename(output_base_name)
        except Exception:
            desired_name = None

        mt_output_path = self._run_profix_scap(wt_output_path, mutation_chain, new_res_id, mutation_residue, desired_output_name=desired_name)

        # Verify mutant produced
        if not os.path.exists(mt_output_path):
            raise RuntimeError("Failed to generate mutant structure file")
        
        # Generate PQR files for both wild-type and mutant structures
        try:
            # Generate PQR for wild-type structure (place in same directory as WT)
            out_base_wt = os.path.join(wt_dir, f"{output_base_name}")
            wt_pqr_path = self._generate_pqr_files(wt_output_path, out_base_wt)
            print(f"Generated wild-type PQR file: {wt_pqr_path}")

            # Generate PQR for mutant structure
            out_base_mt = os.path.join(wt_dir, f"{output_base_name}")
            mt_pqr_path = self._generate_pqr_files(mt_output_path, out_base_mt)
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
if __name__ == "__main__":
    # Example usage
    MT_path = "../../data/Example/mutation_experiment"
    # Only run the example if the provided WT PDB path exists.
    if not os.path.exists(MT_path):
        print(f"[INFO] Example WT path not found ({MT_path}). Skipping example run in __main__. Provide a valid PDB path to run the example.")
    else:
        generator = MutationStructureGenerator(MT_path)
        mutation_string = "1A22_A_M_14_A"
        target_chain = mutation_string[5]
        output_base_name = "1A22_A_M14_A"
        output_path = generator.generate_mutation_from_string(
            mutation_string,
            MT_path,
            output_base_name,
            target_chain
        )
        print(f"[INFO] Generated mutant structure at: {output_path}")