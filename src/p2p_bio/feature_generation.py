"""
Author: Xingjian Xu
Email:xingjianxu@ufl.edu
Data: 06/04/2024
LastEditTime: 11/10/2025
"""
import os
import sys
import warnings
import numpy as np
import torch
import esm
import pandas as pd
from typing import List, Tuple
from Bio.PDB.Structure import Structure
from Bio.PDB.PDBParser import PDBParser
import subprocess
import shutil
from pathlib import Path
# Suppress warnings
warnings.filterwarnings("ignore")

# Fix the import path for config
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(src_dir)
sys.path.insert(0, src_dir)  # Add src directory to Python path

# Import from config
from config import DIR_DATA, TABLE_PDBBIND_V2020_PP, TABLE_SKEMPI_V2_WT, TABLE_SKEMPI_V2_MT

# Import local modules
try:
    from .constant import EleLength, default_cutoff
    from .parser import parser_pdb, parser_pqr
    from .persistent import (
        generate_rips_complex, 
        generate_alpha_shape, 
        generate_flexibility_rigidy_index, 
        generate_persistent_spectra
    )
    from .utils import use_profix, save_pdb
except:
    # Fallback for when running as script
    from p2p_bio.constant import EleLength, default_cutoff
    from p2p_bio.parser import parser_pdb, parser_pqr
    from p2p_bio.persistent import (
        generate_rips_complex, 
        generate_alpha_shape, 
        generate_flexibility_rigidy_index, 
        generate_persistent_spectra
    )
    from p2p_bio.utils import use_profix, save_pdb

class protein_complex:
    def __init__(self, PDBID: str, 
                       filepath: str, 
                       partner1: str, 
                       partner2: str, 
                 check_structure: bool = False, generate_structure_files: bool = True,
                 persistent_homology: bool = False, biophysics: bool = False, sequence: bool = True,
                 read_features: bool = False):
        self.PDBID = PDBID
        self.filepath = filepath
        self.partner1 = partner1
        self.partner2 = partner2

        if read_features:
            self.read_features()
        #=========================================================================================================

        #=========================================================================================================
        # Generate files from a PDB file to other formats
        #=========================================================================================================
        if generate_structure_files:
            self._generate_structure_files()
        if check_structure:
            self._check_structure()
        
        #=========================================================================================================

        #=========================================================================================================
        # Generate features of persistent homology, persistent Laplacian
        # Required input: a pdb file containing the structure of a protein-protein complex
        #=========================================================================================================
        if persistent_homology:
            presistent_features_es, presistent_features_aa = self.persistent_homology_features()
            np.savetxt(os.path.join(self.filepath, self.PDBID+'_persistent_homology_es.txt'), presistent_features_es)
            np.savetxt(os.path.join(self.filepath, self.PDBID+'_persistent_homology_aa.txt'), presistent_features_aa)
        #=========================================================================================================

        #=========================================================================================================
        # Generate features of biophysics
        # Required input: pqr, pdb files, and the results of MIBPB
        # Construct the features for partner 1 and partner 2, seperately
        #=========================================================================================================
        # if biophysics:
        #     biophysics_features_partner1 = self.biophysics_features(partner1)
        #     np.savetxt(os.path.join(self.filepath, self.PDBID+'_'+self.partner1+'_biophysics.txt'), biophysics_features_partner1)
        #     biophysics_features_partner2 = self.biophysics_features(partner2)
        #     np.savetxt(os.path.join(self.filepath, self.PDBID+'_'+self.partner2+'_biophysics.txt'), biophysics_features_partner2)
        #=========================================================================================================

        #=========================================================================================================
        # Generate features of sequence using ESM
        # Required input: sequences are generated from the pdb file of the binding interface
        #=========================================================================================================
        if sequence:
            print("[INFO] Generating sequence features ...")
            self.sequence_features(PDBID, filepath, partner1, partner2)
        #=========================================================================================================

    def read_features(self):
        self.persistent_homology_es = np.loadtxt(os.path.join(self.filepath, self.PDBID+'_persistent_homology_es.txt'))
        self.persistent_homology_aa = np.loadtxt(os.path.join(self.filepath, self.PDBID+'_persistent_homology_aa.txt'))
        self.biophysics_partner1 = np.loadtxt(os.path.join(self.filepath, self.PDBID+'_'+self.partner1+'_biophysics.txt'))
        self.biophysics_partner2 = np.loadtxt(os.path.join(self.filepath, self.PDBID+'_'+self.partner2+'_biophysics.txt'))
        self.sequence = np.loadtxt(os.path.join(self.filepath, self.PDBID+'_esm_features.txt'))

    def sequence_features(self, PDBID, filepath, partner1, partner2):
        '''
        Generate the sequence features using ESM
        NOTE: loading ESM is time-consuming, so it is better to load it once and use it for all complexes
        After save the features, we can load the features directly using this function
        '''
    
        import torch, esm
        
    
        # Load sequences and compute ESM embeddings (mean-pooled per-sequence)
        sequence_list, partner1_chains, partner2_chains = self.generate_sequences()
        if not sequence_list:
            return
        print(f"[INFO] Generating ESM2 features for {self.PDBID} ...")
        # Use ESM2 (fair-esm v2). Choose a small/medium model variant to balance memory.
        model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        batch_converter = alphabet.get_batch_converter()
        model.eval()

        labels, strs, tokens = batch_converter(sequence_list)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        tokens = tokens.to(device)
        with torch.no_grad():
            # pick the top repr layer dynamically (ESM models expose num_layers)
            try:
                repr_layer = model.num_layers
            except Exception:
                repr_layer = 6
            results = model(tokens, repr_layers=[repr_layer], return_contacts=False)
        token_reps = results['representations'][repr_layer]

        # mean-pool residue tokens (exclude BOS/CLS token at index 0 and any padding)
        per_chain_emb = {}
        for i, (chain_id, seq) in enumerate(sequence_list):
            L = len(seq)
            emb = token_reps[i, 1:L+1].mean(0).cpu().numpy()
            per_chain_emb[chain_id] = emb

        # aggregate per-partner by averaging chains belonging to the partner
        p1_embs = [per_chain_emb[c] for c in partner1_chains if c in per_chain_emb]
        p2_embs = [per_chain_emb[c] for c in partner2_chains if c in per_chain_emb]
        if p1_embs:
            p1_vec = np.mean(p1_embs, axis=0)
        else:
            p1_vec = np.zeros(token_reps.size(2), dtype=float)
        if p2_embs:
            p2_vec = np.mean(p2_embs, axis=0)
        else:
            p2_vec = np.zeros_like(p1_vec)

        esm_feature = np.concatenate([p1_vec, p2_vec])
        np.savetxt(os.path.join(self.filepath, self.PDBID + '_esm_features.txt'), esm_feature)
    
        return 

    # @classmethod # can modify class state that applies across state that applies across all instances
    def generate_sequences(self) -> List:

        from Bio.PDB.Polypeptide import PPBuilder

        sequence = []

        pdb_structure = parser_pdb(self.PDBID, self.filepath, self.partner1, self.partner2, default_cutoff) 
        continuous_binding_structure, partner1, partner2 = pdb_structure.read_continuous_binding_interface(partner_update=True)

        ppb = PPBuilder()
        for pp in ppb.build_peptides(continuous_binding_structure, aa_only=False):
            chain_ids = pp[0].parent.id # residue's parent is the chain
            if chain_ids in self.partner1 or chain_ids in self.partner2:
                if len(pp.get_sequence()) > 1022:
                    exit('The sequence is too long. Exit...')
                    # break the sequence into two parts
                    sequence.append((chain_ids+'_1', str(pp.get_sequence()[:1000])))
                    sequence.append((chain_ids+'_2', str(pp.get_sequence()[1000:])))
                    if len(pp.get_sequence()) > 2000:
                        exit('The sequence is too long. Exit...')
                else:
                    sequence.append((chain_ids, str(pp.get_sequence())))

        # check: for some cases, only one residue, need to remove that chain
        sequence_chains = []
        for sequence_pair in sequence:
            sequence_chains.append(sequence_pair[0])
        partner1 = [chain for chain in partner1 if chain in sequence_chains]
        partner2 = [chain for chain in partner2 if chain in sequence_chains]

        return sequence, partner1, partner2

    def _check_structure(self) -> None:
        pdb_file_partner1 = os.path.join(self.filepath, self.PDBID+'_'+self.partner1+'.pdb')
        pdb_file_partner2 = os.path.join(self.filepath, self.PDBID+'_'+self.partner2+'.pdb')

        parser1 = PDBParser(PERMISSIVE=True)
        structure1 = parser1.get_structure(self.PDBID, pdb_file_partner1)

        for chain in structure1[0]:
            if chain.id not in self.partner1:
                os.system(f'rm {self.PDBID}_{self.partner1}.*')

        parser2 = PDBParser(PERMISSIVE=True)
        structure2 = parser2.get_structure(self.PDBID, pdb_file_partner2)

        for chain in structure2[0]:
            if chain.id not in self.partner2:
                os.system(f'rm {self.PDBID}_{self.partner2}.*')

        pass

    def _generate_structure_files(self) -> None:

        pdb_file_name = os.path.join(self.filepath, self.PDBID+'.pdb')
        pdb_file_partner1 = os.path.join(self.filepath, self.PDBID+'_'+self.partner1+'.pdb')
        pdb_file_partner2 = os.path.join(self.filepath, self.PDBID+'_'+self.partner2+'.pdb')

        if not os.path.exists(pdb_file_name):
            os.system('wget https://files.rcsb.org/download/'+self.PDBID+'.pdb')
        pdb_file_name_backup = os.path.join(self.filepath, self.PDBID+'_backup.pdb')
        os.system(f'cp {pdb_file_name} {pdb_file_name_backup}')
        profix_marker_file = os.path.join(self.filepath, f"{self.PDBID}_profix_processing.txt")
        if os.path.exists(profix_marker_file):
            print(f"[INFO] Profix has been applied to {self.PDBID}, skip profix step.")
        else:
            print(f"[INFO] Running profix processing for {self.PDBID} ...")
            use_profix(self.PDBID, self.filepath)
            with open(profix_marker_file, 'w') as f:
                f.write("Profix processing completed.\n")

        pdb_structure = parser_pdb(self.PDBID, self.filepath, self.partner1, self.partner2, default_cutoff) 
        structure = pdb_structure.read_continuous_binding_interface()

        #parser = PDBParser(PERMISSIVE=True)
        #structure = parser.get_structure(self.PDBID, pdb_file_name)

        structure1, structure2 = self._clean_PDB(structure)

        # save the structures
        save_pdb(structure, pdb_file_name)
        save_pdb(structure1, pdb_file_partner1)
        save_pdb(structure2, pdb_file_partner2)

        # call pdb2pqr to generate the pqr files
        pqr_file_partner1 = os.path.join(self.filepath, self.PDBID+'_'+self.partner1+'.pqr')
        pqr_file_partner2 = os.path.join(self.filepath, self.PDBID+'_'+self.partner2+'.pqr')
        try:
            if not os.path.exists(pqr_file_partner1):
                os.system(f'pdb2pqr --ff=AMBER --titration-state-method=propka --keep-chain --with-ph=7.0 {pdb_file_partner1} {pqr_file_partner1}')
            if not os.path.exists(pqr_file_partner2):
                os.system(f'pdb2pqr --ff=AMBER --titration-state-method=propka --keep-chain --with-ph=7.0 {pdb_file_partner2} {pqr_file_partner2}')
        except Exception:
            exit('pdb2pqr is not installed, please install it first')

        pass

    def _clean_PDB(self, structure) -> Tuple[Structure, Structure]:
        '''
        Remove the water molecules and other chains that are not in partner1 and partner2
        '''
        model = structure[0]
        chains_to_remove = [chain.id for chain in model if chain.id not in self.partner1 + self.partner2]
        for chain_id in chains_to_remove:
            model.detach_child(chain_id)
        for chain in model:
            residues_to_remove = [residue.id for residue in chain if residue.id[0] != ' ']
            for residue_id in residues_to_remove:
                chain.detach_child(residue_id)

        # get two structures of partner1 and partner2
        structure1 = structure.copy()
        structure2 = structure.copy()
        detach_chain1 = [chain.id for chain in structure1[0] if chain.id not in self.partner1]
        detach_chain2 = [chain.id for chain in structure2[0] if chain.id not in self.partner2]
        for chain in detach_chain1:
            structure1[0].detach_child(chain)
        for chain in detach_chain2:
            structure2[0].detach_child(chain)

        return structure1, structure2

    def biophysics_features(self, partner) -> np.ndarray:

        # partner: PDBID_ChainIDs.pqr, so as mibpb, pKa, etc.
        filename = self.PDBID + '_' + partner
        pqr_structure = parser_pqr(self.PDBID, self.filepath, filename=filename, chain=partner)

        feature_global = pqr_structure.get_feature_global()
        feature_pKa = pqr_structure.get_feature_pKa()
        feature_electrostatic = pqr_structure.get_feature_electrostatics()

        return np.concatenate((feature_global, feature_pKa, feature_electrostatic), axis=0, dtype=np.float32)

    def persistent_homology_features(self) -> Tuple[np.ndarray, np.ndarray]:

        structure_binding = parser_pdb(self.PDBID, self.filepath, self.partner1, self.partner2, default_cutoff) 

        atoms_es_1 = structure_binding.atoms_element_specific_PPI1 # atoms of protein 1, only consider the 'C', 'N', 'O'
        atoms_es_2 = structure_binding.atoms_element_specific_PPI2 # atoms of protein 2

        args_es = {'length': EleLength, 'atoms_partner1': atoms_es_1, 'atoms_partner2': atoms_es_2}
        feature_rips_es = generate_rips_complex(**args_es)
        feature_alpha_es = generate_alpha_shape(**args_es)
        feature_fri_es = generate_flexibility_rigidy_index(**args_es)
        feature_spectra_es = generate_persistent_spectra(**args_es)

        feature_es = np.concatenate((feature_rips_es, feature_spectra_es, feature_alpha_es, feature_fri_es), axis=0)

        atoms_rs_1 = structure_binding.atoms_residue_type_specific_PPI1 # atoms of protein 1
        atoms_rs_2 = structure_binding.atoms_residue_type_specific_PPI2 # atoms of protein 2

        args_aa = {'length': 5, 'atoms_partner1': atoms_rs_1, 'atoms_partner2': atoms_rs_2}
        feature_rips_aa = generate_rips_complex(**args_aa)
        feature_alpha_aa = generate_alpha_shape(**args_aa)
        # feature_fri_aa = generate_flexibility_rigidy_index(**args_aa)
        args_aa['bins'] = [5, 7, 9, 11]
        feature_spectra_aa = generate_persistent_spectra(**args_aa)

        feature_aa = np.concatenate((feature_rips_aa, feature_spectra_aa, feature_alpha_aa), axis=0)

        return feature_es, feature_aa


    # def concatenate(self):
    #     Feature = np.concatenate((self.rips_dth, self.rips_bar), axis=1)
    #     #print(Feature.flatten().shape)
    #     Feature = Feature.flatten()
    #     #print(self.features)
    #     self.Feature = np.concatenate((Feature, self.features))
    #     self.Feature = self.Feature.reshape(1,len(self.Feature))
    #     #print('laplacian is',self.features.shape)
    #     #print(Feature.shape)

def collect_PDB_feature_to_table(dataname:str = 'PDBbind_V2020_PP'):
    """collect the fundamental features of the dataset, 
    including biophysics, PPI, ESM, and persistent homology features."""
    
    data_partnerA_biophysics = []
    data_partnerB_biophysics = []
    data_esm = []
    data_PPI_aa = []
    data_PPI_es = []
    if dataname == 'PDBbind_V2020_PP':
        df = TABLE_PDBBIND_V2020_PP
        root_folder = os.path.join(DIR_DATA, 'PDBbind_V2020_PP_Table')
    elif dataname == 'SKEMPI_v2_WT':
        df = TABLE_SKEMPI_V2_WT
        root_folder = os.path.join(DIR_DATA, 'SKEMPI_v2_WT_Table')
    elif dataname == 'SKEMPI_v2_MT':
        df = TABLE_SKEMPI_V2_MT
        root_folder = os.path.join(DIR_DATA, 'SKEMPI_v2_MT_Table')
    else:
        raise ValueError(f"Unsupported data type: {dataname}. Supported types are 'PDBbind_V2020_PP', 'SKEMPI_v2_WT', and 'SKEMPI_v2_MT'.")
    os.makedirs(root_folder, exist_ok=True)
    RESULT_PATH = os.path.join(DIR_DATA, dataname)

    for subdir, dirs, files in os.walk(RESULT_PATH):
        for file in files:
            if file.endswith('.txt'): # Only process .txt files
                file_path = os.path.join(subdir, file)
                folder_name = os.path.basename(subdir)
                file_name = os.path.basename(file)  # Get the file name
                print('Processing File'.center(60, '='))
                print(f"Processing folder: {folder_name}, file: {file_name}")
                print('='*60)
                if file_name == '1BP3_YA111V_LA113I_KA115E_DA116Q_EA118K_EA119R_RA120L_QA122E_TA123G_GA126L_RA127I_EA129S':
                    continue
                
                selected_ID_table = df[df['ID'] == folder_name]
                partner1 = list(selected_ID_table['partner1'])[0]
                partner2 = list(selected_ID_table['partner2'])[0]
                print(f"Partner1: {partner1}, Partner2: {partner2}")
                error_ID = []
                try:
                    if partner1+"_biophysics" in file_name:
                        with open(file_path, 'r') as f:
                            content = f.read().strip()  # Read the content of the text file
                            features = content.splitlines()                            
                            feature_dict = {'ID': folder_name}
                            for i, feature in enumerate(features):
                                feature_dict[f'feature_{i+1}'] = feature
                            
                            data_partnerA_biophysics.append(feature_dict)
                    
                    elif partner2+"_biophysics" in file_name:
                        with open(file_path, 'r') as f:
                            content = f.read().strip()  # Read the content of the text file
                            features = content.splitlines()                       
                            feature_dict = {'ID': folder_name}
                            for i, feature in enumerate(features):
                                feature_dict[f'feature_{i+1}'] = feature
                            data_partnerB_biophysics.append(feature_dict)
                    
                    elif "esm" in file_name:
                        with open(file_path, 'r') as f:
                            content = f.read().strip()  # Read the content of the text file
                            features = content.splitlines()                            
                            feature_dict = {'ID': folder_name}
                            for i, feature in enumerate(features):
                                feature_dict[f'feature_{i+1}'] = feature
                            data_esm.append(feature_dict)
                    
                    elif "persistent_homology_aa.txt" in file_name:
                        with open(file_path, 'r') as f:
                            content = f.read().strip()  # Read the content of the text file
                            features = content.splitlines()
                            feature_dict = {'ID': folder_name}
                            for i, feature in enumerate(features):
                                feature_dict[f'feature_{i+1}'] = feature
                            data_PPI_aa.append(feature_dict)

                    elif "persistent_homology_es" in file_name:
                        with open(file_path, 'r') as f:
                            content = f.read().strip()  # Read the content of the text file
                            features = content.splitlines()
                            feature_dict = {'ID': folder_name}
                            for i, feature in enumerate(features):
                                feature_dict[f'feature_{i+1}'] = feature
                            data_PPI_es.append(feature_dict)

                except ValueError as e:
                    print(f"Error: {e}")
                    error_ID.append(file_name)
    
   

    dd_partnerA_biophysics = pd.DataFrame(data_partnerA_biophysics)
    dd_partnerA_biophysics.to_csv(os.path.join(root_folder,"feature_partnerA_biophysics_"+dataname+".csv"))

    dd_partnerB_biophysics = pd.DataFrame(data_partnerB_biophysics)
    dd_partnerB_biophysics.to_csv(os.path.join(root_folder,"feature_partnerB_biophysics_"+dataname+".csv"))
    dd_esm = pd.DataFrame(data_esm)
    dd_esm.to_csv(os.path.join(root_folder,"feature_esm_"+dataname+".csv"))
    dd_PPI_aa = pd.DataFrame(data_PPI_aa)
    dd_PPI_aa.to_csv(os.path.join(root_folder,"feature_PPI_aa_"+dataname+".csv"))  
    dd_PPI_es = pd.DataFrame(data_PPI_es)
    dd_PPI_es.to_csv(os.path.join(root_folder,"feature_PPI_es_"+dataname+".csv"))
    return



def load_training_features(features:str='PPI',
                            dataname:str='PDBbind_V2020_PP')->Tuple[np.ndarray,np.ndarray]:
    """
    Load training features and labels from pre-computed feature tables.
    
    Args:
        features: Feature type to load ('biophysics', 'esm', 'PPI', or 'All')
        dataset: Dataset name ('V2020', 'WT', or 'MT')
        
    Returns:
        tuple: (features array, labels array, ID array)
    """
    if dataname == 'PDBbind_V2020_PP':
        data = TABLE_PDBBIND_V2020_PP.reset_index(drop=True)
        TABLE_PATH = os.path.join(DIR_DATA,'V2020_Table')     
    elif dataname == 'SKEMPI_v2_WT':
        data = TABLE_SKEMPI_V2_WT.reset_index(drop=True)
        TABLE_PATH = os.path.join(DIR_DATA,'WT_Table')
    elif dataname == 'SKEMPI_v2_MT':
        data = TABLE_SKEMPI_V2_MT.reset_index(drop=True)
        TABLE_PATH = os.path.join(DIR_DATA,'MT_Table')
    else:
        raise ValueError(f"Unsupported data type: {dataname}")

    # Get IDs and target values
    id_target_df = data[['ID', 'DDG']].copy()
        
    # Load feature files based on feature type
    
    dfs = {
            'partnerA_bio': pd.read_csv(os.path.join(TABLE_PATH, f"dd_partnerA_biophysics_{dataname}.csv")),
            'partnerB_bio': pd.read_csv(os.path.join(TABLE_PATH, f"dd_partnerB_biophysics_{dataname}.csv")),
            'ppi_aa': pd.read_csv(os.path.join(TABLE_PATH, f"dd_PPI_aa_{dataname}.csv")),
            'ppi_es': pd.read_csv(os.path.join(TABLE_PATH, f"dd_PPI_es_{dataname}.csv")),
            'esm': pd.read_csv(os.path.join(TABLE_PATH, f"dd_esm_{dataname}.csv"))
            }
    try:
        if features == 'biophysics':
            # Filter and merge
            dfs['partnerA_bio'] = dfs['partnerA_bio'][dfs['partnerA_bio']['ID'].isin(id_target_df['ID'])].loc[:, ~dfs['partnerA_bio'].columns.str.contains('^Unnamed')]
            dfs['partnerB_bio'] = dfs['partnerB_bio'][dfs['partnerB_bio']['ID'].isin(id_target_df['ID'])].loc[:, ~dfs['partnerB_bio'].columns.str.contains('^Unnamed')]
            # Merge data
            result_df = pd.merge(dfs['partnerA_bio'], dfs['partnerB_bio'], on='ID', suffixes=('_p1', '_p2'))
            result_df = pd.merge(result_df, id_target_df, on='ID')

        elif features == 'esm':
            dfs['esm'] = pd.read_csv(os.path.join(TABLE_PATH, f"dd_esm_{dataname}.csv"))
            dfs['esm'] = dfs['esm'][dfs['esm']['ID'].isin(id_target_df['ID'])].loc[:, ~dfs['esm'].columns.str.contains('^Unnamed')]
            # Merge with targets
            result_df = pd.merge(dfs['esm'], id_target_df, on='ID')

        elif features == 'PPI':
             # Load PPI features (amino acid and element specific)
            dfs['ppi_aa'] = pd.read_csv(os.path.join(TABLE_PATH, f"dd_PPI_aa_{dataname}.csv"))
            dfs['ppi_es'] = pd.read_csv(os.path.join(TABLE_PATH, f"dd_PPI_es_{dataname}.csv"))
            # Filter and clean
            dfs['ppi_aa'] = dfs['ppi_aa'][dfs['ppi_aa']['ID'].isin(id_target_df['ID'])].loc[:, ~dfs['ppi_aa'].columns.str.contains('^Unnamed')]
            dfs['ppi_es'] = dfs['ppi_es'][dfs['ppi_es']['ID'].isin(id_target_df['ID'])].loc[:, ~dfs['ppi_es'].columns.str.contains('^Unnamed')]
            # Merge data
            result_df = pd.merge(dfs['ppi_aa'], dfs['ppi_es'], on='ID', suffixes=('_aa', '_es'))
            result_df = pd.merge(result_df, id_target_df, on='ID')

        elif features == 'All':
            # Clean all dataframes
            for key, df in dfs.items():
                dfs[key] = df[df['ID'].isin(id_target_df['ID'])].loc[:, ~df.columns.str.contains('^Unnamed')]
            
            # Merge all dataframes
            result_df = pd.merge(dfs['partnerA_bio'], dfs['partnerB_bio'], on='ID', suffixes=('_p1', '_p2'))
            result_df = pd.merge(result_df, dfs['ppi_aa'], on='ID')
            result_df = pd.merge(result_df, dfs['ppi_es'], on='ID', suffixes=('', '_es'))
            result_df = pd.merge(result_df, dfs['esm'], on='ID')
            result_df = pd.merge(result_df, id_target_df, on='ID')
        else:
            raise ValueError(f"Unsupported features: {features}. Must be one of: 'biophysics', 'esm', 'PPI', 'All'")

        # Extract feature columns and target values
        print(f'Merged DataFrame shape: {result_df.shape}')
        feature_columns = [col for col in result_df.columns if col.startswith('feature_')]
        X = result_df[feature_columns].values
        y = result_df['DDG'].values
        ids = result_df['ID'].values
        
        return X, y, ids     
    except FileNotFoundError as e:
        print(f"Error: Required feature file not found: {e}")
        print(f"Make sure to run feature generation for {dataname} dataset first.")
        raise
    except Exception as e:
        print(f"Error loading features: {e}")
        raise













if __name__ == '__main__':
    print("[INFO] This is a feature parser module for protein-protein interaction analysis.")

    pdbID = ["1A22"] #
    for pdb in pdbID:
        # Get partner information from config tables
        selected_ID_table = TABLE_SKEMPI_V2_WT[TABLE_SKEMPI_V2_WT['ID'] == pdb]
        if not selected_ID_table.empty:
            partner1 = list(selected_ID_table['partnerA'])[0]
            partner2 = list(selected_ID_table['partnerB'])[0]
            

            pc = protein_complex(PDBID=pdb,
                                 filepath=os.path.join(DIR_DATA, "Example"),
                                 partner1=partner1,
                                 partner2=partner2,
                                 generate_structure_files=True,
                                 check_structure=True,
                                 persistent_homology=True, 
                                 read_features=False,
                                 sequence=True)
        else:
            print(f"No partner information found for {pdb}")

