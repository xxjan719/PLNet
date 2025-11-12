"""
Author: Xingjian Xu
Email: xingjianxu@ufl.edu
Data: 06/04/2024
LastEditTime: 07/17/2024

Comments:
"""
import os, sys
import warnings
import numpy as np
from typing import Tuple, Optional
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Structure import Structure
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.PDBExceptions import PDBConstructionWarning

warnings.filterwarnings("ignore", category=PDBConstructionWarning)

try:
    from .constant import default_cutoff, ElementList, EleLength, AminoA, AALength, NC_AminoA, AminoA_index
    from .constant import pKa_index, pKa_group, atmtyp_to_ele, three_to_one, ele2index
except:
    from constant import default_cutoff, ElementList, EleLength, AminoA, AALength, NC_AminoA, AminoA_index
    from constant import pKa_index, pKa_group, atmtyp_to_ele, three_to_one, ele2index

# Import config flag for mibpb features
try:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from config import ENABLE_MIBPB_FEATURES
except:
    # Fallback: assume enabled if config not available
    ENABLE_MIBPB_FEATURES = True

class my_atom:
    def __init__(self, position=None, chain=None, resID=None, atype=None, 
                 avtype=None, charge=None, resName=None, radii=None, **kwargs):
        self.pos = np.array(position) if position else None
        self.chain = chain
        self.resID = resID
        self.atype = atype    

        # Properties specific to the verbose description of an Atom
        self.verboseType = avtype
        self.charge = charge 
        self.resName = resName
        self.radii = radii

        # Set defaults for any unspecified properties
        self.Area = kwargs.get('Area', 0.0)
        self.SolvEng = kwargs.get('SolvEng', 0.0)

        # Additional attributes from kwargs
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)

    def position(self, position: list):
        self.pos = np.array(position)

    def atom_energy(self, SolvEng: float):
        self.SolvEng = SolvEng

    def atom_area(self, Area: float):
        self.Area = Area

    def print_atom(self):
        print('Atom: chain: {}, resID: {}, atype: {}, charge: {}, resName: {}, radii: {}, Area: {}, SolvEng: {}'.format(
            self.chain, self.resID, self.atype, self.charge, self.resName, self.radii, self.Area, self.SolvEng))

class parser_pqr:
    def __init__(self, PDBID: str, filepath: str, filename = None, chain = None):
        self.PDBID = PDBID
        self.filepath = filepath
        self.filename = filename
        self.chain = chain

        if self.filename is None:
            self.file_name = os.path.join(self.filepath, self.PDBID)
        else:
            self.file_name = os.path.join(self.filepath, self.filename)

        self._read_pqr() # read the pqr file and collect the atoms
        self._construct_index_list() # construct the index list

    def _read_pqr(self):
        self.atoms = []
        self.charge = []

        f = open(self.file_name+'.pqr', 'r')
        for line in f.readlines():
            if line.startswith('ATOM'):
                if self.chain is not None and line[21] not in list(self.chain):
                    continue
                atom = my_atom(position=[float(line[26:38]), float(line[38:46]), float(line[46:54])], chain=line[21], 
                               atype=atmtyp_to_ele(line[12:14]), avtype=line[11:17], charge=float(line[54:62]), 
                               resName=three_to_one(line[17:20]), resID=int(line[22:26]), radii=float(line[62:69]))
                self.atoms.append(atom)
                self.charge.append(atom.charge)
        f.close()

        self.charge = np.array(self.charge)
        return self.atoms

    def _pairwise_interaction(self, sCut=10., lCut=40
                             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        print('setup pairwise interaction >>>>>>>>>>>>>>>>>>>>>>>>')
        from pyprotein import PyProtein

        PyProt = PyProtein(self.PDBID.encode('utf-8'))
        if not PyProt.loadPQRFile((self.file_name+'.pqr').encode('utf-8')):
            sys.exit('PyProtein reads PQR file filed')
        PyProt.atomwise_interaction(sCut, lCut)
        CLB = PyProt.feature_CLB()
        VDW = PyProt.feature_VDW()
        FRI1, FRI2, FRI3, FRI4, FRI5 = PyProt.feature_FRIs()
        PyProt.Deallocate()

        return CLB, VDW, FRI1, FRI2, FRI3, FRI4, FRI5

    def _construct_index_list(self):
        """ 
        Lists that contains atom index
            first index: 0 mutsite, 1 other near, 2 all
            second index: 0 C, 1 N, 2 O, 3 S, 4 H, 5 heavy, 6 all
        """
        print('constructing index list >>>>>>>>>>>>>>>>>>>>>>>>>>>')
        heavy = ['C', 'N', 'O', 'S']
        index_list = [[] for _ in range(7)]

        # Index of all atoms
        for idx, iAtom in enumerate(self.atoms):
            index_list[ele2index[self.atoms[idx].atype]].append(idx)
            if iAtom.atype in heavy:
                index_list[5].append(idx)
            index_list[6].append(idx)
        self.index_list = index_list
        return self.index_list

    def get_feature_global(self) -> np.ndarray:
        print(f'construct the global features of chains {self.chain} >>>>>>>>>>>>')
        index_array = [np.array([0], int),
                       np.array([1], int),
                       np.array([2], int),
                       np.array([3], int),
                       np.array([4], int),
                       np.array([0,1,2,3], int),
                       np.array([0,1,2,3,4], int)]
        feature_global = []
        for j in range(7):
            feature_global.append(np.sum(self.charge[self.index_list[j]]))
            feature_global.append(np.sum(np.abs(self.charge[self.index_list[j]])))

        CLB, VDW, FRI1, FRI2, FRI3, FRI4, FRI5 = self._pairwise_interaction()
        for j in [0,1,2,3,5]:
            feature_global.append(np.sum(FRI1[self.index_list[j], :][:, index_array[5]]))
            feature_global.append(np.sum(FRI2[self.index_list[j], :][:, index_array[5]]))
            feature_global.append(np.sum(FRI3[self.index_list[j], :][:, index_array[5]]))
            feature_global.append(np.sum(FRI4[self.index_list[j], :][:, index_array[5]]))
            feature_global.append(np.sum(FRI5[self.index_list[j], :][:, index_array[5]]))
            feature_global.append(np.sum(VDW[self.index_list[j], :][:, index_array[5]]))
            feature_global.append(np.sum(CLB[self.index_list[j], :][:, index_array[6]]))
            feature_global.append(np.sum(np.abs(CLB[self.index_list[j], :][:, index_array[6]])))

        # NOTE: in the original code, there is a FeatureGLBother, 
        # which only contains the residue description of the mutsite.
        # However, it can be easily used for the graph neural network.
        return np.array(feature_global)

    def get_feature_pKa(self) -> np.ndarray:
        print('get pKa information >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

        pKa_file = open(self.file_name+'.propka', 'r')
        pKa = []; pKa_name = []; pKa_Ct = 0.; pKa_Nt = 0.
        for line in pKa_file:
            if len(line) < 24:
                continue
            if line[23] == '%':
                if line[0:2] == 'C-': pKa_Ct = float(line[11:16])
                if line[0:2] == 'N+': pKa_Nt = float(line[11:16])
                pKa.append(float(line[11:16]))
                pKa_name.append(line[0:3])
        pKa_file.close()
        pKa = np.array(pKa)

        # Assemble the pKa features
        pKa_features = []
        pKa_features.append(pKa.max())
        pKa_features.append(pKa.min())
        pKa_features.append(pKa.mean())
        pKa_features.append(np.abs(pKa.max()))
        pKa_features.append(np.abs(pKa.min()))
        pKa_features.append(np.abs(pKa.mean()))
        pKa_features.append(pKa.std())
        pKa_features.append(pKa_Ct)
        pKa_features.append(pKa_Nt)

        detail_Abs = np.zeros([7], float)
        detail_Net = np.zeros([7], float)
        for idx in range(pKa.shape[0]):
            if pKa_name[idx] in pKa_group:
                detail_Abs[pKa_index[pKa_name[idx]]] += np.abs(pKa[idx])
                detail_Net[pKa_index[pKa_name[idx]]] += pKa[idx]
        pKa_features.extend(detail_Abs)
        pKa_features.extend(detail_Net)
        return np.array(pKa_features)

    def get_feature_electrostatics(self, h: float = 0.5) -> Optional[np.ndarray]:
        # Check if mibpb features are enabled
        if not ENABLE_MIBPB_FEATURES:
            print('[INFO] MIBPB features disabled - returning None for electrostatics features')
            # Return None when MIBPB features are disabled
            return None

        print('run MIBPB >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        if not os.path.exists(self.file_name+'.englist') or not os.path.exists(self.file_name+'.eng') or \
           not os.path.exists(self.file_name+'.arealist') or not os.path.exists(self.file_name+'.areavolume'):
            os.system('rm -f '+self.file_name+'.englist')
            os.system('rm -f '+self.file_name+'.eng')
            os.system('rm -f '+self.file_name+'.arealist')
            os.system('rm -f '+self.file_name+'.areavolume')
            try:
                os.system('mibpb5 '+self.file_name+' h=%f'%(h))
            except:
                exit('Error: MIBPB failed. Exiting.')
            os.system('mv partition_area.txt '+self.file_name+'.arealist')
            os.system('mv area_volume.dat '+self.file_name+'.areavolume')
        os.system('rm -f bounding_box.txt')
        os.system('rm -f grid_info.txt')
        os.system('rm -f intersection_info.txt')
        os.system('rm -f '+self.file_name+'.dx')

        # ===================== read area list of ESES ======================
        area_list_file = open(self.file_name+'.arealist', 'r')

        atom_area = np.zeros([len(self.atoms)], float)
        line_count = 0
        for idx, line in enumerate(area_list_file):
            _, value = line.split()
            # self.atoms[idx].atom_area(float(value))
            atom_area[idx] = float(value)
            line_count += 1

        # check the number of lines in the file is the same as the number of atoms
        if line_count != len(self.atoms):
            exit(f'Error: number of atoms in the pqr file does not match MIBPB output file,\n \
                   {self.file_name}.arealist.\nExiting...')

        area_list_file.close()
        # ===================================================================

        # ==================== read energy list of MIBPB ====================
        energy_list_file = open(self.file_name+'.englist', 'r')

        atom_solvation_energy = np.zeros([len(self.atoms)], float)
        line_count = 0
        for idx, line in enumerate(energy_list_file):
            # self.atoms[idx].atom_energy(float(line))
            atom_solvation_energy[idx] = float(line)
            line_count += 1

        # check the number of lines in the file is the same as the number of atoms
        if line_count != len(self.atoms):
            exit(f'Error: number of atoms in the pqr file does not match MIBPB output file,\n \
                   {self.file_name}.englist.\nExiting...')

        energy_list_file.close()
        # ===================================================================

        # ================= collect area, volume and energy =================
        area_volume_file = open(self.file_name+'.areavolume', 'r')
        total_area = float(area_volume_file.readline())
        total_volume = float(area_volume_file.readline())
        area_volume_file.close()

        energy_file = open(self.file_name+'.eng', 'r')
        energy_file.readline()
        total_energy = float(energy_file.readline())
        energy_file.close()
        # ===================================================================

        feature_electrostatics = []
        for j in range(7):
            feature_electrostatics.append(np.sum(atom_solvation_energy[self.index_list[j]]))
            feature_electrostatics.append(np.sum(atom_area[self.index_list[j]]))
        feature_electrostatics.append(total_energy)
        feature_electrostatics.append(total_area)
        feature_electrostatics.append(total_volume)

        return np.array(feature_electrostatics)

def test_parser_pqr(PDBID, filepath):
    test_parser = parser_pqr(PDBID, filepath, chain='C')
    # test_parser.pairwise_interaction()
    test_parser.get_feature_global()
    test_parser.get_feature_pKa()
    pass
 
class parser_pdb:
    '''
    This class reads the pdb file by giving the PDBID and the path of the pdb file.
    '''
    def __init__(self, PDBID: str, filepath: str, partner1: str, partner2: str, cutoff: int = default_cutoff):
        self.PDBID = PDBID
        self.cutoff = cutoff

        parser = PDBParser(PERMISSIVE=True)
        pdb_file_name = os.path.join(filepath, self.PDBID+'.pdb') # check if with other extension, change that first
        # get PDB_File and fasta_File
        self.structure = parser.get_structure(self.PDBID, pdb_file_name)

        # update the partner1 and partner2
        self.partner1 = [str(chain.id) for chain in self.structure[0] if chain.id in partner1]
        self.partner2 = [str(chain.id) for chain in self.structure[0] if chain.id in partner2]
        print('Updated partner1: {}; partner2: {}'.format(self.partner1, self.partner2))

        self._read_binding_interface()
        # self._save_binding_interface('./test.pdb')
        self._load_binding_interface_element_specific()
        self._load_binding_interface_residue_type_specific()

    def read_continuous_binding_interface(self, partner_update=False):
        def __remove_non_interacting_residues(structure, primary_partner, secondary_partner):
            chains_to_remove = []
            for ichain in primary_partner:
                interacting_indices = {}
                for i, iresidue in enumerate(structure[ichain]):
                    if 'CA' not in iresidue:
                        continue
                    for jchain in secondary_partner:
                        for jresidue in structure[jchain]:
                            if 'CA' not in jresidue: 
                                continue
                            dist = jresidue['CA'] - iresidue['CA']
                            if dist < self.cutoff:
                                interacting_indices[i] = iresidue.id
                                break
                        if i in interacting_indices:
                            break

                if not interacting_indices:
                    chains_to_remove.append(ichain)
                    continue
                first_interacting = min(interacting_indices)
                last_interacting = max(interacting_indices)

                residues_to_remove = []
                for i, residue in enumerate(structure[ichain]):
                    if i < first_interacting or i > last_interacting:
                        residues_to_remove.append(residue.id)

                for residue_index in residues_to_remove:
                    structure[ichain].detach_child(residue_index)
            return chains_to_remove

        _continuous_binding_structure = self.structure.copy()
        # Remove chains not in partner1 and partner2
        _model = _continuous_binding_structure[0]
        _chains_to_remove = [chain.id for chain in _model if chain.id not in self.partner1 + self.partner2]
        for chain_id in _chains_to_remove:
            _model.detach_child(chain_id)
        for chain in _model:
            residues_to_remove = [residue.id for residue in chain if residue.id[0] != ' ']
            for residue_id in residues_to_remove:
                chain.detach_child(residue_id)

        # Call the helper function for both partners
        chains_to_remove = __remove_non_interacting_residues(_model, self.partner1, self.partner2)
        chains_to_remove += __remove_non_interacting_residues(_model, self.partner2, self.partner1)
        if chains_to_remove:
            for chain_id in chains_to_remove:
                _model.detach_child(chain_id)

        if partner_update:
            partner1_updated = [str(chain.id) for chain in _continuous_binding_structure[0] if chain.id in self.partner1]
            partner2_updated = [str(chain.id) for chain in _continuous_binding_structure[0] if chain.id in self.partner2]
            print('Updated partner1: {}; partner2: {}'.format(partner1_updated, partner2_updated))
            return _continuous_binding_structure, partner1_updated, partner2_updated

        return _continuous_binding_structure

    def _read_binding_interface(self) -> None:
        self.binding_structure = self.structure.copy()
        # Remove chains not in partner1 and partner2
        model = self.binding_structure[0]
        chains_to_remove = [chain.id for chain in model if chain.id not in self.partner1 + self.partner2]
        for chain_id in chains_to_remove:
            model.detach_child(chain_id)
        for chain in model:
            residues_to_remove = [residue.id for residue in chain if residue.id[0] != ' ']
            for residue_id in residues_to_remove:
                chain.detach_child(residue_id)

        def __remove_non_interacting_residues(primary_partner, secondary_partner):
            for ichain in primary_partner:
                iresidue_ids_to_remove = []
                for iresidue in self.binding_structure[0][ichain]:
                    if 'CA' not in iresidue: continue
                    flag_not_remove = False
                    for jchain in secondary_partner:
                        for jresidue in self.binding_structure[0][jchain]:
                            if 'CA' not in jresidue: continue
                            dist = jresidue['CA'] - iresidue['CA']
                            if dist < self.cutoff:
                                flag_not_remove = True
                                break
                        if flag_not_remove:
                            break
                    if not flag_not_remove:
                        iresidue_ids_to_remove.append(iresidue.id)
                for iresidue_id in iresidue_ids_to_remove:
                    self.binding_structure[0][ichain].detach_child(iresidue_id)

        # Call the helper function for both partners
        __remove_non_interacting_residues(self.partner1, self.partner2)
        __remove_non_interacting_residues(self.partner2, self.partner1)
        pass

    def _save_binding_interface(self, save_file_path) -> None: # move to data_extraction.py
        '''
        Needed if we want to save the binding interface as a pdb file and then load it again
        '''
        io = PDBIO()
        io.set_structure(self.binding_structure)
        io.save(save_file_path)
        pass

    def _load_binding_interface_element_specific(self) -> None:
        self.atoms_element_specific_PPI1 = [[] for _ in range(EleLength)]
        self.atoms_element_specific_PPI2 = [[] for _ in range(EleLength)]

        for chain in self.binding_structure[0]:
            for residue in chain:
                for atom in residue:
                    if atom.element in ElementList: # C N O, exclude H, S, etc.
                        _atom = my_atom(atom.coord.tolist(), chain.id, residue.id[1], atom.element)
                        if chain.id in self.partner1:
                            self.atoms_element_specific_PPI1[ElementList.index(_atom.atype)].append(_atom)
                        elif chain.id in self.partner2:
                            self.atoms_element_specific_PPI2[ElementList.index(_atom.atype)].append(_atom)

        # print('partner1 #atoms: C:{}; N:{}; O:{}'.format(len(self.atoms_element_specific_PPI1[0]),len(self.atoms_element_specific_PPI1[1]),len(self.atoms_element_specific_PPI1[2])))
        # print('partner2 #atoms: C:{}; N:{}; O:{}'.format(len(self.atoms_element_specific_PPI2[0]),len(self.atoms_element_specific_PPI2[1]),len(self.atoms_element_specific_PPI2[2])))
        pass

    def _load_binding_interface_residue_specific(self) -> None:
        # This method creates a very large list, which causes overfitting
        self.atoms_residue_specific_PPI1 = [[] for _ in range(AALength)]
        self.atoms_residue_specific_PPI2 = [[] for _ in range(AALength)]

        for chain in self.binding_structure[0]:
            for residue in chain:
                #print(residue.resname, residue.id)
                residue_name = residue.resname
                if residue_name not in AminoA:
                    residue_name = NC_AminoA[residue_name]
                for atom in residue:
                    if atom.element in ElementList:
                        _atom = my_atom(atom.coord.tolist(), chain.id, residue.id[1], atom.element)
                        if chain.id in self.partner1:
                            self.atoms_residue_specific_PPI1[AminoA.index(residue_name)].append(_atom)
                        elif chain.id in self.partner2:
                            self.atoms_residue_specific_PPI2[AminoA.index(residue_name)].append(_atom)
        pass

    def _load_binding_interface_residue_type_specific(self) -> None:
        self.atoms_residue_type_specific_PPI1 = [[] for _ in range(5)]
        self.atoms_residue_type_specific_PPI2 = [[] for _ in range(5)]

        for chain in self.binding_structure[0]:
            for residue in chain:
                #print(residue.resname, residue.id)
                residue_name = residue.resname
                if residue_name not in AminoA:
                    residue_name = NC_AminoA[residue_name]
                residue_index = AminoA_index[residue_name]
                for atom in residue:
                    if atom.element in ElementList:
                        _atom = my_atom(atom.coord.tolist(), chain.id, residue.id[1], atom.element)
                        if chain.id in self.partner1:
                            self.atoms_residue_type_specific_PPI1[residue_index].append(_atom)
                        elif chain.id in self.partner2:
                            self.atoms_residue_type_specific_PPI2[residue_index].append(_atom)
        # print('partner1 #atoms: {}'.format([len(i) for i in self.atoms_residue_specific_PPI1]))
        # print('partner2 #atoms: {}'.format([len(i) for i in self.atoms_residue_specific_PPI2]))
        pass

def test_parser_pdb():
    PDBID = '1A22'
    filepath = '../../data/Example'
    partner1 = 'A'
    partner2 = 'B'
    test_pdb = parser_pdb(PDBID, filepath, partner1, partner2, 16)
    pass

if __name__ =="__main__":
    #test_parser_pqr('1DVF', '.')
    test_parser_pdb()
