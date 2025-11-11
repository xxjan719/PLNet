import numpy as np

# pKa related constants for electrostatics calculations
pKa_index = {'ASP':0, 'GLU':1, 'ARG':2, 'LYS':3, 'HIS':4, 'CYS':5, 'TYR':6}
pKa_group = ['ASP',   'GLU',   'ARG',   'LYS',   'HIS',   'CYS',   'TYR']

# Atom type to element mapping for PQR files
def atmtyp_to_ele(st):
    if len(st.strip()) == 1:
        return st.strip()
    elif st[0] == 'H':
        return 'H'
    elif st == "CA":
        return "CA"
    elif st == "CL":
        return "CL"
    elif st == "BR":
        return "BR"
    else:
        print(st, 'Not in dictionary')
        return

def AAcharge(AA):
    if AA in ['D','E']:
        return -1.
    elif AA in ['R','H','K']:
        return 1.
    else:
        return 0.
default_cutoff = 16
AminoA = ['ARG','HIS','LYS','ASP','GLU','SER','THR','ASN','GLN','CYS',
          'SEF','GLY','PRO','ALA','VAL','ILE','LEU','MET','PHE','TYR',
          'TRP']

AMINO_ACIDS = ['ARG', 'HIS', 'LYS', 'ASP', 'GLU', 'SER', 'THR', 'ASN', 'GLN', 'CYS',
               'SEF', 'GLY', 'PRO', 'ALA', 'VAL', 'ILE', 'LEU', 'MET', 'PHE', 'TYR', 'TRP']


# Non-canonical amino acid mappings
NON_CANONICAL_AA = {
    'LLP': 'LYS', 'M3P': 'LYS', 'MSE': 'MET', 'F2F': 'PHE', 'CGU': 'GLU',
    'MYL': 'LYS', 'TPO': 'THR', 'HSE': 'HIS'
}


AminoA_index = {
    'ARG': 2, 'HIS': 2, 'LYS': 2, 'ASP': 3, 'GLU': 3,
    'SER': 1, 'THR': 1, 'ASN': 1, 'GLN': 1, 'CYS': 4,
    'SEF': 4, 'GLY': 4, 'PRO': 4, 'ALA': 0, 'VAL': 0,
    'ILE': 0, 'LEU': 0, 'MET': 0, 'PHE': 0, 'TYR': 0,
    'TRP': 0
}

AminoA_mapping = {
    'ARG': 'R', 'HIS': 'H', 'LYS': 'K', 'ASP': 'D', 'GLU': 'E',
    'SER': 'S', 'THR': 'T', 'ASN': 'N', 'GLN': 'Q', 'CYS': 'C',
    'SEF': 'U', 'GLY': 'G', 'PRO': 'P', 'ALA': 'A', 'VAL': 'V',
    'ILE': 'I', 'LEU': 'L', 'MET': 'M', 'PHE': 'F', 'TYR': 'Y',
    'TRP': 'W', 'UNK': 'G'
}
NC_AminoA = {'LLP':'LYS','M3P':'LYS','MSE':'MET','F2F':'PHE','CGU':'GLU',
        'MYL':'LYS','TPO':'THR','HSE':'HIS', 'UNK':'GLY', 'PCA':'GLU'}

AALength = len(AminoA)

# def three_to_one(three_letter_code):
#     # Normalize the input to uppercase to handle varying input cases
#     three_letter_code = three_letter_code.upper()
    
#     # Check if it is a non-canonical amino acid
#     if three_letter_code in NC_AminoA:
#         three_letter_code = NC_AminoA[three_letter_code]

#     # Convert the canonical three-letter code to a one-letter code
#     one_letter_code = AminoA_mapping.get(three_letter_code, '?')  # '?' is a placeholder for unknown codes

#     return one_letter_code

# Fallback: define our own amino acid conversion dictionaries
three_to_one = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
        'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
        'SEF': 'S'  # Non-canonical serine
}
    
one_to_three = {v: k for k, v in three_to_one.items()}

ElementList = ['C','N','O']
EleLength = len(ElementList)

R = 8.314/4184 #the universal gs constant
T = 298 # the temperature in kelvin(K);

ele2index = {'C':0, 'N':1, 'O':2, 'S':3, 'H':4}

ss2index = {'H':1, 'E':2, 'G':3, 'S':4, 'B':5, 'T':6, 'I':7, '-':0}

Hydro = ['A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W']
PolarAll = ['S','T','N','Q','R','H','K','D','E']
PolarUncharged = ['S','T','N','Q']
PolarPosCharged = ['R','H','K']
PolarNegCharged = ['D','E']
SpecialCase = ['C','U','G','P']

AminoA_index = {
    'ARG': 2, 'HIS': 2, 'LYS': 2, 'ASP': 3, 'GLU': 3,
    'SER': 1, 'THR': 1, 'ASN': 1, 'GLN': 1, 'CYS': 4,
    'SEF': 4, 'GLY': 4, 'PRO': 4, 'ALA': 0, 'VAL': 0,
    'ILE': 0, 'LEU': 0, 'MET': 0, 'PHE': 0, 'TYR': 0,
    'TRP': 0
}

AAvolume = {'A': 88.6, 'R':173.4, 'D':111.1, 'N':114.1, 'C':108.5, \
            'E':138.4, 'Q':143.8, 'G': 60.1, 'H':153.2, 'I':166.7, \
            'L':166.7, 'K':168.6, 'M':162.9, 'F':189.9, 'P':112.7, \
            'S': 89.0, 'T':116.1, 'W':227.8, 'Y':193.6, 'V':140.0}

AAhydropathy = {'A': 1.8, 'R':-4.5, 'N':-3.5, 'D':-3.5, 'C': 2.5, \
                'E':-3.5, 'Q':-3.5, 'G':-0.4, 'H':-3.2, 'I': 4.5, \
                'L': 3.8, 'K':-3.9, 'M': 1.9, 'F': 2.8, 'P':-1.6, \
                'S':-0.8, 'T':-0.7, 'W':-0.9, 'Y':-1.3, 'V': 4.2}

AAarea = {'A':115., 'R':225., 'D':150., 'N':160., 'C':135., \
          'E':190., 'Q':180., 'G': 75., 'H':195., 'I':175., \
          'L':170., 'K':200., 'M':185., 'F':210., 'P':145., \
          'S':115., 'T':140., 'W':255., 'Y':230., 'V':155.}

AAweight = {'A': 89.094, 'R':174.203, 'N':132.119, 'D':133.104, 'C':121.154, \
            'E':147.131, 'Q':146.146, 'G': 75.067, 'H':155.156, 'I':131.175, \
            'L':131.175, 'K':146.189, 'M':149.208, 'F':165.192, 'P':115.132, \
            'S':105.093, 'T':119.12 , 'W':204.228, 'Y':181.191, 'V':117.148}

AApharma = {'A':[0,1,3,1,1,1],'R':[0,3,3,2,1,1],'N':[0,2,4,1,1,0],'D':[0,1,5,1,2,0],\
            'C':[0,2,3,1,1,0],'E':[0,1,5,1,2,0],'Q':[0,2,4,1,1,0],'G':[0,1,3,1,1,0],\
            'H':[0,3,5,3,1,0],'I':[0,1,3,1,1,2],'L':[0,1,3,1,1,1],'K':[0,2,4,2,1,2],\
            'M':[0,1,3,1,1,2],'F':[1,1,3,1,1,1],'P':[0,1,3,1,1,1],'S':[0,2,4,1,1,0],\
            'T':[0,2,4,1,1,1],'W':[2,2,3,1,1,2],'Y':[1,2,4,1,1,1],'V':[0,1,3,1,1,1]}

Groups = [Hydro, PolarAll, PolarUncharged, PolarPosCharged, PolarNegCharged, SpecialCase]

def atmtyp_to_ele(st):
    if len(st.strip()) == 1:
        return st.strip()
    elif st[0] == 'H':
        return 'H'
    elif st == "CA":
        return "CA"
    elif st == "CL":
        return "CL"
    elif st == "BR":
        return "BR"
    else:
        print(st, 'Not in dictionary')
        return

def AAcharge(AA):
    if AA in ['D','E']:
        return -1.
    elif AA in ['R','H','K']:
        return 1.
    else:
        return 0.

pKa_index = {'ASP':0, 'GLU':1, 'ARG':2, 'LYS':3, 'HIS':4, 'CYS':5, 'TYR':6}
pKa_group = ['ASP',   'GLU',   'ARG',   'LYS',   'HIS',   'CYS',   'TYR']


ElementTau = np.array([6., 1.12, 1.1])
dt = np.dtype([('dim', int), ('birth', float), ('death', float)])

param_combination = [
    {'n_estimators': 25000, 'max_depth': 7 ,'min_samples_split':3, 'learning_rate':0.001,'loss':'squared_error','subsample':0.3,'max_features':'sqrt'}
]