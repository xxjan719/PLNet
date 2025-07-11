 
import gudhi
import numpy as np
from typing import Tuple

# Import from constant module
from .constant import EleLength

def generate_rips_complex(length: int, # number of elements
                          atoms_partner1: list,
                          atoms_partner2: list,
                          interval: float = 1., 
                          birth_cut: float = 2., 
                          death_cut: float =11.,
                          ) -> np.ndarray:
    _PHcut = death_cut+1.
    _bin_length = int((death_cut-birth_cut)/interval)
    _bins = np.linspace(birth_cut, death_cut, _bin_length+1)
    _dt = np.dtype([('dim', int), ('birth', float), ('death', float)])

    def BinID(x):
        for i in range(_bin_length):
            if _bins[i] <= x < _bins[i+1]:
                return i
        return _bin_length # needs update here

    _rips_dth = np.zeros([_bin_length, length**2])
    _rips_bar = np.zeros([_bin_length, length**2])

    for idx_m in range(length):
        for idx_o in range(length):
            length_m = len(atoms_partner1[idx_m])
            length_o = len(atoms_partner2[idx_o])
            matrixA = np.ones((length_m+length_o, length_m+length_o))*100.
            for ii, iatom in enumerate(atoms_partner1[idx_m]):
                for jj, jatom in enumerate(atoms_partner2[idx_o]):
                    dis = np.linalg.norm(iatom.pos-jatom.pos)
                    matrixA[ii, length_m+jj] = dis
                    matrixA[length_m+jj, ii] = dis
            rips_complex = gudhi.RipsComplex(distance_matrix=matrixA, max_edge_length=_PHcut)
            PH = rips_complex.create_simplex_tree().persistence()

            _tmpbars = np.zeros(length_m+length_o, dtype=_dt)
            cnt = 0
            for simplex in PH:
                dim, b, d = int(simplex[0]), float(simplex[1][0]), float(simplex[1][1])
                if d-b < 0.1: continue
                _tmpbars[cnt]['dim']   = dim
                _tmpbars[cnt]['birth'] = b
                _tmpbars[cnt]['death'] = d
                cnt += 1
            for bar in _tmpbars[0:cnt]:
                death = bar['death']
                if death >= death_cut or death < birth_cut: continue
                _death_id = BinID(death)
                _rips_dth[ _death_id, idx_m*EleLength+idx_o] += 1
                _rips_bar[:_death_id, idx_m*EleLength+idx_o] += 1

    return np.concatenate((np.array(_rips_dth).flatten(), np.array(_rips_bar).flatten()), axis=0)

dt = np.dtype([('dim', int), ('birth', float), ('death', float)])

def generate_alpha_shape(length: int, atoms_partner1: list, atoms_partner2: list) -> np.ndarray:
    _alpha_PH12 = np.zeros([length, length, 14])
    for idx_m in range(length):
        for idx_o in range(length):
            points = [iatom.pos for iatom in atoms_partner1[idx_m]] + [jatom.pos for jatom in atoms_partner2[idx_o]]
            alpha_complex = gudhi.AlphaComplex(points = points)
            PH = alpha_complex.create_simplex_tree().persistence()
            tmpbars = np.zeros(len(PH), dtype=dt)
            cnt = 0
            for simplex in PH:
                dim, b, d = int(simplex[0]), float(simplex[1][0]), float(simplex[1][1])
                if d-b < 0.1: continue
                tmpbars[cnt]['dim'], tmpbars[cnt]['birth'], tmpbars[cnt]['death'] = dim, b, d
                cnt += 1
            bars = tmpbars[0:cnt]
            if len(bars[bars['dim'] == 1]['death']) > 0:
                _alpha_PH12[idx_m, idx_o, 0] = np.sum(bars[bars['dim']==1]['death'] - bars[bars['dim']==1]['birth'])
                _alpha_PH12[idx_m, idx_o, 1] = np.max(bars[bars['dim']==1]['death'] - bars[bars['dim']==1]['birth'])
                _alpha_PH12[idx_m, idx_o, 2] = np.mean(bars[bars['dim']==1]['death'] - bars[bars['dim']==1]['birth'])
                _alpha_PH12[idx_m, idx_o, 3] = np.min(bars[bars['dim']==1]['birth'])
                _alpha_PH12[idx_m, idx_o, 4] = np.max(bars[bars['dim']==1]['birth'])
                _alpha_PH12[idx_m, idx_o, 5] = np.min(bars[bars['dim']==1]['death'])
                _alpha_PH12[idx_m, idx_o, 6] = np.max(bars[bars['dim']==1]['death'])
            if len(bars[bars['dim']==2]['death']) > 0:
                _alpha_PH12[idx_m, idx_o, 7]  = np.sum(bars[bars['dim']==2]['death'] - bars[bars['dim']==2]['birth'])
                _alpha_PH12[idx_m, idx_o, 8]  = np.max(bars[bars['dim']==2]['death'] - bars[bars['dim']==2]['birth'])
                _alpha_PH12[idx_m, idx_o, 9]  = np.mean(bars[bars['dim']==2]['death'] - bars[bars['dim']==2]['birth'])
                _alpha_PH12[idx_m, idx_o, 10] = np.min(bars[bars['dim']==2]['birth'])
                _alpha_PH12[idx_m, idx_o, 11] = np.max(bars[bars['dim']==2]['birth'])
                _alpha_PH12[idx_m, idx_o, 12] = np.min(bars[bars['dim']==2]['death'])
                _alpha_PH12[idx_m, idx_o, 13] = np.max(bars[bars['dim']==2]['death'])

    _alpha_PH12_all = np.zeros([14])
    points = []
    for idx in range(length):
        points += [iatom.pos for iatom in atoms_partner1[idx]] + [jatom.pos for jatom in atoms_partner2[idx]]
    alpha_complex = gudhi.AlphaComplex(points = points)
    PH = alpha_complex.create_simplex_tree().persistence()

    tmpbars = np.zeros(len(PH), dtype=dt)
    cnt = 0
    for simplex in PH:
        dim, b, d = int(simplex[0]), float(simplex[1][0]), float(simplex[1][1])
        if d-b < 0.1: continue
        tmpbars[cnt]['dim'], tmpbars[cnt]['birth'], tmpbars[cnt]['death'] = dim, b, d
        cnt += 1
    bars = tmpbars[0:cnt]
    if len(bars[bars['dim'] == 1]['death']) > 0:
        _alpha_PH12_all[0] = np.sum(bars[bars['dim']==1]['death'] - bars[bars['dim']==1]['birth'])
        _alpha_PH12_all[1] = np.max(bars[bars['dim']==1]['death'] - bars[bars['dim']==1]['birth'])
        _alpha_PH12_all[2] = np.mean(bars[bars['dim']==1]['death'] - bars[bars['dim']==1]['birth'])
        _alpha_PH12_all[3] = np.min(bars[bars['dim']==1]['birth'])
        _alpha_PH12_all[4] = np.max(bars[bars['dim']==1]['birth'])
        _alpha_PH12_all[5] = np.min(bars[bars['dim']==1]['death'])
        _alpha_PH12_all[6] = np.max(bars[bars['dim']==1]['death'])
    if len(bars[bars['dim']==2]['death']) > 0:
        _alpha_PH12_all[7]  = np.sum(bars[bars['dim']==2]['death'] - bars[bars['dim']==2]['birth'])
        _alpha_PH12_all[8]  = np.max(bars[bars['dim']==2]['death'] - bars[bars['dim']==2]['birth'])
        _alpha_PH12_all[9]  = np.mean(bars[bars['dim']==2]['death'] - bars[bars['dim']==2]['birth'])
        _alpha_PH12_all[10] = np.min(bars[bars['dim']==2]['birth'])
        _alpha_PH12_all[11] = np.max(bars[bars['dim']==2]['birth'])
        _alpha_PH12_all[12] = np.min(bars[bars['dim']==2]['death'])
        _alpha_PH12_all[13] = np.max(bars[bars['dim']==2]['death'])

    return np.concatenate((_alpha_PH12.flatten(), _alpha_PH12_all), axis=0)

def generate_persistent_spectra(length: int,
                                atoms_partner1: list,
                                atoms_partner2: list,
                                bins: list = [3, 4, 5, 7, 9, 11],
                                ) -> np.ndarray:
    _bins = bins
    _bin_length = len(_bins)
    _features = np.zeros((int(_bin_length*length**2), 8))
    for idx_m in range(length):
        for idx_o in range(length):
            length_m = len(atoms_partner1[idx_m])
            length_o = len(atoms_partner2[idx_o])
            matrixA = np.ones((length_m+length_o, length_m+length_o))*100.
            for ii, iatom in enumerate(atoms_partner1[idx_m]):
                for jj, jatom in enumerate(atoms_partner2[idx_o]):
                    dis = np.linalg.norm(iatom.pos-jatom.pos)
                    matrixA[ii, length_m+jj] = dis
                    matrixA[length_m+jj, ii] = dis
            #print('After computing is',matrixA)
            for idx_cut, cut in enumerate(_bins):
                Laplacian = np.zeros((length_m+length_o, length_m+length_o))
                Laplacian[matrixA<cut] = -1
                Laplacian += np.diagflat(-np.sum(Laplacian, axis=0))
                eigens = np.sort(np.linalg.eigvalsh(Laplacian))
                idx_feat = idx_cut*length**2 + idx_m*length + idx_o
                eigens = eigens[eigens>10**-8]
                if len(eigens) > 0:
                    #sum, min, max, mean, std, var,
                    _features[idx_feat][0] = eigens.sum()
                    _features[idx_feat][1] = eigens.min()
                    _features[idx_feat][2] = eigens.max()
                    _features[idx_feat][3] = eigens.mean()
                    _features[idx_feat][4] = eigens.std()
                    _features[idx_feat][5] = eigens.var()
                    _features[idx_feat][6] = np.dot(eigens, eigens)
                    _features[idx_feat][7] = len(eigens[eigens>10**-8])

    return _features.flatten()

def generate_flexibility_rigidy_index(length: int, atoms_partner1: list, atoms_partner2: list) -> np.ndarray:
    kappa_exp = 2; ElementTau_exp = [5, 2*1.01, 2]
    kappa_lorentz = 4; ElementTau_lorentz = [1, 6*1.01, 6]

    _fri_exp = np.zeros([length, length, 4])
    _fri_lorentz = np.zeros([length, length, 4])
    for idx_m in range(length):
        for idx_o in range(length):
            dists = []
            for iatom in atoms_partner1[idx_m]:
                for jatom in atoms_partner2[idx_o]:
                    dists.append(np.linalg.norm(iatom.pos-jatom.pos))
            dists = np.array(dists)
            dist_exp = np.exp(-np.power(np.true_divide(dists, ElementTau_exp[idx_m]+ElementTau_exp[idx_o]), kappa_exp))
            dist_lorentz = 1./(1+np.power(np.true_divide(dists, ElementTau_lorentz[idx_m]+ElementTau_lorentz[idx_o]), kappa_lorentz))

            _fri_exp[idx_m, idx_o, 0] = np.sum(dist_exp)
            _fri_exp[idx_m, idx_o, 1] = np.max(dist_exp)
            _fri_exp[idx_m, idx_o, 2] = np.mean(dist_exp)
            _fri_exp[idx_m, idx_o, 3] = np.min(dist_exp)

            _fri_lorentz[idx_m, idx_o, 0] = np.sum(dist_lorentz)
            _fri_lorentz[idx_m, idx_o, 1] = np.max(dist_lorentz)
            _fri_lorentz[idx_m, idx_o, 2] = np.mean(dist_lorentz)
            _fri_lorentz[idx_m, idx_o, 3] = np.min(dist_lorentz)

    return np.concatenate((_fri_exp.flatten(), _fri_lorentz.flatten()), axis=0)