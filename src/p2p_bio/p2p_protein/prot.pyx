# distutils: language = c++
cimport cython
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool
import numpy as np
cimport numpy as np

cdef int i, j

# Declare the class with cdef
cdef extern from 'lib/Protein.h':
    cdef cppclass Protein:
        Protein() except +
        Protein(string) except +

        void Deallocate()

        bool loadPQRFile(string)
        bool atomwise_interaction(double short_cutoff, double long_cutoff)
        size_t size()

        double **feature_VDW_()
        double **feature_CLB_()
        double **feature_FRI1()
        double **feature_FRI2()
        double **feature_FRI3()
        double **feature_FRI4()
        double **feature_FRI5()

cdef class PyProtein:
    cdef Protein *c_protein
    #cdef Protein c_protein

    def __cinit__(self, bytes PDBid):
        self.c_protein = new Protein(PDBid)
        #self.c_protein = Protein(PDBid) # both ways work

    def Deallocate(self):
        self.c_protein.Deallocate()

    def loadPQRFile(self, bytes filename):
        return self.c_protein.loadPQRFile(filename)

    def atomwise_interaction(self, double short_cutoff, double long_cutoff):
        return self.c_protein.atomwise_interaction(short_cutoff, long_cutoff)

    def feature_CLB(self):
        PyCLB = np.zeros((self.c_protein.size(), 5))
        CLB = self.c_protein.feature_CLB_()
        for i in range(self.c_protein.size()):
            for j in range(5):
                PyCLB[i][j] = CLB[i][j]
        return PyCLB

    def feature_VDW(self):
        PyVDW = np.zeros((self.c_protein.size(), 5))
        VDW = self.c_protein.feature_VDW_()
        for i in range(self.c_protein.size()):
            for j in range(5):
                PyVDW[i][j] = VDW[i][j]
        return PyVDW

    def feature_FRIs(self):
        #return self.c_protein.feature_FRI1() # doesn't work, cython cannot directly convert double
        PyFRI1 = np.zeros((self.c_protein.size(), 5))
        FRI1 = self.c_protein.feature_FRI1()
        PyFRI2 = np.zeros((self.c_protein.size(), 5))
        FRI2 = self.c_protein.feature_FRI2()
        PyFRI3 = np.zeros((self.c_protein.size(), 5))
        FRI3 = self.c_protein.feature_FRI3()
        PyFRI4 = np.zeros((self.c_protein.size(), 5))
        FRI4 = self.c_protein.feature_FRI4()
        PyFRI5 = np.zeros((self.c_protein.size(), 5))
        FRI5 = self.c_protein.feature_FRI5()
        for i in range(self.c_protein.size()):
            for j in range(5):
                PyFRI1[i][j] = FRI1[i][j]
                PyFRI2[i][j] = FRI2[i][j]
                PyFRI3[i][j] = FRI3[i][j]
                PyFRI4[i][j] = FRI4[i][j]
                PyFRI5[i][j] = FRI5[i][j]
        return PyFRI1, PyFRI2, PyFRI3, PyFRI4, PyFRI5

