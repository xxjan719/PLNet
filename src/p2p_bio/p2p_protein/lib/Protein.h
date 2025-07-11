#ifndef PROTEIN_H
#define PROTEIN_H

#include <iostream>
#include <algorithm>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <math.h>

#define DEFAULT_ATOM_SIZE 1.5
#define TOL 1.E-12
#define MAX 10000
#define S 8.0
#define TABLEX 26
#define TABLEY 76
#define ATOM_TYPES 100

#include "Atom.h"
typedef vector<Atom> Atoms;

//using namespace std;

//double mat_idx[25];

class Protein {
public:
    Protein();
    Protein(string fileName);
    ~Protein();
  
    void Deallocate();
    
    bool loadPQRFile(string fileName);
    bool atomwise_interaction(double short_cutoff, double long_cutoff);

    double **feature_VDW_();
    double **feature_CLB_();
    double **feature_FRI1();
    double **feature_FRI2();
    double **feature_FRI3();
    double **feature_FRI4();
    double **feature_FRI5();

    size_t size() const;
    Atoms::iterator begin(), end();

private:
    string name;
    Atoms atoms;
    int nAtom;
    double **FRI1, **FRI2, **FRI3, **FRI4, **FRI5, **CLB_, **VDW_;
};

#endif
