#include "Protein.h"
#include <map>
std::map<std::string, std::string> three2one {{"ARG", "R"}, {"HIS", "H"}, {"LYS", "K"}, 
    {"ASP", "D"}, {"GLU", "E"}, {"SER", "S"}, {"THR", "T"}, {"ASN", "N"}, {"GLN", "Q"},
    {"CYS", "C"}, {"GLY", "G"}, {"PRO", "P"}, {"ALA", "A"}, {"VAL", "V"}, {"ILE", "I"},
    {"LEU", "L"}, {"MET", "M"}, {"PHE", "F"}, {"TYR", "Y"}, {"TRP", "W"}, 
    {"HID", "H"}, {"HIE", "H"}};

/********************************Constructors********************************/
Protein::Protein() {}

Protein::Protein(string PDBID) {
    this->name = PDBID;

    //mat_idx[0]  = 0; mat_idx[1]  = 1; mat_idx[2]  = 2;  mat_idx[3]  = 3;  mat_idx[4]  = 4;
    //mat_idx[5]  = 1; mat_idx[6]  = 5; mat_idx[7]  = 6;  mat_idx[8]  = 7;  mat_idx[9]  = 8;
    //mat_idx[10] = 2; mat_idx[11] = 6; mat_idx[12] = 9;  mat_idx[13] = 10; mat_idx[14] = 11;
    //mat_idx[15] = 3; mat_idx[16] = 7; mat_idx[17] = 10; mat_idx[18] = 12; mat_idx[19] = 13;
    //mat_idx[20] = 4; mat_idx[21] = 8; mat_idx[22] = 11; mat_idx[23] = 13; mat_idx[24] = 14;

}
Protein::~Protein() {
}
void Protein::Deallocate() {
    printf("deallocate memory\n");
    for (int i=0; i<this->nAtom; i++) {
        free(this->VDW_[i]);
        free(this->FRI1[i]);
        free(this->FRI2[i]);
        free(this->FRI3[i]);
        free(this->FRI4[i]);
        free(this->FRI5[i]);
        free(this->CLB_[i]);
    }
    free(this->VDW_);
    free(this->FRI1);
    free(this->FRI2);
    free(this->FRI3);
    free(this->FRI4);
    free(this->FRI5);
    free(this->CLB_);
}
/******************************Loading Data**********************************/
// Reading the coordinates from the PQR files or from the XZY format
bool Protein::loadPQRFile(string fileName) {
    ifstream in(fileName.c_str());

    if (!in) {
        cerr << "Error opening file pqr or xyz file " << fileName << endl;
        return false;
    }
    double x, y, z, q, r;
    int number=0, PQRnumber, RESnumber;
    string type, junk, residue, chainID;
    while (in && !in.eof())	{
        in >> junk;
        if (junk == "ATOM" || junk == "HETATM")	{
            in >> PQRnumber >> type >> residue >> chainID >> RESnumber >> x >> y >> z >> q >> r;
            if (residue != "WAT") {
                Atom a(x, y, z, r, number, q, type);
                this->atoms.push_back(a);
                number++;
            }
        }
        else {
        /* gobble down the rest of the line */
            in.ignore(255, '\n');
        }
    }
    in.close();
    this->nAtom = number;
    return true;
} // Protein::loadPQRFile(string fileName)
/*******************************Calculation**********************************/
bool Protein::atomwise_interaction(double short_cutoff, double long_cutoff){
    Atoms::iterator iatom, jatom;
    int i, j;
    double dist, ratio, vdw, clb, radSum, temp, rig1, rig2, rig3, rig4, rig5;
    double tau1=.5, tau2=1., tau3=2., tau4=1., tau5=2., nu1=5, nu2=5, nu3=5, nu4=15, nu5=15;

    printf("allocate memory for features: VDW, FRI, and CLB\n");
    this->VDW_ = (double**)calloc(this->nAtom, sizeof(double*));
    this->FRI1 = (double**)calloc(this->nAtom, sizeof(double*));
    this->FRI2 = (double**)calloc(this->nAtom, sizeof(double*));
    this->FRI3 = (double**)calloc(this->nAtom, sizeof(double*));
    this->FRI4 = (double**)calloc(this->nAtom, sizeof(double*));
    this->FRI5 = (double**)calloc(this->nAtom, sizeof(double*));
    this->CLB_ = (double**)calloc(this->nAtom, sizeof(double*));
    for (i=0; i<this->nAtom; i++) {
        this->VDW_[i] = (double*)calloc(5, sizeof(double));
        this->FRI1[i] = (double*)calloc(5, sizeof(double));
        this->FRI2[i] = (double*)calloc(5, sizeof(double));
        this->FRI3[i] = (double*)calloc(5, sizeof(double));
        this->FRI4[i] = (double*)calloc(5, sizeof(double));
        this->FRI5[i] = (double*)calloc(5, sizeof(double));
        this->CLB_[i] = (double*)calloc(5, sizeof(double));
    } // deallocate memory ad Protein::Deallocate()

    for (iatom=this->begin(), i=0; iatom!=this->end(); iatom++, i++) {
        //std::cout << iatom->Type() << " " << iatom->iType() << std::endl;
        for (jatom=iatom+1, j=i+1; jatom!=this->end(); jatom++, j++) {
            dist  = *iatom - *jatom;

            if (dist < long_cutoff) {
                //radSum = iatom->vdWRa() + jatom->vdWRa();
                radSum = iatom->Radius() + jatom->Radius();
                // calculating van der Waals potential
                if (dist < short_cutoff) {
                    ratio = pow((iatom->Radius()+jatom->Radius()), 2) / dist; // original cal.
                    //ratio = radSum / dist;
                    vdw   = pow(ratio, 12) - 2.*pow(ratio, 6);
                    //std::cout << vdw << std::endl;
                    this->VDW_[i][jatom->iType()] += vdw;
                    this->VDW_[j][iatom->iType()] += vdw;
                }

                // calculating coulomb potential
                clb = iatom->Charge()*jatom->Charge() / dist;
                this->CLB_[i][jatom->iType()] += clb;
                this->CLB_[j][iatom->iType()] += clb;

                // calculating flexbility rigidy index
                temp = dist / radSum;
                rig1 = 1./(1.+pow(temp/tau1, nu1));
                this->FRI1[i][jatom->iType()] += rig1;
                this->FRI1[j][iatom->iType()] += rig1;
                rig2 = 1./(1.+pow(temp/tau2, nu2));
                this->FRI2[i][jatom->iType()] += rig2;
                this->FRI2[j][iatom->iType()] += rig2;
                rig3 = 1./(1.+pow(temp/tau3, nu3));
                this->FRI3[i][jatom->iType()] += rig3;
                this->FRI3[j][iatom->iType()] += rig3;
                rig4 = exp(-pow(temp/tau4, nu4));
                this->FRI4[i][jatom->iType()] += rig4;
                this->FRI4[j][iatom->iType()] += rig4;
                rig5 = exp(-pow(temp/tau5, nu5));
                this->FRI5[i][jatom->iType()] += rig5;
                this->FRI5[j][iatom->iType()] += rig5;
            }
        }
    }
    return true;
} //atomwise_interaction
/***************************Accessors & Modifiers****************************/
double **Protein::feature_VDW_() {
    return this->VDW_;
}
double **Protein::feature_CLB_() {
    return this->CLB_;
}
/******************FRI********************/
double **Protein::feature_FRI1() {
    return this->FRI1;
}
double **Protein::feature_FRI2() {
    return this->FRI2;
}
double **Protein::feature_FRI3() {
    return this->FRI3;
}
double **Protein::feature_FRI4() {
    return this->FRI4;
}
double **Protein::feature_FRI5() {
    return this->FRI5;
}
size_t Protein::size() const {
    return this->atoms.size();
}
Atoms::iterator Protein::begin() {
    return this->atoms.begin();
}
Atoms::iterator Protein::end() {
    return this->atoms.end();
}
