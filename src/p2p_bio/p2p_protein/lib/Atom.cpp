#include "Atom.h"


/**************************Constructors**************************************/
// Default constructor
Atom::Atom () {}

// Overloaded constructor
Atom::Atom (double paramX, double paramY, double paramZ, double paramR, int paramN, double paramC, string Type) {
    string AtomType;

    this->x = paramX;
    this->y = paramY;
    this->z = paramZ;
    this->radius = paramR;
    this->number = paramN;
    this->charge = paramC;
    this->type = Type[0];

    if ((int)Type[0]==79) { // O
        this->itype = 2; this->vdwRa = 1.52;
    } else if ((int)Type[0]==78) { // N
        this->itype = 1; this->vdwRa = 1.55;
    } else if ((int)Type[0]==67) { // C
        this->itype = 0; this->vdwRa = 1.7;
    } else if ((int)Type[0]==83) { // S
        this->itype = 3; this->vdwRa = 1.8;
    } else if ((int)Type[0]==72) { // H
        this->itype = 4; this->vdwRa = 1.2;
    //} else if ((int)Type[0]==90) { // Z
    //    this->itype = 5; this->vdwRa = 0.;
    }
    else {
        cout << "new atom type: " << (int)Type[0] << endl;
        abort();
    }
}

// Destructor
Atom::~Atom () {}

// Operators
double Atom::operator- (const Atom _atom) const { // use &_atom should be also OK
    double dx = this->X()-_atom.X();
    double dy = this->Y()-_atom.Y();
    double dz = this->Z()-_atom.Z();
    return sqrt(dx*dx + dy*dy + dz*dz);
}

// Return the area of the rectangle
double Atom::X () const {
    return x;
}
double Atom::Y() const {
    return y;
}
double Atom::Z() const {
    return z;
}
double Atom::Radius() const {
    return radius;
}
int Atom::Number() const {
    return number;
}
double Atom::Charge() const {
    return charge;
}
string Atom::Type() const {
    return type;
}
string Atom::Residue() const {
    return residue;
}
int Atom::ResID() const {
    return resID;
}
int Atom::iType() const {
    return itype;
}
double Atom::vdWRa() const {
    return vdwRa;
}

/***************************Data modifiers***********************************/
void Atom::inX(double paramX) {
    this->x = paramX;
}
void Atom::inY(double paramY) {
    this->y = paramY;
}
void Atom::inZ(double paramZ) {
    this->z = paramZ;
}
void Atom::inRadius(double paramRadius) {
    this->radius = paramRadius;
}
void Atom::inNumber(int paramN) {
    this->number = paramN;
}
void Atom::inCharge(double paramC) {
    this->charge = paramC;
}
void Atom::inType(string Type) {
    this->type = Type;
    if ((int)Type[0]==79) {
        this->itype = 0; this->vdwRa = 1.52;
    }
    else if ((int)Type[0]==78) {
        this->itype = 1; this->vdwRa = 1.55;
    }
    else if ((int)Type[0]==67) {
        this->itype = 2; this->vdwRa = 1.7;
    }
    else if ((int)Type[0]==83) {
        this->itype = 3; this->vdwRa = 1.8;
    }
    else if ((int)Type[0]==72) {
        this->itype = 4; this->vdwRa = 1.2;
    }
    else {
        cout << "new atom type: " << (int)Type[0] << endl;
        abort();
    }
}
void Atom::inResidue(string Residue) {
    this->residue = Residue;
}
void Atom::inResID(int ResID) {
    this->resID = ResID;
}
