#ifndef ATOM_H
#define ATOM_H

#include <iostream>
#include <math.h>
using namespace std;

class Atom {
private:
    double x, y, z;
    double radius;
    int number;
    int PQRnumber; // maybe useful later
    double charge;
    string type;
    string residue;
    int resID;
    int itype;
    double vdwRa;
  
public:
    Atom();
    Atom(double paramX, double paramY, double paramZ, double paramR,
         int paramN, double paramC, string Type);
    ~Atom();
    double operator- (const Atom _atom) const;
    double X() const;
    double Y() const;
    double Z() const;
    double OrigRadius() const;
    double Radius() const;
    int Number() const;
    double Charge() const;
    string Type() const;
    string Residue() const;
    int ResID() const;
    int iType() const;
    double vdWRa() const;
  
    void inX(double paramX);
    void inY(double paramY);
    void inZ(double paramZ);
    void inRadius(double paramRadius);
    void inNumber(int paramN);
    void inCharge(double paramC);
    void inType(string Type);
    void inResidue(string Residue);
    void inResID(int ResID);

    int VDWindex;
};

#endif
