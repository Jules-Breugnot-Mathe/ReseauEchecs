#include "IJ.hpp"

IJ::IJ(int i, int j) : i(i), j(j) {}

IJ::~IJ() {}

IJ::IJ(const IJ& other) : i(other.i), j(other.j) {}

IJ& IJ::operator=(const IJ& other) {
    if (this != &other) {
        this->i = other.i;
        this->j = other.j;
    }
    return *this;
}

int IJ::geti() const {
    return i;
}

int IJ::getj() const {
    return j;
}

bool operator<(const IJ& ij1, const IJ& ij2) {
    if (ij1.geti() < ij2.geti()) {
        return true;
    } else if (ij1.geti() == ij2.geti()) {
        return ij1.getj() < ij2.getj();
    }else
    return false;
}
