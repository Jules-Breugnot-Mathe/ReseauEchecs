#include <vector>
#include <iostream>
#include "Matrix.hpp"
#include "Utils.hpp"



Mat::Mat(int n, int p){ // constructeur par défault, génération aléatoire suivant une distribution de Xavier
    this->row = n;
    this->col = p;
    std::vector<double> u = xavier_init_vector(n, p);
    this->data = u;
}

Mat::~Mat(){} // destructeur (la libération de la mémoire est gérée par <vector>)

Mat::Mat(const Mat & other){ // constructeur par copie
    if (this != &other){
        this->col = other.col; 
        this->row = other.row;
        this->data = other.data; // la réallocation est automatique
    }
}

Mat& Mat::operator=(const Mat & other){ // surcharge de = 
    if (this != &other){
        this->col = other.col; 
        this->row = other.row;
        this->data = other.data; // la réallocation est automatique
    }
    return *this;
}

double& Mat::getcoef(int i, int j){
    return this->data[i*(this->col) + j];
}

std::vector<double> Mat::matve(const std::vector<double> u) const {
    if ((int)u.size() != this->col) {
        throw std::invalid_argument("taille du vecteur incompatible avec le nombre de colonnes");
    }
    std::vector<double> result(this->row, 0.0);
    for (int i = 0; i < this->row; ++i) {
        for (int j = 0; j < this->col; ++j) {
            result[i] += this->data[i * this->col + j] * u[j];
        }
    }
    return result;
}

std::ostream& operator<<(std::ostream& os, const Mat& m) {
    for (int i = 0; i < m.row; ++i) {
        for (int j = 0; j < m.col; ++j) {
            os << m.data[i * m.col + j] << " ";
        }
        os << "\n";
    }
    return os;
}

Mat Mat::operator+(const Mat& other) const {
    if (this->row != other.row || this->col != other.col) {
        throw std::invalid_argument("Matrices de dimensions incompatibles pour l'addition.");
    }

    Mat result(this->row, this->col);
    for (int i = 0; i < this->row * this->col; ++i) {
        result.data[i] = this->data[i] + other.data[i];
    }

    return result;
}

Mat Mat::scale(const Mat& A, double alpha){
    Mat result(this->row, this->col);
    for (int i = 0; i < this->row * this->col; ++i) {
        result.data[i] = A.data[i]*alpha;
    }
    return result;
}

Mat Mat::operator*(const Mat& other) const {
    if (this->col != other.row) {
        throw std::invalid_argument("Dimensions incompatibles pour la multiplication matricielle.");
    }

    Mat result(this->row, other.col);

    for (int i = 0; i < this->row; ++i) {
        for (int j = 0; j < other.col; ++j) {
            double sum = 0.0;
            for (int k = 0; k < this->col; ++k) {
                sum += this->getcoef(i, k) * other.getcoef(k, j);
            }
            result.getcoef(i, j) = sum;
        }
    }

    return result;
}

double Mat::getcoef(int i, int j) const {
    return this->data[i * this->col + j];
}

std::vector<double> Mat::operator*(const std::vector<double>& vec) const {
    if ((int)vec.size() != this->col) {
        throw std::invalid_argument("Taille du vecteur incompatible avec le nombre de colonnes de la matrice.");
    }

    std::vector<double> result(this->row, 0.0);
    for (int i = 0; i < this->row; ++i) {
        for (int j = 0; j < this->col; ++j) {
            result[i] += this->getcoef(i, j) * vec[j];
        }
    }
    return result;
}

int Mat::getnCols(){
    return this->col;
}

int Mat::getnRows(){
    return this->row;
}

int Mat::getnRows() const {
    return row;
}

int Mat::getnCols() const {
    return col;
}
