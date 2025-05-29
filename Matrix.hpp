#ifndef MATRIX_HPP
#define MATRIX_HPP
#include <vector>
#include <iostream>

class Mat{
    private : 
        int row; 
        int col;
        std::vector<double> data;
    public : 
        Mat(int n=0, int p=0);
        ~Mat();
        Mat(const Mat & other);
        Mat& operator=(const Mat & other);
        double& getcoef(int i, int j); // surcharge en accès lecture et modification
        double getcoef(int i, int j) const;  // surcharge accès en lecture seule
        std::vector<double> matve(const std::vector<double> u) const;
        friend std::ostream& operator<<(std::ostream& os, const Mat& m);
        Mat operator+(const Mat& other) const;
        Mat scale(const Mat& A, double alpha);
        Mat operator*(const Mat& other) const;//surcharge produit matrice matrice
        std::vector<double> operator*(const std::vector<double>& vec) const; // surcharge produit matrice vecteur
        int getnRows();//accesseurs mode écriture
        int getnCols();
        int getnRows() const; // accesseurs mode lecture
        int getnCols() const;


};


#endif // MATRIX_HPP