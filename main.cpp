#include "IJ.hpp"
#include "Matrix.hpp"
#include <iostream>
#include "Layer.hpp"
#include "Dense.hpp"
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "Utils.hpp"
#include <algorithm>
using namespace std;


template<class T> std::vector<T> operator+(const std::vector<T> & v1, const std::vector<T> & v2){
    std::vector<T> v3(v1.size());
    for (int i=0; i<v1.size(); i++){
        v3[i] = v1[i] + v2[i];
    }
    return v3;
}


template<class T> std::vector<T> operator*(std::vector<T> v1, T alpha){
    for (int i=0; i<v1.size(); i++){
        v1[i] = v1[i]*alpha;
    }
    return v1;
}



int main(){

    //----------------- PROTOTYPE DE RESEAU ET APPRENTISSAGE-------------------//
    // Construction du réseau Dense
    /*
    Dense Reseau;
    Layer L0(97, 64);
    Layer L1(64, 64);
    Layer L2(64, 1);
    Reseau.getNbCouches() = 4;
    std::vector<Layer> layers{L0, L1, L2};
    Reseau.getNetwork() = layers;
    //entrainement sur 1 epoch
    Reseau.backpropagation("chess_positions.csv", 1, 97, 1, 5);
    Reseau.save_weights("Model.txt");
    */
    /*
    Dense Reseau;
    Reseau.load_weights("Model.txt");
    Reseau.backpropagation("chess_positions.csv", 1, 97, 1, 5);
    Reseau.save_weights("Model.txt");
    */

    
    
    

    
    return 0;
}