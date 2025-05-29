#ifndef LAYER_HPP
#define LAYER_HPP 

#include <iostream>
#include <vector>
#include <cmath>
#include "Matrix.hpp"

class Layer{ // X_n+1 = sigmoid(W_n * X_n  +  B_n)
    private : 
        int input_dim; // dimension du vecteur entrant
        int output_dim; // dimension du vecteur sortant
        Mat W; // matrice des poids de la couche
        std::vector<double> B; // vecteur des biais 
        std::vector<double> X;  // Activation réelle de la couche
        std::vector<double> Z;  // W*X + B (avant activation)
        std::vector<double> X_opti; // acivation optimisée de la couche, ne va être stocké qu'une fois pour
        /*
        //l'ensemble des couches du réseau, on ne stocke pas de vecteurs inutiles pendant la rétropropagation du gradient
        Mat grad_W;
        std::vector<double> grad_B;// pour des raisons d'optimisation du temps de calcul, on stocke des gradients
        std::vector<double> grad_X; //au lieu de réallouer des Mat et des vector à chaque backward_pass
        */
    public : 
        Layer(int input_size=0, int output_size=0);
        ~Layer();
        Layer(const Layer & other);
        Layer& operator=(const Layer & other);
        std::vector<double> pass_forward(std::vector<double> X);
        //std::ostream& operator<<(std::ostream& os, const Layer& layer);
        Mat& getWeight(); // surcharge en mode écriture
        std::vector<double>& getBiais(); // pareil
        std::vector<double>& getActivation();//renvoie l'activation à l'entrée de la couche
        std::vector<double>& getZ();
        std::vector<double>& getXopti(); // pour pouvoir l’affecter directement
        const std::vector<double>& getXopti() const;
        void backward_pass(const std::vector<double>& Y_cible, double alpha); // calcul du gradient d'une couche par rapport
        //à l'activation de la couche n+1 suivante, le score est calculé en norme 2 entre Xn+1 et un Xcible
        int getInputDim() const;
        int getOutputDim() const;
        const Mat& getWeight() const; // surcharge en mode lecture de l'accesseur sur les poids     
        const std::vector<double>& getBiais() const; // pareil
};

#endif // LAYER_HPP