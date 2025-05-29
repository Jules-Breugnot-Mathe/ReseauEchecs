#ifndef DENSE_HPP
#define DENSE_HPP

#include <iostream>
#include <vector>
#include "Layer.hpp"

class Dense {
private:
    int NbCouches;
    std::vector<Layer> Network;
    //Par convention, on 
    //compte la couche de sortie comme une couche à part entière, même si on a seulement n-1 objets Layer dans le réseau.

public:
    Dense(int N = 0);
    ~Dense();
    Dense(const Dense& other);
    Dense& operator=(const Dense& other);
    std::vector<double> pass_forward(std::vector<double> input);
    void backward_pass(const std::vector<double>& donnee, double alpha, const int input_size, const int output_size); // ajuste le réseau à une donnée Y_cible
    void set_layer(int index, int input_dim, int output_dim); // cette surcharge de set_layer défini les dimensions de la couche
    // numéro "index"
    void save_weights(const std::string& filename) const; // sauvegarde dans le fichier Model.txt des poids du réseau
    void load_weights(const std::string& filename);
    void set_layer(int input_dim, const std::vector<int>& hidden_dims, int output_dim); // surcharge de set_layer 
    //qu'on appelera dans la fonction load_weights, cette surcharge définit le nombre
    //de couches du réseau ainssi que les dimensions de toutes les couches 
    const std::vector<Layer>& getNetwork() const; //accesseur en mode lecture seulement
    std::vector<Layer>& getNetwork(); //accesseur en mode écriture
    void backpropagation(const std::string& filename, double alpha, const int input_size, const int output_size, int epochs); //l'algorithme de descente du gradient
    //prenant en compte toutes les données de la base, avec coefficient d'apprentissage et nombre d'itérations (epochs) en paramètre.
    int getNbCouches() const;
    int& getNbCouches();
    std::vector<double> prediction(std::vector<double> input);
    

};

// (possibilité de faire du polymorphisme plus tard en faisant hériter d'une classe Network des classes Dense et Convolutive
//dont les surcharges de backward_pass seront différent, en modifiant également les définitions des attributs Layer
 //selon la classe fille, puisque celles ci seront des filtres de convolution ou des matrices selons
//le type de réseau, en plus de donc modifier le calcul de gradient. De plus, un réseau convolutif me demanderai
//d'implémenter une classe Mat généralisée à plus de 2 dimensions, mais pourrai être adapté à ce projet. Si j'ai le temps)
    

#endif // DENSE_HPP
