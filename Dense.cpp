#include "Dense.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include "Utils.hpp"
#include <string>
#include <vector>
#include <chrono>

Dense::Dense(int N) : NbCouches(N), Network(N) {
}
Dense::~Dense() {
}

Dense::Dense(const Dense & other) {
    this->NbCouches = other.NbCouches;
    this->Network = other.Network;
}

Dense& Dense::operator=(const Dense & other) {
    if (this != &other) {
        this->NbCouches = other.NbCouches;
        this->Network = other.Network;
    }
    return *this;
}

std::vector<double> Dense::pass_forward(std::vector<double> input) { // on propage la donnée d'entrée (observation)
    // dans tout le réseau et on renvoie la sortie (prédiction)
    std::vector<double> activation = input; // activation initiale
    for (int i = 0; i < NbCouches-1; ++i) { // boucle sur le nombre de couche-1, car on a pas d'objet Layer Network(2)
        activation = Network[i].pass_forward(activation); // on appelle transmet l'info dans une couche
    }
    return activation;
}


void Dense::backward_pass(const std::vector<double>& donnee, double alpha, const int input_size, const int output_size) {
    //const int input_size = ((this->getNetwork()[0]).getWeight()).getnCols(); // on récupère la dimension d'entrée du réseau.
    //const int output_size = ((this->getNetwork()[this->getNbCouches() - 1]).getWeight()).getnRows(); // dim de sortie
    //std::cout<<"dimension input : "<<input_size<<" , dimension_output : "<<output_size<<std::endl;
    //on sépare la donnée en input et output : 
    
    std::vector<double> X(input_size);
    std::vector<double> Y_cible(output_size);
    std::copy(donnee.begin(), donnee.begin() + input_size, X.begin());
    std::copy(donnee.begin() + input_size, donnee.end(), Y_cible.begin()); 
    // on transmet la donnée X dans le réseau : 
    std::vector<double> Y_pred = this->pass_forward(X); // l'activation de toutes les couches est mise à jour
    //on calcule le gradient de la derniere Layer par rapport à Y_cible
    Network[this->getNbCouches() - 2].backward_pass(Y_cible, alpha); 
    //dans mon exemple, nbCouches = 3, => nbCouches-2 = 1, donc Network[1] est Layer1.
    std::vector<double>& xopti = ((this->getNetwork())[this->getNbCouches() - 2]).getXopti(); // exemple : 3-2 = 1 => X_opti de Layer1
    //qu'on vient de calculer par rapport à la donnée.  
    for (int i = NbCouches - 3; i >= 0; --i) { // boucle à un élément dans l'exemple : ok! on appelle couche0.backward_pass()
        xopti = (this->Network[i + 1]).getXopti();
        if (xopti.size() != (size_t)Network[i].getOutputDim()) {
            std::cerr << "Erreur : couche " << i << " attend un xopti de taille "
                      << Network[i].getOutputDim() << ", mais a reçu : "
                      << xopti.size() << std::endl;
            exit(EXIT_FAILURE);
        }
        Network[i].backward_pass(xopti, alpha);
        //std::cout<<"appel de backward_pass sur la couche "<<i<<std::endl;
    }
    
}


// Méthode pour configurer une couche donnée (input/output dims)

void Dense::set_layer(int index, int input_dim, int output_dim) {
    if (index < 0 || index >= NbCouches) {
        std::cerr << "Index de couche invalide\n";
        return;
    }
    Network[index] = Layer(input_dim, output_dim);
}

int Dense::getNbCouches() const { return NbCouches; }
int& Dense::getNbCouches() {
    return this->NbCouches;
}



void Dense::save_weights(const std::string& filename) const {
    std::ofstream ofs(filename);
    if (!ofs) {
        std::cerr << "Erreur : impossible d'ouvrir le fichier " << filename << " en écriture." << std::endl;
        return;
    }

    // Récupérer les dimensions du réseau dans un vecteur
    // La taille des dimensions = NbCouches
    // Dimensions = input_dim de la couche 0, output_dim couche 0, output_dim couche 1, ..., output_dim couche NbCouches-2
    // Puis la dernière dimension de sortie = output_dim de la dernière couche (couche NbCouches-2)
    std::vector<int> dims;
    const auto& net = this->getNetwork();
    if (net.empty()) {
        std::cerr << "Erreur : réseau vide, rien à sauvegarder." << std::endl;
        return;
    }

    dims.push_back(net[0].getInputDim());
    for (const auto& layer : net) {
        dims.push_back(layer.getOutputDim());
    }

    // Écrire la première ligne (dimensions)
    ofs << "(";
    for (size_t i = 0; i < dims.size(); ++i) {
        ofs << dims[i];
        if (i != dims.size() - 1) ofs << ", ";
    }
    ofs << ")" << std::endl;

    // Écrire chaque couche
    for (size_t i = 0; i < net.size(); ++i) {
        const Layer& layer = net[i];
        ofs << "Layer : " << i << std::endl;

        ofs << "Mat :" << std::endl;
        ofs << layer.getWeight() << std::endl;

        ofs << "Biais :" << std::endl;
        const std::vector<double>& biais = layer.getBiais();
        for (size_t j = 0; j < biais.size(); ++j) {
            ofs << biais[j];
            if (j != biais.size() - 1) ofs << " ";
        }
        ofs << std::endl;
    }
}



void Dense::load_weights(const std::string& filename) {
    std::ifstream ifs(filename);
    if (!ifs) {
        std::cerr << "Erreur : impossible d'ouvrir le fichier " << filename << std::endl;
        return;
    }

    std::string line;
    if (!std::getline(ifs, line)) {
        std::cerr << "Erreur lecture dimensions" << std::endl;
        return;
    }

    std::vector<int> dims;
    if (!parse_dimensions(line, dims)) {
        std::cerr << "Erreur parsing dimensions" << std::endl;
        return;
    }

    if (dims.size() < 2) {
        std::cerr << "Dimensions insuffisantes pour réseau" << std::endl;
        return;
    }

    this->NbCouches = dims.size();
    this->Network.clear();
    this->Network.reserve(NbCouches - 1);

    // Création des Layers avec constructeur Layer(input_dim, output_dim)
    for (size_t i = 0; i < NbCouches - 1; ++i) {
        Layer layer(dims[i], dims[i+1]);
        this->Network.push_back(layer);
    }

    // Lecture des couches
    for (size_t i = 0; i < NbCouches - 1; ++i) {
        // Layer : i
        if (!std::getline(ifs, line) || line != ("Layer : " + std::to_string(i))) {
            std::cerr << "Erreur lecture header couche " << i << std::endl;
            return;
        }

        // Mat :
        if (!std::getline(ifs, line) || line != "Mat :") {
            std::cerr << "Attendu 'Mat :' couche " << i << std::endl;
            return;
        }

        // Lire matrice poids
        Mat& W = this->Network[i].getWeight();
        if (!read_matrix_from_stream(ifs, W)) {
            std::cerr << "Erreur lecture matrice poids couche " << i << std::endl;
            return;
        }

        // Biais :
        if (!std::getline(ifs, line) || line != "Biais :") {
            std::cerr << "Attendu 'Biais :' couche " << i << std::endl;
            return;
        }

        // Lire vecteur biais
        std::vector<double>& B = this->Network[i].getBiais();
        if (!read_vector_from_stream(ifs, B, this->Network[i].getOutputDim())) {
            std::cerr << "Erreur lecture vecteur biais couche " << i << std::endl;
            return;
        }
    }

    std::cout << "Chargement du réseau terminé avec " << NbCouches << " couches." << std::endl;
}
const std::vector<Layer>& Dense::getNetwork() const{
    return this->Network;
}

std::vector<Layer>& Dense::getNetwork(){
    return this->Network;
}

/*
void Dense::set_layer(int input_dim, const std::vector<int>& hidden_dims, int output_dim) { // surcharge de set_layer
    // défini le nombre tota de couches et leurs dimensions
    Network.clear();
    NbCouches = hidden_dims.size() + 1;

    // Première couche (entrée -> première couche cachée)
    Network.emplace_back(Layer(input_dim, hidden_dims[0]));

    // Couches cachées suivantes
    for (size_t i = 1; i < hidden_dims.size(); ++i) {
        Network.emplace_back(Layer(hidden_dims[i - 1], hidden_dims[i]));
    }

    // Couche de sortie
    Network.emplace_back(Layer(hidden_dims.back(), output_dim));
}
*/


void Dense::backpropagation(const std::string& database, double alpha, const int input_size, const int output_size, int epochs) {
    int N = count_lines_in_csv(database) - 1; //nb de données,  on retire 1 pour l'en-tete des fichiers csv
    std::cout<<"Entrainement sur "<<N<<" donnees : "<<std::endl;
    std::vector<double> X;
    std::vector<double> Y;
    for (int epoch = 0; epoch < epochs; epoch++) {
    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "epoch : " << epoch << std::endl;
    for (int n = 1; n < 2000; n++) { 
        std::vector<double> donnee = lire_ligne_csv(database, n);
        backward_pass(donnee, alpha, input_size, output_size);
        }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Temps d'execution : " << elapsed.count() << " secondes" << std::endl;
    }
    std::cout << "Apprentissage termine" << std::endl;
}
