#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iostream>
#include "Utils.hpp"
#include <random>
#include <chrono>
#include <cmath>
#include <cctype>
#include "Matrix.hpp"

#include <cstdint>
#include "Plateau.hpp"
#include "Dense.hpp"





int count_lines_in_csv(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Erreur : impossible d’ouvrir le fichier " << filename << std::endl;
        return -1;
    }

    int line_count = 0;
    std::string line;
    while (std::getline(file, line)) {
        ++line_count;
    }

    file.close();
    return line_count;
}

std::vector<double> lire_ligne_csv(const std::string& nom_fichier, int ligne_index) {
    std::ifstream fichier(nom_fichier);
    std::string ligne;
    int ligne_courante = 0;

    while (std::getline(fichier, ligne)) {
        if (ligne_courante == ligne_index) {
            std::vector<double> valeurs;
            std::stringstream ss(ligne);
            std::string cellule;

            while (std::getline(ss, cellule, ',')) {
                try {
                    valeurs.push_back(std::stod(cellule));
                } catch (...) {
                    std::cerr << "Conversion invalide : '" << cellule << "' ignorée" << std::endl;
                }
            }
            return valeurs;
        }
        ++ligne_courante;
    }

    return {};
}

int countColumns(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Erreur : impossible d'ouvrir le fichier " << filename << "\n";
        return -1; // Indique une erreur
    }

    std::string line;
    if (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string column;
        int count = 0;
        while (std::getline(ss, column, ',')) {
            count++;
        }
        return count;
    } else {
        std::cerr << "Le fichier est vide.\n";
        return -1; // Indique une erreur
    }
}



std::vector<double> xavier_init_vector(int n_in, int n_out) {
    int size = n_in * n_out;
    std::vector<double> vec(size);

    // Calcul de la borne de l'intervalle de Xavier Uniform
    double limit = std::sqrt(6.0 / (n_in + n_out));

    // Générateur aléatoire avec une seed basée sur l'horloge
    std::mt19937 gen(std::chrono::steady_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<double> dist(-limit, limit);

    // Remplissage du vecteur
    for (int i = 0; i < size; ++i) {
        vec[i] = dist(gen);
    }

    return vec;
}


std::vector<double> xavier_init_bias(int input_size, int output_size) {
    std::vector<double> bias(output_size);

    double limit = std::sqrt(6.0 / (input_size + output_size));
    std::mt19937 gen(std::chrono::steady_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<double> dist(-limit, limit);

    for (int i = 0; i < output_size; ++i) {
        bias[i] = dist(gen);
    }

    return bias;
}


// parse la ligne contenant les dimensions du réseau, format (97, 64, 1) pour mon exemple
bool parse_dimensions(const std::string& line, std::vector<int>& dims) {
    size_t start = line.find('(');
    size_t end = line.find(')');
    if (start == std::string::npos || end == std::string::npos || end <= start) {
        return false;
    }
    std::string inside = line.substr(start + 1, end - start - 1);
    std::stringstream ss(inside);
    dims.clear();
    std::string token;
    while (std::getline(ss, token, ',')) {
        // Enlever espaces autour
        size_t first = token.find_first_not_of(" \t");
        size_t last = token.find_last_not_of(" \t");
        if (first == std::string::npos) return false;
        token = token.substr(first, last - first + 1);
        try {
            dims.push_back(std::stoi(token));
        } catch (...) {
            return false;
        }
    }
    return true;
}


// Format supposé par ligne : chaque ligne correspond à une ligne de la matrice, les valeurs séparées par espaces
//donc adapté à ma surcharge de << pour Mat
bool read_matrix_from_stream(std::istream& is, Mat& mat) {
    int rows = mat.getnRows();
    int cols = mat.getnCols();
    for (int i = 0; i < rows; ++i) {
        std::string line;
        if (!std::getline(is, line)) return false;
        std::stringstream ss(line);
        for (int j = 0; j < cols; ++j) {
            double val;
            if (!(ss >> val)) return false;
            mat.getcoef(i, j) = val;
        }
    }
    return true;
}

// lit un vecteur de biais (une ligne, valeurs séparées par espaces)
bool read_vector_from_stream(std::istream& is, std::vector<double>& vec, int expected_size) {
    std::string line;
    if (!std::getline(is, line)) return false;
    std::stringstream ss(line);
    vec.clear();
    double val;
    while (ss >> val) {
        vec.push_back(val);
    }
    return (int)vec.size() == expected_size;
}


int evaluation(const Plateau& plateau, const Dense& reseau) {
    std::vector<double> input;

    // Liste des 12 bitboards dans l’ordre
    std::vector<uint64_t> bitboards = {
        plateau.getPionsBlancs(), plateau.getPionsNoirs(),
        plateau.getToursBlancs(), plateau.getToursNoirs(),
        plateau.getCavaliersBlancs(), plateau.getCavaliersNoirs(),
        plateau.getFousBlancs(), plateau.getFousNoirs(),
        plateau.getDamesBlancs(), plateau.getDamesNoirs(),
        plateau.getRoiBlanc(), plateau.getRoiNoir()
    };

    // Convertit chaque bitboard en 8 doubles
    for (const auto& bb : bitboards) {
        for (int i = 7; i >= 0; --i) {
            uint8_t byte = (bb >> (i * 8)) & 0xFF;
            input.push_back(static_cast<double>(byte));
        }
    }

    // Vérification finale
    if (input.size() != 96) {
        std::cerr << "Erreur : vecteur d'entrée invalide, taille = " << input.size() << std::endl;
        return -1;
    }

    // Passage dans le réseau
    std::vector<double> output = reseau.pass_forward(input);

    if (output.empty()) {
        std::cerr << "Erreur : le réseau a retourné un vecteur vide." << std::endl;
        return -1;
    }

    // Sortie : arrondie à l’entier
    return static_cast<int>(output[0]);
}