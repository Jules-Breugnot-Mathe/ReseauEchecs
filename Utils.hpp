#ifndef UTILS_HPP
#define UTILS_HPP

#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include "Matrix.hpp"
#include <cstdint>
#include "Dense.hpp"
#include "plateau.hpp" // classe plateau de Corentin


int count_lines_in_csv(const std::string& filename);
std::vector<double> lire_ligne_csv(const std::string& nom_fichier, int ligne_index);
int countColumns(const std::string& filename);
std::vector<double> xavier_init_vector(int n_in, int n_out);
std::vector<double> xavier_init_bias(int input_size, int output_size);
bool parse_dimensions(const std::string& line, std::vector<int>& dims);
bool read_matrix_from_stream(std::istream& is, Mat& mat);
bool read_vector_from_stream(std::istream& is, std::vector<double>& vec, int expected_size);
int evaluation(const Plateau& plateau, const Dense& reseau, bool Tour);
std::vector<double> prediction(Dense& Reseau, std::vector<double> input); // input de dimension 97 pour les échecs

#endif // UTILS_HPP
