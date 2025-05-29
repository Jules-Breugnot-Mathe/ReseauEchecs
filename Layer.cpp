#include "Layer.hpp"
#include <vector>
#include "Utils.hpp"

Layer::Layer(int input_size, int output_size){
    this->input_dim = input_size; 
    this->output_dim = output_size;
    Mat A(output_size, input_size); // matrice aléatoire suivant la distribution de Xavier
    this->W = A;
    std::vector<double> vec = xavier_init_bias(input_size, output_size);// on utilise des vecteurs aléatoires
    this->B = vec;
    std::vector<double> acti = xavier_init_bias(output_size, input_size); // suivant une distribution de probas adaptée à
    //la fonction d'activation sigmoide
    this->X = acti;
    /*
    //on initialise aussi les gradients
    this->grad_B.resize(output_size);
    this->grad_X.resize(input_size);
    this->grad_W = Mat(output_dim, input_dim);
    for (int i=0; i<input_size; i++){
        for (int j=0; j<output_size; j++){
            this->grad_W.getcoef(i, j) = 0.0;
        }
    }
    */
    
}

Layer::~Layer(){}

Layer::Layer(const Layer & other){
    if (this != &other){
        this->input_dim = other.input_dim;
        this->output_dim = other.output_dim;
        this->B = other.B;
        this->W = other.W;
    }
}

Layer& Layer::operator=(const Layer & other){
    this->input_dim = other.input_dim;
    this->output_dim = other.output_dim;
    this->B = other.B;
    this->W = other.W;
    return *this;
}

std::vector<double> Layer::pass_forward(std::vector<double> X) {
    this->X = X;
    this->Z = this->W.matve(X);
    std::vector<double> Y(this->output_dim);
    for (int i = 0; i < this->output_dim; ++i) {
        this->Z[i] += this->B[i];
        Y[i] = 1.0 / (1.0 + std::exp(-this->Z[i]));
    }
    return Y;
}

void Layer::backward_pass(const std::vector<double>& Y_cible, double alpha) {

    if (Y_cible.size() != (size_t)this->output_dim) {
        std::cerr << "Erreur : taille de Y_cible = " << Y_cible.size() 
                  << ", attendue : " << this->output_dim << std::endl;
        exit(EXIT_FAILURE);
    }

    if (this->X.size() != (size_t)this->input_dim) {
        std::cerr << "Erreur : taille de X = " << this->X.size() 
                  << ", attendue : " << this->input_dim << std::endl;
        exit(EXIT_FAILURE);
    }
    //on va calculer les dérivées partielles des coefficients de Layer::W et de Layer::B 
    //par rapport au carré de la distance euclidienne entre la prédiction de la couche et Y_cible

    // Étape 0 : recalcul de Z = W * X + B
    this->Z = this->W.matve(this->X);
    for (int i = 0; i < this->output_dim; ++i) {
        this->Z[i] += this->B[i];
    }

    // Étape 1 : initialisations des gradients
    
    Mat grad_W(this->output_dim, this->input_dim);// on crée la matrice qui contiendra les dérivées partielles (DS/DWij)
    std::vector<double> grad_B(this->output_dim, 0.0);// on crée le vecteur qui contiendra les dérivées partielles (DS/DBi) 
    std::vector<double> grad_Activation(this->input_dim, 0.0); // vecteur des dérivées partielles (DS/DXi)
    

    // Étape 2 : calcul des gradients de W, B, et X
    for (int i = 0; i < this->output_dim; ++i) {
        double sigmoid_i = 1.0 / (1.0 + std::exp(-this->Z[i]));
        double dLoss_dYi = 2.0 * (sigmoid_i - Y_cible[i]);  // dérivée de la perte MSE
        double dYi_dZi = sigmoid_i * (1 - sigmoid_i);       // dérivée de sigmoid

        double delta_i = dLoss_dYi * dYi_dZi;               // dérivée de la perte par rapport à Zi

        // Gradient pour les biais
        grad_B[i] = delta_i;

        for (int j = 0; j < this->input_dim; ++j) {
            grad_W.getcoef(i, j) = delta_i * this->X[j];
            grad_Activation[j] += delta_i * this->W.getcoef(i, j); // pour X_opti
        }
    }
    //std::cout<<"grad de W : "<<grad_W<<std::endl;
    // Étape 3 : mise à jour des poids W et des biais B
    for (int i = 0; i < this->output_dim; ++i) {
        this->B[i] -= alpha * grad_B[i];
        for (int j = 0; j < this->input_dim; ++j) {
            this->W.getcoef(i, j) -= alpha * grad_W.getcoef(i, j);
        }
    }

    // Étape 4 : calcul et stockage de X_opti = X - alpha * grad_X
    X_opti.resize(this->input_dim);
    for (int j = 0; j < this->input_dim; ++j) {
        X_opti[j] = this->X[j] - alpha * grad_Activation[j];
    }
}

Mat& Layer::getWeight(){
    return this->W;
}

std::vector<double>& Layer::getBiais(){
    return this->B;
}

std::vector<double>& Layer::getActivation(){// renvoie l'activation à l'entrée d'une couche
    return this->X;
}

std::vector<double>& Layer::getXopti() {
    return X_opti;
}

std::vector<double>& Layer::getZ(){ // Z = WX + B
    return this->Z;
}

int Layer::getInputDim() const {
    return input_dim;
}

int Layer::getOutputDim() const {
    return output_dim;
}



const Mat& Layer::getWeight() const { // versions en lecture uniquement des accesseurs des poids et biais (*) d'une couche
    return W;
}

const std::vector<double>& Layer::getBiais() const { // (*)
    return B;
}

const std::vector<double>& Layer::getXopti() const {
    return X_opti;
}





/*
void Layer::backward_pass(const std::vector<double>& Y_cible, double alpha) {
    
    this->grad_X.assign(this->input_dim, 0.0);
    this->grad_B.assign(this->output_dim, 0.0);  // idem pour B si nécessaire
    // Optionnel : s'assurer que grad_W a les bonnes dimensions (pas obligatoire si alloué au début)
    if (this->grad_W.getnRows() != this->output_dim || this->grad_W.getnCols() != this->input_dim) {
    this->grad_W = Mat(this->output_dim, this->input_dim);
    }


    if (Y_cible.size() != (size_t)this->output_dim) {
        std::cerr << "Erreur : taille de Y_cible = " << Y_cible.size() 
                  << ", attendue : " << this->output_dim << std::endl;
        exit(EXIT_FAILURE);
    }

    if (this->X.size() != (size_t)this->input_dim) {
        std::cerr << "Erreur : taille de X = " << this->X.size() 
                  << ", attendue : " << this->input_dim << std::endl;
        exit(EXIT_FAILURE);
    }
    //on va calculer les dérivées partielles des coefficients de Layer::W et de Layer::B 
    //par rapport au carré de la distance euclidienne entre la prédiction de la couche 
    //calculée avec l'activation stockée et Y_cible

    // recalcul de Z = W * X + B
    this->Z = this->W.matve(this->X);
    for (int i = 0; i < this->output_dim; ++i) {
        this->Z[i] += this->B[i];
    }
    

    //calcul et mise à jour des gradients de W, B, et X 
    for (int i = 0; i < this->output_dim; ++i) {
        double sigmoid_i = 1.0 / (1.0 + std::exp(-this->Z[i]));
        double dLoss_dYi = 2.0 * (sigmoid_i - Y_cible[i]);  // dérivée de la perte norme 2
        double dYi_dZi = sigmoid_i * (1 - sigmoid_i);       // dérivée de sigmoid

        double delta_i = dLoss_dYi * dYi_dZi;               // dérivée de la perte par rapport à Zi

        // Gradient pour les biais
        this->grad_B[i] = delta_i;

        for (int j = 0; j < this->input_dim; ++j) {
            this->grad_W.getcoef(i, j) = delta_i * this->X[j];
            this->grad_X[j] += delta_i * this->W.getcoef(i, j); // pour X_opti
        }
    }
    //std::cout<<"grad de W : "<<grad_W<<std::endl;
    //  mise à jour des poids W et des biais B , on applique l'algo de la descente du gradient 
    for (int i = 0; i < this->output_dim; ++i) {
        this->B[i] -= alpha * grad_B[i];
        for (int j = 0; j < this->input_dim; ++j) {
            this->W.getcoef(i, j) -= alpha * grad_W.getcoef(i, j);
        }
    }

    // Étape 4 : calcul et stockage de X_opti = X - alpha * grad_X
    X_opti.resize(this->input_dim);
    for (int j = 0; j < this->input_dim; ++j) {
        X_opti[j] = this->X[j] - alpha * this->grad_X[j];
    }
}
*/





/*

*/
