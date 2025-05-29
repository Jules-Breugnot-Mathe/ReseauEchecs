#Implémentation d'un Reseau de Neurones dense, et d'un algorithme de rétropropagation du gradient
#à l'aide de le base de données *"chess_positions.csv"* convertie de pgn à bitboard.

##objectif : 
Ce projet vise à implémenter un algorithme d'apprentissage supervisé dans le but d'évaluer des positions d'échecs
pour, à terme, programmer un moteur d'échecs simple. Ce projet n'utilise pas de bibliothèque de machine learning.

##mise en oeuvre : 
le format bitboard stocke des positions d'echecs sous la forme de 12 nombres de type uint64_t hexadécimaux. 
Chacun de ces nombre peux être représenté par 64 chiffres binaires, donc un plateau de 64 cases.
Par exemple pour les pions blancs, le nombre 0000FF00, soit , en décimal, 
00000000 00000000 00000000 00000000 00000000 00000000 11111111 00000000
nous indique avec des 1 les positions des représentants des pions sur le plateau.
ces nombres sont coupés en paquet de 2 hexadécimaux, puis tous concaténé pour former un vecteur de 96 entrées qui est l'entrée du réseau. 

##rétropopagation du gradient : 
La classe Dense est codée avec un vecteur d'objets de type Layer, représentant les couches.
La classe Layer possède une matrice de poids, un vecteur de biais, et un vecteur d'activation, ainsi qu'un vecteur activation optimale.
La méthode backward_pass de la n-ième couche Layer calcule la distance euclidienne entre la sortie du réseau Y = sigmoid(WX + B) et l'activation de la couche n+1. Ensuite, les dérivées partielles des coefficients des poids, biais, et activation par rapport à cette distance sont calculées, et les gradients sont construit. On applique une méthode de destente de gradient, et on ajuste les coef de la couche n pour minimiser cette distance. On stocke donc dans le vecteur activation optimale l'activation ajustée, sur laquelle on pourra calculer le gradient de la couche n-1. 

##architecture : 
Dans mon cas, j'ai choisi un réseau de couche d'entrée de dimension 96, deux couches cachées de dimension 64 et une couche de sortie scalaire, soit plus de 10300 neurones. L'entrainement s'est déroulé sur 5 epoch en parcourant les 20 000 données de la base.


##exemple d'utilisation pour les échecs : 
```//besoin de la classe Plateau stockant le bitboard

    //le réseau adapté aux échecs est déjà stocké dans le fichier Model.txt, il suffit de le charger avec load_weights
    Dense Reseau;
    Reseau.load_weights("Model.txt");
    Reseau.backpropagation("chess_positions.csv", 1, 97, 1, 5);
    Reseau.save_weights("Model.txt");```

##exemple d'utilisation générale : 
//commenter dans les fichiers Utils.hpp et Utils.cpp les lignes
//#include "plateau.hpp", et les déclaration et implémentation de la fonction evaluation. 
//constuire un réseau de cette manière
```int main(){
    Dense Reseau;
    Layer L0(97, 64);
    Layer L1(64, 64);
    Layer L2(64, 1);
    Reseau.getNbCouches() = 4;
    std::vector<Layer> layers{L0, L1, L2};
    Reseau.getNetwork() = layers;
    //entrainement sur 5 epoch : 
    Reseau.backpropagation("chess_positions.csv", 1, 97, 1, 5);
    Reseau.save_weights("Model.txt");
    return 0;
//utiliser la méthode `std::vector<double> Dense::pass_forward(std::vector<double> input);` pour caluler la sortie correspondant à la prediction.
}```

    