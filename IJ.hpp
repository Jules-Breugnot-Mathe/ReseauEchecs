#ifndef IJ_HPP
#define IJ_HPP

class IJ {
private:
    int i, j;

public:
    IJ(int i, int j);
    virtual ~IJ();
    IJ(const IJ& other);
    IJ& operator=(const IJ& other);

    int geti() const;
    int getj() const;

    friend bool operator<(const IJ& ij1, const IJ& ij2);
};

#endif // IJ_HPP
