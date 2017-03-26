//
// Created by di miao on 12/23/16.
//

#ifndef OO_IGA_KNOTVECTOR_H
#define OO_IGA_KNOTVECTOR_H
#ifndef NDEBUG
#   define ASSERT(condition, message) \
    do { \
        if (! (condition)) { \
            std::cerr << "Assertion `" #condition "` failed in " << __FILE__ \
                      << " line " << __LINE__ << ": " << message << std::endl; \
            std::terminate(); \
        } \
    } while (false)
#else
#   define ASSERT(condition, message) do { } while (false)
#endif
#include <vector>
#include <map>
#include <eigen3/Eigen/Dense>
#include <iostream>

template<typename T>
class KnotVector {
public:
    using knotContainer = std::vector<T>;
    using uniContainer = std::map<T, unsigned>;

    //methods
    KnotVector() {};

    KnotVector(const knotContainer &target);

    KnotVector(const uniContainer &target);

    KnotVector(const KnotVector &target);

    const T& operator[](unsigned i) const;

    void Insert(T knot);

    void printUnique() const;

    void printKnotVector() const;

    void UniformRefine(unsigned r = 1, unsigned multi = 1);

    void RefineSpan(std::pair<T, T>, unsigned r = 1, unsigned multi = 1);

    Eigen::Matrix<T, Eigen::Dynamic, 1> MapToEigen() const;

    unsigned GetDegree() const;

    unsigned GetSize() const;

    void InitClosed(unsigned _deg, T first=T(0.0), T last=T(1.0));

    void InitClosedUniform(unsigned _dof, unsigned _deg, T first=T(0.0), T last=T(1.0));

    KnotVector UniKnotUnion(const KnotVector & vb) const;

    std::vector<std::pair<T,T>> KnotSpans() const;

    knotContainer GetKnotVector() const;

    void Remove(T knot, unsigned multiplicity);
private:
    //Knots with repetitions {0,0,0,.5,1,1,1}
    knotContainer _multiKnots;
    //Unique knots associated with multiplicity {0:3,.5:1,1:3}
    uniContainer _uniKnots;
    //


    void UniQue();

    void MultiPle();

};


#endif //OO_IGA_KNOTVECTOR_H
