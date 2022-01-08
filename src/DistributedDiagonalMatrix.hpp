#pragma once
#include <iostream>
#include <iomanip>
#include <mpi.h>
#include <vector>
#include "Eigen/Dense"
#include "DistributedMatrix.hpp"
#include "DummyDistributedVector.hpp"
// This function is used to simulate a big network latency in order to be able to
// see the benefit of communication avoiding with a small number of cores

class DistributedDiagonalMatrix : public DistributedMatrix{
public:
    DistributedDiagonalMatrix(MPI_Comm& comm, int local_sz);
    //auto toDiagonal();
    void inplaceProduct(DummyDistributedVector& other) const override;
    void product(DummyDistributedVector& out, const DummyDistributedVector & in) const override;
    void print() const override;
};

/*template <typename Derived>
void print_mat(Eigen::EigenBase<Derived>& A);*/
