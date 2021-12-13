#pragma once
#include <iostream>
#include <iomanip>
#include <mpi.h>
#include <vector>
#include "Eigen/Dense"
#include "DummyDistributedVector.hpp"
// This function is used to simulate a big network latency in order to be able to
// see the benefit of communication avoiding with a small number of cores

class DistributedDiagonalMatrix
{
public: /* should be private:*/
    int _local_sz;
    int _rank;
    int _comm_sz;
    MPI_Comm* _comm;

    void _start_of_binary_op(const DummyDistributedVector& other);


public:
    Eigen::VectorXd data;

    DistributedDiagonalMatrix(MPI_Comm& comm, int local_sz);
    void inplaceProduct(DummyDistributedVector& other) const;
    void product(DummyDistributedVector& out, const DummyDistributedVector & in) const;
};
