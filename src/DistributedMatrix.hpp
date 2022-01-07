#pragma once
#include <iostream>
#include <iomanip>
#include <mpi.h>
#include <vector>
#include "Eigen/Dense"
#include "DummyDistributedVector.hpp"


class DistributedMatrix
{
public: /* should be private:*/
    int _local_sz;
    int _rank;
    int _comm_sz;
    MPI_Comm* _comm;

    void _start_of_binary_op(const DummyDistributedVector& other);


public:
    Eigen::VectorXd data;

    DistributedMatrix(MPI_Comm& comm, int local_sz);
    virtual void inplaceProduct(DummyDistributedVector& other) const = 0;
    virtual void product(DummyDistributedVector& out,
      const DummyDistributedVector & in) const = 0;
};
