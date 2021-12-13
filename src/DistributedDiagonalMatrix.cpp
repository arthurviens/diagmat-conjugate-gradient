#include <iostream>
#include <iomanip>
#include <mpi.h>
#include <vector>
#include "Eigen/Dense"
#include "DistributedDiagonalMatrix.hpp"
#include "DummyDistributedVector.hpp"


// This function is used to simulate a big network latency in order to be able to
// see the benefit of communication avoiding with a small number of cores

void DistributedDiagonalMatrix::_start_of_binary_op(const DummyDistributedVector& other)
{
    assert(data.rows() == other.rows());
}

DistributedDiagonalMatrix::DistributedDiagonalMatrix(MPI_Comm& comm, int local_sz)
{
    _comm = &comm;
    MPI_Comm_rank(*_comm, &_rank);
    MPI_Comm_size(*_comm, &_comm_sz);
    _local_sz = local_sz;
    data.resize(_local_sz); data.setZero();
}

void DistributedDiagonalMatrix::inplaceProduct(DummyDistributedVector& other) const
{
    other.data.cwiseProduct(data);
}

void DistributedDiagonalMatrix::product(DummyDistributedVector& out, const DummyDistributedVector & in) const
{
    out.data = data.asDiagonal() * in.data;
}
