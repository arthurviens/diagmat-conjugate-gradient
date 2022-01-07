#include <iostream>
#include <iomanip>
#include <mpi.h>
#include <vector>
#include "Eigen/Dense"
#include "DistributedMatrix.hpp"
#include "DistributedDiagonalMatrix.hpp"
#include "DummyDistributedVector.hpp"


// This function is used to simulate a big network latency in order to be able to
// see the benefit of communication avoiding with a small number of cores


DistributedDiagonalMatrix::DistributedDiagonalMatrix(MPI_Comm& comm, int local_sz)
  : DistributedMatrix(comm, local_sz) {}

void DistributedDiagonalMatrix::inplaceProduct(DummyDistributedVector& other) const
{
    other.data.cwiseProduct(data);
}

void DistributedDiagonalMatrix::product(DummyDistributedVector& out, const DummyDistributedVector & in) const
{
    out.data = data.asDiagonal() * in.data;
}
