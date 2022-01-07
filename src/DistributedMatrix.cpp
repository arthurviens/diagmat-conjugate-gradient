#include <iostream>
#include <iomanip>
#include <mpi.h>
#include <vector>
#include "Eigen/Dense"
#include "DistributedMatrix.hpp"
#include "DummyDistributedVector.hpp"




DistributedMatrix::DistributedMatrix(MPI_Comm& comm, int local_sz)
{
    _comm = &comm;
    MPI_Comm_rank(*_comm, &_rank);
    MPI_Comm_size(*_comm, &_comm_sz);
    _local_sz = local_sz;
    data.resize(_local_sz); data.setZero();
}



void DistributedMatrix::_start_of_binary_op(const DummyDistributedVector& other)
{
    assert(data.rows() == other.rows());
}
