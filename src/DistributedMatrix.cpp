#include <iostream>
#include <iomanip>
#include <mpi.h>
#include <vector>
#include "Eigen/Dense"
#include "Eigen/Core"
#include "DistributedMatrix.hpp"
#include "DummyDistributedVector.hpp"



DistributedMatrix::DistributedMatrix(MPI_Comm& comm, int local_sz)
{
    m_comm = &comm;
    MPI_Comm_rank(*m_comm, &m_rank);
    MPI_Comm_size(*m_comm, &m_comm_sz);
    m_local_sz = local_sz;
    data.resize(m_local_sz); data.setZero();
}


void DistributedMatrix::_start_of_binary_op(const DummyDistributedVector& other)
{
    assert(data.rows() == other.rows());
}
