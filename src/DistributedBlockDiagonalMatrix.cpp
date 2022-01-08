#include <iostream>
#include <iomanip>
#include <mpi.h>
#include <vector>
#include "Eigen/Dense"
#include "DistributedMatrix.hpp"
#include "DistributedBlockDiagonalMatrix.hpp"
#include "DummyDistributedVector.hpp"



DistributedBlockDiagonalMatrix::DistributedBlockDiagonalMatrix(MPI_Comm& comm, int local_sz)
  : DistributedMatrix(comm, local_sz) {}

void DistributedBlockDiagonalMatrix::inplaceProduct(DummyDistributedVector& other) const
{
    other.data.cwiseProduct(data);
}

void DistributedBlockDiagonalMatrix::product(DummyDistributedVector& out, const DummyDistributedVector & in) const
{
    out.data = data.asDiagonal() * in.data;
}

void DistributedBlockDiagonalMatrix::print(std::string display_type) const {
  std::string sep = "\n----------------------------------------\n";

  Eigen::IOFormat CleanFmt(3, 0, ", ", "\n", "[", "]");

  std::cout << "Matrix of size " << data.size() << std::endl;
  if (display_type == "diagonal") {
    std::cout << Eigen::MatrixXd(data.asDiagonal()).format(CleanFmt) << sep;
  } else {
    std::cout << data << sep;
  }
}


// piste
// https://stackoverflow.com/questions/28950857/how-to-construct-block-diagonal-matrix

template <typename Derived>
Eigen::MatrixXd blkdiag(const Eigen::MatrixBase<Derived>& a, int count)
{
    Eigen::MatrixXd bdm = Eigen::MatrixXd::Zero(a.rows() * count, a.cols() * count);
    for (int i = 0; i < count; ++i)
    {
        bdm.block(i * a.rows(), i * a.cols(), a.rows(), a.cols()) = a;
    }

    return bdm;
}
