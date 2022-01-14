#include <iostream>
#include <iomanip>
#include <mpi.h>
#include <vector>
#include "Eigen/Dense"
#include "DistributedMatrix.hpp"
#include "DistributedDiagonalMatrix.hpp"
#include "DummyDistributedVector.hpp"



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

Eigen::MatrixXd DistributedDiagonalMatrix::plainMatrix() const {

  Eigen::MatrixXd fullMatrix(m_local_sz, m_local_sz);
  fullMatrix.setZero();
  for (unsigned int i = 0; i < m_local_sz; ++i) {
    fullMatrix(i, i) = data[i];
  }
  return fullMatrix;
}


void DistributedDiagonalMatrix::print(std::string display_type) const {
  std::string sep = "\n----------------------------------------\n";

  Eigen::IOFormat CleanFmt(3, 0, ", ", "\n", "[", "]");

  std::cout << "Matrix of size " << data.size() << std::endl;
  if (display_type == "diagonal") {
    std::cout << Eigen::MatrixXd(data.asDiagonal()).format(CleanFmt) << sep;
  } else {
    std::cout << data << sep;
  }
}

void DistributedDiagonalMatrix::inv() {
  for (int i = 0; i < m_local_sz; ++i) {
    data(i) = 1 / data(i);
  }
}
