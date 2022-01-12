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
