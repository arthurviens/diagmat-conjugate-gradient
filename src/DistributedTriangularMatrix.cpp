#include <iostream>
#include <iomanip>
#include <mpi.h>
#include <vector>
#include <assert.h>
#include "Eigen/Dense"
#include "DistributedMatrix.hpp"
#include "DistributedTriangularMatrix.hpp"
#include "DummyDistributedVector.hpp"


DistributedTriangularMatrix::DistributedTriangularMatrix(MPI_Comm& comm, int local_sz)
  : DistributedMatrix(comm, local_sz * (local_sz + 1) / 2) {}


void DistributedTriangularMatrix::initFromMatrix(Eigen::MatrixXd A) {
    assert(A.rows() == A.cols());
    int local_sz = A.rows();
    for (int i = 0; i < local_sz; ++i) {
      for (int j = 0; j <= i; ++j) {
        data(i + i * (i-1) / 2 + j) = A(i, j);
      }
    }
  }


void DistributedTriangularMatrix::inplaceProduct(DummyDistributedVector& other) const
{
    other.data.cwiseProduct(data);
}

void DistributedTriangularMatrix::product(DummyDistributedVector& out, const DummyDistributedVector & in) const
{
  double tmp;
  int offset;
  for (int i = 0; i < m_local_sz; ++i) {
    tmp = 0;
    offset = i + i * (i-1) / 2;
    for (int j = 0; j <= i; ++j) {
      tmp += in.data(j) * data(offset + j);
    }
    out.data(i) = tmp;
  }
    //out.data = data.asDiagonal() * in.data;
}

Eigen::MatrixXd DistributedTriangularMatrix::plainMatrix() const {

  Eigen::MatrixXd fullMatrix(m_local_sz, m_local_sz);
  fullMatrix.setZero();
  for (int i = 0; i < m_local_sz; ++i) {
    for (int j = 0; j <= i; ++j) {
      fullMatrix(i, j) = data(i + i * (i-1) / 2 + j);
    }
  }
  return fullMatrix;
}


void DistributedTriangularMatrix::print(std::string display_type) const {
  std::string sep = "\n----------------------------------------\n";

  Eigen::MatrixXd toDisplay = plainMatrix();

  Eigen::IOFormat CleanFmt(3, 0, ", ", "\n", "[", "]");

  std::cout << "Triangular Matrix of size " << data.size() << std::endl;
  if (display_type == "no round") {
    std::cout << toDisplay << sep;
  } else {
    std::cout << toDisplay.format(CleanFmt) << sep;
  }
}

void DistributedTriangularMatrix::inv() {
  std::cout << "Im her triangular" << std::endl;
  Eigen::MatrixXd fullMatrix = plainMatrix();
  Eigen::MatrixXd inverse = fullMatrix.inverse();
  initFromMatrix(inverse);
}
