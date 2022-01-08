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

/*
Eigen::DiagonalMatrix DistributedDiagonalMatrix::toDiagonal() {
  return data.asDiagonal();
}*/

void DistributedDiagonalMatrix::inplaceProduct(DummyDistributedVector& other) const
{
    other.data.cwiseProduct(data);
}

void DistributedDiagonalMatrix::product(DummyDistributedVector& out, const DummyDistributedVector & in) const
{
    out.data = data.asDiagonal() * in.data;
}

void DistributedDiagonalMatrix::print() const {
  std::string sep = "\n----------------------------------------\n";

  //Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", " << ", ";");
  Eigen::IOFormat CleanFmt(3, 0, ", ", "\n", "[", "]");
  //Eigen::IOFormat OctaveFmt(Eigen::StreamPrecision, 0, ", ", ";\n", "", "", "[", "]");
  //Eigen::IOFormat HeavyFmt(Eigen::FullPrecision, 0, ", ", ";\n", "[", "]", "[", "]");


  std::cout << "Matrix of size " << data.size() << std::endl;
  std::cout << data.format(CleanFmt) << sep;
}

/*
template <typename Derived>
void print_mat(Eigen::EigenBase<Derived>& A) {
  std::cout << "in print_mat" << std::endl;
  std::string sep = "\n----------------------------------------\n";

  //Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", " << ", ";");
  Eigen::IOFormat CleanFmt(3, 0, ", ", "\n", "[", "]");
  //Eigen::IOFormat OctaveFmt(Eigen::StreamPrecision, 0, ", ", ";\n", "", "", "[", "]");
  //Eigen::IOFormat HeavyFmt(Eigen::FullPrecision, 0, ", ", ";\n", "[", "]", "[", "]");


  std::cout << "Matrix of size " << A.size() << std::endl;
  std::cout << A.format(CleanFmt) << sep;
}*/
