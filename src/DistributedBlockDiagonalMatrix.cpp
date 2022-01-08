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

void DistributedBlockDiagonalMatrix::print() const {
  std::string sep = "\n----------------------------------------\n";

  //Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", " << ", ";");
  Eigen::IOFormat CleanFmt(3, 0, ", ", "\n", "[", "]");
  //Eigen::IOFormat OctaveFmt(Eigen::StreamPrecision, 0, ", ", ";\n", "", "", "[", "]");
  //Eigen::IOFormat HeavyFmt(Eigen::FullPrecision, 0, ", ", ";\n", "[", "]", "[", "]");

  std::cout << "Matrix of size " << data.size() << std::endl;
  std::cout << data.format(CleanFmt) << sep;
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
