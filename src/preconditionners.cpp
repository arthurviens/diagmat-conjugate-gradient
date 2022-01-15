#include <iostream>
#include <iomanip>
#include <mpi.h>
#include <vector>
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "Eigen/Core"
#include "DistributedMatrix.hpp"
#include "DistributedDiagonalMatrix.hpp"
#include "DistributedBlockDiagonalMatrix.hpp"
#include "DistributedBlockTridiagonalMatrix.hpp"
#include "DummyDistributedVector.hpp"

using namespace Eigen;


DistributedDiagonalMatrix Jacobi(MatrixXd A) {
  VectorXd D = A.diagonal();

  MPI_Comm comm;
  MPI_Comm_dup(MPI_COMM_WORLD, &comm);

  DistributedDiagonalMatrix Mdiag(comm, D.size());
  Mdiag.data = D;
  return Mdiag;
}


DistributedDiagonalMatrix SSOR(MatrixXd A, double omega) {
    MatrixXd E = A.triangularView<Lower>();

    MatrixXd D = E.diagonal().asDiagonal();
    E -= D;

    MatrixXd M = (D - omega * E) * D.inverse() * (D - omega * E.transpose());
    VectorXd Mdiag_vec = M.diagonal();

    MPI_Comm comm;
    MPI_Comm_dup(MPI_COMM_WORLD, &comm);

    DistributedDiagonalMatrix Mdiag(comm, Mdiag_vec.size());

    Mdiag.data = Mdiag_vec;
    return Mdiag;
}
