#include <iostream>
#include <iomanip>
#include <mpi.h>
#include <vector>
#include "Eigen/Dense"
#include "Eigen/Core"
#include "DistributedMatrix.hpp"
#include "DistributedDiagonalMatrix.hpp"
#include "DistributedBlockDiagonalMatrix.hpp"
#include "DistributedBlockTridiagonalMatrix.hpp"
#include "DummyDistributedVector.hpp"


DistributedDiagonalMatrix Jacobi(Eigen::MatrixXd A);

DistributedDiagonalMatrix SSOR(Eigen::MatrixXd A, double omega);
