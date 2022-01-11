#include <iostream>
#include <iomanip>
#include <mpi.h>
#include <vector>
#include <assert.h>
#include "Eigen/Dense"
#include "DistributedMatrix.hpp"
#include "DistributedBlockTridiagonalMatrix.hpp"
#include "DummyDistributedVector.hpp"



DistributedBlockTridiagonalMatrix::DistributedBlockTridiagonalMatrix(MPI_Comm& comm, int nb_blocks_diag, int block_size)
  : DistributedMatrix(comm, nb_blocks_diag * block_size * block_size + 2 * (nb_blocks_diag - 1) * block_size * block_size) {
    m_blocksize = block_size;
    m_nbblocks_diag = nb_blocks_diag;
}


void DistributedBlockTridiagonalMatrix::inplaceProduct(DummyDistributedVector& other) const
{
    other.data.cwiseProduct(data);
}

void DistributedBlockTridiagonalMatrix::product(DummyDistributedVector& out, const DummyDistributedVector & in) const
{
    Eigen::MatrixXd fullMatrix = plainMatrix();

    out.data = fullMatrix * in.data;
}


Eigen::MatrixXd DistributedBlockTridiagonalMatrix::plainMatrix() const {
  int blocksize_squared = m_blocksize * m_blocksize;

  Eigen::MatrixXd fullMatrix(m_blocksize * m_nbblocks_diag, m_blocksize * m_nbblocks_diag);
  fullMatrix.setZero();
  for (unsigned int i = 0; i < (m_nbblocks_diag); ++i) {
    for (int t = -1; t < 2; ++t) {
      if (((i != 0) | (t != -1)) & ((i != m_nbblocks_diag - 1) | (t != 1) )) {
        for (unsigned int j = 0; j < m_blocksize; ++j) {
          for (unsigned int k = 0; k < m_blocksize; ++k) {
            fullMatrix(i * m_blocksize + j, i * m_blocksize + t * m_blocksize + k)
                = data[i * 3 * blocksize_squared + t * blocksize_squared + j * m_blocksize + k];
          }
        }
      }
    }
  }
  return fullMatrix;
}


void DistributedBlockTridiagonalMatrix::print(std::string display_type) const {
  std::string sep = "\n----------------------------------------\n";

  Eigen::MatrixXd toDisplay = plainMatrix();

  Eigen::IOFormat CleanFmt(3, 0, ", ", "\n", "[", "]");

  std::cout << "Block Matrix of size " << data.size() << std::endl;
  if (display_type == "diagonal") {
    std::cout << toDisplay << sep;
  } else {
    std::cout << toDisplay.format(CleanFmt) << sep;
  }
}
