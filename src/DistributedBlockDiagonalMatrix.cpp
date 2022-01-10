#include <iostream>
#include <iomanip>
#include <mpi.h>
#include <vector>
#include <assert.h>
#include "Eigen/Dense"
#include "DistributedMatrix.hpp"
#include "DistributedBlockDiagonalMatrix.hpp"
#include "DummyDistributedVector.hpp"



DistributedBlockDiagonalMatrix::DistributedBlockDiagonalMatrix(MPI_Comm& comm, int nb_blocks, int block_size)
  : DistributedMatrix(comm, nb_blocks * block_size * block_size) {
    m_blocksize = block_size;
    m_nbblocks = nb_blocks;
}


void DistributedBlockDiagonalMatrix::inplaceProduct(DummyDistributedVector& other) const
{
    other.data.cwiseProduct(data);
}

void DistributedBlockDiagonalMatrix::product(DummyDistributedVector& out, const DummyDistributedVector & in) const
{
    Eigen::MatrixXd fullMatrix = plainMatrix();

    out.data = fullMatrix * in.data;
}


Eigen::MatrixXd DistributedBlockDiagonalMatrix::plainMatrix() const {
  int blocksize_squared = m_blocksize * m_blocksize;

  Eigen::MatrixXd fullMatrix(m_blocksize * m_nbblocks, m_blocksize * m_nbblocks);
  fullMatrix.setZero();
  for (unsigned int i = 0; i < (m_nbblocks); ++i) {
    for (unsigned int j = 0; j < m_blocksize; ++j) {
      for (unsigned int k = 0; k < m_blocksize; ++k) {
        //std::cout << i << " x " << j << " x " << k << " = " << i + j + k << " : " << data[i+j+k] << std::endl;
        fullMatrix(i * m_blocksize + j, i * m_blocksize + k) = data[i * blocksize_squared + j * m_blocksize + k];
      }
    }
  }
  return fullMatrix;
}

// [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16]

void DistributedBlockDiagonalMatrix::makeDataSymetric() {
  int blocksize_squared = m_blocksize * m_blocksize;
  data(0) = data[blocksize_squared];
  for (unsigned int i = 0; i < (m_nbblocks); ++i) {
    for (unsigned int j = 0; j < m_blocksize; ++j) {
      for (unsigned int k = 0; k < m_blocksize; ++k) {
        if ((j > k)) {
          data[i * blocksize_squared + j * m_blocksize + k] = data[i * blocksize_squared + k * m_blocksize + j];
        }
      }
    }
  }
}


void DistributedBlockDiagonalMatrix::print(std::string display_type) const {
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
