#include <iostream>
#include <iomanip>
#include <mpi.h>
#include <vector>
#include <assert.h>
#include "Eigen/Dense"
#include "DistributedMatrix.hpp"
#include "DistributedBlockDiagonalMatrix.hpp"
#include "DistributedBlockTridiagonalMatrix.hpp"
#include "DummyDistributedVector.hpp"



DistributedBlockTridiagonalMatrix::DistributedBlockTridiagonalMatrix(MPI_Comm& comm, int nb_blocks_diag, int block_size)
  : DistributedMatrix(comm, nb_blocks_diag * block_size * block_size + 2 * (nb_blocks_diag - 1) * block_size * block_size) {
    m_blocksize = block_size;
    m_nbblocks_diag = nb_blocks_diag;
}


void DistributedBlockTridiagonalMatrix::initFromMatrix(Eigen::MatrixXd base) {
    double f_value;
    int blocksize_squared = m_blocksize * m_blocksize;
    int nrows = base.rows();
    int ncols = base.cols();
    assert(nrows == ncols);
    assert(m_blocksize * m_nbblocks_diag == nrows);

    for (unsigned int i = 0; i < (m_nbblocks_diag); ++i) {
      for (int t = -1; t < 2; ++t) {
        if (((i != 0) | (t != -1)) & ((i != m_nbblocks_diag - 1) | (t != 1) )) {
          for (unsigned int j = 0; j < m_blocksize; ++j) {
            for (unsigned int k = 0; k < m_blocksize; ++k) {
              f_value = base(i * m_blocksize + j, i * m_blocksize + t * m_blocksize + k);
              if (f_value != 0) {
                data[i * 3 * blocksize_squared + t * blocksize_squared + j * m_blocksize + k] = f_value;
              } else {
                data[i * 3 * blocksize_squared + t * blocksize_squared + j * m_blocksize + k] =
                    base(i * m_blocksize + t * m_blocksize + k, i * m_blocksize + j);
              }
            }
          }
        }
      }
    }
}


void DistributedBlockTridiagonalMatrix::inplaceProduct(DummyDistributedVector& other) const
{
    other.data.cwiseProduct(data);
}

void DistributedBlockTridiagonalMatrix::product(DummyDistributedVector& out, const DummyDistributedVector & in) const
{
  int blocksize_squared = m_blocksize * m_blocksize;
  double tmp;

  for (unsigned int i = 0; i < (m_nbblocks_diag); ++i) {
    for (unsigned int j = 0; j < m_blocksize; ++j) {
      tmp = 0;
      for (int t = -1; t < 2; ++t) {
        if (((i != 0) | (t != -1)) & ((i != m_nbblocks_diag - 1) | (t != 1) )) {
          for (unsigned int k = 0; k < m_blocksize; ++k) {
            tmp += in.data[i * m_blocksize + t * m_blocksize + k]
                  * data[i * 3 * blocksize_squared + t * blocksize_squared + j * m_blocksize + k];
            /*std::cout << "Adding " << in.data[i * m_blocksize + t * m_blocksize + k] << " * "
              << data[i * 3 * blocksize_squared + t * blocksize_squared + j * m_blocksize + k] << " (data["
              << i * 3 * blocksize_squared + t * blocksize_squared + j * m_blocksize + k << "] to tmp" << std::endl;*/
          }
        }
      }
      out.data[i * m_blocksize + j] = tmp;
      //std::cout << "Writing " << tmp << " in c[" << m_blocksize * i + j << "]" << std::endl;
    }
  }
}



DistributedDiagonalMatrix DistributedBlockTridiagonalMatrix::extractDiagonal() const {
    MPI_Comm comm;
    MPI_Comm_dup(MPI_COMM_WORLD, &comm);
    DistributedDiagonalMatrix D(comm, m_nbblocks_diag * m_blocksize);
    int blocksize_squared = m_blocksize * m_blocksize;

    for (unsigned int i = 0; i < (m_nbblocks_diag); ++i) {
      for (unsigned int j = 0; j < m_blocksize; ++j) {
          D.data[i * m_blocksize + j] = data[i * 3 * blocksize_squared + j * m_blocksize + j];
      }
    }
    return D;
}


DistributedBlockDiagonalMatrix DistributedBlockTridiagonalMatrix::extractBlockDiagonal() const {
    MPI_Comm comm;
    MPI_Comm_dup(MPI_COMM_WORLD, &comm);
    DistributedBlockDiagonalMatrix D(comm, m_nbblocks_diag, m_blocksize);
    int blocksize_squared = m_blocksize * m_blocksize;

    for (unsigned int i = 0; i < (m_nbblocks_diag); ++i) {
      for (unsigned int j = 0; j < m_blocksize; ++j) {
          for (unsigned int k = 0; k < m_blocksize; ++k) {
          D.data[i * blocksize_squared + j * m_blocksize + k] = data[i * 3 * blocksize_squared + j * m_blocksize + k];
        }
      }
    }
    return D;
}


Eigen::MatrixXd DistributedBlockTridiagonalMatrix::plainMatrix() const {
  double dat;
  int blocksize_squared = m_blocksize * m_blocksize;

  Eigen::MatrixXd fullMatrix(m_blocksize * m_nbblocks_diag, m_blocksize * m_nbblocks_diag);
  fullMatrix.setZero();
  for (unsigned int i = 0; i < (m_nbblocks_diag); ++i) {
    for (int t = -1; t < 2; ++t) {
      if (((i != 0) | (t != -1)) & ((i != m_nbblocks_diag - 1) | (t != 1) )) {
        for (unsigned int j = 0; j < m_blocksize; ++j) {
          for (unsigned int k = 0; k < m_blocksize; ++k) {
            dat = data[i * 3 * blocksize_squared + t * blocksize_squared + j * m_blocksize + k];
            fullMatrix(i * m_blocksize + j, i * m_blocksize + t * m_blocksize + k) = dat;
            //fullMatrix(i * m_blocksize + t * m_blocksize + k, i * m_blocksize + j) = dat;
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

  Eigen::IOFormat CleanFmt(2, 0, ", ", "\n", "[", "]");

  std::cout << "Block Matrix of size " << data.size() << std::endl;
  if (display_type == "diagonal") {
    std::cout << toDisplay << sep;
  } else {
    std::cout << toDisplay.format(CleanFmt) << sep;
  }
}

void DistributedBlockTridiagonalMatrix::inv() {
  int i = 0;
  ++i;
}
