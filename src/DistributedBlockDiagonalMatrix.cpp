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
    out.data = data.asDiagonal() * in.data;
}

void DistributedBlockDiagonalMatrix::print(std::string display_type) const {
  std::string sep = "\n----------------------------------------\n";
  //std::cout << "In print block matrix data size " << data.size() << " block size " << m_blocksize
  //  << " nb blocks " << m_nbblocks << std::endl;

  Eigen::MatrixXd toDisplay(m_blocksize * m_nbblocks, m_blocksize * m_nbblocks);
  for (unsigned int i = 0; i < (m_nbblocks); ++i) {
    for (unsigned int j = 0; j < m_blocksize; ++j) {
      for (unsigned int k = 0; k < m_blocksize; ++k) {
        //std::cout << i << " x " << j << " x " << k << " = " << i + j + k << " : " << data[i+j+k] << std::endl;
        toDisplay(i * m_blocksize + j, i * m_blocksize + k) = data[i + j + k];
      }
    }
  }

  Eigen::IOFormat CleanFmt(3, 0, ", ", "\n", "[", "]");

  std::cout << "Block Matrix of size " << data.size() << std::endl;
  if (display_type == "diagonal") {
    std::cout << toDisplay.diagonal() << sep;
  } else {
    std::cout << toDisplay.format(CleanFmt) << sep;
  }
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
