#pragma once
#include <iostream>
#include <iomanip>
#include <mpi.h>
#include <vector>
#include "Eigen/Dense"
#include "DummyDistributedVector.hpp"
#include "DistributedMatrix.hpp"
#include "DistributedDiagonalMatrix.hpp"
#include "DistributedBlockDiagonalMatrix.hpp"


// This function is used to simulate a big network latency in order to be able to
// see the benefit of communication avoiding with a small number of cores

class DistributedBlockTridiagonalMatrix : public DistributedMatrix{
public:
  unsigned int m_blocksize;
  unsigned int m_nbblocks_diag;

public:
    DistributedBlockTridiagonalMatrix(MPI_Comm& comm,  int nb_blocks, int block_size);
    void inplaceProduct(DummyDistributedVector& other) const override;
    void product(DummyDistributedVector& out, const DummyDistributedVector & in) const override;
    Eigen::MatrixXd plainMatrix() const;
    void print(std::string display_type) const;
    void initFromMatrix(Eigen::MatrixXd);
    DistributedDiagonalMatrix extractDiagonal() const;
    DistributedBlockDiagonalMatrix extractBlockDiagonal() const;
};
