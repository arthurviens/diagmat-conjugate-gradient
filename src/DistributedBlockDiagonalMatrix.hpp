#pragma once
#include <iostream>
#include <iomanip>
#include <mpi.h>
#include <vector>
#include "Eigen/Dense"
#include "DummyDistributedVector.hpp"
#include "DistributedMatrix.hpp"
#include "DistributedDiagonalMatrix.hpp"


// This function is used to simulate a big network latency in order to be able to
// see the benefit of communication avoiding with a small number of cores

class DistributedBlockDiagonalMatrix : public DistributedMatrix{
public:
  unsigned int m_blocksize;
  unsigned int m_nbblocks;

public:
    DistributedBlockDiagonalMatrix(MPI_Comm& comm,  int nb_blocks, int block_size);
    void initProductData();
    void inplaceProduct(DummyDistributedVector& other) const override;
    void product(DummyDistributedVector& out, const DummyDistributedVector & in) const override;
    void print(std::string display_type) const override;
    Eigen::MatrixXd plainMatrix() const override;
    void makeDataSymetric();
    DistributedDiagonalMatrix extractDiagonal() const;
    void inv() override;
};
