#pragma once
#include <iostream>
#include <iomanip>
#include <mpi.h>
#include <vector>
#include "Eigen/Dense"
#include "Eigen/Core"
#include "DummyDistributedVector.hpp"


class DistributedMatrix
{
public: /* should be private:*/
    int m_local_sz;
    int m_rank;
    int m_comm_sz;
    MPI_Comm* m_comm;

    void _start_of_binary_op(const DummyDistributedVector& other);


public:
    Eigen::VectorXd data;

    DistributedMatrix(MPI_Comm& comm, int local_sz);
    virtual void inplaceProduct(DummyDistributedVector& other) const = 0;
    virtual void product(DummyDistributedVector& out,
      const DummyDistributedVector & in) const = 0;
    virtual void print(std::string display_type) const = 0;
    virtual Eigen::MatrixXd plainMatrix() const = 0;
    virtual void inv() = 0;
};
