#pragma once
#include <iostream>
#include <iomanip>
#include <thread>         // std::this_thread::sleep_for
#include <chrono>         // std::chrono::seconds
#include <memory>
#include <mpi.h>
#include <vector>
#include "Eigen/Dense"

// This function is used to simulate a big network latency in order to be able to
// see the benefit of communication avoiding with a small number of cores
void Dummy_MPI_Allreduce(const void *sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
void Dummy_MPI_Iallreduce(const void *sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, MPI_Request& req);

class DummyDistributedVector
{
public: /* should be private:*/
    int _local_sz;
    int _rank;
    int _comm_sz;
    MPI_Comm* _comm;

    void _start_of_binary_op(const DummyDistributedVector& other);

public:

    Eigen::VectorXd data;

    DummyDistributedVector(MPI_Comm& comm, int local_sz);
    unsigned int rows() const;
    bool operator==(const DummyDistributedVector& other) const;
    bool operator!=(const DummyDistributedVector& other) const;
    DummyDistributedVector& operator+=(const DummyDistributedVector& other);
    DummyDistributedVector& operator-=(const DummyDistributedVector& other);
    DummyDistributedVector& axpy(double alpha, DummyDistributedVector& other);
    DummyDistributedVector& operator*=(double alpha);

    double ltransposeProduct(const DummyDistributedVector& other);
    void transposeProduct(double& out, const DummyDistributedVector& other);
    void doubleTransposeProduct(DummyDistributedVector& second_ref, double& out1, double& out2, const DummyDistributedVector& other1, const DummyDistributedVector& other2);
    void IdoubleTransposeProduct(DummyDistributedVector& second_ref, double& out1, double& out2, const DummyDistributedVector& other1, const DummyDistributedVector& other2, MPI_Request& req);
    void itransposeProduct(double& out, const DummyDistributedVector& other, MPI_Request& req);
    void print() const;
};
