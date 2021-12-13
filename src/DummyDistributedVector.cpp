#include <iostream>
#include <iomanip>
#include <thread>         // std::this_thread::sleep_for
#include <chrono>         // std::chrono::seconds
#include <memory>
#include <mpi.h>
#include <vector>
#include "Eigen/Dense"
#include "DummyDistributedVector.hpp"
/**
 * @todo write docs
 */
static bool debug = false;

void Dummy_MPI_Allreduce(const void *sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
{
    int comm_sz;
    MPI_Comm_size(comm, &comm_sz);
    std::this_thread::sleep_for (std::chrono::microseconds(25*comm_sz));
    MPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
}

void Dummy_MPI_Iallreduce(const void *sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, MPI_Request& req)
{
    int comm_sz;
    MPI_Comm_size(comm, &comm_sz);
    std::this_thread::sleep_for (std::chrono::microseconds(25*comm_sz));
    MPI_Iallreduce(sendbuf, recvbuf, count, datatype, op, comm, &req);
}

void DummyDistributedVector::_start_of_binary_op(const DummyDistributedVector& other)
{
    assert(data.rows() == other.rows());
}


DummyDistributedVector::DummyDistributedVector(MPI_Comm& comm, int local_sz)
{
    _comm = &comm;
    MPI_Comm_rank(*_comm, &_rank);
    MPI_Comm_size(*_comm, &_comm_sz);
    _local_sz = local_sz;
    data.resize(_local_sz); data.setZero();
}

unsigned int DummyDistributedVector::rows() const { return data.rows(); }

bool DummyDistributedVector::operator==(const DummyDistributedVector& other) const
{
    return (data == other.data);
}

bool DummyDistributedVector::operator!=(const DummyDistributedVector& other) const
{
    return (data != other.data);
}

DummyDistributedVector& DummyDistributedVector::operator+=(const DummyDistributedVector& other)
{
  _start_of_binary_op(other);
  data.noalias() += other.data;
  return (*this);
}

DummyDistributedVector& DummyDistributedVector::operator-=(const DummyDistributedVector& other)
{
  _start_of_binary_op(other);
  data.noalias() -= other.data;
  return (*this);
}

DummyDistributedVector& DummyDistributedVector::axpy(double alpha, DummyDistributedVector& other)
{
  _start_of_binary_op(other);
  data.noalias() += alpha * other.data;
  return (*this);
}

DummyDistributedVector& DummyDistributedVector::operator*=(double alpha)
{
  data *= alpha;
  return (*this);
}

double DummyDistributedVector::ltransposeProduct(const DummyDistributedVector& other)
{
  return (data.transpose() * other.data);
}

void DummyDistributedVector::transposeProduct(double& out, const DummyDistributedVector& other)
{
  out = ltransposeProduct(other);
  Dummy_MPI_Allreduce(MPI_IN_PLACE, &out, 1, MPI_DOUBLE, MPI_SUM, *_comm);
}

void DummyDistributedVector::doubleTransposeProduct(DummyDistributedVector& second_ref, double& out1, double& out2, const DummyDistributedVector& other1, const DummyDistributedVector& other2)
{
  std::vector<double> double_result(2);
  double_result[0] = ltransposeProduct(other1);
  double_result[1] = second_ref.ltransposeProduct(other2);
  Dummy_MPI_Allreduce(MPI_IN_PLACE, double_result.data(), 2, MPI_DOUBLE, MPI_SUM, *_comm);
  out1 = double_result[0];
  out2 = double_result[1];
}

void DummyDistributedVector::IdoubleTransposeProduct(DummyDistributedVector& second_ref, double& out1, double& out2, const DummyDistributedVector& other1, const DummyDistributedVector& other2, MPI_Request& req)
{
  std::vector<double> double_result(2);
  double_result[0] = ltransposeProduct(other1);
  double_result[1] = second_ref.ltransposeProduct(other2);
  Dummy_MPI_Iallreduce(MPI_IN_PLACE, double_result.data(), 2, MPI_DOUBLE, MPI_SUM, *_comm, req);
  out1 = double_result[0];
  out2 = double_result[1];
}

void DummyDistributedVector::itransposeProduct(double& out, const DummyDistributedVector& other, MPI_Request& req)
{
  out = ltransposeProduct(other);
  MPI_Iallreduce(MPI_IN_PLACE, &out, 1, MPI_DOUBLE, MPI_SUM, *_comm, &req);
}
