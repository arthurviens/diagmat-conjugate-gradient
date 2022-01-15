#include <iostream>
#include <iomanip>
#include <mpi.h>
#include <vector>
#include "Eigen/Dense"
#include "DistributedMatrix.hpp"
#include "DummyDistributedVector.hpp"


class DistributedTriangularMatrix : public DistributedMatrix{
public:
  DistributedTriangularMatrix(MPI_Comm& comm, int local_sz);
  void initFromMatrix(Eigen::MatrixXd A);

  void inplaceProduct(DummyDistributedVector& other) const override;
  void product(DummyDistributedVector& out, const DummyDistributedVector & in) const override;
  Eigen::MatrixXd plainMatrix() const override;
  void print(std::string display_type) const override;
  void inv() override;
};
