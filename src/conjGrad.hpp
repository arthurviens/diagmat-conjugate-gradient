#pragma once
#include <iostream>
#include <iomanip>
#include <mpi.h>
#include <vector>
#include "DummyDistributedVector.hpp"
#include "DistributedDiagonalMatrix.hpp"

extern bool debug;

DummyDistributedVector CG(
    int rank,
    const DistributedDiagonalMatrix &A,
    const DummyDistributedVector &b,
    double rtol=1e-6, int maxiter=1000);

DummyDistributedVector ImprovedCG(
  int rank,
  const DistributedDiagonalMatrix &A,
  const DummyDistributedVector &b,
  double rtol=1e-6, int maxiter=1000);

DummyDistributedVector ChronopoulosGearCG(
    int rank,
    const DistributedDiagonalMatrix &A,
    const DummyDistributedVector &b,
    double rtol=1e-6, int maxiter=1000);

DummyDistributedVector Preconditionned_ChronopoulosGearCG(
    int rank,
    const DistributedDiagonalMatrix &A,
    const DistributedDiagonalMatrix &M,
    const DummyDistributedVector &b,
    double rtol=1e-6, int maxiter=1000);
