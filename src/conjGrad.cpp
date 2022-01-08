#include <iostream>
#include <iomanip>
#include <mpi.h>
#include <vector>
#include "DummyDistributedVector.hpp"
#include "DistributedDiagonalMatrix.hpp"
#include "conjGrad.hpp"


bool debug = false;

DummyDistributedVector CG(
    int rank,
    const DistributedDiagonalMatrix &A,
    const DummyDistributedVector &b,
    double rtol, int maxiter)
{
    // Allocation
    double alpha, gamma, delta;
    double nr, nr0;
    DummyDistributedVector r(b);
    DummyDistributedVector x(b); x.data.setZero();
    DummyDistributedVector q(r); q.data.setZero();

    // Initialization
    int iter=0;
    r.transposeProduct(nr0, r);
    nr = nr0;
    DummyDistributedVector w(r);
    if(rank==0) {
      std::cout<<"Start CG"<<std::endl;
      std::cout<<"    Initial residual: "<< sqrt(nr0) <<std::endl;
      std::cout<<"    Iteration,  Absolute residual,  Relative residual"<<std::endl;
    }

    // CG-Loop
    do {
        A.product(q, w);
        r.transposeProduct(gamma, r);
        w.transposeProduct(delta, q);
        alpha = gamma/delta;

	if(debug && rank==0) {
	  std::cout<<"gamma, delta: "<< gamma << ", " << delta << std::endl;
	}

        x.axpy(alpha, w);
        r.axpy(-alpha, q);


        r.transposeProduct(nr, r);
        w *= (nr/gamma);
        w +=r;

        iter++;
	if((rank==0) && (iter%10 == 0)) {
	  std::cout<< std::setfill(' ') << std::setw(8);
	  std::cout<< iter << "/" << maxiter << "        ";
	  std::cout << std::scientific << sqrt(nr) << "        " << sqrt(nr/nr0) << std::endl;
	}

    } while ((sqrt(nr/nr0)>rtol) && (iter<maxiter));

    if(rank==0) {
      if(sqrt(nr/nr0)<rtol) {
	  std::cout<<"Converged solution"<<std::endl;
      } else {
	  std::cout<<"Not converged solution"<<std::endl;
      }
    }

    return x;
}

DummyDistributedVector ImprovedCG(
  int rank,
  const DistributedDiagonalMatrix &A,
  const DummyDistributedVector &b,
  double rtol, int maxiter)
{
  // Allocation
  double alpha, gamma, delta;
  double nr, nr0;
  DummyDistributedVector r(b);
  DummyDistributedVector x(b); x.data.setZero();
  DummyDistributedVector q(r); q.data.setZero();

  // Initialization
  int iter=0;
  r.transposeProduct(nr0, r);
  nr = nr0;
  DummyDistributedVector w(r);
  if(rank==0) {
    std::cout<<"Start ImprovedCG"<<std::endl;
    std::cout<<"    Initial residual: "<< sqrt(nr0) <<std::endl;
    std::cout<<"    Iteration,  Absolute residual,  Relative residual"<<std::endl;
  }

  // CG-Loop
  do {
      A.product(q, w);
      r.doubleTransposeProduct(w, gamma, delta, r, q);
      alpha = gamma/delta;

if(debug && rank==0) {
  std::cout<<"gamma, delta: "<< gamma << ", " << delta << std::endl;
}

      x.axpy(alpha, w);
      r.axpy(-alpha, q);


      r.transposeProduct(nr, r);
      w *= (nr/gamma);
      w +=r;

      iter++;
if(debug && (rank==0) && (iter%10 == 0)) {
  std::cout<< std::setfill(' ') << std::setw(8);
  std::cout<< iter << "/" << maxiter << "        ";
  std::cout << std::scientific << sqrt(nr) << "        " << sqrt(nr/nr0) << std::endl;
}

  } while ((sqrt(nr/nr0)>rtol) && (iter<maxiter));

  if(rank==0) {
    if(sqrt(nr/nr0)<rtol) {
  std::cout<<"Converged solution"<<std::endl;
    } else {
  std::cout<<"Not converged solution"<<std::endl;
    }
  }

  return x;
}


DummyDistributedVector ChronopoulosGearCG(
    int rank,
    const DistributedDiagonalMatrix &A,
    const DummyDistributedVector &b,
    double rtol, int maxiter)
{
  // Allocation
  double alpha, gamma, delta, beta, prev_gamma;
  double nr, nr0;
  DummyDistributedVector r(b);
  DummyDistributedVector x(b); x.data.setZero();
  DummyDistributedVector q(r); q.data.setZero();
  DummyDistributedVector v(r); v.data.setZero();
  DummyDistributedVector u(r); u.data.setZero();
  DummyDistributedVector z(r); z.data.setZero();
  DummyDistributedVector w(r); w.data.setZero();
  // Initialization
  int iter=0;
  r.transposeProduct(nr0, r);
  nr = nr0;

  if(rank==0) {
    std::cout<<"Start Chronopoulos CG"<<std::endl;
    std::cout<<"    Initial residual: "<< sqrt(nr0) <<std::endl;
    std::cout<<"    Iteration,  Absolute residual,  Relative residual"<<std::endl;
  }

  A.product(v, r);
  MPI_Request req;
  MPI_Status status;

  // CG-Loop
  std::vector<double> values(2);
  do {
    //A.product(q, w);
    values[0] = r.ltransposeProduct(r);
    values[1] = r.ltransposeProduct(v);
    Dummy_MPI_Iallreduce(MPI_IN_PLACE, values.data(), 2, MPI_DOUBLE, MPI_SUM, (*A._comm), req);
    gamma = values[0];
    delta = values[1];

    A.product(u, v);

    MPI_Wait(&req, &status);

    if (iter > 0) {
      beta = gamma / prev_gamma;
      alpha = gamma / (delta - beta * gamma / alpha );
    } else {
      beta = 0;
      alpha = gamma / delta;
    }

    if(debug && rank==0) {
      std::cout<<"gamma, delta: "<< gamma << ", " << delta << std::endl;
    }

    z *= beta; z += u;
    q *= beta; q += v;
    w *= beta; w += r;

    x.axpy(alpha, w);
    r.axpy(-alpha, q);
    v.axpy(-alpha, z);

    if((debug) && (rank==0) && (iter%100 == 0)) {
      std::cout<< std::setfill(' ') << std::setw(8);
      std::cout<< iter << "/" << maxiter << "        ";
      std::cout << std::scientific << sqrt(nr) << "        " << sqrt(nr/nr0) << std::endl;
    }

    prev_gamma = gamma;

    r.transposeProduct(nr, r);
    iter++;
  } while((sqrt(nr/nr0)>rtol) && (iter<maxiter));


  if(rank==0) {
    if(sqrt(nr/nr0)<rtol) {
  std::cout<<"Converged solution"<<std::endl;
    } else {
  std::cout<<"Not converged solution"<<std::endl;
    }
  }

  return x;
}

DummyDistributedVector Preconditionned_ChronopoulosGearCG(
    int rank,
    const DistributedDiagonalMatrix &A,
    const DummyDistributedVector &b,
    double rtol=1e-6, int maxiter=1000)

    {
      // Allocation
      double alpha, beta, gamma, delta, prev_gamma;
      double nr, nr0;
      DummyDistributedVector r(b);
      DummyDistributedVector x(b); x.data.setZero();
      DummyDistributedVector q(r); q.data.setZero();
      DummyDistributedVector v(r); v.data.setZero();
      DummyDistributedVector u(r); u.data.setZero();
      DummyDistributedVector z(r); z.data.setZero();
      DummyDistributedVector w(r); w.data.setZero();


      // Initialization
      int iter=0;
      r.transposeProduct(nr0, r); //u0=M^(-1) r0
      A.product(w,r); // w0 = A U0
      alpha=r.ltransposeProduct(r)/v.ltransposeProduct(r); // alpha0:=(r0,u0)/(w0,u0)
      beta=0;
      gamma=r.ltransposeProduct(r); // gamma0 = (r0,u0)

      if(rank==0) {
        std::cout<<"Start Preconditionned Chronopoulos CG"<<std::endl;
        std::cout<<"    Initial residual: "<< sqrt(nr0) <<std::endl;
        std::cout<<"    Iteration,  Absolute residual,  Relative residual"<<std::endl;
      }

      MPI_Request req;
      MPI_Status status;

        // CG-Loop
        std::vector<double> values(2);
      do {

        values[0] = r.ltransposeProduct(r);
        values[1] = r.ltransposeProduct(v);
        Dummy_MPI_Iallreduce(MPI_IN_PLACE, values.data(), 2, MPI_DOUBLE, MPI_SUM, (*A._comm), req);
        gamma = values[0];
        delta = values[1];

      w *= beta; w += u;
      q *= beta; q += v;
      x.axpy(alpha, w);
      r.axpy(-alpha, q);
      u.transposeProduct(nr, r); // ui+1 = M^-1 ri+1
      A.product(v,r); // ligne 9
      gamma = r.ltransposeProduct(r); //ligne 10
      delta = v.ltransposeProduct(r); //ligne 11
      beta = gamma/prev_gamma; //logne 12
      prev_gamma = gamma;
      alpha = gamma /( delta - beta * gamma / alpha);


      iter++;



      } while((sqrt(nr/nr0)>rtol) && (iter<maxiter));

      if(rank==0) {
        if(sqrt(nr/nr0)<rtol) {
      std::cout<<"Converged solution"<<std::endl;
        } else {
      std::cout<<"Not converged solution"<<std::endl;
        }
      }

      return x;
      }
