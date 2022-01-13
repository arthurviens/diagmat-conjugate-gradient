
#include <iostream>
#include <iomanip>
#include <mpi.h>
#include <vector>
#include "DummyDistributedVector.hpp"
#include "DistributedDiagonalMatrix.hpp"
#include "conjGrad.hpp"


bool debug = true;

DummyDistributedVector GhyselsVanrooseCG(
    int rank,
    const DistributedDiagonalMatrix &A,
    const DistributedDiagonalMatrix &M,
    const DummyDistributedVector &b,
    double rtol, int maxiter
    )
    {
    // Allocation
    double alpha, gamma, delta, beta, prev_gamma;
    DummyDistributedVector r(b);
    DummyDistributedVector x(b); x.data.setZero();
    DummyDistributedVector q(r); q.data.setZero();
    DummyDistributedVector n(r); n.data.setZero();
    DummyDistributedVector u(r); u.data.setZero();
    DummyDistributedVector z(r); z.data.setZero();
    DummyDistributedVector w(r); w.data.setZero();
    DummyDistributedVector p(r); p.data.setZero();
    DummyDistributedVector s(r); s.data.setZero();
    DummyDistributedVector m(r); m.data.setZero();

    // Initialization
    double nr, nr0;
    int iter=0;
    r.transposeProduct(nr0, r);
    nr = nr0;
    M.product(u, r);
    if (debug) {
      std::cout << "Initialisation : u = "; u.print();
    }
    A.product(w,u);

    if (debug) {
      std::cout << "Initialisation : w = "; w.print();
    }

    if(rank==0) {
       std::cout<<"Start Preconditionned Chronopoulos CG"<<std::endl;
       std::cout<<"Initial residual: "<< sqrt(nr0) <<std::endl;
       std::cout<<"Iteration,  Absolute residual,  Relative residual"<<std::endl;
    }

    MPI_Request req;
    MPI_Status status;

    std::vector < double > values(2);
    do {
        std::cout << "iter " << iter << std::endl;
        std::cout << "r " << r.data << std::endl;
        std::cout << "u " << u.data << std::endl;
        values[0] = r.ltransposeProduct(u);
        values[1] = w.ltransposeProduct(u);
        Dummy_MPI_Iallreduce(MPI_IN_PLACE, values.data(), 2, MPI_DOUBLE, MPI_SUM, ( * A._comm), req);
        gamma = values[0];
        delta = values[1];



        M.product(m, w);
        std::cout << "iter = " << iter << " on a m = "; m.print();
        A.product(n, m);

        MPI_Wait( & req, & status);

        if (debug) {
          std::cout << "Gamma " << gamma << " | Delta " << delta << std::endl;
        }
        if (iter > 0) {
            beta = gamma / prev_gamma;
            /*std::cout << "Dividing " << gamma << " by " << prev_gamma
            << " gives " << beta << std::endl;
            std::cout << "Dividing " << gamma << " by " << (delta - beta * gamma / alpha )
            << " gives " << gamma / (delta - beta * gamma / alpha ) << std::endl;
            std::cout << "Delta " << delta << std::endl;
            std::cout << "Beta " << beta << std::endl;
            std::cout << "Gamma " << gamma << std::endl;
            std::cout << "Alpha " << alpha << std::endl;*/
            alpha = gamma / (delta - beta * gamma / alpha );
            //std::cout << "juste after " << alpha << std::endl;
        } else {
            //std::cout << "iter = 0 and gamma " << gamma << " and delta "<< delta << std::endl;
            beta = 0;
            alpha = gamma / delta;
        }

        z *= beta; z += n;
        std::cout << "WE HAVE beta = " << beta << " axpy m = "; m.print();
        std::cout << "BEFORE q = "; q.print();
        q *= beta; q += m;
        std::cout << "AFTER q = "; q.print();
        s *= beta; s += w;
        p *= beta; p += u;
        x.axpy(alpha, p);
        if (debug) {
          std::cout << "-alpha " << -alpha << std::endl;
        }
        r.axpy(-alpha, s);
        std::cout << "before u "; u.print();
        u.axpy(-alpha, q);
        std::cout << "after u "; u.print();
        w.axpy(-alpha, z);

        prev_gamma = gamma;


        if ((rank == 0)) {
          std::cout << std::setfill(' ') << std::setw(8);
          std::cout << iter << "/" << maxiter << "        ";
          std::cout << std::scientific << sqrt(nr) << "        " << sqrt(nr / nr0) << std::endl;
        }


        r.transposeProduct(nr, r);
        iter++;

      } while((sqrt(nr/nr0)>rtol) && (iter<maxiter));

        if(rank==0) {
            if( sqrt(nr/nr0) < rtol)
            {
            std::cout<<"Converged solution"<<std::endl;
            }
            else {
            std::cout<<"Not converged solution"<<std::endl;
            }
        }
        return x;
        }



DummyDistributedVector ChronopoulosGearCG(
  int rank,
  const DistributedDiagonalMatrix & A,
    const DummyDistributedVector & b,
      double rtol, int maxiter) {

  // Allocation
  double alpha, gamma, delta, beta, prev_gamma;
  double nr, nr0;
  DummyDistributedVector r(b);
  DummyDistributedVector x(b);x.data.setZero();
  DummyDistributedVector q(b);q.data.setZero();
  DummyDistributedVector v(r);v.data.setZero();
  DummyDistributedVector u(r);u.data.setZero();
  DummyDistributedVector z(r);z.data.setZero();
  DummyDistributedVector w(r);w.data.setZero();

  // Initialization
  int iter = 0;
  r.transposeProduct(nr0, r);
  nr = nr0;
  if (rank == 0) {
    std::cout << "Start Chronopoulos CG" << std::endl;
    std::cout << "    Initial residual: " << sqrt(nr0) << std::endl;
    std::cout << "    Iteration,  Absolute residual,  Relative residual" << std::endl;
  }

  A.product(v, r);
  MPI_Request req;
  MPI_Status status;

  // CG-Loop
  std::vector < double > values(2);
  do {
    values[0] = r.ltransposeProduct(r);
    values[1] = r.ltransposeProduct(v);
    Dummy_MPI_Iallreduce(MPI_IN_PLACE, values.data(), 2, MPI_DOUBLE, MPI_SUM, ( * A._comm), req);
    gamma = values[0];
    delta = values[1];

    A.product(u, v);
    MPI_Wait( & req, & status);

    if (iter > 0) {
      beta = gamma / prev_gamma;
      alpha = gamma / (delta - beta * gamma / alpha);
    } else {
      beta = 0;
      alpha = gamma / delta;
    }

    if (debug && rank == 0) {
      std::cout << "gamma, delta: " << gamma << ", " << delta << std::endl;
    }

    z *= beta;
    z += u;
    q *= beta;
    q += v;
    w *= beta;
    w += r;

    x.axpy(alpha, w);
    r.axpy(-alpha, q);
    v.axpy(-alpha, z);

    if ((debug) && (rank == 0) && (iter % 100 == 0)) {
      std::cout << std::setfill(' ') << std::setw(8);
      std::cout << iter << "/" << maxiter << "        ";
      std::cout << std::scientific << sqrt(nr) << "        " << sqrt(nr / nr0) << std::endl;
    }

    prev_gamma = gamma;

    r.transposeProduct(nr, r);
    iter++;
  } while ((sqrt(nr / nr0) > rtol) && (iter < maxiter));

  if (rank == 0) {
    if (sqrt(nr / nr0) < rtol) {
      std::cout << "Converged solution" << std::endl;
    } else {
      std::cout << "Not converged solution" << std::endl;
    }
  }

  return(x);
}

DummyDistributedVector CG(
  int rank,
  const DistributedDiagonalMatrix & A,
    const DummyDistributedVector & b,
      double rtol, int maxiter) {

  // Allocation
  double alpha, gamma, delta;
  double nr, nr0;
  DummyDistributedVector r(b);
  DummyDistributedVector x(b);
  x.data.setZero();
  DummyDistributedVector q(r);
  q.data.setZero();

  // Initialization
  int iter = 0;
  r.transposeProduct(nr0, r);
  nr = nr0;
  DummyDistributedVector w(r);
  if (rank == 0) {
    std::cout << "Start CG" << std::endl;
    std::cout << "    Initial residual: " << sqrt(nr0) << std::endl;
    std::cout << "    Iteration,  Absolute residual,  Relative residual" << std::endl;
  }

  // CG-Loop
  do {
    A.product(q, w);
    r.transposeProduct(gamma, r);
    w.transposeProduct(delta, q);
    alpha = gamma / delta;

    if (debug && rank == 0) {
      std::cout << "gamma, delta: " << gamma << ", " << delta << std::endl;
    }

    x.axpy(alpha, w);
    r.axpy(-alpha, q);

    r.transposeProduct(nr, r);
    w *= (nr / gamma);
    w += r;

    iter++;
    if ((rank == 0) && (iter % 10 == 0)) {
      std::cout << std::setfill(' ') << std::setw(8);
      std::cout << iter << "/" << maxiter << "        ";
      std::cout << std::scientific << sqrt(nr) << "        " << sqrt(nr / nr0) << std::endl;
    }

  } while ((sqrt(nr / nr0) > rtol) && (iter < maxiter));

  if (rank == 0) {
    if (sqrt(nr / nr0) < rtol) {
      std::cout << "Converged solution" << std::endl;
    } else {
      std::cout << "Not converged solution" << std::endl;
    }
  }

  return(x);
}

DummyDistributedVector ImprovedCG(
  int rank,
  const DistributedDiagonalMatrix & A,
    const DummyDistributedVector & b,
      double rtol, int maxiter) {

  // Allocation
  double alpha, gamma, delta;
  double nr, nr0;
  DummyDistributedVector r(b);
  DummyDistributedVector x(b);
  x.data.setZero();
  DummyDistributedVector q(r);
  q.data.setZero();

  // Initialization
  int iter = 0;
  r.transposeProduct(nr0, r);
  nr = nr0;
  DummyDistributedVector w(r);
  if (rank == 0) {
    std::cout << "Start ImprovedCG" << std::endl;
    std::cout << "    Initial residual: " << sqrt(nr0) << std::endl;
    std::cout << "    Iteration,  Absolute residual,  Relative residual" << std::endl;
  }

  // CG-Loop
  do {
    A.product(q, w);
    r.doubleTransposeProduct(w, gamma, delta, r, q);
    alpha = gamma / delta;

    if (debug && rank == 0) {
      std::cout << "gamma, delta: " << gamma << ", " << delta << std::endl;
    }

    x.axpy(alpha, w);
    r.axpy(-alpha, q);

    r.transposeProduct(nr, r);
    w *= (nr / gamma);
    w += r;

    iter++;
    if (debug && (rank == 0) && (iter % 10 == 0)) {
      std::cout << std::setfill(' ') << std::setw(8);
      std::cout << iter << "/" << maxiter << "        ";
      std::cout << std::scientific << sqrt(nr) << "        " << sqrt(nr / nr0) << std::endl;
    }

  } while ((sqrt(nr / nr0) > rtol) && (iter < maxiter));

  if (rank == 0) {
    if (sqrt(nr / nr0) < rtol) {
      std::cout << "Converged solution" << std::endl;
    } else {
      std::cout << "Not converged solution" << std::endl;
    }
  }

  return(x);
}
