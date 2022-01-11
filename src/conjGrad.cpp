
#include <iostream>
#include <iomanip>
#include <mpi.h>
#include <vector>
#include "DummyDistributedVector.hpp"
#include "DistributedDiagonalMatrix.hpp"
#include "conjGrad.hpp"


bool debug = false;

DummyDistributedVector GhyselsVanrooseCG(
    int rank,
    const DistributedDiagonalMatrix &A,
    const DummyDistributedVector &b,
    double rtol, int maxiter
    )
    
{
    // Allocation
    double alpha, gamma, delta, beta, prev_gamma;
    DummyDistributedVector r(b);
    DummyDistributedVector x(b); x.data.setZero();
    DummyDistributedVector q(r); q.data.setZero();
    DummyDistributedVector v(r); v.data.setZero();
    DummyDistributedVector u(r); u.data.setZero();
    DummyDistributedVector z(r); z.data.setZero();
    DummyDistributedVector w(r); w.data.setZero();
    DummyDistributedVector p(r); p.data.setZero();
    DummyDistributedVector s(r); s.data.setZero();
    DummyDistributedVector m(r); m.data.setZero();

    // Initialization
    int iter=0;
    M.product(u, r);   
    r.transposeProduct(nr0, r);
    A.product(w,u);
    MPI_Request req;
    MPI_Status status;

    if(rank==0) {
       std::cout<<"Start Preconditionned Chronopoulos CG"<<std::endl;
       std::cout<<"Initial residual: "<< sqrt(nr0) <<std::endl;
       std::cout<<"Iteration,  Absolute residual,  Relative residual"<<std::endl;
    }
    
    do {
        gamma = r.ltransposeProduct(u);
        delta = w.ltransposeProduct(u);
        M.product(u, w);
        A.product(n,m);
        if (iter > 0) {
            beta = gamma / prev_gamma;
            alpha = gamma / (delta - beta * gamma / alpha )
            } else {
                beta = 0;
                alpha = gamma / delta;
                }
        }
        z *= beta; z += n;
        q *= beta; q += m;
        p *= beta; p += u;
        s *= beta; s += w;
        x.axpy(alpha, p);
        r.axpy(-alpha, s);
        u.axpy(-alpha, q);
        w.axpy(-alpha, z);
        iter++;

      } while((sqrt(nr/nr0)>rtol) && (iter<maxiter));
        if(rank==0) {
            if( sqrt(nr/nr0) < rtol){
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
    const DistributedDiagonalMatrix &A,
    const DummyDistributedVector &b,
    double rtol, int maxiter)
{
  // Allocation
  double alpha, gamma, delta, beta, prev_gamma;
  double nr, nr0;
  DummyDistributedVector r(b);
  DummyDistributedVector x(b); x.data.setZero();
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




