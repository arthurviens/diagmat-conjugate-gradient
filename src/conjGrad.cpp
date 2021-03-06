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
    const DistributedMatrix* A,
    const DummyDistributedVector &b,
    double rtol, int maxiter)
{
    // Allocation
    double alpha, gamma, delta;
    double nr, nr0;
    //double ntest;
    DummyDistributedVector r(b);
    DummyDistributedVector rtest(b);
    DummyDistributedVector x(b); x.data.setZero();
    DummyDistributedVector q(r); q.data.setZero();
    DummyDistributedVector w(r);

    // Initialization
    int iter=0;
    r.transposeProduct(nr0, r); //
    nr = nr0;
    if((rank==0) && debug) {
      std::cout<<"Start CG"<<std::endl;
      std::cout<<"    Initial residual: "<< sqrt(nr0) <<std::endl;
      std::cout<<"    Iteration,  Absolute residual,  Relative residual"<<std::endl;
    }

    // CG-Loop
    do {
        A->product(q, w);
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
        w += r;

        if(debug && (rank==0)) {
      	  std::cout<< std::setfill(' ') << std::setw(8);
      	  std::cout<< iter << "/" << maxiter << "        ";
      	  std::cout << std::scientific << sqrt(nr) << "        " << sqrt(nr/nr0) << std::endl;
      	}

        iter++;

    } while ((sqrt(nr/nr0)>rtol) && (iter<maxiter));

    if((rank==0)) {
      if(sqrt(nr/nr0)<rtol) {
	       std::cout << "Converged solution with last residual = " << sqrt(nr) << std::endl;
      } else {
	       std::cout<<"Not converged solution"<<std::endl;
      }
    }
    if (rank == 0) {
      std::cout << "Number of iterations : " << iter << std::endl;
    }

    return x;
}

DummyDistributedVector ImprovedCG(
  int rank,
  const DistributedMatrix *A,
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
  if((rank==0) && debug) {
    std::cout<<"Start ImprovedCG"<<std::endl;
    std::cout<<"    Initial residual: "<< sqrt(nr0) <<std::endl;
    std::cout<<"    Iteration,  Absolute residual,  Relative residual"<<std::endl;
  }

  // CG-Loop
  do {
      A->product(q, w);
      r.doubleTransposeProduct(w, gamma, delta, r, q);
      alpha = gamma/delta;

      if(debug && rank==0) {
        std::cout<<"gamma, delta: "<< gamma << ", " << delta << std::endl;
      }

      x.axpy(alpha, w);
      r.axpy(-alpha, q);


      if(debug && (rank==0)) {
        std::cout<< std::setfill(' ') << std::setw(8);
        std::cout<< iter << "/" << maxiter << "        ";
        std::cout << std::scientific << sqrt(nr) << "        " << sqrt(nr/nr0) << std::endl;
      }

      r.transposeProduct(nr, r);
      w *= (nr/gamma);
      w += r;

      iter++;

  } while ((sqrt(nr/nr0)>rtol) && (iter<maxiter));

  if(rank==0) {
    if(sqrt(nr/nr0)<rtol) {
  std::cout<<"Converged solution"<<std::endl;
    } else {
  std::cout<<"Not converged solution"<<std::endl;
    }
  }
  if (rank == 0) {
    std::cout << "Number of iterations : " << iter << std::endl;
  }

  return x;
}


DummyDistributedVector ChronopoulosGearCG(
    int rank,
    const DistributedMatrix *A,
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

  if((rank==0) && debug) {
    std::cout<<"Start Chronopoulos CG"<<std::endl;
    std::cout<<"    Initial residual: "<< sqrt(nr0) <<std::endl;
    std::cout<<"    Iteration,  Absolute residual,  Relative residual"<<std::endl;
  }

  A->product(v, r);
  MPI_Request req;
  MPI_Status status;

  // CG-Loop
  std::vector<double> values(2);
  do {
    values[0] = r.ltransposeProduct(r);
    values[1] = r.ltransposeProduct(v);
    Dummy_MPI_Iallreduce(MPI_IN_PLACE, values.data(), 2, MPI_DOUBLE, MPI_SUM, (*A->m_comm), req);
    gamma = values[0];
    delta = values[1];

    A->product(u, v);

    MPI_Wait(&req, &status);

    if (iter > 0) {
      beta = gamma / prev_gamma;
      alpha = gamma / (delta - beta * gamma / alpha );
    } else {
      beta = 0;
      alpha = gamma / delta;
    }

    if(debug && (rank==0)) {
      std::cout<<"gamma, delta: "<< gamma << ", " << delta << std::endl;
    }

    z *= beta; z += u;
    q *= beta; q += v;
    w *= beta; w += r;

    x.axpy(alpha, w);
    r.axpy(-alpha, q);
    v.axpy(-alpha, z);

    if(debug && (rank==0)) {
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
  if (rank == 0) {
    std::cout << "Number of iterations : " << iter << std::endl;
  }

  return x;
}

DummyDistributedVector Preconditionned_ChronopoulosGearCG(
    int rank,
    DistributedMatrix* A,
    const DistributedMatrix* M,
    const DummyDistributedVector &b,
    double rtol, int maxiter)
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
      DummyDistributedVector p(r); p.data.setZero();
      DummyDistributedVector s(r); s.data.setZero();


      // Initialization
      int iter=0;
      M->product(u, r); //u0=M^(-1) r0
      r.transposeProduct(nr0, r);
      nr = nr0;

      A->product(w,u); // w0 = A U0
      alpha=r.ltransposeProduct(u)/w.ltransposeProduct(u); // alpha0:=(r0,u0)/(w0,u0)
      beta=0;
      gamma=r.ltransposeProduct(u); // gamma0 = (r0,u0)
      prev_gamma = gamma;
      if((rank==0) && debug) {
        std::cout<<"Start Preconditionned Chronopoulos CG"<<std::endl;
        std::cout<<"    Initial residual: "<< sqrt(nr0) <<std::endl;
        std::cout<<"    Iteration,  Absolute residual,  Relative residual"<<std::endl;
      }

      MPI_Request req;
      MPI_Status status;


        // CG-Loop
        std::vector<double> values(2);
      do {

      p *= beta; p += u;
      s *= beta; s += w;
      x.axpy(alpha, p);
      r.axpy(-alpha, s);
      M->product(u, r); // ui+1 = M^-1 ri+1
      A->product(w,u); // ligne 9

      values[0] = r.ltransposeProduct(u);
      values[1] = w.ltransposeProduct(u);
      Dummy_MPI_Iallreduce(MPI_IN_PLACE, values.data(), 2, MPI_DOUBLE, MPI_SUM, (*A->m_comm), req);
      gamma = values[0];
      delta = values[1];

      MPI_Wait(&req, &status);

      beta = gamma/prev_gamma; //logne 12
      prev_gamma = gamma;
      alpha = gamma /( delta - beta * gamma / alpha);

      r.transposeProduct(nr, r);
      iter++;

      if(debug && (rank==0)) {
        std::cout<< std::setfill(' ') << std::setw(8);
        std::cout<< iter << "/" << maxiter << "        ";
        std::cout << std::scientific << sqrt(nr) << "        " << sqrt(nr/nr0) << std::endl;
      }


      } while((sqrt(nr/nr0)>rtol) && (iter<maxiter));

      if(rank==0) {
        if(sqrt(nr/nr0)<rtol) {
      std::cout<<"Converged solution"<<std::endl;
        } else {
      std::cout<<"Not converged solution"<<std::endl;
        }
      }
      if (rank == 0) {
        std::cout << "Number of iterations : " << iter << std::endl;
      }
      return x;
}



DummyDistributedVector GhyselsVanrooseCG(
    int rank,
    DistributedMatrix *A,
    const DistributedMatrix* M,
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
    alpha = 0; prev_gamma = 0; // Silence warnings
    double nr, nr0;
    int iter=0;
    r.transposeProduct(nr0, r);
    nr = nr0;
    M->product(u, r);
    A->product(w,u);

    if((rank==0) && debug) {
       std::cout<<"Start Preconditionned Chronopoulos CG"<<std::endl;
       std::cout<<"Initial residual: "<< sqrt(nr0) <<std::endl;
       std::cout<<"Iteration,  Absolute residual,  Relative residual"<<std::endl;
    }

    MPI_Request req;
    MPI_Status status;

    std::vector < double > values(2);
    do {
        values[0] = r.ltransposeProduct(u);
        values[1] = w.ltransposeProduct(u);
        Dummy_MPI_Iallreduce(MPI_IN_PLACE, values.data(), 2, MPI_DOUBLE, MPI_SUM, ( *A->m_comm), req);
        gamma = values[0];
        delta = values[1];

        M->product(m, w);
        A->product(n, m);

        MPI_Wait( & req, & status);

        if (iter > 0) {
            beta = gamma / prev_gamma;
            alpha = gamma / (delta - beta * gamma / alpha );
        } else {
            beta = 0;
            alpha = gamma / delta;
        }

        z *= beta; z += n;
        q *= beta; q += m;
        s *= beta; s += w;
        p *= beta; p += u;
        x.axpy(alpha, p);
        r.axpy(-alpha, s);
        u.axpy(-alpha, q);
        w.axpy(-alpha, z);

        prev_gamma = gamma;


        if (debug && (rank==0)) {
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
        if (rank == 0) {
          std::cout << "Number of iterations : " << iter << std::endl;
        }

        return x;
}
