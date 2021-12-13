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


// This function is used to simulate a big network latency in order to be able to
// see the benefit of communication avoiding with a small number of cores

class DistributedDiagonalMatrix
{
public: /* should be private:*/
    int _local_sz;
    int _rank;
    int _comm_sz;
    MPI_Comm* _comm;

    void _start_of_binary_op(const DummyDistributedVector& other)
    {
        assert(data.rows() == other.rows());
    }


public:
    Eigen::VectorXd data;

    DistributedDiagonalMatrix(MPI_Comm& comm, int local_sz)
    {
        _comm = &comm;
        MPI_Comm_rank(*_comm, &_rank);
        MPI_Comm_size(*_comm, &_comm_sz);
        _local_sz = local_sz;
        data.resize(_local_sz); data.setZero();
    }

    void inplaceProduct(DummyDistributedVector& other) const
    {
        other.data.cwiseProduct(data);
    }

    void product(DummyDistributedVector& out, const DummyDistributedVector & in) const
    {
        out.data = data.asDiagonal() * in.data;
    }
};

DummyDistributedVector CG(
    int rank,
    const DistributedDiagonalMatrix &A,
    const DummyDistributedVector &b,
    double rtol=1e-6, int maxiter=1000)
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
  double rtol=1e-6, int maxiter=1000)
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
    double rtol=1e-6, int maxiter=1000)
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

int main (int argc, char *argv[])
{
    int rank, comm_sz;
    int local_sz = 10;
    int solverID = 0;
    int maxiter = 1000;
    double rtol = 1.0e-6;
    int rep=1;

    /* Initialize MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm comm;
    MPI_Comm_dup(MPI_COMM_WORLD, &comm);
    MPI_Comm_size(comm, &comm_sz);
    MPI_Comm_rank(comm, &rank);

    if (comm_sz <= 0)
    {
       if (rank == 0) std::cout << "The number of processors must be positive!" << std::endl;
       MPI_Finalize();
       return(0);
    }

    /* Parse command line */
    {
       int arg_index = 0;
       int print_usage = 0;

       while (arg_index < argc)
       {
	  if ( strcmp(argv[arg_index], "-sz") == 0 )
          {
             arg_index++;
             local_sz = std::atoi(argv[arg_index++]);
          }
          else if ( strcmp(argv[arg_index], "-solver") == 0 )
          {
             arg_index++;
             solverID = atoi(argv[arg_index++]);
          }
          else if ( strcmp(argv[arg_index], "-maxiter") == 0 )
          {
             arg_index++;
             maxiter = atoi(argv[arg_index++]);
          }
          else if ( strcmp(argv[arg_index], "-rtol") == 0 )
          {
             arg_index++;
             rtol = atof(argv[arg_index++]);
          }
          else if ( strcmp(argv[arg_index], "-r") == 0 )
          {
             arg_index++;
             rep = atoi(argv[arg_index++]);
          }
          else if ( strcmp(argv[arg_index], "-g") == 0 )
          {
             arg_index++;
             debug = true;
          }
          else if ( strcmp(argv[arg_index], "-help") == 0 )
          {
             print_usage = 1;
             break;
          }
          else
          {
             arg_index++;
          }
       }

       if ((print_usage) && (rank == 0))
       {
          printf("\n");
          printf("Usage: %s [<options>]\n", argv[0]);
          printf("\n");
          printf("  -sz <n>              : problem size per processor  (default: %d)\n", 10);
          printf("  -maxiter <n>         : maximum number of iteration (default: %d)\n", 1000);
          printf("  -rep     <n>         : number of repetitions (default: %d)\n", 1);
          printf("  -rtol    <f>         : relative tolerance (default: %f)\n", 1.0e-6);
          printf("  -solver <ID>         : solver ID\n");
          printf("                        0 - CG (default)\n");
          printf("                        1 - ImprovedCG\n");
          printf("                        2 - Chronopoulos Gear-CG\n");
          printf("\n");
       }

       if (print_usage)
       {
          MPI_Finalize();
          return (0);
       }
    }




    if (rank == 0) {
       std::cout<<"Starting computation"<<std::endl;
       std::cout << "Number of MPI processes: " << comm_sz << std::endl;
       std::cout << "Local size: " << local_sz << std::endl;
       std::cout << "Solver id: " << solverID << std::endl;
    }


    // Setup of the matrix and rhs
    DistributedDiagonalMatrix A(comm, local_sz);
    A.data.setLinSpaced(local_sz, 1.0, (double) local_sz);
    // A.data.array().pow(k);
    DummyDistributedVector b(comm, local_sz);
    b.data.setOnes();
    // b.data.setRandom();

    DummyDistributedVector x(comm, local_sz);

    double starttime, endtime;
    starttime = MPI_Wtime();

    for(int irep=0; irep<rep; irep++) {
    	if(solverID == 0) x = CG(rank, A, b, rtol, maxiter);
    	else if (solverID == 1) x = ImprovedCG(rank, A, b, rtol, maxiter);
    	else if (solverID == 2) x = ChronopoulosGearCG(rank, A, b, rtol, maxiter);
    	else {
    	  printf("Unknown solver\n");
    	  return(1);
    	}
    }
    endtime = MPI_Wtime();
    if(rank == 0) printf("That took %f seconds\n",endtime-starttime);

    // Just to check solution
    DummyDistributedVector tmp(x);
    A.product(tmp, x);
    tmp -= b;
    double nr;
    tmp.transposeProduct(nr, tmp);
    if(rank==0) std::cout << "Real residual " << sqrt(nr) << std::endl;

    // To print the solution
    // if(rank==0) std::cout << x.data << std::endl;

    if(rank == 0) std::cout<<"Finish computation"<<std::endl;
    MPI_Finalize();
    return 0;
}
