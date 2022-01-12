#include <iostream>
#include <iomanip>
#include <thread>         // std::this_thread::sleep_for
#include <chrono>         // std::chrono::seconds
#include <memory>
#include <mpi.h>
#include <vector>
#include "Eigen/Dense"
#include "DummyDistributedVector.hpp"
#include "DistributedDiagonalMatrix.hpp"
#include "conjGrad.hpp"
/**
 * @todo write docs
 */
extern bool debug;

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
          printf("                        3 - GhyselsVanroose - CG\n");
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
    DistributedDiagonalMatrix M(comm, local_sz);
    M.data.setLinSpaced(local_sz, 1.0, (double) 1.0);
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
      else if (solverID == 3) x = GhyselsVanrooseCG(rank, A, b, rtol, maxiter);
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
