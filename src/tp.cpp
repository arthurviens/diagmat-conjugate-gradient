#include <iostream>
#include <iomanip>
#include <thread>         // std::this_thread::sleep_for
#include <chrono>         // std::chrono::seconds
#include <memory>
#include <mpi.h>
#include <vector>
#include "Eigen/Dense"
#include "Eigen/Core"
#include "Eigen/StdVector"
#include <unsupported/Eigen/SparseExtra>
#include "DummyDistributedVector.hpp"
#include "DistributedMatrix.hpp"
#include "DistributedDiagonalMatrix.hpp"
#include "DistributedTriangularMatrix.hpp"
#include "DistributedBlockDiagonalMatrix.hpp"
#include "DistributedBlockTridiagonalMatrix.hpp"
#include "conjGrad.hpp"
#include "utils.hpp"
#include "preconditionners.hpp"


extern bool debug;


int main (int argc, char *argv[])
{
    int rank, comm_sz;
    int local_sz = 100;
    int solverID = 0;
    int precondID = 0;
    int maxiter = 1000;
    double rtol = 1.0e-6;
    double omega = 1;
    int rep=1;
    int block_size = 10;

    // Initialize MPI

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

    // Parse command line
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
          else if ( strcmp(argv[arg_index], "-precond") == 0 )
          {
             arg_index++;
             precondID = atoi(argv[arg_index++]);
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
          else if ( strcmp(argv[arg_index], "-omega") == 0 )
          {
             arg_index++;
             omega = atof(argv[arg_index++]);
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
          printf("  -omega    <f>        : omega of SSOR precond (default: %f)\n", 1.0);
          printf("  -precond <ID>        : preconditionner ID\n");
          printf("                        0 - Jacobi : Diagonal \n");
          printf("                        1 - Jacobi : Block Diagonal \n");
          printf("                        2 - SSOR \n");
          printf("  -solver <ID>         : solver ID\n");
          printf("                        0 - CG (default)\n");
          printf("                        1 - ImprovedCG\n");
          printf("                        2 - Chronopoulos Gear-CG\n");
          printf("                        3 - Preconditionned Chronopoulos Gear - CG\n");
          printf("                        4 - GhyselsVanroose - CG\n");
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


    DistributedBlockTridiagonalMatrix triblock_B(comm, (local_sz / block_size), block_size);
    DistributedBlockTridiagonalMatrix* triblock_A = &triblock_B;


    if (rank == 0) {
      std::cout << "Beginning matrix reading" << std::endl;
    }
    typedef Eigen::SparseMatrix<double, Eigen::RowMajor>SMatrixXf;
    SMatrixXf Spmat;
    Eigen::loadMarket(Spmat, "nos4.mtx");
    Eigen::MatrixXd Readmat(Spmat); // Read matrix in dense format

    triblock_A->initFromMatrix(Spmat);

    if (rank == 0) {
      std::cout << "Matrix finished reading" << std::endl;
    }

    DistributedMatrix *M;

    DistributedBlockDiagonalMatrix Mb(comm, local_sz/block_size, block_size);
    Mb = triblock_A->extractBlockDiagonal();
    DistributedDiagonalMatrix Mb2(comm, local_sz);
    Mb2 = Jacobi(Readmat);
    DistributedDiagonalMatrix Mb3(comm, local_sz);
    Mb3 = SSOR(Readmat, omega);
    if (precondID == 0) {
      if (rank == 0) {
        std::cout << "Jacobi preconditionner" << std::endl;
      }
      M = &Mb2;
    } else if (precondID == 1) {
      if (rank == 0) {
        std::cout << "Block Jacobi preconditionner" << std::endl;
      }
      M = &Mb;
    } else {
      if (rank == 0) {
        std::cout << "SSOR preconditionner" << std::endl;
      }
      M = &Mb3;
    }



    M->inv();

    DummyDistributedVector b(comm, local_sz);
    //b.data.setOnes();
    b.data.setRandom();
    


    DummyDistributedVector x(comm, local_sz);



    double starttime, endtime;
    starttime = MPI_Wtime();

    if (rank == 0) {
      std::cout << "Beginning computation " << std::endl;
    }

    for(int irep=0; irep<rep; irep++) {
    	if(solverID == 0) x = CG(rank, triblock_A, b, rtol, maxiter);
    	else if (solverID == 1) x = ImprovedCG(rank, triblock_A, b, rtol, maxiter);
    	else if (solverID == 2) x = ChronopoulosGearCG(rank, triblock_A, b, rtol, maxiter);
      else if (solverID == 3) x = Preconditionned_ChronopoulosGearCG(rank, triblock_A, M, b, rtol, maxiter);
      else if (solverID == 4) x = GhyselsVanrooseCG(rank, triblock_A, M, b, rtol, maxiter);
    	else {
    	  printf("Unknown solver\n");
    	  return(1);
    	}
    }
    endtime = MPI_Wtime();
    if(rank == 0) printf("That took %f seconds\n",endtime-starttime);

    // Just to check solution
    DummyDistributedVector r_tmp(b);
    triblock_A->product(r_tmp, x);
    r_tmp -= b;
    double nr;
    r_tmp.transposeProduct(nr, r_tmp);

    if(rank==0) {
      std::cout << "Real residual " << sqrt(nr) << std::endl;
      //r_tmp.print();
    }

    // To print the solution
    // if(rank==0) std::cout << x.data << std::endl;

    if(rank == 0) std::cout<<"Finish computation"<<std::endl;

    MPI_Finalize();

    return 0;
}
