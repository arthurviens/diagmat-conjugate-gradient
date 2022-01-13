#include <iostream>
#include <fstream>
#include <iomanip>
#include <thread>         // std::this_thread::sleep_for
#include <chrono>         // std::chrono::seconds
#include <memory>
#include <mpi.h>
#include <vector>
#include "Eigen/Dense"
#include "Eigen/Core"

#define MAXBUFSIZE  ((int) 1e6)


namespace disp
{
  template <typename Derived>
  void print_matsize(const Eigen::EigenBase<Derived>& b) {
      std::cout << "size (rows, cols): " << b.size() << " (" << b.rows()
                << ", " << b.cols() << ")" << std::endl;
  }

  template <typename Derived>
  void print_mat(Eigen::MatrixBase<Derived>& A) {
    std::string sep = "\n----------------------------------------\n";

    Eigen::IOFormat CleanFmt(3, 0, ", ", "\n", "[", "]");

    std::cout << sep << "Matrix of size " << A.size() << std::endl;
    std::cout << A.format(CleanFmt) << sep;
  }
}
