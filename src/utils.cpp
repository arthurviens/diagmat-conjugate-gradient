#include <iostream>
#include <iomanip>
#include <thread>         // std::this_thread::sleep_for
#include <chrono>         // std::chrono::seconds
#include <memory>
#include <mpi.h>
#include <vector>
#include "Eigen/Dense"
#include "Eigen/Core"

using namespace Eigen;

template <typename Derived>
void print_size(const Eigen::EigenBase<Derived>& b)
{
  std::cout << "size (rows, cols): " << b.size() << " (" << b.rows()
            << ", " << b.cols() << ")" << std::endl;
}
