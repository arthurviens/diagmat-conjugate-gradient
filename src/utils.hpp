#include <iostream>
#include <iomanip>
#include <thread>         // std::this_thread::sleep_for
#include <chrono>         // std::chrono::seconds
#include <memory>
#include <mpi.h>
#include <vector>
#include "Eigen/Dense"
#include "Eigen/Core"

namespace covid
{
  template <typename Derived>
  void print_matsize(const Eigen::EigenBase<Derived>& b);
}
