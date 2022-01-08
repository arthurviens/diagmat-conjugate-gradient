#include <iostream>
#include "utils.hpp"
#include "Eigen/Dense"
#include "Eigen/Core"



namespace covid
{
  template <typename Derived>
  void print_matsize(const Eigen::EigenBase<Derived>& b)
  {
    std::cout << "size (rows, cols): " << b.size() << " (" << b.rows()
              << ", " << b.cols() << ")" << std::endl;
  }
}
