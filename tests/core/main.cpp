#include "../../src/core/common/pch.h"

int main() {

  
  PE::vector_t<double, 3, sizeof(double)> a(4.0, 3, 2);
  PE::matrix_t<double, 4, 3> b(1., 2., 3., 4., 5., 6., 7., 8., 9., 7., 8., 9.);
  for (int i = 0; i < 4; i++) {
    std::cout << (b * a)[i] << std::endl;
  }
  
  PE::vector_t<double, 4> c(1., 2., 3., 4.);
  PE::vector_t<double, 3> d(4., 5., 6.);

  print(outer(c, d));

  std::cout << b(1,1) << std::endl;
  return 0;
}
