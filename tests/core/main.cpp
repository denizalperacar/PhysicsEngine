#include "../../src/core/common/pch.h"

int main() {

  
  PE::vector_t<double, 3, sizeof(double)> a(4.0, 3, 2);
  std::cout << a.r << std::endl;

  return 0;
}
