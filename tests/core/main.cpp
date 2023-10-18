#include "../../src/core/common/pch.h"

int main() {

  
  PE::vector_t<double, 4, sizeof(double)> a(4.0);
  a.print();

  return 0;
}
