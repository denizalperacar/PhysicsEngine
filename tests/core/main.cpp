#include "../../src/core/common/pch.h"
#include "../../src/core/coordinate_frame/frame.h"


int main() {

  
  PE::vector_t<double, 3> a(4.0, 3., 2.);
  PE::matrix_t<double, 4, 3> b(1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.);
  for (int i = 0; i < 4; i++) {
    std::cout << (b * a)[i] << std::endl;
  }
  
  PE::vector_t<double, 4> c(1., 2., 3., 4.);
  PE::vector_t<double, 3> u1(4., 5., 6.);
  PE::matrix_t<double, 3, 3> e = PE::rodrigues_formula(PE::u1<double>(), 90.0, false);

  PE::vector_t<double, 3> pos {0., 2.4414, 0.52};
  PE::htm_t<double> htm(pos);
  htm.set_dcm_from_euler_123_rotation(90., 0., -180., false);
  int n = 10;
  PE::vector_t<double, 3> p;

  for (int i = 0; i < n; i++) {
    double d = PE::length(p);
    double xl = PE::sin(PE::degrees_to_radians(n) * d);  
    double yl = PE::cos(PE::degrees_to_radians(n) * d);  
    
  }



  return 0;
}
