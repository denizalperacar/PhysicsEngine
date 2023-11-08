#include "pch.h"
#include "../../src/core/coordinate_frame/relative_frame.h"
#include "../../src/core/coordinate_frame/absolute_frame.h"
#include "../../src/core/camera/simple_camera.h"


#include <fstream>

void write_json(PE::htm_t<double> &htm, std::ofstream &file) {
  file << "  \"transform_matrix\": [\n";
  PE_UNROLL
  for (int i = 0; i < 4; ++i) {
    file << "    [\n      " << htm.matrix[0][i] << ", " << htm.matrix[1][i] << ", " << htm.matrix[2][i] << ", " << htm.matrix[3][i] << "\n" << "    ],\n";
  }
  file << "  ]\n";
}

void write_image_json(PE::htm_t<double> &htm, std::ofstream &file, std::string name, double sharpness) {
  file << "{\n  \"file_path\": \"" << name << "\",\n";
  file << "  \"sharpness\": " << sharpness << ",\n";
  write_json(htm, file);
  file << "},\n";
}


int main() {

  
  PE::vector_t<double, 3> a(4.0, 3., 2.);
  PE::matrix_t<double, 4, 3> b(1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.);
  for (int i = 0; i < 4; i++) {
    std::cout << (b * a)[i] << std::endl;
  }
  
  PE::vector_t<double, 4> c(1., 2., 3., 4.);
  PE::vector_t<double, 3> u1(4., 5., 6.);
  PE::matrix_t<double, 3, 3> e = PE::rodrigues_formula(PE::u1<double>(), 90.0, false);

  PE::vector_t<double, 3> pos {0., 2.4791, 0.52};
  PE::htm_t<double> htm(pos);
  htm.set_dcm_from_euler_123_rotation(90., 0., -180., false);
  int n = 7; int start{ 160 };
  PE::vector_t<double, 3> p = pos;

  double sharpness = 110.74499252993057;
  std::ofstream file;
  file.open("/home/deniz/codes/PE/PhysicsEngine/tests/core/orientations.txt");

  for (int i = 0; i < n; i++) {
    double ii = i * 10.0 + start;
    double d = PE::length(PE::dvec3(pos[0], pos[1], 0.));
    double xl = PE::sin(PE::degrees_to_radians(ii)) * d;  
    double yl = PE::cos(PE::degrees_to_radians(ii)) * d;  
    p = PE::vector_t<double, 3>(xl, yl, 0.52);
    htm.set_dcm_from_euler_123_rotation(90., 0., (ii + 180) * -1, false);
    htm.set_position(p);
    write_image_json(htm, file, fmt::format("./images/image{}.png", ii), sharpness);
  }
  file.close();


  PE::GlobalFrame<double>& global_frame = PE::GlobalFrame<double>::get_instance();
  PE::GlobalFrame<double>& global_frame2 = PE::GlobalFrame<double>::get_instance();

  PE::RelativeFrame<double> rf(&global_frame, PE::htm_t<double>(PE::rotation_xyz(90., 0., 0., false), PE::vector_t<double, 3>(1., 2., 3.)));
  PE::RelativeFrame<double> ef(&rf, PE::htm_t<double>(PE::rotation_xyz(0., 0., 0., false), PE::vector_t<double, 3>(1., 2., 3.)));

  PE::htm_t<double> res = ef.resolve_frame_in_global();
  PE::print(res);

  std::cout << "Global is singelton: " << ((void*)&global_frame == (void*)&global_frame2) << std::endl;

  PE::print(PE::BLUE<double>);

  return 0;
}
