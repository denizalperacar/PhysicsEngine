#ifndef AC21051C_12E3_4BA7_990D_33EFC5668EDD
#define AC21051C_12E3_4BA7_990D_33EFC5668EDD

#include "common.h"
#include "matrix_base.h"

PE_BEGIN

template <typename T>
PE_HOST_DEVICE matrix_t<T, 3, 3> rodrigues_formula(const vector_t<T, 3>& n, T delta, bool is_radian = true) {
	if (!is_radian) { delta = degrees_to_radians(delta); }
	T c = cos(delta);
	T s = sin(delta);
	T t = 1 - c;
	matrix_t<T, 3, 3> I = matrix_t<T, 3, 3>::identity();
	matrix_t<T, 3, 3> n_nT = outer(n, n);
	matrix_t<T, 3, 3> n_tilde = skew_symmetric_matrix(n);

	return I * c + n_tilde * s + n_nT * t;
}

template <typename T>
PE_HOST_DEVICE matrix_t<T, 3, 3> rotation_x(T delta, bool is_radian = true) {
	if (!is_radian) { delta = degrees_to_radians(delta); }
	T s = sin(delta);
	T c = cos(delta);
	return {1, 0, 0, 0, c, s, 0, -s, c};
}

template <typename T>
PE_HOST_DEVICE matrix_t<T, 3, 3> rotation_y(T delta, bool is_radian = true) {
	if (!is_radian) { delta = degrees_to_radians(delta); }
	T s = sin(delta);
	T c = cos(delta);
	return {c, 0, -s, 0, 1, 0, s, 0, c};
}

template <typename T>
PE_HOST_DEVICE matrix_t<T, 3, 3> rotation_z(T delta, bool is_radian = true) {
	if (!is_radian) { delta = degrees_to_radians(delta); }
	T s = sin(delta);
	T c = cos(delta);
	return {c, s, 0, -s, c, 0, 0, 0, 1};
}

template <typename T>
PE_HOST_DEVICE matrix_t<T, 3, 3> rotation_xy(T ang1, T ang2, bool is_radian = true) {
	if (!is_radian) { 
		ang1 = degrees_to_radians(ang1); 
		ang2 = degrees_to_radians(ang2); 
	}
	T s1 = sin(ang1); T s2 = sin(ang2);
	T c1 = cos(ang1); T c2 = cos(ang2);
	return {c2, s1*s2, -c1*s2, 0, c1, s1, s2, -c2*s1, c1*c2};
}

template <typename T>
PE_HOST_DEVICE matrix_t<T, 3, 3> rotation_xz(T ang1, T ang3, bool is_radian = true) {
	if (!is_radian) { 
		ang1 = degrees_to_radians(ang1); 
		ang3 = degrees_to_radians(ang3); 
	}
	T s1 = sin(ang1); T s3 = sin(ang3);
	T c1 = cos(ang1); T c3 = cos(ang3);
  return {c3, c1*s3, s1*s3, -s3, c1*c3, c3*s1, 0, -s1, c1};
}

template <typename T>
PE_HOST_DEVICE matrix_t<T, 3, 3> rotation_yx(T ang2, T ang1, bool is_radian = true) {
	if (!is_radian) { 
		ang1 = degrees_to_radians(ang1); 
		ang2 = degrees_to_radians(ang2); 
	}
	T s1 = sin(ang1); T s2 = sin(ang2);
	T c1 = cos(ang1); T c2 = cos(ang2);
  return {c2, 0, -s2, s1*s2, c1, c2*s1, c1*s2, -s1, c1*c2};
}

template <typename T>
PE_HOST_DEVICE matrix_t<T, 3, 3> rotation_yz(T ang2, T ang3, bool is_radian = true) {
	if (!is_radian) { 
		ang3 = degrees_to_radians(ang3); 
		ang2 = degrees_to_radians(ang2); 
	}
	T s3 = sin(ang3); T s2 = sin(ang2);
	T c3 = cos(ang3); T c2 = cos(ang2);
  return {c2*c3, s3, -c3*s2, -c2*s3, c3, s2*s3, s2, 0, c2};
}

template <typename T>
PE_HOST_DEVICE matrix_t<T, 3, 3> rotation_zx(T ang3, T ang1, bool is_radian = true) {
	if (!is_radian) { 
		ang3 = degrees_to_radians(ang3); 
		ang1 = degrees_to_radians(ang1); 
	}
	T s3 = sin(ang3); T s1 = sin(ang1);
	T c3 = cos(ang3); T c1 = cos(ang1);
  return {c3, s3, 0, -c1*s3, c1*c3, s1, s1*s3, -c3*s1, c1};
}

template <typename T>
PE_HOST_DEVICE matrix_t<T, 3, 3> rotation_zy(T ang3, T ang2, bool is_radian = true) {
	if (!is_radian) { 
		ang3 = degrees_to_radians(ang3); 
		ang2 = degrees_to_radians(ang2); 
	}
	T s3 = sin(ang3); T s2 = sin(ang2);
	T c3 = cos(ang3); T c2 = cos(ang2);
  return {c2*c3, c2*s3, -s2, -s3, c3, 0, c3*s2, s2*s3, c2};
}

// r1_2_1 = 
template <typename T>
PE_HOST_DEVICE matrix_t<T, 3, 3> rotation_xyx(T ang1, T ang2, T ang3, bool is_radian = true) {
  if (!is_radian) { 
    ang1 = degrees_to_radians(ang1); 
    ang2 = degrees_to_radians(ang2); 
    ang1 = degrees_to_radians(ang1); 
  }
  T s11 = sin(ang1); T s22 = sin(ang2); T s31 = sin(ang3);
  T c11 = cos(ang1); T c22 = cos(ang2); T c31 = cos(ang3);
  return {c22, s11*s22, -c11*s22, s11*s22, c11*c11 - c22*s11*c11, c11*c22*s11 + c11*s11, c11*s22, -c11*c22*s11 - c11*s11, c11*c11*c22 - s11*c11};
}

// r1_2_3 = 
template <typename T>
PE_HOST_DEVICE matrix_t<T, 3, 3> rotation_xyz(T ang1, T ang2, T ang3, bool is_radian = true) {
  if (!is_radian) { 
    ang1 = degrees_to_radians(ang1); 
    ang2 = degrees_to_radians(ang2); 
    ang3 = degrees_to_radians(ang3); 
  }
  T s11 = sin(ang1); T s22 = sin(ang2); T s33 = sin(ang3);
  T c11 = cos(ang1); T c22 = cos(ang2); T c33 = cos(ang3);
  return {c22*c33, c11*s33 + c33*s11*s22, -c11*c33*s22 + s11*s33, -c22*s33, c11*c33 - s11*s22*s33, c11*s22*s33 + c33*s11, s22, -c22*s11, c11*c22};
}

// r1_3_1 = 
template <typename T>
PE_HOST_DEVICE matrix_t<T, 3, 3> rotation_xzx(T ang1, T ang2, T ang3, bool is_radian = true) {
  if (!is_radian) { 
    ang1 = degrees_to_radians(ang1); 
    ang3 = degrees_to_radians(ang3); 
    ang1 = degrees_to_radians(ang1); 
  }
  T s11 = sin(ang1); T s23 = sin(ang2); T s31 = sin(ang3);
  T c11 = cos(ang1); T c23 = cos(ang2); T c31 = cos(ang3);
  return {c31, c11*s31, s11*s31, -c11*s31, c11*c11*c31 - s11*c11, c11*c31*s11 + c11*s11, s11*s31, -c11*c31*s11 - c11*s11, c11*c11 - c31*s11*c11};
}

// r1_3_2 = 
template <typename T>
PE_HOST_DEVICE matrix_t<T, 3, 3> rotation_xzy(T ang1, T ang2, T ang3, bool is_radian = true) {
  if (!is_radian) { 
    ang1 = degrees_to_radians(ang1); 
    ang3 = degrees_to_radians(ang3); 
    ang2 = degrees_to_radians(ang2); 
  }
  T s11 = sin(ang1); T s23 = sin(ang2); T s32 = sin(ang3);
  T c11 = cos(ang1); T c23 = cos(ang2); T c32 = cos(ang3);
  return {c23*c32, c11*c23*s32 + s11*s23, -c11*s23 + c23*s11*s32, -s32, c11*c32, c32*s11, c32*s23, c11*s23*s32 - c23*s11, c11*c23 + s11*s23*s32};
}

// r2_1_2 = 
template <typename T>
PE_HOST_DEVICE matrix_t<T, 3, 3> rotation_yxy(T ang1, T ang2, T ang3, bool is_radian = true) {
  if (!is_radian) { 
    ang2 = degrees_to_radians(ang2); 
    ang1 = degrees_to_radians(ang1); 
    ang2 = degrees_to_radians(ang2); 
  }
  T s12 = sin(ang1); T s21 = sin(ang2); T s32 = sin(ang3);
  T c12 = cos(ang1); T c21 = cos(ang2); T c32 = cos(ang3);
  return {-c12*s21*s21 + c21*s21, s12*s21, -c12*c21*s21 - c21*s21, s12*s21, c12, c21*s12, c12*c21*s21 + c21*s21, -c21*s12, c12*c21*s21 - s21*s21};
}

// r2_1_3 = 
template <typename T>
PE_HOST_DEVICE matrix_t<T, 3, 3> rotation_yxz(T ang1, T ang2, T ang3, bool is_radian = true) {
  if (!is_radian) { 
    ang2 = degrees_to_radians(ang2); 
    ang1 = degrees_to_radians(ang1); 
    ang3 = degrees_to_radians(ang3); 
  }
  T s12 = sin(ang1); T s21 = sin(ang2); T s33 = sin(ang3);
  T c12 = cos(ang1); T c21 = cos(ang2); T c33 = cos(ang3);
  return {c21*c33 + s12*s21*s33, c12*s33, c21*s12*s33 - c33*s21, -c21*s33 + c33*s12*s21, c12*c33, c21*c33*s12 + s21*s33, c12*s21, -s12, c12*c21};
}

// r2_3_1 = 
template <typename T>
PE_HOST_DEVICE matrix_t<T, 3, 3> rotation_yzx(T ang1, T ang2, T ang3, bool is_radian = true) {
  if (!is_radian) { 
    ang2 = degrees_to_radians(ang2); 
    ang3 = degrees_to_radians(ang3); 
    ang1 = degrees_to_radians(ang1); 
  }
  T s12 = sin(ang1); T s23 = sin(ang2); T s31 = sin(ang3);
  T c12 = cos(ang1); T c23 = cos(ang2); T c31 = cos(ang3);
  return {c23*c31, s31, -c31*s23, -c12*c23*s31 + s12*s23, c12*c31, c12*s23*s31 + c23*s12, c12*s23 + c23*s12*s31, -c31*s12, c12*c23 - s12*s23*s31};
}

// r2_3_2 = 
template <typename T>
PE_HOST_DEVICE matrix_t<T, 3, 3> rotation_yzy(T ang1, T ang2, T ang3, bool is_radian = true) {
  if (!is_radian) { 
    ang2 = degrees_to_radians(ang2); 
    ang3 = degrees_to_radians(ang3); 
    ang2 = degrees_to_radians(ang2); 
  }
  T s12 = sin(ang1); T s23 = sin(ang2); T s32 = sin(ang3);
  T c12 = cos(ang1); T c23 = cos(ang2); T c32 = cos(ang3);
  return {c23*c23*c32 - s23*c23, c23*s32, -c23*c32*s23 - c23*s23, -c23*s32, c32, s23*s32, c23*c32*s23 + c23*s23, s23*s32, c23*c23 - c32*s23*c23};
}

// r3_1_2 = 
template <typename T>
PE_HOST_DEVICE matrix_t<T, 3, 3> rotation_zxy(T ang1, T ang2, T ang3, bool is_radian = true) {
  if (!is_radian) { 
    ang3 = degrees_to_radians(ang3); 
    ang1 = degrees_to_radians(ang1); 
    ang2 = degrees_to_radians(ang2); 
  }
  T s13 = sin(ang1); T s21 = sin(ang2); T s32 = sin(ang3);
  T c13 = cos(ang1); T c21 = cos(ang2); T c32 = cos(ang3);
  return {c21*c32 - s13*s21*s32, c21*s32 + c32*s13*s21, -c13*s21, -c13*s32, c13*c32, s13, c21*s13*s32 + c32*s21, -c21*c32*s13 + s21*s32, c13*c21};
}

// r3_1_3 = 
template <typename T>
PE_HOST_DEVICE matrix_t<T, 3, 3> rotation_zxz(T ang1, T ang2, T ang3, bool is_radian = true) {
  if (!is_radian) { 
    ang3 = degrees_to_radians(ang3); 
    ang1 = degrees_to_radians(ang1); 
    ang3 = degrees_to_radians(ang3); 
  }
  T s13 = sin(ang1); T s21 = sin(ang2); T s33 = sin(ang3);
  T c13 = cos(ang1); T c21 = cos(ang2); T c33 = cos(ang3);
  return {-c13*s33*s33 + c33*s33, c13*c33*s33 + c33*s33, s13*s33, -c13*c33*s33 - c33*s33, c13*c33*s33 - s33*s33, c33*s13, s13*s33, -c33*s13, c13};
}

// r3_2_1 = 
template <typename T>
PE_HOST_DEVICE matrix_t<T, 3, 3> rotation_zyx(T ang1, T ang2, T ang3, bool is_radian = true) {
  if (!is_radian) { 
    ang3 = degrees_to_radians(ang3); 
    ang2 = degrees_to_radians(ang2); 
    ang1 = degrees_to_radians(ang1); 
  }
  T s13 = sin(ang1); T s22 = sin(ang2); T s31 = sin(ang3);
  T c13 = cos(ang1); T c22 = cos(ang2); T c31 = cos(ang3);
  return {c22*c31, c22*s31, -s22, -c13*s31 + c31*s13*s22, c13*c31 + s13*s22*s31, c22*s13, c13*c31*s22 + s13*s31, c13*s22*s31 - c31*s13, c13*c22};
}

// r3_2_3 = 
template <typename T>
PE_HOST_DEVICE matrix_t<T, 3, 3> rotation_zyz(T ang1, T ang2, T ang3, bool is_radian = true) {
  if (!is_radian) { 
    ang3 = degrees_to_radians(ang3); 
    ang2 = degrees_to_radians(ang2); 
    ang3 = degrees_to_radians(ang3); 
  }
  T s13 = sin(ang1); T s22 = sin(ang2); T s33 = sin(ang3);
  T c13 = cos(ang1); T c22 = cos(ang2); T c33 = cos(ang3);
  return {c22*c33*c33 - s33*s33, c22*c33*s33 + c33*s33, -c33*s22, -c22*c33*s33 - c33*s33, -c22*s33*s33 + c33*s33, s22*s33, c33*s22, s22*s33, c22};
}


PE_END

#endif /* AC21051C_12E3_4BA7_990D_33EFC5668EDD */
