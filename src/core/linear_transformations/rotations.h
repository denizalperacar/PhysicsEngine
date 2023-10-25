#ifndef AC21051C_12E3_4BA7_990D_33EFC5668EDD
#define AC21051C_12E3_4BA7_990D_33EFC5668EDD

#include "../common/common.h"
#include "matrix_base.h"

PE_BEGIN

template <typename T>
PE_HOST_DEVICE matrix_t<T, 3, 3> rodrigues_formula(const vector_t<T, 3>& n, T angle, bool is_radian = true) {
    if (!is_radian) {
        angle = angle * pi / 180.0;
    }
    T c = cos(angle);
    T s = sin(angle);
    T t = 1 - c;
    matrix_t<T, 3, 3> I = matrix_t<T, 3, 3>::identity();
    matrix_t<T, 3, 3> n_nT = outer(n, n);
    matrix_t<T, 3, 3> n_tilde = skew_symmetric_matrix(n);

    return I * c + n_tilde * s + n_nT * t;
}

template <typename T>
PE_HOST_DEVICE matrix_t<T, 3, 3> rotation_x(T delta, bool is_radian = true) {
    if (!is_radian) {
        delta = delta * pi / 180.0;
    }
    T s = sin(delta);
    T c = cos(delta);
    return {1, 0, 0, 0, c, s, 0, -s, c};
}

template <typename T>
PE_HOST_DEVICE matrix_t<T, 3, 3> rotation_y(T delta, bool is_radian = true) {
    if (!is_radian) {
        delta = delta * pi / 180.0;
    }
    T s = sin(delta);
    T c = cos(delta);
    return {c, 0, -s, 0, 1, 0, s, 0, c};
}

template <typename T>
PE_HOST_DEVICE matrix_t<T, 3, 3> rotation_z(T delta, bool is_radian = true) {
    if (!is_radian) {
        delta = delta * pi / 180.0;
    }
    T s = sin(delta);
    T c = cos(delta);
    return {c, s, 0, -s, c, 0, 0, 0, 1};
}

PE_END

#endif /* AC21051C_12E3_4BA7_990D_33EFC5668EDD */
