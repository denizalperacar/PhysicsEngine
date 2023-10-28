#ifndef B0AF69A6_577B_4BF9_AF88_BB09112AEE6E
#define B0AF69A6_577B_4BF9_AF88_BB09112AEE6E

#include "../common/common.h"
#include "rotation_dyads.h"

PE_BEGIN

#define PE_HTM_FROM_ROT1(dir, name) \
PE_HOST_DEVICE void set_dcm_from_euler_##name##_rotation(T ang1) { \
  return set_dcm(set_dcm_from_euler_##name##_rotation(ang1)); \
} \
PE_HOST_DEVICE void set_dcm_from_euler_##dir##_rotation(T ang1) { \
  return set_dcm(set_dcm_from_euler_##name##_rotation(ang1)); \
}

#define PE_HTM_FROM_ROT2(dir, name) \
PE_HOST_DEVICE void set_dcm_from_euler_##name##_rotation(T ang1, T ang2) { \
  return set_dcm(set_dcm_from_euler_##name##_rotation(ang1, ang2)); \
} \
PE_HOST_DEVICE void set_dcm_from_euler_##dir##_rotation(T ang1, T ang2) { \
  return set_dcm(set_dcm_from_euler_##name##_rotation(ang1, ang2)); \
}

#define PE_HTM_FROM_ROT3(dir, name) \
PE_HOST_DEVICE void set_dcm_from_euler_##name##_rotation(T ang1, T ang2, T ang3) { \
  return set_dcm(set_dcm_from_euler_##name##_rotation(ang1, ang2, ang3)); \
} \
PE_HOST_DEVICE void set_dcm_from_euler_##dir##_rotation(T ang1, T ang2, T ang3) { \
  return set_dcm(set_dcm_from_euler_##name##_rotation(ang1, ang2, ang3)); \
}

#define PE_HTM_FROM_ROT() \
PE_HTM_FROM_ROT1(1, x) \
PE_HTM_FROM_ROT1(2, y) \
PE_HTM_FROM_ROT1(3, z) \
PE_HTM_FROM_ROT2(12, xy) \
PE_HTM_FROM_ROT2(13, xz) \
PE_HTM_FROM_ROT2(21, yx) \
PE_HTM_FROM_ROT2(23, yz) \
PE_HTM_FROM_ROT2(31, zx) \
PE_HTM_FROM_ROT2(22, zy) \
PE_HTM_FROM_ROT3(121, xyx) \
PE_HTM_FROM_ROT3(123, xyz) \
PE_HTM_FROM_ROT3(132, xzy) \
PE_HTM_FROM_ROT3(131, xzx) \
PE_HTM_FROM_ROT3(212, yxy) \
PE_HTM_FROM_ROT3(213, yxz) \
PE_HTM_FROM_ROT3(231, yzy) \
PE_HTM_FROM_ROT3(232, yzx) \
PE_HTM_FROM_ROT3(312, zxy) \
PE_HTM_FROM_ROT3(313, zxz) \
PE_HTM_FROM_ROT3(321, zyx) \
PE_HTM_FROM_ROT3(323, zyz)


// ----------------------------------------------------------------------------


template <typename T>
struct htm_t {
  
  using value_type = T;

  PE_HTM_FROM_ROT()

  PE_HOST_DEVICE htm_t() = default;
  
  PE_HOST_DEVICE htm_t(const matrix_t<T, 4, 4>& htm) : matrix(htm) {}
  
  PE_HOST_DEVICE htm_t(const matrix_t<T, 3, 3>& dcm, const vector_t<T, 3>& position) {
    from_pdcm(dcm, position);
  }

  PE_HOST_DEVICE htm_t(const matrix_t<T, 3, 3>& dcm) {
    set_dcm(dcm);
  }

  PE_HOST_DEVICE htm_t(const quaternion_t<T>& q, const vector_t<T, 3>& position) {
    from_pdcm(q.to_dcm(), position);
  }

  PE_HOST_DEVICE htm_t(vector_t<T, 3> position) {
    set_position(position);
  }

  PE_HOST_DEVICE htm_t(const quaternion_t<T>& q) {
    set_quaternion(q);
  }

  PE_HOST htm_t(const std::vector<htm_t<T>>& rhtm) {
    matrix = rhtm[0].matrix;
    for (int i = 1; i < rhtm.size(); ++i) {
      matrix = matrix * rhtm[i].matrix;
    }
  }

  PE_HOST_DEVICE htm_t<T> operator*(const htm_t<T>& rhs) const {
    return htm_t<T>(matrix * rhs.matrix);
  }

  // returns the inverted form of the htm.
  PE_HOST_DEVICE htm_t<T> get_inverse() const {
    matrix_t<T, 3, 3> dcm = get_dcm();
    // invert dcm
    dcm = dcm.transpose();
    // invert position
    vector_t<T, 3> position = -dcm * get_position();
    return htm_t<T>(dcm, position);
  }

  // an inplace operation
  PE_HOST_DEVICE void invert() {
    matrix_t<T, 3, 3> dcm = get_dcm();
    // invert dcm
    dcm = dcm.transpose();
    // invert position
    vector_t<T, 3> position = -dcm * get_position();
    from_pdcm(dcm, position);
  }

  PE_HOST_DEVICE void set_dcm(const matrix_t<T, 3, 3>& dcm) {
    PE_UNROLL
    for (int i = 0; i < 3; ++i) {
      PE_UNROLL
      for (int j = 0; j < 3; ++j) {
        matrix[i][j] = dcm[i][j];
      }
    }
  }

  PE_HOST_DEVICE void set_position(const vector_t<T, 3>& position) {
    PE_UNROLL
    for (int i = 0; i < 3; ++i) {
      matrix[i][3] = position[i];
    }
  }

  PE_HOST_DEVICE void set_quaternion(const quaternion_t<T>& q) {
    set_dcm(q.to_dcm());
  }

  PE_HOST_DEVICE void from_pdcm(const matrix_t<T, 3, 3>& dcm, const vector_t<T, 3>& position) {
    set_dcm(dcm);
    set_position(position);
  }

  PE_HOST_DEVICE void from_pquat(const quaternion_t<T>& q, const vector_t<T, 3>& position) {
    set_quaternion(q);
    set_position(position);
  }

  PE_HOST_DEVICE void from_htm(const matrix_t<T, 4, 4>& htm) {
    matrix = htm;
  }

  PE_HOST_DEVICE void reset() {
    matrix = matrix_t<T, 4, 4>::identity();
  }

  // get the dcm from the htm
  PE_HOST_DEVICE matrix_t<T, 3, 3> get_dcm() const {
    matrix_t<T, 3, 3> dcm;
    PE_UNROLL
    for (int i = 0; i < 3; ++i) {
      PE_UNROLL
      for (int j = 0; j < 3; ++j) {
        dcm[i][j] = matrix[i][j];
      }
    }
    return dcm;
  }

  PE_HOST_DEVICE vector_t<T, 3> get_position() const {
    vector_t<T, 3> position;
    PE_UNROLL
    for (int i = 0; i < 3; ++i) {
      position[i] = matrix[i][3];
    }
    return position;
  }

  PE_HOST_DEVICE quaternion_t<T> get_quaternion() const {
    return quaternion_t<T>(get_dcm());
  }

  PE_HOST_DEVICE matrix_t<T, 4, 4> get_htm() const {
    return matrix;
  }

  PE_HOST_DEVICE vector_t<T, 4> get_angle_axis() const {
    // Implementation from http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToAngle/index.htm
    vector_t<T, 4> result;
    T angle, x, y, z; // variables for result
    T epsilon = (T)0.01; // margin to allow for rounding errors
    T epsilon2 = (T)0.1; // margin to distinguish between 0 and 180 degrees
    // optional check that input is pure rotation, 'isRotationMatrix' is defined at:
    // http://www.euclideanspace.com/maths/algebra/matrix/orthogonal/rotation/
    if ((abs(matrix[0][1]-matrix[1][0])< epsilon)
      && (abs(matrix[0][2]-matrix[2][0])< epsilon)
      && (abs(matrix[1][2]-matrix[2][1])< epsilon)) {
      // singularity found
      // first check for identity matrix which must have +1 for all terms
      //  in leading diagonaland zero in other terms
      if ((abs(matrix[0][1]+matrix[1][0]) < epsilon2)
        && (abs(matrix[0][2]+matrix[2][0]) < epsilon2)
        && (abs(matrix[1][2]+matrix[2][1]) < epsilon2)
        && (abs(matrix[0][0]+matrix[1][1]+matrix[2][2]-3) < epsilon2)) {
        // this singularity is identity matrix so angle = 0
        result = { 0, 1, 0, 0 }; // zero angle, arbitrary axis
        return result; // zero angle, arbitrary axis
      }
      // otherwise this singularity is angle = 180
      angle = pi;
      T xx = (matrix[0][0]+1)/2;
      T yy = (matrix[1][1]+1)/2;
      T zz = (matrix[2][2]+1)/2;
      T xy = (matrix[0][1]+matrix[1][0])/4;
      T xz = (matrix[0][2]+matrix[2][0])/4;
      T yz = (matrix[1][2]+matrix[2][1])/4;
      if ((xx > yy) && (xx > zz)) { // matrix[0][0] is the largest diagonal term
        if (xx< epsilon) {
          x = 0;
          y = 0.7071;
          z = 0.7071;
        } else {
          x = sqrt(xx);
          y = xy/x;
          z = xz/x;
        }
      } else if (yy > zz) { // matrix[1][1] is the largest diagonal term
        if (yy < epsilon) { 
          x = 0.7071; 
          y = 0; 
          z = 0.7071;
        } else {
          y = sqrt(yy);
          x = xy/y;
          z = yz/y;
        }	
      } else { // matrix[2][2] is the largest diagonal term so base result on this
        if (zz< epsilon) {
          x = 0.7071;
          y = 0.7071;
          z = 0;
        } else {
          z = sqrt(zz);
          x = xz/z;
          y = yz/z;
        }
      }

      result = { angle, x, y, z }; 
      return result; // return 180 deg rotation
	  }
    // as we have reached here there are no singularities so we can handle normally
    double s = sqrt((matrix[2][1] - matrix[1][2])*(matrix[2][1] - matrix[1][2])
      +(matrix[0][2] - matrix[2][0])*(matrix[0][2] - matrix[2][0])
      +(matrix[1][0] - matrix[0][1])*(matrix[1][0] - matrix[0][1])); // used to normalise
    if (abs(s) < 0.001) s=1; 
    // prevent divide by zero, should not happen if matrix is orthogonal and should be
    // caught by singularity test above, but I've left it in just in case
    angle = acos(( matrix[0][0] + matrix[1][1] + matrix[2][2] - 1)/2);
    x = (matrix[2][1] - matrix[1][2])/s;
    y = (matrix[0][2] - matrix[2][0])/s;
    z = (matrix[1][0] - matrix[0][1])/s;
    result = { angle, x, y, z }; 
    return result;
  }

  PE_HOST_DEVICE void print() const {
  #if defined(__CUDA_ARCH__)
    printf("HTM:\n");
    PE_UNROLL
    for (int i = 0; i < 4; ++i) {
      printf("%f %f %f %f\n", matrix[0][i], matrix[1][i], matrix[2][i], matrix[3][i]);
    }
  #else 
    std::cout << "HTM:\n";
    PE_UNROLL
    for (int i = 0; i < 4; ++i) {
      std::cout << matrix[0][i] << " " << matrix[1][i] << " " << matrix[2][i] << " " << matrix[3][i] << "\n";
    }
  #endif
  }

  matrix_t<T, 4, 4> matrix = matrix_t<T, 4, 4>::identity();
};

// ----------------------------------------------------------------------------

// helper functions
#undef PE_HTM_FROM_ROT1
#undef PE_HTM_FROM_ROT2
#undef PE_HTM_FROM_ROT3

#define PE_HTM_FROM_ROT1(dir, name) \
template <typename T> \
PE_HOST_DEVICE htm_t<T> from_euler_##name##_rotation(T ang1) { \
  return htm_t<T>(from_euler_##name##_rotation(ang1)); \
} \
\
template <typename T> \
PE_HOST_DEVICE htm_t<T> from_euler_##dir##_rotation(T ang1) { \
  return htm_t<T>(from_euler_##name##_rotation(ang1)); \
} \
\
template <typename T> \
PE_HOST_DEVICE htm_t<T> from_euler_##name##_rotation(T ang1, const vector_t<T, 3>& position) { \
  return htm_t<T>(from_euler_##name##_rotation(ang1), position); \
} \
\
template <typename T> \
PE_HOST_DEVICE htm_t<T> from_euler_##dir##_rotation(T ang1, const vector_t<T, 3>& position) { \
  return htm_t<T>(from_euler_##name##_rotation(ang1, position)); \
}

#define PE_HTM_FROM_ROT2(dir, name) \
template <typename T> \
PE_HOST_DEVICE htm_t<T> from_euler_##name##_rotation(T ang1, T ang2) { \
  return htm_t<T>(from_euler_##name##_rotation(ang1, ang2)); \
} \
\
template <typename T> \
PE_HOST_DEVICE htm_t<T> from_euler_##dir##_rotation(T ang1, T ang2) { \
  return htm_t<T>(from_euler_##name##_rotation(ang1, ang2)); \
} \
\
template <typename T> \
PE_HOST_DEVICE htm_t<T> from_euler_##name##_rotation(T ang1, T ang2, const vector_t<T, 3>& position) { \
  return htm_t<T>(from_euler_##name##_rotation(ang1, ang2, position)); \
} \
\
template <typename T> \
PE_HOST_DEVICE htm_t<T> from_euler_##dir##_rotation(T ang1, T ang2, const vector_t<T, 3>& position) { \
  return htm_t<T>(from_euler_##name##_rotation(ang1, ang2, position)); \
} \

#define PE_HTM_FROM_ROT3(dir, name) \
template <typename T> \
PE_HOST_DEVICE htm_t<T> from_euler_##name##_rotation(T ang1, T ang2, T ang3) { \
  return htm_t<T>(from_euler_##name##_rotation(ang1, ang2, ang3)); \
} \
\
template <typename T> \
PE_HOST_DEVICE htm_t<T> from_euler_##dir##_rotation(T ang1, T ang2, T ang3) { \
  return htm_t<T>(from_euler_##name##_rotation(ang1, ang2, ang3)); \
} \
\
template <typename T> \
PE_HOST_DEVICE htm_t<T> from_euler_##name##_rotation(T ang1, T ang2, T ang3, const vector_t<T, 3>& position) { \
  return htm_t<T>(from_euler_##name##_rotation(ang1, ang2, ang3, position)); \
} \
\
template <typename T> \
PE_HOST_DEVICE htm_t<T> from_euler_##dir##_rotation(T ang1, T ang2, T ang3, const vector_t<T, 3>& position) { \
  return htm_t<T>(from_euler_##name##_rotation(ang1, ang2, ang3, position)); \
} \

PE_HTM_FROM_ROT()

#undef PE_HTM_FROM_ROT1
#undef PE_HTM_FROM_ROT2
#undef PE_HTM_FROM_ROT3
#undef PE_HTM_FROM_ROT

PE_END

#endif /* B0AF69A6_577B_4BF9_AF88_BB09112AEE6E */
