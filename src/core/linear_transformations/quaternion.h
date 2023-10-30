/*
@file: quaternion.h
@brief: Implementation of quaternions
@author: Deniz A. Acar
*/

#ifndef F0CEA5BD_D9AA_4065_B558_58BA5BEA3B2E
#define F0CEA5BD_D9AA_4065_B558_58BA5BEA3B2E

#include "../common/common.h"
#include "rotation_dyads.h"

PE_BEGIN

template <typename T, uint32_t ALIGNMENT=sizeof(T)>
struct quaternion_t {

  using value_type = T;
  PE_HOST_DEVICE quaternion_t() {
    quat = {0, 0, 0, 1};
  }

  PE_HOST_DEVICE quaternion_t(T w, T x, T y, T z) {
    quat.w = w;
    quat.x = x; 
    quat.y = y;
    quat.z = z;
  }

  PE_HOST_DEVICE quaternion_t(const matrix_t<T, 3, 3>& m) {
    // implemented from http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/index.htm
    T tr = m(0, 0) + m(1, 1) + m(2, 2);

    if (tr > 0) {
      T S = sqrt(tr + 1.0) * 2; // S=4*qw
      quat.w = 0.25 * S;
      quat.x = (m(2, 1) - m(1, 2)) / S;
      quat.y = (m(0, 2) - m(2, 0)) / S; 
      quat.z = (m(1, 0) - m(0, 1)) / S; 
    } else if ((m(0,0) > m(1,1)) && (m(0,0) > m(2,2))) {
      T S = sqrt(1.0 + m(0,0) - m(1,1) - m(2,2)) * 2; // S=4*qx 
      quat.w = (m(2,1) - m(1,2)) / S;
      quat.x = 0.25 * S;
      quat.y = (m(0,1) + m(1,0)) / S; 
      quat.z = (m(0,2) + m(2,0)) / S;
    } else if (m(1, 1) > m(2, 2)) {
      T S = sqrt(1.0 + m(1,1) - m(0,0) - m(2,2)) * 2; // S=4*qy
      quat.w = (m(0,2) - m(2,0)) / S;
      quat.x = (m(0,1) + m(1,0)) / S; 
      quat.y = 0.25 * S;
      quat.z = (m(1,2) + m(2,1)) / S; 
    } else {
  float S = sqrt(1.0 + m(2,2) - m(0,0) - m(1,1)) * 2; // S=4*qz
      quat.w = (m(1,0) - m(0,1)) / S;
      quat.x = (m(0,2) + m(2,0)) / S;
      quat.y = (m(1,2) + m(2,1)) / S;
      quat.z = 0.25 * S;
    }
  }

  PE_HOST_DEVICE quaternion_t(const matrix_t<T, 4, 4>& m) {
    // implemented from http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/index.htm
    T tr = m(0, 0) + m(1, 1) + m(2, 2);

    if (tr > 0) {
      T S = sqrt(tr + 1.0) * 2; // S=4*qw
      quat.w = 0.25 * S;
      quat.x = (m(2, 1) - m(1, 2)) / S;
      quat.y = (m(0, 2) - m(2, 0)) / S; 
      quat.z = (m(1, 0) - m(0, 1)) / S; 
    } else if ((m(0,0) > m(1,1)) && (m(0,0) > m(2,2))) {
      T S = sqrt(1.0 + m(0,0) - m(1,1) - m(2,2)) * 2; // S=4*qx 
      quat.w = (m(2,1) - m(1,2)) / S;
      quat.x = 0.25 * S;
      quat.y = (m(0,1) + m(1,0)) / S; 
      quat.z = (m(0,2) + m(2,0)) / S;
    } else if (m(1, 1) > m(2, 2)) {
      T S = sqrt(1.0 + m(1,1) - m(0,0) - m(2,2)) * 2; // S=4*qy
      quat.w = (m(0,2) - m(2,0)) / S;
      quat.x = (m(0,1) + m(1,0)) / S; 
      quat.y = 0.25 * S;
      quat.z = (m(1,2) + m(2,1)) / S; 
    } else {
  float S = sqrt(1.0 + m(2,2) - m(0,0) - m(1,1)) * 2; // S=4*qz
      quat.w = (m(1,0) - m(0,1)) / S;
      quat.x = (m(0,2) + m(2,0)) / S;
      quat.y = (m(1,2) + m(2,1)) / S;
      quat.z = 0.25 * S;
    }
  }

  PE_HOST_DEVICE quaternion_t(T heading, T attitude, T bank) {
    // implemented from http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/index.htm
    // Assuming the angles are in radians.
    T c1 = cos(heading);
    T s1 = sin(heading);
    T c2 = cos(attitude);
    T s2 = sin(attitude);
    T c3 = cos(bank);
    T s3 = sin(bank);
    quat.w = sqrt(1.0 + c1 * c2 + c1*c3 - s1 * s2 * s3 + c2*c3) / 2.0;
    T w4 = (4.0 * quat.w);
    quat.x = (c2 * s3 + c1 * s3 + s1 * s2 * c3) / w4 ;
    quat.y = (s1 * c2 + s1 * c3 + c1 * s2 * s3) / w4 ;
    quat.z = (-s1 * s3 + c1 * s2 * c3 +s2) / w4 ;
  }

  PE_HOST_DEVICE quaternion_t(const vector_t<T, 3>& n, T angle) {
    // implemented from http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/index.htm
    // Assuming the axis is already normalized.
    T halfAngle = angle / 2.0;
    T s = sin(halfAngle);
    quat.w = cos(halfAngle);
    quat.x = n.x * s;
    quat.y = n.y * s;
    quat.z = n.z * s;
  }

  PE_HOST_DEVICE vector_t<T, 4> to_axisangle() {
    // implemented from http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/index.htm
    // Assuming the quaternion is normalized.
    if (quat.w > 1) normalise(quat);
    vector_t<T, 4> aa;
    T angle = 2 * acos(quat.w);
    T s = sqrt(1 - quat.w * quat.w);
    if (s < 0.001) { // test to avoid divide by zero, s is always positive due to sqrt
      // if s close to zero then direction of axis not important
      aa.x = quat.x; // if it is important that axis is normalised then replace with x=1; y=z=0;
      aa.y = quat.y;
      aa.z = quat.z;
    } else {
      aa.x = quat.x / s; // normalise axis
      aa.y = quat.y / s;
      aa.z = quat.z / s;
    }
    aa.w = angle;
    return aa;
  }

  PE_HOST_DEVICE vector_t<T, 3> to_euler() {
    // implemented from http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToEuler/index.htm
    // Assuming the quaternion is normalized.
    quaternion_t q1 = normalize(quat);
    T test = q1.x*q1.y + q1.z*q1.w;
    T heading, attitude, bank;
    if (test > 0.499) { // singularity at north pole
      heading = 2 * atan2(q1.x,q1.w);
      attitude = pi/2;
      bank = 0;
      return {heading, attitude, bank};
    }
    if (test < -0.499) { // singularity at south pole
      heading = -2 * atan2(q1.x,q1.w);
      attitude = - pi/2;
      bank = 0;
      return {heading, attitude, bank};
    }
    T sqx = q1.x*q1.x;
    T sqy = q1.y*q1.y;
    T sqz = q1.z*q1.z;
    heading = atan2(2*q1.y*q1.w-2*q1.x*q1.z , 1 - 2*sqy - 2*sqz);
    attitude = asin(2*test);
    bank = atan2(2*q1.x*q1.w-2*q1.y*q1.z , 1 - 2*sqx - 2*sqz);
    return {heading, attitude, bank};
  }

  PE_HOST_DEVICE matrix_t<T, 3, 3> to_matrix() {
    // implemented from http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/index.htm
    // Assuming the quaternion is normalized.
    T sqw = quat.w*quat.w;
    T sqx = quat.x*quat.x;
    T sqy = quat.y*quat.y;
    T sqz = quat.z*quat.z;

    matrix_t<T, 3, 3> m;
    
    // invs (inverse square length) is only required if quaternion is not already normalised
    T invs = 1 / (sqx + sqy + sqz + sqw);
    m(0, 0) = ( sqx - sqy - sqz + sqw)*invs ; // since sqw + sqx + sqy + sqz =1/invs*invs
    m(1, 1) = (-sqx + sqy - sqz + sqw)*invs ;
    m(2, 2) = (-sqx - sqy + sqz + sqw)*invs ;
    
    T tmp1 = quat.x*quat.y;
    T tmp2 = quat.z*quat.w;
    m(1, 0) = 2.0 * (tmp1 + tmp2)*invs ;
    m(0, 1) = 2.0 * (tmp1 - tmp2)*invs ;
    
    tmp1 = quat.x*quat.z;
    tmp2 = quat.y*quat.w;
    m(2, 0) = 2.0 * (tmp1 - tmp2)*invs ;
    m(0, 2) = 2.0 * (tmp1 + tmp2)*invs ;
    tmp1 = quat.y*quat.z;
    tmp2 = quat.x*quat.w;
    m(2, 1) = 2.0 * (tmp1 + tmp2)*invs ;
    m(1, 2) = 2.0 * (tmp1 - tmp2)*invs ;
    return m;
  }

  PE_HOST_DEVICE void print() const {
  #if defined(__CUDA_ARCH__) 
    printf("w: %f, x: %f, y: %f, z: %f\n", quat.w, quat.x, quat.y, quat.z);
  #else
    std::cout << "w: " << quat.w << ", x: " << quat.x << ", y: " << quat.y << ", z: " << quat.z << "\n";
  #endif
  }

  vector_t<T, 4, ALIGNMENT> quat;
};

template <typename T>
PE_HOST_DEVICE void print(const quaternion_t<T>& q) {
  q.print();
}

template <typename T>
PE_HOST_DEVICE inline quaternion_t<T> normalize(const quaternion_t<T>& q) {
  T norm = sqrt(q.quat.w * q.quat.w + q.quat.x * q.quat.x + q.quat.y * q.quat.y + q.quat.z * q.quat.z);
  return {q.quat.w / norm, q.quat.x / norm, q.quat.y / norm, q.quat.z / norm};
}

PE_END

#endif /* F0CEA5BD_D9AA_4065_B558_58BA5BEA3B2E */
