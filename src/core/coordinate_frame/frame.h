#ifndef C42C2054_3026_478C_82CD_E6BB0A8DD2DE
#define C42C2054_3026_478C_82CD_E6BB0A8DD2DE

/*
@brief: Implementation of reference frames (static and moving)
@file: frame.h
@author: Deniz A. ACAR
*/


#include "../linear_transformations/homogeneous_transformation_matrix.h"


PE_BEGIN


/*
This abstract class defines the interface for all frames. 
Here it is assumed that the any vector in space is resolved 
as a column vector thus the htm vector multiplication is defined as:
[1] v' = htm * v
where v is a column vector and v' is the transformed vector.
The second assumption that is required or somewhat enforced is that any 
frame defined must be relative to another frame. This is enforced by 
removing the default constructor and requiring the constructors to 
specify a parent Frame.

The second assumption is somehow not efficinet specially when using 
cuda as some calculations might require fetching information from f
ar apart memory locations. As a result two types of abstract frames 
are defined inheriting the Frame class which are the Absolute Frame 
and Relative Frame classes.

Here two types of coordinate frames are implemented. The first type 
is a stationary frame and the second is moving frame.
*/
template <typename T, size_t ALIGNMENT=sizeof(T)>
struct FrameBase {
  using value_type = T;
  
  // no default frame constructor
  PE_HOST_DEVICE virtual htm_t<T> get_htm() const = 0;
  PE_HOST_DEVICE virtual vector_t<T, 3> get_position() const = 0;
  PE_HOST_DEVICE virtual quaternion_t<T> get_quaternion() const = 0;
};

template <typename T, size_t ALIGNMENT>
struct Frame : public FrameBase<T, ALIGNMENT> {
  using value_type = T;
  PE_HOST_DEVICE virtual void set_htm(const htm_t<T>& htm) = 0;
  PE_HOST_DEVICE virtual void set_position(const vector_t<T, 3>& position) = 0;
  PE_HOST_DEVICE virtual void set_quaternion(const quaternion_t<T>& quaternion) = 0;
  PE_HOST_DEVICE virtual htm_t<T> operator*(const htm_t<T>& htm) const = 0;
  PE_HOST_DEVICE virtual htm_t<T> inverted() const = 0;
  PE_HOST_DEVICE virtual void invert() = 0;
  PE_HOST_DEVICE virtual vector_t<T, 3> resolve_in_frame(const vector_t<T, 3>& vec) const = 0;
  PE_HOST_DEVICE virtual vector_t<T, 4> resolve_in_frame(const vector_t<T, 4>& vec) const = 0;
  PE_HOST_DEVICE virtual void normalize_dcm() = 0;
};

template <typename T, size_t ALIGNMENT>
struct AbstractAbsoluteFrame : public Frame<T, ALIGNMENT>{
  using value_type = T;
  PE_HOST_DEVICE virtual Frame<T, ALIGNMENT> operator()(const Frame<T, ALIGNMENT>& frame) const = 0; // resolve in frame
  PE_HOST_DEVICE virtual Frame<T, ALIGNMENT> operator()(const htm_t<T>& htm) const = 0; // resolve in frame
  PE_HOST_DEVICE virtual AbstractAbsoluteFrame<T, ALIGNMENT>& operator=(const htm_t<T>& htm) = 0;

};

// Forward declare the concrete Absolute Frame class
template <typename T, size_t ALIGNMENT=sizeof(T)> 
struct AbsoluteFrame : public AbstractAbsoluteFrame<T, ALIGNMENT> {};

template <typename T, size_t ALIGNMENT>
struct AbstractRelativeFrame : public Frame<T, ALIGNMENT>{
  using value_type = T;
  PE_HOST_DEVICE virtual FrameBase<T, ALIGNMENT>* get_parent() const = 0;
  PE_HOST_DEVICE virtual void set_parent(Frame<T, ALIGNMENT>* parent) = 0;
  PE_HOST_DEVICE virtual void set_parent(Frame<T, ALIGNMENT>& parent) = 0;
  PE_HOST_DEVICE virtual void set_parent(const Frame<T, ALIGNMENT>& frame) = 0;
  PE_HOST_DEVICE virtual htm_t<T> resolve_in_parent() const = 0;
  PE_HOST_DEVICE virtual vector_t<T, 3> resolve_in_parent(const vector_t<T, 3>& vec) const = 0;
  PE_HOST_DEVICE virtual vector_t<T, 4> resolve_in_parent(const vector_t<T, 4>& vec) const = 0;
  PE_HOST_DEVICE virtual bool operator==(const Frame<T, ALIGNMENT>& frame) = 0;
  PE_HOST_DEVICE virtual AbsoluteFrame<T, ALIGNMENT> resolve_frame_in_global() const = 0;
  PE_HOST_DEVICE virtual Frame<T, ALIGNMENT> operator()() const = 0; // resolve in global frame

  PE_HOST_DEVICE virtual AbstractRelativeFrame<T, ALIGNMENT>& operator=(const htm_t<T>& htm) = 0;
};

PE_END


#endif /* C42C2054_3026_478C_82CD_E6BB0A8DD2DE */


