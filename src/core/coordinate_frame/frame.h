#ifndef C42C2054_3026_478C_82CD_E6BB0A8DD2DE
#define C42C2054_3026_478C_82CD_E6BB0A8DD2DE

/*
@brief: Implementation of reference frames (static and moving)
@file: frame.h
@author: Deniz A. ACAR
*/


#include <../common/pch.h>


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
template <typename T, uint32_t ALIGNMENT>
struct Frame {
  using value_type = T;
  
  // no default frame constructor
  Frame() = delete;
  PE_HOST_DEVICE htm_t<T> get_htm() const = 0;
  PE_HOST_DEVICE virtual void set_htm(const htm_t<T>& htm) = 0;
  PE_HOST_DEVICE virtual vector_t<T, 3> get_position() const = 0;
  PE_HOST_DEVICE virtual void set_position(const vector_t<T, 3>& position) = 0;
  PE_HOST_DEVICE virtual quaternion_t<T> get_quaternion() const = 0;
  PE_HOST_DEVICE virtual void set_quaternion(const quaternion_t<T>& quaternion) = 0;
  PE_HOST_DEVICE virtual Frame<T, ALIGNMENT> operator*(const Frame& frame) const = 0;
  PE_HOST_DEVICE virtual Frame<T, ALIGNMENT> operator*(const htm_t<T>& htm) const = 0;
  PE_HOST_DEVICE virtual Frame<T, ALIGNMENT> inverted() const = 0;
  PE_HOST_DEVICE virtual void invert() = 0;
  PE_HOST_DEVICE virtual Frame<T, ALIGNMENT> resolve_in_frame(const Frame<T, ALIGNMENT>& frame) const = 0;
  PE_HOST_DEVICE virtual Frame<T, ALIGNMENT> resolve_in_frame(const htm_t<T>& htm) const = 0;
  PE_HOST_DEVICE virtual Frame<T, ALIGNMENT> operator()(const Frame<T, ALIGNMENT>& frame) const = 0; // resolve in frame
  PE_HOST_DEVICE virtual Frame<T, ALIGNMENT> operator()(const htm_t<T>& htm) const = 0; // resolve in frame
  PE_HOST_DEVICE virtual Frame<T, ALIGNMENT> operator()() const = 0; // resolve in global frame
};

template <typename T, uint32_t ALIGNMENT>
struct AbstractRelativeFrame : public Frame<T, ALIGNMENT>{
  using value_type = T;
  PE_HOST_DEVICE virtual Frame<T, ALIGNMENT>* get_parent() const = 0;
  PE_HOST_DEVICE virtual void set_parent(Frame<T, ALIGNMENT>* parent) = 0;
  PE_HOST_DEVICE virtual void set_parent(Frame<T, ALIGNMENT>& parent) = 0;
  PE_HOST_DEVICE virtual bool operator==(const Frame<T, ALIGNMENT>& frame) = 0;
};

template <typename T, uint32_t ALIGNMENT>
struct AbsoluteFrame : public Frame<T, ALIGNMENT>{
  using value_type = T;
};

PE_END


#endif /* C42C2054_3026_478C_82CD_E6BB0A8DD2DE */


