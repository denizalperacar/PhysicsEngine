/*
@brief: Implementation of relative reference frames
@file: relative_frame.h
@author: Deniz A. ACAR
*/


#ifndef C8D185FE_FC90_4372_85E7_649049D9DA4B
#define C8D185FE_FC90_4372_85E7_649049D9DA4B

#include "../common/pch.h"

PE_BEGIN

template <typename T, size_t ALIGNMENT=sizeof(T)>
struct RelativeFrame : public AbstractRelativeFrame<T, ALIGNMENT> {

  PE_HOST_DEVICE RelativeFrame(
      Frame<T, ALIGNMENT>* parent, 
      const vector_t<T, 3>& position, 
      const quaternion_t<T>& quaternion
  ) {  
    this->parent = parent;
    htm = htm_t<T>(quaternion, position);
  }

  PE_HOST_DEVICE RelativeFrame(
      FrameBase<T, ALIGNMENT>* parent, 
      const vector_t<T, 3>& position, 
      const quaternion_t<T>& quaternion
  ) {  
    this->parent = parent;
    htm = htm_t<T>(quaternion, position);
  }

  PE_HOST_DEVICE RelativeFrame(
      FrameBase<T, ALIGNMENT> *parent_pointer,  
      const htm_t<T>& input_htm
  ) {  
    this->parent = parent_pointer;
    this->htm = input_htm;
  }

  PE_HOST_DEVICE RelativeFrame(
      Frame<T, ALIGNMENT> *parent_pointer,  
      const htm_t<T>& input_htm
  ) {  
    this->parent = parent_pointer;
    this->htm = input_htm;
  }

  PE_HOST_DEVICE RelativeFrame(
      const vector_t<T, 3>& position, 
      const quaternion_t<T>& quaternion
  ) {  
    this->parent = GlobalFrame<T, ALIGNMENT>::get_instance();
    htm = htm_t<T>(quaternion, position);
  }

  PE_HOST_DEVICE virtual htm_t<T> get_htm() const override {
    return htm;
  }

  PE_HOST_DEVICE virtual vector_t<T, 3> get_position() const override {
    return htm.get_position();
  }

  PE_HOST_DEVICE virtual quaternion_t<T> get_quaternion() const override {
    return htm.get_quaternion();
  }

  PE_HOST_DEVICE virtual void set_htm(const htm_t<T>& input_htm) override {
    htm = input_htm;
  }

  PE_HOST_DEVICE virtual void set_position(const vector_t<T, 3>& position) override {
    htm.set_position(position);
  }

  PE_HOST_DEVICE virtual void set_quaternion(const quaternion_t<T>& quaternion) override {
    htm = htm_t(quaternion);
  }

  PE_HOST_DEVICE virtual void set_parent(FrameBase<T, ALIGNMENT>*& parent) override {
    this->parent = parent;
  }

  PE_HOST_DEVICE virtual FrameBase<T, ALIGNMENT>* get_parent() const override {
    return parent;
  }

  PE_HOST_DEVICE virtual htm_t<T> operator*(const htm_t<T>& htm_representation) const override {
    return htm * htm_representation;
  }

  PE_HOST_DEVICE virtual htm_t<T> inverted() const override {
    return htm.get_inverse();
  }

  PE_HOST_DEVICE virtual void invert() override {
    htm.invert();
  }

  PE_HOST_DEVICE virtual vector_t<T, 3> resolve_in_frame(const vector_t<T, 3>& vec) const override {
    return htm * vec;
  }
  PE_HOST_DEVICE virtual vector_t<T, 4> resolve_in_frame(const vector_t<T, 4>& vec) const override {
    return htm * vec;
  }

  PE_HOST_DEVICE virtual htm_t<T> resolve_in_parent() const override {
    return parent->get_htm() * htm;
  }

  PE_HOST_DEVICE virtual vector_t<T, 3> resolve_in_parent(const vector_t<T, 3>& vec) const override {
    return (parent->get_htm() * htm) * vec;
  }

  PE_HOST_DEVICE virtual vector_t<T, 4> resolve_in_parent(const vector_t<T, 4>& vec) const override {
    return (parent->get_htm() * htm) * vec;
  }

  // checks if the address of the supported frame is the same as the given frame
  PE_HOST_DEVICE bool operator==(const Frame<T, ALIGNMENT>& frame) override {
    return this == &frame;
  }

  PE_HOST_DEVICE virtual htm_t<T> resolve_frame_in_global() const override {
    htm_t<T> result_htm = htm; 
    if (parent != &GlobalFrame<T, ALIGNMENT>::get_instance() and parent != nullptr) {
      result_htm = parent->resolve_frame_in_global() * result_htm;
      htm.print();
    }
    return result_htm;
  }

  PE_HOST_DEVICE virtual htm_t<T>operator()() const {
    return resolve_frame_in_global();
  }

  PE_HOST_DEVICE virtual AbstractRelativeFrame<T, ALIGNMENT>& operator=(const htm_t<T>& htm) {
    this->htm = htm;
    return *this;
  }

  PE_HOST_DEVICE void print() {
    htm.print();
  }

  htm_t<T> htm;
  FrameBase<T, ALIGNMENT> *parent = &GlobalFrame<T, ALIGNMENT>::get_instance();
};

template <typename T, size_t ALIGNMENT=sizeof(T)>
void print(const RelativeFrame<T, ALIGNMENT>& frame) {
  print(frame.htm);
} 

PE_END


#endif /* C8D185FE_FC90_4372_85E7_649049D9DA4B */
