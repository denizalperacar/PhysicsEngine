#ifndef C8D185FE_FC90_4372_85E7_649049D9DA4B
#define C8D185FE_FC90_4372_85E7_649049D9DA4B

#include "../common/pch.h"

PE_BEGIN

template <typename T, size_t ALIGNMENT=sizeof(T)>
struct RelativeFrame : public AbstractRelativeFrame<T, ALIGNMENT> {

  PE_HOST_DEVICE RelativeFrame(
      Frame<T, ALIGNMENT> *parent, 
      const vector_t<T, 3>& position, 
      const quaternion_t<T>& quaternion
  ) {  
    this->parent = parent;
    htm = htm_t<T>(quaternion, position);
  }

  PE_HOST_DEVICE RelativeFrame(
      Frame<T, ALIGNMENT> *parent_pointer,  
      const htm_t<T>& input_htm
  ) {  
    this->parent = parent_pointer;
    this->htm = input_htm;
  }

  PE_HOST_DEVICE virtual htm_t<T> get_htm() const override {
    return htm;
  }

  PE_HOST_DEVICE virtual vector_t<T, 3> get_position() const override {
    vector_t<T, 3> pos;
    PE_UNROLL
    for (int i = 0; i < 3; i++) {
      pos[i] = this->position[i];
    }
    return pos;
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
    htm.set_from_quaternion(quaternion);
  }

  PE_HOST_DEVICE virtual void set_parent(const Frame<T, ALIGNMENT>& frame) override {
    parent = &frame;
  }

  PE_HOST_DEVICE virtual void set_parent(Frame<T, ALIGNMENT>* parent) override {
    this->parent = parent;
  }

  PE_HOST_DEVICE virtual void set_parent(Frame<T, ALIGNMENT>& parent) override {
    this->parent = &parent;
  }

  PE_HOST_DEVICE virtual Frame<T, ALIGNMENT>* get_parent() const override {
    return parent;
  }

  PE_HOST_DEVICE virtual Frame<T, ALIGNMENT> operator*(const Frame<T, ALIGNMENT>& frame) const override {
    return Frame<T, ALIGNMENT>(htm * frame.htm);
  }

  PE_HOST_DEVICE virtual Frame<T, ALIGNMENT> operator*(const htm_t<T>& htm_representation) const override {
    return Frame<T, ALIGNMENT>(htm * htm_representation);
  }

  PE_HOST_DEVICE virtual Frame<T, ALIGNMENT> inverted() const override {
    return Frame<T, ALIGNMENT>(htm.inverted());
  }

  PE_HOST_DEVICE virtual void invert() override {
    htm.invert();
  }

  PE_HOST_DEVICE virtual vector_t<T, 3> resolve_in_frame(const vector_t<T, 3>& vec) const {
    return htm * vec;
  }
  PE_HOST_DEVICE virtual vector_t<T, 4> resolve_in_frame(const vector_t<T, 4>& vec) const {
    return htm * vec;
  }

  PE_HOST_DEVICE virtual htm_t<T> resolve_in_parent() const override {
    return parent->htm * htm;
  }

  PE_HOST_DEVICE virtual vector_t<T, 3> resolve_in_parent(const vector_t<T, 3>& vec) const {
    return (parent->htm * htm) * vec;
  }

  PE_HOST_DEVICE virtual vector_t<T, 4> resolve_in_parent(const vector_t<T, 4>& vec) const {
    return (parent->htm * htm) * vec;
  }

  // checks if the address of the supported frame is the same as the given frame
  PE_HOST_DEVICE bool operator==(const Frame<T, ALIGNMENT>& frame) override {
    return this == &frame;
  }

  PE_HOST_DEVICE virtual AbsoluteFrame<T, ALIGNMENT> resolve_frame_in_global() const {
    htm_t<T> result_htm = htm;
    GlobalFrame<T, ALIGNMENT> &global = GlobalFrame<T, ALIGNMENT>::get_instance();
    if (global.is_global(parent) && parent != nullptr) {
      return parent->resolve_frame_in_global(global) * result_htm;
    }
    return result_htm;
  }

  PE_HOST_DEVICE virtual Frame<T, ALIGNMENT> operator()() const {
    return resolve_frame_in_global();
  }

  PE_HOST_DEVICE virtual AbstractRelativeFrame<T, ALIGNMENT>& operator=(const htm_t<T>& htm) {
    this->htm = htm;
    return *this;
  }

  htm_t<T> htm;
  Frame<T, ALIGNMENT> *parent = nullptr;
};



PE_END


#endif /* C8D185FE_FC90_4372_85E7_649049D9DA4B */
