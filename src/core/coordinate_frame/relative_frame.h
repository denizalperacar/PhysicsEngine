#ifndef C8D185FE_FC90_4372_85E7_649049D9DA4B
#define C8D185FE_FC90_4372_85E7_649049D9DA4B

#include "frame.h"

PE_BEGIN

template <typename T, uint32_t ALIGNMENT>
struct RelativeFrame : public AbstractRelativeFrame<T, ALIGNMENT> {

  PE_HOST_DEVICE RelativeFrame(
      Frame<T, ALIGNMENT> *parent, 
      const vector_t<T, 3>& position, 
      const quaternion_t<T>& quaternion
  ) {  
    this->parent = parent;
    PE_UNROLL
    for (int i = 0; i < 3; i++) {
      this->position[i] = position[i];
      this->quaternion[i] = quaternion[i];
    }
    this->quaternion[3] = quaternion[3];
  }

  

  PE_HOST_DEVICE virtual vector_t<T, 3> get_position() const override {
    vector_t<T, 3> pos;
    PE_UNROLL
    for (int i = 0; i < 3; i++) {
      pos[i] = this->position[i];
    }
    return pos;
  }

  PE_HOST_DEVICE virtual htm_t<T> get_htm() const override {
    return htm_t<T>(quaternion, get_position());
  }

  PE_HOST_DEVICE virtual void set_position(const vector_t<T, 3>& position) override {
    this->position = position;
  }

  PE_HOST_DEVICE virtual void set_htm(const htm_t<T>& htm) override {
    quaternion = htm.get_quaternion();
  }

  PE_HOST_DEVICE virtual quaternion_t<T> get_quaternion() const override {
    return quaternion;
  }

  PE_HOST_DEVICE virtual void set_quaternion(const quaternion_t<T>& quaternion) override {
    this->quaternion = quaternion;
  }

  // TODO : check the correctness of this function
  PE_HOST_DEVICE virtual Frame<T, ALIGNMENT> operator*(const Frame<T, ALIGNMENT>& frame) const override {
    return Frame<T, ALIGNMENT>(get_htm() * frame.get_htm());
  }

  // TODO : check the correctness of this function
  PE_HOST_DEVICE virtual Frame<T, ALIGNMENT> operator*(const htm_t<T>& htm) const override {
    return Frame<T, ALIGNMENT>(get_htm() * htm);
  }

  PE_HOST_DEVICE virtual Frame<T, ALIGNMENT> inverted() const override {
    return Frame<T, ALIGNMENT>(get_htm().inverted());
  }

  PE_HOST_DEVICE virtual void invert() override {
    set_htm(get_htm().inverted());
  }

  // TODO : check the correctness of this function
  PE_HOST_DEVICE virtual Frame<T, ALIGNMENT> resolve_in_frame(const Frame<T, ALIGNMENT>& frame) const override {
    return frame.inverted() * (*this);
  }

  // TODO : check the correctness of this function
  PE_HOST_DEVICE virtual Frame<T, ALIGNMENT> resolve_in_frame(const htm_t<T>& htm) const override {
    return htm.inverted() * (*this);
  }

  // TODO : implement this function
  PE_HOST_DEVICE virtual Frame<T, ALIGNMENT> operator()(const Frame<T, ALIGNMENT>& frame) const override {
    return Frame<T, ALIGNMENT>();
  }

  // TODO : implement this function
  PE_HOST_DEVICE virtual Frame<T, ALIGNMENT> operator()(const htm_t<T>& htm) const override {
    return Frame<T, ALIGNMENT>();
  }

  // TODO : implement this function
  PE_HOST_DEVICE virtual Frame<T, ALIGNMENT> operator()() const override {
    return Frame<T, ALIGNMENT>();
  }

  PE_HOST_DEVICE virtual Frame<T, ALIGNMENT>* get_parent() const override {
    return parent;
  }

  PE_HOST_DEVICE virtual void set_parent(Frame<T, ALIGNMENT>* parent) override {
    this->parent = parent;
  }

  PE_HOST_DEVICE virtual void set_parent(Frame<T, ALIGNMENT>& parent) override {
    this->parent = &parent;
  }

  // checks if the address of the supported frame is the same as the given frame
  PE_HOST_DEVICE bool operator==(const Frame<T, ALIGNMENT>& frame) override {
    return this == &frame;
  }

  htm_t<T> htm;
  Frame<T, ALIGNMENT> *parent;
};



PE_END


#endif /* C8D185FE_FC90_4372_85E7_649049D9DA4B */
