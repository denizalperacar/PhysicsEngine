#ifndef A07E53E3_272E_4C66_99FA_D3DE84481CC0
#define A07E53E3_272E_4C66_99FA_D3DE84481CC0


#include "../common/pch.h"

PE_BEGIN

template <typename T, size_t ALIGNMENT>
struct AbsoluteFrame : public Frame<T, ALIGNMENT> {

  PE_HOST_DEVICE AbsoluteFrame(vector_t<T, 3> position, quaternion_t<T> quaternion) {
    htm = htm_t<T>(position, quaternion);
  }

  PE_HOST_DEVICE AbsoluteFrame(htm_t<T> htm) {
    this->htm = htm;
  }

  PE_HOST_DEVICE AbsoluteFrame(vector_t<T, 3> position, matrix_t<T, 3, 3> dcm) {
    htm = htm_t<T>(dcm, position);
  }

  PE_HOST_DEVICE AbsoluteFrame() {
    htm = htm_t<T>();
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

  PE_HOST_DEVICE virtual htm_t<T> resolve_frame_in_global() const override {
    return htm;
  }

  PE_HOST_DEVICE virtual void set_htm(const htm_t<T>& htm) override {
    this->htm = htm;
  }

  PE_HOST_DEVICE virtual void set_position(const vector_t<T, 3>& position) override {
    htm.set_position(position);
  }

  PE_HOST_DEVICE virtual void set_quaternion(const quaternion_t<T>& quaternion) override {
    htm.set_quaternion(quaternion);
  }

  PE_HOST_DEVICE virtual htm_t<T> operator*(const htm_t<T>& other) const override {
    return htm * other;
  }

  PE_HOST_DEVICE virtual htm_t<T> inverted() const override {
    return htm.inverted();
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

  PE_HOST_DEVICE virtual htm_t<T> operator()(const FrameBase<T, ALIGNMENT>& frame) const {
    return frame.get_htm().inverted() * htm;
  }

  PE_HOST_DEVICE virtual htm_t<T> operator()(const htm_t<T>& htm) const {
    return htm.inverted() * htm;
  }

  PE_HOST_DEVICE virtual AbstractAbsoluteFrame<T, ALIGNMENT>& operator=(const htm_t<T>& htm) {
    this->htm = htm;
    return *this;
  }

  htm_t<T> htm;

};


PE_END

#endif /* A07E53E3_272E_4C66_99FA_D3DE84481CC0 */
