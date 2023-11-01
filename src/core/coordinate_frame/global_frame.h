#ifndef B4E38C0D_2463_488C_A440_85CEF99DD6F0
#define B4E38C0D_2463_488C_A440_85CEF99DD6F0

// here the global frame is defined as a special case
// which is a singleton class

#include "frame.h"

PE_BEGIN

template <typename T, uint32_t ALIGNMENT>
struct GlobalFrame : public Frame<T, ALIGNMENT> {

  GlobalFrame() = delete;
  GlobalFrame(const GlobalFrame&) = delete;
  GlobalFrame& operator=(const GlobalFrame&) = delete;
  GlobalFrame(GlobalFrame&&) = delete;
  GlobalFrame& operator=(GlobalFrame&&) = delete;
  ~GlobalFrame() = default;

  static GlobalFrame<T, ALIGNMENT>& get_instance() {
    static GlobalFrame<T, ALIGNMENT> instance;
    return instance;
  }

  /*
  @brief: returns the identity htm
  @return: identity htm
  */
  PE_HOST_DEVICE htm_t<T> get_htm() const override {
    return htm_;
  }

  

  tamplate <uint32_t A>
  PE_HOST_DEVICE virtual void set_htm(const htm_t<T, A>& htm) override {
    htm_ = htm;
  }

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

private:
  htm_t<T, ALIGNMENT> htm_ = htm_t<T, ALIGNMENT>::identity();

};




PE_END





#endif /* B4E38C0D_2463_488C_A440_85CEF99DD6F0 */
