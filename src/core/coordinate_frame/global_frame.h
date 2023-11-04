#ifndef B4E38C0D_2463_488C_A440_85CEF99DD6F0
#define B4E38C0D_2463_488C_A440_85CEF99DD6F0

// here the global frame is defined as a special case
// which is a singleton class

#include "frame.h"

PE_BEGIN

template <typename T, size_t ALIGNMENT>
struct GlobalFrame : public FrameBase<T, ALIGNMENT> {

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
    return htm_<T>();
  }

  PE_HOST_DEVICE virtual vector_t<T, 3> get_position() const override {
    return vector_t<T, 3>();
  }

  PE_HOST_DEVICE virtual quaternion_t<T> get_quaternion() const override {
    return quaternion_t<T>();
  }

  PE_HOST_DEVICE bool is_global(const FrameBase<T, ALIGNMENT>& frame) const {
    return this == &frame;
  }

  PE_HOST_DEVICE bool is_global(const Frame<T, ALIGNMENT>*& frame) const {
    return this == frame;
  }

};

PE_END





#endif /* B4E38C0D_2463_488C_A440_85CEF99DD6F0 */
