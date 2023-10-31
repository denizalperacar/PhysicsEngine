#ifndef B4E38C0D_2463_488C_A440_85CEF99DD6F0
#define B4E38C0D_2463_488C_A440_85CEF99DD6F0

// here the global frame is defined as a special case
// which is a singleton class

#include "frame.h"

PE_BEGIN

template <typename T, uint32_t ALIGNMENT>
struct GlobalFrame : public Frame<T, ALIGNMENT> {

static GlobalFrame<T, ALIGNMENT> instance;


static GlobalFrame<T, ALIGNMENT>& get_instance() {
  static GlobalFrame<T, ALIGNMENT> instance;
  return instance;
}

};




PE_END





#endif /* B4E38C0D_2463_488C_A440_85CEF99DD6F0 */
