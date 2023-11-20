#include "pch.h"
#include "../../src/pbrt/rendering/basic_renderer.cuh"
#include "../../src/pbrt/rendering/render_manager.cuh"

int main (int argc, char** argv) {

  return PE::render_manager<float>(argc, argv);
  
}