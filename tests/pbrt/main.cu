#include "pch.h"
#include "../../src/pbrt/rendering/basic_renderer.cuh"
#include "../../src/pbrt/rendering/render_manager.cuh"



__global__ void fillTextureKernel(cudaSurfaceObject_t surfaceObj, int width, int height) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  // Your filling logic here
  uchar4 color = make_uchar4(255, 0, 0, 255);  // Red color, modify as needed

  surf2Dwrite(color, surfaceObj, x * sizeof(uchar4), y);
}

int main () {

  return PE::render_manager<float>();
}