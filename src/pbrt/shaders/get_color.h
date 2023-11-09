#ifndef CA42C68D_85F7_4DD0_AF1C_5A80DDD44CEC
#define CA42C68D_85F7_4DD0_AF1C_5A80DDD44CEC

#include "pch.h"
#include "color_render.h"

PE_BEGIN

template <typename T>
PE_HOST_DEVICE render_color get_color(vector_t<T, 3>& pixel_color, int samples_per_pixel) {
	// writes the mapped [0, 255] value of each colow componenet
  // TODO reimplement a more efficient version of this
	T r = pixel_color.r;
	T g = pixel_color.g;
	T b = pixel_color.b;

	// scale the color by the number of samples sent to the sensor
	T scale = 1.0f / samples_per_pixel;
	r *= scale;
	g *= scale;
	b *= scale;

	render_color c;

	c.r = (uint8_t)(255.999 * r);
	c.g = (uint8_t)(255.999 * g);
	c.b = (uint8_t)(255.999 * b);

	return c;
	
}

PE_END

#endif /* CA42C68D_85F7_4DD0_AF1C_5A80DDD44CEC */
