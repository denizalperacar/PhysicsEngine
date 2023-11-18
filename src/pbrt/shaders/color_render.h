#ifndef D5111990_4A35_4F99_B428_6BE050EA3504
#define D5111990_4A35_4F99_B428_6BE050EA3504

#include "common.h"
#include "vector_base.h"

PE_BEGIN

class render_color {
public:

	PE_HOST_DEVICE render_color() : r(0), g(0), b(0) {}

	PE_HOST_DEVICE render_color(uint8_t red, uint8_t green, uint8_t blue)
		: r(red), g(green), b(blue) {}

	PE_HOST std::ofstream& draw(std::ofstream& os) {
		os << r << " " << g << " " << b << "\n";
		return os;
	}

	PE_HOST_DEVICE void print() {
		printf("%d %d %d\n", r, g, b);
	}

public:
		uint8_t r;
		uint8_t g;
		uint8_t b;
};

std::ostream& operator<<(std::ostream& os, render_color obj) {
	os << +obj.r << " " << +obj.g << " " << +obj.b << "\n";
	return os;
}

std::fstream& operator<<(std::fstream& os, render_color obj) {
	os << +obj.r << " " << +obj.g << " " << +obj.b << "\n";
	return os;
}

PE_END

#endif /* D5111990_4A35_4F99_B428_6BE050EA3504 */
