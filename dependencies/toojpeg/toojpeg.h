#ifndef DE61610E_27A3_452F_A761_BD29BB1BC142
#define DE61610E_27A3_452F_A761_BD29BB1BC142

// //////////////////////////////////////////////////////////
// toojpeg.h
// written by Stephan Brumme, 2018-2019
// see https://create.stephan-brumme.com/toojpeg/
//
// This is a compact baseline JPEG/JFIF writer, written in C++ (but looks like C for the most part).
// Its interface has only one function: writeJpeg() - and that's it !
//
// basic example:
// => create an image with any content you like, e.g. 1024x768, RGB = 3 bytes per pixel
// auto pixels = new unsigned char[1024*768*3];
// => you need to define a callback that receives the compressed data byte-by-byte from my JPEG writer
// void myOutput(unsigned char oneByte) { fputc(oneByte, myFileHandle); } // save byte to file
// => let's go !
// TooJpeg::writeJpeg(myOutput, mypixels, 1024, 768);
#pragma once

#include <fstream>

namespace TooJpeg
{
  // write one byte (to disk, memory, ...)
  typedef void (*WRITE_ONE_BYTE)(unsigned char);

  // this callback is called for every byte generated by the encoder and behaves similar to fputc
  // if you prefer stylish C++11 syntax then it can be a lambda, too:
  // auto myOutput = [](unsigned char oneByte) { fputc(oneByte, output); };
  // output       - callback that stores a single byte (writes to disk, memory, ...)
  // pixels       - stored in RGB format or grayscale, stored from upper-left to lower-right
  // width,height - image size
  // isRGB        - true if RGB format (3 bytes per pixel); false if grayscale (1 byte per pixel)
  // quality      - between 1 (worst) and 100 (best)
  // downsample   - if true then YCbCr 4:2:0 format is used (smaller size, minor quality loss) instead of 4:4:4, not relevant for grayscale
  // comment      - optional JPEG comment (0/NULL if no comment), must not contain ASCII code 0xFF
  bool writeJpeg(WRITE_ONE_BYTE output, const void* pixels, unsigned short width, unsigned short height,
    bool isRGB = true, unsigned char quality = 90, bool downsample = false, const char* comment = nullptr);

} // namespace TooJpeg
// My main inspiration was Jon Olick's Minimalistic JPEG writer
// ( https://www.jonolick.com/code.html => direct link is https://www.jonolick.com/uploads/7/9/2/1/7921194/jo_jpeg.cpp ).
// However, his code documentation is quite sparse - probably because it wasn't written from scratch and is (quote:) "based on a javascript jpeg writer",
// most likely Andreas Ritter's code: https://github.com/eugeneware/jpeg-js/blob/master/lib/encoder.js
//
// Therefore I wrote the whole lib from scratch and tried hard to add tons of comments to my code, especially describing where all those magic numbers come from.
// And I managed to remove the need for any external includes ...
// yes, that's right: my library has no (!) includes at all, not even #include <stdlib.h>
// Depending on your callback WRITE_ONE_BYTE, the library writes either to disk, or in-memory, or wherever you wish.
// Moreover, no dynamic memory allocations are performed, just a few bytes on the stack.
//
// In contrast to Jon's code, compression can be significantly improved in many use cases:
// a) grayscale JPEG images need just a single Y channel, no need to save the superfluous Cb + Cr channels
// b) YCbCr 4:2:0 downsampling is often about 20% more efficient (=smaller) than the default YCbCr 4:4:4 with only little visual loss
//
// TooJpeg 1.2+ compresses about twice as fast as jo_jpeg (and about half as fast as libjpeg-turbo).
// A few benchmark numbers can be found on my website https://create.stephan-brumme.com/toojpeg/#benchmark
//
// Last but not least you can optionally add a JPEG comment.
//
// Your C++ compiler needs to support a reasonable subset of C++11 (g++ 4.7 or Visual C++ 2013 are sufficient).
// I haven't tested the code on big-endian systems or anything that smells like an apple.
//
// USE AT YOUR OWN RISK. Because you are a brave soul :-)

#endif /* DE61610E_27A3_452F_A761_BD29BB1BC142 */
