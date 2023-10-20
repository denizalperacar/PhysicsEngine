# COORDINATE FRAME

Here two types of coordinate frames are implemented. The first type is a stationary frame and the second is moving frame.

## Stationary frame (SF)

The stationary frame contains only position and orientation. The position of the frame corresponds to the origin of the frame and the orientation is represented as either quaternions, Euler angles or transformation matrices.

Represnting a frame as a hemogenious transformation matrix is also a good idea as only 3 or 4 vector_t<T, 4> are required to be stored.

[ ] The storage decision of the frames is postponed to a later time.
[ ] Store the parent information in the SF.

## Moving Frame (MF)

The moving frame contains 