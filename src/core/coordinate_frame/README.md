# COORDINATE FRAME

## Abstract Frame Class

This abstract class defines the interface for all frames. Here it is assumed that the any vector in space is resolved as a column vector thus the htm vector multiplication is defined as:

[1] v' = htm * v

where v is a column vector and v' is the transformed vector.

The second assumption that is required or somewhat enforced is that any frame defined must be relative to another frame. This is enforced by removing the default constructor and requiring the constructors to specify a parent Frame.

The second assumption is somehow not efficinet specially when using cuda as some calculations might require fetching information from far apart memory locations. As a result two types of abstract frames 
are defined inheriting the Frame class which are the Absolute Frame and Relative Frame classes.

Here two types of coordinate frames are implemented. The first type is a stationary frame and the second is moving frame.

## global frame

global frame is a singleton which will be used inside the RelativeFrame child classes to stop recursive query to be resolved in global frame.

## Stationary frame (SF)

The stationary frame contains only position and orientation. The position of the frame corresponds to the origin of the frame and the orientation is represented as either quaternions, Euler angles or transformation matrices.

Represnting a frame as a hemogenious transformation matrix is also a good idea as only 3 or 4 vector_t<T, 4> are required to be stored.

[ ] The storage decision of the frames is postponed to a later time.
[ ] Store the parent information in the SF.

## Moving Frame (MF)

The moving frame contains 