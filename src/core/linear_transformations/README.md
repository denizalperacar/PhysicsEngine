# TO DO

# Advanced Dynamics code

## Linear Transformations

- [X] dot product

- [x] cross product, cross product matrix, skew symmetric matrix

- [x] determinant

- [x] outer product of vectors

- [x] sqrt matrix

- [x] matrix exponent

- [x] matrix log

- [x] rotation matrix

- [x] inverse matrix

- [x] roudrigues formula for rotation dyad implementation

- [x] implement different sets of euler rotation sequences

- [x] implement quaternions

- [ ] implement hemogenious transformation matrices (htm)

- [x] implement from euler for htm

- [ ] htm from n and theta

- [x] htm from quaternion

- [X] htm to quaternion

- [X] add unit vector return from the matrix

- [X] htm to dcm

- [X] add vector resolution to htm

- [X] htm to euler

- [ ] implement matrix slice accessors

- [X] implement the inverse of transformation matrix i.e. get n and theta (page 24)

- [X] invert transofmation matrices

- [ ] test the matrix to angle axis // the indices might be just the opposite

- [X] add print method for all the classes in the linear tranformation folder

- [X] add quaternion multiplication

- [X] add quaternion inversion

- [ ] construct dcm from 2 vectors


## PBRT Chapter3 

In this chapter authors write a Tuple class which accepts a template template parameter then creat a distinct type 
for different entities that have the same base class. 

For the vector_base here we created a general purpose vector that stores any number of variables. I have decided to create
a second vector_base that is based on both definitions and extend that. The reason is the fact that the same concepts will 
be helpful in the creation of the multi body dynamics engine. Yet another issue is that they might restrict and limit the 
operations that can be performed but that is something that I will see later when I am designing the multi-body engine.

At least having two different vector bases allows me decrease the rebase time when I decide on what I am going to use.

