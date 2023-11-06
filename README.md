# PhysicsEngine

The implementation of this physics engine is influenced by the following libraries:
* tiny_cuda_nn.
* pbr
* project chrono

## Physical Rendering [Light physics]

### Current Milestones

- [ ] Implement the linear transformations

> - [X] implement vector definition

> - [X] Implementmatrix definition

> - [X] Implement quaternions

> - [X] Implement HTM

> - [X] Implement rotation dyads calculations

> - [ ] Test the code

- [X] Implement static reference frames

> - [X] implement tracing the frames to get their resolution in global frame

> - [ ] test the frame resolution in cuda for a large number of random frames

- [ ] Implement different camera models in PBRT

- [ ] Implement a simple ray tracing engine

> - [ ] combine the engine with the previous code

> - [ ] add some gui to control it.

> - [ ] add rendering result directly to gui.
