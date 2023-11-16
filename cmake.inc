option(USE_CUDA "Use CUDA" ON)
find_package(fmt REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOURCE_DIR "${ROOT_DIR}/src/")
set(CORE_DIR   "${SOURCE_DIR}/core/")
set(COMMON_DIR "${CORE_DIR}/common/")
set(VECTOR_DIR "${CORE_DIR}/linear_transformations/")
set(PCH_DIR    "${ROOT_DIR}/include/")
set(JPEG_DIR   "${ROOT_DIR}/dependencies/toojpeg/")

message("-- PCH_DIR" ${PCH_DIR})

file(GLOB cpu_source_files 
  "${CMAKE_SOURCE_DIR}/main.c*"
  "${SOURCE_DIR}/*.cpp" 
  # "${JPEG_DIR}/*.cpp"
)

file(GLOB gpu_source_files 
  "${CMAKE_SOURCE_DIR}/*.cu"
  "${SOURCE_DIR}/*.cu" 
)

if(USE_CUDA)
  add_executable(physics_engine ${cpu_source_files} ${gpu_source_files})
  target_compile_definitions(physics_engine PRIVATE USE_CUDA)
  target_link_libraries(physics_engine fmt::fmt)
else()
  add_executable(physics_engine ${cpu_source_files})
  target_link_libraries(physics_engine fmt::fmt)
endif()

target_include_directories(physics_engine PRIVATE "/usr/local/cuda/include/" "${COMMON_DIR}" ${PCH_DIR} ${VECTOR_DIR})  

# Specify the precompiled header file

set(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS} -Wno-unused-variable -Wno-unused-but-set-variable" )


target_link_libraries(physics_engine fmt::fmt)
message(STATUS "USE_CUDA: ${USE_CUDA}")

target_precompile_headers(physics_engine PRIVATE "${PCH_DIR}/pch.h")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=20012 -w")

if(USE_CUDA)
  set_property(TARGET physics_engine PROPERTY CUDA_ARCHITECTURES 60)
  add_definitions(-DUSE_CUDA)
endif()
