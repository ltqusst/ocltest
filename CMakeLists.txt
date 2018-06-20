cmake_minimum_required (VERSION 3.1)
project (OCLtest)

set(CMAKE_CXX_STANDARD 11)  # for c++11

option(WITH_OPENCV "Test with opencv va_intel" OFF)

if(WITH_OPENCV)
    add_definitions(-DUSE_OPENCV)
endif()

find_package(OpenCV)
message("OPENCV_INCLUDE_DIR=${OpenCV_INCLUDE_DIR}")
message("OpenCV_LIBS=${OpenCV_LIBS}")

#find_package(OpenGL REQUIRED)
#message("OPENGL_INCLUDE_DIR=${OPENGL_INCLUDE_DIR}")
#message("OPENGL_LIBRARIES=${OPENGL_LIBRARIES}")

#find_package(GLEW REQUIRED)
#message("GLEW_INCLUDE_DIR=${GLEW_INCLUDE_DIR}")
#message("GLEW_LIBRARIES=${GLEW_LIBRARIES}")

#find_package(X11 REQUIRED)
#message("X11_INCLUDE_DIR=${X11_INCLUDE_DIR}")
#message("X11_LIBRARIES=${X11_LIBRARIES}")

# make sure correct so links are made:
#     sudo ln -s /usr/lib/x86_64-linux-gnu/libOpenCL.so.1.0.0 /usr/lib/x86_64-linux-gnu/libOpenCL.so
#     sudo ln -s /opt/intel/mediasdk/lib64/libva.so.2 /usr/lib/x86_64-linux-gnu/libva.so
#     sudo ln -s /opt/intel/mediasdk/lib64/libva-drm.so.2 /usr/lib/x86_64-linux-gnu/libva-drm.so

# libva's headers are there
set(VA_INCLUDE_DIR "/opt/intel/mediasdk/include/")
set(VA_LIBRARIES "-lva" "-lva-drm")

# https://github.com/KhronosGroup/OpenCL-Headers.git
find_path(OpenCL_INCLUDE_DIR "CL/cl.h"
          PATHS "/home/hddl/OpenCL-Headers/"
                "/opt/intel/opencl/include/")
                

set(OpenCL_LIBRARIES "-lOpenCL")
message("OpenCL_INCLUDE_DIR=${OpenCL_INCLUDE_DIR}")
message("OpenCL_LIBRARIES=${OpenCL_LIBRARIES}")

add_definitions(-DCOMPILE_MAIN)

add_executable (ocltest ocltest.cpp)

include_directories( ${OpenCL_INCLUDE_DIR} ${VA_INCLUDE_DIR} ${OpenCV_INCLUDE_DIR})

target_link_libraries (ocltest ${OpenCL_LIBRARIES} ${VA_LIBRARIES} ${OpenCV_LIBS} -lpthread)


# embed 
add_custom_target(
    embedKernels
    COMMAND ${CMAKE_COMMAND} -Dsrc=${OCLtest_SOURCE_DIR}/kernels.cl -Ddst=${OCLtest_SOURCE_DIR}/kernels.inl -P ${OCLtest_SOURCE_DIR}/embedKernels.cmake
)
add_custom_target(
    embedKernels2
    COMMAND ${CMAKE_COMMAND} -Dsrc=${OCLtest_SOURCE_DIR}/kernels2.cl -Ddst=${OCLtest_SOURCE_DIR}/kernels2.inl -P ${OCLtest_SOURCE_DIR}/embedKernels.cmake
)

add_dependencies(ocltest embedKernels embedKernels2) 


