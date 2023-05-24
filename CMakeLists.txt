cmake_minimum_required(VERSION 3.10)
project(rknn_UFLD)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wl,--allow-shlib-undefined")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wl,--allow-shlib-undefined")

# rknn api
set(RKNN_API_PATH /home/rpdzkj/dev/rknpu2/runtime/RK3588/Linux/librknn_api)
set(LIB_ARCH aarch64)
set(RKNN_RT_LIB ${RKNN_API_PATH}/${LIB_ARCH}/librknnrt.so)
include_directories(${RKNN_API_PATH}/include)

# opencv
set(OpenCV_DIR /home/rpdzkj/dev/rknpu2/examples/3rdparty/opencv/opencv-linux-aarch64/share/OpenCV)
find_package(OpenCV REQUIRED)

#rga
set(RGA_PATH /home/rpdzkj/dev/rknpu2/examples/3rdparty/rga/RK3588)
set(LIB_ARCH aarch64)
set(RGA_LIB ${RGA_PATH}/lib/Linux//${LIB_ARCH}/librga.so)

include_directories( ${RGA_PATH}/include)
include_directories( ${CMAKE_SOURCE_DIR}/include)


add_executable(${PROJECT_NAME} main.cpp UFLDdetector.h UFLDdetector.cpp)
target_link_libraries(${PROJECT_NAME}
        ${RKNN_RT_LIB}
        ${RGA_LIB}
        ${OpenCV_LIBS}
        )