cmake_minimum_required(VERSION 3.10)
message(STATUS "cmake version ${CMAKE_VERSION}")
project(rknn_yolov5_seg_cam)

set(TOOLCHAIN_DIR /home/talowe/talowe_rv1126b_linux6.1_sdk_v1.0/buildroot/output/rockchip_rv1126b/host)
set(CMAKE_CXX_COMPILER ${TOOLCHAIN_DIR}/bin/aarch64-buildroot-linux-gnu-g++)
set(CMAKE_C_COMPILER ${TOOLCHAIN_DIR}/bin/aarch64-buildroot-linux-gnu-gcc)
set(SYSROOT ${TOOLCHAIN_DIR}/aarch64-buildroot-linux-gnu/sysroot/usr/include)
set(CMAKE_SYSROOT ${TOOLCHAIN_DIR}/aarch64-buildroot-linux-gnu/sysroot)
set(CMAKE_CXX_STANDARD 11)
add_definitions(-g -O0 -ggdb -gdwarf -funwind-tables -rdynamic)
add_definitions(-Wno-write-strings -Wno-return-type)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/ 3rdparty.out)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/utils/ utils.out)

# opencv
include_directories(${SYSROOT})
include_directories(${SYSROOT}/opencv4)
set(OPENCV_LIBS opencv_core opencv_flann opencv_videoio opencv_video opencv_highgui opencv_imgcodecs opencv_imgproc)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")

file(GLOB SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)

#dma
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/allocator/dma)
#drm
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/allocator/drm)

set(postprocess_file rknpu2/postprocess.cc)
set(yolov5_seg_file rknpu2/yolov5_seg.cc)

add_executable(${PROJECT_NAME}
    main.cc
    ${postprocess_file}
    ${yolov5_seg_file}
)

target_link_libraries(${PROJECT_NAME}
      fileutils
      imageutils
      imagedrawing
      ${OPENCV_LIBS}
      ${LIBRKNNRT}
)

# 使用pthread线程库
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(${PROJECT_NAME} Threads::Threads)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LIBRKNNRT_INCLUDES}
)

install(TARGETS ${PROJECT_NAME} DESTINATION .)
install(FILES ${LIBRKNNRT} DESTINATION lib)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../model/coco_80_labels_list.txt DESTINATION model)

file(GLOB RKNN_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../model/*.rknn")
install(FILES ${RKNN_FILES} DESTINATION model)
