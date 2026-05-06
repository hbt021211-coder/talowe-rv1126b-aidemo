# Install script for directory: /home/talowe/rv1126b_aidemo/talowe_yolov8_garbage_sort_cam/cpp

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/talowe/rv1126b_aidemo/talowe_yolov8_garbage_sort_cam/cpp/install/rv1126b_linux_aarch64/rknn_yolov8_garbage_sort_cam")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "TRUE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/home/talowe/talowe_rv1126b_linux6.1_sdk_v1.0/buildroot/output/rockchip_rv1126b/host/bin/aarch64-buildroot-linux-gnu-objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/./rknn_yolov8_garbage_sort_cam" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/./rknn_yolov8_garbage_sort_cam")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/./rknn_yolov8_garbage_sort_cam"
         RPATH "$ORIGIN/../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/." TYPE EXECUTABLE FILES "/home/talowe/rv1126b_aidemo/talowe_yolov8_garbage_sort_cam/cpp/build/build_rknn_yolov8_garbage_sort_cam_rv1126b_linux_aarch64_Release/rknn_yolov8_garbage_sort_cam")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/./rknn_yolov8_garbage_sort_cam" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/./rknn_yolov8_garbage_sort_cam")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/./rknn_yolov8_garbage_sort_cam"
         OLD_RPATH "/home/talowe/rv1126b_aidemo/talowe_yolov8_garbage_sort_cam/cpp/3rdparty/rknpu2/Linux/aarch64:"
         NEW_RPATH "$ORIGIN/../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/home/talowe/talowe_rv1126b_linux6.1_sdk_v1.0/buildroot/output/rockchip_rv1126b/host/bin/aarch64-buildroot-linux-gnu-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/./rknn_yolov8_garbage_sort_cam")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE FILE FILES "/home/talowe/rv1126b_aidemo/talowe_yolov8_garbage_sort_cam/cpp/3rdparty/rknpu2/Linux/aarch64/librknnrt.so")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/model" TYPE FILE FILES "/home/talowe/rv1126b_aidemo/talowe_yolov8_garbage_sort_cam/cpp/../model/coco_4_labels_list.txt")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/model" TYPE FILE FILES "/home/talowe/rv1126b_aidemo/talowe_yolov8_garbage_sort_cam/cpp/../model/garbage_sort.rknn")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/talowe/rv1126b_aidemo/talowe_yolov8_garbage_sort_cam/cpp/build/build_rknn_yolov8_garbage_sort_cam_rv1126b_linux_aarch64_Release/3rdparty.out/cmake_install.cmake")
  include("/home/talowe/rv1126b_aidemo/talowe_yolov8_garbage_sort_cam/cpp/build/build_rknn_yolov8_garbage_sort_cam_rv1126b_linux_aarch64_Release/utils.out/cmake_install.cmake")

endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/talowe/rv1126b_aidemo/talowe_yolov8_garbage_sort_cam/cpp/build/build_rknn_yolov8_garbage_sort_cam_rv1126b_linux_aarch64_Release/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
