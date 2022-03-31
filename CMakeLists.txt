cmake_minimum_required(VERSION 3.6)
project(PyORB)

find_package(Eigen3)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} ${PROJECT_SOURCE_DIR})

#find_package(COLMAP REQUIRED)
#include_directories(${COLMAP_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR})
#link_directories(${COLMAP_LINK_DIRS})


#add_executable(orb_detector
#        main.cc orb.cpp orb.hpp)
#target_link_libraries(orb_detector ${OpenCV_LIBS})


add_subdirectory(pybind11)

pybind11_add_module(pyorb main.cc orb.cpp orb.hpp)
target_link_libraries(pyorb PRIVATE ${OpenCV_LIBS})
