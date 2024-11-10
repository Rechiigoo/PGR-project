cmake_minimum_required(VERSION 3.12.0)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1 ) 

PROJECT(LKG)

set(SOURCES
  lkgp.hpp
  lkgp.cpp
  lf_blend.hpp
  lf_blend.cpp
  spherical_harmonic_visualizer.cpp
  spherical_harmonic_visualizer.hpp
  )

add_library(${PROJECT_NAME} OBJECT ${SOURCES})
target_link_libraries(${PROJECT_NAME} framework)
source_group("${PROJECT_NAME}" FILES ${SOURCES})
