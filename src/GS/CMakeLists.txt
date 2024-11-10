cmake_minimum_required(VERSION 3.12.0)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1 ) 

PROJECT(GS)

set(SOURCES
  geometryShaders.cpp
  geometryShaders.hpp
  gsWireBunny.cpp
  gsFrostBunny.cpp
  )

add_library(${PROJECT_NAME} OBJECT ${SOURCES})
target_link_libraries(${PROJECT_NAME} framework)
source_group("${PROJECT_NAME}" FILES ${SOURCES})
