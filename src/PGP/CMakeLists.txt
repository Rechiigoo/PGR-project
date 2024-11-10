cmake_minimum_required(VERSION 3.12.0)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1 ) 

PROJECT(PGP)

set(SOURCES
  lect/compute.hpp
  lect/compute.cpp
  homeworks/homework1.cpp
  homeworks/homework2.cpp
  homeworks/homework3.cpp
  homeworks/homework4.cpp
  homeworks/homework5.cpp
  )

add_library(${PROJECT_NAME} OBJECT ${SOURCES})
target_link_libraries(${PROJECT_NAME} framework)
source_group("${PROJECT_NAME}" FILES ${SOURCES})
