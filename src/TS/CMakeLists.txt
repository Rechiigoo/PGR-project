cmake_minimum_required(VERSION 3.12.0)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1 ) 

PROJECT(TS)

set(SOURCES
  tessellationLevels.cpp
  bezierSurf.cpp
  tsCircleBunny.cpp
  )

add_library(${PROJECT_NAME} OBJECT ${SOURCES})
target_link_libraries(${PROJECT_NAME} framework)
source_group("${PROJECT_NAME}" FILES ${SOURCES})
