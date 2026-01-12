cmake_minimum_required(VERSION 3.12.0)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1 ) 

PROJECT(STUDENT)
set(CMAKE_TOOLCHAIN_FILE "C:/vcpkg/scripts/buildsystems/vcpkg.cmake")
set(SOURCES
  project.cpp
  blorb.cpp
  hatblorb.cpp
  leather_dmap.jpg 
   "witch.cpp" "robe.cpp" "gorro.cpp" "coat.cpp")

add_library(${PROJECT_NAME} OBJECT ${SOURCES})
target_link_libraries(${PROJECT_NAME} framework)
source_group("${PROJECT_NAME}" FILES ${SOURCES})
