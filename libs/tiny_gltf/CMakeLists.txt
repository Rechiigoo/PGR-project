cmake_minimum_required(VERSION 3.12.0)

project(tiny_gltf)

add_library(${PROJECT_NAME} STATIC
  tiny_gltf.h
  impl.cpp
  )

target_link_libraries(${PROJECT_NAME} PUBLIC
  json
  stb_image
  )
target_include_directories(${PROJECT_NAME} PUBLIC .)

