cmake_minimum_required(VERSION 3.12.0)

project(json)

add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE .)
