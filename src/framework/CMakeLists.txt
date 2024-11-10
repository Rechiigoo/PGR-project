cmake_minimum_required(VERSION 3.12.0)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1 ) 

PROJECT(framework)

set(SOURCES
  DVars.hpp
  Barrier.cpp
  Barrier.h
  FunctionPrologue.h
  methodRegister.hpp
  methodRegister.cpp
  Timer.hpp
  model.cpp
  model.hpp
  bunny.cpp
  bunny.hpp
  OpenGLState.cpp
  OpenGLState.hpp
  methodManager.hpp
  methodManager.cpp
  defineGLSLVersion.cpp
  defineGLSLVersion.hpp
  makeProgram.cpp
  makeProgram.hpp
  )

add_library(${PROJECT_NAME} OBJECT ${SOURCES})

target_link_libraries(${PROJECT_NAME}
  tiny_gltf
  Simple3DApp::Simple3DApp 
  Vars::Vars 
  VarsGLMDecorator::VarsGLMDecorator 
  imguiDormon::imguiDormon 
  imguiVars::imguiVars
  BasicCamera::BasicCamera
  )

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)

option(CLEAR_CMAKE_ROOT_DIR "if this is set, #define CMAKE_ROOT_DIR will be .")
if(NOT CLEAR_CMAKE_ROOT_DIR)
  target_compile_definitions(${PROJECT_NAME} PUBLIC CMAKE_ROOT_DIR="${CMAKE_SOURCE_DIR}")
else()
  #target_compile_definitions(${PROJECT_NAME} PUBLIC CMAKE_ROOT_DIR=".")
endif()

source_group("${PROJECT_NAME}" FILES ${SOURCES})
