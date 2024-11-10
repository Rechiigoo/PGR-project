cmake_minimum_required(VERSION 3.12.0)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1 ) 

PROJECT(PGR)

set(PGR01_SOURCES
  01/emptyWindow.hpp
  01/emptyWindow.cpp
  01/oglFunc.cpp
  01/compileShaders.cpp
  01/compileShaders.hpp
  01/shaders.cpp
  01/shaders.hpp
  01/primitives.hpp
  01/primitives.cpp
  01/buffers.hpp
  01/buffers.cpp
  01/vertexArrays.hpp
  01/vertexArrays.cpp
  01/camera.hpp
  01/camera.cpp
  01/task.hpp
  01/task.cpp
  01/freelookCamera.hpp
  01/freelookCamera.cpp
  )

set(PGR02_SOURCES
  02/uv.hpp
  02/uv.cpp
  02/texture.hpp
  02/texture.cpp
  02/blending.cpp
  02/srgb.cpp
  02/model.hpp
  02/model.cpp
  )

set(PGR03_SOURCES
  03/phong.hpp
  03/phong.cpp
  03/framebuffer.hpp
  03/framebuffer.cpp
  03/shadowMapping.hpp
  03/shadowMapping.cpp
  03/shadowedModel.hpp
  03/shadowedModel.cpp
  03/twoTriangles.hpp
  03/twoTriangles.cpp
  03/stencil.cpp
  03/shadowVolume.cpp
  )

set(SOURCES
  ${PGR01_SOURCES}
  ${PGR02_SOURCES}
  ${PGR03_SOURCES}
  )

add_library(${PROJECT_NAME} OBJECT ${SOURCES})
target_link_libraries(${PROJECT_NAME} framework)
source_group("${PROJECT_NAME}" FILES ${SOURCES})
