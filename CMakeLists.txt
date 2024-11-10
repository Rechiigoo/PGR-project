cmake_minimum_required(VERSION 3.12.0)

project(FitGraphics)

set(CMAKE_CXX_VISIBILITY_PRESET   default)
set(CMAKE_CXX_STANDARD            17     )
set(CMAKE_C_STANDARD              99     )
set(CMAKE_EXPORT_COMPILE_COMMANDS 1      ) 

# visual studio startup project
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})

add_library(glm INTERFACE)
target_include_directories(glm INTERFACE libs/glm-0.9.9.8)

option(SDL_SHARED "" OFF)
option(SDL_STATIC "" ON)
add_subdirectory(libs/SDL-release-2.24.1)

add_subdirectory(libs/MealyMachine)
add_subdirectory(libs/TxtUtils)
add_subdirectory(libs/ArgumentViewer)
add_subdirectory(libs/BasicCamera)
add_subdirectory(libs/SDL2CPP)
add_subdirectory(libs/geGL)
add_subdirectory(libs/Vars)
add_subdirectory(libs/imguiDormon)
add_subdirectory(libs/imguiOpenGLDormon)
add_subdirectory(libs/imguiSDL2Dormon)
add_subdirectory(libs/imguiSDL2OpenGL)
add_subdirectory(libs/imguiVars)
add_subdirectory(libs/Simple3DApp)
add_subdirectory(libs/VarsGLMDecorator)
add_subdirectory(libs/stb_image)
add_subdirectory(libs/json)
add_subdirectory(libs/tiny_gltf)

add_subdirectory(src/framework)
add_subdirectory(src/GMU)
add_subdirectory(src/MISC)
add_subdirectory(src/GS)
add_subdirectory(src/TS)
add_subdirectory(src/LKG)
add_subdirectory(src/PGP)
add_subdirectory(src/student)
add_subdirectory(src/PGR)


#solution files / files that should not be visible to students
set(SOLUTION_DIR "" CACHE STRING "Directory containing solution files")

if(NOT ${SOLUTION_DIR} EQUAL "")
  add_subdirectory(${SOLUTION_DIR} SOLUTION)
endif()

set(APP_SOURCES
  src/framework/main.cpp
  )

add_executable(${PROJECT_NAME} 
  ${APP_SOURCES} 
  )

set_property(TARGET ${PROJECT_NAME} PROPERTY ENABLE_EXPORTS 1)
target_link_libraries(${PROJECT_NAME}
  framework
  PGR
  PGP
  GMU
  GS
  TS
  MISC
  LKG
  STUDENT
  ${SOLUTION_LIBRARY}
  )

if (CMAKE_CROSSCOMPILING)
  target_link_libraries(${PROJECT_NAME}  
    -lmingw32
    -static-libgcc 
    -static-libstdc++
    )
endif()

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_custom_target(run ./${PROJECT_NAME})

