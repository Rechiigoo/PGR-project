/// Homework 1.)
///
/// You can write your homework in (a) OpenGL or (b) Vulkan (harder).
/// 
/// a) OpenGL (standard) homework1:
/// Use indirect call and replace the other draw calls in this file.
/// Use single glMultiDrawArraysIndirect call to render everything.
///
/// You have to create draw indirect buffer that contains correct values (draw commands).
/// You will need these OpenGL functions:
/// glMultiDrawArraysIndirect
/// glCreateBuffers
/// glNamedBufferData
/// glBindBuffer
/// 
/// submit homework1.cpp
///
/// resources/images/pgp/homework01before.png
/// resources/images/pgp/homework01after.png
///
///
///
/// b) Vulkan (harder)
/// Recreate the homework1 in Vulkan and use single vkCmdDrawIndirect to draw everything.
/// Your homework in vulkan has to render multiple objects with multiple instances each.
/// You can render diamonds and triangles or other shapes.
/// It does not have to be in 3D.
/// The goal is to use single Indirect calls per frame.
/// Submit homework1_vulkan.zip with CMake without dependencies that runs on Linux.
/// Preferably:
/// $ cd build
/// $ cmake ..
/// $ make
/// $ ./homework1

#include<geGL/geGL.h>
#include<glm/glm.hpp>
#include<Vars/Vars.h>
#include<geGL/StaticCalls.h>
#include<imguiDormon/imgui.h>
#include<imguiVars/addVarsLimits.h>

#include<framework/methodRegister.hpp>
#include<framework/makeProgram.hpp>
#include<PGR/01/emptyWindow.hpp>

using namespace emptyWindow;
using namespace ge::gl;
using namespace std;

namespace pgp::homework1 {

    std::shared_ptr<ge::gl::Program>program;
    GLuint vbo;
    GLuint drawIdsBuffer;
    GLuint objectDataBuffer;
    GLuint vao;
    GLuint indirectBuffer; //for indirect draw calls

    void onInit(vars::Vars& vars) {
        auto src = R".(
  
  #ifdef  VERTEX_SHADER
  struct ObjectData{
    vec2 position;
    vec2 size    ;
    vec4 color   ;
  };
  
  layout(binding=0,std430)buffer Objects{ObjectData objects[];};
  
  layout(location=0)in vec2 position;
  layout(location=1)in int drawId;
  
  out vec4 vColor;
  
  void main() {
    ObjectData objectData = objects[drawId];
    vColor = objectData.color;
    gl_Position = vec4(position*objectData.size + objectData.position,1,1);
  }
  #endif//VERTEX_SHADER



  #ifdef FRAGMENT_SHADER
  in vec4 vColor;
  out vec4 fColor;
  
  void main(){
    fColor=vColor;
  }
  #endif//FRAGMENT_SHADER

  ).";

        //create shader program
        program = makeProgram(src);

        //buffer data
        std::vector<float>vertices = {
            //triangle vertices
            -1.f,-1.f,
            +1.f,-1.f,
            -1.f,+1.f,

            //diamond vertices
            +0.f,-1.f,
            +1.f,+0.f,
            -1.f,+0.f,
            +0.f,+1.f,
        };
        std::vector<int32_t>drawIds = {
          0,1,2,3,
          4,5,6
        };

        struct ObjectData {
            glm::vec2 position;
            glm::vec2 size;
            glm::vec4 color;
        };
        std::vector<ObjectData>objectData({
            {glm::vec2(+0.5f,-0.5f),glm::vec2(0.03f),glm::vec4(1,0,0,1)},
            {glm::vec2(+0.5f,-0.2f),glm::vec2(0.06f),glm::vec4(0,1,0,1)},
            {glm::vec2(+0.5f,+0.2f),glm::vec2(0.10f),glm::vec4(0,0,1,1)},
            {glm::vec2(+0.5f,+0.5f),glm::vec2(0.13f),glm::vec4(0,1,1,1)},
            {glm::vec2(-0.5f,-0.5f),glm::vec2(0.03f),glm::vec4(1,1,1,1)},
            {glm::vec2(-0.5f,+0.0f),glm::vec2(0.06f),glm::vec4(1,1,0,1)},
            {glm::vec2(-0.5f,+0.5f),glm::vec2(0.10f),glm::vec4(1,0,1,1)},
            });

        //This buffer contains vertex positions.
        glCreateBuffers(1, &vbo);
        glNamedBufferData(vbo, vertices.size() * sizeof(decltype(vertices)::value_type), vertices.data(), GL_STATIC_DRAW);

        //This buffer contains drawIds.
        glCreateBuffers(1, &drawIdsBuffer);
        glNamedBufferData(drawIdsBuffer, drawIds.size() * sizeof(decltype(drawIds)::value_type), drawIds.data(), GL_STATIC_DRAW);

        //This buffer contains positions, sizes and colors of all instances.
        glCreateBuffers(1, &objectDataBuffer);
        glNamedBufferData(objectDataBuffer, objectData.size() * sizeof(decltype(objectData)::value_type), objectData.data(), GL_STATIC_DRAW);


        // vertex array object
        glCreateVertexArrays(1, &vao);

        //attrib 0. is vertex positions
        glVertexArrayAttribFormat(vao, 0, 2, GL_FLOAT, GL_FALSE, 0);
        glVertexArrayVertexBuffer(vao, 0, vbo, 0, static_cast<GLsizei>(sizeof(float) * 2));
        glVertexArrayAttribBinding(vao, 0, 0);
        glEnableVertexArrayAttrib(vao, 0);
        glVertexArrayBindingDivisor(vao, 0, 0);

        //attrib 1. simulates gl_DrawID
        glVertexArrayAttribIFormat(vao, 1, 1, GL_INT, 0);
        glVertexArrayVertexBuffer(vao, 1, drawIdsBuffer, 0, sizeof(int32_t));
        glVertexArrayAttribBinding(vao, 1, 1);
        glEnableVertexArrayAttrib(vao, 1);
        glVertexArrayBindingDivisor(vao, 1, 1);

        //Additions 
        struct DrawArraysIndirectCommand {
            GLuint count;
            GLuint instanceCount;
            GLuint first;
            GLuint baseInstance;
        };

        DrawArraysIndirectCommand commands[2];

        // First draw (triangle)
        commands[0].count = 3;  // vertices per instance
        commands[0].instanceCount = 4;  // instances
        commands[0].first = 0;  // starting vertex
        commands[0].baseInstance = 0;  // base instance index

        // Second draw (diamond)
        commands[1].count = 4;
        commands[1].instanceCount = 3;
        commands[1].first = 3;
        commands[1].baseInstance = 4;

        glCreateBuffers(1, &indirectBuffer);
        glNamedBufferData(indirectBuffer, sizeof(commands), commands, GL_STATIC_DRAW);

        glClearColor(0, 0, 0, 1);
    }



    void onDraw(vars::Vars& vars) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        program->use();

        glBindVertexArray(vao);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, objectDataBuffer);

        /// you will have to replace these draw calls...
        //glDrawArraysInstancedBaseInstance(GL_TRIANGLE_STRIP,0,3,4,0);
        //glDrawArraysInstancedBaseInstance(GL_TRIANGLE_STRIP,3,4,3,4);

        //Draw calls replaced
        glBindBuffer(GL_DRAW_INDIRECT_BUFFER, indirectBuffer);
        glMultiDrawArraysIndirect(GL_TRIANGLE_STRIP, nullptr, 2, 0);
        glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0);

        glBindVertexArray(0);
    }

    void onQuit(vars::Vars& vars) {
        glDeleteBuffers(1, &drawIdsBuffer);
        glDeleteBuffers(1, &objectDataBuffer);
        glDeleteBuffers(1, &vbo);
        glDeleteBuffers(1, &indirectBuffer);
        glDeleteVertexArrays(1, &vao);
        program = nullptr;
        vars.erase("method");
    }


    EntryPoint main = []() {
        methodManager::Callbacks clbs;
        clbs.onDraw = onDraw;
        clbs.onInit = onInit;
        clbs.onQuit = onQuit;
        clbs.onResize = emptyWindow::onResize;
        MethodRegister::get().manager.registerMethod("pgp.homework1", clbs);
        };

}

