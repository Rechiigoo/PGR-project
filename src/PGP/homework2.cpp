///Homework 2. - Geometry Shaders
///
///Replace point rendering with Czech flag rendering.
///Change shaders in order to replace points with Czech flags.
///Do not touch glDrawArrays draw call!
///Do not touch vertex shader!
///
///Vulkan version is not necessary - you are only working with shaders.

#include<geGL/geGL.h>
#include<glm/glm.hpp>
#include<Vars/Vars.h>
#include<geGL/StaticCalls.h>
#include<imguiDormon/imgui.h>
#include<imguiVars/addVarsLimits.h>

#include<framework/methodRegister.hpp>
#include<framework/defineGLSLVersion.hpp>
#include<PGR/01/emptyWindow.hpp>

using namespace emptyWindow;
using namespace ge::gl;
using namespace std;

namespace pgp::homework2{

shared_ptr<Program    >program;
shared_ptr<VertexArray>vao    ;

void onInit(vars::Vars&vars){
  auto vertSrc = defineGLSLVersion()+R".(
  void main() {
    if(gl_VertexID == 0)gl_Position = vec4(-.5,-.5,0,1);
    if(gl_VertexID == 1)gl_Position = vec4(+.5,-.5,0,1);
    if(gl_VertexID == 2)gl_Position = vec4(-.5,+.5,0,1);
    if(gl_VertexID == 3)gl_Position = vec4(+.5,+.5,0,1);
  }).";
  
  ///\todo Homework 2. Reimplement geometry shader.
  /// This geometry shader should replace an input point
  /// with flag of Czech Republic.
  auto geomSrc = defineGLSLVersion()+R".(
  
  layout(points)in;
  layout(points,max_vertices=1)out;
  
  void main(){
    gl_Position = gl_in[0].gl_Position;
    EmitVertex();
    EndPrimitive();
  }
  ).";

  auto fragSrc = defineGLSLVersion()+ R".(
  out vec4 fColor;
  
  void main(){
    fColor = vec4(1,1,1,1);
  }).";

  //create shader program
  auto vs = make_shared<Shader>(GL_VERTEX_SHADER  ,vertSrc);
  auto gs = make_shared<Shader>(GL_GEOMETRY_SHADER,geomSrc);
  auto fs = make_shared<Shader>(GL_FRAGMENT_SHADER,fragSrc);

  program = make_shared<ge::gl::Program>(vs,gs,fs);
  vao     = make_shared<ge::gl::VertexArray>();
  
  glClearColor(0,0,0,1);
}

void onDraw(vars::Vars&vars){
  glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
  program->use();
  vao->bind();
  glDrawArrays(GL_POINTS,0,4);
  vao->unbind();
}

void onQuit(vars::Vars&vars){
  vao     = nullptr;
  program = nullptr;
  vars.erase("method");
}


EntryPoint main = [](){
  methodManager::Callbacks clbs;
  clbs.onDraw        =              onDraw       ;
  clbs.onInit        =              onInit       ;
  clbs.onQuit        =              onQuit       ;
  clbs.onResize      = emptyWindow::onResize     ;
  MethodRegister::get().manager.registerMethod("pgp.homework2",clbs);
};

}

