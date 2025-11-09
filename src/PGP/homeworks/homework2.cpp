///Homework 2. - Geometry Shaders
///
///Replace point rendering with Czech flag rendering.
///Change shaders in order to replace points with Czech flags.
///Do not touch glDrawArrays draw call!
///Do not touch vertex shader!
///
///Vulkan version is not necessary - you are only working with shaders.
///But if you have to...
///
/// resources/images/pgp/homework02before.png
/// resources/images/pgp/homework02after.png

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

namespace pgp::homework2{

shared_ptr<Program    >program;
shared_ptr<VertexArray>vao    ;

void onInit(vars::Vars&vars){ //do NOT touch vertex shader
  auto src = R".(
  #ifdef  VERTEX_SHADER
  void main() {
    if(gl_VertexID == 0)gl_Position = vec4(-.5,-.5,0,1);
    if(gl_VertexID == 1)gl_Position = vec4(+.5,-.5,0,1);
    if(gl_VertexID == 2)gl_Position = vec4(-.5,+.5,0,1);
    if(gl_VertexID == 3)gl_Position = vec4(+.5,+.5,0,1);
  }
  #endif//VERTEX_SHADER
  


  #ifdef  GEOMETRY_SHADER
  ///\todo Homework 2. Reimplement geometry shader.
  /// The geometry shader should replace an input point
  /// with flag of Czech Republic.
  /// every point should be replaced with Czech Republic flag.

  layout(points)in;
  layout(triangle_strip,max_vertices=12)out; //triangle strip for czech flag?

  out vec3 color; //color to fragment shader
  
  void main(){
    vec4 center_pos = gl_in[0].gl_Position;
    float size = 0.1;
    float aspec_ratio = 1.5;

    //corners of flag
    vec4 topl = center_pos + vec4(-size*aspec_ratio, size, 0, 0);
    vec4 topr = center_pos + vec4(size*aspec_ratio, size, 0, 0);
    vec4 botl = center_pos + vec4(-size*aspec_ratio, -size, 0, 0);
    vec4 botr = center_pos + vec4(size*aspec_ratio, -size, 0, 0);

    //white half
    color = vec3(1.0);
    gl_Position = topl; EmitVertex();
    gl_Position = topr; EmitVertex();
    gl_Position = center_pos + vec4(-size * aspec_ratio,0,0,0);
    EmitVertex();
    gl_Position = center_pos + vec4(size * aspec_ratio,0,0,0);
    EmitVertex();
    EndPrimitive();

    //red half
    color = vec3(1.0, 0.0, 0.0);
    gl_Position = center_pos + vec4(-size * aspec_ratio,0,0,0);
    EmitVertex();
    gl_Position = center_pos + vec4(size * aspec_ratio,0,0,0);
    EmitVertex();
    gl_Position = botl; EmitVertex();
    gl_Position = botr; EmitVertex();
    EndPrimitive();

    //blue triangle
    color = vec3(0.0, 0.0, 1.0);
    gl_Position = topl + vec4(0,0,0.001,0);
    EmitVertex();
    gl_Position = center_pos + vec4(0,0,0.001,0);
    EmitVertex();
    gl_Position = botl + vec4(0,0,0.001,0);
    EmitVertex();
    EndPrimitive();
    
  }
  #endif//GEOMETRY_SHADER


  #ifdef FRAGMENT_SHADER
  in vec3 color;
  out vec4 fColor;
  
  void main(){
    fColor = vec4(color,1.0);
  }
  #endif//FRAGMENT_SHADER
  ).";

  //create shader program
  program = makeProgram(src);
  vao     = make_shared<ge::gl::VertexArray>();
  
  glClearColor(0,0,0,1);
}

void onDraw(vars::Vars&vars){
  glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
  program->use();
  vao->bind();
  glDrawArrays(GL_POINTS,0,4); //Do not touch
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

