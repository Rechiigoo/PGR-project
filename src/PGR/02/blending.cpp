#include "glm/geometric.hpp"
#include<glm/glm.hpp>
#include<glm/gtc/type_ptr.hpp>
#include<SDL.h>
#include<Vars/Vars.h>
#include<geGL/geGL.h>
#include<geGL/StaticCalls.h>
#include<imguiDormon/imgui.h>
#include<imguiVars/addVarsLimits.h>
#include<framework/FunctionPrologue.h>
#include<framework/methodRegister.hpp>
#include<framework/makeProgram.hpp>
#include<PGR/01/emptyWindow.hpp>
#include<PGR/01/compileShaders.hpp>

using namespace ge::gl;
using namespace compileShaders;
using namespace std;

namespace blending{

/**
 * Mathematics of free look camera
 */

glm::mat4 proj         = glm::mat4(1.f); /// final projection matrix
glm::mat4 view         = glm::mat4(1.f); /// final view matrix
glm::mat4 viewRotation = glm::mat4(1.f); /// rotation part of view
glm::vec3 position     = glm::vec3(0.f); /// camera position
float     angleX       = 0.f           ; /// camera angle around x axis
float     angleY       = 0.f           ; /// camera angle around y axis

/**
 * @brief This function computes projection matrix
 *
 * @param vars input/output variables
 */
void computeProjectionMatrix(vars::Vars&vars){
  auto width  = vars.getUint32("event.resizeX");
  auto height = vars.getUint32("event.resizeY");
  auto near   = vars.getFloat ("method.near"  );
  auto far    = vars.getFloat ("method.far"   );

  auto aspect = static_cast<float>(width) / static_cast<float>(height);
  proj = glm::perspective(glm::half_pi<float>(),aspect,near,far);
}

/**
 * @brief This function computes free look view matrix
 *
 * @param vars input/output variables
 */
void computeFreelookMatrix(vars::Vars&vars){
  viewRotation = 
    glm::rotate(glm::mat4(1.f),angleX,glm::vec3(1.f,0.f,0.f))*
    glm::rotate(glm::mat4(1.f),angleY,glm::vec3(0.f,1.f,0.f));

  view = 
    viewRotation*
    glm::translate(glm::mat4(1.f),position);
}

/**
 * Controls of the free look camera
 */

std::map<int,bool>keys; /// which keys are down

void onKeyDown(vars::Vars&vars){
  auto key = vars.getInt32("event.key");
  keys[key] = true;
}

void onKeyUp(vars::Vars&vars){
  auto key = vars.getInt32("event.key");
  keys[key] = false;
}

void onMouseMotion(vars::Vars&vars){
  auto xrel = vars.getInt32("event.mouse.xrel");
  auto yrel = vars.getInt32("event.mouse.yrel");

  auto sensitivity = vars.getFloat("method.sensitivity"    );

  if(vars.getBool("event.mouse.right")){
    angleX += sensitivity * yrel;
    angleY += sensitivity * xrel;

    angleX = glm::clamp(angleX,-glm::half_pi<float>(),glm::half_pi<float>());
  }
}

void onUpdate(vars::Vars&vars){
  auto dt = vars.getFloat("event.dt");

  auto keyVector = glm::vec3(
      keys[SDLK_a     ]-keys[SDLK_d    ],
      keys[SDLK_LSHIFT]-keys[SDLK_SPACE],
      keys[SDLK_w     ]-keys[SDLK_s    ]);

  position += dt * keyVector*glm::mat3(viewRotation);
  computeFreelookMatrix(vars);
}



shared_ptr<Program>prg;
shared_ptr<VertexArray>vao;

std::string const source = R".(
#ifdef VERTEX_SHADER

uniform mat4 view = mat4(1.f);
uniform mat4 proj = mat4(1.f);

out vec4 vColor;
void main(){

  
  int vID = gl_VertexID%6;
  int qID = gl_VertexID/6;
  vec2 c = vec2((0x31>>vID)&1,(0x2C>>vID)&1);

  vColor = vec4(0.5,0.5+qID*0.1,1,0.5);
  gl_Position = proj*view*vec4(c*2.f-1.f,qID-1,1.f);

}
#endif

#ifdef FRAGMENT_SHADER
in vec4 vColor;
layout(location=0)out vec4 fColor;
void main(){
  fColor = vColor;
}
#endif
).";



void onInit(vars::Vars&vars){
  vars.addFloat("method.sensitivity"    ,  0.01f);
  vars.addFloat("method.near"           ,  0.10f);
  vars.addFloat("method.far"            ,100.00f);
  vars.addFloat("method.orbit.angleX"   ,  0.50f);
  vars.addFloat("method.orbit.angleY"   ,  0.50f);
  vars.addFloat("method.orbit.distance" ,  4.00f);
  vars.addFloat("method.orbit.zoomSpeed",  0.10f);

  prg = makeProgram(source);
  vao = make_shared<VertexArray>();

  glClearColor(0.0,0.3,0,1);
  glEnable(GL_DEPTH_TEST);

  computeProjectionMatrix(vars);
}

GLenum selectEnum(vars::Vars&vars,char const*name,char const**names,GLenum*ids,int n,std::string const&varName){
  auto&selected = vars.addOrGetInt32(varName,0);
  ImGui::Begin("vars");
  ImGui::ListBox(name,&selected,names,n);
  ImGui::End();
  hide(vars,varName);
  return ids[selected];
}

GLenum selectEquation(vars::Vars&vars){
  char const* names[]={
    "GL_FUNC_ADD             ",
    "GL_FUNC_SUBTRACT        ",
    "GL_FUNC_REVERSE_SUBTRACT",
    "GL_MIN                  ",
    "GL_MAX                  ",
  };
  GLenum ids[]={
    GL_FUNC_ADD             ,
    GL_FUNC_SUBTRACT        ,
    GL_FUNC_REVERSE_SUBTRACT,
    GL_MIN                  ,
    GL_MAX                  ,
  };
  auto n = (int32_t)sizeof(names)/sizeof(char const*);
  return selectEnum(vars,"equation",names,ids,n,"method.selectedEquation");
}

GLenum selectFactor(vars::Vars&vars,std::string const&name){
  char const* names[]={
    "GL_ZERO                    ",
    "GL_ONE                     ",
    "GL_SRC_COLOR               ",
    "GL_ONE_MINUS_SRC_COLOR     ",
    "GL_DST_COLOR               ",
    "GL_ONE_MINUS_DST_COLOR     ",
    "GL_SRC_ALPHA               ",
    "GL_ONE_MINUS_SRC_ALPHA     ",
    "GL_DST_ALPHA               ",
    "GL_ONE_MINUS_DST_ALPHA     ",
    "GL_CONSTANT_COLOR          ",
    "GL_ONE_MINUS_CONSTANT_COLOR",
    "GL_CONSTANT_ALPHA          ",
    "GL_ONE_MINUS_CONSTANT_ALPHA",
  };
  GLenum ids[]={
    GL_ZERO                    ,
    GL_ONE                     ,
    GL_SRC_COLOR               ,
    GL_ONE_MINUS_SRC_COLOR     ,
    GL_DST_COLOR               ,
    GL_ONE_MINUS_DST_COLOR     ,
    GL_SRC_ALPHA               ,
    GL_ONE_MINUS_SRC_ALPHA     ,
    GL_DST_ALPHA               ,
    GL_ONE_MINUS_DST_ALPHA     ,
    GL_CONSTANT_COLOR          ,
    GL_ONE_MINUS_CONSTANT_COLOR,
    GL_CONSTANT_ALPHA          ,
    GL_ONE_MINUS_CONSTANT_ALPHA,
  };
  auto n = (int32_t)sizeof(names)/sizeof(char const*);
  return selectEnum(vars,name.c_str(),names,ids,n,"method.selected"+name);
}

void onDraw(vars::Vars&vars){
  auto equation = selectEquation(vars);
  auto sfactor  = selectFactor(vars,"sfactor");
  auto dfactor  = selectFactor(vars,"dfactor");
  computeProjectionMatrix(vars);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  vao->bind();
  prg->use();

  prg->setMatrix4fv("view",glm::value_ptr(view));
  prg->setMatrix4fv("proj",glm::value_ptr(proj));

  glEnable(GL_BLEND);
  glBlendFunc(sfactor,dfactor);
  glBlendEquation(equation);

  glDrawArrays(GL_TRIANGLES,0,6*3);
}

void onQuit(vars::Vars&vars){
  prg = nullptr;
  vao = vao;
  vars.erase("method");
}

EntryPoint main = [](){
  methodManager::Callbacks clbs;
  clbs.onDraw        =              onDraw       ;
  clbs.onInit        =              onInit       ;
  clbs.onQuit        =              onQuit       ;
  clbs.onResize      = emptyWindow::onResize     ;
  clbs.onKeyDown     =              onKeyDown    ;
  clbs.onKeyUp       =              onKeyUp      ;
  clbs.onMouseMotion =              onMouseMotion;
  clbs.onUpdate      =              onUpdate     ;
  MethodRegister::get().manager.registerMethod("pgr02.blending",clbs);
};

}
