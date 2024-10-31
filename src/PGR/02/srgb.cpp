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

namespace srgb{

/**
 * Mathematics of free look camera
 */

glm::mat4 proj         = glm::mat4(1.f); /// final projection matrix
glm::mat4 view         = glm::mat4(1.f); /// final view matrix
glm::mat4 viewRotation = glm::mat4(1.f); /// rotation part of view
glm::vec3 position     = glm::vec3(0.f,0.f,-3.f); /// camera position
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

auto vsSrc = R".(
#version 460
uniform mat4 view = mat4(1.);
uniform mat4 proj = mat4(1.);

out vec2 vPos;

void main(){
  gl_Position = vec4(0.f,0.f,0.f,1.f);

  if(gl_VertexID>=6)return;
  vPos = vec2((0x32>>gl_VertexID)&1,(0x2c>>gl_VertexID)&1)*2.f-1.f;
  gl_Position = vec4(vPos,0,1);
}
).";

auto fsSrc = R".(
#version 460
out vec4 fColor;

in vec2 vPos;

uniform mat4 view;
uniform mat4 proj;

struct Ray{
  vec3 s;
  vec3 d;
  float minT;
  float matT;
};

struct Sphere{
  vec3 c;
  float r;
  vec3 color;
  float metalness;
  float roughness;
  float opticalDensity;
};

float raySphere(Ray ray,Sphere s){
  float a = dot(ray.d,ray.d);
  float b = dot(ray.s-s.c,ray.d)*2.f;
  float c = dot(ray.s-s.c,ray.s-s.c)-s.r*s.r;
  float d = b*b-4.f*a*c;
  float t0 = (-b - sqrt(d))/(2.f*a);
  float t1 = (-b + sqrt(d))/(2.f*a);
  if(t0 < 0.f && t1 < 0.f)return -1.f;
  if(t0 < 0.f)return t1;
  if(t1 < 0.f)return t0;
  return min(t0,t1);
}

const float size = 100000;
const float roomSize = 1000;

const Sphere scene[]=Sphere[](
   Sphere(vec3(-size,0    ,    0),size-roomSize,vec3(1,0,0),0,1,100)
  ,Sphere(vec3(+size,0    ,    0),size-roomSize,vec3(0,1,0),0,1,100)
  ,Sphere(vec3(    0,0    ,-size),size-roomSize,vec3(.5   ),0,1,100)
  ,Sphere(vec3(    0,-size,    0),size-roomSize,vec3(.5   ),0,1,100)
  ,Sphere(vec3(    0,+size,    0),size-roomSize,vec3(0,0,1),0,1,100)


  ,Sphere(vec3(-700),300,vec3(.3),0,1,1.3)
  ,Sphere(vec3(0  ,0    ,0),10,vec3(0,0,1),0,1,100)
  ,Sphere(vec3(30 ,0    ,0),20,vec3(0,0,1),1,0,100)
  ,Sphere(vec3(60 ,0    ,0),20,vec3(.5,.5,0),.5,.5,100)
  ,Sphere(vec3(300,-300 ,0),200,vec3(0,1,1),1,0,100)
);

struct HitPoint{
  float t;
  vec3 position;
  vec3 normal;
  vec3 color;
  float metalness;
  float roughness;
  float opticalDensity;
};



HitPoint rayTrace(Ray ray){
  HitPoint hp;
  float t=10e10;
  int id=scene.length();
  for(int i=0;i<scene.length();++i){
    float newt = raySphere(ray,scene[i]);
    if(newt < 0.f)continue;
    if(newt < t){
      id = i;
      t = newt;
    }
  }

  if(id >= scene.length()){
    hp.t = -1.f;
    hp.position = ray.s + ray.d*1000000.f;
    hp.normal   = normalize(-ray.d);
    hp.color    = vec3(0,0,0);
    hp.metalness = 0;
    hp.roughness = 1;
    hp.opticalDensity = 100;
    return hp;
  }

  hp.t = t;
  hp.position  = ray.s + ray.d*t;
  hp.normal    = normalize(hp.position-scene[id].c);
  hp.color     = scene[id].color;

  //if(id == 3)hp.color = vec3(noise(hp.position,8u));//vec3(1,0,1);
  hp.metalness = scene[id].metalness;
  hp.roughness = scene[id].roughness;
  hp.opticalDensity = scene[id].opticalDensity;
  return hp;
}

Ray generateFlatScreenRay(){
  vec4 dir = inverse(proj*view)*vec4(vPos,1,1);
  dir.xyz/=dir.w;
  vec3 pos = vec3(inverse(proj*view)*vec4(0,0,0,1));

  Ray ray;
  ray.s=pos;
  ray.d=normalize(dir.xyz);
  return ray;
}



void main(){
  Ray ray;

  ray = generateFlatScreenRay();

  HitPoint hp = rayTrace(ray);

  vec3 L = normalize(vec3(100));

  float dF = max(dot(hp.normal,L),0);

  fColor = vec4(vec3(dF),1);
}

).";



void onInit(vars::Vars&vars){
  vars.addFloat("method.sensitivity"    ,  0.01f);
  vars.addFloat("method.near"           ,  0.10f);
  vars.addFloat("method.far"            ,100.00f);
  vars.addFloat("method.orbit.angleX"   ,  0.50f);
  vars.addFloat("method.orbit.angleY"   ,  0.50f);
  vars.addFloat("method.orbit.distance" ,  20.00f);
  vars.addFloat("method.orbit.zoomSpeed",  0.10f);

  auto vs = std::make_shared<ge::gl::Shader>(GL_VERTEX_SHADER  ,vsSrc);
  auto fs = std::make_shared<ge::gl::Shader>(GL_FRAGMENT_SHADER,fsSrc);
  prg = std::make_shared<ge::gl::Program>(vs,fs);
  vao = make_shared<VertexArray>();

  glClearColor(0.0,0.3,0,1);
  glEnable(GL_DEPTH_TEST);

  computeProjectionMatrix(vars);
}


void onDraw(vars::Vars&vars){
  computeProjectionMatrix(vars);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  if(vars.addOrGetBool("method.useSRGB"))
    glEnable(GL_FRAMEBUFFER_SRGB);
  else
    glDisable(GL_FRAMEBUFFER_SRGB);

  vao->bind();
  prg->use();

  prg->setMatrix4fv("view",glm::value_ptr(view));
  prg->setMatrix4fv("proj",glm::value_ptr(proj));

  glDrawArrays(GL_TRIANGLES,0,2*3);

  glDisable(GL_FRAMEBUFFER_SRGB);
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
  MethodRegister::get().manager.registerMethod("pgr02.srgb",clbs);
};

}
