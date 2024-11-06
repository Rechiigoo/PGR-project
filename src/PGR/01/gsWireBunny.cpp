#include<Vars/Vars.h>
#include<geGL/StaticCalls.h>
#include<geGL/geGL.h>
#include<framework/methodRegister.hpp>
#include<framework/FunctionPrologue.h>
#include<BasicCamera/OrbitCamera.h>
#include<BasicCamera/PerspectiveCamera.h>
#include<PGR/01/emptyWindow.hpp>
#include<PGR/02/model.hpp>
#include<PGR/03/phong.hpp>
#include<framework/bunny.hpp>

namespace gsWireBunny{

using namespace ge::gl;

std::string vsSrc = R".(
out vec4 vColor;

uniform float iTime = 0;
uniform mat4 proj = mat4(1.f);
uniform mat4 view = mat4(1.f);
uniform mat4 model = mat4(1.f);

layout(location=0)in vec3 position;
layout(location=1)in vec3 normal  ;

void main(){
  vColor = vec4(normal,1);
  gl_Position = proj*view*model*vec4(position,1);
}
).";

std::string  gsSrc = R".(
layout(triangles)in;
layout(line_strip,max_vertices=4)out;

in vec4 vColor[];
out vec4 gColor;

void main(){
  vec4 a = (gl_in[0].gl_Position + gl_in[1].gl_Position)/2.f;
  vec4 b = (gl_in[1].gl_Position + gl_in[2].gl_Position)/2.f;
  vec4 c = (gl_in[2].gl_Position + gl_in[0].gl_Position)/2.f;

  vec4 ac = (vColor[0] + vColor[1])/2.f;
  vec4 bc = (vColor[1] + vColor[2])/2.f;
  vec4 cc = (vColor[2] + vColor[0])/2.f;

  gl_Position = a;
  gColor = ac;
  EmitVertex();

  gl_Position = b;
  gColor = bc;
  EmitVertex();

  gl_Position = c;
  gColor = cc;
  EmitVertex();

  gl_Position = a;
  gColor = ac;
  EmitVertex();

  EndPrimitive();
}
).";

std::string const fsSrc = R".(
in vec4 gColor;

out vec4 fColor;
void main(){
  fColor = vec4(gColor);
}
).";

void initBunny(vars::Vars&vars){
  FUNCTION_PROLOGUE("method.bunny");
  vars.reCreate<Program>("method.bunny.prg",
      std::make_shared<Shader>(GL_VERTEX_SHADER  ,"#version 460\n"+vsSrc),
      std::make_shared<Shader>(GL_GEOMETRY_SHADER,"#version 460\n"+gsSrc),
      std::make_shared<Shader>(GL_FRAGMENT_SHADER,"#version 460\n"+fsSrc)
      );

  vars.addUint32("method.bunny.nofIndices",sizeof(bunnyIndices)/sizeof(VertexIndex));
  auto vbo = vars.reCreate<Buffer>("method.bunny.vbo",sizeof(bunnyVertices),bunnyVertices);
  auto ebo = vars.reCreate<Buffer>("method.bunny.ebo",sizeof(bunnyIndices ),bunnyIndices );
  auto vao = vars.reCreate<VertexArray>("method.bunny.vao");
  vao->addAttrib(vbo,0,3,GL_FLOAT,sizeof(BunnyVertex),0);
  vao->addAttrib(vbo,1,3,GL_FLOAT,sizeof(BunnyVertex),sizeof(BunnyVertex::position));
  vao->addElementBuffer(ebo);
}

void drawScene(vars::Vars&vars){
  ge::gl::VertexArray*vao = nullptr;
  ge::gl::Program    *prg = nullptr;
  GLenum primitive = GL_TRIANGLES;
  uint32_t nofVertices = 0;
  bool hasIndices = false;

  if(vars.has("method.scene.vaoName")){
    auto&vaoName = vars.getString("method.scene.vaoName");
    vao = vars.get<ge::gl::VertexArray>(vaoName);
  }else return;
  if(vars.has("method.scene.prgName")){
    auto&prgName = vars.getString("method.scene.prgName");
    prg = vars.get<ge::gl::Program>(prgName);
  }else return;
  if(vars.has("method.scene.primitive")){
    primitive = *vars.get<GLenum>("method.scene.primitive");
  }else return;
  if(vars.has("method.scene.nofVertices")){
    nofVertices = vars.getUint32("method.scene.nofVertices");
  }else return;
  if(vars.has("method.scene.hasIndices")){
    hasIndices = vars.getBool("method.scene.hasIndices");
  }else return;

  vao->bind();
  prg->use();

  if(vars.has("method.scene.proj")){
    auto proj = *vars.get<glm::mat4>("method.scene.proj");
    prg->setMatrix4fv("proj",glm::value_ptr(proj));
  }

  if(vars.has("method.scene.view")){
    auto view = *vars.get<glm::mat4>("method.scene.view");
    prg->setMatrix4fv("view",glm::value_ptr(view));
  }

  if(hasIndices)
    glDrawElements(primitive,nofVertices,GL_UNSIGNED_INT,0);
  else
    glDrawArrays(primitive,0,nofVertices);

  vao->unbind();
}

void setScene(vars::Vars&vars    ,
    std::string const&vao        ,
    std::string const&prg        ,
    GLenum            primitive  ,
    uint32_t          nofVertices,
    bool              hasIndices ){
  vars .addOrGetString  ("method.scene.vaoName"    ) = vao        ;
  vars .addOrGetString  ("method.scene.prgName"    ) = prg        ;
  *vars.addOrGet<GLenum>("method.scene.primitive"  ) = primitive  ;
  vars.addOrGetUint32   ("method.scene.nofVertices") = nofVertices;
  vars.addOrGetBool     ("method.scene.hasIndices" ) = hasIndices ;
}

void drawBunny(vars::Vars&vars){
  initBunny(vars);

  setScene(vars,
    "method.bunny.vao"                       ,
    "method.bunny.prg"                       ,
    GL_TRIANGLES                             ,
    vars.getUint32("method.bunny.nofIndices"),
    true                                     );

  drawScene(vars);
}


void onInit(vars::Vars&vars){
  model::setUpCamera(vars);
  vars.get<basicCamera::OrbitCamera>("method.view")->setDistance(5.f);

  glClearColor(0.1,0.1,0.1,1);
  glEnable(GL_DEPTH_TEST);
}

void onDraw(vars::Vars&vars){
  model::computeProjectionMatrix(vars);

  glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
  *vars.addOrGet<glm::mat4>("method.scene.view") = vars.getReinterpret<basicCamera::CameraTransform >("method.view")->getView      ();
  *vars.addOrGet<glm::mat4>("method.scene.proj") = vars.getReinterpret<basicCamera::CameraProjection>("method.proj")->getProjection();
  drawBunny(vars);
}

void onQuit(vars::Vars&vars){
  vars.erase("method");
}

EntryPoint main = [](){
  methodManager::Callbacks clbs;
  clbs.onDraw        =              onDraw       ;
  clbs.onInit        =              onInit       ;
  clbs.onQuit        =              onQuit       ;
  clbs.onResize      = emptyWindow::onResize     ;
  clbs.onMouseMotion = model      ::onMouseMotion;
  MethodRegister::get().manager.registerMethod("pgr01.gsWireBunny",clbs);
};

}
