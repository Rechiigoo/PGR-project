#include<Vars/Vars.h>
#include<geGL/StaticCalls.h>
#include<geGL/geGL.h>
#include<framework/methodRegister.hpp>
#include<BasicCamera/OrbitCamera.h>
#include<BasicCamera/PerspectiveCamera.h>
#include<PGR/01/emptyWindow.hpp>
#include<PGR/01/vertexArrays.hpp>
#include<framework/model.hpp>


using namespace ge::gl;
using namespace vertexArrays;

#ifndef CMAKE_ROOT_DIR
#define CMAKE_ROOT_DIR "."
#endif

namespace model{

struct GPUTexture{
  GPUTexture(::Texture const&t){

// STUDENT TASK
// LOAD TEXTURE TO GPU
// look inside PGR/02/texture.cpp
// use declared GLuint id;

  }
  ~GPUTexture(){
    if(id!=0)glDeleteTextures(1,&id);
  }
  GLuint id;
};


struct GPUBuffer{
  GPUBuffer(::Buffer const&b){
    glCreateBuffers(1,&id);
    glNamedBufferData(id,b.size,b.ptr,GL_DYNAMIC_DRAW);
  }
  ~GPUBuffer(){
    glDeleteBuffers(1,&id);
  }
  GLuint id;
};

struct GPUMesh{
  GPUMesh(Mesh const&m,std::vector<std::shared_ptr<GPUBuffer>>const&b){
    glCreateVertexArrays(1,&vao);
    if(m.indexType == IndexType::UINT32)indexType = GL_UNSIGNED_INT  ;
    if(m.indexType == IndexType::UINT16)indexType = GL_UNSIGNED_SHORT;
    if(m.indexType == IndexType::UINT8 )indexType = GL_UNSIGNED_BYTE ;
    auto setAtt = [&](GLuint id,VertexAttrib const&att){
      if(att.type != AttributeType::EMPTY || att.bufferId < 0)
        setVertexAttrib(vao,id,(GLint)att.type,GL_FLOAT,b.at(att.bufferId)->id,att.offset,att.stride);
    };
    setAtt(0,m.position);
    setAtt(1,m.normal  );
    setAtt(2,m.texCoord);
    if(m.indexBuffer != -1)
      glVertexArrayElementBuffer(vao,b.at(m.indexBuffer)->id);
    hasIndices     = m.indexBuffer != -1;
    nofIndices     = m.nofIndices;
    indexOffset    = m.indexOffset;
    diffuseColor   = m.diffuseColor;
    diffuseTexture = m.diffuseTexture;
  }
  ~GPUMesh(){
    glDeleteVertexArrays(1,&vao);
  }
  GLenum    indexType      = GL_UNSIGNED_INT;
  GLuint    vao            = 0              ;
  bool      hasIndices     = false          ;
  uint32_t  nofIndices     = 0              ;
  uint64_t  indexOffset    = 0              ;
  glm::vec4 diffuseColor   = glm::vec4(1.f) ;
  int       diffuseTexture = -1             ;
};

struct GPUNode{
  GPUNode(Node const&n){
    modelMatrix = n.modelMatrix;
    mesh        = n.mesh       ;
    for(auto const&c:n.children)
      children.emplace_back(c);
  }
  glm::mat4           modelMatrix = glm::mat4(1.f);
  int                 mesh        = -1            ;
  std::vector<GPUNode>children                    ;
};

struct GPUModel{
  GPUModel(Model const&mdl){
    for(auto const&b:mdl.buffers)
      buffers.emplace_back(std::make_shared<GPUBuffer>(b));
    for(auto const&t:mdl.textures)
      textures.emplace_back(std::make_shared<GPUTexture>(t));
    for(auto const&m:mdl.meshes)
      meshes.emplace_back(std::make_shared<GPUMesh>(m,buffers));
    for(auto const&r:mdl.roots)
      roots.emplace_back(r);
  }
  std::vector<std::shared_ptr<GPUTexture>>textures;
  std::vector<std::shared_ptr<GPUMesh   >>meshes  ;
  std::vector<GPUNode                    >roots   ;
  std::vector<std::shared_ptr<GPUBuffer >>buffers ;
};

void drawNode(
    GPUNode  const&node,
    GPUModel const*model,
    ge::gl::Program*prg,
    glm::mat4 const&modelMatrix){

// STUDENT TASK
// draw node
//
// general pseodo code is:
// if(node.mesh >= 0){ 
//   //draw node
// }
//
// for(n node.children){
//   drawNode(n)
// }
//
// You have to set up the texture, vertex array, diffuseColor, useTexture variable in shader, modelMatrix
// and call glDrawElements or glDrawArrays is mesh has indices.
//
// set vertex array
// glBindVertexArray(mesh->vao)
//
// set up the texture
// glBindTextureUnit(0,...)
//
// set useTexture/diffuseColor uniform variables in shader
// prg->set1i("useTexture",1/0)
// prg->set4fv("diffuseColor",glm::value_ptr(mesh->diffuseColor))
//
// set the model matrix:
// prg->setMatrix4fv("model",glm::value_ptr(___MODEL___MATRIX___))
//
// call draw call
// glDrawElements(GL_TRIANGLES,mesh->nofIndices,mesh->indexType,(GLvoid*)mesh->indexOffset);

  for(auto const&n:node.children){
    drawNode(n,model,prg,modelMatrix*node.modelMatrix);
  }
}


void drawModel(GPUModel const*model,glm::mat4 const&proj,glm::mat4 const&view,ge::gl::Program*prg){

  prg->setMatrix4fv("proj",glm::value_ptr(proj));
  prg->setMatrix4fv("view",glm::value_ptr(view));

  for(auto const&root:model->roots)
    drawNode(root,model,prg,glm::mat4(1.f));
}

std::string const source = R".(

#ifdef VERTEX_SHADER
uniform mat4 view  = mat4(1.f);
uniform mat4 proj  = mat4(1.f);
uniform mat4 model = mat4(1.f);

layout(location = 0)in vec3 position;
layout(location = 1)in vec3 normal  ;
layout(location = 2)in vec2 texCoord;

out vec2 vCoord;

void main(){
  vCoord  = texCoord;

  gl_Position = proj*view*model*vec4(position,1.f);
}
#endif

#ifdef FRAGMENT_SHADER
in vec2 vCoord;
in vec3 vNormal;
in vec3 vPosition;

uniform sampler2D diffuseTexture;
uniform vec4      diffuseColor = vec4(1.f);
uniform int       useTexture   = 0;

uniform vec3 lightPosition = vec3(100,100,100);

layout(location=0)out vec4 fColor;
void main(){
  if(useTexture == 1){
    fColor = texture(diffuseTexture,vCoord);
    if(fColor.a < 0.01)discard;
  }else{
    fColor = diffuseColor;
  }
}
#endif
).";


void setUpCamera(vars::Vars&vars){
  vars.addFloat("method.sensitivity"    ,   0.01f);
  vars.addFloat("method.near"           ,   0.10f);
  vars.addFloat("method.far"            ,1000.00f);
  vars.addFloat("method.orbit.zoomSpeed",   0.10f);
  vars.reCreate<basicCamera::OrbitCamera      >("method.view");
  vars.reCreate<basicCamera::PerspectiveCamera>("method.proj");
}

void onInit(vars::Vars&vars){
  setUpCamera(vars);

  vars.reCreate<ge::gl::Program>("method.prg",
      std::make_shared<ge::gl::Shader>(GL_VERTEX_SHADER    ,"#version 460\n#define   VERTEX_SHADER\n"  +source),
      std::make_shared<ge::gl::Shader>(GL_FRAGMENT_SHADER  ,"#version 460\n#define FRAGMENT_SHADER\n"  +source)
      );

  vars.reCreate<ge::gl::VertexArray>("method.vao");

  ModelData mdl;
  //mdl.load(std::string(CMAKE_ROOT_DIR)+"/resources/models/konfucius/scene.gltf");
  mdl.load(std::string(CMAKE_ROOT_DIR)+"/resources/models/nyra/scene.gltf");
  //mdl.load(std::string(CMAKE_ROOT_DIR)+"/resources/models/china.glb");
  //mdl.load(std::string(CMAKE_ROOT_DIR)+"/resources/models/triss/scene.gltf");
  auto m = mdl.getModel();
  auto gm = GPUModel(m);

  vars.reCreate<GPUModel>("method.model",m);

  glClearColor(0.1,0.1,0.1,1);
  glEnable(GL_DEPTH_TEST);
}

void computeProjectionMatrix(vars::Vars&vars){
  auto width  = vars.getUint32("event.resizeX");
  auto height = vars.getUint32("event.resizeY");
  auto near   = vars.getFloat ("method.near"  );
  auto far    = vars.getFloat ("method.far"   );

  float aspect = static_cast<float>(width) / static_cast<float>(height);
  auto pr = vars.get<basicCamera::PerspectiveCamera>("method.proj");
  pr->setFar(far);
  pr->setNear(near);
  pr->setAspect(aspect);
}


void onDraw(vars::Vars&vars){
  computeProjectionMatrix(vars);

  auto proj = vars.get<basicCamera::PerspectiveCamera>("method.proj")->getProjection();
  auto view = vars.get<basicCamera::OrbitCamera      >("method.view")->getView      ();

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  auto prg = vars.get<ge::gl::Program    >("method.prg");
  auto vao = vars.get<ge::gl::VertexArray>("method.vao");

  prg
    ->setMatrix4fv("proj",glm::value_ptr(proj))
    ->setMatrix4fv("view",glm::value_ptr(view))
    ->use();
  vao->bind();

  glDrawArrays(GL_TRIANGLE_STRIP,0,4);

  drawModel(vars.get<GPUModel>("method.model"),proj,view,prg);
}

void onMouseMotion(vars::Vars&vars){

  auto width  = vars.getInt32("event.resizeX");
  auto height = vars.getInt32("event.resizeY");
  auto xrel = vars.getInt32("event.mouse.xrel");
  auto yrel = vars.getInt32("event.mouse.yrel");

  auto sensitivity = vars.getFloat("method.sensitivity"    );
  auto zoomSpeed   = vars.getFloat("method.orbit.zoomSpeed");

  auto&view = *vars.get<basicCamera::OrbitCamera>("method.view");

  if(vars.getBool("event.mouse.left")){
    view.addXAngle(sensitivity * yrel);
    view.addYAngle(sensitivity * xrel);
  }
  if(vars.getBool("event.mouse.middle")){
    view.addXPosition(+view.getDistance() * xrel /
                              float(width) * 2.f);
    view.addYPosition(-view.getDistance() * yrel /
                              float(height) * 2.f);
  }
  if(vars.getBool("event.mouse.right")){
    view.addDistance(zoomSpeed * yrel);
  }
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
  clbs.onMouseMotion =              onMouseMotion;
  MethodRegister::get().manager.registerMethod("pgr02.model",clbs);
};

}
