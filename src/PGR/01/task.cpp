#include<Vars/Vars.h>
#include<geGL/StaticCalls.h>
#include<framework/methodRegister.hpp>
#include<PGR/01/emptyWindow.hpp>
#include<PGR/01/compileShaders.hpp>
#include<PGR/01/vertexArrays.hpp>

using namespace ge::gl;
using namespace emptyWindow;
using namespace compileShaders;

namespace pgr_task01{

void onInit(vars::Vars&){
  glClearColor(0.1f,0.2f,0.2f,1.0f);
}

void onDraw(vars::Vars&){
  glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
}

void onQuit(vars::Vars&){

}

EntryPoint main = [](){
  methodManager::Callbacks clbs;
  clbs.onInit   = onInit;
  clbs.onQuit   = onQuit;
  clbs.onDraw   = onDraw;
  clbs.onResize = emptyWindow::onResize;
  MethodRegister::get().manager.registerMethod("pgr01.task",clbs);
};

}
