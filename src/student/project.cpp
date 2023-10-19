/**
 * This file can be used as template for student project.
 *
 * There is a lot of hidden functionality not documented
 * If you have questions about the framework
 * email me: imilet@fit.vutbr.cz
 */

#include<glm/geometric.hpp>
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

using namespace ge::gl;
using namespace std;

namespace student::project{

void onKeyDown(vars::Vars&vars){
  ///which key was pressed
  auto key = vars.getInt32("event.key");
}

void onKeyUp(vars::Vars&vars){
  ///which key was released
  auto key = vars.getInt32("event.key");
}

void onMouseMotion(vars::Vars&vars){
  ///relative mouse movement
  auto xrel = vars.getInt32("event.mouse.xrel");
  auto yrel = vars.getInt32("event.mouse.yrel");

  ///mouse buttons
  auto left  = vars.getBool("event.mouse.left"  );
  auto mid   = vars.getBool("event.mouse.middle");
  auto right = vars.getBool("event.mouse.right" );

}

void onUpdate(vars::Vars&vars){
  ///time delta
  auto dt = vars.getFloat("event.dt");

}



void onInit(vars::Vars&vars){
  //create custom variable into namespace method with default value
  vars.addFloat("method.sensitivity"    ,  0.1f);

  /// add limits to variable
  addVarsLimitsF(vars,"method.sensitivity",-1.f,+1.f,0.01f);

  vars.addUint32("method.bufferSize",1024*sizeof(float));

  glClearColor(1,0,0,1);
}

void reactToSensitivityChange(vars::Vars&vars){
  /// this will end the function immidiatelly
  /// if none of listed variables was changed
  if(notChanged(vars,"method.all",__FUNCTION__,
        {"method.sensitivity",//list of variables
        }))return;

  std::cerr << "sensitivity was changed to: " << vars.getFloat("method.sensitivity") << std::endl;

}

void createBuffer(vars::Vars&vars){
  if(notChanged(vars,"method.all",__FUNCTION__,{"method.bufferSize"}))return;

  auto size = vars.getUint32("method.bufferSize");
  vars.reCreate<Buffer>("method.buffer",size);

  std::cerr << "buffer on GPU was recreated" << std::endl;
}

void onDraw(vars::Vars&vars){
  reactToSensitivityChange(vars);
  createBuffer(vars);


  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  /// addOrGet... with create variable if it does not exist
  /// and return its value
  /// it will be automatically added to GUI
  if(vars.addOrGetBool("method.shouldPrint")){
    std::cerr << "Hello World!" << std::endl;
  }

}

void onResize(vars::Vars&vars){
  /// size of the screen
  auto width  = vars.getUint32("event.resizeX");
  auto height = vars.getUint32("event.resizeY");

  glViewport(0,0,width,height);
}

void onQuit(vars::Vars&vars){
  /// if you created everything inside "method" namespace
  /// this line should be enough to clear everything...
  vars.erase("method");
}

/// this will register your method into menus and stuff like that...
/// it is using static value initialization and singleton concept...
EntryPoint main = [](){
  /// table of callbacks
  methodManager::Callbacks clbs;
  clbs.onDraw        = onDraw       ;
  clbs.onInit        = onInit       ;
  clbs.onQuit        = onQuit       ;
  clbs.onResize      = onResize     ;
  clbs.onKeyDown     = onKeyDown    ;
  clbs.onKeyUp       = onKeyUp      ;
  clbs.onMouseMotion = onMouseMotion;
  clbs.onUpdate      = onUpdate     ;

  /// register method
  MethodRegister::get().manager.registerMethod("student.project",clbs);
};

}
