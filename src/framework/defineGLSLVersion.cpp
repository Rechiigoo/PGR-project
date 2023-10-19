#include<framework/defineGLSLVersion.hpp>
#include<sstream>
#include<geGL/StaticCalls.h>

using namespace ge::gl;

std::string defineGLSLVersion(){
  std::stringstream ss;
  GLint major;
  GLint minor;
  glGetIntegerv(GL_MAJOR_VERSION,&major);
  glGetIntegerv(GL_MAJOR_VERSION,&minor);
  ss << "#version ";
  ss << major;
  ss << minor;
  ss << "0";
  ss << std::endl;
  return ss.str();
}
