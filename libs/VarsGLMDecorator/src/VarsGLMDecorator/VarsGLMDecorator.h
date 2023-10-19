#pragma once

#include <iostream>
#include <glm/glm.hpp>

template<typename T>
class VarsGLMDecorator: public T{
  public:
    glm::uvec2& addUVec2(std::string const&n,glm::uvec2 const&v = glm::uvec2(0u));
    glm::uvec2& addUVec2(std::string const&n,uint32_t x=0,uint32_t y=0);
    glm::uvec2& getUVec2(std::string const&n);
    glm::uvec3& addUVec3(std::string const&n,glm::uvec3 const&v = glm::uvec3(0u));
    glm::uvec3& addUVec3(std::string const&n,uint32_t x=0,uint32_t y=0,uint32_t z=0);
    glm::uvec3& getUVec3(std::string const&n);
    glm::uvec4& addUVec4(std::string const&n,glm::uvec4 const&v = glm::uvec4(0u));
    glm::uvec4& addUVec4(std::string const&n,uint32_t x=0,uint32_t y=0,uint32_t z=0,uint32_t w=0);
    glm::uvec4& getUVec4(std::string const&n);
    glm::vec2& addVec2(std::string const&n,glm::vec2 const&v = glm::vec2(0.f));
    glm::vec2& addVec2(std::string const&n,float x = 0.f,float y = 0.f);
    glm::vec2& getVec2(std::string const&n);
    glm::vec3& addVec3(std::string const&n,glm::vec3 const&v = glm::vec3(0.f));
    glm::vec3& addVec3(std::string const&n,float x = 0.f,float y = 0.f,float z = 0.f);
    glm::vec3& getVec3(std::string const&n);
    glm::vec4& addVec4(std::string const&n,glm::vec4 const&v = glm::vec4(0.f));
    glm::vec4& addVec4(std::string const&n,float x = 0.f,float y = 0.f,float z = 0.f,float w = 0.f);
    glm::vec4& getVec4(std::string const&n);
    glm::mat2& addMat2(std::string const&n,glm::mat2 const&v = glm::mat2(1.f));
    glm::mat2& getMat2(std::string const&n);
    glm::mat3& addMat3(std::string const&n,glm::mat3 const&v = glm::mat3(1.f));
    glm::mat3& getMat3(std::string const&n);
    glm::mat4& addMat4(std::string const&n,glm::mat4 const&v = glm::mat4(1.f));
    glm::mat4& getMat4(std::string const&n);
    glm::mat2x3& addMat2x3(std::string const&n,glm::mat2x3 const&v = glm::mat2x3(1.f));
    glm::mat2x3& getMat2x3(std::string const&n);
    glm::mat2x4& addMat2x4(std::string const&n,glm::mat2x4 const&v = glm::mat2x4(1.f));
    glm::mat2x4& getMat2x4(std::string const&n);
    glm::mat3x2& addMat3x2(std::string const&n,glm::mat3x2 const&v = glm::mat3x2(1.f));
    glm::mat3x2& getMat3x2(std::string const&n);
    glm::mat3x4& addMat3x4(std::string const&n,glm::mat3x4 const&v = glm::mat3x4(1.f));
    glm::mat3x4& getMat3x4(std::string const&n);
};


template<typename T>glm::uvec2&  VarsGLMDecorator<T>::addUVec2(std::string const&n,glm::uvec2 const&v){
  return *T::template add<glm::uvec2>(n,v);
}
template<typename T>glm::uvec2&  VarsGLMDecorator<T>::addUVec2(std::string const&n,uint32_t x,uint32_t y){
  return *T::template add<glm::uvec2>(n,x,y);
}
template<typename T>glm::uvec2&  VarsGLMDecorator<T>::getUVec2(std::string const&n){
  return *T::template get<glm::uvec2>(n);
}
template<typename T>glm::uvec3&  VarsGLMDecorator<T>::addUVec3(std::string const&n,glm::uvec3 const&v){
  return *T::template add<glm::uvec3>(n,v);
}
template<typename T>glm::uvec3&  VarsGLMDecorator<T>::addUVec3(std::string const&n,uint32_t x,uint32_t y,uint32_t z){
  return *T::template add<glm::uvec3>(n,x,y,z);
}
template<typename T>glm::uvec3&  VarsGLMDecorator<T>::getUVec3(std::string const&n){
  return *T::template get<glm::uvec3>(n);
}
template<typename T>glm::uvec4&  VarsGLMDecorator<T>::addUVec4(std::string const&n,glm::uvec4 const&v){
  return *T::template add<glm::uvec4>(n,v);
}
template<typename T>glm::uvec4&  VarsGLMDecorator<T>::addUVec4(std::string const&n,uint32_t x,uint32_t y,uint32_t z,uint32_t w){
  return *T::template add<glm::uvec4>(n,x,y,z,w);
}
template<typename T>glm::uvec4&  VarsGLMDecorator<T>::getUVec4(std::string const&n){
  return *T::template get<glm::uvec4>(n);
}
template<typename T>glm::vec2&  VarsGLMDecorator<T>::addVec2(std::string const&n,glm::vec2 const&v){
  return *T::template add<glm::vec2>(n,v);
}
template<typename T>glm::vec2&  VarsGLMDecorator<T>::addVec2(std::string const&n,float x,float y){
  return *T::template add<glm::vec2>(n,x,y);
}
template<typename T>glm::vec2&  VarsGLMDecorator<T>::getVec2(std::string const&n){
  return *T::template get<glm::vec2>(n);
}
template<typename T>glm::vec3&  VarsGLMDecorator<T>::addVec3(std::string const&n,glm::vec3 const&v){
  return *T::template add<glm::vec3>(n,v);
}
template<typename T>glm::vec3&  VarsGLMDecorator<T>::addVec3(std::string const&n,float x,float y,float z){
  return *T::template add<glm::vec3>(n,x,y,z);
}
template<typename T>glm::vec3&  VarsGLMDecorator<T>::getVec3(std::string const&n){
  return *T::template get<glm::vec3>(n);
}
template<typename T>glm::vec4&  VarsGLMDecorator<T>::addVec4(std::string const&n,glm::vec4 const&v){
  return *T::template add<glm::vec4>(n,v);
}
template<typename T>glm::vec4&  VarsGLMDecorator<T>::addVec4(std::string const&n,float x,float y,float z,float w){
  return *T::template add<glm::vec4>(n,x,y,z,w);
}
template<typename T>glm::vec4&  VarsGLMDecorator<T>::getVec4(std::string const&n){
  return *T::template get<glm::vec4>(n);
}
template<typename T>glm::mat2&  VarsGLMDecorator<T>::addMat2(std::string const&n,glm::mat2 const&v){
  return *T::template add<glm::mat2>(n,v);
}
template<typename T>glm::mat2&  VarsGLMDecorator<T>::getMat2(std::string const&n){
  return *T::template get<glm::mat2>(n);
}
template<typename T>glm::mat3&  VarsGLMDecorator<T>::addMat3(std::string const&n,glm::mat3 const&v){
  return *T::template add<glm::mat3>(n,v);
}
template<typename T>glm::mat3&  VarsGLMDecorator<T>::getMat3(std::string const&n){
  return *T::template get<glm::mat3>(n);
}
template<typename T>glm::mat4&  VarsGLMDecorator<T>::addMat4(std::string const&n,glm::mat4 const&v){
  return *T::template add<glm::mat4>(n,v);
}
template<typename T>glm::mat4&  VarsGLMDecorator<T>::getMat4(std::string const&n){
  return *T::template get<glm::mat4>(n);
}
template<typename T>glm::mat2x3&  VarsGLMDecorator<T>::addMat2x3(std::string const&n,glm::mat2x3 const&v){
  return *T::template add<glm::mat2x3>(n,v);
}
template<typename T>glm::mat2x3&  VarsGLMDecorator<T>::getMat2x3(std::string const&n){
  return *T::template get<glm::mat2x3>(n);
}
template<typename T>glm::mat2x4&  VarsGLMDecorator<T>::addMat2x4(std::string const&n,glm::mat2x4 const&v){
  return *T::template add<glm::mat2x4>(n,v);
}
template<typename T>glm::mat2x4&  VarsGLMDecorator<T>::getMat2x4(std::string const&n){
  return *T::template get<glm::mat2x4>(n);
}
template<typename T>glm::mat3x2&  VarsGLMDecorator<T>::addMat3x2(std::string const&n,glm::mat3x2 const&v){
  return *T::template add<glm::mat3x2>(n,v);
}
template<typename T>glm::mat3x2&  VarsGLMDecorator<T>::getMat3x2(std::string const&n){
  return *T::template get<glm::mat3x2>(n);
}
template<typename T>glm::mat3x4&  VarsGLMDecorator<T>::addMat3x4(std::string const&n,glm::mat3x4 const&v){
  return *T::template add<glm::mat3x4>(n,v);
}
template<typename T>glm::mat3x4&  VarsGLMDecorator<T>::getMat3x4(std::string const&n){
  return *T::template get<glm::mat3x4>(n);
}
