#pragma once

#include<memory>
#include<imguiSDL2OpenGL/imguisdl2opengl_export.h>
#include <imguiDormon/imgui.h>

struct SDL_Window;
typedef union SDL_Event SDL_Event;

namespace imguiSDL2OpenGL{
  class Imgui;
  class ImguiImpl;
}

class imguiSDL2OpenGL::Imgui{
  public:
    IMGUISDL2OPENGL_EXPORT Imgui(SDL_Window*window);
	IMGUISDL2OPENGL_EXPORT ~Imgui();
	IMGUISDL2OPENGL_EXPORT void newFrame(SDL_Window*window);
	IMGUISDL2OPENGL_EXPORT void render(SDL_Window*window,void*context);
	IMGUISDL2OPENGL_EXPORT bool processEvent(SDL_Event const* event);
  protected:
    friend class ImguiImpl;
    std::unique_ptr<ImguiImpl>impl;
};

