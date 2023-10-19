#pragma once

#include<memory>
#include<imguiSDL2Dormon/imguisdl2dormon_export.h>

struct SDL_Window;
typedef union SDL_Event SDL_Event;

namespace imguiSDL2Dormon{
  class EventHandlerImpl;
  class EventHandler;
}

class imguiSDL2Dormon::EventHandler{
  public:
    IMGUISDL2DORMON_EXPORT EventHandler(SDL_Window*window,void*sdl_gl_context = nullptr);
	IMGUISDL2DORMON_EXPORT ~EventHandler();
	IMGUISDL2DORMON_EXPORT void newFrame(SDL_Window*window);
	IMGUISDL2DORMON_EXPORT bool processEvent(SDL_Event const* event);
  protected:
    friend class EventHandlerImpl;
    std::unique_ptr<EventHandlerImpl>impl;
};

