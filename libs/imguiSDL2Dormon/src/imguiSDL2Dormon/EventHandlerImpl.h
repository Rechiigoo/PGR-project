#pragma once

#include <SDL.h>
#include <imguiDormon/imgui.h>

namespace imguiSDL2Dormon{

class EventHandlerImpl{
  public:
    EventHandlerImpl(SDL_Window*window,void*sdl_gl_context = nullptr);
    ~EventHandlerImpl();
    void newFrame(SDL_Window*window);
    bool processEvent(SDL_Event const* event);
    void updateMousePosAndButtons();
    void updateMouseCursor();
    bool processMouseWheel(SDL_Event const*event);
    bool processMouseButtonDown(SDL_Event const*event);
    bool processTextInput(SDL_Event const*event);
    bool processKey(SDL_Event const*event);
    void fillKeyMap();
    void fillMouseCursors();
    void setMousePosition();
    void captureMouse();
    bool isCursorNotVisible();
    void hideCursor();
    void showAndSetCursor();
    bool wasCursorNotChanged();
    SDL_Window* window = nullptr;
    Uint64      time = 0;
    bool        mousePressed[3] = { false, false, false };
    SDL_Cursor* mouseCursors[ImGuiMouseCursor_COUNT] = { 0 };
    char*       clipboardTextData = nullptr;
};

}
