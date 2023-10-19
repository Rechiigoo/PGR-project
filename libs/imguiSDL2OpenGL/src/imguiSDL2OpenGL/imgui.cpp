#include <imguiDormon/imgui.h>
#include <imguiSDL2Dormon/EventHandler.h>
#include <imguiOpenGLDormon/Renderer.h>
#include <imguiSDL2OpenGL/imgui.h>
#include <SDL.h>

namespace imguiSDL2OpenGL{

class ImguiImpl{
  public:
    ImguiImpl(SDL_Window*window){
      IMGUI_CHECKVERSION();
      ImGui::CreateContext();
      ImGuiIO& io = ImGui::GetIO(); (void)io;
      imguiEventHandler = std::make_unique<imguiSDL2Dormon::EventHandler>(window);
      imguiRenderer = std::make_unique<imguiOpenGLDormon::Renderer>();
      ImGui::StyleColorsDark();

    }
    ~ImguiImpl(){
      imguiRenderer = nullptr;
      imguiEventHandler = nullptr;
      ImGui::DestroyContext();
    }
    bool processEvent(SDL_Event const* event){
      return imguiEventHandler->processEvent(event);
    }
    void newFrame(SDL_Window*window){
      imguiEventHandler->newFrame(window);
      ImGui::NewFrame();
    }
    void render(SDL_Window*window,void*context){
      ImGui::Render();
      SDL_GL_MakeCurrent(window, context);
      imguiRenderer->render(ImGui::GetDrawData());
    }
    std::unique_ptr<imguiSDL2Dormon::EventHandler>imguiEventHandler;
    std::unique_ptr<imguiOpenGLDormon::Renderer>imguiRenderer;
};

Imgui::Imgui(SDL_Window*window){
  impl = std::make_unique<ImguiImpl>(window);
}

Imgui::~Imgui(){
}

void Imgui::newFrame(SDL_Window*window){
  impl->newFrame(window);
}

bool Imgui::processEvent(SDL_Event const*event){
  return impl->processEvent(event);
}

void Imgui::render(SDL_Window*window,void*context){
  impl->render(window,context);
}
}
