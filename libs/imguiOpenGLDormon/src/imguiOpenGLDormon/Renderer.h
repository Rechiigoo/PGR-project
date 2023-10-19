#pragma once

#include<memory>
#include<imguiOpenGLDormon/imguiopengldormon_export.h>

struct ImDrawData;

namespace imguiOpenGLDormon{
  class RendererImpl;
  class Renderer;
}

struct ImDrawData;
class imguiOpenGLDormon::Renderer{
  public:
    IMGUIOPENGLDORMON_EXPORT Renderer(size_t version = 150);
	IMGUIOPENGLDORMON_EXPORT ~Renderer();
	IMGUIOPENGLDORMON_EXPORT void render(ImDrawData*data);
  private:
    friend class ImguiOpenGLImpl;
    std::unique_ptr<RendererImpl>impl;
};

