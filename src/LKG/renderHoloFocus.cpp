#include<fstream>
#include<SDL.h>
#include<glm/glm.hpp>
#include<glm/gtc/matrix_transform.hpp>
#include<glm/gtc/type_ptr.hpp>
#include<Vars/Vars.h>
#include<geGL/geGL.h>
#include<geGL/StaticCalls.h>
#include<framework/methodRegister.hpp>
#include<BasicCamera/OrbitCamera.h>
#include<BasicCamera/FreeLookCamera.h>
#include<BasicCamera/PerspectiveCamera.h>
#include<framework/Timer.hpp>
#include<framework/Barrier.h>
#include<framework/FunctionPrologue.h>
#include <imguiSDL2OpenGL/imgui.h>
#include <imguiVars/addVarsLimits.h>
#include<stb_image.h>

using namespace ge::gl;

namespace renderHoloFocus{

#ifndef CMAKE_ROOT_DIR
#define CMAKE_ROOT_DIR "."
#endif

void createTrianglesProgram(vars::Vars&vars){
  if(notChanged(vars,"all",__FUNCTION__,{}))return;

  std::string const vsSrc = R".(
  void main(){
  }
  ).";

  std::string const gsSrc = R".(

  layout(points)in;
  layout(triangle_strip,max_vertices=30)out;

  uniform mat4 projection;
  uniform mat4 view;
  out vec3 gPosition;
  out vec3 gNormal;
  out vec4 gColor;

  void genTriangle(vec3 a,vec3 b,vec3 c,vec3 color){
    vec3 n = normalize(cross(b-a,c-a));

    gPosition = a;
    gNormal = n;
    gColor = vec4(color,1);
    gl_Position = projection*view*vec4(a,1);
    EmitVertex();

    gPosition = b;
    gNormal = n;
    gColor = vec4(color,1);
    gl_Position = projection*view*vec4(b,1);
    EmitVertex();

    gPosition = c;
    gNormal = n;
    gColor = vec4(color,1);
    gl_Position = projection*view*vec4(c,1);
    EmitVertex();

    EndPrimitive();
  }

  void main(){
    genTriangle(vec3(-1,0,-1),vec3(+1,0,-1),vec3(+1,+1,-1),vec3(1,0,0));
    genTriangle(vec3(-3,0,-2),vec3(+3,0,-2),vec3(+3,+3,-2),vec3(0,1,0));
    genTriangle(vec3(-10,0,-4),vec3(+10,0,-4),vec3(+10,+10,-4),vec3(0,0,1));
    //gColor = vec4(0,0,0,1);
    //gl_Position = vec4(-1,-1,0,1);EmitVertex();
    //gl_Position = vec4(4,-1,0,1);EmitVertex();
    //gl_Position = vec4(-1,4,0,1);EmitVertex();
    //EndPrimitive();
  }

  ).";

  std::string const fsSrc = R".(
  out vec4 fColor;
  in vec4 gColor;
  in vec3 gPosition;
  in vec3 gNormal;
  uniform mat4 view     = mat4(1);
  uniform vec3 lightPos = vec3(0,0,10);
  uniform uvec2 size = uvec2(1024,512);

  void main(){
    vec3 cameraPos = vec3(inverse(view)*vec4(0,0,0,1));
    vec3 N = normalize(gNormal);
    vec3 L = normalize(lightPos - gPosition);
    vec3 V = normalize(cameraPos - gPosition);
    vec3 R = -reflect(L,N);
    float df = abs(dot(N,L));
    float sf = pow(abs(dot(R,V)),100);
    fColor = gColor * df + vec4(sf);
    uvec2 pixSize = uvec2(1,1);
    //if(any(greaterThanEqual(gl_FragCoord.xy,size)))discard;
    uvec2 coord = uvec2(gl_FragCoord.xy/size);
    uint offset = 0;//coord.x + coord.y*5;

    //if((uint(gl_FragCoord.x)%size.x) >= 100+offset && (uint(gl_FragCoord.y)%size.y) >= 100 && (uint(gl_FragCoord.x)%size.x) < 100+pixSize.x+offset &&(uint(gl_FragCoord.y)%size.y) < 100+pixSize.y)
    //  fColor = vec4(0,1,0,1);
//    fColor = gColor;
  }
  ).";

  auto vs = std::make_shared<ge::gl::Shader>(GL_VERTEX_SHADER,
      "#version 450\n",
      vsSrc
      );
  auto gs = std::make_shared<ge::gl::Shader>(GL_GEOMETRY_SHADER,
      "#version 450\n",
      gsSrc
      );
  auto fs = std::make_shared<ge::gl::Shader>(GL_FRAGMENT_SHADER,
      "#version 450\n",
      fsSrc
      );
  vars.reCreate<ge::gl::Program>("trianglesProgram",vs,gs,fs);

}

void drawTriangles(vars::Vars&vars,glm::mat4 const&view,glm::mat4 const&proj){
  createTrianglesProgram(vars);
  ge::gl::glEnable(GL_DEPTH_TEST);
  vars.get<ge::gl::Program>("method.trianglesProgram")
    ->setMatrix4fv("view"      ,glm::value_ptr(view))
    ->setMatrix4fv("projection",glm::value_ptr(proj))
    ->use();
  ge::gl::glDrawArrays(GL_POINTS,0,1);
}

void drawTriangles(vars::Vars&vars){
  auto view = vars.getReinterpret<basicCamera::CameraTransform>("view");
  auto projection = vars.get<basicCamera::PerspectiveCamera>("projection");
  drawTriangles(vars,view->getView(),projection->getProjection());
}


void createView(vars::Vars&vars){
  if(notChanged(vars,"all",__FUNCTION__,{"method.useOrbitCamera"}))return;

  if(vars.getBool("method.useOrbitCamera"))
    vars.reCreate<basicCamera::OrbitCamera>("method.view");
  else
    vars.reCreate<basicCamera::FreeLookCamera>("method.view");
}

void createProjection(vars::Vars&vars){
  if(notChanged(vars,"all",__FUNCTION__,{"method.windowSize","method.camera.fovy","method.camera.near","method.camera.far"}))return;

  auto windowSize = vars.get<glm::uvec2>("method.windowSize");
  auto width = windowSize->x;
  auto height = windowSize->y;
  auto aspect = (float)width/(float)height;
  auto nearv = vars.getFloat("method.camera.near");
  auto farv  = vars.getFloat("method.camera.far" );
  auto fovy = vars.getFloat("method.camera.fovy");

  vars.reCreate<basicCamera::PerspectiveCamera>("method.projection",fovy,aspect,nearv,farv);
}

void createCamera(vars::Vars&vars){
  createProjection(vars);
  createView(vars);
}

void create3DCursorProgram(vars::Vars&vars){
  if(notChanged(vars,"all",__FUNCTION__,{}))return;

  std::string const vsSrc = R".(
  uniform mat4 projection = mat4(1);
  uniform mat4 view       = mat4(1);
  uniform mat4 origView   = mat4(1);
  uniform float distance = 10;
  out vec3 vColor;
  void main(){
    float size = 1;
    vColor = vec3(1,0,0);

    if(gl_VertexID == 0)gl_Position = projection * view * inverse(origView) * vec4(vec2(-10  ,+0   )*size,-distance,1);
    if(gl_VertexID == 1)gl_Position = projection * view * inverse(origView) * vec4(vec2(+10  ,+0   )*size,-distance,1);
    if(gl_VertexID == 2)gl_Position = projection * view * inverse(origView) * vec4(vec2(-10  ,+0.1 )*size,-distance,1);
    if(gl_VertexID == 3)gl_Position = projection * view * inverse(origView) * vec4(vec2(+0   ,-10  )*size,-distance,1);
    if(gl_VertexID == 4)gl_Position = projection * view * inverse(origView) * vec4(vec2(+0   ,+10  )*size,-distance,1);
    if(gl_VertexID == 5)gl_Position = projection * view * inverse(origView) * vec4(vec2(+0.1 ,-10  )*size,-distance,1);
  }
  ).";

  std::string const fsSrc = R".(
  out vec4 fColor;
  in vec3 vColor;
  void main(){
    fColor = vec4(vColor,1);
  }
  ).";

  auto vs = std::make_shared<ge::gl::Shader>(GL_VERTEX_SHADER,
      "#version 450\n",
      vsSrc
      );
  auto fs = std::make_shared<ge::gl::Shader>(GL_FRAGMENT_SHADER,
      "#version 450\n",
      fsSrc
      );
  vars.reCreate<ge::gl::Program>("method.cursorProgram",vs,fs);

}

void draw3DCursor(vars::Vars&vars,glm::mat4 const&view,glm::mat4 const&proj){
  create3DCursorProgram(vars);
  auto origView = vars.getReinterpret<basicCamera::CameraTransform>("method.view")->getView();
  vars.get<ge::gl::Program>("method.cursorProgram")
    ->setMatrix4fv("projection",glm::value_ptr(proj))
    ->setMatrix4fv("view"      ,glm::value_ptr(view))
    ->setMatrix4fv("origView"  ,glm::value_ptr(origView))
    ->set1f       ("distance"  ,vars.getFloat("method.quiltRender.d"))
    ->use();
  ge::gl::glDrawArrays(GL_TRIANGLES,0,6);
}

void draw3DCursor(vars::Vars&vars){
  auto view = vars.getReinterpret<basicCamera::CameraTransform>("method.view");
  auto projection = vars.get<basicCamera::PerspectiveCamera>("method.projection");
  draw3DCursor(vars,view->getView(),projection->getProjection());
}

void loadColorTexture(vars::Vars&vars){
  if(notChanged(vars,"all",__FUNCTION__,{"method.quiltFileName"}))return;
  int w,h,channels;
  uint8_t* data = stbi_load(vars.getString("method.quiltFileName").c_str(),&w,&h,&channels,0);

  uint8_t* flippedData = new uint8_t[w*h*channels];
  for(int y=0;y<h;++y)
    for(int x=0;x<w;++x)
      for(int c=0;c<channels;++c)
        flippedData[((h-y-1)*w+x)*channels+c]=data[(y*w+x)*channels+c];

  GLenum format;
  GLenum type = GL_UNSIGNED_BYTE;
  if(channels == 3){
    format = GL_RGB;
  }
  if(channels == 4){
    format = GL_RGBA;
  }

  auto colorTex = vars.reCreate<ge::gl::Texture>(
      "method.quiltTex",GL_TEXTURE_2D,GL_RGB8,1,w,h);
  ge::gl::glPixelStorei(GL_UNPACK_ROW_LENGTH,w);
  ge::gl::glPixelStorei(GL_UNPACK_ALIGNMENT ,1);
  ge::gl::glTextureSubImage2D(colorTex->getId(),0,0,0,w,h,format,type,flippedData);
  delete[]flippedData;
  stbi_image_free(data);
}


void loadTextures(vars::Vars&vars){
  loadColorTexture(vars);
}

void createHoloProgram(vars::Vars&vars){
  if(notChanged(vars,"all",__FUNCTION__,{}))return;

  std::string const vsSrc = R".(
  #version 450 core

  out vec2 texCoords;

  void main(){
    texCoords = vec2(gl_VertexID&1,gl_VertexID>>1);
    gl_Position = vec4(texCoords*2-1,0,1);
  }
  ).";

  std::string const fsSrc = R".(
  #version 450 core
  
  in vec2 texCoords;
  
  layout(location=0)out vec4 fragColor;
  
  uniform int showQuilt = 0;
  uniform int showAsSequence = 0;
  uniform uint selectedView = 0;
  // HoloPlay values
  uniform float pitch = 354.677f;
  uniform float tilt = -0.113949f;
  uniform float center = -0.400272f;
  uniform float invView = 1.f;
  uniform float flipX;
  uniform float flipY;
  uniform float subp = 0.00013f;
  uniform int ri = 0;
  uniform int bi = 2;
  uniform vec4 tile = vec4(5,9,45,45);
  uniform vec4 viewPortion = vec4(0.99976f, 0.99976f, 0.00f, 0.00f);
  uniform vec4 aspect;
  uniform uint drawOnlyOneImage = 0;
  
  layout(binding=0)uniform sampler2D screenTex;
  
  uniform float focus = 0.f;

  vec2 texArr(vec3 uvz)
  {
      // decide which section to take from based on the z.
 

      float z = floor(uvz.z * tile.z);
      float focusMod = focus*(1-2*clamp(z/tile.z,0,1));
      float x = (mod(z, tile.x) + clamp(uvz.x+focusMod,0,1)) / tile.x;
      float y = (floor(z / tile.x) + uvz.y) / tile.y;
      return vec2(x, y) * viewPortion.xy;
  }
  
  void main()
  {
  	vec3 nuv = vec3(texCoords.xy, 0.0);
  
  	vec4 rgb[3];
  	for (int i=0; i < 3; i++) 
  	{
  		nuv.z = (texCoords.x + i * subp + texCoords.y * tilt) * pitch - center;
  		//nuv.z = mod(nuv.z + ceil(abs(nuv.z)), 1.0);
  		//nuv.z = (1.0 - invView) * nuv.z + invView * (1.0 - nuv.z);
  		nuv.z = fract(nuv.z);
  		nuv.z = (1.0 - nuv.z);
      if(drawOnlyOneImage == 1){
        if(uint(nuv.z *tile.z) == selectedView || uint(nuv.z *tile.z) == 19)
  		    rgb[i] = texture(screenTex, texArr(nuv));
        else
          rgb[i] = vec4(0);
      }else{
  		  rgb[i] = texture(screenTex, texArr(nuv));
      }
  		//rgb[i] = vec4(nuv.z, nuv.z, nuv.z, 1.0);
  	}
  
      if(showQuilt == 0)
        fragColor = vec4(rgb[ri].r, rgb[1].g, rgb[bi].b, 1.0);
      else{
        if(showAsSequence == 0)
          fragColor = texture(screenTex, texCoords.xy);
        else{
          uint sel = min(selectedView,uint(tile.x*tile.y-1));
          fragColor = texture(screenTex, texCoords.xy/vec2(tile.xy) + vec2(vec2(1.f)/tile.xy)*vec2(sel%uint(tile.x),sel/uint(tile.x)));
          
        }
      }
  }
  ).";

  auto vs = std::make_shared<ge::gl::Shader>(GL_VERTEX_SHADER,
      vsSrc
      );
  auto fs = std::make_shared<ge::gl::Shader>(GL_FRAGMENT_SHADER,
      fsSrc
      );
  auto prg = vars.reCreate<ge::gl::Program>("method.holoProgram",vs,fs);
  prg->setNonexistingUniformWarning(false);
}

class Quilt{
  public:
    glm::uvec2 counts = glm::uvec2(5,9);
    glm::uvec2 baseRes = glm::uvec2(380,238);
    glm::uvec2 res = glm::uvec2(1024,512);
    std::shared_ptr<ge::gl::Framebuffer>fbo;
    std::shared_ptr<ge::gl::Texture>color;
    std::shared_ptr<ge::gl::Texture>depth;
    vars::Vars&vars;
    void createTextures(){
      if(notChanged(vars,"all",__FUNCTION__,{"method.quiltRender.texScale","method.quiltRender.texScaleAspect"}))return;
      float texScale = vars.getFloat("method.quiltRender.texScale");
      float texScaleAspect =  vars.getFloat("method.quiltRender.texScaleAspect");
      auto newRes = glm::uvec2(glm::vec2(baseRes) * texScale * glm::vec2(texScaleAspect,1.f));
      if(newRes == res)return;
      res = newRes;
      std::cerr << "reallocate quilt textures - " << res.x << " x " << res.y << std::endl;
      fbo = std::make_shared<ge::gl::Framebuffer>();
      color = std::make_shared<ge::gl::Texture>(GL_TEXTURE_2D,GL_RGB8,1,res.x*counts.x,res.y*counts.y);
      color->texParameteri(GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);
      color->texParameteri(GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
      depth = std::make_shared<ge::gl::Texture>(GL_TEXTURE_RECTANGLE,GL_DEPTH24_STENCIL8,1,res.x*counts.x,res.y*counts.y);
      depth->texParameteri(GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);
      depth->texParameteri(GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
      fbo->attachTexture(GL_COLOR_ATTACHMENT0,color);
      fbo->attachTexture(GL_DEPTH_ATTACHMENT,depth);
      GLenum buffers[] = {GL_COLOR_ATTACHMENT0};
      fbo->drawBuffers(1,buffers);
    }
    Quilt(vars::Vars&vars):vars(vars){
      createTextures();
    }
    void draw(std::function<void(glm::mat4 const&view,glm::mat4 const&proj)>const&fce,glm::mat4 const&centerView,glm::mat4 const&centerProj){
      createTextures();
      GLint origViewport[4];
      ge::gl::glGetIntegerv(GL_VIEWPORT,origViewport);

      fbo->bind();
      ge::gl::glClearColor(1,0,0,1);
      ge::gl::glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
      size_t counter = 0;

      for(size_t j=0;j<counts.y;++j)
        for(size_t i=0;i<counts.x;++i){
          ge::gl::glViewport(i*(res.x),j*(res.y),res.x,res.y);


          float const fov = glm::radians<float>(vars.getFloat("method.quiltRender.fov"));
          float const size = vars.getFloat("method.quiltRender.size");
          float const camDist  =  size / glm::tan(fov * 0.5f); /// ?
          float const viewCone = glm::radians<float>(vars.getFloat("method.quiltRender.viewCone")); /// ?
          float const aspect = static_cast<float>(res.x) / static_cast<float>(res.y);
          float const viewConeSweep = -camDist * glm::tan(viewCone);
			    float const projModifier = 1.f / (size * aspect);
          auto const numViews = counts.x * counts.y;
          float currentViewLerp = 0.f; // if numviews is 1, take center view
          if (numViews > 1)
            currentViewLerp = (float)counter / (numViews - 1) - 0.5f;

          // .5*size*tan(cone)/tan(fov/2)
            // .5*tan(cone)/tan(fov/2)/aspect
          

          glm::mat4 view = centerView;
          glm::mat4 proj = centerProj;

          float t = (float)counter / (float)(numViews - 1);

          float d = vars.addOrGetFloat("method.quiltRender.d",0.70f);
          addVarsLimitsF(vars,"method.quiltRender.d",0,400,0.01);
          //float S = vars.addOrGetFloat("quiltRender.S",0.422f);
          //addVarsLimitsF(vars,"quiltRender.S",0,4,0.01);
          //view[3][0] += d - 2*d*t;
          //proj[2][0] += (d - 2*d*t)*S;
          
          float S = 0.5f*d*glm::tan(viewCone);
          float s = S-2*t*S;
          view[3][0] += s;
          proj[2][0] += s/(d*aspect*glm::tan(vars.getFloat("method.camera.fovy")/2));

          //std::cerr <<currentViewLerp * viewConeSweep << std::endl;
          //std::cerr << "view[3][0] " << view[3][0] << std::endl;
          //std::cerr << "proj[2][0] " << proj[2][0] << std::endl;
          //view[3][0] += currentViewLerp * viewConeSweep;
          //proj[2][0] += currentViewLerp * viewConeSweep * projModifier;

          fce(view,proj);
          counter++;
        }
      fbo->unbind();

      ge::gl::glViewport(origViewport[0],origViewport[1],origViewport[2],origViewport[3]);
    }
};

void drawHolo(vars::Vars&vars){
  loadTextures(vars);
  createHoloProgram(vars);

  if(vars.getBool("method.renderQuilt")){
    vars.get<Quilt>("method.quilt")->color->bind(0);
  }else{
    vars.get<ge::gl::Texture>("method.quiltTex")->bind(0);
  }
  vars.get<ge::gl::Program>("method.holoProgram")
    ->set1i ("showQuilt"       ,                vars.getBool       ("method.showQuilt"            ))
    ->set1i ("showAsSequence"  ,                vars.getBool       ("method.showAsSequence"       ))
    ->set1ui("selectedView"    ,                vars.getUint32     ("method.selectedView"         ))
    ->set1i ("showQuilt"       ,                vars.getBool       ("method.showQuilt"            ))
    ->set1f ("pitch"           ,                vars.getFloat      ("method.quiltView.pitch"      ))
    ->set1f ("tilt"            ,                vars.getFloat      ("method.quiltView.tilt"       ))
    ->set1f ("center"          ,                vars.getFloat      ("method.quiltView.center"     ))
    ->set1f ("invView"         ,                vars.getFloat      ("method.quiltView.invView"    ))
    ->set1f ("subp"            ,                vars.getFloat      ("method.quiltView.subp"       ))
    ->set1i ("ri"              ,                vars.getInt32      ("method.quiltView.ri"         ))
    ->set1i ("bi"              ,                vars.getInt32      ("method.quiltView.bi"         ))
    ->set4fv("tile"            ,glm::value_ptr(*vars.get<glm::vec4>("method.quiltView.tile"       )))
    ->set4fv("viewPortion"     ,glm::value_ptr(*vars.get<glm::vec4>("method.quiltView.viewPortion")))
    ->set1ui("drawOnlyOneImage",                vars.getBool       ("method.drawOnlyOneImage"     ))
    ->set1f ("focus"           ,                vars.getFloat      ("method.quiltView.focus"      ))
    ->use();

  ge::gl::glDrawArrays(GL_TRIANGLE_STRIP,0,4);
}


void onResize(vars::Vars&vars){
  auto windowSize = vars.get<glm::uvec2>("method.windowSize");
  windowSize->x = vars.addOrGetUint32("event.resizeX");
  windowSize->y = vars.addOrGetUint32("event.resizeY"); 
  vars.updateTicks("method.windowSize");
  ge::gl::glViewport(0,0,windowSize->x,windowSize->y);
  std::cerr << "resize(" << windowSize->x << "," << windowSize->y << ")" << std::endl;
}

void onInit(vars::Vars&vars){
  //auto args = vars.add<argumentViewer::ArgumentViewer>("args",argc,argv);
  //auto const quiltFile = args->gets("--quilt","","quilt image 5x9");
  //auto const showHelp = args->isPresent("-h","shows help");
  //if (showHelp || !args->validate()) {
  //  std::cerr << args->toStr();
  //  exit(0);
  //}

  //vars.addString("quiltFileName",quiltFile);
  vars.addString("method.quiltFileName",std::string(CMAKE_ROOT_DIR)+"/resources/images/witcher1.jpg");

  auto width  = vars.addOrGetUint32("event.resizeX");
  auto height = vars.addOrGetUint32("event.resizeY");

  vars.add<ge::gl::VertexArray>("method.emptyVao");
  vars.add<glm::uvec2>("method.windowSize",width,height);
  vars.addFloat("method.input.sensitivity",0.01f);
  vars.addFloat("method.camera.fovy",glm::half_pi<float>());
  vars.addFloat("method.camera.near",.1f);
  vars.addFloat("method.camera.far",1000.f);
  vars.addBool ("method.useOrbitCamera",false);

  vars.addFloat      ("method.quiltView.pitch"      ,354.677f);
  vars.addFloat      ("method.quiltView.tilt"       ,-0.113949f);
  vars.addFloat      ("method.quiltView.center"     ,-0.400272);
  vars.addFloat      ("method.quiltView.invView"    ,1);
  vars.addFloat      ("method.quiltView.subp"       ,0.000130208);
  vars.addInt32      ("method.quiltView.ri"         ,0);
  vars.addInt32      ("method.quiltView.bi"         ,2);
  vars.add<glm::vec4>("method.quiltView.tile"       ,5.00f, 9.00f, 45.00f, 45.00f);
  vars.add<glm::vec4>("method.quiltView.viewPortion",0.99976f, 0.99976f, 0.00f, 0.00f);
  vars.addFloat      ("method.quiltView.focus"      ,0.00f);
  addVarsLimitsF(vars,"method.quiltView.focus",-1,+1,0.001f);
  vars.addBool ("method.showQuilt");
  vars.addBool ("method.renderQuilt");
  vars.addBool ("method.renderScene",false);
  vars.addBool ("method.showAsSequence",false);
  vars.addBool ("method.drawOnlyOneImage",false);
  vars.addUint32("method.selectedView",0);
  addVarsLimitsU(vars,"method.selectedView",0,44);
  addVarsLimitsF(vars,"method.quiltView.tilt",-10,10,0.01);

  vars.addFloat("method.quiltRender.size",5.f);
  vars.addFloat("method.quiltRender.fov",90.f);
  vars.addFloat("method.quiltRender.viewCone",10.f);
  vars.addFloat("method.quiltRender.texScale",1.64f);
  addVarsLimitsF(vars,"method.quiltRender.texScale",0.1f,5,0.01f);
  vars.addFloat("method.quiltRender.texScaleAspect",0.745f);
  addVarsLimitsF(vars,"method.quiltRender.texScaleAspect",0.1f,10,0.01f);
  

  vars.add<Quilt>("method.quilt",vars);

  createCamera(vars);
  
  GLint dims[4];
  ge::gl::glGetIntegerv(GL_MAX_VIEWPORT_DIMS, dims);
  std::cerr << "maxFramebuffer: " << dims[0] << " x " << dims[1] << std::endl;

  ImGui::GetStyle().ScaleAllSizes(4.f);
  ImGui::GetIO().FontGlobalScale = 4.f;
}

void onMouseMotion(vars::Vars&vars){
}

//template<bool DOWN>
//void key(vars::Vars&vars){
//  auto keys = vars.get<std::map<SDL_Keycode, bool>>("input.keyDown");
//  (*keys)[event.key.keysym.sym] = DOWN;
//  if(event.key.keysym.sym == SDLK_f && DOWN){
//    fullscreen = !fullscreen;
//    if(fullscreen)
//      window->setFullscreen(sdl2cpp::Window::FULLSCREEN_DESKTOP);
//    else
//      window->setFullscreen(sdl2cpp::Window::WINDOW);
//  }
//}

std::map<SDL_Keycode,bool>keys;

void onKeyDown(vars::Vars&vars){
  keys[vars.getInt32("event.key")] = true;
  if(vars.getInt32("event.key") == SDLK_f){
    auto&full = vars.addOrGetBool("method.isFullscreen");
    full = !full;
    auto window = vars.get<SDL_Window*>("window");
    SDL_SetWindowFullscreen(*window,full);
  }
}

void onKeyUp(vars::Vars&vars){
  keys[vars.getInt32("event.key")] = false;
}

void onUpdate(vars::Vars&vars){
}

void onDraw(vars::Vars&vars){
  ge::gl::glClear(GL_DEPTH_BUFFER_BIT);
  createCamera(vars);
  basicCamera::CameraTransform*view;

  if(vars.getBool("method.useOrbitCamera"))
    view = vars.getReinterpret<basicCamera::CameraTransform>("method.view");
  else{
    auto freeView = vars.get<basicCamera::FreeLookCamera>("method.view");
    float freeCameraSpeed = 0.01f;
    for (int a = 0; a < 3; ++a)
      freeView->move(a, float((keys)["d s"[a]] - (keys)["acw"[a]]) *
                            freeCameraSpeed);
    view = freeView;
  }


  ge::gl::glClearColor(0.1f,0.1f,0.1f,1.f);
  ge::gl::glClear(GL_COLOR_BUFFER_BIT);

  vars.get<ge::gl::VertexArray>("method.emptyVao")->bind();
  auto drawScene = [&](glm::mat4 const&view,glm::mat4 const&proj){
    vars.get<ge::gl::VertexArray>("method.emptyVao")->bind();
    /////drawGrid(vars,view,proj);
    vars.get<ge::gl::VertexArray>("method.emptyVao")->unbind();
    //vars.get<ge::gl::VertexArray>("emptyVao")->bind();
    //drawTriangles(vars,view,proj);
    //vars.get<ge::gl::VertexArray>("emptyVao")->bind();
    /////drawBunny(vars,view,proj);
    if(vars.addOrGetBool("method.quiltRender.drawCursor",true)){
      vars.get<ge::gl::VertexArray>("method.emptyVao")->bind();
      draw3DCursor(vars,view,proj);
      vars.get<ge::gl::VertexArray>("method.emptyVao")->unbind();
    }
  };
  auto drawSceneSimple = [&](){
    auto view = vars.getReinterpret<basicCamera::CameraTransform>("method.view")->getView();
    auto proj = vars.getReinterpret<basicCamera::CameraProjection>("method.projection")->getProjection();
    drawScene(view,proj);
  };
  if(vars.getBool("method.renderScene")){
    drawSceneSimple();
  }
  else{
    auto quilt = vars.get<Quilt>("method.quilt");
    vars.get<ge::gl::VertexArray>("method.emptyVao")->bind();
    quilt->draw(
        drawScene,
        vars.getReinterpret<basicCamera::CameraTransform>("method.view")->getView(),
        vars.getReinterpret<basicCamera::CameraProjection>("method.projection")->getProjection()
        );
    vars.get<ge::gl::VertexArray>("method.emptyVao")->unbind();
    vars.get<ge::gl::VertexArray>("method.emptyVao")->bind();
    drawHolo(vars);
  return;
    vars.get<ge::gl::VertexArray>("method.emptyVao")->unbind();
  }

  vars.get<ge::gl::VertexArray>("method.emptyVao")->unbind();


}


void onQuit(vars::Vars&vars){
  ImGui::GetIO().FontGlobalScale = 1.f;
  ImGui::GetStyle().ScaleAllSizes(.25f);
  vars.erase("method");
}

EntryPoint main = [](){
  methodManager::Callbacks clbs;
  clbs.onInit        = onInit       ;
  clbs.onQuit        = onQuit       ;
  clbs.onDraw        = onDraw       ;
  clbs.onResize      = onResize     ;
  clbs.onKeyDown     = onKeyDown    ;
  clbs.onKeyUp       = onKeyUp      ;
  clbs.onMouseMotion = onMouseMotion;
  clbs.onUpdate      = onUpdate     ;
  MethodRegister::get().manager.registerMethod("lkg.renderHoloFocus",clbs);
};

}
