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
//additional libraries
#include <PGR/02/model.hpp> //camera model
#include<BasicCamera/OrbitCamera.h>
#include<BasicCamera/PerspectiveCamera.h>
#include<PGR/01/compileShaders.hpp>
#include<framework/bunny.hpp>
#include<PGR/03/phong.hpp>

using namespace ge::gl;
using namespace std;
using namespace compileShaders;

namespace student::project {

    GLuint prg = 0;
    GLuint vao = 0;
    GLuint vbo = 0;
    GLuint ebo = 0;

    glm::mat4 proj = glm::mat4(1.f);
    glm::mat4 view = glm::mat4(1.f);

    GLuint viewUniform;
    GLuint projUniform;

    std::string const source = R".(
#ifdef VERTEX_SHADER

uniform mat4 view = mat4(1.f);
uniform mat4 proj = mat4(1.f);

out vec3 vColor;

 layout(location=0)in vec3 position;  
 layout(location=1)in vec3 color   ;

  void main(){
    mat4 pv = proj * view;
    gl_Position = pv*vec4(position,1);
    vColor = color;

  }
#endif

#ifdef FRAGMENT_SHADER
in vec3 vColor;
layout(location=0)out vec4 fColor;
void main(){
  fColor = vec4(vColor,1);
}
#endif
).";

    void computeProjectionMatrix(vars::Vars& vars) {
        auto width = vars.getUint32("event.resizeX");
        auto height = vars.getUint32("event.resizeY");
        auto near = vars.getFloat("method.near");
        auto far = vars.getFloat("method.far");

        float aspect = static_cast<float>(width) / static_cast<float>(height);
        proj = glm::perspective(glm::half_pi<float>(), aspect, near, far);
    }

    void computeViewMatrix(vars::Vars& vars) {
        auto angleX = vars.getFloat("method.orbit.angleX");
        auto angleY = vars.getFloat("method.orbit.angleY");
        auto distance = vars.getFloat("method.orbit.distance");
        view =
            glm::translate(glm::mat4(1.f), glm::vec3(0.f, 0.f, -distance)) *
            glm::rotate(glm::mat4(1.f), angleX, glm::vec3(1.f, 0.f, 0.f)) *
            glm::rotate(glm::mat4(1.f), angleY, glm::vec3(0.f, 1.f, 0.f));
    }


    void onKeyDown(vars::Vars& vars) {
        ///which key was pressed
        auto key = vars.getInt32("event.key");
        auto sensitivity = vars.getFloat("method.sensitivity");
        auto& angleX = vars.getFloat("method.orbit.angleX");
        auto& angleY = vars.getFloat("method.orbit.angleY");
        if (key == SDLK_a)angleY += sensitivity;
        if (key == SDLK_d)angleY -= sensitivity;
        if (key == SDLK_w)angleX += sensitivity;
        if (key == SDLK_s)angleX -= sensitivity;
    }

    void onKeyUp(vars::Vars& vars) {
        ///which key was released
        auto key = vars.getInt32("event.key");
    }

    void onMouseMotion(vars::Vars& vars) {
        ///relative mouse movement
        auto xrel = vars.getInt32("event.mouse.xrel");
        auto yrel = vars.getInt32("event.mouse.yrel");

        ///mouse buttons
        auto left = vars.getBool("event.mouse.left");
        auto mid = vars.getBool("event.mouse.middle");
        auto right = vars.getBool("event.mouse.right");

    }

    void onUpdate(vars::Vars& vars) {
        ///time delta
        auto dt = vars.getFloat("event.dt");

    }



    void onInit(vars::Vars& vars) {
        //model::setUpCamera(vars);

        //vars.addUint32("method.bufferSize", 1024 * sizeof(float));
        vars.addUint32("method.vaoSize", sizeof(vao));
        vars.addFloat("method.sensitivity", 0.01f); //Camera settings
        vars.addFloat("method.near", 0.10f);
        vars.addFloat("method.far", 100.00f);
        vars.addFloat("method.orbit.angleX", 0.50f);
        vars.addFloat("method.orbit.angleY", 0.50f);
        vars.addFloat("method.orbit.distance", 4.00f);
        vars.addFloat("method.orbit.zoomSpeed", 0.10f);
        /// add limits to variable
        addVarsLimitsF(vars, "method.sensitivity", -1.f, +1.f, 0.01f);

        glClearColor(0.1, 0.1,0.1,1);

        prg = createProgram({
        createShader(GL_VERTEX_SHADER  ,"#version 460\n#define VERTEX_SHADER\n" + source),
        createShader(GL_FRAGMENT_SHADER,"#version 460\n#define FRAGMENT_SHADER\n" + source),
        });

        viewUniform = glGetUniformLocation(prg, "view");
        projUniform = glGetUniformLocation(prg, "proj");

        glCreateBuffers(1, &vbo);
        glNamedBufferData(vbo, sizeof(bunnyVertices), bunnyVertices, GL_DYNAMIC_DRAW);

        glCreateBuffers(1, &ebo);
        glNamedBufferData(ebo, sizeof(bunnyIndices), bunnyIndices, GL_DYNAMIC_DRAW);

        glCreateVertexArrays(1, &vao);
        glVertexArrayAttribBinding(vao, 0, 0);
        glEnableVertexArrayAttrib(vao, 0);
        glVertexArrayVertexBuffer(vao, 0,
            vbo,
            sizeof(float) * 0,//offset FOR VERTICES
            sizeof(float) * 6);//stride
        glVertexArrayAttribBinding(vao, 1, 1);
        glEnableVertexArrayAttrib(vao, 1);
        glVertexArrayVertexBuffer(vao, 1,
            vbo,
            sizeof(float) * 3,//offset FOR NORMALS
            sizeof(float) * 6);//stride
        glVertexArrayElementBuffer(vao, ebo);
        glEnable(GL_DEPTH_TEST);

        computeProjectionMatrix(vars);
    }

    void reactToSensitivityChange(vars::Vars& vars) {
        /// this will end the function immediatelly
        /// if none of listed variables was changed
        if (notChanged(vars, "method.all", __FUNCTION__,
            { "method.sensitivity",//list of variables
            }))return;

        std::cerr << "sensitivity was changed to: " << vars.getFloat("method.sensitivity") << std::endl;

    }

    void createBuffer(vars::Vars& vars) {
        if (notChanged(vars, "method.all", __FUNCTION__, { "method.vaoSize" }))return;

        auto size = vars.getUint32("method.vaoSize");
        vars.reCreate<Buffer>("method.vao", size);

        //vars.reCreate<Buffer>("method.buffer", size);

        std::cerr << "buffer on GPU was recreated" << std::endl;
    }

    void onDraw(vars::Vars& vars) {
        reactToSensitivityChange(vars);
        createBuffer(vars);


        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        /// addOrGet... with create variable if it does not exist
        /// and return its value
        /// it will be automatically added to GUI
        if (vars.addOrGetBool("method.shouldPrint")) {
            std::cerr << "Hello World!" << std::endl;
        }

        computeProjectionMatrix(vars);
        computeViewMatrix(vars);

        glBindVertexArray(vao);
        glUseProgram(prg);

        glProgramUniformMatrix4fv(prg, viewUniform, 1, GL_FALSE, glm::value_ptr(view));
        glProgramUniformMatrix4fv(prg, projUniform, 1, GL_FALSE, glm::value_ptr(proj));
        glDrawElements(GL_TRIANGLES, sizeof(bunnyIndices)/sizeof(uint32_t), GL_UNSIGNED_INT,nullptr);

    }

    void onResize(vars::Vars& vars) {
        /// size of the screen
        auto width = vars.getUint32("event.resizeX");
        auto height = vars.getUint32("event.resizeY");

        glViewport(0, 0, width, height);
    }

    void onQuit(vars::Vars& vars) {
        /// if you created everything inside "method" namespace
        /// this line should be enough to clear everything...
        vars.erase("method");
    }

    /// this will register your method into menus and stuff like that...
    /// it is using static value initialization and singleton concept...
    EntryPoint main = []() {
        /// table of callbacks
        methodManager::Callbacks clbs;
        clbs.onDraw = onDraw;
        clbs.onInit = onInit;
        clbs.onQuit = onQuit;
        clbs.onResize = onResize;
        clbs.onKeyDown = onKeyDown;
        clbs.onKeyUp = onKeyUp;
        clbs.onMouseMotion = onMouseMotion;
        clbs.onUpdate = onUpdate;

        /// register method
        MethodRegister::get().manager.registerMethod("student.project", clbs);
        };

}
