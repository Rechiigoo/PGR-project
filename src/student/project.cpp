/**
 * This file can be used as template for student project.
 *
 * There is a lot of hidden functionality not documented
 * If you have questions about the framework
 * email me: imilet@fit.vutbr.cz
 */


#include "stb_image.h"
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
#include "blorb.hpp"
#include "hatblorb.hpp"
#include<PGR/03/phong.hpp>
#include<PGR/03/shadowedModel.hpp>
#include <cstddef>

using namespace ge::gl;
using namespace std;
using namespace compileShaders;
using namespace shadowedModel;

namespace student::project {

    GLuint prg = 0;
    GLuint vao = 0;
    GLuint vbo = 0;
    GLuint ebo = 0;

    glm::mat4 proj = glm::mat4(1.f);
    glm::mat4 view = glm::mat4(1.f);

    GLuint viewUniform;
    GLuint projUniform;

    std::string const leather = R".(
// Vertex shader part
#ifdef VERTEX_SHADER
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoord;
layout (location = 3) in vec3 aTangent;
layout (location = 4) in vec3 aBitangent;

out VS_OUT {
    vec2 texCoord;
    mat3 TBN;
    vec3 fragPos;
    vec3 viewPos;
} vs_out;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform vec3 cameraPos;

void main()
{
    vec3 T = normalize(mat3(model) * aTangent);
    vec3 B = normalize(mat3(model) * aBitangent);
    vec3 N = normalize(mat3(model) * aNormal);
    vs_out.TBN = mat3(T, B, N);

    vec4 worldPos = model * vec4(aPos, 1.0);
    vs_out.fragPos = worldPos.xyz;
    vs_out.viewPos = cameraPos;
    vs_out.texCoord = aTexCoord;

    gl_Position = projection * view * worldPos;
}
#endif

// Fragment shader part
#ifdef FRAGMENT_SHADER
in VS_OUT {
    vec2 texCoord;
    mat3 TBN;
    vec3 fragPos;
    vec3 viewPos;
} fs_in;

out vec4 FragColor;

uniform sampler2D diffuseMap;
uniform sampler2D normalMap;
uniform sampler2D heightMap;

uniform vec3 lightPos;
uniform vec3 lightColor;

float heightScale = 0.05;   // leather usually small height variation

vec2 parallaxMapping(vec2 texCoords, vec3 viewDir)
{
    float height = texture(heightMap, texCoords).r;
    return texCoords + viewDir.xy * (height * heightScale);
}

void main()
{
    vec3 viewDir = normalize(fs_in.TBN * (fs_in.viewPos - fs_in.fragPos));
    vec2 texCoords = parallaxMapping(fs_in.texCoord, viewDir);

    if (texCoords.x < 0.0 || texCoords.y < 0.0 ||
        texCoords.x > 1.0 || texCoords.y > 1.0)
        discard;

    vec3 normal = texture(normalMap, texCoords).rgb;
    normal = normalize(normal * 2.0 - 1.0);   // tangent-space normal

    vec3 fragToLight = fs_in.TBN * normalize(lightPos - fs_in.fragPos);

    float diff = max(dot(fragToLight, normal), 0.0);
    vec3 reflectDir = reflect(-fragToLight, normal);
    vec3 viewDirection = normalize(fs_in.viewPos - fs_in.fragPos);
    float spec = pow(max(dot(reflectDir, viewDirection), 0.0), 16.0);

    vec3 color = texture(diffuseMap, texCoords).rgb;
    vec3 lighting = (diff * color + 0.2 * color + 0.1 * spec) * lightColor;

    FragColor = vec4(lighting, 1.0);
}
#endif
).";


    std::string const furshader = R".(
    #ifdef VERTEX_SHADER
    layout(location = 0) in vec3 position;
    layout(location = 1) in vec3 normal;

    out vec3 vNormal;
    out vec3 vPosition;

    uniform mat4 model = mat4(1.f);; 
    uniform mat4 view = mat4(1.f);;
    uniform mat4 proj = mat4(1.f);;

    void main() {
        vNormal = mat3(transpose(inverse(model))) * normal;
        vPosition = vec3(model * vec4(position, 1.0));
        gl_Position = proj * view * vec4(vPosition, 1.0);
    }
    #endif

#ifdef GEOMETRY_SHADER

layout(triangles) in;
layout(triangle_strip, max_vertices = 18) out;

in vec3 vNormal[];
in vec3 vPosition[];

out vec3 gNormal;
out vec3 gPosition;
out float gLayer;

uniform mat4 view  = mat4(1.0);
uniform mat4 proj  = mat4(1.0);
uniform int furLayers = 15;
uniform float furLength = 0.25;

// simple hash for randomness
float hash(float n) { return fract(sin(n) * 43758.5453); }

void main() {
    for (int layer = 0; layer < furLayers; ++layer) {

        float t = float(layer) / float(furLayers);
        float offset = t * furLength;     

        for (int i = 0; i < 3; ++i) {

            // Per-vertex randomization
            float r1 = hash(float(layer * 13 + i * 17));
            float r2 = hash(float(layer * 19 + i * 23));

            // small sideways jitter
            vec3 tangentJitter = normalize(
                vec3(r1 - 0.5, r2 - 0.5, r1 * r2 - 0.5)
            );

            // mix normal + jitter, more jitter toward the tip
            vec3 spikeDir = normalize(
                mix(vNormal[i], tangentJitter, t * 0.7)
            );

            vec3 displacedPos = vPosition[i] + spikeDir * offset;

            gPosition = displacedPos;
            gNormal   = spikeDir;
            gLayer    = t;

            gl_Position = proj * view * vec4(displacedPos, 1.0);
            EmitVertex();
        }

        EndPrimitive();
    }
}
#endif

    #ifdef FRAGMENT_SHADER
    in vec3 gNormal;
    in vec3 gPosition;
    in float gLayer;

    out vec4 fColor;

    uniform vec3 lightDir = normalize(vec3(-0.5, -1.0, -0.3));
    uniform vec3 baseColor = vec3(0, 0, 0); // black
    uniform vec3 tipColor  = vec3(1, 1, 1); // white tips

    void main() {
        vec3 N = normalize(gNormal);
        float light = max(dot(N, -lightDir), 0.2);

       
        vec3 color = mix(baseColor, tipColor, gLayer);

        
        float alpha = 1.0 - pow(gLayer, 2.0);

        fColor = vec4(color * light, alpha);
    }
    #endif
).";

    std::string const source2 = R".(
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

std::string const source = R".(

#ifdef VERTEX_SHADER
uniform mat4 view  = mat4(1.f);
uniform mat4 proj  = mat4(1.f);
uniform mat4 model = mat4(1.f);

layout(location = 0)in vec3 position;
layout(location = 1)in vec3 normal  ;
layout(location = 2)in vec2 texCoord;

out vec2 vCoord;
out vec3 vNormal;
out vec3 vPosition;
out vec3 vCamPosition;
void main(){
  vCoord  = texCoord;
  vNormal = normal  ;
  vPosition = vec3(model*vec4(position,1.f));
  vCamPosition = vec3(inverse(view)*vec4(0,0,0,1));
  gl_Position = proj*view*model*vec4(position,1.f);
}
#endif

#ifdef FRAGMENT_SHADER
in vec3 vPosition;
in vec2 vCoord;
in vec3 vNormal;
in vec3 vCamPosition;

uniform vec3  lightPosition = vec3(30,30,30)   ;
uniform vec3  lightColor    = vec3(1,1,1)      ;
uniform vec3  lightAmbient  = vec3(0.3,0.1,0.0);
uniform float shininess     = 60.f             ;


uniform sampler2D diffuseTexture;
uniform vec4      diffuseColor = vec4(1.f);
uniform int       useTexture   = 0;
uniform vec3 difcolor = vec3(1.f);

layout(location=0)out vec4 fColor;
void main(){
  vec3 diffuseColor = difcolor;
  fColor = vec4(vNormal,1);

  if(useTexture == 1)
    diffuseColor = texture(diffuseTexture,vCoord).rgb;
  else
    diffuseColor = diffuseColor.rgb;


  vec3 finalColor = phongLighting(
      vPosition          ,
      normalize(vNormal) ,
      lightPosition      ,
      vCamPosition       ,
      lightColor         ,
      lightAmbient       ,
      diffuseColor       ,
      shininess          ,
      0.f                );

    fColor = vec4(finalColor,1.f);

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
        auto xrel = vars.getInt32("event.mouse.xrel");
        auto yrel = vars.getInt32("event.mouse.yrel");

        auto sensitivity = vars.getFloat("method.sensitivity");
        auto zoomSpeed = vars.getFloat("method.orbit.zoomSpeed");
        auto& angleX = vars.getFloat("method.orbit.angleX");
        auto& angleY = vars.getFloat("method.orbit.angleY");
        auto& distance = vars.getFloat("method.orbit.distance");

        if (vars.getBool("event.mouse.middle")) {
            angleX += sensitivity * yrel;
            angleY += sensitivity * xrel;

            angleX = glm::clamp(angleX, -glm::half_pi<float>(), glm::half_pi<float>());
        }
        if (vars.getBool("event.mouse.right")) {
            distance += zoomSpeed * yrel;

            distance = glm::clamp(distance, 0.f, 100.f);
        }

    }

    void onUpdate(vars::Vars& vars) {
        ///time delta
        auto dt = vars.getFloat("event.dt");

    }

    void updateColor(vars::Vars& vars) {
        if (notChanged(vars, "all", __FUNCTION__, { "blue", "red", "green"})) return;
        GLuint loc = glGetUniformLocation(prg, "difcolor");
        if (loc != -1) {
            glUniform3f(loc, vars.getFloat("red"), vars.getFloat("green"), vars.getFloat("blue"));
        }
    }

    GLuint loadTexture(const char* path) {
        GLuint texID;
        glGenTextures(1, &texID);
        glBindTexture(GL_TEXTURE_2D, texID);


        int width, height, nrChannels;
        unsigned char* data = stbi_load(path, &width, &height, &nrChannels, 0);
        if (!data) {
            std::cerr << "Failed to load image: " << path << std::endl;
            return 0;
        }


        GLenum format = GL_RGB;
        if (nrChannels == 1) format = GL_RED;
        else if (nrChannels == 3) format = GL_RGB;
        else if (nrChannels == 4) format = GL_RGBA;


        glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);


        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);


        stbi_image_free(data);
        return texID;
    }

    void generateSphericalUVsAndTangents(
        const std::vector<glm::vec3>& positions,
        const std::vector<glm::vec3>& normals,
        const std::vector<unsigned int>& indices,
        std::vector<glm::vec2>& texcoords,
        std::vector<glm::vec3>& tangents,
        std::vector<glm::vec3>& bitangents)
    {
        size_t vertexCount = positions.size();

        texcoords.resize(vertexCount);
        tangents.assign(vertexCount, glm::vec3(0.0f));
        bitangents.assign(vertexCount, glm::vec3(0.0f));

        // ------------------------------------------------------------
        // Step 1: Generate spherical UV
        // ------------------------------------------------------------
        for (size_t i = 0; i < vertexCount; ++i)
        {
            glm::vec3 p = glm::normalize(positions[i]);

            float u = atan2(p.z, p.x) / (2.0f * glm::pi<float>()) + 0.5f;
            float v = p.y * 0.5f + 0.5f;

            texcoords[i] = glm::vec2(u, v);
        }

        // ------------------------------------------------------------
        // Step 2: Compute face tangents
        // ------------------------------------------------------------
        for (size_t i = 0; i < indices.size(); i += 3)
        {
            unsigned int i0 = indices[i];
            unsigned int i1 = indices[i + 1];
            unsigned int i2 = indices[i + 2];

            const glm::vec3& p0 = positions[i0];
            const glm::vec3& p1 = positions[i1];
            const glm::vec3& p2 = positions[i2];

            const glm::vec2& uv0 = texcoords[i0];
            const glm::vec2& uv1 = texcoords[i1];
            const glm::vec2& uv2 = texcoords[i2];

            glm::vec3 e1 = p1 - p0;
            glm::vec3 e2 = p2 - p0;

            glm::vec2 dUV1 = uv1 - uv0;
            glm::vec2 dUV2 = uv2 - uv0;

            float denom = dUV1.x * dUV2.y - dUV1.y * dUV2.x;
            float f = (fabs(denom) < 1e-6f) ? 1.0f : 1.0f / denom;

            glm::vec3 T(
                f * (dUV2.y * e1.x - dUV1.y * e2.x),
                f * (dUV2.y * e1.y - dUV1.y * e2.y),
                f * (dUV2.y * e1.z - dUV1.y * e2.z)
            );

            glm::vec3 B(
                f * (-dUV2.x * e1.x + dUV1.x * e2.x),
                f * (-dUV2.x * e1.y + dUV1.x * e2.y),
                f * (-dUV2.x * e1.z + dUV1.x * e2.z)
            );

            tangents[i0] += T; tangents[i1] += T; tangents[i2] += T;
            bitangents[i0] += B; bitangents[i1] += B; bitangents[i2] += B;
        }

        // ------------------------------------------------------------
        // Step 3: Orthonormalize (per vertex)
        // ------------------------------------------------------------
        for (size_t i = 0; i < vertexCount; ++i)
        {
            const glm::vec3& n = normals[i];

            glm::vec3& t = tangents[i];
            glm::vec3& b = bitangents[i];

            // Gram-Schmidt: make T orthogonal to N
            t = glm::normalize(t - n * glm::dot(n, t));

            // Compute handedness
            glm::vec3 computedB = glm::cross(n, t);
            float handedness = (glm::dot(computedB, b) < 0.0f) ? -1.0f : 1.0f;

            b = glm::normalize(glm::cross(n, t) * handedness);
        }
    }
    
    void updateShaders(vars::Vars& vars) {
        if (notChanged(vars, "method.all", __FUNCTION__, { "method.shaderToggle" })) return;

        glDeleteProgram(prg); // clean up old program

        if (vars.getUint32("method.shaderToggle") == 0) { //normal phong shader
            prg = createProgram({
                createShader(GL_VERTEX_SHADER,   "#version 460\n#define VERTEX_SHADER\n" + source),
                createShader(GL_FRAGMENT_SHADER, "#version 460\n#define FRAGMENT_SHADER\n" + phong::phongLightingShader + source),
                });
        }
        else if (vars.getUint32("method.shaderToggle") == 1) { //fur shader
            prg = createProgram({
                createShader(GL_VERTEX_SHADER,   "#version 460\n#define VERTEX_SHADER\n" + furshader),
                createShader(GL_GEOMETRY_SHADER, "#version 460\n#define GEOMETRY_SHADER\n" + furshader),
                createShader(GL_FRAGMENT_SHADER, "#version 460\n#define FRAGMENT_SHADER\n" + furshader),
                });
        }
        else {
            prg = createProgram({ //default shader with normals
                createShader(GL_VERTEX_SHADER,   "#version 460\n#define VERTEX_SHADER\n" + source2),
                createShader(GL_FRAGMENT_SHADER, "#version 460\n#define FRAGMENT_SHADER\n" + source2),
                });
        }

        viewUniform = glGetUniformLocation(prg, "view");
        projUniform = glGetUniformLocation(prg, "proj");
    }

    void createBuffer(vars::Vars& vars) {
        glCreateBuffers(1, &vbo);
        glNamedBufferData(vbo, sizeof(blorbVertices), blorbVertices, GL_DYNAMIC_DRAW);

        glCreateBuffers(1, &ebo);
        glNamedBufferData(ebo, sizeof(blorbIndices), blorbIndices, GL_DYNAMIC_DRAW);

        glCreateVertexArrays(1, &vao);
        glVertexArrayAttribBinding(vao, 0, 0);
        glEnableVertexArrayAttrib(vao, 0);
        glVertexArrayVertexBuffer(vao, 0,
            vbo,
            sizeof(float) * 0,//offset FOR VERTICES
            sizeof(float) * sizeof(BlorbVertex)/4);//stride
        glVertexArrayAttribBinding(vao, 1, 1);
        glEnableVertexArrayAttrib(vao, 1);
        glVertexArrayVertexBuffer(vao, 1,
            vbo,
            sizeof(float) * 3,//offset FOR NORMALS
            sizeof(float) * sizeof(BlorbVertex) / 4);//stride
        glVertexArrayElementBuffer(vao, ebo);
    }

    void onInit(vars::Vars& vars) {
        //model::setUpCamera(vars);

        //vars.addUint32("method.bufferSize", 1024 * sizeof(float));
        vars.addFloat("method.sensitivity", 0.1f); //Camera settings
        vars.addFloat("method.near", 0.10f);
        vars.addFloat("method.far", 100.00f);
        vars.addFloat("method.orbit.angleX", 0.50f);
        vars.addFloat("method.orbit.angleY", 0.50f);
        vars.addFloat("method.orbit.distance", 6.00f);
        vars.addFloat("method.orbit.zoomSpeed", 0.10f);
        vars.addUint32("method.shaderToggle",0);
        /// add limits to variable
        addVarsLimitsF(vars, "method.sensitivity", -1.f, +1.f, 0.1f);
        //add diffuse_color variables
        vars.addFloat("red",1.f);
        vars.addFloat("green", 1.f);
        vars.addFloat("blue", 1.f);
        //add limits to the colors
        addVarsLimitsF(vars, "red", 0.f, +1.f, 0.01f);
        addVarsLimitsF(vars, "green", 0.f, +1.f, 0.01f);
        addVarsLimitsF(vars, "blue", 0.f, +1.f, 0.01f);

        glClearColor(0.1, 0.1,0.1,1);


        glEnable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        //glDepthMask(GL_FALSE);

        createBuffer(vars);

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

    void onDraw(vars::Vars& vars) {
        reactToSensitivityChange(vars);
        //createBuffer(vars);
        updateShaders(vars);


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
        updateColor(vars); //update the diffuse color

        glProgramUniformMatrix4fv(prg, viewUniform, 1, GL_FALSE, glm::value_ptr(view));
        glProgramUniformMatrix4fv(prg, projUniform, 1, GL_FALSE, glm::value_ptr(proj));
        glDrawElements(GL_TRIANGLES, sizeof(blorbIndices)/sizeof(uint32_t), GL_UNSIGNED_INT,nullptr);
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
        vars.erase("red");
        vars.erase("green");
        vars.erase("blue");
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
